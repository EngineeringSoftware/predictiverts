from typing import *

from datetime import datetime
from pathlib import Path
import re
from tqdm import tqdm
import traceback
from urllib.error import HTTPError
from urllib.request import urlopen
from xml.etree import ElementTree

from seutil import LoggingUtils, IOUtils, BashUtils, TimeUtils, GitHubUtils
from seutil.project import Project

from pts.Environment import Environment
from pts.Macros import Macros
from pts.data.ProjectData import ProjectData
from pts.Utils import Utils


class DataCollector:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self):
        self.repos_downloads_dir: Path = Macros.repos_downloads_dir
        self.repos_results_dir: Path = Macros.repos_results_dir
        self.collected_projects_list = []

        return

    def collect_data(self, **options):
        which = Utils.get_option_as_list(options, "which")
        for item in which:
            self.logger.info(f"Collecting data: {item}; options: {options}")
            if item == "eval-data":
                self.collect_eval_data(**options)
            elif item == "test-method-data":
                self.collect_test_method(**options)
            elif item == "method-data-collect":
                project = options.get("project")
                self.collect_project_with_shas(project)
            else:
                raise NotImplementedError
            # end if
        # end for

    def collect_project_with_shas(self, project: str):
        """
        Create the project_name-{revision} directory in _results and collect all the methods in
        that revision, for extracting the methods as features of the model
        """
        from pts.collector.ProjectParser import ProjectParser
        # First collect all shas used in eval
        eval_ag_file = Macros.eval_data_dir / "mutated-eval-data" / f"{project}-ag.json"
        eval_ag_data_list = IOUtils.load(eval_ag_file)
        sha_list = set()
        for eval_data in eval_ag_data_list:
            sha_list.add(eval_data["commit"].split('-')[0])
        # end for
        pp = ProjectParser()
        for revision in sha_list:
            pp.parse_project(project, revision)
            self.collect_test_methods_with_shas(project, revision)
        # end for

    def collect_test_methods_with_shas(self, project: str, revision: str):
        """Make the test2methods.json for the projects"""
        from collections import defaultdict
        project_result_dir = self.repos_results_dir / project / revision / "collector"
        project_method_file = project_result_dir / "method-data.json"
        method_list = IOUtils.load(project_method_file)
        test_class_2_methods = defaultdict(list)
        for m in method_list:
            if "src/test" in m["path"] or "Test.java" in m["path"]:
                class_name = m["path"].split('/')[-1].split('.java')[0]
                test_class_2_methods[class_name].append(m["id"])
            # end if
        # end for
        IOUtils.dump(project_result_dir / "test2methods.json", test_class_2_methods)
    # end for

    def collect_test_method(self, **options):
        """Collect and make the dictionary: Test class name: [test method id]"""
        from collections import defaultdict
        projects = Utils.get_option_as_list(options, "projects")
        for project in projects:
            project_result_dir = self.repos_results_dir / project / "collector"
            project_method_file = project_result_dir / "method-data.json"
            method_list = IOUtils.load(project_method_file)
            test_class_2_methods = defaultdict(list)
            for m in method_list:
                if "src/test" in m["path"] or "Test.java" in m["path"]:
                    test_class_2_methods[m["class_name"]].append(m["id"])
                # end if
            # end for
            IOUtils.dump(project_result_dir/"test2methods.json", test_class_2_methods)
        # end for

    def collect_eval_data(self, **options):
        projects = Utils.get_option_as_list(options, "projects")
        shas = Utils.get_option_as_list(options, "shas")
        num_sha = options.get("num_sha", 20)
        proj_dict = {}
        for p, sh in zip(projects, shas):
            proj_dict[p] = sh
        # end for
        from pts.collector.eval_data_collection import main
        main(proj_dict)

    def download_projects(self, project_list: Dict):
        for project, sha in project_list.items():
            try:
                # Project name is user_repo
                project_url = self.parse_repo_name(project)

                # Query if the repo exists and is public on GitHub - private repo will block and waste time on git clone
                if not self.check_github_url(project_url):
                    self.logger.warning(f"Project {project} no longer available.")
                    continue
                # end if

                # Download project and parse

                downloads_dir = self.repos_downloads_dir / project
                results_dir = self.repos_results_dir / project

                IOUtils.rm_dir(results_dir)
                IOUtils.mk_dir(results_dir)

                # Clone the repo if not exists
                if not downloads_dir.exists():
                    with IOUtils.cd(self.repos_downloads_dir):
                        with TimeUtils.time_limit(300):
                            BashUtils.run(f"git clone {project_url} {project}", expected_return_code=0)
                        # end with
                        if downloads_dir.exists():
                            with IOUtils.cd(downloads_dir):
                                BashUtils.run(f"git checkout {sha}", expected_return_code=0)
                            # end with
                        else:
                            self.logger.warning(f"{project} is not downloaded!")
                    # end with
                # end if
            except KeyboardInterrupt:
                self.logger.warning(f"KeyboardInterrupt")
                break
            except:
                self.logger.warning(f"Collection for project {project_url} failed, error was: {traceback.format_exc()}")
        # end for
        return

    def collect_projects(self, project_urls_file: Path,
                         skip_collected: bool,
                         beg: int = None,
                         cnt: int = None,
                         ):
        # 1. Load urls
        project_urls = IOUtils.load(project_urls_file, IOUtils.Format.txt).splitlines()
        invalid_project_urls = list()

        # Limit the number of projects to collect
        if beg is None:  beg = 0
        if cnt is None:  cnt = len(project_urls)

        project_urls = project_urls[beg:beg + cnt]

        for pi, project_url in enumerate(project_urls):
            self.logger.info(f"Project {beg + pi + 1}/{len(project_urls)}({beg}-{beg + cnt}): {project_url}")

            try:
                # Project name is user_repo
                user_repo = self.parse_github_url(project_url)
                if user_repo is None:
                    self.logger.warning(f"URL {project_url} is not a valid GitHub repo URL.")
                    invalid_project_urls.append(project_url)
                    continue
                # end if
                project_name = f"{user_repo[0]}_{user_repo[1]}"

                if skip_collected and self.is_project_collected(project_name, project_url):
                    self.logger.info(f"Project {project_name} already collected.")
                    continue
                # end if

                # Query if the repo exists and is public on GitHub - private repo will block and waste time on git clone
                if not self.check_github_url(project_url):
                    self.logger.warning(f"Project {project_name} no longer available.")
                    invalid_project_urls.append(project_url)
                    continue
                # end if

                # Try to access pom.xml
                pom_xml_url = project_url[:-len(".git")] + "/blob/master/pom.xml"
                try:
                    urlopen(pom_xml_url)
                except HTTPError:
                    # Delete this project
                    self.logger.info(
                        f"Project {project_name} does not seem to be a Maven project. Moving to nouse set")
                    continue
                # end try

                # Download project and parse
                self.collect_project(project_name, project_url)
            except KeyboardInterrupt:
                self.logger.warning(f"KeyboardInterrupt")
                break
            except:
                self.logger.warning(f"Collection for project {project_url} failed, error was: {traceback.format_exc()}")
        # end for
        IOUtils.dump(Macros.results_dir/"collected-projects.txt", self.collected_projects_list, IOUtils.Format.txt)
        return

    def collect_project(self, project_name: str, project_url: str):
        Environment.require_collector()

        # 0. Download repo
        downloads_dir = self.repos_downloads_dir / project_name
        results_dir = self.repos_results_dir / project_name

        IOUtils.rm_dir(results_dir)
        IOUtils.mk_dir(results_dir)

        # Clone the repo if not exists
        if not downloads_dir.exists():
            with IOUtils.cd(self.repos_downloads_dir):
                with TimeUtils.time_limit(300):
                    BashUtils.run(f"git clone {project_url} {project_name}", expected_return_code=0)
                # end with
            # end with
        # end if

        # 1. Check junit version,
        if not self.check_junit(downloads_dir/"pom.xml"):
            self.logger.info(f"Project {project_name} does not satisfy the dependency requirements.")
            IOUtils.rm(downloads_dir)
            IOUtils.rm(results_dir)
            return

        # 2. Use Javaparser to parse project
        project_data = ProjectData.create()
        project_data.name = project_name
        project_data.url = project_url

        # Get revision (SHA)
        with IOUtils.cd(downloads_dir):
            git_log_out = BashUtils.run(f"git rev-parse HEAD", expected_return_code=0).stdout
            project_data.revision = git_log_out

        project_data_file = results_dir / "project.json"
        IOUtils.dump(project_data_file, IOUtils.jsonfy(project_data), IOUtils.Format.jsonPretty)

        # Prepare config
        log_file = results_dir / "collector-log.txt"
        output_dir = results_dir / "collector"

        config = {
            "collect": True,
            "projectDir": str(downloads_dir),
            "projectDataFile": str(project_data_file),
            "logFile": str(log_file),
            "outputDir": str(output_dir),
        }
        config_file = results_dir / "collector-config.json"
        IOUtils.dump(config_file, config, IOUtils.Format.jsonPretty)

        self.logger.info(f"Starting the Java collector. Check log at {log_file} and outputs at {output_dir}")
        rr = BashUtils.run(f"java -jar {Environment.collector_jar} {config_file}", expected_return_code=0)
        if rr.stderr:
            self.logger.warning(f"Stderr of collector:\n{rr.stderr}")
        # end if

        self.collected_projects_list.append(project_url)

        return

    REQUIREMENTS = {
        "junit": lambda v: int(v) == 4
    }

    @classmethod
    def check_junit(cls, pom_file: Path) -> bool:
        """Parse the pom xml file and check whether it satisfies the requirements specified"""
        namespaces = {'xmlns': 'http://maven.apache.org/POM/4.0.0'}

        tree = ElementTree.parse(pom_file)
        root = tree.getroot()

        deps = root.findall(".//xmlns:dependency", namespaces=namespaces)
        for d in deps:
            artifact_id = d.find("xmlns:artifactId", namespaces=namespaces).text
            if artifact_id in cls.REQUIREMENTS.keys():
                version = d.find("xmlns:version", namespaces=namespaces).text.split(".")[0]
                if cls.REQUIREMENTS[artifact_id](version):
                    return True
                else:
                    return False
        return False

    RE_GITHUB_URL = re.compile(r"https://github\.com/(?P<user>[^/]+)/(?P<repo>.+?)(\.git)?")

    @classmethod
    def parse_repo_name(cls, project_name):
        """ Parse the repo name to get github url."""
        github_user_name = project_name.split('_')[0]
        github_project_name = project_name.split('_')[1]

        github_url = f"https://github.com/{github_user_name}/{github_project_name}.git"

        return github_url

    @classmethod
    def parse_github_url(cls, github_url) -> Tuple[str, str]:
        """
        Parses a GitHub repo URL and returns the user name and repo name. Returns None if the URL is invalid.
        """
        m = cls.RE_GITHUB_URL.fullmatch(github_url)
        if m is None:
            return None
        else:
            return m.group("user"), m.group("repo")
        # end if

    def is_project_collected(self, project_name, project_url):
        return project_name in self.collected_projects_list or project_url in self.collected_projects_list

    @classmethod
    def check_github_url(cls, github_url):
        try:
            urlopen(github_url)
            return True
        except HTTPError:
            return False

    def get_github_top_repos(self):

        # 1000 top starred projects
        repositories = GitHubUtils.search_repos(q="topic:java language:java", sort="stars", order="desc",
                                                max_num_repos=1000)
        for repo in repositories:
            # Create a Project instance from the repo
            project = Project()
            project.url = GitHubUtils.ensure_github_api_call(lambda g: repo.clone_url)
            project.data["user"] = GitHubUtils.ensure_github_api_call(lambda g: repo.owner.login)
            project.data["repo"] = GitHubUtils.ensure_github_api_call(lambda g: repo.name)
            project.full_name = f"{project.data['user']}_{project.data['repo']}"
            project.data["branch"] = GitHubUtils.ensure_github_api_call(lambda g: repo.default_branch)

