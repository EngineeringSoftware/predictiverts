import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from typing import *
from os import listdir
from datetime import datetime
from pathlib import Path
import re
from tqdm import tqdm
import traceback
from urllib.error import HTTPError
from urllib.request import urlopen
from xml.etree import ElementTree
import sys

from seutil import LoggingUtils, IOUtils, BashUtils, TimeUtils, GitHubUtils
from seutil.project import Project

from pts.Environment import Environment
from pts.Macros import Macros
from pts.data.ProjectData import ProjectData
from pts.collector.DataCollector import DataCollector
from pts.collector.mutation.XMLParser import XMLParser
from pts.collector.mutation.RecoverMut import RecoverMut

def searchFile(dir_root, file_name):
    for dir_path, subpaths, files in os.walk(dir_root):
        for f in files:
            if f == file_name:
                return dir_path + '/' + f
    return None

class MutantCollector:
    """Code for collecting mutants, assume repos have been downloaded"""
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self, proj_list: List[str]):
        self.repos_downloads_dir: Path = Macros.repos_downloads_dir
        self.repos_results_dir: Path = Macros.repos_results_dir
        self.proj_list = proj_list
        self.result_dir = Macros.repos_results_dir

        return

    def locate_source_file(self, sourceFile: str, className: str, source_dir: Path):
        file_class = sourceFile.split(".")[0]
        path_list = re.split(r'[.$]', className)
        try:
            filename = "/".join(path_list[:path_list.index(file_class) + 1]) + ".java"
            mutated_files = BashUtils.run(f"find {source_dir} -wholename *{filename}").stdout.splitlines()
            if len(mutated_files) > 1:
                self.logger.warning(f"More than one file is found: {mutated_files}")
                return None
            else:
                mutated_file = mutated_files[0]
        except IndexError:
            self.logger.warning("Can not find source file!")
            mutated_file = None
        except ValueError:
            self.logger.warning(f"{file_class} not in the path list.")
            mutated_file = None
        return mutated_file

    @classmethod
    def get_flacky_tests(cls, project) -> List[str]:
        from pts.defectsep.run_defectsep import LANG_FLAKY_TESTS, MATH_FLAKY_TESTS, TIME_FLAKY_TESTS
        example = project.split('_')[1].lower()
        if example.startswith('lang-'):
            flaky_tests = LANG_FLAKY_TESTS
        elif example.startswith('math-'):
            flaky_tests = MATH_FLAKY_TESTS
        elif example.startswith('time-'):
            flaky_tests = TIME_FLAKY_TESTS
        else:
            return []
        return flaky_tests

    @classmethod
    def exclude_project_flacky_tests(cls, example: str, flaky_tests):
        project = example.split('_')[1].split('-')[0]
        example_number = int(example.split('_')[1].split('-')[1])
        if project.lower() == 'time':
            cls.excludeFlakyTestsFromTime_CommentSuite(example, flaky_tests)
        elif project.lower() == 'lang':
            if example_number in range(28, 42):
                cls.excludeFlakyTests_DeleteFile(example, flaky_tests)
            elif example_number in range(42, 54):
                cls.excludeFlakyTestsFromLang_CommentSuite(example, flaky_tests)
        elif project.lower() == 'math':
            cls.excludeFlakyTests_DeleteFile(example, flaky_tests)

    @classmethod
    def excludeFlakyTestsFromTime_CommentSuite(cls, example, flaky_tests):
        project = example.split('-')[0]
        example_number = int(example.split('-')[1])
        repo_dir = Macros.repos_downloads_dir/example
        BashUtils.run(f"cp {Macros.repos_downloads_dir}/pom.xml {repo_dir}", expected_return_code=0)
        for ft in flaky_tests:
            if ft == 'TestDateTimeZone' or ft == 'TestPeriodType':
                suite_file = repo_dir /'src/test/java/org/joda/time/TestAll.java'
            else:
                suite_file = repo_dir /'src/test/java/org/joda/time/format/TestAll.java'
            if not os.path.isfile(suite_file):
                continue
            fr = open(suite_file, 'r')
            lines = fr.readlines()
            fr.close()
            for i in range(len(lines)):
                if 'suite.addTest(' + ft + '.suite' in lines[i]:
                    lines[i] = '//' + lines[i]
            fw = open(suite_file, 'w')
            fw.write(''.join(lines))
            fw.close()
        # Delete those test files.
        for ft in flaky_tests:
            ft_file = searchFile(repo_dir, ft + '.java')
            if ft_file:
                print(f"Remove {ft_file}.")
                os.remove(ft_file)

    @classmethod
    def excludeFlakyTestsFromLang_CommentSuite(cls, example, flaky_tests):
        project = example.split('-')[0]
        example_number = int(example.split('-')[1])
        repo_dir = Macros.repos_downloads_dir / example
        for ft in flaky_tests:
            if ft == 'ToStringBuilderTest':
                suite_file = repo_dir + \
                             '/src/test/org/apache/commons/lang/builder/BuilderTestSuite.java'
            if not os.path.isfile(suite_file):
                continue
            fr = open(suite_file, 'r')
            lines = fr.readlines()
            fr.close()
            for i in range(len(lines)):
                if 'suite.addTestSuite(' + ft + '.class' in lines[i]:
                    lines[i] = '//' + lines[i]
            fw = open(suite_file, 'w')
            fw.write(''.join(lines))
            fw.close()

    @classmethod
    def excludeFlakyTests_DeleteFile(cls, example, flaky_tests):
        project = example.split('-')[0]
        example_number = int(example.split('-')[1])
        repo_dir = Macros.repos_downloads_dir/example
        for ft in flaky_tests:
            ft_file = searchFile(repo_dir, ft + '.java')
            if ft_file:
                print(f"Remove {ft_file}.")
                os.remove(ft_file)


    @classmethod
    def run_pit(cls, proj_dir: Path):
        """
        Run Pit on the proj_dir to generate mutants and mutate the given project.
        :param proj_dir: the directory of the cloned project
        """
        cls.logger.info(f"Start to run mutation testing using PIT in {proj_dir}.")
        # first make sure the pit extension has been installed
        with IOUtils.cd(f"{Macros.tools_dir}/pit-extension"):
            BashUtils.run("mvn package && cp target/pit-extension-1.0-SNAPSHOT.jar ${M2_HOME}/lib/ext",
                          expected_return_code=0)
        try:
            # IOUtils.rm_dir(proj_dir / "target")
            with IOUtils.cd(f"{proj_dir}"):
                BashUtils.run(f"mvn test")
                BashUtils.run(f"mvn compile -DskipTests", expected_return_code=0)
                BashUtils.run(f"mvn org.pitest:pitest-maven:mutationCoverage &> pit-log.txt", expected_return_code=0)

        except RuntimeError:
            cls.logger.info(f"Error occurred while running PIT in project {proj_dir}, the exception is  {traceback.format_exc()}.")
        return

    def extract_muts(self, default=True, binary=False, context=True):
        """
        Extract mutants from the PIT xml report.
        First parse the xml report file, then recover the code changes from the descriptions and line numbers in the
        report. And then save the data into mutant-data.json
        :context : True will return the new code with the context (5 lines above and 5 line below).
        """
        xml_parser = XMLParser()
        self.logger.info("Start to recover mutations from PIT report")
        proj_names: List = self.proj_list
        proj_result_dirs: List = [Path(self.repos_results_dir / proj) for proj in proj_names]  # a list of project dirs
        for proj, proj_dir in zip(proj_names, proj_result_dirs):
            proj_data = list()
            self.logger.info(f"Look at project {proj}, and first parse the pit report.")
            if not Path(proj_dir/"mutations.xml").exists():
                BashUtils.run(f"cp {self.repos_downloads_dir/proj}/target/pit-reports/*/mutations.xml {proj_dir}/",
                          expected_return_code=0)
            if binary:
                xml_parser.parse_pit_report(Path(proj_dir / "mutations.xml"), proj, proj_dir)
            else:
                xml_parser.parse_pit_report_labels(Path(proj_dir / "mutations.xml"), proj, proj_dir)
            try:
                if default:
                    reports = proj_dir/f"{proj}-default-mutation-report.json"
                else:
                    reports = proj_dir/f"{proj}-all-mutation-report.json"
                objs = IOUtils.load_json_stream(reports)
                cnt = 0
                for i, mut in enumerate(objs):
                    # find the source file first
                    source_file = self.locate_source_file(mut["sourceFile"], mut["mutatedClass"], Macros.repos_downloads_dir / proj)
                    mut["sourceFile"] = source_file
                    if source_file:
                        mr = RecoverMut()  # initialize a mutants recover
                        if context:
                            new_code, old_code, line_list = mr.recover_changed_file(mut["mutator"], mut)
                            if new_code != "" or old_code != "":
                                line_number = int(mut["lineNumber"])
                                mut["context"] = ' '.join(line_list[line_number - 5: line_number + 5])
                        else:
                            new_code, old_code = mr.recover_code_changes(mut["mutator"], mut)
                        mut["path"] = str(Path(source_file).relative_to(Macros.repos_downloads_dir/proj))
                        if new_code != "" or old_code != "":
                            mut["old_code"] = old_code.rstrip()
                            mut["new_code"] = new_code.rstrip()
                            cnt += 1
                            proj_data.append(mut)
                    # end if
                # end for
                self.logger.info(f"In total {cnt} mutants")
                print(f"In total {cnt} mutants recovered in project {proj}.")
                if binary:
                    IOUtils.dump(proj_dir/"collector"/"mutant-data.json", proj_data)
                else:
                    IOUtils.dump(proj_dir / "collector" / "tri-mutant-data.json", proj_data)
            except RuntimeError:
                print(RuntimeError)
                self.logger.info(f"Error occurred while recovering code changes")
                self.logger.warning(f"Collection for project {proj} failed, error was: {traceback.format_exc()}")
            # end try
        # end for

    def extract_muts_w_changed_files(self, sha: str, default=True):
        """
        First parse the xml report file, then recover the code changes from the descriptions and line numbers in the
        report. And then create a new branch to commit the changed files. This branch will be used in future to run
        STARTS/Ekstazi for selecting tests.
        """
        xml_parser = XMLParser()
        self.logger.info("Start to recover mutations from PIT report")
        proj_names: List = self.proj_list
        proj_result_dirs: List = [Path(self.repos_results_dir / proj) for proj in proj_names]  # a list of project dirs
        for proj, proj_dir in zip(proj_names, proj_result_dirs):
            # Make a mutated project dir
            exp_sha = sha
            IOUtils.mk_dir(self.repos_results_dir/f"mutated-{proj}")
            mutated_repo_dir = self.repos_results_dir/f"mutated-{proj}"
            proj_data = list()
            self.logger.info(f"Look at project {proj}, and first parse the pit report.")

            xml_parser.parse_pit_report(Path(proj_dir / "mutations.xml"), proj, proj_dir)
            try:
                if default:
                    reports = proj_dir / f"{proj}-default-mutation-report.json"
                else:
                    reports = proj_dir / f"{proj}-all-mutation-report.json"
                objs = IOUtils.load_json_stream(reports)
                cnt = 0
                for i, mut in enumerate(objs):
                    # find the source file first
                    source_file = self.locate_source_file(mut["sourceFile"], mut["mutatedClass"],
                                                          Macros.repos_downloads_dir / proj)
                    mut["sourceFile"] = source_file
                    if source_file:
                        mr = RecoverMut()  # initialize a mutants recover
                        new_code, old_code, changed_lines = mr.recover_changed_file(mut["mutator"], mut)
                        mut["path"] = str(Path(source_file).relative_to(Macros.repos_downloads_dir / proj))
                        if changed_lines != []:
                            mut["old_code"] = old_code.rstrip()
                            mut["new_code"] = new_code.rstrip()
                            cnt += 1
                            mut["id"] = cnt
                            proj_data.append(mut)
                            # first create a new repo
                            with IOUtils.cd(f"{self.repos_downloads_dir}/{proj}"):
                                BashUtils.run(f"git checkout master")
                                BashUtils.run(f"git reset --hard {exp_sha}")
                                BashUtils.run(f"git checkout -b mutation-{cnt} {exp_sha}")
                                with open(source_file, "w") as f:
                                    f.writelines(changed_lines)
                                # end with
                                try:
                                    BashUtils.run("mvn clean install -DskipTests", expected_return_code=0)
                                except:
                                    print(mut)
                                BashUtils.run(f"git add {source_file}")
                                BashUtils.run(f"git commit -m \"mutation # {cnt}.\"")
                                BashUtils.run(f"git checkout master")
                                BashUtils.run(f"git reset --hard {exp_sha}")
                                
                            # end with
                    # end if
                # end for
                self.logger.info(f"In total {cnt} mutants")
                IOUtils.dump(proj_dir / "collector" / "mutant-data.json", proj_data)
            except RuntimeError:
                print(RuntimeError)
                self.logger.info(f"Error occurred while recovering code changes")
                self.logger.warning(f"Collection for project {proj} failed, error was: {traceback.format_exc()}")
            except:
                print(f"Unexpected error: {sys.exc_info()[0]}")
                print(mut)
                raise
            # end try
        # end for

    def recover_mutants(self, mut, proj, task_id):
        # create a new repo
        BashUtils.run(f"cp -r {Macros.repos_downloads_dir/proj} /tmp/tmp-{task_id}")
        # find the source file first
        source_file = self.locate_source_file(mut["sourceFile"], mut["mutatedClass"], Path(f"/tmp/tmp-{task_id}"))
        mut["sourceFile"] = source_file
        if source_file:
            mr = RecoverMut()  # initialize a mutants recover
            new_code, old_code, changed_lines = mr.recover_changed_file(mut["mutator"], mut)
            mut["path"] = str(Path(source_file).relative_to(Macros.repos_downloads_dir / proj))
            if changed_lines != []:
                mut["old_code"] = old_code.rstrip()
                mut["new_code"] = new_code.rstrip()
                # first create a new repo
                with IOUtils.cd(f"/tmp/tmp-{task_id}"):
                    BashUtils.run(f"git checkout master")
                    BashUtils.run(f"git reset --hard {exp_sha}")
                    BashUtils.run(f"git checkout -b mutation-{cnt} {exp_sha}")
                    with open(source_file, "w") as f:
                        f.writelines(changed_lines)
                    # end with
                    BashUtils.run(f"git add {source_file}")
                    BashUtils.run(f"git commit -m \"mutation # {cnt}.\"")
                    BashUtils.run(f"git checkout master")
                    BashUtils.run(f"git reset --hard {exp_sha}")

                # end with
        # end if

    # end for

    def concurrent_run_starts_proj(self, proj: str):
        """Outdated. Run STARTS on each branch concurrently."""
        from pts.collector.eval_data_collection import collect_starts_tests_mut
        proj_mutant_file = self.result_dir / proj / "collector" / "mutant-data.json"
        objs = IOUtils.load_json_stream(proj_mutant_file)
        total_mut = 0
        for _ in objs:
            total_mut += 1
        output_dir = self.repos_results_dir / proj / "collect" / "starts-results"
        IOUtils.mk_dir(output_dir)
        with ProcessPoolExecutor(12) as executor:
            futures = [executor.submit(collect_starts_tests_mut, proj, b_id, output_dir) for b_id in range(1, total_mut+1)]
            for f in tqdm(as_completed(futures), total=total_mut):
                pass

    def build_test_dictionary(self):
        """
        Build a dict to find all test classes (do not consider inner class) and the methods in it.
        Note: currently only store the test class name and empty method list.
        """
        for proj in self.proj_list:
            test_class_2_method_dict = defaultdict(list)
            proj_method_file = self.result_dir / proj / "collector" / "method-data.json"
            method_list = IOUtils.load(proj_method_file)
            test_class_file_list = BashUtils.run(f"find {Macros.repos_downloads_dir}/{proj}/target/"
                                            f"surefire-reports/ -name \"*.txt\"").stdout.split("\n")
            for cf in test_class_file_list:
                if cf != "":
                    class_name = cf.split('/')[-1].split('.')[-2]
                    test_class_2_method_dict[class_name].append("POS")
                # end if
            # end if
            # for m in method_list:
            #     if "test" in m["path"]:
            #         test_class_2_method_dict[m["class_name"]].append(m["id"])
            #     # end if
            # # end for
            IOUtils.dump(self.result_dir / proj / "collector" / "test2method.json", test_class_2_method_dict)
        # end for

    def add_mutated_method_id(self, binary=False):
        """Add mutatedMethod id in the mutant-data.json file so as to get the changed method body."""
        for proj in self.proj_list:
            method_2_id = dict()
            proj_method_file = self.result_dir / proj / "collector" / "method-data.json"
            if binary:
                proj_mutant_file = self.result_dir / proj / "collector" / "mutant-data.json"
            else:
                proj_mutant_file = self.result_dir / proj / "collector" / "tri-mutant-data.json"
            method_list = IOUtils.load(proj_method_file)
            for m in method_list:
                key = m["path"] + "_" + m["name"]
                method_2_id[key] = m["id"]
            objs = IOUtils.load_json_stream(proj_mutant_file)
            new_mutant_list = list()
            for obj in objs:
                path = obj["path"]
                method_name = obj["mutatedMethod"]
                try:
                    id = method_2_id[path + "_" + method_name]
                    obj["mutatedMethodID"] = id
                except KeyError:
                    self.logger.info(f"[WARN]: Cannot find method {method_name}!")
                    continue
                new_mutant_list.append(obj)
            # end for
            if binary:
                tgt_file = "mutant-data.json"
            else:
                tgt_file = "tri-mutant-data.json"
            IOUtils.dump(self.result_dir / proj / "collector" / tgt_file, new_mutant_list)
        # end for






