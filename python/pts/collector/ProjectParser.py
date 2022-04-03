from typing import *

from datetime import datetime
from pathlib import Path
import re
from tqdm import tqdm
import traceback
from urllib.error import HTTPError
from urllib.request import urlopen
from os import listdir

from seutil import LoggingUtils, IOUtils, BashUtils, TimeUtils, GitHubUtils
from seutil.project import Project

from pts.Environment import Environment
from pts.Macros import Macros
from pts.data.ProjectData import ProjectData


class ProjectParser:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self):
        self.repos_downloads_dir: Path = Macros.repos_downloads_dir
        self.repos_results_dir: Path = Macros.repos_results_dir

        return

    def parse_projects(self):
        proj_list = listdir(self.repos_downloads_dir)
        for proj in proj_list:
            self.parse_project(proj)

    def parse_project(self, project_name: str, revision=None):
        Environment.require_collector()

        downloads_dir = self.repos_downloads_dir / project_name
        if revision is None:
            results_dir = self.repos_results_dir / project_name
        else:
            results_dir = self.repos_results_dir / project_name / revision

        IOUtils.mk_dir(results_dir)

        if revision is None:
            with IOUtils.cd(downloads_dir):
                git_hash = BashUtils.run("git rev-parse HEAD", expected_return_code=0).stdout
        else:
            git_hash = revision
            with IOUtils.cd(downloads_dir):
                BashUtils.run(f"git checkout {git_hash} -f", expected_return_code=0)

        # Use Javaparser to parse project
        project_data = ProjectData.create()
        project_data.name = project_name
        project_data.sha = git_hash

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

        return
