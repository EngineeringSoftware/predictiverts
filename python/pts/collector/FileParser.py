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


class FileParser:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self):
        self.repos_downloads_dir: Path = Macros.repos_downloads_dir
        self.repos_results_dir: Path = Macros.repos_results_dir
        return

    def parse_java_file(self, project_name: str, java_file: str, revision: str):
        Environment.require_collector()

        downloads_dir = self.repos_downloads_dir / project_name
        results_dir = self.repos_results_dir / project_name

        IOUtils.mk_dir(results_dir)

        # with IOUtils.cd(downloads_dir):
        #     git_hash = BashUtils.run("git rev-parse HEAD", expected_return_code=0).stdout

        # Use Javaparser to parse project
        project_data = ProjectData.create()
        project_data.name = project_name
        project_data.revision = revision

        project_data_file = results_dir / "project-parse.json"
        IOUtils.dump(project_data_file, IOUtils.jsonfy(project_data), IOUtils.Format.jsonPretty)

        # Prepare config
        log_file = results_dir / "parser-log.txt"
        output_dir = results_dir / "parser"

        config = {
            "parse": True,
            "projectDir": str(downloads_dir),
            "logFile": str(log_file),
            "projectDataFile": str(project_data_file),
            "outputDir": str(output_dir),
            "javaFile": java_file,
            "revision": revision
        }
        config_file = results_dir / "parser-config.json"
        IOUtils.dump(config_file, config, IOUtils.Format.jsonPretty)

        self.logger.info(f"Starting the Java collector. Check log at {log_file} and outputs at {output_dir}")
        rr = BashUtils.run(f"java -jar {Environment.collector_jar} {config_file}", expected_return_code=0)
        if rr.stderr:
            self.logger.warning(f"Stderr of collector:\n{rr.stderr}")
        # end if

        return
