from typing import *

from pathlib import Path

from seutil import LoggingUtils, IOUtils, BashUtils

from pts.Macros import Macros


class Environment:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.INFO)

    # ----------
    # Environment variables
    # ----------
    is_debug: bool = False
    random_seed: int = 14
    is_parallel: bool = False

    # ----------
    # Special repos
    # ----------

    @classmethod
    def require_data(cls):
        if cls.is_parallel:  return
        cls.require_special_repo(Macros.data_dir, "data")

    @classmethod
    def require_results(cls):
        if cls.is_parallel:  return
        cls.require_special_repo(Macros.results_dir, "results")

    @classmethod
    def get_git_url(cls):
        with IOUtils.cd(Macros.project_dir):
            return BashUtils.run(f"git config --get remote.origin.url", expected_return_code=0).stdout.strip()

    @classmethod
    def require_special_repo(cls, directory: Path, branch: str):
        cls.logger.info(f"Updating {directory} to {branch} branch")
        if directory.exists():
            if not directory.is_dir() or not (directory/".git").is_dir():
                LoggingUtils.log_and_raise(cls.logger, f"Path {directory} already exists but is not a proper git repository!", Exception)
            # end if

            with IOUtils.cd(directory):
                BashUtils.run(f"git pull", expected_return_code=0)
            # end with
        else:
            IOUtils.mk_dir(directory)
            with IOUtils.cd(directory):
                BashUtils.run(f"git clone --single-branch -b {branch} -- {cls.get_git_url()} .", expected_return_code=0)
            # end with
        # end if

    # ----------
    # Tools
    # ----------

    collector_installed = False
    collector_jar = str(Macros.collector_dir / "target" / f"collector-{Macros.collector_version}.jar")

    @classmethod
    def require_collector(cls):
        if cls.is_parallel:  return
        if not cls.collector_installed:
            cls.logger.info("Require collector, installing ...")
            with IOUtils.cd(Macros.collector_dir):
                BashUtils.run(f"mvn clean install -DskipTests", expected_return_code=0)
            # end with
            cls.collector_installed = True
        else:
            cls.logger.debug("Require collector, and already installed")
        # end if
        return
