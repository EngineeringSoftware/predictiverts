import os
from typing import *
from pathlib import Path
import pkg_resources
import random
import sys
import time
from seutil import CliUtils, IOUtils, LoggingUtils, BashUtils

from pts.Environment import Environment
from pts.Macros import Macros
from pts.Utils import Utils

# Check seutil version

EXPECTED_SEUTIL_VERSION = "0.5.1"
if pkg_resources.get_distribution("seutil").version < EXPECTED_SEUTIL_VERSION:
    print(
        f"seutil version does not meet expectation! Expected version: {EXPECTED_SEUTIL_VERSION}, current installed version: {pkg_resources.get_distribution('seutil').version}",
        file=sys.stderr)
    print(
        f"Hint: either upgrade seutil, or modify the expected version (after confirmation that the version will work)",
        file=sys.stderr)
    sys.exit(-1)
# end if


logging_file = Macros.python_dir / "experiment.log"
LoggingUtils.setup(filename=str(logging_file))

logger = LoggingUtils.get_logger(__name__)

# =========
# CONST
proj_logs = {
    "apache_commons-lang" : "ba8c6f6d", # used to be "4b718f6e"
    "apache_commons-net" : "dfd5f19d",
    "apache_commons-validator" : "97bb5737",
    "apache_commons-csv" : "2210c0b0",
    "asterisk-java_asterisk-java" : "a630a125",
    "apache_commons-configuration" : "801f4f4b",
    "frizbog_gedcom4j": "fcf39a01",
    "mikera_vectorz": "1e6769ef",
    "Bukkit_Bukkit": "8a1dbc38",
    "zeroturnaround_zt-exec": "acfe9d41",
}

defects4j_projs = {
    "apache_Lang-36": "9101b3c551ddc0721ba7ec4185a79adf0c03329d"
}


ML_models = [
    "Fail-Basic",
    "Fail-Code",
    "Fail-ABS",
    "Ekstazi-Basic",
    "Ekstazi-Code",
    "Ekstazi-ABS"
]


# ===================
# script for Download projects
def clone_checkout_projects():
    from pts.collector.DataCollector import DataCollector
    dc = DataCollector()
    dc.download_projects(proj_logs)


# script to run pit
def run_pit():
    for p in proj_logs:
        if (Macros.repos_downloads_dir/p).exists():
            try:
                with IOUtils.cd(Macros.repos_downloads_dir/p):
                    try:
                        BashUtils.run("mvn test", expected_return_code=0)
                    except:
                        print(f"{p} can not pass the test suite.")
                    try:
                        p_name = p.split('_')[1]
                        BashUtils.run(f"mvn org.pitest:pitest-maven:mutationCoverage > ../../docs/{p_name}-pit-log.txt", expected_return_code=0)
                    except:
                        print(f"{p} can not run pit.")
                # end with
            except:
                print("Unexpected error.")
        else:
            print(f"{p} does not exist!")
        # end if
    # end for


# ================
# Functions for processing data

def process_data(**options):
    from pts.processor.DataProcessor import DataProcessor
    dp = DataProcessor()
    dp.process(**options)


def clean_mutant_data(**options):
    """The class name in mutant-data-rts-tool.json is not correct"""
    project = options.get("project")
    collector_dir = Macros.repos_results_dir / project / "collector"

    test_list = IOUtils.load(collector_dir/"test2method.json").keys()
    rts_data = IOUtils.load(collector_dir/"mutant-data-rts-tool.json")
    for dp in rts_data:
        if isinstance(dp["killingTests"], list):
            for tc in dp["killingTests"]:
                test_class = tc[0].split('.')[-1]
                tc_list = tc[0].split('.')
                for r_tc in test_list:
                    if r_tc[:-1] == test_class:
                        tc_list[-1] = r_tc
                        tc[0] = '.'.join(tc_list)
                        tc[1] = tc[0]
                        print(f"New test_class {tc}")
                        break
                    # end if
                # end for
            # end for
        # end if
    # end for
    IOUtils.dump(collector_dir/"mutant-data-rts-tool-new.json", rts_data)


# =======================================
# Build mutation dataset

def get_mutant_data(**options):
    from pts.collector.mutation.MutantCollector import MutantCollector
    proj_list: List = Utils.get_option_as_list(options, "projects")
    which = options.get("which", "code")
    binary = Utils.get_option_as_boolean(options, "binary")
    mc = MutantCollector(proj_list)
    if which == "shas":
        sha = options.get("sha")
        mc.extract_muts_w_changed_files(sha)
    elif which == "code":
        mc.extract_muts(binary=binary)


def parse_pit_report(**options):
    """Parse the mutations.xml to get report json file."""
    from pts.collector.mutation.XMLParser import XMLParser
    xml_parser = XMLParser()
    project = options.get("project")
    proj_dir = Macros.repos_results_dir / project
    xml_parser.parse_pit_report(Path(proj_dir / "mutations.xml"), project, proj_dir)


def collect_tests_from_rtstool_changed_by_mutant(**options):
    from pts.collector.mutation.rtstool_tests_collector import main1
    proj = options.get("project")
    main1(proj)


def index_mutant_data(**options):
    from pts.collector.mutation.MutantCollector import MutantCollector
    projs: List[str] = Utils.get_option_as_list(options, "projects")
    binary = Utils.get_option_as_boolean(options, "binary")
    mc = MutantCollector(projs)
    mc.build_test_dictionary()
    mc.add_mutated_method_id(binary)


def run_starts_concurrent(**options):
    from pts.collector.mutation.MutantCollector import MutantCollector
    projs: List[str] = Utils.get_option_as_list(options, "projects")
    mc = MutantCollector(projs)
    for proj in projs:
        mc.concurrent_run_starts_proj(proj)


def test_runtime():
    '''test whether collecting time for each SHA works'''
    from pts.eval.eval_time import test_run_time_model
    test_run_time_model("apache_commons-validator")


def augment_eval_data(**options):
    from pts.collector.eval_data_collection import mutate_git_patch_multiprocess
    project = options.get("project")
    train_sha = proj_logs[project]
    mutate_git_patch_multiprocess(project)

# Run model

def run_ensemble_models(**options):
    """Get results for ensemble models
    Users must provide a list of sub models to ensemble
    """
    from pts.models.Ensemble import Ensemble
    models = Utils.get_option_as_list(options, "models")
    project = options.get("project")
    ensemble_model = Ensemble(models)
    ensemble_model.ensembling(project)


def run_boosting(**options):
    """Ensemble the six models together"""
    from pts.models.AdamBoost import AdamBoost
    project = options.get("project")
    model_data_dir = Macros.data_dir / "model-data" / "rank-model" / project.split('_')[1]
    sub_models = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS"]
    adamBoost = AdamBoost(model_data_dir, sub_models)
    adamBoost.AdaBoost_run(project)


def run_rank_model(**options):
    from pts.models.rank_model.TestSelectionModel import TestSelectionModel
    mode = options.get("mode", "train")
    proj = options.get("project", "apache-commons-dbcp").split('_')[1]
    data_type = options.get("which", "STARTS")
    config = options.get("config", "hybrid.yaml")
    config_file = IOUtils.load(Macros.config_dir/config)
    if "model_data_dir" not in config_file:
        model_data_dir = Macros.model_data_dir / "rank-model" / proj / data_type
    else:
        model_data_dir = Path(Macros.model_data_dir / config_file["model_data_dir"] /proj / data_type)
    if mode == "train":
        # Update the data set used
        if not model_data_dir.exists():
            IOUtils.mk_dir(model_data_dir)
        with IOUtils.cd(model_data_dir):
            if "Ekstazi" in data_type:
                BashUtils.run(f"cp {model_data_dir}/../Ekstazi/*.json {model_data_dir}/", expected_return_code=0)
            elif "Fail" in data_type:
                BashUtils.run(f"cp {model_data_dir}/../Fail/*.json {model_data_dir}/", expected_return_code=0)
        # end
        model = TestSelectionModel(config, model_data_dir)
        model.mode = "train"
        model.run_train()
    elif mode == "test":
        start_time = time.time()
        from pts.models.rank_model.TestSelectionModel import eval_model
        BashUtils.run(f"cp {model_data_dir.parent}/test.json {model_data_dir}", expected_return_code=0)
        eval_model(model_data_dir / "saved_models" / "best_model", model_data_dir)
        running_time = time.time() - start_time
        IOUtils.dump(model_data_dir / "results" / "total-running-time.json", {"time": running_time})
    elif mode == "analysis":
        from pts.models.rank_model.MetricsCollect import MetricsCollect
        mc = MetricsCollect()
        mc.collect_metrics(["Analyze-model"], project=options.get("project"), data_type=data_type)


def run_random_model(**options):
    from pts.models.RandomSelect import run_random_model
    project = options.get("project")
    eval_data_dir = Macros.eval_data_dir / "mutated-eval-data" / f"{project}-ag.json"
    output_dir = Macros.results_dir / "modelResults" / project.split('_')[1] / "Random"
    IOUtils.mk_dir(output_dir)
    run_random_model(eval_data_dir, output_dir)


def run_tfidf_baseline(**options):
    from pts.models.TFIDFbaseline import run_TFIDF_baseline
    project = options.get("project")
    run_TFIDF_baseline(project)


def preprocess_bm25_data(**options):
    from pts.models.BM25Baseline import pre_processing
    project = options.get("project")
    pre_processing(project)


def run_bm25_baseline(**options):
    from pts.models.BM25Baseline import run_BM25_baseline
    project = options.get("project")
    run_BM25_baseline(project)


def run_ealrts_baseline(**options):
    from pts.models.EALRTSBaseline import run_EALRTS_baseline
    project = options.get("project")
    model_type = options.get("model")
    run_EALRTS_baseline(project, model_type)


def run_linenum_baseline(**options):
    from pts.models.mutantbaseline import run_mutant_baseline_model
    project = options.get("project")
    search_span = options.get("search_span", 10)
    use_deleted = Utils.get_option_as_boolean(options, "use_deleted")
    all_covered = Utils.get_option_as_boolean(options, "all_covered")
    eval_data_dir = Macros.eval_data_dir / "mutated-eval-data" / f"{project}-ag.json"
    output_dir = Macros.results_dir / "modelResults" / project.split('_')[1] / "Baseline"
    IOUtils.mk_dir(output_dir)
    run_mutant_baseline_model(eval_data_dir, output_dir, project, search_span=search_span, use_deleted=use_deleted, all_covered=all_covered)


def collect_tools_data_for_mutant(**options):
    from pts.collector.mutation.rtstool_tests_collector import collect_rts_mutants
    proj = options.get("project")
    target_sha = proj_logs[proj]
    collect_rts_mutants(proj, target_sha)


# =====================================
# Make Plots and Tables
def make_plots(**options):
    from pts.Plot import Plot
    plot_maker = Plot()
    plot_maker.make_plots(**options)
    return

def make_tables(**options):
    from pts.Table import Table
    which = Utils.get_option_as_list(options, "which")
    table_maker = Table()
    table_maker.make_tables(which, options)


def filter_eval_data_line_num_and_ekstazi_tests(**options):
    from pts.collector.eval_data_collection import filter_project_change_lines_and_ekstazi_tests
    filter_project_change_lines_and_ekstazi_tests()


def filter_eval_data(**options):
    from pts.collector.eval_data_collection import filter_shas
    if options:
        proj = options.get("project")
        sha = Macros.eval_projects[proj]
        filter_shas({proj: sha})
    else:
        filter_shas(Macros.eval_projects)


def get_start_sha(**options):
    from pts.collector.eval_data_collection import get_start_shas
    get_start_shas(Macros.eval_projects)


# Collect eval data from real world shas
def collect_eval_data(**options):
    from pts.collector.eval_data_collection import filter_shas, main
    if options:
        proj = options.get("project")
        sha = Macros.eval_projects[proj]
        filter_shas({proj: sha})
        main({proj: sha})
    else:
        filter_shas(Macros.eval_projects)
        main(Macros.eval_projects)


def collect_qualified_name(**options):
    from pts.collector.eval_data_collection import qualified_test_name
    if options:
        proj = options.get("project")
        sha = Macros.eval_projects[proj]
        qualified_test_name({proj: sha})
    else:
        qualified_test_name(Macros.eval_projects)


def collect_eval_data_time(**options):
    from pts.collector.eval_data_collection import qualified_test_name_time
    if options:
        proj = options.get("project")
        sha = Macros.eval_projects[proj]
        qualified_test_name_time({proj: sha})
    else:
        qualified_test_name_time(Macros.eval_projects)


# Collect eval data by adding the deleted line numbers
def collect_eval_data_delete_line_number(**options):
    from pts.collector.eval_data_collection import collect_eval_data_adding_delete
    if options:
        proj = options.get("project")
        sha = Macros.eval_projects[proj]
        collect_eval_data_adding_delete({proj: sha})
    else:
        collect_eval_data_adding_delete(Macros.eval_projects)


# Collect eval data from real world shas without updating dependencies after first execution of RTS tools
def collect_eval_data_no_update(**options):
    from pts.collector.eval_data_collection import collect_eval_data_without_updating_dependencies
    if options:
        proj = options.get("project")
        sha = Macros.eval_projects[proj]
        collect_eval_data_without_updating_dependencies({proj: sha}, proj_logs)
    else:
        collect_eval_data_without_updating_dependencies(Macros.eval_projects, proj_logs)


# Test if project can compile
def if_project_can_compile(**options):
    from pts.collector.eval_data_collection import compile_project
    if options:
        proj = options.get("project")
        sha = proj_logs[proj]
        compile_project({proj: sha})
    else:
        compile_project(proj_logs)


def collect_eval_data_defect4j(**options):
    from pts.collector.eval_data_collection import collect_eval_data_defects4j
    proj = options.get("project")
    collect_eval_data_defects4j(proj)


def eval_real_failed_tests(**options):
    from pts.models.rank_model.model_runner import eval_real_tests
    rule = options.get("rule", None)
    if rule:
        eval_real_tests(rule=True)
    else:
        eval_real_tests(rule=False)

# python -m pts.main get_total_selection_time --projects="apache_commons-lang apache_commons-net apache_commons-validator
# apache_commons-csv asterisk-java_asterisk-java apache_commons-configuration frizbog_gedcom4j mikera_vectorz Bukkit_Bukkit zeroturnaround_zt-exec"
# --models="Fail-Basic Fail-Code Fail-ABS Ekstazi-Basic Ekstazi-Code Ekstazi-ABS BM25Baseline Fail-Basic-BM25Baseline
# Ekstazi-Basic-BM25Baseline boosting Ekstazi STARTS randomforest xgboost"
def get_total_selection_time(**options):
    from pts.models.rank_model.model_runner import get_total_selection_time
    projs = options.get("projects").split()
    models = options.get("models").split()
    for proj in projs:
        try:
            ekstazi_subset_models = set(models)
            ekstazi_subset_models.remove("randomforest")
            ekstazi_subset_models.remove("STARTS")
            get_total_selection_time(proj, ekstazi_subset_models, "Ekstazi")

            starts_subset_models = set(models)
            starts_subset_models.remove("Ekstazi")
            get_total_selection_time(proj, starts_subset_models, "STARTS")
        except Exception as e:
            print(e)
            continue


# python -m pts.main run_test_one_by_one --projects="apache_commons-lang apache_commons-net apache_commons-validator
# apache_commons-csv asterisk-java_asterisk-java apache_commons-configuration frizbog_gedcom4j mikera_vectorz zeroturnaround_zt-exec"
def run_test_one_by_one(**options):
    from pts.models.rank_model.model_runner import run_test_one_by_one
    projs = options.get("projects").split()
    run_test_one_by_one(projs)

#==========================
# Collect metrics
def collect_metrics(**options):
    from pts.collector.MetricsCollector import MetricsCollector
    collector = MetricsCollector()
    collector.collect_metrics(**options)

# Collect data
def collect_data(**options):
    from pts.collector.DataCollector import DataCollector
    collector = DataCollector()
    collector.collect_data(**options)


#==========================
# Collect projects

def collect_dependency_graph(**options):
    from pts.collector.min_dis_graph import main
    function_name = options.get("function_name")
    main(function_name)


def collect_projs_for_mt(**options):
    from pts.collector.DataCollector import DataCollector
    dc = DataCollector()
    dc.collect_projects(Macros.results_dir/"projects-github.txt", True)


def check_sha_first_time_failed_tests(**options):
    from pts.collector.eval_data_collection import check_sha_first_time_failed_tests
    projects = options.get("projects").split()
    print(projects)
    check_sha_first_time_failed_tests(projects)


def check_select_rate_outlier(**options):
    from pts.eval.eval_check import check_selection_rate_outlier
    projects = options.get("projects").split()
    models = options.get("models").split()
    check_selection_rate_outlier(projects, models)


def check_not_safe_selection(**options):
    from pts.eval.eval_check import check_not_safe_selection
    projects = options.get("projects").split()
    check_not_safe_selection(projects)


def check_line_number(**options):
    from pts.eval.eval_check import check_line_number
    projects = options.get("projects").split()
    check_line_number(projects)


def check_changed_file(**options):
    from pts.eval.eval_check import eval_changed_file
    projects = options.get("projects").split()
    eval_changed_file(projects)


def update_changed_file(**options):
    from pts.eval.eval_check import update_changed_file
    projects = options.get("projects").split()
    update_changed_file(projects)


def check_mutated_changed_file(**options):
    from pts.eval.eval_check import check_mutated_changed_file
    projects = options.get("projects").split()
    check_mutated_changed_file(projects)


def check_newly_added_tests(**options):
    from pts.eval.eval_check import check_newly_added_tests
    projects = options.get("projects").split()
    check_newly_added_tests(projects)


def check_bm25_performs_well(**options):
    from pts.eval.eval_check import check_bm25_performs_well
    projects = options.get("projects").split()
    models = options.get("models").split()
    if "TFIDFBaseline" in models:
        models.remove("TFIDFBaseline")
    check_bm25_performs_well(projects, models)


def check_diff_between_models(**options):
    from pts.eval.eval_check import check_diff_between_models
    projects = options.get("projects").split()
    models = options.get("models").split()
    if "TFIDFBaseline" in models:
        models.remove("TFIDFBaseline")
    check_diff_between_models(projects, models)


def check_first_time_failed_test():
    from pts.eval.eval_check import check_first_time_failed_test
    check_first_time_failed_test()


def collect_training_sha_info(**options):
    from pts.collector.eval_data_collection import collect_training_sha_info
    projs = options.get("project")
    collect_training_sha_info(projs)


#==== Parse projects to get method
def parse_projects(**options):
    from pts.collector.ProjectParser import ProjectParser
    proj = options.get("project")
    pp = ProjectParser()
    pp.parse_project(proj)


# ==========
# Main

def normalize_options(opts: dict) -> dict:
    # Set a different log file
    if "log_path" in opts:
        logger.info(f"Switching to log file {opts['log_path']}")
        LoggingUtils.setup(filename=opts['log_path'])
    # end if

    # Set debug mode
    if "debug" in opts and str(opts["debug"]).lower() != "false":
        Environment.is_debug = True
        logger.debug("Debug mode on")
        logger.debug(f"Command line options: {opts}")
    # end if

    # Set parallel mode - all automatic installations are disabled
    if "parallel" in opts and str(opts["parallel"]).lower() != "false":
        Environment.is_parallel = True
        logger.warning(f"Parallel mode on")
    # end if

    # Set/report random seed
    if "random_seed" in opts:
        Environment.random_seed = int(opts["random_seed"])
    else:
        Environment.random_seed = time.time_ns()
    # end if
    random.seed(Environment.random_seed)
    logger.info(f"Random seed is {Environment.random_seed}")

    # Automatically update data and results repo
    # Environment.require_data()
    # Environment.require_results()
    return opts

def add_file_dict():
    from pts.collector.eval_data_collection import add_file_diff
    add_file_diff(Path(Macros.data_dir/"raw-data"/"apache_commons-dbcp.json"), "apache_commons-dbcp")

def clone_projects_defectsEP(**options):
    """Clone all the projects used in defectsEP."""
    from pts.defectsep.run_defectsep import cloneProject, setD4JExamples
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))  # Dir of this script
    DEFECTS4J_BIN = SCRIPT_DIR + '/defects4j/framework/bin/defects4j'
    _DOWNLOADS_DIR = SCRIPT_DIR + '/_downloads'
    math_enable = options.get("math-enable", True)
    examples = setD4JExamples(math_enable)
    for example in examples:
        project = example.split('-')[0]
        example_number = example.split('-')[1]
        cloneProject(project, example_number)

def collect_eval_date(**options):
    from pts.collector.eval_data_collection import collect_date
    if options:
        proj = options.get("project")
        sha = Macros.eval_projects[proj]
        collect_date({proj: sha})
    else:
        collect_date(Macros.eval_projects)


# Short scripts, helper functions
def remove_dollar_sign_from_training_data():
    from pts.processor.RankProcess import RankProcessor
    RankProcessor.remove_dollar_signs_from_training_data()


def split_train_valid_data():
    from pts.processor.RankProcess import RankProcessor
    RankProcessor.split_valid_data_from_train()


if __name__ == "__main__":
    CliUtils.main(sys.argv[1:], globals(), normalize_options)
