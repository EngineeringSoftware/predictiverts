import re
from os import listdir
from pathlib import Path
from typing import *
import numpy as np
from collections import defaultdict
from seutil import IOUtils, LoggingUtils, latex
from seutil import BashUtils
from pts.Macros import Macros
from pts.Environment import Environment
from pts.Utils import Utils
from sklearn.metrics import recall_score
from statistics import mean
from pts.models.BM25Baseline import tokenize
from pts.main import proj_logs

class MetricsCollector:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self):
        self.output_dir = Macros.results_dir / "metrics"
        IOUtils.mk_dir(self.output_dir)
        self.ML_MODELS = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS"]
        self.PROJECTS = list(proj_logs.keys())
        self.PARTS = ["Ekstazi", "STARTS"]

    def collect_metrics(self, **options):
        which = Utils.get_option_as_list(options, "which")

        for item in which:
            self.logger.info(f"Collecting metrics: {item}; options: {options}")
            if item == "raw-dataset":
                self.collect_metrics_raw_dataset()
            elif item == "raw-eval-dataset-stats":
                self.collect_metrics_raw_eval_dataset()
            elif item == "model-data":
                self.collect_metrics_model_data()
            elif item == "pit-mutants":
                self.collect_metrics_mutations(**options)
            elif item == "mutated-eval-data":
                self.collect_metrics_mutated_eval_data(**options)
            elif item == "rank-model-eval-results":
                self.collect_rank_eval_results(**options)
            elif item == "rank-model-eval-results-IR-baseline":
                self.collect_rank_eval_results_IR_baseline(**options)
            elif item == "rank-model-eval-results-EALRTS-baseline":
                self.collect_rank_eval_results_EALRTS_baseline(**options)
            elif item == "boosting-eval-results-collect":
                self.collect_boosting_eval_results(**options)
            elif item == "move-rank-models-results":
                project_list = options.get("projects").split()
                model_list = Utils.get_option_as_list(options, "models")
                self.move_rank_models_results(project_list, model_list)
            elif item == "ensemble-model-results":
                self.collect_ensemble_models_eval_results(**options)
            elif item == "update-rank-eval-results":
                self.update_rank_eval_results(**options)
            elif item == "update-select-time-results":
                self.update_select_time_results(**options)
            elif item == "subset-selection-rate":
                self.update_subset_selection_rate(**options)
            elif item == "test-selection-metrics":
                self.collect_test_selection_metrics(**options)
            elif item == "average-metric-across-projects":
                self.collect_average_best_safe_selection_rate(**options)
            elif item == "select-from-subset-execution-time":
                projects = options.get("projects").split()
                models = options.get("models").split()
                self.select_from_subset_execution_time(projects, models)
            elif item == "select-from-subset-execution-time-change-selection-time":
                projects = options.get("projects").split()
                models = options.get("models").split()
                self.select_from_subset_execution_time(projects, models, True)
            elif item == "stats-reported-in-eval":
                # calculated stats and make latex macros reported in the paper
                # calculate stats: # improved, avg sel rate
                self.get_ml_improved_cases()
                self.get_ml_time_improved_cases()
                self.get_model_avg_best_selection_rate("Ekstazi")
                self.get_model_avg_best_selection_rate("STARTS")
                # make macros which can be used in eval.tex
                self.make_macros_paper_eval(self.PARTS)
            elif item == "split-eval-data":
                type = options.get("type")
                projects = options.get("projects").split()
                models = options.get("models").split()
                self.split_eval_data(type, projects, models)
            else:
                self.logger.warning(f"No such metrics {item}")
                raise NotImplementedError
            # end if
        # end for

    def make_macros_paper_eval(self, subsets: List[str]):
        """Write macros to files which will be used in eval.tex

        Macros:
        f"number-ml-better-than-{subset}"
        f"number-ml-shorter-than-{subset}"
        f"avg-best-select-rate-{model}-{subset}",
        f"avg-best-select-rate-{subset}",
        f"rate-{model}-reduce-{subset}"
        """
        fmt = ",.2f"
        macro_file = latex.File(Macros.paper_dir / "tables" / f"numbers-eval-used.tex")
        # f"number-ml-better-than-{subset}"
        ml_improved_metric_file = Macros.metrics_dir / "stats-models-better-than-paRTS.json"
        result_dict = IOUtils.load(ml_improved_metric_file)
        for subset in subsets:
            key = f"models-better-than-{subset}"
            macro_file.append_macro(latex.Macro(f"number-ml-better-than-{subset}", f"{len(result_dict[key])}"))
        # end for
        ml_shorter_metric_file = Macros.metrics_dir / "stats-models-shorter-than-paRTS.json"
        result_dict = IOUtils.load(ml_shorter_metric_file)
        for subset in subsets:
            key = f"models-shorter-than-{subset}"
            macro_file.append_macro(latex.Macro(f"number-ml-shorter-than-{subset}", f"{len(result_dict[key])}"))
        # end for

        # f"avg-best-select-rate-{model}-{subset}", f"avg-best-select-rate-{subset}", f"rate-{model}-reduce-{subset}"
        for subset in subsets:
            avg_best_select_rate_metric_file = Macros.metrics_dir/f"stats-model-{subset}-avg-best-select-rate.json"
            result_dict = IOUtils.load(avg_best_select_rate_metric_file)
            macro_file.append_macro(latex.Macro(f"avg-best-select-rate-{subset}", f"{result_dict[subset]*100:{fmt}}\%"))
            # end for
            for model in self.ML_MODELS:
                macro_file.append_macro(
                    latex.Macro(f"avg-best-select-rate-{model}-{subset}", f"{result_dict[model]*100:{fmt}}\%"))
                reduce_rate = (result_dict[subset] - result_dict[model]) / result_dict[subset]
                macro_file.append_macro(latex.Macro(f"rate-{model}-reduce-{subset}", f"{reduce_rate*100:{fmt}}\%"))
            # end for
        # end for
        macro_file.save()

    def get_ml_improved_cases(self):
        """Calculate cases that ml improves precision and save in metrics file.

        Output: write a dictionary containing the list of (project, model) Tuples
        to stats-models-better-than-paRTS.json
        """
        better_than_starts = []
        better_than_ekstazi = []
        for proj in self.PROJECTS:
            metric_file = Macros.results_dir / "metrics" / f"stats-{proj}-eval-results.json"
            metrics = IOUtils.load(metric_file)
            ekstazi_best_safe_select_rate = metrics["Fail-Basic"]["Ekstazi-best-safe-select-rate"]
            starts_best_safe_select_rate = metrics["Fail-Basic"]["STARTS-best-safe-select-rate"]
            for model in self.ML_MODELS:
                ekstazi_subset = metrics[model]["Ekstazi-select-subset-rate"]
                starts_subset = metrics[model]["STARTS-select-subset-rate"]
                if ekstazi_subset < ekstazi_best_safe_select_rate:
                    better_than_ekstazi.append((proj, model))
                if starts_subset < starts_best_safe_select_rate:
                    better_than_starts.append((proj, model))
            # end for
        # end for
        results = {
            "models-better-than-Ekstazi": better_than_ekstazi,
            "models-better-than-STARTS": better_than_starts
        }
        IOUtils.dump(Macros.results_dir/"metrics"/f"stats-models-better-than-paRTS.json", results)

    def get_ml_time_improved_cases(self):
        """Calculate cases that ml shortens the execution time and save in metrics file.

        Output: write a dictionary containing the list of (project, model) Tuples
        to stats-models-shorter-than-paRTS.json
        """
        results = {}
        for subset in self.PARTS:
            better_than_pa = []
            for proj in self.PROJECTS:
                metric_file = Macros.metrics_dir / f"stats-{proj}-avg-end2end-{subset}-subset-execution-time.json"
                metrics = IOUtils.load(metric_file)
                subset_execution_time = metrics[subset]
                for model in self.ML_MODELS:
                    model_execution_time = metrics[model]
                    if model_execution_time < subset_execution_time:
                        better_than_pa.append((proj, model))
                # end for
            # end for
            results[f"models-shorter-than-{subset}"] = better_than_pa
        # end for

        IOUtils.dump(Macros.metrics_dir / f"stats-models-shorter-than-paRTS.json", results)

    def get_model_avg_best_selection_rate(self, subset: str):
        """Compute avg best safe selection rate for all projects for each model on specified subset.

        Output: write a dictionary of models' average best selection rate across all the projects to
        stats-model-{subset}-avg-best-select-rate.json
        """
        result = {}
        for model in self.ML_MODELS + ["Ekstazi", "STARTS"]:
            model_best_selection_rate_list = []
            for proj in self.PROJECTS:
                metric_file = Macros.results_dir / "metrics" / f"stats-{proj}-eval-results.json"
                metrics = IOUtils.load(metric_file)
                if model in self.PARTS:
                    model_best_safe_select_rate = metrics["Fail-Basic"][f"{model}-best-safe-select-rate"]
                else:
                    model_best_safe_select_rate = metrics[model][f"{subset}-select-subset-rate"]
                model_best_selection_rate_list.append(model_best_safe_select_rate)
            # end for
            result[model] = mean(model_best_selection_rate_list)
        # end for

        IOUtils.dump(Macros.metrics_dir/f"stats-model-{subset}-avg-best-select-rate.json", result)

    def move_rank_models_results(self, project_list: List[str], model_list: List[str]):
        """Copy models' results from data dir to the result dir for sync."""
        for project in project_list:
            proj_name = project.split('_')[1]
            proj_result_dir = Macros.model_data_dir / "rank-model" / proj_name
            IOUtils.mk_dir(Macros.results_dir / "modelResults" / proj_name)
            for model in model_list:
                IOUtils.mk_dir(Macros.results_dir / "modelResults" / proj_name / model)
                pm_result_dir = Macros.results_dir / "modelResults" / proj_name / model
                with IOUtils.cd(proj_result_dir / model / "results"):
                    BashUtils.run(f"cp *.json {pm_result_dir}", expected_return_code=0)
                    BashUtils.run(f"cp test-output/*.json {pm_result_dir}")
        # end for

    def collect_average_best_safe_selection_rate(self, **options):
        """Cacluate the average best safe selection rates across all (10) projects we have.
        Models: 6 models, combination models
        step1: load the existing json files for all the models and projects
        step2: calculate the average
        step3: store in a new json file
        """

        ML_MODELS = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS"]
        COMB_MODELS = ["Ekstazi-Basic-BM25Baseline", "Fail-Basic-BM25Baseline"]
        ENSEMBLE_MODELS = ["boosting"]
        TOOLS = ["Ekstazi", "STARTS"]
        best_safe_selection_rate = defaultdict(list)
        starts_best_safe_selection_rate = defaultdict(list)
        ekstazi_best_safe_selection_rate = defaultdict(list)
        avg_best_safe_selection_rate = {}

        # First load the 6 models results for select from all tests
        avg_best_safe_selection_rate["All"] = {}
        projects = Utils.get_option_as_list(options, "projects")
        for proj in projects:
            ml_model_metrics = IOUtils.load(Macros.metrics_dir / f"stats-{proj}-eval-results.json")
            for model in ML_MODELS:
                best_safe_selection_rate[model].append(ml_model_metrics[model]["best-safe-select-rate"])
            # end for
            for model in TOOLS:
                best_safe_selection_rate[model].append(ml_model_metrics["Fail-Basic"][f"{model}-best-safe-select-rate"])
            # end for
            for model in COMB_MODELS:
                comb_model_metrics = IOUtils.load(Macros.metrics_dir / f"stats-{proj}-{model}-eval-results.json")
                best_safe_selection_rate[model].append(comb_model_metrics[model]["best-safe-select-rate"])
            # end for
            for model in ENSEMBLE_MODELS:
                ensemble_model_metrics = IOUtils.load(Macros.metrics_dir / f"stats-{proj}-boosting-eval-results.json")
                best_safe_selection_rate[model].append(ensemble_model_metrics[model]["best-safe-select-rate"])
            # end for
        # get the average across project
        for model, results in best_safe_selection_rate.items():
            avg_best_safe_selection_rate["All"][model] = mean(results)
        # end for

        # Second load the 6 models results for select from Ekstazi tests
        avg_best_safe_selection_rate["Ekstazi"] = {}
        for proj in projects:
            ml_model_metrics = IOUtils.load(Macros.metrics_dir / f"stats-{proj}-eval-results.json")
            for model in ML_MODELS:
                ekstazi_best_safe_selection_rate[model].append(ml_model_metrics[model]["Ekstazi-select-subset-rate"])
            # end for
            for model in TOOLS:
                ekstazi_best_safe_selection_rate[model].append(
                    ml_model_metrics["Fail-Basic"][f"{model}-best-safe-select-rate"])
            # end for
            for model in COMB_MODELS:
                comb_model_metrics = IOUtils.load(Macros.metrics_dir / f"stats-{proj}-{model}-eval-results.json")
                ekstazi_best_safe_selection_rate[model].append(comb_model_metrics[model]["Ekstazi-select-subset-rate"])
            # end for
            for model in ENSEMBLE_MODELS:
                ensemble_model_metrics = IOUtils.load(Macros.metrics_dir / f"stats-{proj}-boosting-eval-results.json")
                ekstazi_best_safe_selection_rate[model].append(
                    ensemble_model_metrics[model]["Ekstazi-select-subset-rate"])
            # end for
        # end for
        for model, results in ekstazi_best_safe_selection_rate.items():
            avg_best_safe_selection_rate["Ekstazi"][model] = mean(results)
        # end for

        # Third load the 6 models results for select from STARTS tests
        avg_best_safe_selection_rate["STARTS"] = {}
        for proj in projects:
            ml_model_metrics = IOUtils.load(Macros.metrics_dir / f"stats-{proj}-eval-results.json")
            for model in ML_MODELS:
                starts_best_safe_selection_rate[model].append(ml_model_metrics[model]["STARTS-select-subset-rate"])
            # end for
            for model in TOOLS:
                starts_best_safe_selection_rate[model].append(
                    ml_model_metrics["Fail-Basic"][f"{model}-best-safe-select-rate"])
            # end for
            for model in COMB_MODELS:
                comb_model_metrics = IOUtils.load(Macros.metrics_dir / f"stats-{proj}-{model}-eval-results.json")
                starts_best_safe_selection_rate[model].append(comb_model_metrics[model]["STARTS-select-subset-rate"])
            # end for
            for model in ENSEMBLE_MODELS:
                ensemble_model_metrics = IOUtils.load(Macros.metrics_dir / f"stats-{proj}-boosting-eval-results.json")
                starts_best_safe_selection_rate[model].append(
                    ensemble_model_metrics[model]["STARTS-select-subset-rate"])
            # end for
        # end for

        # get the average across project
        for model, results in starts_best_safe_selection_rate.items():
            avg_best_safe_selection_rate["STARTS"][model] = mean(results)
        # end for

        IOUtils.dump(Macros.metrics_dir / "stats-avg-best-safe-selection-rate.json", avg_best_safe_selection_rate)

    @classmethod
    def collect_random_eval_results(cls, project: str):
        src_results_file = Macros.results_dir / "modelResults" / project.split('_')[1] / "Random" / "random-model" \
                                                                                                    "-eval-results.json"
        random_apfd_list = []
        random_best_select_rate = []
        random_first_fail_test_rank_list = []
        random_recall_list = []
        random_select_rate_list = []
        random_results = IOUtils.load(src_results_file)
        for s, stats in random_results.items():
            random_apfd_list.append(stats["random-apfd"])
            random_best_select_rate.append(stats["random-best-select-rate"])
            random_first_fail_test_rank_list.append(stats["random-first-fail-test-rank"])
            random_recall_list.append(stats["random-recall"])
            random_select_rate_list.append(stats["random-select-rate"])
        # end for
        avg_apfd = sum(random_apfd_list) / len(random_apfd_list)
        avg_best_select_rate = sum(random_best_select_rate) / len(random_best_select_rate)
        avg_first_test_rank = sum(random_first_fail_test_rank_list) / len(random_first_fail_test_rank_list)
        avg_random_recall = sum(random_recall_list) / len(random_recall_list)
        avg_select_rate = sum(random_select_rate_list) / len(random_select_rate_list)
        return avg_apfd, avg_best_select_rate, avg_first_test_rank, 100 * avg_random_recall, avg_select_rate

    @classmethod
    def collect_baseline_eval_results(cls, project: str, search_span: int):
        """Collect the line-number based baseline. 10-line, 20-line"""
        src_results_file = Macros.results_dir / "modelResults" / project.split('_')[
            1] / "Baseline" / f"baseline-{search_span}-model" \
                              "-eval-results.json"
        baseline_recall_list = []
        baseline_select_rate_list = []
        baseline_not_covered_files = []
        baseline_results = IOUtils.load(src_results_file)
        no_test_select = 0
        for s, stats in baseline_results.items():
            baseline_recall_list.append(stats["baseline-recall"])
            baseline_select_rate_list.append(stats["baseline-select-rate"])
            baseline_not_covered_files.append(stats["not-covered-file"])
            if stats["baseline-select-rate"] == 0.0:
                no_test_select += 1
        # end for
        avg_baseline_recall = sum(baseline_recall_list) / len(baseline_recall_list)
        avg_select_rate = sum(baseline_select_rate_list) / len(baseline_select_rate_list)
        avg_not_covered_file = sum(baseline_not_covered_files) / len(baseline_not_covered_files)
        no_test_select_pct = no_test_select / len(baseline_not_covered_files)
        return 100 * avg_baseline_recall, avg_select_rate, avg_not_covered_file, no_test_select_pct

    @classmethod
    def collect_del_baseline_eval_results(cls, project: str, search_span: int):
        """Collect the line-number based baseline.del-10-model"""
        src_results_file = Macros.results_dir / "modelResults" / project.split('_')[
            1] / "Baseline" / f"baseline-del-{search_span}-model" \
                              "-eval-results.json"
        baseline_recall_list = []
        baseline_select_rate_list = []
        baseline_not_covered_files = []
        baseline_results = IOUtils.load(src_results_file)
        no_test_select = 0
        for s, stats in baseline_results.items():
            baseline_recall_list.append(stats["baseline-recall"])
            baseline_select_rate_list.append(stats["baseline-select-rate"])
            baseline_not_covered_files.append(stats["not-covered-file"])
            if stats["baseline-select-rate"] == 0.0:
                no_test_select += 1
        # end for
        avg_baseline_recall = sum(baseline_recall_list) / len(baseline_recall_list)
        avg_select_rate = sum(baseline_select_rate_list) / len(baseline_select_rate_list)
        avg_not_covered_file = sum(baseline_not_covered_files) / len(baseline_not_covered_files)
        no_test_select_pct = no_test_select / len(baseline_not_covered_files)
        return 100 * avg_baseline_recall, avg_select_rate, avg_not_covered_file, no_test_select_pct

    @classmethod
    def collect_all_covered_baseline_eval_results(cls, project: str, search_span: int):
        """Collect the line-number based baseline.all-covered-10-model"""
        src_results_file = Macros.results_dir / "modelResults" / project.split('_')[
            1] / "Baseline" / f"baseline-all-mutant-{search_span}-model-eval-results.json"
        baseline_recall_list = []
        baseline_select_rate_list = []
        baseline_not_covered_files = []
        baseline_results = IOUtils.load(src_results_file)
        no_test_select = 0
        for s, stats in baseline_results.items():
            baseline_recall_list.append(stats["baseline-recall"])
            baseline_select_rate_list.append(stats["baseline-select-rate"])
            baseline_not_covered_files.append(stats["not-covered-file"])
            if stats["baseline-select-rate"] == 0.0:
                no_test_select += 1
        # end for
        avg_baseline_recall = sum(baseline_recall_list) / len(baseline_recall_list)
        avg_select_rate = sum(baseline_select_rate_list) / len(baseline_select_rate_list)
        avg_not_covered_file = sum(baseline_not_covered_files) / len(baseline_not_covered_files)
        no_test_select_pct = no_test_select / len(baseline_not_covered_files)
        return 100 * avg_baseline_recall, avg_select_rate, avg_not_covered_file, no_test_select_pct

    def update_rank_eval_results(self, **options):
        """ Update eval results for with Ekstazi and STARTS with no dependencies update."""
        project = options.get("project")
        project_results = {}
        raw_eval_data_no_update = IOUtils.load(Macros.raw_eval_data_no_dep_updated_dir / f"{project}.json")
        mutated_eval_data = IOUtils.load(Macros.data_dir / "mutated-eval-data" / f"{project}-ag.json")
        ekstazi_selection_rate = []
        starts_selection_rate = []
        ekstazi_recall_rate = []
        starts_recall_rate = []
        for mutated_eval_data_item in mutated_eval_data:
            currentSHA = mutated_eval_data_item["commit"][:8]
            for raw_eval_data_no_update_item in raw_eval_data_no_update:
                if raw_eval_data_no_update_item["commit"] == currentSHA:
                    all_tests = raw_eval_data_no_update_item["passed_test_list"]
                    ekstazi_test_list = raw_eval_data_no_update_item["ekstazi_test_list_no_dep_update"]
                    starts_test_list = raw_eval_data_no_update_item["starts_test_list_no_dep_update"]
                    failed_test_list = mutated_eval_data_item["failed_test_list"]
                    ekstazi_test_label_list = [0 for _ in range(len(all_tests))]
                    starts_test_label_list = [0 for _ in range(len(all_tests))]
                    failed_test_label_list = [0 for _ in range(len(all_tests))]
                    for index, test in enumerate(all_tests):
                        if test in ekstazi_test_list:
                            ekstazi_test_label_list[index] = 1
                        if test in starts_test_list:
                            starts_test_label_list[index] = 1
                        if test in failed_test_list:
                            failed_test_label_list[index] = 1

                    ekstazi_selection_rate.append(len(ekstazi_test_list) / len(all_tests))
                    starts_selection_rate.append(len(starts_test_list) / len(all_tests))

                    ekstazi_recall_rate_item = recall_score(failed_test_label_list, ekstazi_test_label_list)
                    if ekstazi_recall_rate_item < 1:
                        print(mutated_eval_data_item["commit"], "ekstazi recall rate < 1")
                    ekstazi_recall_rate.append(ekstazi_recall_rate_item)

                    starts_recall_rate_item = recall_score(failed_test_label_list, starts_test_label_list)
                    starts_recall_rate.append(starts_recall_rate_item)
                    if starts_recall_rate_item < 1:
                        print(mutated_eval_data_item["commit"], "starts recall rate < 1")
                    break
        print("ekstazi select rate", ekstazi_selection_rate)
        print("starts select rate", starts_selection_rate)
        project_results["avg-Ekstazi-no-update-select-rate"] = sum(ekstazi_selection_rate) / len(ekstazi_selection_rate)
        project_results["avg-STARTS-no-update-select-rate"] = sum(starts_selection_rate) / len(starts_selection_rate)
        project_results["avg-Ekstazi-no-update-recall"] = sum(ekstazi_recall_rate) / len(ekstazi_recall_rate) * 100
        project_results["avg-STARTS-no-update-recall"] = sum(starts_recall_rate) / len(starts_recall_rate) * 100
        metric_file = Macros.results_dir / "metrics" / f"stats-{project}-eval-results-no-dep-update.json"
        IOUtils.dump(metric_file, project_results, IOUtils.Format.jsonNoSort)

    def update_select_time_results(self, **options):
        project = options.get("project")
        mutated_eval_data = IOUtils.load(f'{Macros.data_dir}/mutated-eval-data-adding-time/{project}.json')
        if len(mutated_eval_data) == 0:
            print(project, "empty")
            return
        model_to_time = defaultdict(float)
        for mutated_eval_data_item in mutated_eval_data:
            for modelname, modeltime in mutated_eval_data_item.items():
                if modelname.endswith("_time"):
                    model_to_time[modelname.replace("_time", "-time")] += modeltime
        for modelname, modeltime in model_to_time.items():
            model_to_time[modelname] = modeltime / len(mutated_eval_data)
        metric_file = Macros.results_dir / "metrics" / f"stats-{project}-select-time.json"
        IOUtils.dump(metric_file, model_to_time, IOUtils.Format.jsonNoSort)

    def update_subset_selection_rate(self, **options):
        from pts.models.rank_model.MetricsCollect import MetricsCollect
        mc = MetricsCollect()
        project = options.get("project")
        metric_file = IOUtils.load(Macros.results_dir / "metrics" / f"stats-{project}-eval-results.json")
        for m in ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS"]:
            for tool in ["STARTS", "Ekstazi"]:
                full, subset = mc.get_best_selection_rate(project, data_type=m, subset=tool)
                metric_file[m]["best-safe-select-rate"] = full
                metric_file[m][f"{tool}-best-safe-select-rate"] = subset
            # end for
        # end for
        IOUtils.dump(Macros.results_dir / "metrics" / f"stats-{project}-eval-results.json", metric_file)

    def collect_test_selection_metrics(self, **options):
        """Get metrics: the number of missed failed tests, pct newly added missed tests etc."""
        project = options.get("project")
        project_results = {}
        model_results_dir = Macros.model_data_dir / "rank-model" / project.split('_')[1] / "Fail-Code" / "results"
        overall_models_results = IOUtils.load(model_results_dir / "test-output" / "per-file-result.json")
        project_results["total_selected_failed_tests"] = overall_models_results["total_selected_failed_tests"]
        project_results["total_missed_failed_tests"] = overall_models_results["total_missed_failed_tests"]
        project_results["pct_newly_added_missed_failed_tests"] = overall_models_results[
            "pct_newly_add_missed_failed_tests"]
        metric_file = Macros.results_dir / "metrics" / f"stats-{project}-test-selection-metrics.json"
        IOUtils.dump(metric_file, project_results)

    def collect_rank_eval_results_IR_baseline(self, **options):
        project = options.get("project")
        project_results = {}
        for m in ["TFIDFBaseline", "BM25Baseline"]:
            project_results[m] = self.collect_rank_eval_results_for_each_model(m, project)
        metric_file = Macros.results_dir / "metrics" / f"stats-{project}-IR-baseline-eval-results.json"
        IOUtils.dump(metric_file, project_results, IOUtils.Format.jsonPretty)

    def collect_rank_eval_results_EALRTS_baseline(self, **options):
        project = options.get("project")
        project_results = {}
        if project == "apache_commons-csv":
            return
        for m in ["randomforest", "xgboost"]:
            project_results[m] = self.collect_rank_eval_results_for_each_model(m, project)
        metric_file = Macros.results_dir / "metrics" / f"stats-{project}-EALRTS-eval-results.json"
        IOUtils.dump(metric_file, project_results, IOUtils.Format.jsonPretty)

    def collect_rank_eval_results_triplet(self, **options):
        project = options.get("project")
        project_results = {}
        m = "triplet"
        project_results[m] = self.collect_rank_eval_results_for_each_model(m, project)
        metric_file = Macros.results_dir / "metrics" / f"stats-{project}-triplet-eval-results.json"
        IOUtils.dump(metric_file, project_results, IOUtils.Format.jsonPretty)

    def collect_boosting_eval_results(self, **options):
        """Collect metrics and save the results of boosting models"""
        project_results = {}
        project = options.get("project")
        project_results["boosting"] = self.collect_rank_eval_results_for_each_model("boosting", project)
        metric_file = Macros.results_dir / "metrics" / f"stats-{project}-boosting-eval-results.json"
        IOUtils.dump(metric_file, project_results, IOUtils.Format.jsonPretty)

    def collect_ensemble_models_eval_results(self, **options):
        """Collect metrics and save to results/metrics"""
        models = Utils.get_option_as_list(options, "models")
        project_results = {}
        m_name = "-".join(models)
        project = options.get("project")
        project_results[m_name] = self.collect_rank_eval_results_for_each_model(m_name, project)
        metric_file = Macros.results_dir / "metrics" / f"stats-{project}-{m_name}-eval-results.json"
        IOUtils.dump(metric_file, project_results, IOUtils.Format.jsonPretty)

    # m is the model name and project is the project name
    def collect_rank_eval_results_for_each_model(self, m: str, project: str):
        res = {}

        model_results_dir = Macros.results_dir / "modelResults" / project.split('_')[1] / m
        sha_2_best_selection_rate = IOUtils.load(model_results_dir / "best-select-rate-per-SHA.json")
        tmp = []
        for k, v in sha_2_best_selection_rate.items():
            if type(v) != str:
                tmp.append(v)
        res["avg-best-select-rate"] = sum(tmp) / len(tmp)

        sha_2_perfect_selection_rate = IOUtils.load(model_results_dir / "perfect-select-rate-per-SHA.json")
        tmp = []
        for k, v in sha_2_perfect_selection_rate.items():
            tmp.append(v)
        res["avg-perfect-select-rate"] = sum(tmp) / len(tmp)

        sha_2_best_rank = IOUtils.load(model_results_dir / "best-rank-per-SHA.json")
        tmp = []
        for k, v in sha_2_best_rank.items():
            tmp.append(v)
        res["avg-highest-rank"] = sum(tmp) / len(tmp)
        res["worst-highest-rank"] = max(tmp)

        sha_2_lowests_threshold = IOUtils.load(model_results_dir / "lowest-threshold-per-SHA.json")
        tmp = []
        for k, v in sha_2_lowests_threshold.items():
            tmp.append(v)
        res["avg-best-threshold"] = sum(tmp) / len(tmp)

        sha_2_apfd = IOUtils.load(model_results_dir / "apfd-per-sha.json")
        tmp = []
        for k, v in sha_2_apfd.items():
            tmp.append(v)
        res["avg-apfd"] = sum(tmp) / len(tmp)

        sha_2_tools_selection_rate = IOUtils.load(model_results_dir / "tools-select-rate-per-SHA.json")
        ek = []
        sts = []
        for k, v in sha_2_tools_selection_rate.items():
            ek.append(v["Ekstazi"])
            sts.append(v["STARTS"])
        res["avg-Ekstazi-select-rate"] = sum(ek) / len(ek)
        res["avg-STARTS-select-rate"] = sum(sts) / len(sts)
        res["Ekstazi-best-safe-select-rate"] = max(ek)
        res["STARTS-best-safe-select-rate"] = max(sts)
        best_safe_selection_rate = IOUtils.load(model_results_dir / "best-safe-selection-rate.json")
        res["best-safe-select-rate"] = best_safe_selection_rate["best-safe-selection-rate"]
        res["avg-safe-select-rate"] = best_safe_selection_rate["avg-safe-selection-rate"]
        res["Ekstazi-select-subset-rate"] = best_safe_selection_rate["Ekstazi-subset-best-safe-selection-rate"]
        res["Ekstazi-avg-select-subset-rate"] = best_safe_selection_rate["Ekstazi-subset-avg-safe-selection-rate"]
        res["STARTS-select-subset-rate"] = best_safe_selection_rate["STARTS-subset-best-safe-selection-rate"]
        res["STARTS-avg-select-subset-rate"] = best_safe_selection_rate["STARTS-subset-avg-safe-selection-rate"]

        if (Macros.model_data_dir / "rank-model" / project.split('_')[
            1] / m / "results" / "test-output" / "per-file-result.json").exists():
            overall_models_results = IOUtils.load(Macros.model_data_dir / "rank-model" / project.split('_')[
                1] / m / "results" / "test-output" / "per-file-result.json")
            res["avg-recall"] = overall_models_results["recall"]
            res["selection-rate"] = overall_models_results["selected_pct"]
            res["avg-Ekstazi-recall"] = overall_models_results["ekstazi_recall"]
            res["avg-STARTS-recall"] = overall_models_results["starts_recall"]
        return res

    def collect_rank_eval_results(self, **options):
        """ Collect eval results for projets."""
        project = options.get("project")
        project_results = {}
        for m in self.ML_MODELS:
            project_results[m] = self.collect_rank_eval_results_for_each_model(m, project)
        # end for
        project_results["Random"] = {}
        project_results["Random"]["avg-apfd"], project_results["Random"]["avg-best-select-rate"], \
        project_results["Random"]["avg-highest-rank"], project_results["Random"]["avg-recall"], \
        project_results["Random"]["selection-rate"] = self.collect_random_eval_results(project)
        project_results["Random"]["best-safe-select-rate"] = -1.0
        project_results["Random"]["avg-best-threshold"] = -1.0
        project_results["Random"]["worst-highest-rank"] = 1.0

        # Hardcode for two baselines for now TODO
        project_results["Baseline-10"] = {}
        project_results["Baseline-10"]["avg-recall"], project_results["Baseline-10"]["selection-rate"], \
        project_results["Baseline-10"]["avg-not-covered-file"], project_results["Baseline-10"]["pct-no-test-select"] \
            = self.collect_baseline_eval_results(project, search_span=10)
        project_results["Baseline-10"]["avg-apfd"], project_results["Baseline-10"]["avg-best-select-rate"], \
        project_results["Baseline-10"]["avg-highest-rank"], project_results["Baseline-10"]["best-safe-select-rate"], \
        project_results["Baseline-10"]["avg-best-threshold"], project_results["Baseline-10"][
            "worst-highest-rank"] = -1.0, -1.0, -1.0, -1.0, -1.0, -1.0

        project_results["Baseline-20"] = {}
        project_results["Baseline-20"]["avg-recall"], project_results["Baseline-20"]["selection-rate"], \
        project_results["Baseline-20"]["avg-not-covered-file"], project_results["Baseline-20"]["pct-no-test-select"] \
            = self.collect_baseline_eval_results(project, search_span=20)
        project_results["Baseline-20"]["avg-apfd"], project_results["Baseline-20"]["avg-best-select-rate"], \
        project_results["Baseline-20"]["avg-highest-rank"], project_results["Baseline-20"]["best-safe-select-rate"], \
        project_results["Baseline-20"]["avg-best-threshold"], project_results["Baseline-20"][
            "worst-highest-rank"] = -1.0, -1.0, -1.0, -1.0, -1.0, -1.0

        project_results["Baseline-del"] = {}
        project_results["Baseline-del"]["avg-recall"], project_results["Baseline-del"]["selection-rate"], \
        project_results["Baseline-del"]["avg-not-covered-file"], project_results["Baseline-del"]["pct-no-test-select"] \
            = self.collect_del_baseline_eval_results(project, search_span=10)
        project_results["Baseline-del"]["avg-apfd"], project_results["Baseline-del"]["avg-best-select-rate"], \
        project_results["Baseline-del"]["avg-highest-rank"], project_results["Baseline-del"]["best-safe-select-rate"], \
        project_results["Baseline-del"]["avg-best-threshold"], project_results["Baseline-del"][
            "worst-highest-rank"] = -1.0, -1.0, -1.0, -1.0, -1.0, -1.0

        project_results["Baseline-all"] = {}
        project_results["Baseline-all"]["avg-recall"], project_results["Baseline-all"]["selection-rate"], \
        project_results["Baseline-all"]["avg-not-covered-file"], project_results["Baseline-all"]["pct-no-test-select"] \
            = self.collect_all_covered_baseline_eval_results(project, search_span=10)
        project_results["Baseline-all"]["avg-apfd"], project_results["Baseline-all"]["avg-best-select-rate"], \
        project_results["Baseline-all"]["avg-highest-rank"], project_results["Baseline-all"]["best-safe-select-rate"], \
        project_results["Baseline-all"]["avg-best-threshold"], project_results["Baseline-all"][
            "worst-highest-rank"] = -1.0, -1.0, -1.0, -1.0, -1.0, -1.0

        metric_file = Macros.results_dir / "metrics" / f"stats-{project}-eval-results.json"
        IOUtils.dump(metric_file, project_results, IOUtils.Format.jsonNoSort)

    def collect_metrics_mutated_eval_data(self, **options):
        """
        Collect stats for the mutated eval data using universal-mutator
        """
        project = options.get("project")
        data_dir = Macros.eval_data_dir / "mutated-eval-data" / f"{project}-ag.json"
        fail_test_class = 0
        total_shas = set()
        total_mutants = 0
        val_data = IOUtils.load(data_dir)
        cleaned_val_data = []

        for d in val_data:
            if len(d["failed_test_list"]) == 0:
                continue
            total_shas.add(d["commit"].split('-')[0])
            total_mutants += 1
            fail_test_class += len(d["failed_test_list"])
            cleaned_val_data.append(d)
        IOUtils.dump(Macros.eval_data_dir / "mutated-eval-data" / f"{project}-ag.json", cleaned_val_data)

        dataset_stats = {
            "eval-shas": len(total_shas),
            "mutants": total_mutants,
            "fail-test": fail_test_class
        }
        proj = project.split('_')[1]
        metric_file = Macros.results_dir / "metrics" / f"stats-{project}-mutated-eval-dataset.json"
        IOUtils.dump(metric_file, dataset_stats, IOUtils.Format.jsonNoSort)

    def collect_metrics_mutations(self, **options):
        """
        Collect metrics of the mutants for each project.
        Two files are generated:
        stats-project-mutants.json: the mutants generated by PIT for the projects
        stats-project-recover-mutants.json: the mutants can be recovered
        """
        projs: List[str] = Utils.get_option_as_list(options, "projects")
        # mutants_stats = dict()
        for proj in projs:
            proj_mutant_stats = dict()
            data_dir = Macros.repos_results_dir / proj / "collector" / "mutant-data.json"
            objs = IOUtils.load_json_stream(data_dir)
            for obj in objs:
                if obj["mutator"] not in proj_mutant_stats:
                    proj_mutant_stats[obj["mutator"]] = {
                        "KILLED": 0,
                        "NO_COVERAGE": 0,
                        "SURVIVED": 0,
                        "TIMED_OUT": 0,
                        "total": 0
                    }
                # end if
                proj_mutant_stats[obj["mutator"]][obj["status"]] += 1
                proj_mutant_stats[obj["mutator"]]["total"] += 1
            # end for
            # Check whethre mutators are complete
            if "InvertNegsMutator" not in proj_mutant_stats:
                proj_mutant_stats["InvertNegsMutator"] = {
                    "KILLED": 0,
                    "NO_COVERAGE": 0,
                    "SURVIVED": 0,
                    "TIMED_OUT": 0,
                    "total": 0
                }
            # end if
            IOUtils.dump(self.output_dir / f"stats-{proj}-recover-mutants.json", proj_mutant_stats)
        # end for

        # mutants_stats = dict()
        for proj in projs:
            proj_mutant_stats = dict()
            data_dir = Macros.repos_results_dir / proj / f"{proj}-default-mutation-report.json"
            objs = IOUtils.load_json_stream(data_dir)
            for obj in objs:
                if obj["mutator"] not in proj_mutant_stats:
                    proj_mutant_stats[obj["mutator"]] = {
                        "KILLED": 0,
                        "NO_COVERAGE": 0,
                        "SURVIVED": 0,
                        "TIMED_OUT": 0,
                        "total": 0
                    }
                # end if
                proj_mutant_stats[obj["mutator"]][obj["status"]] += 1
                proj_mutant_stats[obj["mutator"]]["total"] += 1
            # end for
            IOUtils.dump(self.output_dir / f"stats-{proj}-mutants.json", proj_mutant_stats)
        # end for

        pit_running_time = None
        runTime = re.compile(".*Total time: (.*)")
        for proj in projs:
            with IOUtils.cd(Macros.project_dir / "docs"):
                project_name = proj.split('_')[1]
                if (Path(f"{project_name}-pit-log.txt")).exists():
                    with open(f"{project_name}-pit-log.txt") as f:
                        for line in f.readlines():
                            match_res = runTime.match(line)
                            if match_res:
                                pit_running_time = match_res.group(1)
                                break
                            # end if
                        # end for
                    # end with
                else:
                    print(f"{project_name} log file does not exist.")
                    raise FileNotFoundError
            pit_log_file = IOUtils.load(self.output_dir / f"stats-{proj}-pitlog.json")
            if pit_running_time:
                pit_log_file["time"] = pit_running_time.strip()
            else:
                print(f"{project_name} log file does not contain time.")
                raise FileNotFoundError
            IOUtils.dump(self.output_dir / f"stats-{proj}-pitlog.json", pit_log_file)
        # end for

    def collect_metrics_seq2pred(self):
        """Get the stats for mutation test data."""
        model_data_dir = Macros.data_dir / "model-data" / "seq2pred-data"
        mut_dataset_stats = dict()
        for data_type in [Macros.train, Macros.test]:
            objs = IOUtils.load_json_stream(model_data_dir / f"{data_type}.json")
            failed_pairs = 0
            passed_pairs = 0
            for obj in objs:
                if obj["label"] == 1:
                    failed_pairs += 1
                elif obj["label"] == 0:
                    passed_pairs += 1
            # end for
            mut_dataset_stats[data_type] = {
                "passed": passed_pairs,
                "failed": failed_pairs
            }
        # end for
        IOUtils.dump(self.output_dir / "mutation-dataset-stats.json", mut_dataset_stats)

    def collect_metrics_model_data(self):
        """Get and save the stat for the dataset for model"""
        model_data_stats: Dict[str, Dict] = dict()
        for data_type in [Macros.train, Macros.test]:
            data_list = IOUtils.load(Macros.model_data_dir / f"{data_type}.json")
            num_data_point = len(data_list)
            total_failed_test_case = [len(data_point["raw_data"]["failed_test_list"]) for data_point in data_list]
            total_passed_test_case = [len(data_point["raw_data"]["passed_test_list"]) for data_point in data_list]
            stats = {
                "FAILED_BUILD_NUM": num_data_point,
                "AVG_FAIL_PER_BUILD": np.array(total_failed_test_case).mean(),
                "AVG_PASSED_PER_BUILD": np.array(total_passed_test_case).mean()
            }
            model_data_stats[data_type] = stats
        # end for
        IOUtils.dump(self.output_dir / "model-data-stats.json", model_data_stats)

    def collect_metrics_raw_eval_dataset(self):
        """Get and save the per-project stats of eval dataset."""
        raw_dataset_stats: List[Dict] = list()
        proj_list = listdir(Macros.raw_eval_data_dir)
        projects = [Path(Macros.raw_eval_data_dir / proj) for proj in listdir(Macros.raw_eval_data_dir)]
        for proj_file, proj_name in zip(projects, proj_list):
            raw_data_list = IOUtils.load(proj_file)
            total_shas = 0
            changed_lines = list()
            failed_shas = list()
            total_tests_num = 0
            ekstazi_passed_tests_num = 0
            ekstazi_failed_tests_num = 0
            starts_passed_tests_num = 0
            starts_failed_test_num = 0

            for data_point in raw_data_list:
                changed_lines_per_sha = 0
                diff_per_file = data_point["diff_per_file"]
                for filename, changed_code in diff_per_file.items():
                    if filename.endswith(".java"):
                        changed_lines_per_sha += len(changed_code.splitlines())
                if changed_lines_per_sha > 1000:
                    print(
                        f"{proj_name} {data_point['prev_commit']} {data_point['commit']} changes more than 1000 lines")
                    continue
                total_shas += 1
                if data_point["failed_test_list"]:
                    failed_shas.append(data_point["commit"])
                total_tests_num += len(data_point["failed_test_list"]) + len(data_point["passed_test_list"])
                ekstazi_passed_tests_num += len(data_point["ekstazi_test_list"])
                starts_passed_tests_num += len(data_point["starts_test_list"])
                ekstazi_failed_tests_num += len(data_point["ekstazi_failed_test_list"])
                starts_failed_test_num += len(data_point["starts_failed_test_list"])

                changed_lines.append(changed_lines_per_sha)

            proj_data_stats = {
                "PROJ_NAME": proj_name.split("_")[1].split(".")[0],
                "TOTAL_SHAS": total_shas,
                "FAILED_TEST_SHAS": len(failed_shas),
                "TOTAL_TESTS": total_tests_num,
                "EKSTAZI_PASSED_TEST_NUM": ekstazi_passed_tests_num,
                "STARTS_PASSED_TEST_NUM": starts_passed_tests_num,
                "EKSTAZI_FAILED_TEST_NUM": ekstazi_failed_tests_num,
                "STARTS_FAILED_TEST_NUM": starts_failed_test_num,
                "CHANGED_LINES": sum(changed_lines),
                "AVG_CHANGED_LINES": sum(changed_lines) / len(raw_data_list),
                "FAILED_SHA_LIST": " ".join(failed_shas)
            }
            raw_dataset_stats.append(proj_data_stats)
        # end for
        IOUtils.dump(self.output_dir / "raw-eval-dataset-stats.json", raw_dataset_stats)

    def collect_metrics_raw_dataset(self):
        """Get and save the per-project stats of raw dataset."""
        raw_dataset_stats: List[Dict] = list()
        data_dir = Macros.data_dir / "raw-data"
        proj_list = listdir(data_dir)
        project_list = list()
        projects = [Path(data_dir / proj) for proj in listdir(data_dir)]
        for proj_file, proj_name in zip(projects, proj_list):
            total_test_case = list()
            raw_data_list = IOUtils.load(proj_file)
            num_data_point = len(raw_data_list)
            total_failed_test_case = list()
            total_passed_test_case = list()
            total_test_methods = list()
            failed_sha_list = list()
            failed_build = 0
            for index, data_point in enumerate(raw_data_list):
                if data_point["min_dis"]:
                    failed_build += 1
                total_failed_test_case.append(len(data_point["failed_test_list"]))
                total_passed_test_case.append(len(data_point["passed_test_list"]))
                total_test_case.append(len(data_point["passed_test_list"]) + len(data_point["failed_test_list"]))
                if data_point["tests_cases_num"]:
                    total_test_methods.append(sum(data_point["tests_cases_num"].values()))
                else:
                    total_test_methods.append(0)
                failed_sha_list.append(data_point["commit"][:8])
            proj_data_stats = {
                "FAILED_TO_BUILD": failed_build,
                "PROJ_NAME": proj_name.split("_")[1].split(".")[0],
                # in fact it is failed test num
                "FAILED_BUILD_NUM": num_data_point,
                "AVG_FAIL_PER_BUILD": np.array(total_failed_test_case).mean(),
                "AVG_PASSED_PER_BUILD": np.array(total_passed_test_case).mean(),
                "MAX_TEST_CASE_PER_BUILD": max(total_test_case),
                "MIN_TEST_CASE_PER_BUILD": min(total_test_case),
                "MAX_TEST_METHOD_PER_BUILD": max(total_test_methods),
                "MIN_TEST_METHOD_PER_BUILD": min(total_test_methods),
                "FAILED_SHA_LIST": " ".join(failed_sha_list)
            }
            raw_dataset_stats.append(proj_data_stats)
            project_list.append(proj_name.split("_")[1].split(".")[0])
        # end for
        IOUtils.dump(self.output_dir / "raw-dataset-stats.json", raw_dataset_stats)
        IOUtils.dump(self.output_dir / "project-list.json", project_list)

    @staticmethod
    def get_subset_select_test_list(subset, prediction_scores, labels, tool_labels, num_of_selected_tests, test_list):
        """Get a list of tests names that will be selected by the model.

        Note the selection rate is fixed for all the shas in the project
        """
        if num_of_selected_tests <= 0:
            return []

        scores_with_index = [(s, index) for index, s in enumerate(prediction_scores)]
        num_of_selected_tests = int(num_of_selected_tests)
        if subset == "All":
            subset_prediction_scores = scores_with_index
        else:
            subset_prediction_scores = []
            for i in range(len(labels)):
                if tool_labels[i] == 1:
                    subset_prediction_scores.append(scores_with_index[i])
        # end if

        sorted_preds = sorted(subset_prediction_scores, key=lambda x: (x[0], x[1]))
        sorted_preds_index = [x[1] for x in sorted_preds][-num_of_selected_tests:]

        return [test_list[index] for index in sorted_preds_index]

    @staticmethod
    def get_subset_select_test_list_selection_rate_change(data_item: dict, subset: str, test_list: List[str]) -> List[str]:
        """Get a list of tests names that will be selected by the model.

        data_item: the dict to store model predictions on one commit
        subset: STARTS/Ekstazi/All
        """
        selected_test_index_list = []
        if subset == "STARTS":
            tool_labels = data_item["STARTS_labels"]
        elif subset == "Ekstazi":
            tool_labels = data_item["Ekstazi_labels"]
        elif subset == "All":
            tool_labels = data_item["labels"]
        else:
            raise NotImplementedError

        scores_with_index = [(s, index) for index, s in enumerate(data_item["prediction_scores"])]

        if subset == "All":
            model_starts_preds = scores_with_index
            starts_selected_labels = data_item["labels"]
        else:
            # Add the results of models selection intersecting tools
            model_starts_preds = []
            starts_selected_labels = []
            for i in range(len(tool_labels)):
                if tool_labels[i] == 1:
                    starts_selected_labels.append(data_item["labels"][i])
                    model_starts_preds.append(scores_with_index[i])  # [(s1, 0), (s2, 1) ...]
            # end for
        # end if
        num_all_failure = sum(starts_selected_labels)
        fail_founded = 0

        sorted_preds = sorted(model_starts_preds, key=lambda x: (x[0], x[1]))
        sorted_preds_index = [x[1] for x in sorted_preds]
        for rank, t in enumerate(sorted_preds_index[::-1]):
            selected_test_index_list.append(t)
            if data_item["labels"][t] > 0:
                fail_founded += 1
            if fail_founded == num_all_failure:
                if rank == len(sorted_preds_index) - 1:
                    break
                elif data_item["prediction_scores"][t] != data_item["prediction_scores"][sorted_preds_index[: \
                :-1][rank + 1]]:
                    break
            # end if
        # end for
        return [test_list[index] for index in selected_test_index_list]

    def select_from_subset_execution_time(self, projs, models, change_selection_rate=False):
        """Get the execution time for tests selected by models.

        For each model, extract the list of selected tests to make the recall 100%,
        the execution time = total time for execute the selected tests.
        Save the execution time to Macros.metrics_dir/stats-*-.json
        """
        # ../../results/metrics/stats-Bukkit_Bukkit-boosting-eval-results.json
        # stats-Bukkit_Bukkit-IR-baseline-eval-results.json
        # stats-Bukkit_Bukkit-EALRTS-eval-results.json
        # stats-Bukkit_Bukkit-Fail-Basic-BM25Baseline-eval-results.json
        for proj in projs:
            all_res = {}
            ekstazi_subset_res = {}
            starts_subset_res = {}

            for model in models:
                all_res[model] = {}
                ekstazi_subset_res[model] = {}
                starts_subset_res[model] = {}

                # get best safe selection rate
                if model == "boosting" or model == "Ekstazi-Basic-BM25Baseline" or model == "Fail-Basic-BM25Baseline":
                    select_rate_info = IOUtils.load(
                        f"{Macros.results_dir}/metrics/stats-{proj}-{model}-eval-results.json")
                if model == "Fail-Code" or model == "Fail-Basic" or model == "Fail-ABS" or model == "Ekstazi-Code" or model == "Ekstazi-Basic" or model == "Ekstazi-ABS":
                    select_rate_info = IOUtils.load(f"{Macros.results_dir}/metrics/stats-{proj}-eval-results.json")
                if model == "BM25Baseline":
                    select_rate_info = IOUtils.load(
                        f"{Macros.results_dir}/metrics/stats-{proj}-IR-baseline-eval-results.json")
                if model == "randomforest" or model == "xgboost":
                    if proj == "apache_commons-csv":
                        continue
                    select_rate_info = IOUtils.load(
                        f"{Macros.results_dir}/metrics/stats-{proj}-EALRTS-eval-results.json")
                all_best_safe_selection_rate = select_rate_info[model]["best-safe-select-rate"]
                ekstazi_subset_best_safe_selection_rate = select_rate_info[model]["Ekstazi-select-subset-rate"]
                starts_subset_best_safe_selection_rate = select_rate_info[model]["STARTS-select-subset-rate"]

                # select subset test
                time_for_each_sha = IOUtils.load(
                    f"{Macros.eval_data_dir}/mutated-eval-data/{proj}-ag-time-for-each-test.json")
                per_sha_results = IOUtils.load(
                    f"{Macros.results_dir}/modelResults/{proj.split('_')[-1]}/{model}/per-sha-result.json")
                mutated_eval_data = IOUtils.load(f"{Macros.eval_data_dir}/mutated-eval-data/{proj}-ag.json")

                for per_sha_result in per_sha_results:
                    commit = per_sha_result["commit"]
                    sha = commit[:8]
                    test_to_time = time_for_each_sha[sha]

                    test_list = []
                    for mutated_eval_item in mutated_eval_data:
                        if mutated_eval_item["commit"] == commit:
                            test_list = mutated_eval_item["qualified_failed_test_list"] + mutated_eval_item[
                                "qualified_passed_test_list"]
                            break
                    assert len(test_list) > 0

                    prediction_scores = per_sha_result["prediction_scores"]
                    labels = per_sha_result["labels"]
                    ekstazi_labels = per_sha_result["Ekstazi_labels"]
                    starts_labels = per_sha_result["STARTS_labels"]

                    test_index_scores = [(i, s) for i, s in enumerate(prediction_scores)]

                    # select from all
                    num_of_selected_tests = all_best_safe_selection_rate * len(test_list)
                    if change_selection_rate:
                        selected_test_list = MetricsCollector.get_subset_select_test_list_selection_rate_change(per_sha_result, "All", test_list)
                    else:
                        selected_test_list = MetricsCollector.get_subset_select_test_list("All", prediction_scores, labels,
                                                                                      labels, num_of_selected_tests, test_list)
                    all_time = self.select_time(selected_test_list, test_to_time)
                    all_res[model][commit] = all_time
                    # select from Ekstazi
                    num_of_selected_tests = ekstazi_subset_best_safe_selection_rate * len(test_list)
                    if change_selection_rate:
                        selected_test_list = MetricsCollector.get_subset_select_test_list_selection_rate_change(per_sha_result, "Ekstazi", test_list)
                    else:
                        selected_test_list = MetricsCollector.get_subset_select_test_list("Ekstazi", prediction_scores, labels,
                                                                                      ekstazi_labels, num_of_selected_tests, test_list)
                    ekstazi_subset_time = self.select_time(selected_test_list, test_to_time)
                    ekstazi_subset_res[model][commit] = ekstazi_subset_time
                    # select from STARTS
                    num_of_selected_tests = starts_subset_best_safe_selection_rate * len(test_list)
                    if change_selection_rate:
                        selected_test_list = MetricsCollector.get_subset_select_test_list_selection_rate_change(per_sha_result, "STARTS", test_list)
                    else:
                        selected_test_list = MetricsCollector.get_subset_select_test_list("STARTS", prediction_scores, labels,
                                                                                      starts_labels, num_of_selected_tests, test_list)
                    starts_subset_time = self.select_time(selected_test_list, test_to_time)
                    starts_subset_res[model][commit] = starts_subset_time

                    if "Ekstazi" not in all_res:
                        all_res["Ekstazi"] = {}
                    if "Ekstazi" not in ekstazi_subset_res:
                        ekstazi_subset_res["Ekstazi"] = {}
                    if commit not in all_res["Ekstazi"]:
                        all_res["Ekstazi"][commit] = self.select_time(self.select_from_tool(ekstazi_labels, test_list),
                                                                      test_to_time)
                    if commit not in ekstazi_subset_res["Ekstazi"]:
                        ekstazi_subset_res["Ekstazi"][commit] = self.select_time(
                            self.select_from_tool(ekstazi_labels, test_list), test_to_time)

                    if "STARTS" not in all_res:
                        all_res["STARTS"] = {}
                    if "STARTS" not in starts_subset_res:
                        starts_subset_res["STARTS"] = {}
                    if commit not in all_res["STARTS"]:
                        all_res["STARTS"][commit] = self.select_time(self.select_from_tool(starts_labels, test_list),
                                                                     test_to_time)
                    if commit not in starts_subset_res["STARTS"]:
                        starts_subset_res["STARTS"][commit] = self.select_time(
                            self.select_from_tool(starts_labels, test_list), test_to_time)

            if change_selection_rate:
                IOUtils.dump(f"{Macros.metrics_dir}/stats-{proj}-execution-time-all-change-selection-rate.json", all_res,
                             IOUtils.Format.jsonPretty)
                IOUtils.dump(f"{Macros.metrics_dir}/stats-{proj}-execution-time-ekstazi-change-selection-rate.json", ekstazi_subset_res,
                             IOUtils.Format.jsonPretty)
                IOUtils.dump(f"{Macros.metrics_dir}/stats-{proj}-execution-time-starts-change-selection-rate.json", starts_subset_res,
                             IOUtils.Format.jsonPretty)
            else:
                IOUtils.dump(f"{Macros.metrics_dir}/stats-{proj}-execution-time-all.json", all_res,
                             IOUtils.Format.jsonPretty)
                IOUtils.dump(f"{Macros.metrics_dir}/stats-{proj}-execution-time-ekstazi.json", ekstazi_subset_res,
                             IOUtils.Format.jsonPretty)
                IOUtils.dump(f"{Macros.metrics_dir}/stats-{proj}-execution-time-starts.json", starts_subset_res,
                             IOUtils.Format.jsonPretty)

    def select_time(self, test_list, test_to_time):
        time = 0
        for test in test_list:
            if test in test_to_time:
                time += test_to_time[test]
            else:
                print(test, "not in test list")
        return time

    def select_from_tool(self, tool_labels, test_list):
        selected_test_list = []
        for index, value in enumerate(tool_labels):
            if value == 1:
                selected_test_list.append(test_list[index])
        return selected_test_list

    def split_eval_data(self, type: str, projects: List, models: List):
        '''
        split the data of per-sha-result based on the rule defined by us
        type: 1. simple-rule: changed tests are failed tests
        2. newly-added-tests: failed tests include newly added tests (newly added tests mean that these tests do not exist
        in train SHA)
        3. killed-tests: failed tests are all killed tests of PIT
        '''
        for project in projects:
            print("project", project)
            proj = project.split("_")[-1]
            mutated_eval_data = IOUtils.load(f"{Macros.eval_data_dir}/mutated-eval-data/{project}-ag.json")
            # tests of training SHA
            train_tests = IOUtils.load(f"{Macros.repos_results_dir}/{project}/collector/test2methods.json")

            # killed mutants of training SHA
            if type == "killed-tests":
                mutant_train_data = IOUtils.load(f"{Macros.repos_results_dir}/{project}/collector/mutant-data.json")
                killed_tests = set()
                for mutant_train_data_item in mutant_train_data:
                    if mutant_train_data_item["status"] == "KILLED" and mutant_train_data_item["killingTests"]:
                        for killing_test_pair in mutant_train_data_item["killingTests"]:
                            killed_tests.add(killing_test_pair[0].split(".")[-1])
            type_list = []
            not_type_list = []

            no_simple_rule_type_list = []
            no_simple_rule_not_type_list = []
            for mutated_eval_data_item in mutated_eval_data:
                sha = mutated_eval_data_item["commit"]
                failed_tests = mutated_eval_data_item["failed_test_list"]
                changed_files = mutated_eval_data_item["diff_per_file"].keys()
                changed_classes = [changed_file.split("/")[-1].replace(".java", "") for changed_file in changed_files]
                if type == "simple-rule":
                    # Define the new simple rule by first tokenizing the class name and compare the overlap
                    simple_case = True
                    tokenized_changed_classes_set = set()
                    for changed_class in changed_classes:
                        tokenized_changed_classes_set = set.union(tokenized_changed_classes_set, set(tokenize(
                            changed_class.replace("Test", "")).split()))
                    # end for
                    for failed_test in failed_tests:
                        tokenized_failed_test = set(tokenize(failed_test.replace("Test", "")).split())
                        if (len(tokenized_failed_test.intersection(tokenized_changed_classes_set)) / len(
                                tokenized_failed_test)) < 0.5:
                            simple_case = False
                            break
                        # end if
                    # end for
                    if simple_case:
                        type_list.append(sha)
                    else:
                        not_type_list.append(sha)
                elif type == "newly-added-tests":
                    simple_rule_shas = IOUtils.load(
                        f"{Macros.results_dir}/modelResults/{proj}/shalist-simple-rule.json")
                    if not set(failed_tests).issubset(train_tests.keys()):
                        if sha in simple_rule_shas:
                            type_list.append(sha)
                        else:
                            no_simple_rule_type_list.append(sha)
                    else:
                        if sha in simple_rule_shas:
                            not_type_list.append(sha)
                        else:
                            no_simple_rule_not_type_list.append(sha)

                elif type == "killed-tests":
                    simple_rule_shas = IOUtils.load(
                        f"{Macros.results_dir}/modelResults/{proj}/shalist-simple-rule.json")
                    if set(failed_tests).issubset(killed_tests):
                        if sha in simple_rule_shas:
                            type_list.append(sha)
                        else:
                            no_simple_rule_type_list.append(sha)
                    else:
                        if sha in simple_rule_shas:
                            not_type_list.append(sha)
                        else:
                            no_simple_rule_not_type_list.append(sha)

            if type == "newly-added-tests" or type == "killed-tests":
                IOUtils.dump(f"{Macros.results_dir}/modelResults/{proj}/shalist-{type}.json", type_list)
                IOUtils.dump(f"{Macros.results_dir}/modelResults/{proj}/shalist-not-{type}.json",
                             not_type_list)
                IOUtils.dump(f"{Macros.results_dir}/modelResults/{proj}/shalist-no-simple-rule-{type}.json",
                             no_simple_rule_type_list)
                IOUtils.dump(f"{Macros.results_dir}/modelResults/{proj}/shalist-no-simple-rule-not-{type}.json",
                             no_simple_rule_not_type_list)
            else:
                print(len(type_list) / (len(type_list) + len(not_type_list)))
                IOUtils.dump(f"{Macros.results_dir}/modelResults/{proj}/shalist-{type}.json", type_list)
                IOUtils.dump(f"{Macros.results_dir}/modelResults/{proj}/shalist-not-{type}.json",
                             not_type_list)
