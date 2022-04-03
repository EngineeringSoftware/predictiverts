import logging
from typing import *
from collections import defaultdict
from pathlib import Path
import numpy as np

from seutil import IOUtils, LoggingUtils
from seutil import latex

from pts.Environment import Environment
from pts.Macros import Macros
from pts.Utils import Utils
from sklearn import metrics


class Table:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    COLSEP = "COLSEP"
    ROWSEP = "ROWSEP"

    SYMBOLS = [
        r"\alpha", r"\beta", r"\gamma", r"\delta",
        r"\epsilon", r"\zeta", r"\eta", r"\theta",
        r"\iota", r"\kappa", r"\lambda", r"\mu",
        r"\nu", r"\tau", r"\pi", r"\rho",
    ]

    MODELS = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS",
              "BM25Baseline", "randomforest"]
    ENSEMBLE_MODELS = ["Fail-Basic-BM25Baseline", "Ekstazi-Basic-BM25Baseline"]
    BOOSTING_MODELS = ["boosting"]
    TOOLS = ["Ekstazi", "STARTS"]

    def __init__(self):
        self.tables_dir: Path = Macros.paper_dir / "tables"
        IOUtils.mk_dir(self.tables_dir)
        self.metrics_dir: Path = Macros.results_dir / "metrics"

        return

    def make_tables(self, which, options):

        for item in which:
            if item == "dataset-raw-stats":
                self.make_numbers_raw_dataset_metrics()
                self.make_table_raw_dataset_metrics()
            elif item == "dataset-model-stats":
                self.make_numbers_model_data_metrics()
                self.make_table_model_data_metrics()
            elif item == "project-pit-mutants":
                project = options.get("project")
                self.make_numbers_mutation_metrics(project)
                self.make_table_project_mutations_metrics(project)
            elif item == "raw-eval-dataset-stats":
                # projects = Utils.get_option_as_list(options, "projects")
                self.make_numbers_raw_eval_data_metrics()
                self.make_table_raw_eval_data_metrics()
            elif item == "mutated-eval-dataset-stats":
                project = options.get("project")
                self.make_table_mutated_eval_dataset_metrics(project)
            elif item == "rank-model-eval-results":
                project = options.get("project")
                self.make_numbers_rank_model_eval_results(project)
                self.append_numbers_rank_model_eval_results(project)
                self.append_numbers_select_time_results(project)
                self.make_table_rank_models_results(project)
            elif item == "rank_model_IR_baseline_eval_results":
                project = options.get("project")
                self.make_numbers_rank_model_IR_baseline_eval_results(project)
            elif item == "ensemble-model-numbers":
                project = options.get("project")
                model_name = options.get("model")
                self.make_numbers_ensemble_model_eval_results(project, model_name)
            elif item == "boosting-model-numbers":
                project = options.get("project")
                model_name = options.get("model", "boosting")
                self.make_numbers_boosting_model_eval_results(project, model_name)
            elif item == "rank_model_eval_results_with_no_deps_update":
                project = options.get("project")
                self.append_numbers_rank_model_eval_results(project)
                self.make_table_rank_models_results(project)
            elif item == "mutant-line-mapping-baseline-stats":
                search_span = Utils.get_option_as_list(options, "search_span")
                project = options.get("project")
                # self.make_numbers_line_mapping_results(search_span, project)
                self.make_table_line_mapping_results(search_span, project)
            elif item == "select_time_update":
                project = options.get("project")
                self.append_numbers_select_time_results(project)
                self.make_table_rank_models_results(project)
            elif item == "cmp-best-safe-select-rate":
                projects = options.get("projects").split()
                self.make_table_cmp_best_safe_select_rate(projects)
            elif item == "EALRTS-numbers":
                projects = options.get("projects").split()
                self.make_EALRTS_numbers(projects)
            elif item == "cmp-avg-safe-select-rate":
                projects = options.get("projects").split()
                self.make_table_cmp_avg_safe_select_rate(projects)
            elif item == "avg-best-safe-select-rate":
                self.make_numbers_avg_best_safe_select_rate()
                self.make_table_avg_best_safe_select_rate()
            elif item == "pct-newly-added-tests":
                projects = options.get("projects").split()
                # self.make_numbers_pct_newly_added_tests(projects)
                self.make_table_pct_newly_added_tests(projects)
            elif item == "perfect-best-safe-selection-rate":
                projects = options.get("projects").split()
                self.make_numbers_perfect_best_safe_selection_rate(projects)
            elif item == "confusion-matrices-table":
                from itertools import combinations
                project = options.get("project")
                self.make_numbers_confusion_matrices(project)
                models_comb = combinations(self.MODELS, 2)
                for model1, model2 in models_comb:
                    self.make_table_confusion_matrices(project, model1, model2)
            elif item == "correct-confusion-matrices-table":
                from itertools import combinations
                project = options.get("project")
                self.make_numbers_correct_confusion_matrices(project)
                models_comb = combinations(self.MODELS, 2)
                for model1, model2 in models_comb:
                    self.make_table_correct_confusion_matrices(project, model1, model2)
            elif item == "auc-score-number":
                project = options.get("project")
                self.make_numbers_auc_score(project)
            elif item == "real-failed-test-stats":
                self.make_numbers_real_failed_test_metrics()
            elif item == "real-failed-test-no-rule-stats":
                self.make_numbers_real_failed_test_no_rule_metrics(True)
                self.make_numbers_real_failed_test_no_rule_metrics(False)
            elif item == "auc-recall-selection-stats":
                subsets = options.get("subsets").split()
                projects = options.get("projects").split()
                data_types = options.get("datatypes").split()
                print(data_types)
                self.make_numbers_auc_recall_selection(subsets, projects, data_types)

            elif item == "selection-time-stats":
                projects = options.get("projects").split()
                data_types = options.get("datatypes").split()
                rts_tools = options.get("rtstools").split()
                ekstazi_subset_models = data_types + ["Ekstazi"]
                ekstazi_subset_models.remove("randomforest")
                self.make_numbers_selection_time(projects, ekstazi_subset_models, "Ekstazi")
                starts_subset_models = data_types + ["STARTS"]
                self.make_numbers_selection_time(projects, starts_subset_models, "STARTS")
            elif item == "selection-time-table":
                projects = options.get("projects").split()
                data_types = options.get("datatypes").split()
                rts_tools = options.get("rtstools").split()
                ekstazi_subset_models = data_types + ["Ekstazi"]
                ekstazi_subset_models.remove("randomforest")
                self.make_table_selection_time(projects, ekstazi_subset_models, "Ekstazi")
                starts_subset_models = data_types + ["STARTS"]
                self.make_table_selection_time(projects, starts_subset_models, "STARTS")
            elif item == "paper-dataset-table":
                projects = options.get("projects").split()
                self.make_table_dataset_table(projects)
            elif item == "real-failure-results-paper-table":
                self.make_table_real_failures_paper()
            elif item == "subset-execution-time":
                """Make numbers macros for execution time and the table for that"""
                projects = options.get("projects").split()
                data_types = options.get("datatypes").split()
                subset = options.get("subset", "Ekstazi")
                self.make_table_execution_time_paper(projects, data_types, subset)
            elif item == "subset-end-to-end-time":
                """Make numbers macros for end to end time and the table for that"""
                projects = options.get("projects").split()
                data_types = options.get("datatypes").split()
                subset = options.get("subset", "Ekstazi")
                change_selection_rate = options.get("change-selection-rate", "False")
                if change_selection_rate == "True":
                    self.make_macros_end_to_end_time(projects, subset, list(self.MODELS), True)
                    self.make_table_end_to_end_time(projects, subset, list(self.MODELS), True)
                else:
                    self.make_macros_end_to_end_time(projects, subset, list(self.MODELS))
                    self.make_table_end_to_end_time(projects, subset, list(self.MODELS))
            elif item == "bm25-perfect-numbers":
                """make numbers macros for the cases where bm25 has perfect selection rate"""
                self.make_numbers_bm25_perfect()
            elif item == "bm25-perfect-table":
                projects = options.get("projects").split()
                self.make_table_bm25_perfect(projects)
            elif item == "dataset-raw-stats":
                """stats (avg-changed-files, avg-tests ...) of raw eval data 
                    \input{tables/numbers-results-raw-dataset-metrics.tex}
                    \input{tables/table-results-raw-dataset-metrics.tex}"""
                self.make_numbers_raw_dataset_metrics()
                self.make_table_raw_dataset_metrics()
            elif item == "intermediate-dataset-raw-stats":
                """ stats of filtering raw eval data (50 commits for each project)
                \input{tables/numbers-intermediate-raw-dataset-metrics.tex}
                \input{tables/table-intermediate-raw-dataset-metrics.tex}"""
                self.make_numbers_intermediate_raw_dataset_metrics()
                self.make_table_intermediate_raw_dataset_metrics()
            else:
                raise NotImplementedError

    def make_numbers_selection_time(self, projects, models, subset):
        res = {}
        file = latex.File(self.tables_dir / f"numbers-{subset.lower()}-selection-time.tex")
        for proj in projects:
            res[proj] = defaultdict(list)
            time_for_proj = IOUtils.load(self.metrics_dir / f"{proj}-{subset.lower()}-selection-time.json")
            # time_for_proj = IOUtils.load(f"{Macros.eval_data_dir}/mutated-eval-data/{proj}_time.json")
            for time_for_item in time_for_proj:
                for model in models:
                    if (proj == "apache_commons-csv") and (model == "randomforest" or model == "xgboost"):
                        continue
                    res[proj][model].append(time_for_item[model])

            for model in models:
                if (proj == "apache_commons-csv") and (model == "randomforest" or model == "xgboost"):
                    continue
                if len(res[proj][model]) == 0:
                    print(proj, model, "no time data collected")
                    continue
                v = sum(res[proj][model])/len(res[proj][model])
                fmt = f",d" if type(v) == int else f",.2f"
                file.append_macro(latex.Macro(f"{proj}-{model}-{subset.lower()}-avg-selection-time", f"{v:{fmt}}"))
                file.append_macro(latex.Macro(f"{proj}-{model}-{subset.lower()}-median-selection-time", f"{np.median(res[proj][model]):{fmt}}"))
                file.append_macro(latex.Macro(f"{proj}-{model}-{subset.lower()}-min-selection-time", f"{min(res[proj][model]):{fmt}}"))
                file.append_macro(latex.Macro(f"{proj}-{model}-{subset.lower()}-max-selection-time", f"{max(res[proj][model]):{fmt}}"))
                file.append_macro(latex.Macro(f"{proj}-{model}-{subset.lower()}-stddev-selection-time", f"{np.std(res[proj][model]):{fmt}}"))
        file.save()

    def make_table_selection_time(self, projects, models, subset):
        """Make selection time table about (put in appendix) avg, min, avg, median, stddev.
        Two tables containing the Percentage of selection time compared with Ekstazi and STARTS
        """
        table_type_dict = {'avg': "Average", "min": "Minimal", "max": "Maximal",
                           "stddev": "Standard deviation", "median": "Median"}

        number_file = self.tables_dir / f"numbers-{subset.lower()}-selection-time.tex"
        for data_type in table_type_dict:
            models_selection_time = latex.Macro.load_from_file(number_file)
            file = latex.File(self.tables_dir / f"table-{data_type}-selection-time-{subset.lower()}-subset.tex")
            # Header
            file.append(r"\begin{table}[h]")
            file.append(r"\centering")

            file.append(r"\caption{" + table_type_dict[data_type] + " selection time (seconds) combining models with "+ subset + "}")
            file.append(r"\scalebox{0.9}{")
            if subset.lower() == "ekstazi":
                file.append(r"\begin{tabular}{l|rrr|rrr|r|r}")
            elif subset.lower() == "starts":
                file.append(r"\begin{tabular}{l|rrr|rrr|r|r|r}")
            file.append(r"\hline")
            file.append(r"\multirow{2}{*}{")
            file.append(r"\textbf{Projects}} &")
            file.append(
                r"\multicolumn{3}{c|}{\textbf{Fail+}} &"
                r" \multicolumn{3}{c|}{\textbf{Ekstazi+}} &"
                r" \multicolumn{1}{c|}{\textbf{Baseline}}")
            if subset.lower() == "ekstazi":
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi}}}")
            elif subset.lower() == "starts":
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{randomforest}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{STARTS}}}")
            file.append(r"\\")
            file.append(r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{Basic} & \textbf{Code} &"
                        r" \textbf{ABS} & \textbf{BM25} &")
            file.append(r"\\")
            file.append(r"\hline")

            for proj in projects:
                project_name = proj.split('_')[1]
                file.append(r"\UseMacro{" + project_name + "}")

                # First find the max and min values
                max_time = -1
                min_time = 10000
                for model in models:
                    key = f"{proj}-{model}-{subset.lower()}-{data_type}-selection-time"
                    if (model == "randomforest" or model == "xgboost") and (proj == "apache_commons-csv"):
                        continue
                    val = models_selection_time.get(key).value
                    # handle exception str float
                    if isinstance(val, str):
                        val = float(val.replace(',', ''))
                    if val > max_time:
                        max_time = val
                    elif val < min_time:
                        min_time = val
                    # end if

                for model in models:
                    if (model == "randomforest" or model == "xgboost") and (proj == "apache_commons-csv"):
                        file.append(r" & N/A")
                        continue
                    key = f"{proj}-{model}-{subset.lower()}-{data_type}-selection-time"
                    val = models_selection_time.get(key).value
                    if isinstance(val, str):
                        val = float(val.replace(',', ''))
                    if val == min_time:
                        file.append(r" & \textbf{" + latex.Macro(key).use() + '}')
                    elif val == max_time:
                        file.append(r" & \cellcolor{black!20}" + latex.Macro(key).use() + "")
                    else:
                        file.append(r" & " + latex.Macro(key).use())
                file.append(r"\\")

            # Footer
            file.append(r"\bottomrule")
            file.append(r"\end{tabular}")
            file.append(r"}")
            file.append(r"\end{table}")
            file.save()

            # Make the table showing selection rate is x% of STARTS and Ekstazi

            file = latex.File(self.tables_dir / f"table-avg-selection-time-wrt-{subset.lower()}.tex")
            models_selection_time = latex.Macro.load_from_file(number_file)
            # Header
            file.append(r"\begin{table}[h]")
            file.append(r"\centering")
            file.append(r"\caption{Average selection time for models compared with " + subset + " (percent) }")
            file.append(r"\label{table:selection-time:pct:" + subset.lower() + "}")
            file.append(r"\scalebox{0.9}{")
            if subset.lower() == "ekstazi":
                file.append(r"\begin{tabular}{l|rrr|rrr|r}")
            elif subset.lower() == "starts":
                file.append(r"\begin{tabular}{l|rrr|rrr|r|r}")
            file.append(r"\hline")
            file.append(r"\multirow{2}{*}{")
            file.append(r"\textbf{Projects}} &")
            file.append(
                r"\multicolumn{3}{c|}{\textbf{Fail+}} &"
                r" \multicolumn{3}{c|}{\textbf{Ekstazi+}} &"
                r" \multicolumn{1}{c|}{\textbf{Baseline}}")
            if subset.lower() == "starts":
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{randomforest}}}")

            file.append(r"\\")
            file.append(r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{Basic} & \textbf{Code} &"
                        r" \textbf{ABS} & \textbf{BM25}")
            file.append(r"\\")
            file.append(r"\hline")

            for proj in projects:
                pct_dict = {}
                project_name = proj.split('_')[1]
                file.append(r"\UseMacro{" + project_name + "}")
                # First get tool average time
                tool_avg_time = models_selection_time.get(f"{proj}-{subset}-{subset.lower()}-avg-selection-time").value
                if isinstance(tool_avg_time, str):
                    tool_avg_time = float(tool_avg_time.replace(',', ''))
                # end if

                # Get percentage for each model, store in dictionary and find the max and min values
                for model in models:
                    if model in self.TOOLS:
                        continue
                    key = f"{proj}-{model}-{subset.lower()}-avg-selection-time"
                    if (model == "randomforest" or model == "xgboost") and (proj == "apache_commons-csv"):
                        continue
                    val = models_selection_time.get(key).value
                    # handle exception str float
                    if isinstance(val, str):
                        val = float(val.replace(',', ''))
                    # end if
                    pct_dict[model] = round(100*val/tool_avg_time, 2)
                # end for
                maximal = max(pct_dict.values())
                minimal = min(pct_dict.values())

                # Actually make tables
                for model in models:
                    if model in self.TOOLS:
                        continue
                    if (model == "randomforest" or model == "xgboost") and (proj == "apache_commons-csv"):
                        file.append(r" & N/A")
                        continue
                    model_avg_selection_pct = pct_dict[model]

                    # bold or gray the cell in table
                    if model_avg_selection_pct == minimal:
                        file.append(r" & \textbf{" + str(model_avg_selection_pct) + '}')
                    elif model_avg_selection_pct == maximal:
                        file.append(r" & \cellcolor{black!20}" + str(model_avg_selection_pct) + "")
                    else:
                        file.append(r" & " + str(model_avg_selection_pct))
                file.append(r"\\")

            # Footer
            file.append(r"\bottomrule")
            file.append(r"\end{tabular}")
            file.append(r"}")
            file.append(r"\end{table}")
            file.save()

    def make_numbers_auc_score(self, project: str):
        """Make the macro file for the auc score."""

        file = latex.File(self.tables_dir / f"numbers-{project}-auc-score.tex")
        for subset in ["All", "STARTS", "Ekstazi"]:
            stats_auc_file = Macros.metrics_dir / f"stats-{project}-{subset}-subset-auc-score.json"
            stats_auc = IOUtils.load(stats_auc_file)
            for model, auc in stats_auc.items():
                fmt = ",.2f"
                file.append_macro(latex.Macro(f"{project}-{model}-{subset}-subset-auc-score", f"{auc:{fmt}}"))
        # end for
        file.save()

    def make_numbers_correct_confusion_matrices(self, project: str):
        """Make the macros for the correct confusion matrix"""
        proj = project.split('_')[1]
        file = latex.File(self.tables_dir / f"numbers-{proj}-correct-confusion-matrices.tex")
        source_json = IOUtils.load(f"{Macros.metrics_dir}/correct-confusion-matrices-{project}.json")
        for k, stats in source_json.items():
            m1 = k.split('=')[0]
            m2 = k.split('=')[1]
            total_num_tests = stats["both_wrong"] + stats["both_correct"] + stats["m1_correct_m2_wrong"] + \
                              stats["m1_wrong_m2_correct"]
            # Fill in the macros
            fmt = f",.2f"
            v = stats["both_correct"] / total_num_tests
            file.append_macro(latex.Macro(f"{proj}-{m1}-{m2}-both-correct-pct", f"{v:{fmt}}"))
            v = stats["both_wrong"] / total_num_tests
            file.append_macro(latex.Macro(f"{proj}-{m1}-{m2}-both-wrong-pct", f"{v:{fmt}}"))
            v = stats["m1_correct_m2_wrong"] / total_num_tests
            file.append_macro(latex.Macro(f"{proj}-{m1}-{m2}-correct-wrong-pct", f"{v:{fmt}}"))
            v = stats["m1_wrong_m2_correct"] / total_num_tests
            file.append_macro(latex.Macro(f"{proj}-{m1}-{m2}-wrong-correct-pct", f"{v:{fmt}}"))

        file.save()

    def make_numbers_confusion_matrices(self, project: str):
        """Make the macros file for the confusion matrix"""
        proj = project.split('_')[1]
        file = latex.File(self.tables_dir / f"numbers-{proj}-confusion-matrices.tex")
        source_json = IOUtils.load(f"{Macros.metrics_dir}/confusion-matrices-{project}.json")
        for k, stats in source_json.items():
            m1 = k.split('=')[0]
            m2 = k.split('=')[1]
            total_num_tests = stats["agree_fail_num"] + stats["agree_pass_num"] + stats["m1_fail_m2_pass_num"] + \
                              stats["m1_pass_m2_fail_num"]
            # Fill in the macros
            fmt = f",.2f"
            v = stats["agree_fail_num"] / total_num_tests
            file.append_macro(latex.Macro(f"{proj}-{m1}-{m2}-agree-fail-pct", f"{v:{fmt}}"))
            v = stats["agree_pass_num"] / total_num_tests
            file.append_macro(latex.Macro(f"{proj}-{m1}-{m2}-agree-pass-pct", f"{v:{fmt}}"))
            v = stats["m1_fail_m2_pass_num"] / total_num_tests
            file.append_macro(latex.Macro(f"{proj}-{m1}-{m2}-fail-pass-pct", f"{v:{fmt}}"))
            v = stats["m1_pass_m2_fail_num"] / total_num_tests
            file.append_macro(latex.Macro(f"{proj}-{m1}-{m2}-pass-fail-pct", f"{v:{fmt}}"))

        file.save()

    def make_table_confusion_matrices(self, project: str, model1: str, model2: str):
        """Make table for confusion matrices given two models."""
        proj = project.split('_')[1]
        file = latex.File(self.tables_dir / f"table-{proj}-{model1}-{model2}-confusion-matrix.tex")
        # Header
        file.append(r"\begin{table*}")
        file.append(r"\centering")
        file.append(r"\small")
        file.append(r"\caption{Confusion matrix for " + model1 + " and " + model2 + f" {proj}" + ".}")
        file.append(r"\vspace{-11pt}")
        file.append(r"\begin{tabular}{l|l|l}")
        file.append(r"\hline")
        file.append(r"\textbf{Confusion matrix} & ")
        file.append(r"\textbf{" + model1 + " selected tests} &")
        file.append(r"\textbf{" + model1 + r" not selected tests} \\")
        file.append(r"\hline")

        file.append(r"\textbf{" + model2 + " selected tests} ")
        file.append(r" & " + latex.Macro(f"{proj}-{model1}-{model2}-agree-fail-pct").use())
        file.append(r" & " + latex.Macro(f"{proj}-{model1}-{model2}-pass-fail-pct").use())
        file.append(r"\\")

        file.append(r"\textbf{" + model2 + " not selected tests} ")
        file.append(r" & " + latex.Macro(f"{proj}-{model1}-{model2}-fail-pass-pct").use())
        file.append(r" & " + latex.Macro(f"{proj}-{model1}-{model2}-agree-pass-pct").use())
        file.append(r"\\")
        # Footer
        file.append(r"\hline")
        file.append(r"\end{tabular}")
        file.append(r"\end{table*}")
        file.save()

    def make_table_correct_confusion_matrices(self, project: str, model1: str, model2: str):
        """Make table for confusion matrices given two models."""
        proj = project.split('_')[1]
        file = latex.File(self.tables_dir / f"table-{proj}-{model1}-{model2}-correct-confusion-matrix.tex")
        # Header
        file.append(r"\begin{table*}")
        file.append(r"\centering")
        file.append(r"\small")
        file.append(r"\caption{Confusion matrix for mistakes " + model1 + " and " + model2 + f" {proj}" + ".}")
        file.append(r"\vspace{-11pt}")
        file.append(r"\begin{tabular}{l|l|l}")
        file.append(r"\hline")
        file.append(r"\textbf{Confusion matrix} & ")
        file.append(r"\textbf{" + model1 + " correct prediction} &")
        file.append(r"\textbf{" + model1 + r" wrong prediction} \\")
        file.append(r"\hline")

        file.append(r"\textbf{" + model2 + " correct prediction} ")
        file.append(r" & " + latex.Macro(f"{proj}-{model1}-{model2}-both-correct-pct").use())
        file.append(r" & " + latex.Macro(f"{proj}-{model1}-{model2}-wrong-correct-pct").use())
        file.append(r"\\")

        file.append(r"\textbf{" + model2 + " wrong prediction} ")
        file.append(r" & " + latex.Macro(f"{proj}-{model1}-{model2}-correct-wrong-pct").use())
        file.append(r" & " + latex.Macro(f"{proj}-{model1}-{model2}-both-wrong-pct").use())
        file.append(r"\\")
        # Footer
        file.append(r"\hline")
        file.append(r"\end{tabular}")
        file.append(r"\end{table*}")
        file.save()

    def make_numbers_pct_newly_added_tests(self, projs: List[str]):
        """Make the macros file for pct of newly added tests."""
        file = latex.File(self.tables_dir / "numbers-pct-newly-added-missed-tests.tex")
        for proj in projs:
            source_json = IOUtils.load(f"{Macros.metrics_dir}/stats-{proj}-test-selection-metrics.json")
            for k, v in source_json.items():
                if k.startswith("pct") and v < 0:
                    v = 0.0
                fmt = f",d" if type(v) == int else f",.2f"
                file.append_macro(latex.Macro(f"{proj}-{k}", f"{v:{fmt}}"))
        file.save()

    def make_table_pct_newly_added_tests(self, projs: List[str]):
        """Make the table for newly added tests"""
        file = latex.File(self.tables_dir / f"table-pct-newly-added-missed-tests.tex")
        # Header
        file.append(r"\begin{table*}")
        file.append(r"\centering")
        file.append(r"\small")
        file.append(r"\caption{Percentage of newly added test for the missed failed tests.}")
        file.append(r"\vspace{-5pt}")
        file.append(r"\begin{tabular}{l|p{2.5cm}p{2cm}p{2cm}}")
        file.append(r"\hline")
        file.append(r"\textbf{Projects} & ")
        file.append(r"\textbf{Pct. newly added missed failed tests} &")
        file.append(r"\textbf{\# missed failed tests} &")
        file.append(r"\textbf{\# selected failed tests} \\")
        file.append(r"\hline")

        for proj in projs:
            file.append(f"{proj.split('_')[-1]}")
            source_json = IOUtils.load(f"{Macros.metrics_dir}/stats-{proj}-test-selection-metrics.json")
            for k in source_json.keys():
                file.append(r" & " + latex.Macro(f"{proj}-{k}").use())
            file.append(r"\\")
        # Footer
        file.append(r"\hline")
        file.append(r"\end{tabular}")
        file.append(r"\end{table*}")
        file.save()

    def make_numbers_line_mapping_results(self, search_span: List[int], proj: str):
        file = latex.File(
            self.tables_dir / f"numbers-{proj}-line-mapping-eval-results-metrics.tex")
        rank_models_results = IOUtils.load(Macros.metrics_dir / f"stats-{proj}-eval-results.json")
        for sp in search_span:
            model = f"Baseline-{sp}"
            stats = rank_models_results[model]
            for k, v in stats.items():
                if v < 0:
                    file.append_macro(latex.Macro(f"{proj}-{model}-{k}", "n/a"))
                else:
                    fmt = f",d" if type(v) == int else f",.2f"
                    file.append_macro(latex.Macro(f"{proj}-{model}-{k}", f"{v:{fmt}}"))
            # end for
        # end for
        # Hard code the baselines
        model = f"Baseline-del"
        stats = rank_models_results[model]
        for k, v in stats.items():
            if v < 0:
                file.append_macro(latex.Macro(f"{proj}-{model}-{k}", "n/a"))
            else:
                fmt = f",d" if type(v) == int else f",.2f"
                file.append_macro(latex.Macro(f"{proj}-{model}-{k}", f"{v:{fmt}}"))
        model = f"Baseline-all"
        stats = rank_models_results[model]
        for k, v in stats.items():
            if v < 0:
                file.append_macro(latex.Macro(f"{proj}-{model}-{k}", "n/a"))
            else:
                fmt = f",d" if type(v) == int else f",.2f"
                file.append_macro(latex.Macro(f"{proj}-{model}-{k}", f"{v:{fmt}}"))
        file.save()

    def make_table_line_mapping_results(self, search_span: List[int], proj: str):
        """Create table for stats of mutant line mapping baseline models."""
        file = latex.File(self.tables_dir / f"table-{proj}-line-mapping-eval-results.tex")
        stats_dict = IOUtils.load(Macros.metrics_dir / f"stats-{proj}-eval-results.json")
        # Header
        file.append(r"\begin{table*}")
        file.append(r"\centering")
        file.append(r"\small")
        file.append(r"\caption{Eval results line mapping baselines on " + proj.split('_')[1] + "}")
        file.append(r"\vspace{-5pt}")
        file.append(r"\begin{tabular}{l|cccc}")
        file.append(r"\hline")

        file.append(r"\textbf{Models} & ")
        file.append("Avg. sel-rate &")
        file.append("Avg. recall &")
        file.append("Avg. not covered files &")
        file.append(r"pct. no test selected \\")
        file.append(r"\hline")
        for sp in search_span:
            model = f"Baseline-{sp}"
            file.append(model)
            for k in ["selection-rate", "avg-recall", "avg-not-covered-file", "pct-no-test-select"]:
                file.append(" & " + latex.Macro(f"{proj}-{model}-{k}").use())
            # end for
            file.append(r"\\")
        # end for
        # Hard code the baseline
        model = "Baseline-del"
        file.append(model)
        for k in ["selection-rate", "avg-recall", "avg-not-covered-file", "pct-no-test-select"]:
            file.append(" & " + latex.Macro(f"{proj}-{model}-{k}").use())
        # end for
        file.append(r"\\")
        model = "Baseline-all"
        file.append(model)
        for k in ["selection-rate", "avg-recall", "avg-not-covered-file", "pct-no-test-select"]:
            file.append(" & " + latex.Macro(f"{proj}-{model}-{k}").use())
        # end for
        file.append(r"\\")
        # Footer
        file.append(r"\hline")
        file.append(r"\end{tabular}")
        file.append(r"\end{table*}")
        file.save()

    def make_numbers_rank_model_eval_results(self, proj: str):
        # proj_for_macro = ''.join(i for i in proj if not i.isdigit())
        file = latex.File(
            self.tables_dir / f"numbers-{proj}-rank-model-eval-results-metrics.tex")
        rank_models_results = IOUtils.load(Macros.metrics_dir / f"stats-{proj}-eval-results.json")
        for model, sts in rank_models_results.items():
            for k, v in sts.items():
                if v < 0:
                    file.append_macro(latex.Macro(f"{proj}-{model}-{k}", "n/a"))
                else:
                    fmt = f",d" if type(v) == int else f",.2f"
                    file.append_macro(latex.Macro(f"{proj}-{model}-{k}", f"{v:{fmt}}"))
            # end for
        # end for
        file.save()

    def make_numbers_rank_model_IR_baseline_eval_results(self, proj: str):
        # proj_for_macro = ''.join(i for i in proj if not i.isdigit())
        file = latex.File(
            self.tables_dir / f"numbers-{proj}-rank-model-IR-baseline-eval-results-metrics.tex")
        rank_models_results = IOUtils.load(Macros.metrics_dir / f"stats-{proj}-IR-baseline-eval-results.json")
        for model, sts in rank_models_results.items():
            for k, v in sts.items():
                if v < 0:
                    file.append_macro(latex.Macro(f"{proj}-{model}-{k}", "n/a"))
                else:
                    fmt = f",d" if type(v) == int else f",.2f"
                    file.append_macro(latex.Macro(f"{proj}-{model}-{k}", f"{v:{fmt}}"))
            # end for
        # end for
        file.save()

    def make_EALRTS_numbers(self, projects: List[str]):
        """Make macro numbers for EALRTS models"""
        for proj in projects:
            file = latex.File(self.tables_dir / f"numbers-{proj}-EALRTS-eval-results-metrics.tex")
            for model_name in ["xgboost", "randomforest"]:
                if proj == "apache_commons-csv":
                    file.append_macro(latex.Macro(f"{proj}-{model_name}-STARTS-subset-best-safe-selection-rate", -1.0))
                    file.append_macro(latex.Macro(f"{proj}-{model_name}-Ekstazi-subset-best-safe-selection-rate", -1.0))
                    file.append_macro(latex.Macro(f"{proj}-{model_name}-best-safe-selection-rate", -1.0))
                    continue
                try:
                    rank_models_results = IOUtils.load(Macros.results_dir / "modelResults" / proj.split('_')[1] / model_name / f"best-safe-selection-rate.json")
                except FileNotFoundError:
                    rank_models_results = IOUtils.load(Macros.results_dir / "modelResults" / proj.split('_')[1] / model_name / f"STARTS-best-selection-rate-per-sha.json")
                    v = max(rank_models_results.values())
                    fmt = f",d" if type(v) == int else f",.2f"
                    file.append_macro(latex.Macro(f"{proj}-{model_name}-STARTS-subset-best-safe-selection-rate", f"{v:{fmt}}"))
                    file.append_macro(latex.Macro(f"{proj}-{model_name}-Ekstazi-subset-best-safe-selection-rate", f"{v:{fmt}}"))
                    file.append_macro(latex.Macro(f"{proj}-{model_name}-best-safe-selection-rate", f"{v:{fmt}}"))
                for metric, v in rank_models_results.items():
                    fmt = f",d" if type(v) == int else f",.2f"
                    file.append_macro(latex.Macro(f"{proj}-{model_name}-{metric}", f"{v:{fmt}}"))
                # end for
            # end for
            file.save()
        # end for

    def make_numbers_ensemble_model_eval_results(self, proj: str, model_name: str):
        """Make macro numbers for ensemble model. Hardcode for Fail-Code-BM25Baseline"""
        file = latex.File(
            self.tables_dir / f"numbers-{proj}-rank-model-ensemble-eval-results-metrics.tex", is_append=True)
        rank_models_results = IOUtils.load(Macros.metrics_dir / f"stats-{proj}-{model_name}-eval-results.json")
        for model, sts in rank_models_results.items():
            for k, v in sts.items():
                if v < 0:
                    file.append_macro(latex.Macro(f"{proj}-{model}-{k}", "n/a"))
                else:
                    fmt = f",d" if type(v) == int else f",.2f"
                    file.append_macro(latex.Macro(f"{proj}-{model}-{k}", f"{v:{fmt}}"))
            # end for
        # end for
        file.save()

    def make_numbers_boosting_model_eval_results(self, proj: str, model_name: str):
        """Make macro numbers for the boosting model"""
        file = latex.File(
            self.tables_dir / f"numbers-{proj}-boosting-model-eval-results-metrics.tex")
        rank_models_results = IOUtils.load(Macros.metrics_dir / f"stats-{proj}-boosting-eval-results.json")
        for model, sts in rank_models_results.items():
            for k, v in sts.items():
                if v < 0:
                    file.append_macro(latex.Macro(f"{proj}-{model}-{k}", "n/a"))
                else:
                    fmt = f",d" if type(v) == int else f",.2f"
                    file.append_macro(latex.Macro(f"{proj}-{model}-{k}", f"{v:{fmt}}"))
            # end for
        # end for
        file.save()

    def append_numbers_rank_model_eval_results(self, proj: str):
        file = latex.File(
            self.tables_dir / f"numbers-{proj}-rank-model-eval-results-metrics.tex", is_append=True)
        rank_models_results = IOUtils.load(
            Macros.results_dir / "metrics" / f"stats-{proj}-eval-results-no-dep-update.json")
        for key, value in rank_models_results.items():
            fmt = f",d" if type(value) == int else f",.2f"
            file.append_macro(latex.Macro(f"{proj}-{key}", f"{value:{fmt}}"))
        file.save()

    def append_numbers_select_time_results(self, proj: str):
        file = latex.File(
            self.tables_dir / f"numbers-{proj}-rank-model-eval-results-metrics.tex", is_append=True)
        rank_models_results = IOUtils.load(
            Macros.results_dir / "metrics" / f"stats-{proj}-select-time.json")
        for key, value in rank_models_results.items():
            fmt = f",d" if type(value) == int else f",.2f"
            file.append_macro(latex.Macro(f"{proj}-{key}", f"{value:{fmt}}"))
        file.save()

    def make_table_rank_models_results(self, proj: str):
        # Create RESULTS TABLE for proj
        # create results f1, recall, precision table
        file = latex.File(self.tables_dir / f"table-{proj}-rank-models-results.tex")
        # Header
        file.append(r"\begin{table*}")
        file.append(r"\centering")
        file.append(r"\scriptsize")
        file.append(r"\caption{Eval results for " + proj.split('_')[1] + "}")
        file.append(r"\vspace{-5pt}")
        file.append(r"\begin{tabular}{l|ccc|ccccc}")
        file.append(r"\hline")

        file.append(r"\textbf{Models} & ")
        file.append("Avg. best sel-rate &")
        file.append("Best/safe sel-rate &")
        file.append("Avg. best th &")
        file.append("Worst. high. f-test & ")
        file.append("Avg. APFD &")
        file.append("Avg. recall &")
        file.append("Avg. sel-rate &")
        file.append(r"Avg. time(s) \\")
        file.append(r"\hline")

        models = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS", "Random",
                  "Baseline-10", "Baseline-20", "Baseline-del", "Baseline-all"]

        for model in models:
            # for model, sts in stats_dict.items():
            file.append(model)
            for k in ["avg-best-select-rate", "best-safe-select-rate", "avg-best-threshold", "worst-highest-rank",
                      "avg-apfd", "avg-recall", "selection-rate", "time"]:
                if k == "time" and (model.startswith("Baseline") or model.startswith("Random")):
                    # baseline models do not need to count time
                    file.append(" & n/a")
                else:
                    # First add Recall and Precision for each baseline
                    file.append(" & " + latex.Macro(f"{proj}-{model}-{k}").use())
            # end for
            file.append(r"\\")
            if model == "Fail-ABS" or model == "Ekstazi-ABS" or model == "Baseline-all":
                file.append(r"\hline")
        # end for

        # Triplet model
        file.append("Ensemble")
        for k in ["avg-best-select-rate", "best-safe-select-rate", "avg-best-threshold", "worst-highest-rank",
                  "avg-apfd", "avg-recall", "selection-rate", "time"]:
            file.append(" & " + latex.Macro(f"{proj}-Fail-Code-BM25Baseline-{k}").use())
        # end for
        file.append(r"\\")
        file.append(r"\hline")

        # information retrieval models
        IRBaselines = ["TFIDFBaseline", "BM25Baseline"]
        for model in IRBaselines:
            file.append(model)
            for k in ["avg-best-select-rate", "best-safe-select-rate", "avg-best-threshold", "worst-highest-rank",
                      "avg-apfd"]:
                file.append(" & " + latex.Macro(f"{proj}-{model}-{k}").use())
            file.append(" & n/a")  # Avg. recall
            file.append(" & n/a")  # Avg. sel-rate
            file.append(" & n/a")  # Avg. time
            file.append(r"\\")
        file.append(r"\hline")
        # end for

        model = models[0]
        for tool in ["Ekstazi", "Ekstazi-no-update", "STARTS", "STARTS-no-update"]:
            file.append(tool)
            if tool in ["Ekstazi-no-update", "STARTS-no-update"]:
                file.append(" & " + latex.Macro(f"{proj}-avg-{tool}-select-rate").use())
                file.append(" & n/a")
            else:
                file.append(" & " + latex.Macro(f"{proj}-{model}-avg-{tool}-select-rate").use())
                file.append(" & " + latex.Macro(f"{proj}-{model}-{tool}-best-safe-select-rate").use())
            file.append(" & n/a")
            file.append(" & n/a")
            file.append(" & n/a")
            if tool in ["Ekstazi-no-update", "STARTS-no-update"]:
                file.append(" & " + latex.Macro(f"{proj}-avg-{tool}-recall").use())
                file.append(" & " + latex.Macro(f"{proj}-avg-{tool}-select-rate").use())
                file.append(" & n/a")
            else:
                file.append(" & " + latex.Macro(f"{proj}-{model}-avg-{tool}-recall").use())
                file.append(" & " + latex.Macro(f"{proj}-{model}-avg-{tool}-select-rate").use())
                file.append(" & " + latex.Macro(f"{proj}-{tool.lower()}_select-time").use())
            file.append(r"\\")
            if tool == "STARTS-no-update" or tool == "Ekstazi-no-update":
                file.append(r"\hline")
        # end for
        file.append("perfect")
        file.append(" & " + latex.Macro(f"{proj}-{model}-avg-perfect-select-rate").use())
        file.append(" & n/a")
        file.append(" & n/a")
        file.append(" & n/a")
        file.append(" & n/a")
        file.append(" & 100.00")
        file.append("& " + latex.Macro(f"{proj}-{model}-avg-perfect-select-rate").use())
        file.append(" & n/a")
        file.append(r"\\")
        # Footer
        file.append(r"\hline")
        file.append(r"\end{tabular}")
        file.append(r"\end{table*}")
        file.save()

    def make_table_mutated_eval_dataset_metrics(self, proj: str):
        # First step create numbers file
        file = latex.File(self.tables_dir / f"numbers-{proj}-mutated-eval-dataset-metrics.tex")
        res_metric = IOUtils.load(Macros.metrics_dir / f"stats-{proj}-mutated-eval-dataset.json")
        for k, v in res_metric.items():
            fmt = f",d" if type(v) == int else f",.2f"
            file.append_macro(latex.Macro(f"{proj}-mutated-eval-data-{k}", f"{v: {fmt}}"))
        # end for
        file.save()
        # Second step create table for eval-data
        file = latex.File(self.tables_dir / f"table-{proj}-mutated-eval-data-metrics.tex")
        heads = list(res_metric.keys())
        # Header
        file.append(r"\begin{table*}")
        file.append(r"\centering")
        file.append(r"\caption{Mutated eval data statistics for " + proj.split('_')[1] + "}")
        file.append(r"\vspace{-5pt}")
        file.append(r"\begin{tabular}{l|ccc}")
        file.append(r"\hline")

        file.append("{" + proj.split('_')[1] + "}")
        for h in heads:
            file.append(" & " + latex.Macro(h).use())
        file.append(r"\\")
        file.append(r"\hline")
        for h in heads:
            file.append("&" + latex.Macro(f"{proj}-mutated-eval-data-{h}").use())
        file.append(r"\\")
        # end for
        # Footer
        file.append(r"\hline")
        file.append(r"\end{tabular}")
        file.append(r"\end{table*}")
        file.save()

    def make_numbers_raw_dataset_metrics(self):
        # TODO: add test shas per project numbers.
        file = latex.File(self.tables_dir / f"numbers-raw-dataset-metrics.tex")

        dataset_metrics = IOUtils.load(Macros.results_dir / "metrics" / "raw-dataset-stats.json", IOUtils.Format.json)
        for proj_stats in dataset_metrics:
            name = proj_stats["PROJ_NAME"]
            for k, v in proj_stats.items():
                if k != "PROJ_NAME":
                    if type(v) == float:
                        fmt = f",.2f"
                    elif type(v) == int:
                        fmt = f",d"
                    else:
                        fmt = ""
                    file.append_macro(latex.Macro(f"{name}-{k}", f"{v:{fmt}}"))
                # end if
            # end for
        # end for
        file.save()

    def make_numbers_mutation_metrics(self, proj: str):
        """
        Make latex macro number file about the mutants generated from each project, and the status.
        source metric file: - stats-{project}-mutants.json
                            - stats-{project}-recover-mutants.json
        """
        file = latex.File(self.tables_dir / f"numbers-{proj}-mutations-metrics.tex")
        mut_metrics = IOUtils.load(self.metrics_dir / f"stats-{proj}-mutants.json", IOUtils.Format.json)
        pit_log = IOUtils.load(self.metrics_dir / f"stats-{proj}-pitlog.json", IOUtils.Format.json)
        proj_total_mut = 0
        proj_total_num = {
            "KILLED": 0,
            "NO_COVERAGE": 0,
            "TIMED_OUT": 0,
            "SURVIVED": 0
        }
        for mt_typ, stat in mut_metrics.items():
            for st, num in stat.items():
                fmt = f",d" if type(num) == int else f",.2f"
                file.append_macro(latex.Macro(f"{proj}-{mt_typ}-{st}", f"{num:{fmt}}"))
                if st == "total":
                    proj_total_mut += num
                if st in proj_total_num:
                    proj_total_num[st] += num
            # end for
        # end for

        fmt = f",d"
        file.append_macro(latex.Macro(f"{proj}-ALL", f"{proj_total_mut:{fmt}}"))
        for st, num in proj_total_num.items():
            file.append_macro(latex.Macro(f"{proj}-{st}-ALL", f"{num: {fmt}}"))
        # end for
        for k, v in pit_log.items():
            if k != "project" and k != "time":
                fmt = f",d" if type(v) == int else f",.2f"
                file.append_macro(latex.Macro(f"{proj}-{k}-ALL", f"{v: {fmt}}"))
            # end if
            elif k == "time":
                file.append_macro(latex.Macro(f"{proj}-{k}-ALL", v))
        # end for
        # Collect mutants recovered by us
        mut_metrics = IOUtils.load(Macros.results_dir / "metrics" / f"stats-{proj}-recover-mutants.json",
                                   IOUtils.Format.json)
        proj_total_mut = 0
        proj_total_num = {
            "KILLED": 0,
            "NO_COVERAGE": 0,
            "TIMED_OUT": 0,
            "SURVIVED": 0
        }
        for mt_typ, stat in mut_metrics.items():
            for st, num in stat.items():
                fmt = f",d" if type(num) == int else f",.2f"
                file.append_macro(latex.Macro(f"{proj}-{mt_typ}-{st}-recovered", f"{num:{fmt}}"))
                if st == "total":
                    proj_total_mut += num
            # end for
        # end for
        fmt = f",d"
        file.append_macro(latex.Macro(f"{proj}-ALL-recovered", f"{proj_total_mut:{fmt}}"))
        for st, num in proj_total_num.items():
            file.append_macro(latex.Macro(f"{proj}-{st}-ALL-recovered", f"{num: {fmt}}"))
        # end for
        file.save()

    def make_numbers_model_data_metrics(self):
        file = latex.File(self.tables_dir / f"numbers-model-data-metrics.tex")

        dataset_metrics = IOUtils.load(Macros.results_dir / "metrics" / "model-data-stats.json", IOUtils.Format.json)
        for data_type in [Macros.train, Macros.test]:
            for k, v in dataset_metrics[data_type].items():
                fmt = f",d" if type(v) == int else f",.2f"
                file.append_macro(latex.Macro(f"{data_type}-{k}", f"{v:{fmt}}"))
            # end for
        # end for
        file.save()

    def make_numbers_raw_eval_data_metrics(self):
        """
        Make latex macro number file about the changed line number of each SHA in each project.
        """
        file = latex.File(self.tables_dir / f"numbers-raw-eval-dataset-stats.tex")
        eval_data_metrics = IOUtils.load(Macros.results_dir / "metrics" / f"raw-eval-dataset-stats.json",
                                         IOUtils.Format.json)
        for eval_data_of_each_project in eval_data_metrics:
            for typename, num in eval_data_of_each_project.items():
                proj = eval_data_of_each_project["PROJ_NAME"]
                if typename != "PROJ_NAME" and typename != "FAILED_SHA_LIST":
                    fmt = f",d" if type(num) == int else f",.2f"
                    file.append_macro(latex.Macro(f"{proj}-raw-eval-{typename}", f"{num:{fmt}}"))
        # end for
        file.save()

    def make_table_raw_eval_data_metrics(self):
        file = latex.File(self.tables_dir / "table-raw-eval-data-metrics.tex")
        # Header
        file.append(r"\begin{table*}")
        file.append(r"\centering")
        file.append(r"\caption{Stats of Raw Eval Data.}")
        file.append(r"\vspace{-11pt}")
        file.append(r"\begin{tabular}{l|p{1.2cm}p{1.2cm}p{1.2cm}p{1.2cm}p{1.2cm}p{1.2cm}p{1.2cm}p{1.2cm}p{1.2cm}}")
        file.append(r"\hline")

        file.append(r"\textbf{Project} & "
                    r"\textbf{\# Total SHAs} &"
                    r"\textbf{\# Failed SHAs} &"
                    r"\textbf{\# Total Tests} &"
                    r"\textbf{\# EKSTAZI Passed Tests} &"
                    r"\textbf{\# EKSTAZI Failed Tests} &"
                    r"\textbf{\# STARTS Passed Tests} &"
                    r"\textbf{\# STARTS Failed Tests} &"
                    r"\textbf{\# Changed Lines} & "
                    r"\textbf{\# Avg Changed Lines} \\")
        file.append(r"\hline")

        for proj in Macros.raw_eval_projects:
            file.append(f"{proj}")
            for m in ["TOTAL_SHAS", "FAILED_TEST_SHAS", "TOTAL_TESTS", "EKSTAZI_PASSED_TEST_NUM",
                      "EKSTAZI_FAILED_TEST_NUM", "STARTS_PASSED_TEST_NUM", "STARTS_FAILED_TEST_NUM", "CHANGED_LINES",
                      "AVG_CHANGED_LINES"]:
                key = f"{proj}-raw-eval-{m}"
                file.append(" & " + latex.Macro(key).use())
                # end if
            # end for
            file.append(r"\\")
            # end for

            # Footer
        file.append(r"\hline")
        file.append(r"\end{tabular}")
        file.append(r"\end{table*}")
        file.save()

    def make_table_model_data_metrics(self):
        file = latex.File(self.tables_dir / "table-model-data-metrics.tex")
        # Header
        file.append(r"\begin{table*}")
        file.append(r"\centering")

        file.append(r"\begin{tabular}{l|ccc}")
        file.append(r"\hline")

        file.append(r"\textbf{Data type} & "
                    r"\textbf{\# failed SHA} & "
                    r"\textbf{avg. failed test classes} &"
                    r"\textbf{avg. passed tet classes} \\")
        file.append(r"\hline")

        for data_type in [Macros.train, Macros.test]:
            file.append(f"{data_type}")
            for m in ["FAILED_BUILD_NUM", "AVG_FAIL_PER_BUILD", "AVG_PASSED_PER_BUILD"]:
                key = f"{data_type}-{m}"
                file.append(" & " + latex.Macro(key).use())
                # end if
            # end for
            file.append(r"\\")
            # end for

            # Footer
        file.append(r"\hline")
        file.append(r"\end{tabular}")
        file.append(r"\end{table*}")
        file.save()

    def make_table_raw_dataset_metrics(self):
        file = latex.File(self.tables_dir / "table-raw-dataset-metrics.tex")
        proj_list = IOUtils.load(self.metrics_dir / "project-list.json")
        # Header
        file.append(r"\begin{table*}")
        file.append(r"\caption{Stats of Raw Eval Data.}")
        file.append(r"\vspace{-11pt}")
        file.append(r"\centering")

        file.append(r"\begin{tabular}{l|p{1.3cm}p{1.3cm}p{1.3cm}p{1.3cm}p{1.3cm}p{1.3cm}p{1.3cm}p{1.3cm}}")
        file.append(r"\hline")

        file.append(r"\textbf{Projects} & "
                    r"\textbf{\# failed SHA} & "
                    r"\textbf{\# failed to build} & "
                    r"\textbf{avg. failed test classes} &"
                    r"\textbf{avg. passed test classes} &"
                    r"\textbf{max \# test classes} &"
                    r"\textbf{min \# test classes} &"
                    r"\textbf{max \# test methods} &"
                    r"\textbf{min \# test methods} \\")
        file.append(r"\hline")

        for proj in proj_list:
            file.append(f"{proj}")
            for m in ["FAILED_BUILD_NUM", "FAILED_TO_BUILD", "AVG_FAIL_PER_BUILD", "AVG_PASSED_PER_BUILD",
                      "MAX_TEST_CASE_PER_BUILD",
                      "MIN_TEST_CASE_PER_BUILD", "MAX_TEST_METHOD_PER_BUILD", "MIN_TEST_METHOD_PER_BUILD"]:
                key = f"{proj}-{m}"
                file.append(" & " + latex.Macro(key).use())
                # end if
            # end for
            file.append(r"\\")
            # end for

            # Footer
        file.append(r"\hline")
        file.append(r"\end{tabular}")
        file.append(r"\end{table*}")
        file.save()

    def make_table_project_mutations_metrics(self, project: str):
        """ Generate table showing the mutants generated by PIT for project."""
        file = latex.File(self.tables_dir / f"table-{project}-PIT-mutations-metrics.tex")
        metrics = IOUtils.load(self.metrics_dir / f"stats-{project}-mutants.json")
        # headers = [str(h) for h in metrics["BooleanFalseReturnValsMutator"].keys()]
        proj_name = project.split("_")[1]

        # Header
        file.append(r"\begin{table*}")
        file.append(r"\footnotesize")
        file.append(r"\centering")
        file.append(r"\caption{\TCPITMutants{" + project.split('_')[1] + "}}")
        file.append(r"\vspace{-5pt}")
        file.append(r"\begin{tabular}{l|cccccc|cc}")
        file.append(r"\hline")

        file.append(r"\textbf{Projects} & "
                    r" \textbf{Mutator} ")
        for h in ["total", "KILLED", "SURVIVED", "NO_COVERAGE", "TIMED_OUT", "TOTAL_TIME", "TEST_NUM"]:
            file.append(r"& \textbf{" + latex.Macro(f"TH-mutant-{h}").use() + "}")
        file.append(r"\\")
        file.append(r"\hline")

        mutators = list(metrics.keys())
        mutators = set(mutators)

        mul_row = len(mutators)
        file.append(r"\multirow{" + f"{mul_row + 1}" + r"}{*}{\rotatebox[origin=c]{90}{" + f"{proj_name}" + r"}}")
        for mt_typ in mutators:
            file.append(f"& {mt_typ}")
            for st in ["total", "KILLED", "SURVIVED", "NO_COVERAGE", "TIMED_OUT"]:
                key = f"{project}-{mt_typ}-{st}"
                file.append(" & " + latex.Macro(key).use())
            # end if
            for i in range(2):
                file.append(" & n/a")
            # end for
            file.append(r"\\")
        # end for
        file.append(r"\hline")
        file.append(f"& All")
        key = f"{project}-ALL"
        file.append(" & " + latex.Macro(key).use())
        for st in ["KILLED", "SURVIVED", "NO_COVERAGE", "TIMED_OUT", "time", "test_num"]:
            key = f"{project}-{st}-ALL"
            file.append(" & " + latex.Macro(key).use())
        file.append(r"\\")
        file.append(r"\hline")
        # end for

        # Footer
        # file.append(r"\hline")
        file.append(r"\end{tabular}")
        file.append(r"\end{table*}")
        file.save()

    def make_numbers_avg_best_safe_select_rate(self):
        """Make numbers for the average best safe selection rate"""
        metrics_file = Macros.metrics_dir / "stats-avg-best-safe-selection-rate.json"
        numbers_file = latex.File(self.tables_dir / "numbers-rank-model-avg-subset-select-results-metrics.tex")
        best_safe_selection_rates = IOUtils.load(metrics_file)
        for subset, stats in best_safe_selection_rates.items():
            for model, number in stats.items():
                fmt = f",d" if type(number) == int else f",.2f"
                numbers_file.append_macro(
                    latex.Macro(f"avg-{model}-{subset}-select-best-safe-selection-rate", f"{number: {fmt}}"))
            # end for
        # end for
        numbers_file.save()

    def make_table_avg_best_safe_select_rate(self):
        """Make the table for average best safe selection rate for all the models"""
        file = latex.File(self.tables_dir / f"table-avg-best-safe-select-rate.tex")
        loaded_latex = latex.Macro.load_from_file(
            self.tables_dir / "numbers-rank-model-avg-subset-select-results-metrics.tex")
        models = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS",
                  "Fail-Basic-BM25Baseline", "Ekstazi-Basic-BM25Baseline", "boosting", "Ekstazi", "STARTS"]
        field_names = ["All", "Ekstazi", "STARTS"]

        file.append(r"\begin{table*}")
        file.append(
            r"\caption{Comparison of average best safe selection rate of models that select from the full set of tests.\vspace{-5pt}}")
        file.append(r"\small")
        file.append(r"\begin{tabular}{l|ccccccccc|cc}")
        file.append(r"\hline")
        file.append(r"\textbf{Subset} ")
        for m in models:
            file.append(r"& \textbf{" + latex.Macro(m).use() + "}")
        file.append(r"\\")
        file.append(r"\hline")

        for subset in field_names:

            values = []
            for model in models:
                key = f"avg-{model}-{subset}-select-best-safe-selection-rate"

                values.append(float(loaded_latex.get(key).value))
            # end for
            values.sort()
            file.append(subset)
            for model in models:
                key = f"avg-{model}-{subset}-select-best-safe-selection-rate"
                try:
                    if float(loaded_latex.get(key).value) == values[0]:
                        file.append(r" & \textbf{" + latex.Macro(key).use() + "}")
                    elif float(loaded_latex.get(key).value) == values[-1]:
                        file.append(r" & \cellcolor{black} " + latex.Macro(key).use() + "")
                    else:
                        file.append(" & " + latex.Macro(key).use())
                except AttributeError:
                    pass
            file.append(r"\\")

        file.append(r"\bottomrule")
        file.append(r"\end{tabular}")
        file.append(r"\end{table*}")
        file.save()

    def make_table_dataset_table(self, projects: List[str]):
        """
        Make the table for the dataset: training and eval for the dataset.tex
        :param projects:
        :return:
        """
        from pts.main import proj_logs
        file = latex.File(self.tables_dir / f"table-dataset.tex")
        file.append(r"\begin{table*}")
        file.append(r"\caption{Projects and Statistics of dataset.\vspace{-5pt}}")
        file.append(r"\label{tab:dataset}")
        file.append(r"\centering")
        file.append(r"\begin{tabular}{l|ll|ll}")
        file.append(r"\hline")
        file.append(r"\multirow{2}{*}{")
        file.append(r"\textbf{Projects}} &")
        file.append(
            r"\multicolumn{2}{c|}{\textbf{Train}} & \multicolumn{2}{c}{\textbf{Evaluation}}")
        file.append(r"\\")
        file.append(r" & \textbf{SHA}")
        file.append(r"& \textbf{\# Mutants}")
        file.append(r"& \textbf{\# SHAs}")
        file.append(r"& \textbf{\# Mutants}")
        file.append(r"\\")
        file.append(r"\hline")

        # Start to fill in the numbers

        for project in projects:
            project_name = project.split('_')[1]
            file.append(r"\UseMacro{" + project_name + "}")
            file.append(f" & {proj_logs[project]}")
            macro = f"{project}-ALL"
            file.append("& " + latex.Macro(macro).use())
            macro = f"{project}-mutated-eval-data-eval-shas"
            file.append(f"& " + latex.Macro(macro).use())
            macro = f"{project}-mutated-eval-data-mutants"
            file.append(f"& " + latex.Macro(macro).use())
            file.append(r"\\")
        # end for
        file.append(r"\bottomrule")
        file.append(r"\end{tabular}")
        file.append(r"\end{table*}")
        file.save()

    def make_table_real_failures_paper(self):
        """Table: Comparison of best safe selection rate of models that select from the full set of tests"""
        file = latex.File(self.tables_dir / f"table-real-failure-eval-results.tex")
        models = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS",
                  "BM25Baseline"]
        # ensemble_models = ["Fail-Basic-BM25Baseline", "Ekstazi-Basic-BM25Baseline"]
        # boosting_models = ["boosting"]
        tools = ["Ekstazi", "STARTS"]
        field_names = ["ekstazi", "starts"]

        for field_name in field_names:
            models_2_best_num = defaultdict(int)
            models_2_worst_num = defaultdict(int)

            file.append(r"\begin{table}[h]")
            file.append(r"\centering")
            if field_name == "all":
                file.append(
                    r"\caption{Comparison of best safe selection rate of models that select from the full set of tests for real failed tests.\vspace{-5pt}}")
                file.append(r"\label{table:compare:real-failure-rates}")
                file.append(r"\begin{tabular}{l|rrr|rrr|r|rr|r|rr}")
                file.append(r"\hline")
                file.append(r"\multirow{2}{*}{")
                file.append(r"\textbf{SHAs}} &")
                file.append(
                    r"\multicolumn{3}{c|}{\textbf{Fail}} & \multicolumn{3}{c|}{\textbf{\Ekstazi}} & \multicolumn{1}{c|}{\textbf{Baseline}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Fail-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{boosting}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{STARTS}}}")
                file.append(r"\\")
                file.append(
                    r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{BM25} &  &  &  &  &")
            elif field_name == "ekstazi":
                models = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS",
                          "BM25Baseline",
                          "Ekstazi"]
                file.append(
                    r"\caption{Comparison of best safe selection rate of models that select from subset of Ekstazi for real failing tests.\vspace{-5pt}}")
                file.append(r"\label{table:compare:real-failure-rates}")
                file.append(r"\scalebox{0.9}{")
                file.append(r"\begin{tabular}{l|rrr|rrr|r|r}")
                file.append(r"\hline")
                file.append(r"\multirow{2}{*}{")
                file.append(r"\textbf{SHAs}} &")
                file.append(
                    r"\multicolumn{3}{c|}{\textbf{Fail}} & \multicolumn{3}{c|}{\textbf{\Ekstazi}} & \multicolumn{1}{c|}{\textbf{Baseline}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi}}}")
                file.append(r"\\")
                file.append(
                    r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{BM25} ")
            elif field_name == "starts":
                models = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS",
                          "BM25Baseline", "randomforest",
                          "STARTS"]
                file.append(
                    r"\caption{Comparison of best safe selection rate of models that select from subset of STARTS for real failing tests.\vspace{-5pt}}")
                file.append(r"\label{table:compare:real-failure-rates}")
                file.append(r"\scalebox{0.9}{")
                file.append(r"\begin{tabular}{l|rrr|rrr|r|r|r}")
                file.append(r"\hline")
                file.append(r"\multirow{2}{*}{")
                file.append(r"\textbf{SHAs}} &")
                file.append(
                    r"\multicolumn{3}{c|}{\textbf{Fail}} & \multicolumn{3}{c|}{\textbf{\Ekstazi}} & \multicolumn{1}{c|}{\textbf{Baseline}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{randomforest}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{STARTS}}}")
                file.append(r"\\")
                file.append(
                    r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{BM25} &  ")

            file.append(r"\\")
            file.append(r"\hline")

            first_time_failed_tests = IOUtils.load(
                f"{Macros.raw_eval_data_dir}/first_time_failed_tests_evaled_no_rule.json")

            for first_time_failed_test_item in first_time_failed_tests:
                model_2_best_safe_select_rate = defaultdict(float)
                commit = first_time_failed_test_item["commit"]
                project = first_time_failed_test_item["project"].split("_")[-1]
                file.append(f"{project}-{commit}")

                # Load the model results
                values = []
                model_results_latex = latex.Macro.load_from_file(
                    self.tables_dir / f"numbers-real-failed-test_no_rule.tex")
                # Log the results for all the models including tools
                for model in models:
                    if model not in tools:
                        key = f"no-rule-{commit}-{model}-{field_name}_best_selection_rate"
                    else:
                        key = f"no-rule-{commit}-Fail-Code-{model.lower()}_selection_rate"
                    if not ((field_name == "ekstazi" and model == "Ekstazi") or (
                            field_name == "starts" and model == "STARTS")):
                        values.append(float(model_results_latex.get(key).value))

                values.sort()

                for model in models:
                    if model in tools:
                        key = f"no-rule-{commit}-Fail-Code-{model.lower()}_selection_rate"
                    else:
                        key = f"no-rule-{commit}-{model}-{field_name}_best_selection_rate"
                    if not (field_name != "all" and model in tools):
                        model_2_best_safe_select_rate[model] = float(model_results_latex.get(key).value)
                    if not ((field_name == "ekstazi" and model == "Ekstazi") or (
                            field_name == "starts" and model == "STARTS")):
                        if float(model_results_latex.get(key).value) == values[0]:
                            file.append(r" & \textbf{" + latex.Macro(key).use() + "}")
                        elif float(model_results_latex.get(key).value) == values[-1]:
                            file.append(r" & \cellcolor{black!20} " + latex.Macro(key).use() + "")
                        else:
                            file.append(" & " + latex.Macro(key).use())
                    else:
                        file.append(" & " + latex.Macro(key).use())
                    # end if
                # end for

                # Count number of best
                models_2_best_num, models_2_worst_num = collect_best_worst_num(model_2_best_safe_select_rate,
                                                                               models_2_best_num,
                                                                               models_2_worst_num)
                file.append(r"\\")
            # end for
            # Start to add rows for num of best and num of worst
            file.append(r"\hline")
            file.append(r"\# Best")
            for model in models:
                if model in tools and field_name != "all":
                    file.append(f"& N/A")
                else:
                    file.append(f"& {models_2_best_num[model]}")
            # end for
            file.append(r"\\")
            file.append(r"\# Worst")
            for model in models:
                if model in tools and field_name != "all":
                    file.append(f"& N/A")
                else:
                    file.append(f"& {models_2_worst_num[model]}")
            # end for
            file.append(r"\\")

            file.append(r"\bottomrule")
            file.append(r"\end{tabular}")
            file.append(r"}")  # close \scalebox{0.75}{
            file.append(r"\end{table}")
        # end for
        file.save()

    def make_table_execution_time_paper(self, projects: List[str], models, subset: str):
        """
        First make macro numbers for exectution time, then make table for it.
        subset should be Ekstazi or STARTS
        """
        from statistics import mean
        macro_file = latex.File(self.tables_dir / f"numbers-models-{subset}-execution-time.tex")
        fmt = f",.2f"
        for project in projects:
            metrics_file_path = self.metrics_dir / f"stats-{project}-execution-time-{subset.lower()}.json"
            metrics = IOUtils.load(metrics_file_path)
            for model, stats in metrics.items():
                if (model == "randomforest" or model == "xgboost") and (project == "apache_commons-csv"):
                    continue
                time_list = []
                for sha, time in stats.items():
                    time_list.append(time)
                # end for
                average_execution_time = mean(time_list)
                macro_file.append_macro(
                    latex.Macro(f"{model}-{project}-{subset}-avg-execution-time", f"{average_execution_time:{fmt}}"))
            # end for
        # end for
        macro_file.save()

        # Start to make table
        file = latex.File(self.tables_dir / f"table-{subset.lower()}-subset-execution-time.tex")
        file.append(r"\begin{table*}")
        file.append(r"\centering")
        file.append(
            r"\caption{Average execution time (seconds) combining models with " + subset + r".\vspace{-5pt}}")
        file.append(r"\label{table:execution-time:" + subset.lower() + "}")
        file.append(r"\begin{tabular}{l|rrr|rrr|r|rr|r|rr|r}")
        file.append(r"\hline")
        file.append(r"\multirow{2}{*}{")
        file.append(r"\textbf{Projects}} &")
        file.append(
            r"\multicolumn{3}{c|}{\textbf{Fail}} & \multicolumn{3}{c|}{\textbf{\Ekstazi}} & \multicolumn{1}{c|}{\textbf{Baseline}}")
        file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Fail-Basic-BM25Baseline}}}")
        file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi-Basic-BM25Baseline}}}")
        file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{boosting}}}")
        file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{randomforest}}}")
        file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{xgboost}}}")
        if subset == "Ekstazi":
            file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi}}}")
        elif subset == "STARTS":
            file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{STARTS}}}")
        file.append(r"\\")

        file.append(
            r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{BM25} &  &  &  &  &  & ")

        file.append(r"\\")
        file.append(r"\hline")

        # Start to fill in the macros
        for project in projects:
            project_name = project.split('_')[1]
            file.append(r"\UseMacro{" + project_name + "}")

            # Load the model results to find the maximun and min
            values = []
            model_results_latex = latex.Macro.load_from_file(
                self.tables_dir / f"numbers-models-{subset}-execution-time.tex")
            # Log the results for all the models including tools
            for model in models:
                if (model == "randomforest" or model == "xgboost") and (project == "apache_commons-csv"):
                    continue
                key = f"{model}-{project}-{subset}-avg-execution-time"
                values.append(float(model_results_latex.get(key).value))
            # end for

            tool_key = f"{subset}-{project}-{subset}-avg-execution-time"
            values.append(float(model_results_latex.get(tool_key).value))
            values.sort()

            # Start to fill in macros
            for model in models + [subset]:
                if (model == "randomforest" or model == "xgboost") and (project == "apache_commons-csv"):
                    file.append(" & NAN")
                    continue
                key = f"{model}-{project}-{subset}-avg-execution-time"
                v = float(model_results_latex.get(key).value)
                if v == values[0]:
                    file.append(r" & \textbf{" + latex.Macro(key).use() + "}")
                elif v == values[-1]:
                    file.append(r" & \cellcolor{black!20}" + latex.Macro(key).use() + "")
                else:
                    file.append(" & " + latex.Macro(key).use())
                # end if
            # end for

            file.append(r"\\")
        # end for
        file.append(r"\bottomrule")
        file.append(r"\end{tabular}")
        file.append(r"\end{table*}")
        file.save()

    def make_macros_end_to_end_time(self, projects: List[str], subset: str, models, change_selection_rate=False):
        """
        First make macro numbers for end to end time, then make table for it.
        subset should be Ekstazi or STARTS
        """
        models.append(subset)
        if subset.lower() == "ekstazi" and "randomforest" in models:
            models.remove("randomforest")

        if change_selection_rate:
            macro_file = latex.File(self.tables_dir / f"numbers-models-{subset.lower()}-end-to-end-time-change-selection-rate.tex")
        else:
            macro_file = latex.File(self.tables_dir / f"numbers-models-{subset.lower()}-end-to-end-time.tex")
        fmt = f",.2f"

        for project in projects:
            results = {}
            # selection_time_metrics_file_path = Macros.eval_data_dir / "mutated-eval-data" / f"{project}_time.json"
            selection_time_metrics_file_path = self.metrics_dir / f"{project}-{subset.lower()}-selection-time.json"
            if change_selection_rate:
                execution_time_metrics_file_path = self.metrics_dir / f"stats-{project}-execution-time-{subset.lower()}-change-selection-rate.json"
            else:
                execution_time_metrics_file_path = self.metrics_dir / f"stats-{project}-execution-time-{subset.lower()}.json"
            selection_time_metrics = IOUtils.load(selection_time_metrics_file_path)
            execution_time_metrics = IOUtils.load(execution_time_metrics_file_path)
            subset_selection_time = []
            for selection_time_item in selection_time_metrics:
                subset_selection_time.append(selection_time_item[subset])

            for model in models:
                if (model == "randomforest" or model == "xgboost") and (project == "apache_commons-csv"):
                    continue
                selection_time_list = []
                execution_time_list = []
                stats = execution_time_metrics[model]
                for sha, time in stats.items():
                    execution_time_list.append(time)
                for selection_time_item in selection_time_metrics:
                    print(selection_time_item[model])
                    selection_time_list.append(selection_time_item[model])
                # end for
                end_to_end_time_list = []
                if model.lower() == "ekstazi" or model.lower() == "starts":
                    end_to_end_time_list = [a + b for a, b in zip(selection_time_list, execution_time_list)]
                else:
                    end_to_end_time_list = [a + b + c for a, b, c in
                                            zip(selection_time_list, execution_time_list, subset_selection_time)]

                average_end_to_end_time = (sum(end_to_end_time_list)) / len(end_to_end_time_list)
                results[model] = average_end_to_end_time
                min_end_to_end_time = min(end_to_end_time_list)
                max_end_to_end_time = max(end_to_end_time_list)
                median_end_to_end_time = np.median(end_to_end_time_list)
                stddev_end_to_end_time = np.std(end_to_end_time_list)
                if change_selection_rate:
                    macro_file.append_macro(
                        latex.Macro(f"{model}-{project}-{subset}-avg-end-to-end-time-change-selection-rate", f"{average_end_to_end_time:{fmt}}"))
                    macro_file.append_macro(
                        latex.Macro(f"{model}-{project}-{subset}-min-end-to-end-time-change-selection-rate", f"{min_end_to_end_time:{fmt}}"))
                    macro_file.append_macro(
                        latex.Macro(f"{model}-{project}-{subset}-max-end-to-end-time-change-selection-rate", f"{max_end_to_end_time:{fmt}}"))
                    macro_file.append_macro(
                        latex.Macro(f"{model}-{project}-{subset}-median-end-to-end-time-change-selection-rate", f"{median_end_to_end_time:{fmt}}"))
                    macro_file.append_macro(
                        latex.Macro(f"{model}-{project}-{subset}-stddev-end-to-end-time-change-selection-rate", f"{stddev_end_to_end_time:{fmt}}"))
                else:
                    macro_file.append_macro(
                        latex.Macro(f"{model}-{project}-{subset}-avg-end-to-end-time", f"{average_end_to_end_time:{fmt}}"))
                    macro_file.append_macro(
                        latex.Macro(f"{model}-{project}-{subset}-min-end-to-end-time", f"{min_end_to_end_time:{fmt}}"))
                    macro_file.append_macro(
                        latex.Macro(f"{model}-{project}-{subset}-max-end-to-end-time", f"{max_end_to_end_time:{fmt}}"))
                    macro_file.append_macro(
                        latex.Macro(f"{model}-{project}-{subset}-median-end-to-end-time", f"{median_end_to_end_time:{fmt}}"))
                    macro_file.append_macro(
                        latex.Macro(f"{model}-{project}-{subset}-stddev-end-to-end-time", f"{stddev_end_to_end_time:{fmt}}"))
            # end for
            IOUtils.dump(Macros.metrics_dir/f"stats-{project}-avg-end2end-{subset}-subset-execution-time.json", results)
        # end for
        macro_file.save()

    def make_table_end_to_end_time(self, projects: List[str], subset: str, models: List[str], change_selection_rate=False):
        # Start to make table
        # This is based on best safe selection rate
        for data_type in ["avg", "min", "max", "stddev", "median"]:
            if change_selection_rate:
                file = latex.File(
                    self.tables_dir / f"table-{subset.lower()}-subset-{data_type}-end-to-end-time-change-selection-rate.tex")
            else:
                file = latex.File(self.tables_dir / f"table-{subset.lower()}-subset-{data_type}-end-to-end-time.tex")
            file.append(r"\begin{table}[t]")
            file.append(r"\centering")
            if data_type == "avg":
                if change_selection_rate:
                    file.append(
                        r"\caption{Average end-to-end testing time (seconds) combining models with " + subset + r". (change selection rate)}")
                else:
                    file.append(
                        r"\caption{Average end-to-end testing time (seconds) combining models with " + subset + r".}")
            elif data_type == "min":
                if change_selection_rate:
                    file.append(
                        r"\caption{Minimal end-to-end testing time (seconds) combining models with " + subset + r". (change selection rate)}")
                else:
                    file.append(
                        r"\caption{Minimal end-to-end testing time (seconds) combining models with " + subset + r".}")
            elif data_type == "max":
                if change_selection_rate:
                    file.append(
                        r"\caption{Maximal end-to-end testing time (seconds) combining models with " + subset + r". (change selection rate)}")
                else:
                    file.append(
                        r"\caption{Maximal end-to-end testing time (seconds) combining models with " + subset + r".}")
            elif data_type == "stddev":
                if change_selection_rate:
                    file.append(
                        r"\caption{Standard deviation end-to-end testing time (seconds) combining models with " + subset + r". (change selection rate)}")
                else:
                    file.append(
                        r"\caption{Standard deviation end-to-end testing time (seconds) combining models with " + subset + r".}")
            elif data_type == "median":
                if change_selection_rate:
                    file.append(
                        r"\caption{Median end-to-end testing time  (seconds) combining models with " + subset + r". (change selection rate)}")
                else:
                    file.append(
                        r"\caption{Median end-to-end testing time  (seconds) combining models with " + subset + r".}")
            if data_type == "avg":
                file.append(r"\label{table:end-to-end-time:" + subset.lower() + "}")
            # end if
            file.append(r"\scalebox{0.9}{")

            file.append(r"\begin{tabular}{l|rrr|rrr|r|r|r}") if subset == "STARTS" else \
                file.append(r"\begin{tabular}{l|rrr|rrr|r|r}")
            file.append(r"\hline")
            file.append(r"\multirow{2}{*}{")
            file.append(r"\textbf{Projects}} &")
            file.append(
                r"\multicolumn{3}{c|}{\textbf{Fail+}} &"
                r" \multicolumn{3}{c|}{\textbf{Ekstazi+}} &"
                r" \multicolumn{1}{c|}{\textbf{Baseline}}")

            if subset == "Ekstazi":
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi}}}")
                file.append(r"\\")
                file.append(
                    r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{Basic} & \textbf{Code} &"
                    r" \textbf{ABS} & \textbf{BM25} ")
            elif subset == "STARTS":
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{randomforest}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{STARTS}}}")
                file.append(r"\\")
                file.append(
                    r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{Basic} & \textbf{Code} & "
                    r"\textbf{ABS} & \textbf{BM25} &   &  ")
            # end if
            file.append(r"\\")
            file.append(r"\hline")

            # Start to fill in the macros
            for project in projects:
                project_name = project.split('_')[1]
                file.append(r"\UseMacro{" + project_name + "}")

                # Load the model results to find the maximun and min
                values = []
                if change_selection_rate:
                    model_results_latex = latex.Macro.load_from_file(
                        self.tables_dir / f"numbers-models-{subset.lower()}-end-to-end-time-change-selection-rate.tex")
                else:
                    model_results_latex = latex.Macro.load_from_file(
                        self.tables_dir / f"numbers-models-{subset.lower()}-end-to-end-time.tex")
                # Log the results for all the models including tools
                for model in models:
                    if (model == "randomforest" or model == "xgboost") and (project == "apache_commons-csv" or
                                                                            subset == "Ekstazi"):
                        continue
                    if change_selection_rate:
                        key = f"{model}-{project}-{subset}-{data_type}-end-to-end-time-change-selection-rate"
                    else:
                        key = f"{model}-{project}-{subset}-{data_type}-end-to-end-time"
                    values.append(float(model_results_latex.get(key).value))
                # end for

                if change_selection_rate:
                    tool_key = f"{subset}-{project}-{subset}-{data_type}-end-to-end-time-change-selection-rate"
                else:
                    tool_key = f"{subset}-{project}-{subset}-{data_type}-end-to-end-time"
                values.append(float(model_results_latex.get(tool_key).value))
                values.sort()

                # Start to fill in macros
                for model in models + [subset]:
                    if model == "randomforest" and subset == "Ekstazi":
                        continue
                    if (model == "randomforest" or model == "xgboost") and (project == "apache_commons-csv"):
                        file.append(" & N/A")
                        continue

                    if change_selection_rate:
                        key = f"{model}-{project}-{subset}-{data_type}-end-to-end-time-change-selection-rate"
                    else:
                        key = f"{model}-{project}-{subset}-{data_type}-end-to-end-time"

                    v = float(model_results_latex.get(key).value)

                    if v == values[0]:
                        file.append(r" & \textbf{" + latex.Macro(key).use() + "}")
                    elif v == values[-1]:
                        file.append(r" & \cellcolor{black!20}" + latex.Macro(key).use() + "")
                    else:
                        file.append(" & " + latex.Macro(key).use())
                    # end if
                # end for

                file.append(r"\\")
            # end for
            file.append(r"\bottomrule")
            file.append(r"\end{tabular}")
            file.append(r"}")  # } for scale box
            file.append(r"\end{table}")
            file.save()

    def make_table_auc_paper(self, projects: List[str], subset: str):
        """
        First make macro numbers for AUC, then make table for it.
        subset should be Ekstazi or STARTS
        """
        file = latex.File(self.tables_dir / f"table-subset-selection-auc.tex")

        # Start to make table
        models = ["Fail-Basic", "Fail-Code", "Fail-ABS",
                  "BM25Baseline", "Fail-Basic-BM25Baseline", "Ekstazi-Basic-BM25Baseline", "boosting",
                  "xgboost", "randomforest"]
        ensemble_models = self.ENSEMBLE_MODELS
        boosting_models = self.BOOSTING_MODELS
        tools = self.TOOLS
        field_names = ["All", "Ekstazi", "STARTS"]

        for field_name in field_names:
            models_2_best_num = defaultdict(int)
            models_2_worst_num = defaultdict(int)
            file.append(r"\begin{table*}")
            file.append(r"\centering")
            if field_name == "All":
                file.append(
                    r"\caption{Area under curve for selecting from all tests.\vspace{-5pt}}")
                file.append(r"\label{table:auc:all}")
                file.append(r"\begin{tabular}{l|rrr|r|rr|r|rr}")
                file.append(r"\hline")
                file.append(r"\multirow{2}{*}{")
                file.append(r"\textbf{Projects}} &")
                file.append(
                    r"\multicolumn{3}{c|}{\textbf{Fail}} & \multicolumn{1}{c|}{\textbf{Baseline}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Fail-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{boosting}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{randomforest}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{xgboost}}}")
                file.append(r"\\")
                file.append(
                    r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{BM25} &  & & & &")
            elif field_name == "Ekstazi":
                file.append(
                    r"\caption{Area under curve for selecting from Ekstazi selected tests.\vspace{-2pt}}")
                file.append(r"\label{table:auc:ekstazi}")
                file.append(r"\begin{tabular}{l|rrr|r|rr|r|rr}")
                file.append(r"\hline")
                file.append(r"\multirow{2}{*}{")
                file.append(r"\textbf{Projects}} &")
                file.append(
                    r"\multicolumn{3}{c|}{\textbf{Fail}} & \multicolumn{1}{c|}{\textbf{Baseline}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Fail-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{boosting}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{randomforest}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{xgboost}}}")
                file.append(r"\\")
                file.append(
                    r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{BM25} &  & & & &")
            elif field_name == "STARTS":
                file.append(
                    r"\caption{Area under curve for selecting from STARTS selected tests.\vspace{-2pt}}")
                file.append(r"\label{table:auc:starts}")
                file.append(r"\begin{tabular}{l|rrr|r|rr|r|rr}")
                file.append(r"\hline")
                file.append(r"\multirow{2}{*}{")
                file.append(r"\textbf{Projects}} &")
                file.append(
                    r"\multicolumn{3}{c|}{\textbf{Fail}} & \multicolumn{1}{c|}{\textbf{Baseline}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Fail-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{boosting}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{randomforest}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{xgboost}}}")
                file.append(r"\\")
                file.append(
                    r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{BM25} &  & & & &")
            file.append(r"\\")
            file.append(r"\hline")

            for project in projects:
                proj = project.split('_')[1]

                model_2_subset_auc_scores = defaultdict(float)
                values = []
                model_results_latex = latex.Macro.load_from_file(
                    self.tables_dir / f"numbers-{project}-auc-score.tex")
                # Log the results for all the models including tools
                for model in models:
                    key = f"{project}-{model}-{field_name}-subset-auc-score"
                    try:
                        values.append(float(model_results_latex.get(key).value))
                    except AttributeError:
                        continue
                # end for
                values.sort()
                file.append(proj)
                for model in models:
                    key = f"{project}-{model}-{field_name}-subset-auc-score"
                    try:
                        model_2_subset_auc_scores[model] = float(model_results_latex.get(key).value)
                    except AttributeError:
                        file.append(r" & N/A")
                        continue
                    if float(model_results_latex.get(key).value) == values[0]:
                        file.append(r" & \textbf{" + latex.Macro(key).use() + "}")
                    elif float(model_results_latex.get(key).value) == values[-1]:
                        file.append(r" & \cellcolor{black!20}" + latex.Macro(key).use() + "")
                    else:
                        file.append(" & " + latex.Macro(key).use())
                    # end if
                # end for

                # Count number of best
                models_2_best_num, models_2_worst_num = collect_best_worst_num(model_2_subset_auc_scores,
                                                                               models_2_best_num,
                                                                               models_2_worst_num)
                file.append(r"\\")
                # end for
                # Start to add rows for num of best and num of worst
            file.append(r"\hline")
            file.append(r"\# Best")
            for model in models:
                if model in tools and field_name != "all":
                    file.append(f"& N/A")
                else:
                    file.append(f"& {models_2_worst_num[model]}")
            # end for
            file.append(r"\\")
            file.append(r"\# Worst")
            for model in models:
                if model in tools and field_name != "all":
                    file.append(f"& N/A")
                else:
                    file.append(f"& {models_2_best_num[model]}")
            # end for
            file.append(r"\\")

            file.append(r"\bottomrule")
            file.append(r"\end{tabular}")
            file.append(r"\end{table*}")
            # end for
        file.save()

    def make_table_cmp_avg_safe_select_rate(self, projects: List[str]):
        """
        Make the table for summarizing the avg safe select rate for All, Ekstazi and STARTS.
        """
        file = latex.File(self.tables_dir / f"table-cmp-avg-safe-select-rate.tex")
        models = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS",
                  "BM25Baseline", "Fail-Basic-BM25Baseline", "Ekstazi-Basic-BM25Baseline", "boosting",
                  "Ekstazi", "STARTS"]
        ensemble_models = ["Fail-Basic-BM25Baseline", "Ekstazi-Basic-BM25Baseline"]
        boosting_models = ["boosting"]
        tools = ["Ekstazi", "STARTS"]
        field_names = ["avg-safe-select-rate", "Ekstazi-avg-select-subset-rate", "STARTS-avg-select-subset-rate"]

        for field_name in field_names:
            models_2_best_num = defaultdict(int)
            models_2_worst_num = defaultdict(int)
            file.append(r"\begin{table*}")
            file.append(r"\centering")
            if field_name == "avg-safe-select-rate":
                file.append(
                    r"\caption{Comparison of average safe selection rate of models that select from the full set of tests.\vspace{-5pt}}")
                file.append(r"\label{table:compare:rates}")
                file.append(r"\begin{tabular}{l|rrr|rrr|r|rr|r|rr}")
                file.append(r"\hline")
                file.append(r"\multirow{2}{*}{")
                file.append(r"\textbf{Projects}} &")
                file.append(
                    r"\multicolumn{3}{c|}{\textbf{Fail+}} & \multicolumn{3}{c|}{\textbf{Ekstazi+}} & \multicolumn{1}{c|}{\textbf{Baseline}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Fail-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{boosting}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{STARTS}}}")
                file.append(r"\\")
                file.append(
                    r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{BM25} &  &  &  &")
            elif field_name == "Ekstazi-avg-select-subset-rate":
                file.append(
                    r"\caption{Comparison of average safe selection rate of models that select from subset of Ekstazi.\vspace{-5pt}}")
                file.append(r"\label{table:combine:ekstazi}")
                models = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS",
                          "BM25Baseline", "Fail-Basic-BM25Baseline", "Ekstazi-Basic-BM25Baseline", "boosting",
                          "Ekstazi"]
                file.append(r"\begin{tabular}{l|rrr|rrr|r|rr|r|rr}")
                file.append(r"\hline")
                file.append(r"\multirow{2}{*}{")
                file.append(r"\textbf{Projects}} &")
                file.append(
                    r"\multicolumn{3}{c|}{\textbf{Fail+}} & \multicolumn{3}{c|}{\textbf{Ekstazi+}} & \multicolumn{1}{c|}{\textbf{Baseline}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Fail-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{boosting}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi}}}")
                file.append(r"\\")
                file.append(
                    r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{BM25} &  &   & ")
            elif field_name == "STARTS-avg-select-subset-rate":
                file.append(
                    r"\caption{Comparison of average safe selection rate of models that select from subset of STARTS.\vspace{-5pt}}")
                file.append(r"\label{table:combine:starts}")
                models = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS",
                          "BM25Baseline", "Fail-Basic-BM25Baseline", "Ekstazi-Basic-BM25Baseline",
                          "boosting", "STARTS"]
                file.append(r"\begin{tabular}{l|rrr|rrr|r|rr|r|rr}")
                file.append(r"\hline")
                file.append(r"\multirow{2}{*}{")
                file.append(r"\textbf{Projects}} &")
                file.append(
                    r"\multicolumn{3}{c|}{\textbf{Fail+}} & \multicolumn{3}{c|}{\textbf{Ekstazi+}} & \multicolumn{1}{c|}{\textbf{Baseline}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Fail-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{boosting}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{STARTS}}}")
                file.append(r"\\")
                file.append(
                    r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{BM25} &  &  &")

            file.append(r"\\")
            file.append(r"\hline")

            for project in projects:
                # First want to make sure we have numbers for the select subset
                # self.make_numbers_rank_model_eval_results(project)
                loaded_latex = latex.Macro.load_from_file(
                    self.tables_dir / f"numbers-{project}-rank-model-eval-results-metrics.tex")
                model_2_best_safe_select_rate = defaultdict(float)
                values = []
                for model in models:
                    key = f"{project}-{model}-{field_name}"
                    if not ((field_name == "Ekstazi-avg-select-subset-rate" and model == "Ekstazi") or (
                            field_name == "STARTS-avg-select-subset-rate" and model == "STARTS")):
                        if model in ensemble_models:
                            ensemble_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-rank-model-ensemble-eval-results-metrics.tex")
                            values.append(float(ensemble_latex.get(key).value))
                        elif model in boosting_models:
                            boosting_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-boosting-model-eval-results-metrics.tex")
                            values.append(float(boosting_latex.get(key).value))
                        elif "Baseline" in model:
                            baseline_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-rank-model-IR-baseline-eval-results-metrics.tex")
                            values.append(float(baseline_latex.get(key).value))
                        elif model not in tools:
                            values.append(float(loaded_latex.get(key).value))
                        else:  # Else the we need the Tool-best-select-rate
                            key = f"{project}-Fail-Basic-{model}-best-safe-select-rate"
                            values.append(float(loaded_latex.get(key).value))
                    # end if
                values.sort()
                project_name = project.split("_")[1]
                file.append(r"\UseMacro{" + project_name + "}")
                for model in models:
                    key = f"{project}-{model}-{field_name}"
                    try:
                        if model in ensemble_models:
                            loaded_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-rank-model-ensemble-eval-results-metrics.tex")
                        elif model in boosting_models:
                            loaded_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-boosting-model-eval-results-metrics.tex")
                        elif "Baseline" in model:
                            loaded_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-rank-model-IR-baseline-eval-results-metrics.tex")
                        elif model in tools:
                            key = f"{project}-Fail-Basic-{model}-best-safe-select-rate"
                            loaded_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-rank-model-eval-results-metrics.tex")
                        else:
                            loaded_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-rank-model-eval-results-metrics.tex")
                        # end if
                        if not (field_name != "avg-safe-select-rate" and model in tools):
                            model_2_best_safe_select_rate[model] = float(loaded_latex.get(key).value)
                        if not ((field_name == "Ekstazi-avg-select-subset-rate" and model == "Ekstazi") or (
                                field_name == "STARTS-avg-select-subset-rate" and model == "STARTS")):
                            if float(loaded_latex.get(key).value) == values[0]:
                                file.append(r" & \textbf{" + latex.Macro(key).use() + "}")
                            elif float(loaded_latex.get(key).value) == values[-1]:
                                file.append(r" & \cellcolor{black!20}" + latex.Macro(key).use() + "")
                            else:
                                file.append(" & " + latex.Macro(key).use())
                        else:
                            file.append(" & " + latex.Macro(key).use())
                    except AttributeError:
                        pass

                models_2_best_num, models_2_worst_num = collect_best_worst_num(model_2_best_safe_select_rate,
                                                                               models_2_best_num,
                                                                               models_2_worst_num)
                file.append(r"\\")
            # end for
            # Add rows to show the num of best and worst
            file.append(r"\hline")
            file.append(r"\# Best")
            for model in models:
                if model in tools and field_name != "avg-safe-select-rate":
                    file.append(f"& N/A")
                else:
                    file.append(f"& {models_2_best_num[model]}")
            # end for
            file.append(r"\\")
            file.append(r"\# Worst")
            for model in models:
                if model in tools and field_name != "avg-safe-select-rate":
                    file.append(f"& N/A")
                else:
                    file.append(f"& {models_2_worst_num[model]}")
            # end for
            file.append(r"\\")

            file.append(r"\bottomrule")
            file.append(r"\end{tabular}")
            file.append(r"\end{table*}")
        file.save()

    def make_table_cmp_best_safe_select_rate(self, projects: List[str]):
        """Make the table for summarizing the safe select rate for Ekstazi and STARTS.

        Note: the params are tuned for FASE 2022 paper format.
        """
        file = latex.File(self.tables_dir / f"table-cmp-best-safe-select-rate.tex")
        models = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS",
                  "BM25Baseline", "randomforest", "Ekstazi", "STARTS"]
        ensemble_models = ["Fail-Basic-BM25Baseline", "Ekstazi-Basic-BM25Baseline"]
        boosting_models = ["boosting"]
        tools = ["Ekstazi", "STARTS"]
        baselines = ["randomforest"]
        field_names = ["Ekstazi-select-subset-rate", "STARTS-select-subset-rate"]

        for field_name in field_names:
            models_2_best_num = defaultdict(int)
            models_2_worst_num = defaultdict(int)
            file.append(r"\begin{table}[t]")
            file.append(r"\centering")
            if field_name == "best-safe-select-rate":
                file.append(
                    r"\caption{Comparison of best safe selection rate of models that select from the full set of tests.}")
                file.append(r"\label{table:compare:rates}")
                file.append(r"\scalebox{1}{")
                file.append(r"\begin{tabular}{l|rrr|rrr|r|rr|r|rr|rr}")
                file.append(r"\hline")
                file.append(r"\multirow{2}{*}{")
                file.append(r"\textbf{Projects}} &")
                file.append(
                    r"\multicolumn{3}{c|}{\textbf{Fail+}} & \multicolumn{3}{c|}{\textbf{Ekstazi+}} & \multicolumn{1}{c|}{\textbf{Baseline}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Fail-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{boosting}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{randomforest}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{xgboost}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{STARTS}}}")
                file.append(r"\\")
                file.append(
                    r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{BM25} &  &  &  & & &")
            elif field_name == "Ekstazi-select-subset-rate":
                file.append(
                    r"\caption{Comparison of best safe selection rate of models that select from subset of Ekstazi.}")
                file.append(r"\label{table:combine:ekstazi}")
                file.append(r"\scalebox{1}{")
                models = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS",
                          "BM25Baseline", "Ekstazi"]
                file.append(r"\begin{tabular}{l|rrr|rrr|r|r}")
                file.append(r"\hline")
                file.append(r"\multirow{2}{*}{")
                file.append(r"\textbf{Projects}} &")
                file.append(
                    r"\multicolumn{3}{c|}{\textbf{Fail+}} & \multicolumn{3}{c|}{\textbf{Ekstazi+}} & \multicolumn{1}{c|}{\textbf{Baseline}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi}}}")
                file.append(r"\\")
                file.append(
                    r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{BM25} ")
            elif field_name == "STARTS-select-subset-rate":
                file.append(
                    r"\caption{Comparison of best safe selection rate of models that select from subset of STARTS.}")
                file.append(r"\label{table:combine:starts}")
                file.append(r"\scalebox{1}{")
                models = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS",
                          "BM25Baseline",
                          "randomforest", "STARTS"]
                file.append(r"\begin{tabular}{l|rrr|rrr|r|r|r}")
                file.append(r"\hline")
                file.append(r"\multirow{2}{*}{")
                file.append(r"\textbf{Projects}} &")
                file.append(
                    r"\multicolumn{3}{c|}{\textbf{Fail+}} & \multicolumn{3}{c|}{\textbf{Ekstazi+}} & \multicolumn{1}{c|}{\textbf{Baseline}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{randomforest}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{STARTS}}}")
                file.append(r"\\")
                file.append(
                    r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{BM25} &  ")

            file.append(r"\\")
            file.append(r"\hline")

            for project in projects:
                # First want to make sure we have numbers for the select subset
                # self.make_numbers_rank_model_eval_results(project)
                loaded_latex = latex.Macro.load_from_file(
                    self.tables_dir / f"numbers-{project}-rank-model-eval-results-metrics.tex")
                model_2_best_safe_select_rate = defaultdict(float)
                values = []
                for model in models:
                    key = f"{project}-{model}-{field_name}"
                    if not ((field_name == "Ekstazi-select-subset-rate" and model == "Ekstazi") or (
                            field_name == "STARTS-select-subset-rate" and model == "STARTS")):
                        if model in ensemble_models:
                            ensemble_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-rank-model-ensemble-eval-results-metrics.tex")
                            values.append(float(ensemble_latex.get(key).value))
                        elif model in boosting_models:
                            boosting_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-boosting-model-eval-results-metrics.tex")
                            values.append(float(boosting_latex.get(key).value))
                        elif "Baseline" in model:
                            baseline_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-rank-model-IR-baseline-eval-results-metrics.tex")
                            values.append(float(baseline_latex.get(key).value))
                        elif model in baselines:
                            if project == "apache_commons-csv":
                                continue
                            ealrts_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-EALRTS-eval-results-metrics.tex")
                            if field_name == "best-safe-select-rate":
                                key = f"{project}-{model}-best-safe-selection-rate"
                            elif field_name == "Ekstazi-select-subset-rate":
                                key = f"{project}-{model}-Ekstazi-subset-best-safe-selection-rate"
                            elif field_name == "STARTS-select-subset-rate":
                                key = f"{project}-{model}-STARTS-subset-best-safe-selection-rate"
                            values.append(float(ealrts_latex.get(key).value))
                        elif model not in tools:
                            values.append(float(loaded_latex.get(key).value))
                        else:  # Else the we need the Tool-best-select-rate
                            key = f"{project}-Fail-Basic-{model}-best-safe-select-rate"
                            values.append(float(loaded_latex.get(key).value))
                    # end if
                values.sort()
                project_name = project.split("_")[1]
                file.append(r"\UseMacro{" + project_name + "}")
                for model in models:
                    key = f"{project}-{model}-{field_name}"
                    try:
                        if model in ensemble_models:
                            loaded_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-rank-model-ensemble-eval-results-metrics.tex")
                        elif model in baselines:
                            if field_name == "best-safe-select-rate":
                                key = f"{project}-{model}-best-safe-selection-rate"
                            elif field_name == "Ekstazi-select-subset-rate":
                                key = f"{project}-{model}-Ekstazi-subset-best-safe-selection-rate"
                            elif field_name == "STARTS-select-subset-rate":
                                key = f"{project}-{model}-STARTS-subset-best-safe-selection-rate"
                            loaded_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-EALRTS-eval-results-metrics.tex")
                        elif model in boosting_models:
                            loaded_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-boosting-model-eval-results-metrics.tex")
                        elif "Baseline" in model:
                            loaded_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-rank-model-IR-baseline-eval-results-metrics.tex")
                        elif model in tools:
                            key = f"{project}-Fail-Basic-{model}-best-safe-select-rate"
                            loaded_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-rank-model-eval-results-metrics.tex")
                        else:
                            loaded_latex = latex.Macro.load_from_file(
                                self.tables_dir / f"numbers-{project}-rank-model-eval-results-metrics.tex")
                        # end if
                        if not (field_name != "best-safe-select-rate" and model in tools):
                            model_2_best_safe_select_rate[model] = float(loaded_latex.get(key).value)
                        if not ((field_name == "Ekstazi-select-subset-rate" and model == "Ekstazi") or (
                                field_name == "STARTS-select-subset-rate" and model == "STARTS")):
                            if float(loaded_latex.get(key).value) < 0:
                                file.append(r"& N/A")
                            elif float(loaded_latex.get(key).value) == values[0]:
                                file.append(r" & \textbf{" + latex.Macro(key).use() + "}")
                            elif float(loaded_latex.get(key).value) == values[-1]:
                                file.append(r" & \cellcolor{black!20}" + latex.Macro(key).use() + "")
                            else:
                                file.append(" & " + latex.Macro(key).use())
                        else:
                            file.append(" & " + latex.Macro(key).use())
                    except AttributeError:
                        pass

                models_2_best_num, models_2_worst_num = collect_best_worst_num(model_2_best_safe_select_rate,
                                                                               models_2_best_num,
                                                                               models_2_worst_num)
                file.append(r"\\")
            # end for
            # Add rows to show the num of best and worst
            file.append(r"\hline")
            file.append(r"\# Best")
            for model in models:
                if model in tools and field_name != "best-safe-select-rate":
                    file.append(f"& N/A")
                else:
                    file.append(f"& {models_2_best_num[model]}")
            # end for
            file.append(r"\\")
            file.append(r"\# Worst")
            for model in models:
                if model in tools and field_name != "best-safe-select-rate":
                    file.append(f"& N/A")
                else:
                    file.append(f"& {models_2_worst_num[model]}")
            # end for
            file.append(r"\\")

            file.append(r"\bottomrule")
            file.append(r"\end{tabular}")
            file.append(r"}") # close \scalebox{0.75}{
            file.append(r"\end{table}")
        file.save()

    def make_numbers_perfect_best_safe_selection_rate(self, projects: List[str]):
        """
        Suppose there is a perfect model, get the best safe selection rate
        """
        file = latex.File(self.tables_dir / f"numbers-perfect-best-safe-selection-rate.tex")
        fmt = f",.2f"
        for project in projects:
            mutated_eval_data = IOUtils.load(f"{Macros.data_dir}/mutated-eval-data/{project}-ag.json")
            largest_rate = 0
            for mutated_eval_item in mutated_eval_data:
                current_rate = len(mutated_eval_item["failed_test_list"]) / len(
                    mutated_eval_item["failed_test_list"] + mutated_eval_item["passed_test_list"])
                if largest_rate < current_rate:
                    largest_rate = current_rate
            file.append_macro(latex.Macro(f"{project}-perfect-best-safe-selection-rate", f"{largest_rate:{fmt}}"))
        file.save()

    def make_numbers_real_failed_test_metrics(self):
        MODELS = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS", "BM25Baseline",
                  "Fail-Basic-BM25Baseline", "Ekstazi-Basic-BM25Baseline", "boosting"]

        file = latex.File(self.tables_dir / f"numbers-real-failed-test.tex")
        fmt = f",.3f"
        first_time_failed_tests = IOUtils.load(f"{Macros.raw_eval_data_dir}/first_time_failed_tests_evaled.json")
        for first_time_failed_test_item in first_time_failed_tests:
            commit = first_time_failed_test_item["commit"]
            for model in MODELS:
                model_of_commit = first_time_failed_test_item[model]
                for key, value in model_of_commit.items():
                    file.append_macro(latex.Macro(f"{commit}-{model}-{key}", f"{value:{fmt}}"))
        file.save()

    def make_table_real_failed_test(self):
        MODELS = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS", "BM25Baseline",
                  "Fail-Basic-BM25Baseline", "Ekstazi-Basic-BM25Baseline"]
        other_metrics = ["perfect", "Ekstazi", "STARTS"]
        field_names = ["all"]
        file = latex.File(self.tables_dir / f"table-real-failed-test.tex")

        for field_name in field_names:
            file.append(r"\begin{table*}")
            if field_name == "all":
                file.append(
                    r"\caption{Eval on real failed tests: best selection rate from all tests.(Rule based selection)\vspace{-2pt}}")
            elif field_name == "ekstazi":
                file.append(
                    r"\caption{Eval on real failed tests: best selection rate from Ekstazi selected tests.(Rule based selection)\vspace{-2pt}}")
            elif field_name == "starts":
                file.append(
                    r"\caption{Eval on real failed tests: best selection rate from STARTS selected tests.(Rule based selection)\vspace{-2pt}}")
            file.append(r"\begin{scriptsize}")
            file.append(r"\centering")
            file.append(r"\begin{tabular}{l|ccccccccc|ccc}")
            file.append(r"\hline")

            file.append(r"\textbf{commit}")
            for m in MODELS:
                file.append(r"& \textbf{" + latex.Macro(m).use() + "}")
            for other_metric in other_metrics:
                file.append(r"& \textbf{" + other_metric + "}")
            file.append(r"\\")
            file.append(r"\hline")

            first_time_failed_tests = IOUtils.load(f"{Macros.raw_eval_data_dir}/first_time_failed_tests_evaled.json")
            for first_time_failed_test_item in first_time_failed_tests:
                commit = first_time_failed_test_item["commit"]
                project = first_time_failed_test_item["project"].split("_")[-1]
                file.append(r"\makecell[c]{" + project + r"\\" + commit + "}")

                min_value = 1.1
                max_value = 0
                model_results_latex = latex.Macro.load_from_file(
                    self.tables_dir / f"numbers-real-failed-test.tex")

                for model in MODELS:
                    key = f"{commit}-{model}-{field_name}_best_selection_rate"
                    if float(model_results_latex.get(key).value) < min_value:
                        min_value = float(model_results_latex.get(key).value)
                    if float(model_results_latex.get(key).value) > max_value:
                        max_value = float(model_results_latex.get(key).value)
                for model in MODELS:
                    key = f"{commit}-{model}-{field_name}_best_selection_rate"
                    if float(model_results_latex.get(key).value) == min_value:
                        file.append(r" & \textbf{" + latex.Macro(key).use() + "}")
                    elif float(model_results_latex.get(key).value) == max_value:
                        file.append(r" & \cellcolor{black!20}" + latex.Macro(key).use() + "")
                    else:
                        file.append(" & " + latex.Macro(key).use())
                file.append(" & " + latex.Macro(f"{commit}-Fail-Code-perfect_selection_rate").use())
                file.append(" & " + latex.Macro(f"{commit}-Fail-Code-ekstazi_selection_rate").use())
                file.append(" & " + latex.Macro(f"{commit}-Fail-Code-starts_selection_rate").use())
                file.append(r"\\")
                file.append(r"\hline")
            file.append(r"\bottomrule")
            file.append(r"\end{tabular}")
            file.append(r"\end{scriptsize}")
            file.append(r"\end{table*}")
        file.save()

    def make_numbers_real_failed_test_no_rule_metrics(self, changed_failed: bool):
        MODELS = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS", "BM25Baseline",
                  "Fail-Basic-BM25Baseline", "Ekstazi-Basic-BM25Baseline", "boosting"]
        if not changed_failed:
            file = latex.File(self.tables_dir / f"numbers-real-failed-test-no-rule.tex")
            fmt = f",.3f"
            first_time_failed_tests = IOUtils.load(
                f"{Macros.raw_eval_data_dir}/first_time_failed_tests_evaled_no_rule.json")
            for first_time_failed_test_item in first_time_failed_tests:
                commit = first_time_failed_test_item["commit"]
                for model in MODELS:
                    model_of_commit = first_time_failed_test_item[model]
                    for key, value in model_of_commit.items():
                        if key == "prediction_scores":
                            continue
                        file.append_macro(
                            latex.Macro(f"changed-not-failed-no-rule-{commit}-{model}-{key}", f"{value:{fmt}}"))
            file.save()

        else:
            file = latex.File(self.tables_dir / f"numbers-real-failed-test-changed-failed-no-rule.tex")
            fmt = f",.3f"
            first_time_failed_tests = IOUtils.load(
                f"{Macros.raw_eval_data_dir}/first_time_failed_tests_evaled_change_failed_no_rule.json")
            for first_time_failed_test_item in first_time_failed_tests:
                commit = first_time_failed_test_item["commit"]
                for model in MODELS:
                    model_of_commit = first_time_failed_test_item[model]
                    for key, value in model_of_commit.items():
                        if key == "prediction_scores":
                            continue
                        file.append_macro(
                            latex.Macro(f"changed-failed-no-rule-{commit}-{model}-{key}", f"{value:{fmt}}"))
            file.save()

    def make_table_real_failed_no_rule_test(self, changed_failed: bool):
        # MODELS = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS", "BM25Baseline",
        #           "Fail-Basic-BM25Baseline", "Ekstazi-Basic-BM25Baseline", "boosting"]
        other_metrics = ["perfect", "Ekstazi", "STARTS"]
        field_names = ["all", "ekstazi", "starts"]
        if not changed_failed:
            file = latex.File(self.tables_dir / f"table-real-failed-test-no-rule.tex")
        else:
            file = latex.File(self.tables_dir / f"table-real-failed-test-no-rule-changed-failed.tex")

        for field_name in field_names:
            file.append(r"\begin{table*}")

            if field_name == "all":
                if not changed_failed:
                    title = r"\caption{Eval on real failed tests: best selection rate from all tests. \vspace{-2pt}}"
                else:
                    title = r"\caption{Eval on real failed tests: best selection rate from all tests. (Changed test failed) \vspace{-2pt}}"
            elif field_name == "ekstazi":
                if not changed_failed:
                    title = r"\caption{Eval on real failed tests: best selection rate from Ekstazi selected tests. \vspace{-2pt}}"
                else:
                    title = r"\caption{Eval on real failed tests: best selection rate from Ekstazi selected tests.(Changed test failed) \vspace{-2pt}}"
            elif field_name == "starts":
                if not changed_failed:
                    title = r"\caption{Eval on real failed tests: best selection rate from STARTS selected tests. \vspace{-2pt}}"
                else:
                    title = r"\caption{Eval on real failed tests: best selection rate from STARTS selected tests.(Changed test failed)\vspace{-2pt}}"

            # file.append(r"\begin{scriptsize}")
            file.append(r"\centering")
            file.append(r"\tiny")

            if field_name == "all":
                models = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS",
                          "BM25Baseline", "Fail-Basic-BM25Baseline", "Ekstazi-Basic-BM25Baseline", "boosting", "perfect",
                          "Ekstazi", "STARTS"]
                # file.append(r"\label{table:compare:real-failure-rates}")
                file.append(title)
                file.append(r"\begin{tabular}{l|rrr|rrr|r|rr|r|rrr}")
                file.append(r"\hline")
                file.append(r"\multirow{2}{*}{")
                file.append(r"\textbf{SHAs}} &")
                file.append(
                    r"\multicolumn{3}{c|}{\textbf{Fail}} & \multicolumn{3}{c|}{\textbf{\Ekstazi}} & \multicolumn{1}{c|}{\textbf{Baseline}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Fail-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{boosting}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{perfect}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{STARTS}}}")
                file.append(r"\\")
                file.append(
                    r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{BM25} &  &  &  &  &")
            elif field_name == "ekstazi":
                models = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS",
                          "BM25Baseline", "Fail-Basic-BM25Baseline", "Ekstazi-Basic-BM25Baseline", "boosting", "perfect",
                          "Ekstazi"]
                file.append(title)
                # file.append(r"\label{table:compare:real-failure-rates}")
                file.append(r"\begin{tabular}{l|rrr|rrr|r|rr|r|rr}")
                file.append(r"\hline")
                file.append(r"\multirow{2}{*}{")
                file.append(r"\textbf{SHAs}} &")
                file.append(
                    r"\multicolumn{3}{c|}{\textbf{Fail}} & \multicolumn{3}{c|}{\textbf{\Ekstazi}} & \multicolumn{1}{c|}{\textbf{Baseline}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Fail-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{boosting}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{perfect}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi}}}")
                file.append(r"\\")
                file.append(
                    r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{BM25} &  &  &  & &")
            elif field_name == "starts":
                models = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS",
                          "BM25Baseline", "Fail-Basic-BM25Baseline", "Ekstazi-Basic-BM25Baseline", "boosting", "perfect",
                          "STARTS"]
                file.append(title)
                # file.append(r"\label{table:compare:real-failure-rates}")
                file.append(r"\begin{tabular}{l|rrr|rrr|r|rr|r|rr}")
                file.append(r"\hline")
                file.append(r"\multirow{2}{*}{")
                file.append(r"\textbf{SHAs}} &")
                file.append(
                    r"\multicolumn{3}{c|}{\textbf{Fail}} & \multicolumn{3}{c|}{\textbf{\Ekstazi}} & \multicolumn{1}{c|}{\textbf{Baseline}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Fail-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{Ekstazi-Basic-BM25Baseline}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{boosting}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{perfect}}}")
                file.append(r"& \multirow{2}{*}{\textbf{\UseMacro{STARTS}}}")
                file.append(r"\\")
                file.append(
                    r"& \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{Basic} & \textbf{Code} & \textbf{ABS} & \textbf{BM25} &  &  &  & &")

            file.append(r"\\")
            file.append(r"\hline")

            if not changed_failed:
                first_time_failed_tests = IOUtils.load(
                    f"{Macros.raw_eval_data_dir}/first_time_failed_tests_evaled_no_rule.json")
            else:
                first_time_failed_tests = IOUtils.load(
                    f"{Macros.raw_eval_data_dir}/first_time_failed_tests_evaled_change_failed_no_rule.json")

            if not changed_failed:
                model_results_latex = latex.Macro.load_from_file(
                    self.tables_dir / f"numbers-real-failed-test-no-rule.tex")
            else:
                model_results_latex = latex.Macro.load_from_file(
                    self.tables_dir / f"numbers-real-failed-test-changed-failed-no-rule.tex")

            for index, first_time_failed_test_item in enumerate(first_time_failed_tests):
                commit = first_time_failed_test_item["commit"]
                project = first_time_failed_test_item["project"].split("_")[-1]
                file.append(r"\makecell[c]{" + project + commit + "}")

                min_value = 1.1
                max_value = 0

                for model in models:
                    if not changed_failed:
                        if model in other_metrics:
                            key = f"changed-not-failed-no-rule-{commit}-Fail-Code-{model.lower()}_selection_rate"
                        else:
                            key = f"changed-not-failed-no-rule-{commit}-{model}-{field_name}_best_selection_rate"
                    else:
                        if model in other_metrics:
                            key = f"changed-failed-no-rule-{commit}-Fail-Code-{model.lower()}_selection_rate"
                        else:
                            key = f"changed-failed-no-rule-{commit}-{model}-{field_name}_best_selection_rate"

                    if float(model_results_latex.get(key).value) < min_value:
                        min_value = float(model_results_latex.get(key).value)
                    if float(model_results_latex.get(key).value) > max_value:
                        max_value = float(model_results_latex.get(key).value)
                for model in models:
                    if not changed_failed:
                        if model in other_metrics:
                            key = f"changed-not-failed-no-rule-{commit}-Fail-Code-{model.lower()}_selection_rate"
                        else:
                            key = f"changed-not-failed-no-rule-{commit}-{model}-{field_name}_best_selection_rate"
                    else:
                        if model in other_metrics:
                            key = f"changed-failed-no-rule-{commit}-Fail-Code-{model.lower()}_selection_rate"
                        else:
                            key = f"changed-failed-no-rule-{commit}-{model}-{field_name}_best_selection_rate"

                    if float(model_results_latex.get(key).value) == min_value:
                        file.append(r" & \textbf{" + latex.Macro(key).use() + "}")
                    elif float(model_results_latex.get(key).value) == max_value:
                        file.append(r" & \cellcolor{black!20}" + latex.Macro(key).use() + "")
                    else:
                        file.append(" & " + latex.Macro(key).use())

                file.append(r"\\")
                # print(index, len(first_time_failed_tests))
                if index != len(first_time_failed_tests) - 1:
                    file.append(r"\hline")
            file.append(r"\bottomrule")
            file.append(r"\end{tabular}")
            # file.append(r"\end{scriptsize}")
            file.append(r"\end{table*}")
        file.save()

    def make_numbers_auc_recall_selection(self, subsets, projs, data_types, first=False):
        fmt = f",.3f"
        if first:
            file = latex.File(self.tables_dir / f"numbers-first-failure-auc-recall-selection.tex")
        else:
            file = latex.File(self.tables_dir / f"numbers-auc-recall-selection.tex")
        for subset in subsets:
            for proj in projs:
                proj = proj.split("_")[-1]
                for data_type in data_types:
                    if first:
                        x_data = IOUtils.load(
                            Macros.data_dir / "plot-data" / f"{proj}_rankModel_select_first_failure_subset_{subset}_{data_type}_x.json")
                        y_data = IOUtils.load(
                            Macros.data_dir / "plot-data" / f"{proj}_rankModel_select_first_failure_subset_{subset}_{data_type}_y.json")
                    else:
                        x_data = IOUtils.load(
                            Macros.data_dir / "plot-data" / f"{proj}_rankModel_select_subset_{subset}_{data_type}_x.json")
                        y_data = IOUtils.load(
                            Macros.data_dir / "plot-data" / f"{proj}_rankModel_select_subset_{subset}_{data_type}_y.json")

                    auc_x_data = list(x_data)
                    auc_x_data.append(1)
                    auc_y_data = list(y_data)
                    auc_y_data.append(1)
                    aucs = round(metrics.auc(auc_x_data, auc_y_data), 3)
                    if first:
                        file.append_macro(
                            latex.Macro(f"recall-first-failure-selection-auc-{subset}-{proj}-{data_type}",
                                        f"{aucs:{fmt}}"))
                    else:
                        file.append_macro(
                            latex.Macro(f"recall-selection-auc-{subset}-{proj}-{data_type}", f"{aucs:{fmt}}"))
        file.save()

    def make_table_auc_recall_selection(self, subsets, projs, data_types, first=False):
        if first:
            file = latex.File(self.tables_dir / f"table-first-failure-auc-recall-selection.tex")
        else:
            file = latex.File(self.tables_dir / f"table-auc-recall-selection.tex")
        for subset in subsets:
            file.append(r"\begin{table*}")
            if subset == "All":
                if first:
                    file.append(
                        r"\caption{Area under curve for selecting the first failed test from all tests.\vspace{-2pt}}")
                else:
                    file.append(r"\caption{Area under curve for selecting from all tests.\vspace{-2pt}}")
            elif subset == "Ekstazi":
                if first:
                    file.append(
                        r"\caption{Area under curve for selecting the first failed test from Ekstazi selected tests.\vspace{-2pt}}")
                else:
                    file.append(
                        r"\caption{Area under curve for selecting from Ekstazi selected tests.\vspace{-2pt}}")
            elif subset == "STARTS":
                if first:
                    file.append(
                        r"\caption{Area under curve for selecting the first failed test from all tests from STARTS selected tests.\vspace{-2pt}}")
                else:
                    file.append(
                        r"\caption{Area under curve for selecting from all tests from STARTS selected tests.\vspace{-2pt}}")
            file.append(r"\footnotesize")
            file.append(r"\centering")
            file.append(r"\begin{tabular}{l|ccccccccccc}")
            file.append(r"\hline")

            file.append(r"\textbf{Projects}")
            for m in data_types:
                file.append(r"& \textbf{" + latex.Macro(m).use() + "}")
            file.append(r"\\")
            file.append(r"\hline")

            for proj in projs:
                proj = proj.split("_")[-1]
                file.append(proj)

                min_value = 1.1
                max_value = 0
                if first:
                    model_results_latex = latex.Macro.load_from_file(
                        self.tables_dir / f"numbers-first-failure-auc-recall-selection.tex")
                else:
                    model_results_latex = latex.Macro.load_from_file(
                        self.tables_dir / f"numbers-auc-recall-selection.tex")

                for data_type in data_types:
                    if first:
                        key = f"recall-first-failure-selection-auc-{subset}-{proj}-{data_type}"
                    else:
                        key = f"recall-selection-auc-{subset}-{proj}-{data_type}"
                    if float(model_results_latex.get(key).value) < min_value:
                        min_value = float(model_results_latex.get(key).value)
                    if float(model_results_latex.get(key).value) > max_value:
                        max_value = float(model_results_latex.get(key).value)

                for data_type in data_types:
                    if first:
                        key = f"recall-first-failure-selection-auc-{subset}-{proj}-{data_type}"
                    else:
                        key = f"recall-selection-auc-{subset}-{proj}-{data_type}"
                    if float(model_results_latex.get(key).value) == min_value:
                        file.append(r" & \textbf{" + latex.Macro(key).use() + "}")
                    elif float(model_results_latex.get(key).value) == max_value:
                        file.append(r" & \cellcolor{black!20}" + latex.Macro(key).use() + "")
                    else:
                        file.append(" & " + latex.Macro(key).use())
                file.append(r"\\")
                file.append(r"\hline")
            file.append(r"\bottomrule")
            file.append(r"\end{tabular}")
            file.append(r"\end{table*}")
        file.save()

    def make_numbers_bm25_perfect(self):
        file = latex.File(self.tables_dir / f"numbers-bm25-perfect.tex")
        fmt = f",.2f"
        bm25_perfect_list = IOUtils.load(f"{Macros.eval_data_dir}/mutated-eval-data/bm25_perfect.json")
        for bm25_perfect_item in bm25_perfect_list:
            project = bm25_perfect_item["project"]
            for key, value in bm25_perfect_item.items():
                if key == "project":
                    continue
                file.append_macro(latex.Macro(f"{project}-{key}", f"{value:{fmt}}"))
        file.save()

    def make_table_bm25_perfect(self, projects: List):
        file = latex.File(self.tables_dir / f"table-bm25-perfect.tex")
        file.append(r"\begin{table*}")
        file.append(r"\caption{BM25 perfect statistics.\vspace{-5pt}}")
        file.append(r"\centering")
        file.append(r"\begin{tabular}{l|ccc|cc}")
        file.append(r"\hline")
        file.append(r"\textbf{project}")
        file.append(r"& \textbf{\# bm25 perfect}")
        file.append(r"& \bfseries\makecell[ct]{\# bm25 perfect \\ but others not}")
        file.append(r"& \textbf{\# total}")
        file.append(r"& \textbf{bm25 perfect rate}")
        file.append(r"& \bfseries\makecell[ct]{bm25 perfect but \\ others not rate}")
        file.append(r"\\")
        file.append(r"\hline")

        # Start to fill in the numbers
        for project in projects:
            project_name = project.split('_')[1]
            file.append(project_name)
            file.append(f"& " + latex.Macro(f"{project}-bm25_perfect").use())
            file.append(f"& " + latex.Macro(f"{project}-bm25_perfect_but_others_not").use())
            file.append(f"& " + latex.Macro(f"{project}-total").use())
            file.append(f"& " + latex.Macro(f"{project}-bm25_perfect_rate").use())
            file.append(f"& " + latex.Macro(f"{project}-bm25_perfect_but_others_not_rate").use())
            file.append(r"\\")
        # end for
        file.append(r"\bottomrule")
        file.append(r"\end{tabular}")
        file.append(r"\end{table*}")
        file.save()

def collect_best_worst_num(model_2_best_select_rate, models_2_best_num, models_2_worst_num):
    """Collect numbers of each model to be the best and worst"""
    select_rates = [select_rate for select_rate in list(model_2_best_select_rate.values()) if select_rate >= 0]
    best_select_rate = min(select_rates)
    worst_select_rate = max(select_rates)
    for model, select_rate in model_2_best_select_rate.items():
        if select_rate < 0:
            continue
        if select_rate == best_select_rate:
            models_2_best_num[model] += 1
        if select_rate == worst_select_rate:
            models_2_worst_num[model] += 1
        # end if
    # end for
    return models_2_best_num, models_2_worst_num

    def make_numbers_raw_dataset_metrics(self):
        file = latex.File(self.tables_dir / f"numbers-results-raw-dataset-metrics.tex")

        dataset_metrics = IOUtils.load(Macros.results_dir / "metrics" / "intermediate-raw-eval-dataset.json",
                                       IOUtils.Format.json)
        for proj_stats in dataset_metrics:
            project = proj_stats["project-name"]
            proj_raw_eval_data = IOUtils.load(Macros.eval_data_dir / "raw-eval-data" / f"{project}.json",
                                           IOUtils.Format.json)
            sum_changed_files = 0
            sum_java_files = 0
            sum_add_code_java_files = 0
            sum_added_lines = 0
            sum_deleted_lines = 0
            sum_ekstazi_selected_tests = 0
            sum_starts_selected_tests = 0
            sum_ekstazi_selection_time = 0
            sum_starts_selection_time = 0
            min_tests = 10000
            max_tests = 0
            sum_tests = 0
            shalist = []
            for proj_raw_eval_data_item in proj_raw_eval_data:
                shalist.append((proj_raw_eval_data_item["prev_commit"], proj_raw_eval_data_item["commit"]))
                sum_changed_files += proj_raw_eval_data_item["changed_files_num"]
                sum_java_files += proj_raw_eval_data_item["changed_java_files_num"]
                sum_ekstazi_selected_tests += len(proj_raw_eval_data_item["ekstazi_test_list"])
                sum_starts_selected_tests += len(proj_raw_eval_data_item["starts_test_list"])
                sum_ekstazi_selection_time += proj_raw_eval_data_item["ekstazi_select_time"]
                sum_starts_selection_time += proj_raw_eval_data_item["starts_select_time"]
                sum_tests += len(proj_raw_eval_data_item["passed_test_list"])
                min_tests = min(min_tests, len(proj_raw_eval_data_item["passed_test_list"]))
                max_tests = max(max_tests, len(proj_raw_eval_data_item["passed_test_list"]))
                # cur_sha_line_code_change = 0
                for file_name, line_list in proj_raw_eval_data_item["diff_line_number_list_per_file"].items():
                    sum_add_code_java_files += 1
                    # cur_sha_line_code_change += len(line_list)
                    sum_added_lines += len(line_list)
                # print(f'{project}, ({proj_raw_eval_data_item["prev_commit"]}, {proj_raw_eval_data_item["commit"]}), changes {cur_sha_line_code_change}')
                # print(f'{project}, ({proj_raw_eval_data_item["prev_commit"]}, {proj_raw_eval_data_item["commit"]}), changes {len(proj_raw_eval_data_item["diff_code"].split())}')
                for file_name, line_list in proj_raw_eval_data_item["deleted_line_number_list_per_file"].items():
                    sum_deleted_lines += len(line_list)
            num_shas = len(proj_raw_eval_data)
            file.append_macro(latex.Macro(f"{project}-raw-eval-data-avg-changed-files", int(sum_changed_files / num_shas)))
            file.append_macro(latex.Macro(f"{project}-raw-eval-data-avg-changed-java-files", int(sum_java_files / num_shas)))
            file.append_macro(latex.Macro(f"{project}-raw-eval-data-avg-add-code-java-files", int(sum_add_code_java_files / num_shas)))
            file.append_macro(latex.Macro(f"{project}-raw-eval-data-avg-add-lines", int(sum_added_lines / num_shas)))
            file.append_macro(latex.Macro(f"{project}-raw-eval-data-avg-delete-lines", int(sum_deleted_lines / num_shas)))
            file.append_macro(latex.Macro(f"{project}-raw-eval-data-avg-ekstazi-selected-tests", int(sum_ekstazi_selected_tests / num_shas)))
            file.append_macro(latex.Macro(f"{project}-raw-eval-data-avg-starts-selected-tests", int(sum_starts_selected_tests / num_shas)))
            file.append_macro(latex.Macro(f"{project}-raw-eval-data-avg-ekstazi-selection-time", '{0:.3g}'.format(sum_ekstazi_selection_time / num_shas)))
            file.append_macro(latex.Macro(f"{project}-raw-eval-data-avg-starts-selection-time", '{0:.3g}'.format(sum_starts_selection_time / num_shas)))
            file.append_macro(latex.Macro(f"{project}-raw-eval-data-min-tests", min_tests))
            file.append_macro(latex.Macro(f"{project}-raw-eval-data-max-tests", max_tests))
            file.append_macro(latex.Macro(f"{project}-raw-eval-data-avg-tests", int(sum_tests / num_shas)))
            # end for
        # end for
        file.save()

    def make_numbers_intermediate_raw_dataset_metrics(self):
        file = latex.File(self.tables_dir / f"numbers-intermediate-raw-dataset-metrics.tex")
        dataset_metrics = IOUtils.load(Macros.results_dir / "metrics" / "intermediate-raw-eval-dataset.json",
                                       IOUtils.Format.json)
        for proj_stats in dataset_metrics:
            name = proj_stats["project-name"].split("_")[-1]
            for k, v in proj_stats.items():
                if k != "project-name":
                    if type(v) == float:
                        fmt = f",.2f"
                    elif type(v) == int:
                        fmt = f",d"
                    else:
                        fmt = ""
                    file.append_macro(latex.Macro(f"{name}-raw-data-{k}", f"{v:{fmt}}"))
                # end if
            # end for
        # end for
        file.save()


    def make_table_raw_dataset_metrics(self):
        file = latex.File(self.tables_dir / "table-results-raw-dataset-metrics.tex")
        dataset_metrics = IOUtils.load(Macros.results_dir / "metrics" / "intermediate-raw-eval-dataset.json",
                                       IOUtils.Format.json)
        # Header
        file.append(r"\begin{table*}")
        file.append(r"\caption{Stats of Raw Eval Data.}")
        file.append(r"\vspace{-5pt}")
        file.append(r"\centering")

        file.append(r"\begin{tabular}{l|p{1cm}|p{1cm}|p{1.2cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1.1cm}|p{1.1cm}}")
        file.append(r"\hline")

        file.append(r"\textbf{Projects} & "
                    r"\textbf{\# avg changed files} & "
                    r"\textbf{\# avg changed java files} & "
                    r"\textbf{\# avg add code java files} &"
                    r"\textbf{\# avg added lines} &"
                    r"\textbf{\# avg deleted lines} &"
                    r"\textbf{min \# tests} &"
                    r"\textbf{max \# tests} &"
                    r"\textbf{avg \# tests} &"
                    r"\textbf{\# avg Ekstazi selected tests} &"
                    r"\textbf{\# avg STARTS selected tests} &"
                    r"\textbf{avg Ekstazi selection time(s)} &"
                    r"\textbf{avg STARTS selection time(s)} \\")

        file.append(r"\hline")

        for proj_stats in dataset_metrics:
            project = proj_stats["project-name"]
            proj = proj_stats["project-name"].split("_")[-1]
            file.append(r"\UseMacro{" + proj + "}")
            for m in ["avg-changed-files", "avg-changed-java-files", "avg-add-code-java-files", "avg-add-lines",
                      "avg-delete-lines", "min-tests", "max-tests", "avg-tests", "avg-ekstazi-selected-tests",
                      "avg-starts-selected-tests", "avg-ekstazi-selection-time", "avg-starts-selection-time"]:
                key = f"{project}-raw-eval-data-{m}"
                file.append(" & " + latex.Macro(key).use())
                # end if
            # end for
            file.append(r"\\")
            # end for
            # Footer
        file.append(r"\hline")
        file.append(r"\end{tabular}")
        file.append(r"{\raggedright There are 50 SHAs for each project \par}")
        file.append(r"{\raggedright \# avg changed files: total number of changed files over 50 SHAs / 50 \par}")
        file.append(r"{\raggedright \# avg changed java files: total number of changed java files over 50 SHAs / 50 \par}")
        file.append(r"{\raggedright \# avg add code java files: total number of java files that add code over 50 SHAs / 50 "
                    r"(exclude those java files that only deleting code or changing comments) \par}")
        file.append(r"{\raggedright \# avg added lines: total number of lines of added code over 50 SHAs / 50 \par}")
        file.append(r"{\raggedright \# avg deleted lines: total number of lines of deleted code over 50 SHAs / 50 \par}")
        file.append(r"{\raggedright min \# tests: for the 50 SHAs, min(number of test classes per SHA) \par}")
        file.append(r"{\raggedright max \# tests: for the 50 SHAs, max(number of test classes per SHA) \par}")
        file.append(r"{\raggedright avg \# tests: total number of test classes over 50 SHAs / 50  \par}")
        file.append(r"{\raggedright \# avg Ekstazi selected tests: total number of tests selected by Ekstazi over 50 SHAs / 50 \par}")
        file.append(r"{\raggedright Bukkit, \# avg Ekstazi selected tests is 0: no tests depend on the changed code and no tests are selected \par}")
        file.append(r"{\raggedright \# avg STARTS selected tests: total number of tests selected by STARTS over 50 SHAs / 50 \par}")
        file.append(r"\end{table*}")
        file.save()

    def make_table_intermediate_raw_dataset_metrics(self):
        file = latex.File(self.tables_dir / "table-intermediate-raw-dataset-metrics.tex")
        # Header
        file.append(r"\begin{table*}")
        file.append(r"\caption{Stats of Filtering Raw Eval Data.}")
        file.append(r"\vspace{-5pt}")
        file.append(r"\centering")

        file.append(r"\begin{tabular}{l|p{1.8cm}|p{1.8cm}|p{1.8cm}|p{1.8cm}|p{1.8cm}|p{2cm}}")
        file.append(r"\hline")

        file.append(r"\textbf{Projects} & "
                    r"\textbf{\# SHAs checked} & "
                    r"\textbf{\# SHAs collected} & "
                    r"\textbf{\# SHAs with compile failure} &"
                    r"\textbf{\# SHAs with no bytecode change} &"
                    r"\textbf{\# SHAs just delete lines} &"
                    r"\textbf{\# SHAs change over 1000 lines java code} \\")
        file.append(r"\hline")

        dataset_metrics = IOUtils.load(Macros.results_dir / "metrics" / "intermediate-raw-eval-dataset.json",
                                       IOUtils.Format.json)
        for proj_stats in dataset_metrics:
            proj = proj_stats["project-name"].split("_")[-1]
            file.append(r"\UseMacro{" + proj + "}")
            for m in ["num-sha-checked", "num-sha-result", "num-sha-compile-failure", "num-sha-no-bytecode-change",
                      "num-sha-only-delete-lines", "num-sha-1000-lines"]:
                key = f"{proj}-raw-data-{m}"
                file.append(" & " + latex.Macro(key).use())
                # end if
            # end for
            file.append(r"\\")
            # end for
            # Footer
        file.append(r"\hline")
        file.append(r"\end{tabular}")
        file.append(r"{\raggedright \par}")
        file.append(r"{\raggedright For each row, 1st column = the sum of next four columns(2nd, 3rd, 4th, 5th) \par}")
        file.append(r"{\raggedright Selection rule: start from the training SHA, compare every adjacent SHAs, skip when "
                    r"code cannot compile/no bytecode change/only deleting java code \par}")
        file.append(r"{\raggedright Did not skip SHAs that change java code (exclude comments) over 1000 lines this time \par}")
        file.append(r"{\raggedright apache/commons-net, (efedd6dc 7fabd004), 'rename package', changes 3749 lines \par}")
        file.append(r"{\raggedright apache/commons-net, (1f1cb3cb 75203979), 'use final', changes 2944 lines \par}")
        file.append(r"{\raggedright apache/commons-net, (0da76503 10b792c0), 'rename', changes 1166 lines \par}")
        file.append(r"{\raggedright apache/commons-csv, (ed6adc70 9daee904), 'add try statement', changes 1093 lines \par}")
        file.append(r"{\raggedright apache/commons-configuration, (d7280877 fa5dbfaf), 'use final', changes 8456 lines \par}")
        file.append(r"{\raggedright frizbog/gedcom4j, (0d236c5a f701e8c0), 'add multiple new tests', changes 1265 lines \par}")
        file.append(r"{\raggedright frizbog/gedcom4j, (c1542505, c5875df8), 'merge branch', changes 16657 lines \par}")
        file.append(r"{\raggedright commons-lang changed the training SHA from 4b718f6e to ba8c6f6d, because the old training SHA has test failure,"
                    r"also, the 50 consecutive SHA mutant pair cannot be mutated successfully, and 15 SHAs have test failure\par}")
        file.append(r"\end{table*}")
        file.save()
