from pathlib import Path
from statistics import mean
from typing import *
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from seutil import IOUtils, LoggingUtils, MiscUtils, latex
from collections import defaultdict, OrderedDict

from pts.Environment import Environment
from pts.Macros import Macros
from pts.Utils import Utils
from sklearn import metrics

import matplotlib.cm as mplcm
import matplotlib.colors as colors

class Plot:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    MARKERS = [
        ".", "^", "o", "*", "s", "x", "1", ">", "h", "D", "d", "H", "1", "2"
    ]
    COLORS = ['b', 'r', 'g']
    output_dir = Macros.plot_dir

    def make_plots(self, **options):
        which = Utils.get_option_as_list(options, "which")
        for item in which:
            if item == "rank-model-plot-data":
                # Make recall vs selection rate for selecting all tests
                project = options.get("project")
                subset = options.get("subset", None)
                data_type = Utils.get_option_as_list(options, "data_type")
                output_dir = Macros.data_dir / "plot-data"
                self.get_subset_plot_data(project, data_type, output_dir, subset="All")
            elif item == "subset-recall-selection-data":
                project = options.get("project")
                subset = options.get("subset", None)
                data_type = Utils.get_option_as_list(options, "data_type")
                output_dir = Macros.data_dir / "plot-data"
                self.get_subset_plot_data(project, data_type, output_dir, subset)
            elif item == "first-failure-recall-selection-data":
                project = options.get("project")
                subset = options.get("subset", None)
                data_type = Utils.get_option_as_list(options, "data_type")
                output_dir = Macros.data_dir / "plot-data"
                self.get_subset_first_failure_plot_data(project, data_type, output_dir, subset)
            elif item == "perfect-subset-recall-selection-data":
                project = options.get("project")
                subset = options.get("subset", None)
                output_dir = Macros.data_dir / "plot-data"
                self.get_perfect_subset_plot_data(project, output_dir, subset)
            elif item == "plot-first-failure-recall-selection":
                project = options.get("project")
                subset = options.get("subset", None)
                data_type = Utils.get_option_as_list(options, "data_type")
                output_dir = Macros.results_dir / "plots"
                self.plot_first_failure_recall_selection_rate(project, data_type, output_dir, subset)
            elif item == "plot-subset-recall-selection":
                project = options.get("project")
                subset = options.get("subset", None)
                data_type = Utils.get_option_as_list(options, "data_type")
                output_dir = Macros.results_dir / "plots"
                self.plot_recall_subset_selection_rate(project, data_type, output_dir, subset)
            elif item == "plot-recall-selection-rate":
                project = options.get("project")
                subset = options.get("subset", None)
                data_type = Utils.get_option_as_list(options, "data_type")
                output_dir = Macros.results_dir / "plots"
                self.plot_recall_subset_selection_rate(project, data_type, output_dir, subset="All")
            elif item == "tools-perfect-data":
                project = options.get("project")
                tool = options.get("tool")
                output_dir = Macros.data_dir / "plot-data"
                self.plot_tool_perfect(project, tool, output_dir)
            elif item == "rank-models-barplot":
                models = Utils.get_option_as_list(options, "data_type")
                proj = options.get("project").split('_')[1]
                self.rank_models_bar_plot(proj, models)
            elif item == "rank-model-test-rank-barplot":
                models = Utils.get_option_as_list(options, "data_type")
                proj = options.get("project").split('_')[1]
                self.rank_models_best_rank_bar_plot(proj, models)
            elif item == "best-select-rate-boxplot":
                models = Utils.get_option_as_list(options, "data_type")
                proj = options.get("project")
                self.best_select_rate_box_plot(proj, models)
            elif item == "best-select-rate-boxplot-for-types":
                models = Utils.get_option_as_list(options, "data_type")
                proj = options.get("project")
                types = Utils.get_option_as_list(options, "types")
                for type in types:
                    self.best_select_rate_box_plot(proj, models, type)
            elif item == "num_of_test_cases_raw_eval_data_lineplot":
                projs = options.get("projects")
                self.num_of_test_cases_raw_eval_data_lineplot(projs)
            elif item == "num_of_test_cases_mutated_eval_data_lineplot":
                proj = options.get("project")
                self.num_of_test_cases_mutated_eval_data_lineplot(proj)
            elif item == "ROC-curve-data":
                project = options.get("project")
                data_type = Utils.get_option_as_list(options, "data_type")
                output_dir = Macros.data_dir / "plot-data"
                self.get_ROC_curve_plot_data(project, data_type, output_dir)
            elif item == "plot-roc-curve":
                project = options.get("project")
                data_type = Utils.get_option_as_list(options, "data_type")
                output_dir = Macros.results_dir / "plots"
                self.plot_ROC_curce(project, data_type, output_dir)
            elif item == "pr-curve-data":
                project = options.get("project")
                data_type = Utils.get_option_as_list(options, "data_type")
                output_dir = Macros.data_dir / "plot-data"
                self.get_precision_recall_curve_data(project, data_type, output_dir)
            elif item == "plot-pr-curve":
                project = options.get("project")
                data_type = Utils.get_option_as_list(options, "data_type")
                output_dir = Macros.results_dir / "plots"
                self.plot_precision_recall_curve(project, data_type, output_dir)
            elif item == "recall-vs-selection-boxplot-plots-layout":
                projects = options.get("projects").split()
                self.recall_vs_selection_boxplot_plots_layout(projects)
            elif item == "number-of-changed-files-vs-select-rate-plots-layout":
                projects = options.get("projects").split()
                self.number_of_changed_files_vs_select_rate_plots_layout(projects)
            elif item == "num-changed-files-vs-avg-selection-rate-barplot":
                projects = options.get("projects").split()
                models = options.get("models").split()
                models.append("perfect")
                type = options.get("type")
                self.num_changed_files_vs_avg_selection_rate_barplot(projects, models, type)
            elif item == "scatter-plot-changed-files-selection-rate":
                projects = options.get("projects").split()
                self.best_select_rate_changed_files_scatter_plot(projects)
            elif item == "boxplot-end-to-end-time":
                projects = options.get("projects").split()
                subset = options.get("subset")
                models = options.get("models").split()
                self.boxplot_end_to_end_time(projects, subset, models)
            elif item == "boxplot-selection-time":
                projects = options.get("projects").split()
                subset = options.get("subset")
                models = options.get("models").split()
                self.boxplot_selection_time(projects, subset, models)
            else:
                raise NotImplementedError

    def best_select_rate_changed_files_scatter_plot(self, projects: List[str]):
        """Make a scatter plot to show # changed files v.s. best safe selection rate"""
        # legends = ["BM25Baseline", "Fail-Code"]
        legends = ["BM25Baseline"]
        for i, dt in enumerate(legends):
            x_val = []
            y_val = []
            for project in projects:
                mutated_eval_data = IOUtils.load(Macros.eval_data_dir / "mutated-eval-data" / f"{project}-ag.json")
                per_sha_preds = IOUtils.load(
                Macros.results_dir / "modelResults" / project.split('_')[1] / dt / "best-select-rate-per-SHA.json")
                for eval_data, commit in zip(mutated_eval_data, per_sha_preds):
                    # first check whether they are the same sha
                    assert eval_data["commit"] == commit
                    x_val.append(eval_data["changed_files_num"])
                    y_val.append(per_sha_preds[commit])
                # end for
            # end for
            plt.scatter(x_val, y_val, c=self.COLORS[i])
        # end for
        plt.legend(legends)
        plt.ylabel("Safe selection rate")
        plt.xlabel("Number of changed files")
        plt.title(f"Safe selection rate and number of changed files.")
        plt.savefig(self.output_dir / f"fig-changed-files-safe-select-rate.jpg")
        # end for


    def plot_tool_perfect(self, project: str, tool_type: str, output_dir: Path):
        from pts.models.rank_model.MetricsCollect import MetricsCollect
        mc = MetricsCollect()
        proj = project.split("_")[1]
        x_values = np.arange(0.0, 1.04, 0.02)
        y_values = mc.collect_metrics([f"{tool_type}-perfect"], project, "", 1)
        # end for
        IOUtils.dump(output_dir / f"{proj}_{tool_type}_recall_selection_x.json", x_values.tolist())
        IOUtils.dump(output_dir / f"{proj}_{tool_type}_recall_selection_y.json", y_values)

    def plot_first_failure_recall_selection_rate(self, project: str, data_type: List[str], output_dir: Path, subset: str):
        """ Plot recall vs selection rate for different models for selecting the first failed tests form all tests."""
        proj = project.split('_')[1]
        legends = []
        aucs = {}
        max_x = 0
        for i, dt in enumerate(data_type):
            x_data = IOUtils.load(
            Macros.data_dir / "plot-data" / f"{proj}_rankModel_select_first_failure_subset_{subset}_{dt}_x.json")
            y_data = IOUtils.load(
            Macros.data_dir / "plot-data" / f"{proj}_rankModel_select_first_failure_subset_{subset}_{dt}_y.json")
            legends.append(dt)
            plt.plot([0] + x_data, [0] + y_data, marker=self.MARKERS[i])
            max_x = max(x_data) if max(x_data) > max_x else max_x

            auc_x_data = list(x_data)
            auc_x_data.append(1)
            auc_y_data = list(y_data)
            auc_y_data.append(1)
            aucs[dt] = round(metrics.auc(auc_x_data, auc_y_data), 2)

        plt.legend(legends, loc='lower right')
        plt.ylabel("Test Recall")

        # show auc in x_label, add a new line every four models
        auc_label = ""
        auc_index = 0
        for dt, auc_value in aucs.items():
            auc_label += dt + ":" + str(auc_value) + " "
            auc_index += 1
            if auc_index % 3 == 0:
                auc_label += "\n"

        plt.xlabel("Selection Rate\n" + auc_label)
        if subset != "All":
            x_max_limit = max_x
        else:
            x_max_limit = 1
        plt.axis([0, x_max_limit, 0, 1])
        # plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', axis='x', color='#999999', linestyle='--', alpha=0.2)

        # this added code snippet is used to add title in plots
        plt.title(f"First failure Recall vs Selection rate for {proj}, {subset}.")
        plt.tight_layout()
        plt.savefig(output_dir / f"fig-{proj}-rank-models-select-first-failure-subset-{subset}-results.eps")

    def get_perfect_subset_plot_data(self, project: str, output_dir: Path, subset: str):
        """
        Get the recall v.s. selection rate plot data for the Ekstazi subset and STARTS subset for the perfect model
        Note that, here selection_rate = (# selected test) / (# tests selected by tools)
        recall = (# failed tests) / (# failed test selected by tools)
        Note: the data required to make the plot should exist in results/modelResults directory
        """
        from sklearn.metrics import recall_score
        if subset == "All":
            tool_labels = "labels"
        else:
            tool_labels = f"{subset}_labels"
        proj = project.split("_")[1]
        dt = "Fail-Code"
        per_sha_preds = IOUtils.load(
            Macros.results_dir / "modelResults" / project.split('_')[1] / dt / "per-sha-result.json")
        x_scaled_values = []
        y_values = []
        for select_rate in np.arange(0.0, 1.04, 0.02):
            starts_subset_recall_per_sha = []
            reach_one = False
            for data_item in per_sha_preds:
                num_of_all_tests = len(data_item[tool_labels])
                # Get labels and predictions
                if subset == "All":
                    starts_selected_labels = data_item["labels"]
                else:
                    # Add the results of models selection intersecting tools
                    starts_selected_labels = []
                    for i in range(len(data_item[tool_labels])):
                        if data_item[tool_labels][i] == 1:
                            starts_selected_labels.append(data_item["labels"][i])
                    # end for
                # end if

                positive_index_list = [i for i, x in enumerate(starts_selected_labels) if x>0]

                select_size = min(int(select_rate * num_of_all_tests), len(starts_selected_labels))

                starts_subset_preds = np.zeros(len(starts_selected_labels))

                if select_size > 0:
                    # if select_size is 0 then select nothing
                    # actual_select_size = min(select_size, sum(starts_selected_labels))
                    actual_select_size = select_size
                    for index in positive_index_list[: actual_select_size]:
                        starts_subset_preds[index] = 1.0
                # end if
                # Calculate recall
                if sum(starts_selected_labels) == 0:
                    pass
                else:
                    recall_this_sha = recall_score(starts_selected_labels, starts_subset_preds)
                    starts_subset_recall_per_sha.append(recall_this_sha)
            # Get the avg. recall for the project in this select rate
            recall = sum(starts_subset_recall_per_sha) / len(starts_subset_recall_per_sha)
            if recall == 1.0 and reach_one is False:
                best_select_rate = select_rate
                reach_one = True
            # end if
            y_values.append(recall)
            x_scaled_values.append(select_rate)
            if reach_one is True:
                break

        IOUtils.dump(output_dir / f"{proj}_rankModel_select_subset_{subset}_perfect_x.json", x_scaled_values)
        IOUtils.dump(output_dir / f"{proj}_rankModel_select_subset_{subset}_perfect_y.json", y_values)
    # end for

    def get_subset_first_failure_plot_data(self, project: str, data_type: List[str], output_dir: Path, subset: str):
        """
        Get the recall (over the first failure test across all sha:mutant) v.s. selection rate plot data for All, Ekstazi,
        and SATRST subset
        :param project:
        :return:
        """
        from sklearn.metrics import recall_score
        if subset == "All":
            tool_labels = "labels"
        else:
            tool_labels = f"{subset}_labels"
        proj = project.split("_")[1]
        for dt in data_type:
            per_sha_preds = IOUtils.load(
                Macros.results_dir / "modelResults" / project.split('_')[1] / dt / "per-sha-result.json")
            x_scaled_values = []
            y_values = []
            best_select_rate = 0
            for select_rate in np.arange(0.0, 1.04, 0.02):
                starts_subset_recall_per_sha = []
                reach_one = False
                for data_item in per_sha_preds:
                    num_of_all_tests = len(data_item[tool_labels])
                    # Get labels and predictions
                    if subset == "All":
                        model_starts_preds = data_item["prediction_scores"]
                        starts_selected_labels = data_item["labels"]
                    else:
                        # Add the results of models selection intersecting tools
                        model_starts_preds = []
                        starts_selected_labels = []
                        for i in range(len(data_item[tool_labels])):
                            if data_item[tool_labels][i] == 1:
                                starts_selected_labels.append(data_item["labels"][i])
                                model_starts_preds.append(data_item["prediction_scores"][i])
                        # end for
                    # end if
                    select_size = min(int(select_rate * num_of_all_tests), len(model_starts_preds))
                    model_starts_preds_with_index = [(x, i) for i, x in enumerate(model_starts_preds)]
                    sorted_model_starts_preds = sorted(model_starts_preds_with_index, key=lambda x: (x[0], x[1]))
                    sorted_preds_index = [x[1] for x in sorted_model_starts_preds]
                    # model_starts_preds_sort_index = model_starts_preds.argsort()

                    starts_subset_preds = np.zeros(len(starts_selected_labels))
                    if select_size > 0:
                        # if select_size is 0 then select nothing
                        starts_subset_preds[sorted_preds_index[-select_size:]] = 1
                    # end if
                    # Calculate recall
                    if sum(starts_selected_labels) == 0:
                        pass
                    else:
                        recall_this_sha = recall_score(starts_selected_labels, starts_subset_preds)
                        if recall_this_sha > 0:
                            starts_subset_recall_per_sha.append(1.0)
                        else:
                            starts_subset_recall_per_sha.append(0)
                # Get the avg. recall for the project in this select rate
                recall = sum(starts_subset_recall_per_sha) / len(starts_subset_recall_per_sha)
                if recall == 1.0 and reach_one is False:
                    best_select_rate = select_rate
                    reach_one = True
                # end if
                y_values.append(recall)
                x_scaled_values.append(select_rate)
                if reach_one is True:
                    break

            IOUtils.dump(output_dir / f"{proj}_rankModel_select_first_failure_subset_{subset}_{dt}_x.json", x_scaled_values)
            IOUtils.dump(output_dir / f"{proj}_rankModel_select_first_failure_subset_{subset}_{dt}_y.json", y_values)
        # end for


    def get_subset_plot_data(self, project: str, data_type: List[str], output_dir: Path, subset: str):
        """
        Get the recall v.s. selection rate plot data for the Ekstazi subset and STARTS subset.
        Note that, here selection_rate = (# selected test) / (# tests selected by tools)
        recall = (# failed tests) / (# failed test selected by tools)
        Note: the data required to make the plot should exist in results/modelResults directory
        """
        from sklearn.metrics import recall_score
        if subset == "All":
            tool_labels = "labels"
        else:
            tool_labels = f"{subset}_labels"
        proj = project.split("_")[1]
        for dt in data_type:
            per_sha_preds = IOUtils.load(
                Macros.results_dir / "modelResults" / project.split('_')[1] / dt / "per-sha-result.json")
            x_scaled_values = []
            y_values = []
            best_select_rate = 0
            for select_rate in np.arange(0.0, 1.04, 0.02):
                starts_subset_recall_per_sha = []
                reach_one = False
                for data_item in per_sha_preds:
                    num_of_all_tests = len(data_item[tool_labels])
                    # Get labels and predictions
                    if subset == "All":
                        num_of_candidate_tests = len(data_item[tool_labels])
                        model_starts_preds = data_item["prediction_scores"]
                        starts_selected_labels = data_item["labels"]
                    else:
                        # Add the results of models selection intersecting tools
                        num_of_candidate_tests = sum(data_item[tool_labels])
                        model_starts_preds = []
                        starts_selected_labels = []
                        for i in range(len(data_item[tool_labels])):
                            if data_item[tool_labels][i] == 1:
                                starts_selected_labels.append(data_item["labels"][i])
                                model_starts_preds.append(data_item["prediction_scores"][i])
                        # end for
                    # end if
                    select_size = min(int(select_rate * num_of_all_tests), len(model_starts_preds))
                    model_starts_preds_with_index = [(x, i) for i, x in enumerate(model_starts_preds)]
                    sorted_model_starts_preds = sorted(model_starts_preds_with_index, key=lambda x: (x[0], x[1]))
                    sorted_preds_index = [x[1] for x in sorted_model_starts_preds]
                    # model_starts_preds_sort_index = model_starts_preds.argsort()

                    starts_subset_preds = np.zeros(len(starts_selected_labels))
                    if select_size > 0:
                        # if select_size is 0 then select nothing
                        starts_subset_preds[sorted_preds_index[-select_size:]] = 1
                    # end if
                    # Calculate recall
                    if sum(starts_selected_labels) == 0:
                        pass
                    else:
                        recall_this_sha = recall_score(starts_selected_labels, starts_subset_preds)

                        starts_subset_recall_per_sha.append(recall_this_sha)
                # Get the avg. recall for the project in this select rate
                recall = sum(starts_subset_recall_per_sha) / len(starts_subset_recall_per_sha)
                if recall == 1.0 and reach_one is False:
                    best_select_rate = select_rate
                    reach_one = True
                # end if
                y_values.append(recall)
                x_scaled_values.append(select_rate)
                if reach_one is True:
                    break

            IOUtils.dump(output_dir / f"{proj}_rankModel_select_subset_{subset}_{dt}_x.json", x_scaled_values)
            IOUtils.dump(output_dir / f"{proj}_rankModel_select_subset_{subset}_{dt}_y.json", y_values)
        # end for

    def get_precision_recall_curve_data(self, project: str, data_type: List[str], output_dir: Path):
        """Get precision recall curve data for the projects and models."""
        from sklearn.metrics import precision_recall_curve
        proj = project.split('_')[1]
        # roc_auc_score_models = {}
        for dt in data_type:
            per_sha_preds = IOUtils.load(
                Macros.model_data_dir / "rank-model" / project.split('_')[1] / dt / "results" /
                "per-sha-result.json")
            labels_all_sha = []
            prediction_scores_all_sha = []
            for data_item in per_sha_preds:
                labels_all_sha.extend(data_item["labels"])
                prediction_scores_all_sha.extend(data_item["prediction_scores"])
            # end for

            precision, recall, _ = precision_recall_curve(labels_all_sha, prediction_scores_all_sha, pos_label=1)

            IOUtils.dump(output_dir / f"{proj}_{dt}_precision.json", precision.tolist())
            IOUtils.dump(output_dir / f"{proj}_{dt}_recall.json", recall.tolist())
        # end for

    def plot_precision_recall_curve(self, project: str, data_type: List[str], output_dir: Path):
        """Make plot precision recall curve."""
        proj = project.split('_')[1]
        legends = []
        for i, dt in enumerate(data_type):
            pr = IOUtils.load(
                Macros.data_dir / "plot-data" / f"{proj}_{dt}_precision.json")
            rc = IOUtils.load(
                Macros.data_dir / "plot-data" / f"{proj}_{dt}_recall.json")
            legends.append(dt)
            plt.plot(rc, pr, marker=self.MARKERS[i])
        # end for
        plt.legend(legends, loc="upper right")
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.axis([0, 1, 0, 1])
        plt.minorticks_on()
        plt.grid(b=True, which='minor', axis='x', color='#999999', linestyle='--', alpha=0.2)
        plt.savefig(output_dir / f"fig-{proj}-PR-curve.eps")

    def get_ROC_curve_plot_data(self, project: str, data_type: List[str], output_dir: Path):
        """Get ROC curve data for the projects and models."""
        from sklearn.metrics import roc_curve, roc_auc_score
        proj = project.split('_')[1]
        roc_auc_score_models = {}
        for dt in data_type:
            per_sha_preds = IOUtils.load(
                Macros.model_data_dir / "rank-model" / project.split('_')[1] / dt / "results" /
                "per-sha-result.json")
            labels_all_sha = []
            prediction_scores_all_sha = []
            for data_item in per_sha_preds:
                labels_all_sha.extend(data_item["labels"])
                prediction_scores_all_sha.extend(data_item["prediction_scores"])
            # end for

            fpr, tpr, thresholds = roc_curve(labels_all_sha, prediction_scores_all_sha, pos_label=1)
            # plot the roc curve for the model

            roc_score = roc_auc_score(labels_all_sha, prediction_scores_all_sha)
            roc_auc_score_models[f"{dt}_roc_auc_score"] = roc_score
            IOUtils.dump(output_dir / f"{proj}_{dt}_ROC_false_positive_rate.json", fpr.tolist())
            IOUtils.dump(output_dir / f"{proj}_{dt}_ROC_true_positive_rate.json", tpr.tolist())
        # end for
        IOUtils.dump(Macros.metrics_dir/ f"stats-{proj}-ROC_AUC_scores.json", roc_auc_score_models)

    def plot_ROC_curce(self, project: str, data_type: List[str], output_dir: Path):
        """Plot ROC curve for all the models"""
        proj = project.split('_')[1]
        legends = []
        for i, dt in enumerate(data_type):
            fpr = IOUtils.load(
                Macros.data_dir / "plot-data" / f"{proj}_{dt}_ROC_false_positive_rate.json")
            tpr = IOUtils.load(
                Macros.data_dir / "plot-data" / f"{proj}_{dt}_ROC_true_positive_rate.json")
            legends.append(dt)
            plt.plot(fpr, tpr, marker=self.MARKERS[i])
        # end for
        plt.legend(legends, loc='lower right')
        plt.ylabel("True postive Rate")
        plt.xlabel("False Positive Rate")
        plt.axis([0, 1, 0, 1])
        plt.minorticks_on()
        plt.grid(b=True, which='minor', axis='x', color='#999999', linestyle='--', alpha=0.2)
        plt.savefig(output_dir / f"fig-{proj}-ROC-curve.eps")

    def plot_recall_subset_selection_rate(self, project: str, data_type: List[str], output_dir: Path, subset: str):
        """ Plot recall vs selection rate for different models for selecting subset of tools's selection
        This model also save auc into metric files.
        """
        proj = project.split('_')[1]
        legends = []
        aucs = {}
        max_x = 0
        for i, dt in enumerate(data_type):
            x_data = IOUtils.load(
                Macros.data_dir / "plot-data" / f"{proj}_rankModel_select_subset_{subset}_{dt}_x.json")
            y_data = IOUtils.load(
                Macros.data_dir / "plot-data" / f"{proj}_rankModel_select_subset_{subset}_{dt}_y.json")

            plt.plot([0] + x_data, [0] + y_data, marker=self.MARKERS[i])
            max_x = max(x_data) if max(x_data) > max_x else max_x

            auc_x_data = list(x_data)
            auc_x_data.append(1)
            auc_y_data = list(y_data)
            auc_y_data.append(1)
            aucs[dt] = round(metrics.auc(auc_x_data, auc_y_data), 2)
            dt_name = dt if dt != "randomforest" else "EALRTS"
            legends.append(dt_name)
        plt.legend(legends, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
        plt.ylabel("Test Recall")
        
        # show auc in x_label, add a new line every four models
        stats_auc_file = Macros.metrics_dir / f"stats-{project}-{subset}-subset-auc-score.json"
        auc_stats = {}
        auc_label = ""
        auc_index = 0
        for dt, auc_value in aucs.items():
            auc_label += dt + ":" + str(auc_value) + " "
            auc_index += 1
            if auc_index % 2 == 0:
                auc_label += "\n"
            # Save to metric file
            auc_stats[dt] = auc_value

        IOUtils.dump(stats_auc_file, auc_stats)
        
        plt.xlabel("Selection Rate")
        if subset != "All":
            x_max_limit = 1
        else:
            x_max_limit = 1
        plt.axis([0, x_max_limit, 0, 1])
        # plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', axis='x', color='#999999', linestyle='--', alpha=0.2)

        # this added code snippet is used to add title in plots
        best_safe_select_rate = 0
        try:
            if subset == "All":
                mutated_eval_data = latex.Macro.load_from_file(Macros.paper_dir / "tables" / f"numbers-perfect-best-safe-selection-rate.tex")
                best_safe_select_rate = mutated_eval_data.get(f"{project}-perfect-best-safe-selection-rate").value
                print(best_safe_select_rate)
            else:
                eval_results = IOUtils.load(Macros.metrics_dir / f"stats-{project}-eval-results.json")
                best_safe_select_rate = format(eval_results["Fail-Basic"][f"{subset}-best-safe-select-rate"], '.2f')
        except Exception as e:
            print(e)
        plt.title(f"Recall vs Selection rate for {proj} combining with {subset}")
        plt.tight_layout()
        plt.savefig(output_dir / f"fig-{proj}-rank-models-select-subset-{subset}-results.eps")

    def rank_models_bar_plot(self, proj: str, models: List[str]):
        """ Draw bar plots per sha to show the best selection rate for each SHA and model. """
        plt.cla()
        plt.figure(figsize=(14, 6))
        new_models = list()
        all_models_results = defaultdict(dict)
        for model in models:
            result_dir = Macros.model_data_dir / "rank-model" / proj / model / "results" / "best-select-rate-per-SHA.json"
            sha_results = IOUtils.load(result_dir)
            # ipdb.set_trace()
            for k, v in sha_results.items():
                if k != "model":
                    all_models_results[k][model] = v
            # end for
        # end for
        # Load results for tools
        result_dir = Macros.model_data_dir / "rank-model" / proj / model / "results" / "tools-select-rate-per-SHA.json"
        sha_results = IOUtils.load(result_dir)
        for k, res in sha_results.items():
            for t, v in res.items():
                all_models_results[k][t] = v
        # end for
        perfect_result_dir = Macros.model_data_dir / "rank-model" / proj / model / "results" / "perfect-select-rate-per-SHA.json"
        sha_results = IOUtils.load(perfect_result_dir)
        for k, res in sha_results.items():
            all_models_results[k]["PERFECT"] = res
        # end for

        models.extend(["Ekstazi", "STARTS", "PERFECT"])
        X = np.arange(len(list(all_models_results.keys())))
        tick_label = [t for t in list(all_models_results.keys())]
        bar_width = 0.1

        for index, model in enumerate(models):
            model_y = np.array([all_models_results[x][model] for x in list(all_models_results.keys())])
            for x_text, y_text in zip(X, model_y):
                plt.text(x_text + bar_width * index, y_text + 0.005, '%.1f' % y_text, ha='center', va='bottom',
                         fontsize=1)

            plt.bar(X + bar_width * index, model_y, bar_width, align="center", label=model, alpha=0.7)
        plt.xticks(fontproperties='monospace', size=5, rotation=90)
        plt.xticks(X + bar_width * len(models) / 2, tick_label)
        ax = plt.subplot(111)
        ax.legend(bbox_to_anchor=(1.12, 1.1))
        plt.xlabel("SHAs")
        plt.ylabel("Selection rate to get 100 Recall")
        plt.title("Selection rate for models across eval data")
        plt.savefig(f"{Macros.plot_dir}/figure-{proj}-rank-models-selection-rate-barplot.eps")

    def rank_models_best_rank_bar_plot(self, proj: str, models: List[str]):
        """ Draw bar plots per sha to show the best rank for each SHA to find the failed test and model. """
        plt.cla()
        plt.figure(figsize=(14, 6))
        new_models = list()
        all_models_results = defaultdict(dict)
        for model in models:
            result_dir = Macros.model_data_dir / "rank-model" / proj / model / "results" / "best-rank-per-SHA.json"
            sha_results = IOUtils.load(result_dir)
            for k in sha_results:
                if k != "model":
                    v = sha_results[k]
                    all_models_results[k][model] = v
            # end for
        # end for

        X = np.arange(len(list(all_models_results.keys())))
        tick_label = [t for t in list(all_models_results.keys())]
        bar_width = 0.1

        for index, model in enumerate(models):
            model_y = np.array([all_models_results[x][model] for x in list(all_models_results.keys())])
            for x_text, y_text in zip(X, model_y):
                plt.text(x_text + bar_width * index, y_text + 0.005, '%.1f' % y_text, ha='center', va='bottom',
                         fontsize=1)

            plt.bar(X + bar_width * index, model_y, bar_width, align="center", label=model, alpha=0.7)
        plt.xticks(fontproperties='monospace', size=5, rotation=90)
        plt.xticks(X + bar_width * len(models) / 2, tick_label)
        ax = plt.subplot(111)
        ax.legend(bbox_to_anchor=(1.12, 1.1))
        plt.xlabel("sha+mutant")
        plt.ylabel("rank out of 30 tests")
        plt.title("The rank for the failed test")
        plt.savefig(f"{Macros.plot_dir}/figure-{proj}-rank-models-rank-barplot.eps")


    def best_select_rate_box_plot_for_each_subset(self, project: str, models: List[str], subset: str, type=""):
        data = []
        proj = project.split('_')[1]
        if type != "":
            # type includes "simple-rule", "not-simple-rule", "newly-added-tests", "not-newly-added-tests",
            # "killed-tests", "not-killed-tests"
            shalist = IOUtils.load(
                f"{Macros.results_dir}/modelResults/{proj}/shalist-{type}.json")

        for model in models:
            if subset == "all":
                best_select_rate_file = Macros.results_dir / "modelResults" / proj / model / "best-select-rate-per-SHA.json"
            elif subset == "ekstazi":
                best_select_rate_file = Macros.results_dir / "modelResults" / proj / model /  "Ekstazi-best-selection-rate-per-sha.json"
            elif subset == "starts":
                best_select_rate_file = Macros.results_dir / "modelResults" / proj / model /  "STARTS-best-selection-rate-per-sha.json"
            best_select_rate_per_sha = IOUtils.load(best_select_rate_file)
            if type == "":
                data.append(list(best_select_rate_per_sha.values()))
            else:
                percent = format(len(shalist) / len(best_select_rate_per_sha), '.2f')
                list_for_current_model = []
                for commit, value in best_select_rate_per_sha.items():
                    if commit in shalist:
                        list_for_current_model.append(value)
                data.append(list_for_current_model)

        # Tools
        starts_select_rate = []
        ekstazi_select_rate = []
        best_select_rate_file = Macros.results_dir / "modelResults" / proj / "Fail-Code" / "tools-select-rate-per-SHA.json"
        tool_best_select_rates = IOUtils.load(best_select_rate_file)
        for k, v in tool_best_select_rates.items():
            if type == "":
                starts_select_rate.append(v["STARTS"])
                ekstazi_select_rate.append(v["Ekstazi"])
            else:
                if k in shalist:
                    starts_select_rate.append(v["STARTS"])
                    ekstazi_select_rate.append(v["Ekstazi"])

        # end for
        data.append(starts_select_rate)
        data.append(ekstazi_select_rate)
        # Perfect
        best_select_rate_file = Macros.results_dir / "modelResults" / proj / "Fail-Code" / "perfect-select-rate-per-SHA.json"
        perfect_best_select_rates = IOUtils.load(best_select_rate_file)
        if type == "":
            data.append(list(perfect_best_select_rates.values()))
        else:
            list_for_perfect = []
            for commit, value in perfect_best_select_rates.items():
                if commit in shalist:
                    list_for_perfect.append(value)
            data.append(list_for_perfect)

        figure, axes = plt.subplots()
        axes.xaxis.set_tick_params(labelsize=8)
        m_labels = []
        for m in models:
            if "BM25Baseline" in m:
               m_labels.append(m.replace("BM25Baseline", "BM25"))
            elif m == "boosting":
                m_labels.append("bagging")
            else:
                m_labels.append(m)
        # end for
        if type == "":
            plt.title(f"Best selection rate for {proj} , {subset}")
        elif type == "simple-rule":
            plt.title(f"Best selection rate for {proj}, {subset}, \n failed tests match changed classes, {percent}")
        elif type == "not-simple-rule":
            plt.title(f"Best selection rate for {proj}, {subset}, \n not all failed tests match changed classes, {percent}")
        elif type == "newly-added-tests":
            plt.title(f"Best selection rate for {proj}, {subset}, \n failed tests match changed classes \n failed tests include newly added tests, {percent}")
        elif type == "not-newly-added-tests":
            plt.title(f"Best selection rate for {proj}, {subset}, \n failed tests match changed classes \n all failed tests are not newly added tests, {percent}")
        elif type == "no-simple-rule-newly-added-tests":
            plt.title(f"Best selection rate for {proj}, {subset}, \n not all failed tests match changed classes \n failed tests include newly added tests, {percent}")
        elif type == "no-simple-rule-not-newly-added-tests":
            plt.title(f"Best selection rate for {proj}, {subset}, \n not all failed tests match changed classes \n all failed tests are not newly added tests, {percent}")
        elif type == "killed-tests":
            plt.title(f"Best selection rate for {proj}, {subset}, \n failed tests match changed classes \n all failed tests are killed tests of PIT, {percent}")
        elif type == "not-killed-tests":
            plt.title(f"Best selection rate for {proj}, {subset}, \n failed tests match changed classes \n not all failed tests are killed tests of PIT, {percent}")
        elif type == "no-simple-rule-killed-tests":
            plt.title(f"Best selection rate for {proj}, {subset}, \n not all failed tests match changed classes \n all failed tests are killed tests of PIT, {percent}")
        elif type == "no-simple-rule-not-killed-tests":
            plt.title(f"Best selection rate for {proj}, {subset}, \n not all failed tests match changed classes \n not all failed tests are killed tests of PIT, {percent}")

        plt.xticks(rotation=90)
        plt.boxplot(data, labels=m_labels + ["STARTS", "Ekstazi", "Perfect"])
        plt.ylabel("Best Select Rate")
        # plt.title(project + " Best Select Rate")
        plt.tight_layout()
        if type == "":
            plt.savefig(f"{Macros.paper_dir}/figs/fig-{project}-{subset}-best-select-rate-boxplot.eps")
        else:
            plt.savefig(f"{Macros.paper_dir}/figs/fig-{project}-{subset}-{type}-best-select-rate-boxplot.eps")


    def best_select_rate_box_plot(self, project: str, models: List[str], type=""):
        for subset in ["all", "ekstazi", "starts"]:
            self.best_select_rate_box_plot_for_each_subset(project, models, subset, type)


    def num_of_test_cases_raw_eval_data_lineplot(self, projects: List[str]):
        for proj in projects.split():
            raw_eval_path = Macros.raw_eval_data_adding_time_dir / f"{proj}.json"
            # print(raw_eval_path)
            raw_eval_json = IOUtils.load(raw_eval_path)
            num_test_list = []
            for raw_eval_item in raw_eval_json:
                num_test_list.append(len(raw_eval_item["passed_test_list"]))
            plt.plot([i for i in range(len(num_test_list))], num_test_list, label=proj.split("_")[-1])
        plt.xlabel("index of SHA")
        plt.ylabel("number of test cases")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, prop={'size': 5})
        plt.savefig(f"{Macros.plot_dir}/figure-num-of-test-cases-raw-eval-data.eps")

    def num_of_test_cases_mutated_eval_data_lineplot(self, proj: str):
        mutated_eval_path = Macros.data_dir / "mutated-eval-data" / f"{proj}-ag.json"
        mutated_eval_json = IOUtils.load(mutated_eval_path)
        raw_eval_path = Macros.raw_eval_data_adding_time_dir / f"{proj}.json"
        raw_eval_json = IOUtils.load(raw_eval_path)
        training_path = Macros.data_dir / "train_tests.json"
        training_json = IOUtils.load(training_path)
        raw_shalist = []
        for raw_eval_item in raw_eval_json:
            raw_shalist.append(raw_eval_item["commit"])
        sha_index_to_num_tests = {}
        sha_index_to_date = {}
        for mutated_eval_item in mutated_eval_json:
            sha = mutated_eval_item["commit"][:8]
            sha_index = raw_shalist.index(sha)
            date = mutated_eval_item["date"][:10]
            sha_index_to_date[sha_index] = date
            sha_index_to_num_tests[sha_index] = len(mutated_eval_item["failed_test_list"]) + len(
                mutated_eval_item["passed_test_list"])
        shalist = []
        num_test_list = []

        # first is the training sha
        for training_item in training_json:
            if training_item["project"] == proj:
                shalist.append(training_item["sha"] + " " + training_item["date"][:10])
                num_test_list.append(len(training_item["failed_test_list"]) + len(training_item["passed_test_list"]))
                break

        for k in sorted(sha_index_to_num_tests):
            shalist.append(raw_shalist[k] + " " + sha_index_to_date[k])
            num_test_list.append(sha_index_to_num_tests[k])
        if len(shalist) > 10:
            plt.tick_params(axis='both', labelsize=3)
        else:
            plt.tick_params(axis='both', labelsize=5.5)
        plt.plot(shalist, num_test_list)
        plt.ylim(bottom=0, top=max(num_test_list) + 5)
        plt.xlabel("SHA+mutant pair")
        plt.ylabel("number of test cases")
        plt.xticks(range(len(shalist)), [s.replace(" ", "\n") for s in shalist], rotation=90)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(f"{Macros.plot_dir}/figure-{proj}-num-of-test-cases-mutated-eval-data.eps", bbox_inches='tight',
                    pad_inches=0)

    def num_changed_files_vs_avg_selection_rate_barplot(self, projects: List, models: List[str], type: str):
        '''
        x-axis: the number of changed files
        y-axis: avg selection rate or best safe selection rate
        '''
        for project in projects:
            try:                
                plt.cla()
                #plt.figure(figsize=(14, 6))

                fig = plt.figure()
                NUM_COLORS = len(models)
                #cm = plt.get_cmap('gist_rainbow')
                #cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
                #scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
                ax = fig.add_subplot(111)
                #ax.set_prop_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
                #ax[0].set_prop_cycle(color=['#00FFFF','#0000FF','#DC143C' , '#A52A2A','#7FFF00','#8A2BE2','#000000','#D2691E','#008B8B','#B8860B', '#006400', '#FF8C00','#8B0000','#2F4F4F','#FF1493','#FFD700', '#00FF00', '#BA55D3', '#FF4500', '#800080'])
                ax.set_prop_cycle('color',plt.cm.Spectral(np.linspace(0,1,len(models))))
                proj = project.split("_")[-1]
                simple_rule_commits = IOUtils.load(f"{Macros.results_dir}/modelResults/{proj}/shalist-simple-rule.json")

                sha_to_num_changed_files = Counter()
                eval_path = Macros.eval_data_dir / "mutated-eval-data" / f"{project}-ag.json"
                eval_results = IOUtils.load(eval_path)
                for eval_result in eval_results:
                    commit = eval_result["commit"]
                    if (type.startswith("simple-rule") and commit in simple_rule_commits) or (type.startswith("no-simple-rule") and commit not in simple_rule_commits):
                        sha_to_num_changed_files[commit] = len(eval_result["diff_line_number_list_per_file"].keys())
                
                all_models_results = OrderedDict()
                for num_changed_files in sorted(sha_to_num_changed_files.values()):
                    all_models_results[num_changed_files] = {}
                    for model in models:
                        all_models_results[num_changed_files][model] = []
                
                for model in models:
                    if model == "perfect":
                        result_dir = Macros.results_dir / "modelResults" / proj / "Fail-Basic" / "perfect-select-rate-per-SHA.json"
                    else:
                        result_dir = Macros.results_dir / "modelResults" / proj / model / "best-select-rate-per-SHA.json"
                    sha_results = IOUtils.load(result_dir)
                    for sha_mutant, selection_rate in sha_results.items():
                        if sha_mutant not in sha_to_num_changed_files.keys():
                            continue
                        num_changed_files = sha_to_num_changed_files[sha_mutant]
                        all_models_results[num_changed_files][model].append(selection_rate)
                    # end for
                # end for
                
                X = np.arange(len(list(all_models_results.keys())))
                tick_label = [t for t in list(all_models_results.keys())]
                bar_width = 0.08

                for index, model in enumerate(models):
                    if "average" in type:
                        model_y = np.array([round(mean(all_models_results[num_changed_files][model]), 2) for num_changed_files in list(all_models_results.keys())])
                    elif "safe" in type:
                        model_y = np.array(
                            [round(max(all_models_results[num_changed_files][model]), 2) for num_changed_files in
                             list(all_models_results.keys())])
                    for x_text, y_text in zip(X, model_y):
                        plt.text(x_text + bar_width * index, y_text + 0.005, '%.01f' % y_text, ha='center', va='bottom',
                                 fontsize=1)

                    plt.bar(X + bar_width * index, model_y, bar_width, align="center", label=model, alpha=0.7)
                    plt.xticks(fontproperties='monospace', size=5)
                    plt.xticks(X + bar_width * len(models) / 2, tick_label)
                plt.xlabel("number of changed files")
                if "average" in type:
                    plt.ylabel("average selection rate")
                    if type.startswith("simple-rule"):
                        plt.title(f"{proj}, failed tests match changed classes \n Average selection rate vs. number of changed files")
                    else:
                        plt.title(f"{proj}, not all failed tests match changed classes \n Average selection rate vs. number of changed files")
                elif "safe" in type:
                    plt.ylabel("best safe selection rate")
                    if type.startswith("simple-rule"):
                        plt.title(f"{proj}, failed tests match changed classes \n Best safe selection rate vs. number of changed files")
                    else:
                        plt.title(f"{proj}, not all failed tests match changed classes \n Best safe selection rate vs. number of changed files")
                #plt.legend(bbox_to_anchor=(1.05, 1))
                plt.legend(ncol=3, bbox_to_anchor=(-0.15, 1.2), loc="lower left")
                plt.tight_layout()
                plt.savefig(f"{Macros.paper_dir}/figs/figure-{proj}-{type}-selection-rate-histogram.eps")
            except Exception as e:
                print(e)
                continue

    def recall_vs_selection_boxplot_plots_layout(self, projects):
        """Assemble figures to recall_vs_selection.tex which is put in appendix.tex"""

        file = latex.File(Macros.paper_dir / "figs" / "recall_vs_selection_boxplot.tex")
        for subset in ["ekstazi", "starts"]:
            half_proj_num = len(projects) // 2
            for i in range(half_proj_num):
                project = projects[2*i]
                proj = project.split("_")[-1]
                file.append(r"\begin{figure*}[h]")
                file.append(r"\centering")
                file.append(r"\begin{subfigure}[h]{0.48\textwidth}")
                if subset == "all":
                    recall_vs_selection_filename = f"fig-{proj}-rank-models-select-subset-All-results.eps"
                elif subset == "ekstazi":
                    recall_vs_selection_filename = f"fig-{proj}-rank-models-select-subset-Ekstazi-results.eps"
                elif subset == "starts":
                    recall_vs_selection_filename = f"fig-{proj}-rank-models-select-subset-STARTS-results.eps"
                file.append(r"\includegraphics[width=\textwidth]{figs/"+recall_vs_selection_filename+"}")
                file.append(r"\end{subfigure}")
                file.append(r"\hspace{1em}%")
                file.append(r"\begin{subfigure}[h]{0.48\textwidth}")
                project = projects[2*i+1]
                proj = project.split("_")[-1]
                if subset == "all":
                    recall_vs_selection_filename = f"fig-{proj}-rank-models-select-subset-All-results.eps"
                elif subset == "ekstazi":
                    recall_vs_selection_filename = f"fig-{proj}-rank-models-select-subset-Ekstazi-results.eps"
                elif subset == "starts":
                    recall_vs_selection_filename = f"fig-{proj}-rank-models-select-subset-STARTS-results.eps"
                file.append(r"\includegraphics[width=\textwidth]{figs/"+recall_vs_selection_filename+"}")
                file.append(r"\end{subfigure}")
                file.append(r"\end{figure*}")
        file.save()

    def number_of_changed_files_vs_select_rate_plots_layout(self, projects):
        file = latex.File(Macros.paper_dir / "figs" / "number_of_changed_files_vs_select_rate_plots.tex")
        for project in projects:
            proj = project.split("_")[-1]
            for type in ["simple-rule-average", "no-simple-rule-average", "simple-rule-safe", "no-simple-rule-safe"]:
                file.append(r"\begin{figure*}[h]")
                file.append(r"\centering")
                filename = f"figure-{proj}-{type}-selection-rate-histogram.eps"
                file.append(r"\includegraphics[width=0.4\textwidth]{figs/" + filename + "}")
                file.append(r"\end{figure*}")
            # file.append(r"\clearpage")
        file.save()

    def boxplot_end_to_end_time(self, projects, subset, models):
        """Script to make boxplots for models' end2end time."""

        MODELS = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS", "randomforest",
                  "BM25Baseline"]
        TOOLS = ["Ekstazi", "STARTS"]
        for project in projects:
            data = {}
            selection_time_metrics_file_path = Macros.metrics_dir / f"{project}-{subset.lower()}-selection-time.json"
            execution_time_metrics_file_path = Macros.metrics_dir / f"stats-{project}-execution-time-{subset.lower()}.json"
            selection_time_metrics = IOUtils.load(selection_time_metrics_file_path)
            execution_time_metrics = IOUtils.load(execution_time_metrics_file_path)
            subset_selection_time = []
            for selection_time_item in selection_time_metrics:
                subset_selection_time.append(selection_time_item[subset])
            for model, stats in execution_time_metrics.items():
                if model not in MODELS and model not in TOOLS:
                    continue
                if (model == "randomforest" or model == "xgboost") and (
                        project == "apache_commons-csv" or subset == "Ekstazi"):
                    continue
                selection_time_list = []
                execution_time_list = []
                for sha, time in stats.items():
                    execution_time_list.append(time)
                for selection_time_item in selection_time_metrics:
                    selection_time_list.append(selection_time_item[model])
                # end for
                if model.lower() == "ekstazi" or model.lower() == "starts":
                    end_to_end_time_list = [a+b for a, b in zip(selection_time_list, execution_time_list)]
                else:
                    end_to_end_time_list = [a+b+c for a, b, c in zip(selection_time_list, execution_time_list, subset_selection_time)]
                data[model] = end_to_end_time_list

            # end for
            # Creating plot
            xlabels = []
            data_to_plot = []

            for model in models + [subset]:
                if model in data:
                    model_name = model if model != "randomforest" else "EALRTS"
                    xlabels.append(model_name + "\n" + "{:.2f}".format(np.std(data[model])))
                    data_to_plot.append(data[model])
            plt.figure()
            plt.boxplot(data_to_plot, labels=xlabels)
            plt.title(f"End-to-end testing time (seconds) for {project.split('_')[1]},\n combining with {subset}")
            plt.xticks(rotation=90)
            plt.ylabel("Time (s)")
            plt.tight_layout()
            plt.savefig(f"{Macros.paper_dir}/figs/fig-{project}-{subset}-end-to-end-time-boxplot.eps")
            plt.clf()
        # end for


    def boxplot_selection_time(self, projects, subset, models):
        """Script to make boxplots for models' selection time."""
        MODELS = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS",
                  "BM25Baseline", "randomforest"]
        TOOLS = ["Ekstazi", "STARTS"]
        for project in projects:
            data = {}
            selection_time_metrics_file_path = Macros.metrics_dir / f"{project}-{subset.lower()}-selection-time.json"
            selection_time_metrics = IOUtils.load(selection_time_metrics_file_path)
            subset_selection_time = []
            for selection_time_item in selection_time_metrics:
                subset_selection_time.append(selection_time_item[subset])
            for model in MODELS + [subset]:
                selection_time_list = []
                if (model == "randomforest" or model == "xgboost") and (
                        project == "apache_commons-csv" or subset == "Ekstazi"):
                    continue

                for selection_time_item in selection_time_metrics:
                    selection_time_list.append(selection_time_item[model])
                # end for
                data[model] = selection_time_list
            # end for
            # Creating plot
            xlabels = []
            data_to_plot = []

            for model in MODELS + TOOLS:
                if model in data:
                    model_name = model if model != "randomforest" else "EALRTS"
                    xlabels.append(model_name + "\n" + "{:.2f}".format(np.std(data[model])))
                    data_to_plot.append(data[model])
            plt.figure()
            plt.boxplot(data_to_plot, labels=xlabels)
            plt.title(f"Selection time (seconds) for {project.split('_')[1]},\ncombining with {subset}")
            plt.xticks(rotation=90)
            plt.ylabel("Time (s)")
            plt.tight_layout()
            plt.savefig(f"{Macros.paper_dir}/figs/fig-{project}-{subset}-subset-selection-time-boxplot.eps")
            plt.clf()
        # end for