from typing import List
from sklearn.metrics import recall_score, precision_score, f1_score
import numpy as np

from pts.Utils import Utils
from pts.models.rank_model.TestSelectionModel import load_model
from seutil import IOUtils, BashUtils
from pts.Macros import Macros
import os

class MetricsCollect:

    MODELS = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS"]

    def __init__(self):
        self.project = ""
        return

    def collect_metrics(self, which: List[str], project: str, data_type: str="Fail-Basic", select_rate: float=0, **options):
        self.project = project
        for item in which:
            if item == "Analyze-model":
                model_name = options.get("model_name", "rank-model")
                self.analyze_model_results(project, data_type, model_name)
            else:
                raise NotImplementedError
            # end if
        # end for

    def get_all_combinations_correct_confusion_matrix(self):
        from itertools import combinations
        models_comb = combinations(self.MODELS, 2)
        confusion_matrices_all = {}
        for model1, model2 in models_comb:
            confusion_matrix = self.make_correct_confusion_matrix(model1, model2)
            confusion_matrices_all[f"{model1}={model2}"] = confusion_matrix
        # end for
        IOUtils.dump(Macros.metrics_dir / f"correct-confusion-matrices-{self.project}.json", confusion_matrices_all)

    def get_all_combinations_confusion_matrix(self):
        from itertools import combinations
        models_comb = combinations(self.MODELS, 2)
        confusion_matrices_all = {}
        for model1, model2 in models_comb:
            confusion_matrix = self.make_confusion_matrix(model1, model2)
            confusion_matrices_all[f"{model1}={model2}"] = confusion_matrix
        # end for
        IOUtils.dump(Macros.metrics_dir/f"confusion-matrices-{self.project}.json", confusion_matrices_all)

    def make_confusion_matrix(self, model1, model2):
        m1_per_sha_preds = IOUtils.load(
            Macros.results_dir / "modelResults" / self.project.split('_')[1] / model1 / "per-sha-result.json")
        m2_per_sha_preds = IOUtils.load(
            Macros.results_dir / "modelResults" / self.project.split('_')[1] / model2 / "per-sha-result.json")
        m1_pred_scores = []
        m2_pred_scores = []
        for m1_data, m2_data in zip(m1_per_sha_preds, m2_per_sha_preds):
            m1_pred_scores.extend(m1_data["prediction_scores"])
            m2_pred_scores.extend(m2_data["prediction_scores"])
        # end for

        agree_fail_num = 0
        agree_pass_num = 0
        m1_fail_m2_pass_num = 0
        m1_pass_m2_fail_num = 0
        for m1_score, m2_score in zip(m1_pred_scores, m2_pred_scores):
            if m1_score >= 0.5 and m2_score >= 0.5:
                agree_fail_num += 1
            elif m1_score < 0.5 and m2_score < 0.5:
                agree_pass_num += 1
            elif m1_score >= 0.5 and m2_score < 0.5:
                m1_fail_m2_pass_num += 1
            elif m1_score < 0.5 and m2_score >= 0.5:
                m1_pass_m2_fail_num += 1
            # end if
        # end for

        confusion_matrix = {
            "agree_fail_num":agree_fail_num,
            "agree_pass_num": agree_pass_num,
            "m1_fail_m2_pass_num": m1_fail_m2_pass_num,
            "m1_pass_m2_fail_num": m1_pass_m2_fail_num
        }
        return confusion_matrix

    def make_correct_confusion_matrix(self, model1, model2):
        """Build the matrix to show the correct and mistakes made by models."""
        m1_per_sha_preds = IOUtils.load(
            Macros.results_dir / "modelResults" / self.project.split('_')[1] / model1 / "per-sha-result.json")
        m2_per_sha_preds = IOUtils.load(
            Macros.results_dir / "modelResults" / self.project.split('_')[1] / model2 / "per-sha-result.json")
        m1_pred_scores = []
        m2_pred_scores = []
        labels = []
        for m1_data, m2_data in zip(m1_per_sha_preds, m2_per_sha_preds):
            m1_pred_scores.extend(m1_data["prediction_scores"])
            m2_pred_scores.extend(m2_data["prediction_scores"])
            labels.extend(m2_data["labels"])
        # end for

        both_correct_num = 0
        both_wrong_num = 0
        m1_correct_m2_wrong_num = 0
        m1_wrong_m2_correct_num = 0

        for m1_score, m2_score, label in zip(m1_pred_scores, m2_pred_scores, labels):
            if m1_score >= 0.5 and m2_score >= 0.5:
                if label == 1:
                    both_correct_num += 1
                else:
                    both_wrong_num += 1
            elif m1_score < 0.5 and m2_score < 0.5:
                if label == 1:
                    both_wrong_num += 1
                else:
                    both_correct_num += 1
            elif m1_score >= 0.5 and m2_score < 0.5:
                if label == 1:
                    m1_correct_m2_wrong_num += 1
                else:
                    m1_wrong_m2_correct_num += 1
            elif m1_score < 0.5 and m2_score >= 0.5:
                if label == 1:
                    m1_wrong_m2_correct_num += 1
                else:
                    m1_correct_m2_wrong_num += 1
            # end if
        # end for

        confusion_matrix = {
            "both_wrong": both_wrong_num,
            "both_correct": both_correct_num,
            "m1_correct_m2_wrong": m1_correct_m2_wrong_num,
            "m1_wrong_m2_correct": m1_wrong_m2_correct_num
        }
        return confusion_matrix

    def get_recall_Ekstazi(self, project: str):
        per_sha_preds = IOUtils.load(
            Macros.results_dir / "modelResults" / project.split('_')[1] / "Fail-Basic" / "per-sha-result.json")
        select_rates = np.arange(0.0, 1.04, 0.02)
        select_rates.tolist()
        recall_list = []
        for select_rate in select_rates:
            starts_subset_recall_per_sha = []
            for data_item in per_sha_preds:
                num_of_candidate_tests = len(data_item["Ekstazi_labels"])
                starts_selected_labels = []
                for i in range(len(data_item["Ekstazi_labels"])):
                    if data_item["Ekstazi_labels"][i] == 1:
                        starts_selected_labels.append(data_item["labels"][i])
                # end for
                select_size = int(select_rate * num_of_candidate_tests)
                starts_perfect_preds = [0 for _ in range(len(starts_selected_labels))]
                if select_size > 0:
                    for i in range(min(select_size, len(starts_selected_labels))):
                        starts_perfect_preds[i] = 1
                # end if
                if sum(starts_selected_labels) == 0:
                    pass
                else:
                    starts_subset_recall_per_sha.append(recall_score(starts_selected_labels, starts_perfect_preds))
            recall_list.append(sum(starts_subset_recall_per_sha) / len(starts_subset_recall_per_sha))
        return recall_list

    def get_recall_STARTS(self, project: str):
        per_sha_preds = IOUtils.load(
            Macros.results_dir / "modelResults" / project.split('_')[1] / "Fail-Basic" / "per-sha-result.json")
        select_rates = np.arange(0.0, 1.04, 0.02)
        select_rates.tolist()
        recall_list = []
        for select_rate in select_rates:
            starts_subset_recall_per_sha = []
            for data_item in per_sha_preds:
                num_of_candidate_tests = len(data_item["STARTS_labels"])
                starts_selected_labels = []
                for i in range(len(data_item["STARTS_labels"])):
                    if data_item["STARTS_labels"][i] == 1:
                        starts_selected_labels.append(data_item["labels"][i])
                # end for
                select_size = int(select_rate * num_of_candidate_tests)
                starts_perfect_preds = [0 for _ in range(len(starts_selected_labels))]
                if select_size > 0:
                    for i in range(min(select_size, len(starts_selected_labels))):
                        starts_perfect_preds[i] = 1
                # end if
                if sum(starts_selected_labels) == 0:
                    pass
                else:
                    starts_subset_recall_per_sha.append(recall_score(starts_selected_labels, starts_perfect_preds))
            recall_list.append(sum(starts_subset_recall_per_sha) / len(starts_subset_recall_per_sha))
        return recall_list

    def get_best_selection_rate(self, project: str, data_type: str, model_name="rank-model", subset=None):
        """Get the best/safe selection rate for:

        1. selecting from the full list of the test cases, subset=All
        2. selecting from the subset of STARTS/Ekstazi, subset=STARTS or Ekstazi
        return: the best safe selection rate and average safe selection rate
        """
        from statistics import mean
        per_sha_preds = IOUtils.load(
            Macros.results_dir / "modelResults" / project.split('_')[1] / data_type / "per-sha-result.json")
        sha_2_safe_selection_rate = {}
        for data_item in per_sha_preds:
            labels = data_item["labels"]
            if subset == "STARTS":
                tool_labels = data_item["STARTS_labels"]
            elif subset == "Ekstazi":
                tool_labels = data_item["Ekstazi_labels"]
            elif subset == "All":
                tool_labels = data_item["labels"]

            if subset == "All":
                model_starts_preds = data_item["prediction_scores"]
                starts_selected_labels = data_item["labels"]
            else:
                # Add the results of models selection intersecting tools
                model_starts_preds = []
                starts_selected_labels = []
                for i in range(len(tool_labels)):
                    if tool_labels[i] == 1:
                        starts_selected_labels.append(data_item["labels"][i])
                        model_starts_preds.append(data_item["prediction_scores"][i])
                # end for
            # end if
            num_all_failure = sum(starts_selected_labels)
            fail_founded = 0

            pred_with_index = [(x, i) for i, x in enumerate(model_starts_preds)]
            sorted_preds = sorted(pred_with_index, key=lambda x: (x[0], x[1]))
            sorted_preds_index = [x[1] for x in sorted_preds]

            for rank, t in enumerate(sorted_preds_index[::-1]):
                if starts_selected_labels[t] > 0:
                    fail_founded += 1
                if fail_founded == num_all_failure:
                    if rank == len(sorted_preds_index)-1 or model_starts_preds[t] != model_starts_preds[sorted_preds_index[::-1][rank+1]]:
                        sha_2_safe_selection_rate[data_item["commit"]] = (rank+1) / len(labels)
                        break
            # end for
        # end for
        if subset:
            filename = subset+"-best-selection-rate-per-sha.json"
        else:
            filename = "best-selection-rate-per-sha.json"
        IOUtils.dump(Macros.results_dir / "modelResults" / project.split('_')[1] / data_type /
                     filename, sha_2_safe_selection_rate, IOUtils.Format.jsonNoSort)
        return max(sha_2_safe_selection_rate.values()), mean(list(sha_2_safe_selection_rate.values()))


    def get_recall_starts_subset(self, project: str, data_type: str, select_rate: float) -> (float, float):
        """
        Given the select rate for the tests select from STARTS, return the recall and selection rate w.r.t all tests
        in the this sha.
        """
        per_sha_preds = IOUtils.load(
            Macros.results_dir / "modelResults" / project.split('_')[1] / data_type / "per-sha-result.json")
        starts_subset_recall_per_sha = []
        select_rate_all_shas = []
        for data_item in per_sha_preds:
            num_of_candidate_tests = sum(data_item["STARTS_labels"])
            num_of_all_tests = len(data_item["STARTS_labels"])
            # Add the results of models selection intersecting STARTS
            model_starts_preds = []
            starts_selected_labels = []
            for i in range(len(data_item["STARTS_labels"])):
                if data_item["STARTS_labels"][i] == 1:
                    starts_selected_labels.append(data_item["labels"][i])
                    model_starts_preds.append(data_item["prediction_scores"][i])
            # end for
            select_size = int(select_rate * num_of_candidate_tests)
            select_rate_all_shas.append(select_size/num_of_all_tests)
            model_starts_preds = np.array(model_starts_preds)
            starts_subset_preds = np.zeros(len(starts_selected_labels))
            if select_size > 0:
                starts_subset_preds[model_starts_preds.argsort(kind="mergesort")[-select_size:]] = 1
            # end if
            if sum(starts_selected_labels) == 0:
                pass
            else:
                starts_subset_recall_per_sha.append(recall_score(starts_selected_labels, starts_subset_preds))
        return sum(starts_subset_recall_per_sha) / len(starts_subset_recall_per_sha), max(select_rate_all_shas)

    def get_recall_ekstazi_subset(self, project: str, data_type: str, select_rate: float):
        per_sha_preds = IOUtils.load(
            Macros.results_dir / "modelResults" / project.split('_')[1] / data_type / "per-sha-result.json")
        starts_subset_recall_per_sha = []
        select_rate_all_shas = []
        for data_item in per_sha_preds:
            num_of_candidate_tests = sum(data_item["Ekstazi_labels"])
            num_of_all_tests = len(data_item["Ekstazi_labels"])
            # Add the results of models selection intersecting STARTS
            model_starts_preds = []
            starts_selected_labels = []
            for i in range(len(data_item["Ekstazi_labels"])):
                if data_item["Ekstazi_labels"][i] == 1:
                    starts_selected_labels.append(data_item["labels"][i])
                    model_starts_preds.append(data_item["prediction_scores"][i])
            # end for
            select_size = int(select_rate * num_of_candidate_tests)
            select_rate_all_shas.append(select_size / num_of_all_tests)
            model_starts_preds = np.array(model_starts_preds)
            starts_subset_preds = np.zeros(len(starts_selected_labels))
            if select_size > 0:
                starts_subset_preds[model_starts_preds.argsort(kind="mergesort")[-select_size:]] = 1
            # end if
            if sum(starts_selected_labels) == 0:
                pass
            else:
                starts_subset_recall_per_sha.append(recall_score(starts_selected_labels, starts_subset_preds))
        return sum(starts_subset_recall_per_sha) / len(starts_subset_recall_per_sha), max(select_rate_all_shas)

    def get_additional_metrics(self, project: str, data_type: str, threshold: float=0.5):
        """Calculate the average recall and selection rate across mutant sha"""
        from pts.models.rank_model.utils import compute_score
        from statistics import mean
        per_sha_preds = IOUtils.load(
            Macros.results_dir / "modelResults" / project.split('_')[1] / data_type / "per-sha-result.json")
        sha_2_recall = []
        sha_2_select_rate = []
        result_dict = {}

        for data_item in per_sha_preds:
            labels = data_item["labels"]
            prediction_scores = np.array(data_item["prediction_scores"])

            preds = np.zeros(len(labels))
            preds[prediction_scores >= threshold] = 1
            preds = preds.tolist()
            p, r, f = compute_score(predicted_labels=preds, gold_labels=labels)
            selection_rate = sum(preds) / len(preds)
            sha_2_select_rate.append(selection_rate)
            sha_2_recall.append(r*100)
        # end for
        result_dict["recall"] = mean(sha_2_recall)
        result_dict["threshold"] = threshold
        result_dict["selected_pct"] = mean(sha_2_select_rate)
        result_dict["ekstazi_recall"] = 0
        result_dict["starts_recall"] = 0

        results_dump_dir = Macros.model_data_dir / "rank-model" / project.split('_')[1] / data_type / "results" / "test-output"
        IOUtils.mk_dir(results_dump_dir)
        IOUtils.dump(results_dump_dir/"per-file-result.json", result_dict)

    def analyze_model_results(self, project: str, data_type: str, model_name: str):
        """Calculate the metrics based on the stored prediction results of models."""
        from statistics import mean
        from pts.models.rank_model.TestSelectionModel import calculate_apfd
        proj_result_dir = Macros.model_data_dir / "rank-model" / project.split('_')[1]
        pm_result_dir = Macros.results_dir / "modelResults" / project.split('_')[1] / data_type
        if os.path.exists(proj_result_dir / data_type / "results"):
            with IOUtils.cd(proj_result_dir / data_type / "results"):
                BashUtils.run(f"cp *.json {pm_result_dir}", expected_return_code=0)

        per_sha_preds = IOUtils.load(
            Macros.results_dir / "modelResults" / project.split('_')[1] / data_type / "per-sha-result.json")

        sha_2_best_selection_rate = {}
        sha_2_apfd = {}
        sha_2_lowest_threshold = {}
        sha_2_first_fail_threshold = {}
        sha_2_perfect_selection_rate = {}
        sha_2_tools_selection_rate = {}
        sha_2_best_rank = {}

        for data_item in per_sha_preds:
            try:
                # Evaluate results
                labels = data_item["labels"]
                ekstazi_labels = data_item["Ekstazi_labels"]
                starts_labels = data_item["STARTS_labels"]
                prediction_scores = np.array(data_item["prediction_scores"])
                sha = data_item["commit"]
                num_of_candidate_tests = len(labels)
                num_failed_test = sum(labels)
                test_founded = 0
                first_failed_test_founded = False

                # get the best rank and best selection rate
                fail_test_rank = []

                pred_with_index = [(x, i) for i, x in enumerate(prediction_scores.tolist())]
                sorted_preds = sorted(pred_with_index, key=lambda x: (x[0], x[1]))
                sorted_preds_index = [x[1] for x in sorted_preds]

                for rank, t in enumerate(sorted_preds_index[::-1]):
                    if labels[t] > 0 and first_failed_test_founded is False:
                        sha_2_best_rank[sha] = (rank + 1) / num_of_candidate_tests
                        sha_2_first_fail_threshold[sha] = float(prediction_scores[t])
                        first_failed_test_founded = True
                    # end if
                    if labels[t] > 0:
                        test_founded += 1
                        fail_test_rank.append(rank + 1)
                    if test_founded == num_failed_test:
                        # when there are multiple tests whose prediction scores equal to threshold, choose all of them for safety
                        # if rank == len(sorted_preds_index)-1 or prediction_scores[rank] != prediction_scores[rank+1]:
                        if rank == len(sorted_preds_index)-1 or prediction_scores[t] != prediction_scores[sorted_preds_index[::-1][rank+1]]:
                            select_rate = (rank + 1) / num_of_candidate_tests
                            sha_2_best_selection_rate[sha] = select_rate
                            sha_2_lowest_threshold[sha] = float(prediction_scores[t])
                            break
                    # end if
                # end for

                APFD = calculate_apfd\
                    (fail_test_rank, num_of_candidate_tests)
                sha_2_apfd[sha] = APFD
                sha_2_perfect_selection_rate[sha] = num_failed_test / num_of_candidate_tests

                eks_select_rate = sum(ekstazi_labels) / len(ekstazi_labels)
                sts_select_rate = sum(starts_labels) / len(starts_labels)

                sha_2_tools_selection_rate[sha] = {
                    "Ekstazi": eks_select_rate,
                    "STARTS": sts_select_rate
                }
            except Exception as e:
                print(e)
                continue
        if not (Macros.model_data_dir / model_name / project.split('_')[1] / data_type / "results" / "test-output" / "per-file-result.json").exists():
            self.get_additional_metrics(project, data_type)

        starts_subset_best_safe_selection_rate, starts_subset_avg_safe_selection_rate = self.get_best_selection_rate(project, data_type, model_name, "STARTS")
        ekstazi_subset_best_safe_selection_rate, ekstazi_subset_avg_safe_selection_rate = self.get_best_selection_rate(project, data_type, model_name, "Ekstazi")

        best_safe_selection_rate = max(sha_2_best_selection_rate.values())
        avg_safe_selection_rate = mean(list(sha_2_best_selection_rate.values()))

        best_safe_selection_rate_result = {
            "best-safe-selection-rate": best_safe_selection_rate,
            "avg-safe-selection-rate": avg_safe_selection_rate,
            "STARTS-subset-best-safe-selection-rate": starts_subset_best_safe_selection_rate,
            "STARTS-subset-avg-safe-selection-rate": starts_subset_avg_safe_selection_rate,
            "Ekstazi-subset-best-safe-selection-rate": ekstazi_subset_best_safe_selection_rate,
            "Ekstazi-subset-avg-safe-selection-rate": ekstazi_subset_avg_safe_selection_rate
        }

        IOUtils.dump(Macros.results_dir / "modelResults" / project.split('_')[1] / data_type /
                     "best-select-rate-per-SHA.json", sha_2_best_selection_rate, IOUtils.Format.jsonNoSort)
        IOUtils.dump(Macros.results_dir / "modelResults" / project.split('_')[1] / data_type /
                     "apfd-per-sha.json", sha_2_apfd, IOUtils.Format.jsonNoSort)
        IOUtils.dump(Macros.results_dir / "modelResults" / project.split('_')[1] / data_type /
                     "lowest-threshold-per-SHA.json", sha_2_lowest_threshold,
                     IOUtils.Format.jsonNoSort)
        IOUtils.dump(Macros.results_dir / "modelResults" / project.split('_')[1] / data_type /
                     "tools-select-rate-per-SHA.json", sha_2_tools_selection_rate)
        IOUtils.dump(Macros.results_dir / "modelResults" / project.split('_')[1] / data_type /
                     "perfect-select-rate-per-SHA.json", sha_2_perfect_selection_rate, IOUtils.Format.jsonNoSort)
        IOUtils.dump(Macros.results_dir / "modelResults" / project.split('_')[1] / data_type /
                     "best-rank-per-SHA.json", sha_2_best_rank, IOUtils.Format.jsonNoSort)
        IOUtils.dump(Macros.results_dir / "modelResults" / project.split('_')[1] / data_type /
                     "first-fail-test-threshold-per-SHA.json", sha_2_first_fail_threshold)
        IOUtils.dump(Macros.results_dir / "modelResults" / project.split('_')[1] / data_type /
                     "best-safe-selection-rate.json", best_safe_selection_rate_result, IOUtils.Format.jsonNoSort)
