import time
from typing import List
from pathlib import Path
import torch
from seutil import IOUtils, BashUtils
import numpy as np

from pts.models.rank_model.TestSelectionModel import load_model, calculate_apfd
from pts.processor.data_utils.SubTokenizer import SubTokenizer
from pts.processor.RankProcess import RankProcessor
from pts.Macros import *
from pts.collector.eval_data_collection import diff_per_file_for_each_SHA
from pts.main import proj_logs
from os import listdir
from os.path import isfile, join
from pts.collector.eval_data_collection import test_list_from_surefile_reports

maven_home = os.getenv('M2_HOME')


def process_eval_data_for_one_sha(eval_data_item, subset="All"):
    """Process the eval data for one sha exactly suitable for the model."""
    data_list = list()

    # do sanity check
    if "diff_per_file" not in eval_data_item:
        bad_sha = eval_data_item["commit"]
        print(f"This sha {bad_sha} does not have diff per file, the model can not run. Please fix the bug!")
        return

    if subset == "All":
        passed_test_clss: List = eval_data_item["passed_test_list"]
        failed_test_clss: List = eval_data_item["failed_test_list"]
    elif subset == "Ekstazi":
        failed_test_clss: List = list(
            set(eval_data_item["failed_test_list"]).intersection(set(eval_data_item["ekstazi_test_list"])))
        passed_test_clss: List = list(set(eval_data_item["ekstazi_test_list"]).difference(set(failed_test_clss)))
    elif subset == "STARTS":
        failed_test_clss: List = list(
            set(eval_data_item["failed_test_list"]).intersection(set(eval_data_item["starts_test_list"])))
        passed_test_clss: List = list(set(eval_data_item["starts_test_list"]).difference(set(failed_test_clss)))
    else:
        raise NotImplementedError

    ekstazi_selected: List = eval_data_item["ekstazi_test_list"]
    starts_selected: List = eval_data_item["starts_test_list"]

    # tuple of (ChangedFile.java, added_code)
    changed_files_code = [(c.split('/')[-1], eval_data_item["diff_per_file"][c]) for c in
                          eval_data_item["diff_per_file"]]

    if subset == "All":
        num_test_class = len(failed_test_clss) + len(passed_test_clss)
    elif subset == "Ekstazi":
        num_test_class = len(eval_data_item["ekstazi_test_list"])
    elif subset == "STARTS":
        num_test_class = len(eval_data_item["starts_test_list"])
    else:
        raise NotImplementedError
    assert num_test_class == (len(failed_test_clss) + len(passed_test_clss))

    num_changed_file = len(changed_files_code)

    start_time = time.time()
    if len(failed_test_clss) <= 0:
        print(f"{subset} unsafe")
    for cls, code in changed_files_code:
        # Ignore non java files
        if "java" not in cls:
            continue
        code_diffs = SubTokenizer.sub_tokenize_java_like(code)
        # get mutator
        mutator_type = RankProcessor.abstract_mutated_type(code)
        for ftc in failed_test_clss:
            # get labels from RTS tools
            if ftc in ekstazi_selected:
                ekstazi_label = 1
            else:
                ekstazi_label = 0
            if ftc in starts_selected:
                starts_label = 1
            else:
                starts_label = 0
            # get test code
            pos_test_input = SubTokenizer.sub_tokenize_java_like(ftc)
            mutated_clss = SubTokenizer.sub_tokenize_java_like(cls.split('.')[-2])
            mutated_method = []
            data_point = {
                "sha": eval_data_item["commit"],
                "label": 1,
                "changed_class_name": mutated_clss,
                "changed_method_name": mutated_method,
                "code_diff": code_diffs,
                "abstract_code_diff": mutator_type,
                "pos_test_class": pos_test_input,
                "neg_test_class": pos_test_input,
                "ekstazi_label": ekstazi_label,
                "starts_label": starts_label,
                "num_test_class": num_test_class,
                "num_changed_files": num_changed_file
            }
            data_list.append(data_point)
        # end for
        for ptc in passed_test_clss:
            # get labels from RTS tools
            if ptc in ekstazi_selected:
                ekstazi_label = 1
            else:
                ekstazi_label = 0
            if ptc in starts_selected:
                starts_label = 1
            else:
                starts_label = 0
            # get test code
            neg_test_input = SubTokenizer.sub_tokenize_java_like(ptc)
            mutated_clss = SubTokenizer.sub_tokenize_java_like(cls.split('.')[-2])
            mutated_method = []
            data_point = {
                "sha": eval_data_item["commit"],
                "label": 0,
                "changed_method_name": mutated_method,
                "changed_class_name": mutated_clss,
                "code_diff": code_diffs,
                "abstract_code_diff": mutator_type,
                "pos_test_class": neg_test_input,
                "neg_test_class": neg_test_input,
                "ekstazi_label": ekstazi_label,
                "starts_label": starts_label,
                "num_test_class": num_test_class,
                "num_changed_files": num_changed_file
            }
            data_list.append(data_point)
        # end for
    end_time = time.time()
    return data_list, end_time - start_time


# def get_selection_time_per_sha(test_data_item, model_saving_dir: Path):
#     """Caller function to load trained and run one data point, return the running time"""
#     start_time = time.time()
#     model = load_model(model_saving_dir)
#     model.mode = "test"
#     test_data, _ = process_eval_data_for_one_sha(test_data_item)
#     num_of_candidate_tests = test_data[0]["num_test_class"]
#     num_of_changed_files = test_data[0]["num_changed_files"]
#
#     # Start to run the model
#     model.eval()
#     test_batches = model.create_pair_batches(mode="test", dataset=test_data)
#
#     # self.logger.info(f"Number of changed files in {s} is {num_of_changed_files}."
#     #                  f"Number of candidate test classes is {num_of_candidate_tests}")
#
#     s_pred_scores = []
#     s_starts_labels = []
#     s_labels = []
#     s_ekstazi_labels = []
#
#     with torch.no_grad():
#         for b, batch_data in enumerate(test_batches):
#             pos_score, _ = model.forward(batch_data)
#             s_pred_scores.extend([element.item() for element in pos_score.flatten()])
#             s_labels.extend([element.item() for element in batch_data.label.flatten()])
#             s_starts_labels.extend([element.item() for element in batch_data.starts_label.flatten()])
#             s_ekstazi_labels.extend([element.item() for element in batch_data.ekstazi_label.flatten()])
#         # end for
#     # end with
#
#     # aggregate prediction scores
#     prediction_scores = np.zeros(int(num_of_candidate_tests))
#     for i in range(0, len(s_pred_scores), num_of_candidate_tests):
#         tmp = np.array(s_pred_scores[i: i + num_of_candidate_tests])
#         prediction_scores = np.maximum(prediction_scores, tmp)
#     # end for
#     preds = np.zeros(num_of_candidate_tests)
#     preds[prediction_scores >= 0.5] = 1
#
#     run_time = time.time() - start_time
#     return run_time

def get_total_selection_time(proj, models, subset):
    # preparation
    institution = proj.split("_")[0]
    project_name = proj.split("_")[1]
    if Path(f"{Macros.repos_downloads_dir}/{institution}_{project_name}_ekstazi").is_dir():
        BashUtils.run(f"rm -rf {Macros.repos_downloads_dir}/{institution}_{project_name}_ekstazi")
    BashUtils.run(
        f"git clone https://github.com/{institution}/{project_name} {Macros.repos_downloads_dir}/{institution}_{project_name}_ekstazi")

    if Path(f"{Macros.repos_downloads_dir}/{institution}_{project_name}_starts").is_dir():
        BashUtils.run(f"rm -rf {Macros.repos_downloads_dir}/{institution}_{project_name}_starts")
    BashUtils.run(
        f"git clone https://github.com/{institution}/{project_name} {Macros.repos_downloads_dir}/{institution}_{project_name}_starts")

    result = []
    mutated_eval_data = IOUtils.load(f"{Macros.eval_data_dir}/mutated-eval-data/{proj}-ag.json")
    for data_item in mutated_eval_data:
        result_item = {}
        result_item["commit"] = data_item["commit"]
        for model in models:
            model_res = run_model_end_to_end(proj, model, data_item, subset, eval=False)
            result_item[model] = model_res["time"]
            print(model, result_item)
        result.append(result_item)
    IOUtils.dump(f"{Macros.metrics_dir}/{proj}-{subset.lower()}-selection-time.json", result, IOUtils.Format.jsonPretty)


def eval_real_tests(rule=False):
    """Run 10 models on real failure tests and dump the results to json files.

    Return:
    first_time_failed_tests_evaled_change_failed_no_rule.json
    first_time_failed_tests_evaled_no_rule.json
    """
    result = []
    failed_test_changed_res = []
    failed_test_not_changed_res = []
    MODELS = ["Fail-Basic", "Fail-Code", "Fail-ABS", "Ekstazi-Basic", "Ekstazi-Code", "Ekstazi-ABS", "BM25Baseline",
              "Fail-Basic-BM25Baseline", "Ekstazi-Basic-BM25Baseline", "boosting", "randomforest"]
    first_time_failed_tests = IOUtils.load(f"{Macros.raw_eval_data_dir}/first_time_failed_tests.json")
    for data_item in first_time_failed_tests:

        test_changed = False
        changed_files = data_item["diff_per_file"].keys()
        failed_tests = data_item["first_time_failed_test_list"]
        for changed_file in changed_files:
            if changed_file.replace(".java", "").split("/")[-1] in failed_tests:
                test_changed = True
                break

        for model in MODELS:
            if data_item["project"] == "apache_commons-csv" and model == "randomforest":
                continue
            ekstazi_test_list = set(data_item["ekstazi_test_list"])
            starts_test_list = set(data_item["starts_test_list"])
            model_res = run_model_end_to_end(data_item["project"], model, data_item, "All", rule)
            data_item[model] = model_res

        if rule:
            if test_changed:
                failed_test_changed_res.append(data_item)
            else:
                failed_test_not_changed_res.append(data_item)
        else:
            result.append(data_item)

    if rule:
        IOUtils.dump(f"{Macros.raw_eval_data_dir}/first_time_failed_tests_evaled_change_failed_rule.json",
                     failed_test_changed_res)
        IOUtils.dump(f"{Macros.raw_eval_data_dir}/first_time_failed_tests_evaled_rule.json",
                     failed_test_not_changed_res)
    else:
        IOUtils.dump(f"{Macros.raw_eval_data_dir}/first_time_failed_tests_evaled_no_rule.json", result)


def process_ml_model(project: str, data_type: str, test_data_item, rule=False, subset="All"):
    project_name = project.split("_")[-1]
    model_saving_dir = Macros.model_data_dir / "rank-model" / project_name / data_type / "saved_models" / "best_model"
    model = load_model(model_saving_dir)
    model.mode = "test"

    with IOUtils.cd(f"{Macros.repos_downloads_dir}/{project}"):
        # get test classes
        testclasses = [f for f in listdir(".") if isfile(join(".", f)) and "Test" in f]
        # for each model, collect its related features
        if data_type != "Fail-Basic" and data_type != "Ekstazi-Basic":
            diff_per_file_for_each_SHA(test_data_item["prev_commit"], test_data_item["commit"][:8])

    test_data, abs_time = process_eval_data_for_one_sha(test_data_item, subset)
    if not test_data:
        return [], [], [], [], abs_time
    num_of_candidate_tests = test_data[0]["num_test_class"]

    # Start to run the model
    model.eval()
    test_batches = model.create_pair_batches(mode="test", dataset=test_data, batch_size=128)

    s_pred_scores = []
    s_starts_labels = []
    s_labels = []
    s_ekstazi_labels = []

    with torch.no_grad():
        for b, batch_data in enumerate(test_batches):
            pos_score, _ = model.forward(batch_data)
            s_pred_scores.extend([element.item() for element in pos_score.flatten()])
            s_labels.extend([element.item() for element in batch_data.label.flatten()])
            s_starts_labels.extend([element.item() for element in batch_data.starts_label.flatten()])
            s_ekstazi_labels.extend([element.item() for element in batch_data.ekstazi_label.flatten()])
        # end for
    # end with

    prediction_scores = np.zeros(int(num_of_candidate_tests))
    for i in range(0, len(s_pred_scores), num_of_candidate_tests):
        tmp = np.array(s_pred_scores[i: i + num_of_candidate_tests])
        prediction_scores = np.maximum(prediction_scores, tmp)

    # rule means rule-based selection: always select tests whose class name match the changed class name
    if rule:
        modified_test_class = [d["changed_class_name"] for d in test_data if "test" in d["changed_class_name"]]
        test_index = []
        for t in modified_test_class:
            test_index.extend(
                [i for i, d in enumerate(test_data[:num_of_candidate_tests]) if d["pos_test_class"] == t])
        for i in test_index:
            prediction_scores[i] = 1

    labels = s_labels[:num_of_candidate_tests]
    ekstazi_labels = s_ekstazi_labels[:num_of_candidate_tests]
    starts_labels = s_starts_labels[:num_of_candidate_tests]
    # print("process ml labels", labels)
    # print("process ml ekstazi labels", ekstazi_labels)
    return prediction_scores, labels, ekstazi_labels, starts_labels, abs_time


def max_abs_normalize(scores: List):
    """Normalize the scores"""
    max_score = max(scores)
    return [s / max_score for s in scores]


def run_model_end_to_end(project: str, data_type: str, test_data_item, subset: str,
                         rule=False, eval=True):
    from pts.models.BM25Baseline import pre_proecessing_for_each_sha
    from pts.models.BM25Baseline import run_BM25_baseline_for_each_sha
    from pts.models.EALRTSBaseline import run_EALRTS_baseline_for_each_sha
    """
    Caller function to load trained and run one data point, return the evaluation result
    test_data_item is expected to be deleted in the future
    """
    res = {}

    num_of_candidate_tests = len(test_data_item["passed_test_list"]) + len(test_data_item["failed_test_list"])

    if test_data_item["ekstazi_failed_test_list"] and test_data_item["ekstazi_failed_test_list"][0] not in \
            test_data_item["ekstazi_test_list"]:
        test_data_item["ekstazi_test_list"] += test_data_item["ekstazi_failed_test_list"]
    if test_data_item["starts_failed_test_list"] and test_data_item["starts_failed_test_list"][0] not in test_data_item[
        "starts_test_list"]:
        test_data_item["starts_test_list"] += test_data_item["starts_failed_test_list"]

    if len(test_data_item["ekstazi_test_list"]) != len(set(test_data_item["ekstazi_test_list"])):
        test_data_item["ekstazi_test_list"] = list(set(test_data_item["ekstazi_test_list"]))
    if len(test_data_item["starts_test_list"]) != len(set(test_data_item["starts_test_list"])):
        test_data_item["starts_test_list"] = list(set(test_data_item["starts_test_list"]))

    abs_time = 0
    if data_type == "Ekstazi":
        with IOUtils.cd(f"{Macros.repos_downloads_dir}/{project}_ekstazi"):
            BashUtils.run(f"git checkout {test_data_item['commit']}")
            BashUtils.run(f"cp {Macros.tools_dir}/ekstazi-extension-1.0-SNAPSHOT.jar {maven_home}/lib/ext")
            BashUtils.run(f"mvn clean test-compile {Macros.SKIPS}")
            start_time = time.time()
            BashUtils.run(f"mvn ekstazi:select {Macros.SKIPS}")
            BashUtils.run(f"rm {maven_home}/lib/ext/ekstazi-extension-1.0-SNAPSHOT.jar")
    elif data_type == "STARTS":
        with IOUtils.cd(f"{Macros.repos_downloads_dir}/{project}_starts"):
            BashUtils.run(f"git checkout {test_data_item['commit']}")
            BashUtils.run(f"mvn clean test-compile {Macros.SKIPS}")
            start_time = time.time()
            BashUtils.run(f"mvn starts:select {Macros.SKIPS}")
    elif data_type == "BM25Baseline":
        start_time = time.time()
        test_data_item = pre_proecessing_for_each_sha(project, test_data_item, subset)
        bm25_baseline_res = run_BM25_baseline_for_each_sha(test_data_item, rule, subset)
        test_data_item.pop('data_objects', None)
        test_data_item.pop('queries', None)
        labels = bm25_baseline_res["labels"]
        ekstazi_labels = bm25_baseline_res["Ekstazi_labels"]
        starts_labels = bm25_baseline_res["STARTS_labels"]
        prediction_scores = bm25_baseline_res["prediction_scores"]
    elif data_type == "Fail-Basic-BM25Baseline":
        from operator import add
        start_time = time.time()
        test_data_item = pre_proecessing_for_each_sha(project, test_data_item, subset)
        bm25_baseline_res = run_BM25_baseline_for_each_sha(test_data_item, rule, subset)

        test_data_item.pop('data_objects', None)
        test_data_item.pop('queries', None)
        labels = bm25_baseline_res["labels"]
        ekstazi_labels = bm25_baseline_res["Ekstazi_labels"]
        starts_labels = bm25_baseline_res["STARTS_labels"]
        bm25_prediction_scores = bm25_baseline_res["prediction_scores"]

        ml_prediction_scores, _, _, _, abs_time = process_ml_model(project, "Fail-Basic", test_data_item, rule)

        ir_model_scores = max_abs_normalize(bm25_prediction_scores)
        ml_model_scores = ml_prediction_scores
        prediction_scores = list(map(add, ml_model_scores, ir_model_scores))
    elif data_type == "Ekstazi-Basic-BM25Baseline":
        from operator import add
        start_time = time.time()
        test_data_item = pre_proecessing_for_each_sha(project, test_data_item, subset)
        bm25_baseline_res = run_BM25_baseline_for_each_sha(test_data_item, rule, subset)

        test_data_item.pop('data_objects', None)
        test_data_item.pop('queries', None)
        labels = bm25_baseline_res["labels"]
        ekstazi_labels = bm25_baseline_res["Ekstazi_labels"]
        starts_labels = bm25_baseline_res["STARTS_labels"]
        bm25_prediction_scores = bm25_baseline_res["prediction_scores"]

        ml_prediction_scores, _, _, _, abs_time = process_ml_model(project, "Ekstazi-Basic", test_data_item, rule)

        ir_model_scores = max_abs_normalize(bm25_prediction_scores)
        ml_model_scores = ml_prediction_scores
        prediction_scores = list(map(add, ml_model_scores, ir_model_scores))
    elif data_type == "boosting":
        from operator import add
        start_time = time.time()
        trained_ensemble_models = IOUtils.load(
            Macros.model_data_dir / "rank-model" / project.split('_')[1] / "boosting" / "boosting-model.json")
        prediction_scores = [0 for _ in range(num_of_candidate_tests)]
        for m, weight in trained_ensemble_models:
            scores, labels, ekstazi_labels, starts_labels, abs_time = process_ml_model(project, m, test_data_item, rule,
                                                                                       subset)
            model_scores = list(map(lambda x: x * weight, scores))
            prediction_scores = list(map(add, prediction_scores, model_scores))
    elif data_type == "randomforest" or data_type == "xgboost":
        # these two projects do not have EALRTS models
        if project == "apache_commons-csv":
            if eval:
                res = {"time": float("inf"),
                       "all_best_rank": 1, "all_first_fail_threshold": 0, "all_best_selection_rate": 1,
                       "all_sha_2_lowest_threshold": 0,
                       "ekstazi_best_rank": 1, "ekstazi_first_fail_threshold": 0, "ekstazi_best_selection_rate": 1,
                       "ekstazi_sha_2_lowest_threshold": 0,
                       "starts_best_rank": 1, "starts_first_fail_threshold": 0, "starts_best_selection_rate": 1,
                       "starts_sha_2_lowest_threshold": 0,
                       "perfect_selection_rate": 1, "ekstazi_selection_rate": 1, "starts_selection_rate": 1,
                       "prediction_scores": []}
            else:
                res = {"time": float("inf")}
            return res
        start_time = time.time()
        ealrts_baseline_res, compile_time = run_EALRTS_baseline_for_each_sha(project, data_type, test_data_item, subset)
        labels = ealrts_baseline_res["labels"]
        ekstazi_labels = ealrts_baseline_res["Ekstazi_labels"]
        starts_labels = ealrts_baseline_res["STARTS_labels"]
        prediction_scores = ealrts_baseline_res["prediction_scores"]
    else:
        start_time = time.time()
        prediction_scores, labels, ekstazi_labels, starts_labels, abs_time = process_ml_model(project, data_type,
                                                                                              test_data_item, rule,
                                                                                              subset)
    run_time = time.time() - start_time
    if data_type == "randomforest" or data_type == "xgboost":
        run_time -= compile_time
    if "ABS" in data_type:
        res["time"] = run_time
    else:
        res["time"] = run_time - abs_time
    # print(data_type, run_time)
    # print("abs time", abs_time)
    # Evaluate results
    if eval:
        num_failed_test = sum(labels)

        all_subset_res = select_from_subset("All", prediction_scores, labels, labels, num_of_candidate_tests)
        ekstazi_subset_res = select_from_subset("Ekstazi", prediction_scores, labels, ekstazi_labels,
                                                num_of_candidate_tests)
        starts_subset_res = select_from_subset("STARTS", prediction_scores, labels, starts_labels,
                                               num_of_candidate_tests)
        res["all_best_rank"] = all_subset_res["best_rank"]
        res["all_first_fail_threshold"] = all_subset_res["first_fail_threshold"]
        res["all_best_selection_rate"] = all_subset_res["best_selection_rate"]
        res["all_sha_2_lowest_threshold"] = all_subset_res["sha_2_lowest_threshold"]

        res["ekstazi_best_rank"] = ekstazi_subset_res["best_rank"]
        res["ekstazi_first_fail_threshold"] = ekstazi_subset_res["first_fail_threshold"]
        res["ekstazi_best_selection_rate"] = ekstazi_subset_res["best_selection_rate"]
        res["ekstazi_sha_2_lowest_threshold"] = ekstazi_subset_res["sha_2_lowest_threshold"]

        res["starts_best_rank"] = starts_subset_res["best_rank"]
        res["starts_first_fail_threshold"] = starts_subset_res["first_fail_threshold"]
        res["starts_best_selection_rate"] = starts_subset_res["best_selection_rate"]
        res["starts_sha_2_lowest_threshold"] = starts_subset_res["sha_2_lowest_threshold"]

        res["perfect_selection_rate"] = num_failed_test / num_of_candidate_tests
        if len(ekstazi_labels) == 0:
            res["ekstazi_selection_rate"] = 1
        else:
            res["ekstazi_selection_rate"] = sum(ekstazi_labels) / len(ekstazi_labels)
        if len(starts_labels) == 0:
            res["starts_selection_rate"] = 1
        else:
            res["starts_selection_rate"] = sum(starts_labels) / len(starts_labels)

        res["prediction_scores"] = list(prediction_scores)
        # consider selecting from the subset of Ekstazi/STARTS
    return res


def select_from_subset(subset, prediction_scores, labels, tool_labels, num_of_candidate_tests):
    '''
    subset: "All", "Ekstazi", "STARTS"
    prediction_scores: list of float
    labels: list of 0/1
    tool_labels: list of 0/1
    num_of_candidate_tests: int
    '''
    res = {}
    if subset == "All":
        subset_prediction_scores = prediction_scores
        subset_labels = labels
    else:
        subset_prediction_scores = []
        subset_labels = []
        for i in range(len(tool_labels)):
            if tool_labels[i] == 1:
                subset_prediction_scores.append(prediction_scores[i])
                subset_labels.append(labels[i])

    # print("subset", subset)
    # print("subset labels", subset_labels)
    # print("subset labels len", len(subset_labels))
    # print("subset prediction scores", subset_prediction_scores)
    # print("number of candidate tests", num_of_candidate_tests)
    # subset_prediction_scores = -1 * subset_prediction_scores
    subset_prediction_scores = np.array(subset_prediction_scores)
    # print("argsorted prediction scores", -subset_prediction_scores.argsort(kind="mergesort")[::-1])
    num_failed_test = sum(subset_labels)
    test_found = 0
    first_failed_test_found = False
    # get the best rank and best selection rate
    fail_test_rank = []

    res["best_rank"] = 0
    res["first_fail_threshold"] = 0
    res["best_selection_rate"] = 0
    res["sha_2_lowest_threshold"] = 0

    sorted_index = subset_prediction_scores.argsort(kind="mergesort")[::-1]
    for rank, t in enumerate(sorted_index):
        if subset_labels[t] > 0 and first_failed_test_found is False:
            res["best_rank"] = (rank + 1) / num_of_candidate_tests
            res["first_fail_threshold"] = subset_prediction_scores[t]
            first_failed_test_found = True
        # end if
        if subset_labels[t] > 0:
            test_found += 1
            fail_test_rank.append(rank + 1)
        if test_found == num_failed_test:
            if rank == len(subset_prediction_scores) - 1 or subset_prediction_scores[t] != \
                    prediction_scores[sorted_index[rank + 1]]:
                select_rate = (rank + 1) / num_of_candidate_tests
                res["best_selection_rate"] = select_rate
                print("best selection rate", select_rate)
                res["sha_2_lowest_threshold"] = subset_prediction_scores[t]
                break
        # end if
    # end for
    APFD = calculate_apfd(fail_test_rank, num_of_candidate_tests)
    res["apfd"] = APFD
    return res


def run_test_one_by_one(projs):
    """Get test execution time for every test in the repository.

    Output: dict: commit:test -> execution time
    """
    for proj in projs:
        res = {}
        mutated_eval_data = IOUtils.load(f"{Macros.eval_data_dir}/mutated-eval-data/{proj}-ag.json")
        with IOUtils.cd(f"{Macros.repos_downloads_dir}/{proj}"):
            for mutated_eval_item in mutated_eval_data:
                commit = mutated_eval_item["commit"][:8]
                if commit not in res:
                    res[commit] = {}
                    BashUtils.run(f"git checkout -f {commit}")
                    BashUtils.run(f"mvn clean test {Macros.SKIPS}")
                    qualified_failed_test_list, qualified_passed_test_list, _ = test_list_from_surefile_reports(True)
                    test_list = qualified_failed_test_list + qualified_passed_test_list
                    BashUtils.run(f"mvn clean test-compile {Macros.SKIPS}")
                    for test in test_list:
                        start_time = time.time()
                        BashUtils.run(f"mvn test -Dtest={test} {Macros.SKIPS}")
                        end_time = time.time()
                        exe_time = end_time - start_time
                        res[commit][test] = exe_time
        IOUtils.dump(f"{Macros.eval_data_dir}/mutated-eval-data/{proj}-ag-time-for-each-test.json", res,
                     IOUtils.Format.jsonPretty)
