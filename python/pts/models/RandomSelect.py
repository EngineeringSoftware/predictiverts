import random
from typing import List
from pts.models.rank_model.TestSelectionModel import calculate_apfd
from pathlib import Path
from seutil import IOUtils


def shuffle_list(input: List):
    result = input.copy()
    random.shuffle(result)
    return result


def get_best_selection_rate(ordered_test_suite: List[str], failed_test_list: List[str]):
    failed_test_founded = 0
    failed_test_rank = []
    num_candidate_test = len(ordered_test_suite)
    for rank, t in enumerate(ordered_test_suite):
        if t in failed_test_list:
            failed_test_founded += 1
            failed_test_rank.append(rank + 1)
            if failed_test_founded == len(failed_test_list):
                return round((rank + 1) / num_candidate_test, 4), calculate_apfd(failed_test_rank,
                                                                                    num_candidate_test)


def get_first_fail_test_rank(ordered_test_suite: List[str], failed_test_list: List[str]):
    for rank, t in enumerate(ordered_test_suite):
        if t in failed_test_list:
            return round((rank + 1) / len(ordered_test_suite), 4)


def calculate_recall_list(pred: List[str], truth: List[str]):
    true_positive = list(set(pred).intersection(set(truth)))
    return len(true_positive) / len(truth)


def random_model_select(all_test_suite: List[str], failed_test_list: List[str], sample_times=10000):
    best_selection_rate_list = []
    apfd_list = []
    first_fail_test_list = []
    recall_list = []
    select_rate_list = []
    for iter in range(sample_times):
        shuffled_test = shuffle_list(all_test_suite)
        best_selection_rate, apfd = get_best_selection_rate(shuffled_test, failed_test_list)
        first_fail_test = get_first_fail_test_rank(shuffled_test, failed_test_list)
        random_select_size = int(0.75 * len(all_test_suite))
        randomly_selected = random.sample(all_test_suite, random_select_size)
        recall = calculate_recall_list(randomly_selected, failed_test_list)
        random_select_rate = random_select_size / len(all_test_suite)
        # update list
        best_selection_rate_list.append(best_selection_rate)
        apfd_list.append(apfd)
        first_fail_test_list.append(first_fail_test)
        recall_list.append(recall)
        select_rate_list.append(random_select_rate)
    # end for
    random_results = {
        "random-best-select-rate": sum(best_selection_rate_list) / len(best_selection_rate_list),
        "random-apfd": sum(apfd_list) / len(apfd_list),
        "random-first-fail-test-rank": sum(first_fail_test_list) / len(first_fail_test_list),
        "random-recall": sum(recall_list) / len(recall_list),
        "random-select-rate": sum(select_rate_list) / len(select_rate_list)
    }
    return random_results


def run_random_model(eval_data_dir: Path, output_dir: Path):
    eval_ag_data = IOUtils.load(eval_data_dir)
    random_eval_results = {}
    for s_data in eval_ag_data:
        all_test_suite = s_data["failed_test_list"] + s_data["passed_test_list"]
        failed_test_list = s_data["failed_test_list"]
        random_result = random_model_select(all_test_suite, failed_test_list)
        random_eval_results[s_data["commit"]] = random_result
    # end for
    IOUtils.dump(output_dir/"random-model-eval-results.json", random_eval_results)
