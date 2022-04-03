from gensim.summarization.bm25 import BM25
import string
from seutil import BashUtils, IOUtils
from typing import List
from pts.main import Macros
import javalang
from functools import reduce
from pts.models.TFIDFbaseline import get_tf_idf_query_similarity


def max_abs_normalize(scores):
    """Normalize the scores"""
    max_score = max(scores)
    return [s / max_score for s in scores]


def getBM25sims(trainDocs, query):
    bm25model = BM25([i.split() for i in trainDocs])
    sims = bm25model.get_scores(query.split())
    return sims


def build_bm25_model(trainDocs):
    return BM25([i.split() for i in trainDocs])


def tokenize(s):
    result = ""
    buffer = ""
    for word in s.split():
        if word.isupper():
            if len(word) > 1:
                result += word.lower() + " "
        else:
            for c in word:
                if c in string.ascii_lowercase:
                    buffer += c
                elif c in string.ascii_uppercase:
                    if buffer != "":
                        if len(buffer) > 1:
                            result += buffer + " "
                        buffer = ""
                    buffer += c.lower()
                else:
                    if buffer != "":
                        if len(buffer) > 1:
                            result += buffer + " "
                        buffer = ""
            if buffer != "":
                if len(buffer) > 1:
                    result += buffer + " "
                buffer = ""
    return result


def parse_file(SHA: str, filepath: str):
    counter = {}
    try:
        with open(filepath) as f:
            content = f.read().replace("\n", " ")
            tokens = javalang.tokenizer.tokenize(content)
            for i in tokens:
                name = type(i).__name__
                if name == 'Operator' or "Integer" in name or "Floating" in name or name == 'Separator':
                    # print(name)
                    continue
                else:
                    if i.value not in counter:
                        counter[i.value] = 1
                    else:
                        counter[i.value] += 1

    except Exception as e:
        print(f"{SHA}, {filepath}, {e}")
    return reduce((lambda x, key: (key + " ") * counter[key] + x), counter, "")


def pre_proecessing_for_each_sha(project, eval_data_item, subset="All"):
    SHA = eval_data_item["commit"][:8]
    if subset == "All":
        failed_test_list = eval_data_item["failed_test_list"]
        passed_test_list = eval_data_item["passed_test_list"]
        total_test_list = failed_test_list + passed_test_list
    elif subset == "Ekstazi":
        total_test_list = eval_data_item["ekstazi_test_list"]
    elif subset == "STARTS":
        total_test_list = eval_data_item["starts_test_list"]

    changed_files = eval_data_item["diff_per_file"].keys()

    project_folder = Macros.repos_downloads_dir / f"{project}"
    if not project_folder.exists():
        BashUtils.run(f"git clone https://github.com/{project.replace('_', '/')} f{project_folder}")

    test_content = {}
    change_content = {}
    with IOUtils.cd(project_folder):
        BashUtils.run(f"git checkout {SHA}")
        for test in total_test_list:
            filepath = BashUtils.run(f"find . -name {test}.java").stdout
            if filepath == "":
                print(test, "filepath is empty")
            if len(filepath.split("\n")) > 1:
                filepath = filepath.split("\n")[0]
            test_content[test] = parse_file(SHA, filepath)

        for changed_file in changed_files:
            change_content[changed_file] = parse_file(SHA, changed_file)
        eval_data_item["data_objects"] = test_content
        eval_data_item["queries"] = change_content
    return eval_data_item


def pre_processing(project: str):
    eval_data_json = IOUtils.load(Macros.eval_data_dir / "mutated-eval-data" / f"{project}-ag.json")
    res = []
    for eval_data_item in eval_data_json:
        eval_data_item = pre_proecessing_for_each_sha(project, eval_data_item)
        res.append(eval_data_item)
    res_path = Macros.eval_data_dir / "mutated-eval-data" / f"{project}-ag-preprocessed.json"
    IOUtils.dump(res_path, res, IOUtils.Format.jsonPretty)


def run_BM25_baseline_for_each_sha(eval_data_item, rule=False, subset="All"):
    """Run BM25 baseline for the given SHA and return the results in the form of dictionary."""
    res_item = {}
    changed_files = eval_data_item["diff_per_file"].keys()
    failed_test_list = eval_data_item["failed_test_list"]
    passed_test_list = eval_data_item["passed_test_list"]
    if subset == "All":
        test_list = failed_test_list + passed_test_list
    elif subset == "Ekstazi":
        test_list = eval_data_item["ekstazi_test_list"]
    elif subset == "STARTS":
        test_list = eval_data_item["starts_test_list"]
    else:
        raise NotImplementedError
    if not test_list:
        res_item["labels"] = []
        res_item["prediction_scores"] = []
        res_item["commit"] = eval_data_item["commit"]
        res_item["Ekstazi_labels"] = []
        res_item["STARTS_labels"] = []
        return res_item
    trainDocs = [tokenize(eval_data_item["data_objects"].get(i, "")) for i in test_list]  # TEST
    query = tokenize(" ".join([eval_data_item["queries"].get(i, "") for i in changed_files]))
    BM25sims = getBM25sims(trainDocs, query)

    ekstazi_labels = []
    starts_labels = []
    labels = []
    for test in test_list:
        if test in eval_data_item["ekstazi_test_list"]:
            ekstazi_labels.append(1)
        else:
            ekstazi_labels.append(0)
        if test in eval_data_item["starts_test_list"]:
            starts_labels.append(1)
        else:
            starts_labels.append(0)
        if test in failed_test_list:
            labels.append(1)
        else:
            labels.append(0)
    res_item["labels"] = labels
    if rule:
        BM25sims = max_abs_normalize(BM25sims)
        changed_classes = [t.split("/")[-1].replace(".java", "") for t in
                           eval_data_item["diff_line_number_list_per_file"].keys()]
        modified_test_class = [t for t in changed_classes if "Test" in t]
        for index, t in enumerate(test_list):
            if t in modified_test_class:
                BM25sims[index] = 1
    res_item["prediction_scores"] = BM25sims
    res_item["commit"] = eval_data_item["commit"]
    res_item["Ekstazi_labels"] = ekstazi_labels
    res_item["STARTS_labels"] = starts_labels
    return res_item


def run_BM25_baseline(project: str):
    # testDump, key is test name, value is a dict, "fail", "time", "doc"
    processed_eval_data_json = IOUtils.load(
        Macros.eval_data_dir / "mutated-eval-data" / f"{project}-ag-preprocessed.json")
    res = []
    for eval_data_item in processed_eval_data_json:
        res_item = run_BM25_baseline_for_each_sha(eval_data_item)
        res.append(res_item)
    output_dir = Macros.model_data_dir / "rank-model" / project.split('_')[1] / "BM25Baseline" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    IOUtils.dump(output_dir / "per-sha-result.json", res, IOUtils.Format.jsonPretty)


# Code for getting BM25 for training data (mutants)
# Step1: first get the contents of files for the training sha
def pre_proecessing_for_training_sha(project, changed_file_list: List[str]):
    """
    Extract the contents for the PIT tool mutants, test files' contents, source files' contents.
    param: name of the project
    param: a list of all mutated files' paths
    return a dict of file => contents
    """
    from pts.main import proj_logs
    training_SHA = proj_logs[project]
    data_item = {}

    collected_results_dir = Macros.repos_results_dir / project / "collector"
    if (collected_results_dir / "method-data.json").exists():
        test_class_2_methods = IOUtils.load(collected_results_dir / "test2meth.json")
    else:
        raise FileNotFoundError("Can not find test2meth.json.")
    total_test_list = list(test_class_2_methods.keys())

    project_folder = Macros.repos_downloads_dir / f"{project}"
    if not project_folder.exists():
        BashUtils.run(f"git clone https://github.com/{project.replace('_', '/')} f{project_folder}")

    test_content = {}
    change_content = {}
    with IOUtils.cd(project_folder):
        BashUtils.run(f"git checkout {training_SHA}")
        for test in total_test_list:
            filepath = BashUtils.run(f"find . -name {test}.java").stdout
            if filepath == "":
                print(test, "filepath is empty")
            if len(filepath.split("\n")) > 1:
                filepath = filepath.split("\n")[0]
            test_content[test] = parse_file(training_SHA, filepath)

        for changed_file in changed_file_list:
            change_content[changed_file] = parse_file(training_SHA, changed_file)
        data_item["data_objects"] = test_content
        data_item["queries"] = change_content
    return data_item


# Step2: get documents and queries for the training mutants
def get_BM25_score(data_item, changed_file: str, bm25_model):
    """
    Given the name of the tests and the changed file path, from the pre-processed data items,
    the BM25 scores will be returned.
    """

    query = tokenize(data_item["queries"].get(changed_file, ""))
    BM25sims = bm25_model.get_scores(query.split())

    return BM25sims
