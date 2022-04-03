import pickle
from seutil import BashUtils, IOUtils
from pts.main import Macros
import pandas as pd
from pts.processor.EALRTSPreprocessor import EALRTSProcessor
import os

def run_EALRTS_baseline(project: str, model_type: str):
    proj = project.split("_")[-1]

    preprocessed_eval_data_path = f"{Macros.model_data_dir}/EALRTS/{project}/eval_reducedData.txt"
    # load eval data file and saved model
    eval_data_df = pd.read_csv(preprocessed_eval_data_path, sep=",", header=None)
    eval_data_df.columns = ["id", "test runs", "File cardinality", "Target cardinality", "minimal distance",
                            "connected tests", "change history", "failure rate", "failed test", "test name"]
    saved_model_path = f"{Macros.model_data_dir}/EALRTS/{project}/{model_type}.pkl"
    saved_model = pickle.load(open(saved_model_path, 'rb'))

    if model_type == "randomforest":
        x_test = eval_data_df.drop(["id", "failed test", "File cardinality", "test name"], axis=1)
    elif model_type == "xgboost":
        x_test = eval_data_df.drop(["id", "failed test", "test name"], axis=1)
    y_pred = saved_model.predict_proba(x_test)
    y_pred_one = y_pred[:, 1].tolist()

    eval_data_json = IOUtils.load(Macros.eval_data_dir / "mutated-eval-data" / f"{project}-ag.json")

    res = []
    for eval_data_item in eval_data_json:
        res_item = {}
        commit = eval_data_item["commit"]
        failed_test_list = eval_data_item["failed_test_list"]
        passed_test_list = eval_data_item["passed_test_list"]
        ekstazi_labels = []
        starts_labels = []

        prediction_scores = []
        for test in eval_data_item["failed_test_list"] + eval_data_item["passed_test_list"]:
            if test in eval_data_item["ekstazi_test_list"]:
                ekstazi_labels.append(1)
            else:
                ekstazi_labels.append(0)
            if test in eval_data_item["starts_test_list"]:
                starts_labels.append(1)
            else:
                starts_labels.append(0)
            prediction_score = 0
            for index, row in eval_data_df.iterrows():
                if row["id"] == commit and row["test name"].split(".")[-1] == test:
                    prediction_score = y_pred_one[index]
            prediction_scores.append(prediction_score)

        labels = [1 for _ in range(len(failed_test_list))] + [0 for _ in range(len(passed_test_list))]
        res_item["labels"] = labels
        res_item["prediction_scores"] = prediction_scores
        res_item["commit"] = eval_data_item["commit"]
        res_item["Ekstazi_labels"] = ekstazi_labels
        res_item["STARTS_labels"] = starts_labels
        res.append(res_item)

    output_dir = Macros.results_dir / "modelResults" / proj / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    IOUtils.dump(output_dir / "per-sha-result.json", res, IOUtils.Format.jsonPretty)


def run_EALRTS_baseline_for_each_sha(project: str, model_type: str, eval_data_item, subset="All"):
    failure_rate_dict = {}
    if not os.path.exists(f"{Macros.model_data_dir}/EALRTS/{project}/reducedData.txt"):
        # this project does not have training data
        raise RuntimeError(f"{project} does not have training data")
    reduced_data = pd.read_csv(f"{Macros.model_data_dir}/EALRTS/{project}/reducedData.txt", sep=",", header=None,
                               error_bad_lines=False)
    with open(f"{Macros.model_data_dir}/EALRTS/{project}/finalData.txt") as infile:
        for index, line in enumerate(infile):
            line = line.rstrip("\n")
            test_name = line.split(",")[8].strip()
            failure_rate = reduced_data.iat[index, 7]
            failure_rate_dict[test_name] = failure_rate

    sha_data_list, compile_time = EALRTSProcessor().process_eval_for_each_SHA(project, eval_data_item, failure_rate_dict)
    # load eval data file and saved model
    # eval_data_df = pd.read_csv(preprocessed_eval_data_path, sep=",", header=None)
    eval_data_df_columns = ["id", "test runs", "File cardinality", "Target cardinality", "minimal distance",
                            "connected tests", "change history", "failure rate", "failed test", "test name"]
    eval_data_df = pd.DataFrame(sha_data_list, columns=eval_data_df_columns)
    if eval_data_df.empty:
        res_item = {}
        res_item["labels"] = []
        res_item["prediction_scores"] = []
        res_item["commit"] = eval_data_item["commit"]
        res_item["Ekstazi_labels"] = []
        res_item["STARTS_labels"] = []
        return res_item, compile_time
    saved_model_path = f"{Macros.model_data_dir}/EALRTS/{project}/{model_type}.pkl"
    saved_model = pickle.load(open(saved_model_path, 'rb'))

    if model_type == "randomforest":
        x_test = eval_data_df.drop(["id", "failed test", "File cardinality", "test name"], axis=1)
    elif model_type == "xgboost":
        x_test = eval_data_df.drop(["id", "failed test", "test name"], axis=1)
    y_pred = saved_model.predict_proba(x_test)
    y_pred_one = y_pred[:, 1].tolist()

    res_item = {}
    if subset == "All":
        test_list = eval_data_item["failed_test_list"] + eval_data_item["passed_test_list"]
    elif subset == "Ekstazi":
        test_list = eval_data_item["ekstazi_test_list"]
    elif subset == "STARTS":
        test_list = eval_data_item["starts_test_list"]
    else:
        raise NotImplementedError

    commit = eval_data_item["commit"]
    failed_test_list = eval_data_item["failed_test_list"]
    passed_test_list = eval_data_item["passed_test_list"]
    ekstazi_labels = []
    starts_labels = []

    prediction_scores = []
    for test in test_list:
        if test in eval_data_item["ekstazi_test_list"]:
            ekstazi_labels.append(1)
        else:
            ekstazi_labels.append(0)
        if test in eval_data_item["starts_test_list"]:
            starts_labels.append(1)
        else:
            starts_labels.append(0)
        prediction_score = 0
        for index, row in eval_data_df.iterrows():
            if row["id"] == commit and row["test name"].split(".")[-1] == test:
                prediction_score = y_pred_one[index]
        prediction_scores.append(prediction_score)

    labels = []
    for test in test_list:
        if test in failed_test_list:
            labels.append(1)
        else:
            labels.append(0)
    res_item["labels"] = labels
    res_item["prediction_scores"] = prediction_scores
    res_item["commit"] = eval_data_item["commit"]
    res_item["Ekstazi_labels"] = ekstazi_labels
    res_item["STARTS_labels"] = starts_labels
    return res_item, compile_time
