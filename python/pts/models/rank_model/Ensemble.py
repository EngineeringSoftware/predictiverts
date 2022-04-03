import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import joblib
import time
import ipdb
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from seutil import IOUtils, LoggingUtils
from collections import Counter
from sklearn.metrics import recall_score, precision_score, f1_score

from pts.Macros import Macros
from pts.models.rank_model.CodeEmbeddingStore import CodeEmbeddingStore
from pts.models.rank_model.DiffTestModel import DiffTestModel
from pts.models.rank_model.utils import DiffTestBatchData, DiffPairBatchData, read_data_from_file, hinge_loss, compute_score

class Ensemble:
    """ This class ensemble two or more models to do the test selection."""

    def __init__(self, sub_models: List[str], project: str):
        self.sub_models = {k: None for k in sub_models}
        self.proj_name = project.split('_')[1]

    def load_model(self, model_path: Path):
        """Loads a pretrained model from model_path."""
        print('Loading model from: {}'.format(model_path))
        if torch.cuda.is_available():
            model = torch.load(model_path)
            model.torch_device_name = 'gpu'
            model.cuda()
            for c in model.children():
                c.cuda()
        else:
            model = torch.load(model_path, map_location='cpu')
            model.torch_device_name = 'cpu'
            model.cpu()
            for c in model.children():
                c.cpu()
        return model

    def run_evaluation(self):
        # load the trained sub models
        for m in self.sub_models:
            model_dir = Macros.model_data_dir / "rank-model" / self.proj_name / m / "saved_models" / "best_model"
            self.sub_models[m] = self.load_model(model_dir)
            self.sub_models[m].mode = "test"
            self.sub_models[m].eval()
        # end for

        test_data_file = Macros.model_data_dir / "rank-model" / self.proj_name  / "test.json"

        test_data_list = read_data_from_file(test_data_file)

        sha_2_data = defaultdict(list)
        for test_data_sha in test_data_list:
            if "sha" not in test_data_sha:
                print(test_data_sha)
            sha_2_data[test_data_sha["sha"]].append(test_data_sha)

        all_predictions = []
        all_labels = []
        all_ekstazi_labels = []
        all_starts_labels = []

        model_pred_scores = {}

        for s, s_data in sha_2_data.items():
            # HARD CODE for now !!!!
            aggregate_prediction_scores = np.zeros(30)
            model_pred_scores[s] = {}
            # get the predictions scores for each model per sha
            for m, model in self.sub_models.items():
                test_batches = model.create_pair_batches(mode="test", dataset=s_data)
                num_of_changed_files = len(set(["".join(t["code_diff"]) for t in s_data]))

                print(f"Number of changed files: {num_of_changed_files}")

                s_pred_scores = []
                s_starts_labels = []
                s_labels = []
                s_ekstazi_labels = []

                with torch.no_grad():
                    for b, batch_data in enumerate(test_batches):
                        print(f"Testing SHA: {s}")
                        sys.stdout.flush()
                        pos_score, _ = model.forward(batch_data)
                        s_pred_scores.extend([element.item() for element in pos_score.flatten()])
                        s_labels.extend([element.item() for element in batch_data.label.flatten()])
                        s_starts_labels.extend([element.item() for element in batch_data.starts_label.flatten()])
                        s_ekstazi_labels.extend([element.item() for element in batch_data.ekstazi_label.flatten()])
                    # end for
                # end with
                num_of_candidate_tests = int(len(s_pred_scores) / num_of_changed_files)

                print(f"Num of tests is {num_of_candidate_tests}.")
                prediction_scores = np.zeros(int(num_of_candidate_tests))
                for i in range(0, len(s_pred_scores), num_of_candidate_tests):
                    tmp = np.array(s_pred_scores[i: i + num_of_candidate_tests])
                    prediction_scores += tmp
                num_files = len(s_pred_scores) / int(num_of_candidate_tests)
                # prediction_scores /= num_files
                model_pred_scores[s][m] = prediction_scores
                aggregate_prediction_scores += prediction_scores
            # end for
            preds = np.zeros(num_of_candidate_tests)
            # Aggregate the scores for all the models for this SHA, currently the threshold is 0.5
            # aggregate_prediction_scores /= len(self.sub_models)
            preds[aggregate_prediction_scores > 1.2] = 1
            labels = s_labels[:num_of_candidate_tests]
            # ipdb.set_trace()

            ekstazi_labels = s_ekstazi_labels[:num_of_candidate_tests]
            starts_labels = s_starts_labels[:num_of_candidate_tests]
            compute_score(predicted_labels=preds, gold_labels=labels)

            all_predictions.extend(preds)
            all_ekstazi_labels.extend(ekstazi_labels)
            all_starts_labels.extend(starts_labels)
            all_labels.extend(labels)

        compute_score(predicted_labels=all_predictions, gold_labels=all_labels)

        prec = precision_score(all_labels, all_predictions)
        rec = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        ek_prec = precision_score(all_ekstazi_labels, all_predictions)
        ek_rec = recall_score(all_ekstazi_labels, all_predictions)
        ek_f1 = f1_score(all_ekstazi_labels, all_predictions)

        sts_prec = precision_score(all_starts_labels, all_predictions)
        sts_rec = recall_score(all_starts_labels, all_predictions)
        sts_f1 = f1_score(all_starts_labels, all_predictions)

        # average selected test class
        model_sel_num = sum(all_predictions)
        ek_sel_num = sum(all_ekstazi_labels)
        sts_sel_num = sum(all_starts_labels)
        total_num = len(all_predictions)

        result = {
            "precision": 100 * prec,
            "recall": 100 * rec,
            "f1": f1,
            "selected_pct": float(model_sel_num) / total_num,
            "ekstazi_precision": 100 * ek_prec,
            "ekstazi_recall": 100 * ek_rec,
            "ekstazi_f1": 100 * ek_f1,
            "ekstazi_selected_pct": float(ek_sel_num) / total_num,
            "starts_precision": 100 * sts_prec,
            "starts_recall": 100 * sts_rec,
            "starts_f1": 100 * sts_f1,
            "starts_selected_pct": float(sts_sel_num) / total_num,
        }

        print(result)
