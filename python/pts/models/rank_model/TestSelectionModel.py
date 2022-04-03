import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List
from pts.main import proj_logs

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
from pts.processor.data_utils.SubTokenizer import SubTokenizer
from pts.processor.RankProcess import RankProcessor
from pts.Macros import Macros
from pts.models.rank_model.CodeEmbeddingStore import CodeEmbeddingStore
from pts.models.rank_model.DiffTestModel import DiffTestModel
from pts.models.rank_model.utils import DiffPairBM25BatchData, DiffTestBM25BatchData, DiffTestBatchData, DiffPairBatchData, read_data_from_file, hinge_loss, \
    compute_score, find_missing_tests
from pts.collector.eval_data_collection import remove_comment

def threshold_valid_loss(scores):
    """
    Way to get validation results for early stopping, this approach we assume the positive results should be
    greater than 0.5 score.
    """
    labels = []
    for _ in range(scores.shape[0]):
        labels.append(1)
        labels.append(0)
    scores = scores.flatten().detach().cpu().numpy()
    preds = np.zeros(scores.size)
    preds[scores > 0.5] = 1
    return labels, preds.tolist()

def pairwise_rank_loss(scores):
    """
    Use the pairwise rank to get valid loss for early stopping
    """
    return [0 for _ in range(scores.shape[0])], scores.argmax(-1).tolist()

def get_first_fail_selection_rate(model_saving_dir: Path, test_data_dir: Path):
    """
    Get the 'perfect' selection rate to get recal 1.0
    First get the lowest threshold for all the SHA:mutant, and use this threshold for all to get the average
    selection rate.
    """
    model = load_model(model_saving_dir)
    model.mode = "test"

    sha_2_first_fail_test_threshold: dict = IOUtils.load(
        model.model_results_dir / "first-fail-test-threshold-per-SHA.json")
    first_threshold = min(sha_2_first_fail_test_threshold.values())

    result_dict = model.run_evaluation(test_data_file=test_data_dir, threshold=first_threshold, save=False,
                                       return_result=True)
    best_selection_rate = {"first-fail-test-selection-rate": result_dict["selected_pct"]}
    IOUtils.dump(model.model_results_dir / "first-fail-test-selection-rate.json", best_selection_rate,
                 IOUtils.Format.jsonNoSort)

def get_safe_selection_rate(model_saving_dir: Path, test_data_dir: Path):
    """
    Get the 'perfect' selection rate to get recal 1.0
    First get the lowest threshold for all the SHA:mutant, and use this threshold for all to get the average
    selection rate.
    """
    model = load_model(model_saving_dir)
    model.mode = "test"

    sha_2_lowest_threshold: dict = IOUtils.load(model.model_results_dir / "lowest-threshold-per-SHA.json")
    most_safe_threshold = min(sha_2_lowest_threshold.values())

    result_dict = model.run_evaluation(test_data_file=test_data_dir, threshold=most_safe_threshold, save=False,
                                       return_result=True)
    best_selection_rate = {"best-safe-selection-rate": result_dict["selected_pct"]}
    IOUtils.dump(model.model_results_dir / "best-safe-selection-rate.json", best_selection_rate,
                 IOUtils.Format.jsonNoSort)


def eval_model(model_saving_dir: Path, model_data_dir: Path):
    """Caller function to load model and run model evaluation."""
    model = load_model(model_saving_dir)
    model.mode = "test"
    test_data_dir = model_data_dir / "test.json"
    model.model_results_dir = model_data_dir / "results"
    model.run_evaluation(test_data_file=test_data_dir)


def eval_model_per_sha(model_saving_dir: Path, test_data_dir: Path):
    """Caller function to load model and eval model per sha."""
    model = load_model(model_saving_dir)
    model.mode = "test"
    model.evaluate_per_sha(model, test_data_file=test_data_dir)



def load_model(model_path):
    """Loads a pretrained model from model_path."""
    print('Loading model from: {}'.format(model_path))
    sys.stdout.flush()
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


def calculate_apfd(fail_test_rank: List[int], num_of_test: int):
    """Return APFD for a given test ordering."""
    try:
        apfd = 1 - sum(fail_test_rank) / (num_of_test * len(fail_test_rank)) + 1 / (2 * num_of_test)
        return round(apfd, 4)
    except Exception as e:
        print(e)
        return 0


class TestSelectionModel(nn.Module):
    """Predict confidence score for a given code diff and test."""

    def __init__(self, config_file, model_data_dir: Path):
        super(TestSelectionModel, self).__init__()

        # load config file
        base_config_file = Macros.config_dir / config_file
        self.config = IOUtils.load(base_config_file, IOUtils.Format.jsonPretty)
        self.model_data_dir = model_data_dir
        self.train_add_data_file = None
        if "triplet" in self.config:
            self.train_add_data_file = self.model_data_dir / "train_add.json"
        self.train_data_file = self.model_data_dir / "train.json"
        self.valid_data_file = self.model_data_dir / "valid.json"
        self.test_data_file = self.model_data_dir / "test.json"
        # load hyper-param
        self.load_hyper_params()

        # set up logging
        logging_file = self.model_data_dir / "model.log"
        LoggingUtils.setup(filename=str(logging_file))
        self.logger = LoggingUtils.get_logger(__name__, LoggingUtils.INFO)
        self.logger.info("===> initializing the model ...")
        if "additional_features" in self.config:
            self.additional_features = self.config["additional_features"]
            self.output_layer = nn.Linear(self.last_layer_dim+1, 1)
        else:
            self.additional_features = None
            self.output_layer = nn.Linear(self.last_layer_dim, 1)
        # create vocab
        self.create_vocab()
        # set up sub-modules
        self.diff_test_model = DiffTestModel(self.config, self.embed_size, self.embedding_store, self.hidden_size,
                                             self.cross_feature_size, self.encoder_layers, self.dropout, self.num_heads)
        self.logger.info(repr(
            self.diff_test_model))  # feature_dim = self.config["cross_feature_dim"] + self.config["test_feature_dim"]
        
        
        self.rankLoss = nn.MarginRankingLoss(1.0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.sigmoid = nn.Sigmoid()
        # Create dir for the model
        IOUtils.mk_dir(self.model_data_dir / "saved_models")
        IOUtils.mk_dir(self.model_data_dir / "results")
        self.model_save_dir = self.model_data_dir / "saved_models"
        self.model_results_dir = self.model_data_dir / "results"

    def load_hyper_params(self):
        self.learning_rate = self.config["learning_rate"]
        self.embed_size = self.config["embedding_size"]
        self.hidden_size = self.config["hidden_size"]
        self.encoder_layers = self.config["layers"]
        self.max_epoch = self.config["max_epochs"]
        self.patience = self.config["patience"]
        self.dropout = self.config["dropout"]
        self.num_heads = self.config["num_heads"]
        self.cross_feature_size = self.config["cross_feature_size"]
        self.last_layer_dim = self.config["last_layer_dim"]
        if "triplet" in self.config:
            self.margin = self.config["margin_large"]
            self.small_margin = self.config["margin_small"]
        else:
            self.margin = self.config["margin"]
        self.torch_device_name = self.config["device_name"]
        self.diff_features: List = self.config["diff_features"]
        self.test_features: List = self.config["test_features"]
        self.batch_size = self.config["batch_size"]

    def create_vocab(self):
        self.code_token_counter = Counter()
        code_lengths = []

        train_data = IOUtils.load_json_stream(self.train_data_file)
        valid_data = IOUtils.load(self.valid_data_file)

        print("===> creating vocabs ...")
        self.logger.info("===> creating vocabs ...")
        for dt in train_data:
            code_diff = []
            for feature in self.diff_features:
                try:
                    code_diff.extend(dt[feature])
                except:
                    pass
            self.code_token_counter.update(code_diff)
            code_lengths.append(len(code_diff))

            pos_test_code = []
            for feature in self.test_features:
                pos_test_code.extend(dt[f"pos_{feature}"])
            self.code_token_counter.update(pos_test_code)
            code_lengths.append(len(pos_test_code))

            neg_test_code = []
            for feature in self.test_features:
                neg_test_code.extend(dt[f"neg_{feature}"])
            self.code_token_counter.update(neg_test_code)
            code_lengths.append(len(neg_test_code))
        # end for

        if self.train_add_data_file:
            train_add_data = IOUtils.load_json_stream(self.train_add_data_file)
            for dt in train_add_data:
                code_diff = []
                for feature in self.diff_features:
                    try:
                        code_diff.extend(dt[feature])
                    except:
                        pass
                self.code_token_counter.update(code_diff)
                code_lengths.append(len(code_diff))

                pos_test_code = []
                for feature in self.test_features:
                    pos_test_code.extend(dt[f"pos_{feature}"])
                self.code_token_counter.update(pos_test_code)
                code_lengths.append(len(pos_test_code))

                neg_test_code = []
                for feature in self.test_features:
                    neg_test_code.extend(dt[f"neg_{feature}"])
                self.code_token_counter.update(neg_test_code)
                code_lengths.append(len(neg_test_code))
            # end for

        code_counts = np.asarray(sorted(self.code_token_counter.values()))
        code_threshold = int(np.percentile(code_counts, self.config["vocab_cut_off_pct"])) + 1
        self.max_code_length = int(np.percentile(np.asarray(sorted(code_lengths)),
                                                 self.config["length_cut_off_pct"]))
        self.embedding_store = CodeEmbeddingStore(code_threshold, self.config["embedding_size"],
                                                  self.code_token_counter,
                                                  self.config["dropout"])

    def get_device(self):
        """Returns the proper device."""
        if self.torch_device_name == 'gpu':
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def forward(self, batch_data, additional_features=None):
        pos_output, neg_output = self.diff_test_model.forward(batch_data, self.get_device())  # BS, output_size

        pos_logits = self.sigmoid(self.output_layer(pos_output))
        neg_logits = self.sigmoid(self.output_layer(neg_output))
        # convert feature vectors to probabilities
        if self.mode == "test":
            return pos_logits, neg_logits
        # Try: without log_softmax
        # scores = torch.cat((pos_logits, neg_logits), -1)  # BS, 2
        # log_prob = torch.nn.functional.log_softmax(scores, dim=-1)
        # pos_log_prob, neg_log_prob = log_prob[:, 0], log_prob[:, 1]
        return pos_logits[:, 0], neg_logits[:, 0]

    def create_pair_batches(self, mode="train", batch_size=32, shuffle=True, dataset=None):
        if "batch_size" in self.__dict__.keys():
            batch_size = self.batch_size
        batches = []
        if mode == "train":
            objs = IOUtils.load_json_stream(self.train_data_file)
            dataset = []
            for obj in objs:
                dataset.append(obj)
            if shuffle:
                random.shuffle(dataset)
        elif mode == "train-add":
            objs = IOUtils.load_json_stream(self.train_add_data_file)
            dataset = []
            for obj in objs:
                dataset.append(obj)
            if shuffle:
                random.shuffle(dataset)
        elif mode == "valid":
            objs = IOUtils.load_json_stream(self.valid_data_file)
            dataset = []
            for obj in objs:
                dataset.append(obj)
            if shuffle:
                random.shuffle(dataset)
        elif mode == "test":
            if not dataset:
                dataset = read_data_from_file(self.test_data_file)

        curr_idx = 0
        while curr_idx < len(dataset):
            start_idx = curr_idx
            end_idx = min(start_idx + batch_size, len(dataset))

            code_diff_token_ids = []  # length: batch size
            code_diff_lengths = []
            pos_test_token_ids = []
            pos_test_lengths = []
            neg_test_token_ids = []
            neg_test_lengths = []

            pos_bm25_scores = []
            neg_bm25_scores = []

            labels = []
            ekstazi_labels = []
            starts_labels = []

            for i in range(start_idx, end_idx):
                # diff features
                code_diff = []
                for feature in self.diff_features:
                    code_diff.extend(dataset[i][feature])
                code_diff_ids = self.embedding_store.get_padded_code_ids(code_diff, self.max_code_length)
                code_diff_length = min(len(code_diff), self.max_code_length)
                code_diff_token_ids.append(code_diff_ids)
                code_diff_lengths.append(code_diff_length)
                # test code
                pos_test_code = []
                for feature in self.test_features:
                    pos_test_code.extend(dataset[i][f"pos_{feature}"])
                pos_test_code_ids = self.embedding_store.get_padded_code_ids(pos_test_code, self.max_code_length)
                pos_test_code_length = min(len(pos_test_code), self.max_code_length)
                pos_test_token_ids.append(pos_test_code_ids)
                pos_test_lengths.append(pos_test_code_length)

                neg_test_code = []
                for feature in self.test_features:
                    neg_test_code.extend(dataset[i][f"neg_{feature}"])
                neg_test_code_ids = self.embedding_store.get_padded_code_ids(neg_test_code, self.max_code_length)
                neg_test_code_length = min(len(neg_test_code), self.max_code_length)
                neg_test_token_ids.append(neg_test_code_ids)
                neg_test_lengths.append(neg_test_code_length)

                # features
                if self.additional_features:
                    for f in self.additional_features:
                        if f == "bm25":
                            pos_bm25_scores.append(dataset[i]["pos_test_bm25"])
                            neg_bm25_scores.append(dataset[i]["neg_test_bm25"])

                if mode == "test":
                    label = dataset[i]["label"]
                    labels.append(label)
                    ekstazi_label = dataset[i]["ekstazi_label"]
                    ekstazi_labels.append(ekstazi_label)
                    starts_label = dataset[i]["starts_label"]
                    starts_labels.append(starts_label)
            if mode == "train" or mode == "valid":
                if self.additional_features:
                    batches.append(
                        DiffPairBM25BatchData(
                            torch.tensor(code_diff_token_ids, dtype=torch.int64, device=self.get_device()),
                            torch.tensor(code_diff_lengths, dtype=torch.int64, device=self.get_device()),
                            torch.tensor(pos_test_token_ids, dtype=torch.int64, device=self.get_device()),
                            torch.tensor(pos_test_lengths, dtype=torch.int64, device=self.get_device()),
                            torch.tensor(pos_bm25_scores, dtype=torch.float, device=self.get_device()),
                            torch.tensor(neg_test_token_ids, dtype=torch.int64, device=self.get_device()),
                            torch.tensor(neg_test_lengths, dtype=torch.int64, device=self.get_device()),
                            torch.tensor(neg_bm25_scores, dtype=torch.float, device=self.get_device()))
                    )
                else:
                    batches.append(
                        DiffPairBatchData(torch.tensor(code_diff_token_ids, dtype=torch.int64, device=self.get_device()),
                        torch.tensor(code_diff_lengths, dtype=torch.int64, device=self.get_device()),
                        torch.tensor(pos_test_token_ids, dtype=torch.int64, device=self.get_device()),
                        torch.tensor(pos_test_lengths, dtype=torch.int64, device=self.get_device()),
                        torch.tensor(neg_test_token_ids, dtype=torch.int64, device=self.get_device()),
                        torch.tensor(neg_test_lengths, dtype=torch.int64, device=self.get_device()))
                    )
            elif mode == "test":
                if self.additional_features:
                    batches.append(DiffTestBM25BatchData(torch.tensor(code_diff_token_ids, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(code_diff_lengths, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(pos_test_token_ids, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(pos_test_lengths, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(pos_bm25_scores, dtype=torch.float, device=self.get_device()),
                                      torch.tensor(neg_test_token_ids, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(neg_test_lengths, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(neg_bm25_scores, dtype=torch.float, device=self.get_device()),
                                      torch.tensor(labels, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(ekstazi_labels, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(starts_labels, dtype=torch.int64, device=self.get_device()))

                    )
                else:
                    batches.append(
                        DiffTestBatchData(torch.tensor(code_diff_token_ids, dtype=torch.int64, device=self.get_device()),
                                          torch.tensor(code_diff_lengths, dtype=torch.int64, device=self.get_device()),
                                          torch.tensor(pos_test_token_ids, dtype=torch.int64, device=self.get_device()),
                                          torch.tensor(pos_test_lengths, dtype=torch.int64, device=self.get_device()),
                                          torch.tensor(neg_test_token_ids, dtype=torch.int64, device=self.get_device()),
                                          torch.tensor(neg_test_lengths, dtype=torch.int64, device=self.get_device()),
                                          torch.tensor(labels, dtype=torch.int64, device=self.get_device()),
                                          torch.tensor(ekstazi_labels, dtype=torch.int64, device=self.get_device()),
                                          torch.tensor(starts_labels, dtype=torch.int64, device=self.get_device()))
                    )
            curr_idx = end_idx
        return batches

    def run_gradient_step(self, batch_data, margin="large"):
        self.optimizer.zero_grad()
        pos_log_prob, neg_log_prob = self.forward(batch_data)
        if margin == "small":
            loss = hinge_loss(pos_log_prob, neg_log_prob, self.small_margin)
        else:
            loss = hinge_loss(pos_log_prob, neg_log_prob, self.margin)
        loss.backward()
        self.optimizer.step()
        return float(loss.cpu())

    def run_train(self):
        """Train the prediction model"""
        self.logger.info("===> start to train the model ... ")
        # load train and valid data
        valid_batches = self.create_pair_batches(mode="valid")
        if self.torch_device_name == "gpu":
            self.cuda()
        best_loss = float('inf')
        best_prec = 0
        patience_tally = 0

        # start training
        for epoch in range(self.max_epoch):
            if patience_tally > self.patience:
                print('Terminating')
                self.logger.info("===> terminating the training ...")
                break
            self.train()
            train_batches = self.create_pair_batches(mode="train")

            train_loss = 0

            for batch_iter, batch_data in enumerate(train_batches):
                train_loss += self.run_gradient_step(batch_data)

            if self.train_add_data_file:
                train_add_batches = self.create_pair_batches(mode="train-add")
                for batch_iter, batch_data in enumerate(train_add_batches):
                    train_loss += self.run_gradient_step(batch_data, margin="small")
            # end if
            # do validation at each epoch
            self.eval()
            validation_loss = 0
            validation_predictions = []
            validation_labels = []
            with torch.no_grad():
                for batch_data in valid_batches:
                    # validation loss
                    pos_score, neg_score = self.forward(batch_data)
                    valid_loss = hinge_loss(pos_score, neg_score, 0.2)
                    validation_loss += float(valid_loss.cpu())
                    scores = torch.cat((pos_score.unsqueeze(1), neg_score.unsqueeze(1)), -1)  # BS, 2
                    # labels, preds = pairwise_rank_loss(scores)
                    labels, preds = threshold_valid_loss(scores)
                    validation_labels.extend(labels)
                    validation_predictions.extend(preds)

            # validation_precision = (len(validation_labels) - sum(validation_predictions)) / len(validation_labels)
            validation_precision, validation_recall, validation_f1 = compute_score(
                validation_predictions, validation_labels, verbose=False)

            if validation_f1 > best_prec:
                torch.save(self, self.model_save_dir / "best_model")
                saved = True
                best_prec = validation_f1
                patience_tally = 0
            else:
                saved = False
                patience_tally += 1
            print('Epoch: {}'.format(epoch + 1))
            print('Training loss: {}'.format(train_loss / len(train_batches)))
            print('Validation loss: {}'.format(validation_loss))
            print(f"Validation precision is {validation_precision}, recall is {validation_recall}, f1 is {validation_f1}.")
            if saved:
                print('Saved')
            print('-----------------------------------')

    def aggregate_eval_data(self, test_data_list):
        """ Helper function to aggregate test data from each sha."""
        sha_2_data = defaultdict(list)
        for test_data_sha in test_data_list:
            sha_2_data[test_data_sha["sha"]].append(test_data_sha)

        self.logger.info(f"In total {len(sha_2_data)} shas in eval...")

        return sha_2_data

    def run_evaluation(self, select_rate=None, test_data_file=None, threshold=0.5, save=True, return_result=False, subset=None):
        """Predicts labels for the (diff, test) pairs."""
        self.eval()
        self.mode = "test"
        if test_data_file:
            self.test_data_file = test_data_file

        test_data_list = read_data_from_file(self.test_data_file)

        sha_2_data = self.aggregate_eval_data(test_data_list)

        all_predictions = []
        all_labels = []
        all_ekstazi_labels = []
        all_starts_labels = []
        sha_2_time = []

        selected_failed_tests_num = 0
        missed_failed_tests_num = 0

        out_of_scope_test = 0
        in_scope_test = 0

        recall_per_sha = []
        precision_per_sha = []
        f1_per_sha = []
        eks_recall_per_sha = []
        starts_recall_per_sha = []
        starts_subset_recall_per_sha = []
        ekstazi_subset_recall_per_sha = []

        # to save for further process
        prediction_per_sha = []

        for s, s_data in sha_2_data.items():
            test_batches = self.create_pair_batches(mode="test", dataset=s_data)

            num_of_candidate_tests = s_data[0]["num_test_class"]
            num_of_changed_files = s_data[0]["num_changed_files"]

            self.logger.info(f"Number of changed files in {s} is {num_of_changed_files}."
                             f"Number of candidate test classes is {num_of_candidate_tests}")

            s_pred_scores = []
            s_starts_labels = []
            s_labels = []
            s_ekstazi_labels = []

            # Start to do prediction in this SHA
            with torch.no_grad():
                start_time = time.time()
                for b, batch_data in enumerate(test_batches):
                    print(f"Testing SHA: {s}")
                    sys.stdout.flush()
                    pos_score, _ = self.forward(batch_data)
                    s_pred_scores.extend([element.item() for element in pos_score.flatten()])
                    s_labels.extend([element.item() for element in batch_data.label.flatten()])
                    s_starts_labels.extend([element.item() for element in batch_data.starts_label.flatten()])
                    s_ekstazi_labels.extend([element.item() for element in batch_data.ekstazi_label.flatten()])
                # end for
                run_time = time.time() - start_time
                self.logger.info(f"Running time to do prediction is {run_time} seconds.")
                sha_2_time.append(run_time)
            # end with
            prediction_scores = np.zeros(int(num_of_candidate_tests))
            for i in range(0, len(s_pred_scores), num_of_candidate_tests):
                tmp = np.array(s_pred_scores[i: i + num_of_candidate_tests])
                prediction_scores = np.maximum(prediction_scores, tmp)
            # num_files = len(s_pred_scores) / int(num_of_candidate_tests)

            # prediction_scores /= num_of_changed_files
            preds = np.zeros(num_of_candidate_tests)
            if select_rate is None:
                preds[prediction_scores >= threshold] = 1
            else:
                select_size = int(select_rate * num_of_candidate_tests)
                if select_size > 0:
                    preds[prediction_scores.argsort()[-select_size:]] = 1
            labels = s_labels[:num_of_candidate_tests] 
            ekstazi_labels = s_ekstazi_labels[:num_of_candidate_tests]
            starts_labels = s_starts_labels[:num_of_candidate_tests]
            preds = preds.tolist()
            # ad-hoc processing
            modified_test_class = [d["changed_class_name"] for d in s_data if "test" in d["changed_class_name"]]
            test_index = []
            for t in modified_test_class:
                test_index.extend(
                    [i for i, d in enumerate(s_data[:num_of_candidate_tests]) if d["pos_test_class"] == t])

            # To see how many failed are detected and how many are not
            for i, p in enumerate(preds):
                if p > 0 and labels[i] > 0:
                    selected_failed_tests_num += 1
                elif labels[i] > 0:
                    missed_failed_tests_num += 1
            # end for
            p, r, f = compute_score(predicted_labels=preds, gold_labels=labels)
            if r < 1:
                project_name = str(test_data_file).split('/')[-3]
                for k in proj_logs:
                    if k.split('_')[1] == project_name:
                        project = k
                t1, t2 = find_missing_tests(preds, labels, s_data, project)
                out_of_scope_test += t1
                in_scope_test += t2

            if select_rate is not None:
                # Add the results of models selection intersecting STARTS
                model_starts_preds = []
                starts_selected_labels = []
                for i in range(len(starts_labels)):
                    if starts_labels[i] == 1:
                        starts_selected_labels.append(labels[i])
                        model_starts_preds.append(prediction_scores[i])
                # end for
                select_size = int(select_rate * num_of_candidate_tests)
                model_starts_preds = np.array(model_starts_preds)
                starts_subset_preds = np.zeros(len(starts_selected_labels))
                if select_size > 0:
                    starts_subset_preds[model_starts_preds.argsort()[-select_size:]] = 1
                # end if
                if sum(starts_selected_labels) == 0:
                    pass
                else:
                    starts_subset_recall_per_sha.append(recall_score(starts_selected_labels, starts_subset_preds))

                # Add the results of models selection intersecting Ekstazi
                model_ekstazi_preds = []
                ekstazi_selected_labels = []
                for i in range(len(ekstazi_labels)):
                    if ekstazi_labels[i] == 1:
                        ekstazi_selected_labels.append(labels[i])
                        model_ekstazi_preds.append(prediction_scores[i])
                # end for
                select_size = int(select_rate * num_of_candidate_tests)
                model_ekstazi_preds = np.array(model_ekstazi_preds)
                ekstazi_subset_preds = np.zeros(len(ekstazi_selected_labels))
                if select_size > 0:
                   ekstazi_subset_preds[model_ekstazi_preds.argsort()[-select_size:]] = 1
                # end if
                if sum(ekstazi_selected_labels) == 0:
                    pass
                else:
                    ekstazi_subset_recall_per_sha.append(recall_score(ekstazi_selected_labels, ekstazi_subset_preds))
                    
            recall_per_sha.append(recall_score(labels, preds))
            precision_per_sha.append(precision_score(labels, preds))
            eks_recall_per_sha.append((recall_score(labels, ekstazi_labels)))
            starts_recall_per_sha.append((recall_score(labels, starts_labels)))
            f1_per_sha.append(f1_score(labels, preds))

            all_predictions.extend(preds)
            all_ekstazi_labels.extend(ekstazi_labels)
            all_starts_labels.extend(starts_labels)
            all_labels.extend(labels)

            prediction_per_sha.append({
                "commit": s,
                "prediction_scores": prediction_scores.tolist(),
                "Ekstazi_labels": ekstazi_labels,
                "STARTS_labels": starts_labels,
                "labels": labels
            })

        if subset == "STARTS":
            return sum(starts_subset_recall_per_sha)/len(starts_subset_recall_per_sha)
        if subset == "Ekstazi":
            return sum(ekstazi_subset_recall_per_sha)/len(ekstazi_subset_recall_per_sha)

        compute_score(predicted_labels=all_predictions, gold_labels=all_labels)
        prec = sum(precision_per_sha) / len(precision_per_sha)
        rec = sum(recall_per_sha) / len(recall_per_sha)
        f1 = sum(f1_per_sha) / len(f1_per_sha)

        ek_prec = 0
        ek_rec = sum(eks_recall_per_sha) / len(eks_recall_per_sha)
        ek_f1 = 0

        sts_prec = 0
        sts_rec = sum(starts_recall_per_sha) / len(starts_recall_per_sha)
        sts_f1 = 0

        # average selected test class
        model_sel_num = sum(all_predictions)
        ek_sel_num = sum(all_ekstazi_labels)
        sts_sel_num = sum(all_starts_labels)
        total_num = len(all_predictions)
        if in_scope_test+out_of_scope_test == 0:
            pct = -1
        else:
            pct = out_of_scope_test/(in_scope_test+out_of_scope_test)
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
            "total_missed_failed_tests": missed_failed_tests_num,
            "total_selected_failed_tests": selected_failed_tests_num,
            "pct_newly_add_missed_failed_tests": pct
        }

        if save:  # will change to per_file later if we use other approach to do prediction
            IOUtils.mk_dir(self.model_results_dir / "test-output")
            IOUtils.dump(self.model_results_dir / "test-output" / "per-file-result.json", result,
                         IOUtils.Format.jsonPretty)
            IOUtils.dump(self.model_results_dir / "running-time-per-sha.json", sha_2_time)
            print(result)
            IOUtils.dump(self.model_results_dir / f"per-sha-result.json", prediction_per_sha, IOUtils.Format.json)
        elif return_result:
            return result
        else:
            return rec
