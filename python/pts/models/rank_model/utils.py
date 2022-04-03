import os

import torch
from typing import List, NamedTuple
import json
from seutil import IOUtils
from pts.Macros import Macros

SUREFILE = "target/surefire-reports"


class DiffTestBatchData(NamedTuple):
    """Stores tensorized batch used in test prediction model."""
    code_seqs: torch.Tensor
    code_lengths: torch.Tensor
    pos_test_seqs: torch.Tensor
    pos_test_lengths: torch.Tensor
    neg_test_seqs: torch.Tensor
    neg_test_lengths: torch.Tensor
    label: torch.Tensor
    ekstazi_label: torch.Tensor
    starts_label: torch.Tensor


class DiffPairBatchData(NamedTuple):
    """Stores tensorized batch used in test prediction model."""
    code_seqs: torch.Tensor
    code_lengths: torch.Tensor
    pos_test_seqs: torch.Tensor
    pos_test_lengths: torch.Tensor
    neg_test_seqs: torch.Tensor
    neg_test_lengths: torch.Tensor

class DiffTestBM25BatchData(NamedTuple):
    """Stores tensorized batch used in test prediction model."""
    code_seqs: torch.Tensor
    code_lengths: torch.Tensor
    pos_test_seqs: torch.Tensor
    pos_test_lengths: torch.Tensor
    pos_bm25: torch.Tensor
    neg_test_seqs: torch.Tensor
    neg_test_lengths: torch.Tensor
    neg_bm25: torch.Tensor
    label: torch.Tensor
    ekstazi_label: torch.Tensor
    starts_label: torch.Tensor

class DiffPairBM25BatchData(NamedTuple):
    code_seqs: torch.Tensor
    code_lengths: torch.Tensor
    pos_test_seqs: torch.Tensor
    pos_test_lengths: torch.Tensor
    pos_bm25: torch.Tensor
    neg_test_seqs: torch.Tensor
    neg_test_lengths: torch.Tensor
    neg_bm25: torch.Tensor

def read_data_from_file(filename):
    """Reads in data in the format used for model."""
    with open(filename) as f:
        data = json.load(f)
    return data


def hinge_loss(S_pos, S_neg, hinge_margin):
    """ calculate the hinge loss
        S_pos: pos score Variable (BS,)
        S_neg: neg score Variable (BS,)
        hinge_margin: hinge margin
        returns: batch-averaged loss value
    """
    cost = torch.mean((hinge_margin - (S_pos - S_neg)) *
                      ((hinge_margin - (S_pos - S_neg)) > 0).float())
    return cost


def compute_score(predicted_labels, gold_labels, verbose=True):
    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0

    assert (len(predicted_labels) == len(gold_labels))

    for i in range(len(gold_labels)):
        if gold_labels[i]:
            if predicted_labels[i]:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if predicted_labels[i]:
                false_positives += 1
            else:
                true_negatives += 1

    if verbose:
        print('True positives: {}'.format(true_positives))
        print('False positives: {}'.format(false_positives))
        print('True negatives: {}'.format(true_negatives))
        print('False negatives: {}'.format(false_negatives))

    try:
        precision = true_positives / (true_positives + false_positives)
    except:
        precision = 0.0
    try:
        recall = true_positives / (true_positives + false_negatives)
    except:
        recall = 0.0
    try:
        f1 = 2 * ((precision * recall) / (precision + recall))
    except:
        f1 = 0.0
    print(f"precision: {precision}, recall: {recall}, f1: {f1}")
    return precision, recall, f1


def get_test_from_surefile_reports(project: str):
    test_class_list = []
    with IOUtils.cd(Macros.repos_downloads_dir / project):
        subfolders = [x[0] for x in os.walk(".") if x[0].endswith(SUREFILE)]
        if not subfolders:
            return []
        for surefire_folder in subfolders:
            for test_log in os.listdir(surefire_folder):
                if test_log.endswith(".txt"):
                    test_class_list.append(test_log.split('.')[-2].lower())
        # end for
        return test_class_list


def find_missing_tests(preds, labels, sha_data, project):
    """Check the missing tests and see whether it exists in the training data."""
    out_of_scope_test = 0
    in_scope_test = 0
    test_class_list = get_test_from_surefile_reports(project)
    for i in range(len(preds)):
        if labels[i] == 1 and preds[i] != 1:
            test_class_name = "".join(sha_data[i]["pos_test_class"])
            if test_class_name not in test_class_list:
                out_of_scope_test += 1
            else:
                in_scope_test += 1
    return out_of_scope_test, in_scope_test
