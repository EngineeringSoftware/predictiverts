from typing import *

import random
from os import listdir
from pathlib import Path
from seutil import LoggingUtils, IOUtils, BashUtils
import numpy as np
from collections import defaultdict
import os

from pts.Environment import Environment
from pts.Macros import Macros
from pts.Utils import Utils


class DataProcessor:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    TRAIN_RATIO = 0.8
    TEST_RATIO = 0.2

    EXTENSIONS = [".java", ".xml", ".bin", ".yml", "other"]

    BACKTRACK = 5
    """
    {
        proj_name: str,
        raw_data:
        Feats: {
               - # commits made to files in the last 5 commits (scalar)
               - # files touched (scalar)
               - # distinct authors (scalar)
               - # failure rate for a test case in the last 5 commits (float)
               - # test cases in one test class 
               - extensions of changed files (one hot vector)
               - # common tokens in the path
    }
    """

    def process(self, **options):
        """Entry point to run different processors"""
        which = options.get("which")
        if which == "Rank":
            label_type = options.get("label_type", "Fail")
            from pts.processor.RankProcess import RankProcessor
            projs: List[str] = Utils.get_option_as_list(options, "project")
            processor = RankProcessor(
                Path(options.get("output_dir", Macros.data_dir / "model-data" / "rank-model")))
            typ = options.get("type", "train")
            if typ == "train":
                if label_type == "Fail":
                    processor.process_train_fail(projs)
                elif label_type == "Ekstazi":
                    processor.process_train_tools(projs)
            elif typ == "test":
                # currently hard-coded for ag data
                for proj in projs:
                    src_data_dir = Macros.eval_data_dir/ "mutated-eval-data" / f"{proj}-ag.json"
                    processor.process_eval(src_data_dir, proj)
        elif which == "Rank-bm25":
            from pts.processor.RankBM25Processor import RankBM25Processor
            label_type = options.get("label_type", "Fail")
            projs: List[str] = Utils.get_option_as_list(options, "project")
            processor = RankBM25Processor(
                Path(options.get("output_dir", Macros.data_dir / "model-data" / "rank-bm25-model")))
            typ = options.get("type", "train")
            if typ == "train":
                if label_type == "Fail":
                    processor.process_train_fail(projs)
                elif label_type == "STARTS":
                    processor.process_train_tools(projs)
                elif label_type == "triplet":
                    processor.process_triplet_labels(projs)
            elif typ == "test":
                for proj in projs:
                    src_data_dir = Macros.eval_data_dir / "mutated-eval-data" / f"{proj}-ag.json"
                    processor.process_eval(src_data_dir, proj)
        elif which == "DecisionTree":
            self.process_project_data(Path(options.get("output_dir")))
        elif which == "EALRTS":
            from pts.processor.EALRTSPreprocessor import EALRTSProcessor
            project = options.get("project")
            processor = EALRTSProcessor()
            processor.process_eval(Macros.eval_data_dir/"mutated-eval-data"/f"{project}-ag.json", project)  # zeroturnaround_zt-exec-ag.json
        else:
            raise NotImplemented

    def process_project_data(self, output_dir: Path):
        """Process the projects data file to get the dataset which can be used by the model"""
        data_dir = Macros.data_dir / "raw-data"
        proj_list = listdir(data_dir)
        projects = [Path(data_dir / proj) for proj in listdir(data_dir)]
        extension_dict = dict()
        for ext in self.EXTENSIONS:
            extension_dict[ext] = 0
        for proj_file, proj_name in zip(projects, proj_list):
            processed_data_list: List[dict] = list()
            raw_data_list = IOUtils.load(proj_file)
            num_data_point = len(raw_data_list)
            for index, data_point in enumerate(raw_data_list):
                if not data_point["min_dis"]:
                    continue
                # initialize values
                commits_history = 0
                cardinality = 0
                extension_dict = extension_dict.fromkeys(extension_dict, 0)
                test_case_fail_rates = dict()
                test_case_min_dis = defaultdict(lambda: float("inf"))
                test_case_common_tokens = defaultdict(lambda: 0)
                labels = defaultdict(lambda: 0)
                failed_test_list = data_point["failed_test_list"]
                passed_test_list = data_point["passed_test_list"]
                all_test_list = failed_test_list + passed_test_list
                # extract extension feature
                extensions = set([os.path.splitext(f)[1] for f in data_point["changed_files"]])
                # extract triggered test classes
                # triggered_test = data_point["num_target"]
                # if len(triggered_test) != 0:
                #     for k,v in triggered_test.items():
                #         cardinality += v
                cardinality = data_point["num_target"]
                # end for
                # end if

                for ext in extensions:
                    if ext != "":
                        if ext in self.EXTENSIONS:
                            extension_dict[ext] += 1
                        else:
                            extension_dict["other"] += 1
                # label the data and extract common tokens in path
                for test_case in all_test_list:
                    if test_case in failed_test_list:
                        labels[test_case] = 1
                    else:
                        labels[test_case] = 0
                    # end if
                    test_case_common_tokens[test_case] = sum([len(set(test_case.split(".")).
                                                                  intersection(set(file.split("/"))))
                                                              for file in data_point["changed_files"]])
                    test_case_fail_rates[test_case] = 0
                # end for
                # extract min distance
                for test_class in failed_test_list + passed_test_list:
                    test_case_min_dis[test_class] = float("inf")
                for _, dis in data_point["min_dis"].items():
                    for test_class, distance in dis.items():
                        if distance != -1 and distance < test_case_min_dis[test_class]:
                            test_case_min_dis[test_class] = distance
                        # end if
                    # end for
                # end for
                for k, v in test_case_min_dis.items():
                    if v == float("inf"):
                        test_case_min_dis[k] = -1
                    # end if
                # end for

                # get test fail rate and changed files history
                if index + self.BACKTRACK < num_data_point:
                    check_indexes = [index + 1 + i for i in range(self.BACKTRACK)]
                else:
                    check_indexes = range(index + 1, num_data_point)
                for t_index in check_indexes:
                    if set(data_point["changed_files"]) & set(raw_data_list[t_index]["changed_files"]):
                        commits_history += 1
                    intersection_failed_test_cases = list(
                        set(all_test_list).intersection(set(raw_data_list[t_index]["failed_test_list"])))
                    for t in intersection_failed_test_cases:
                        test_case_fail_rates[t] += 1
                for k, v in test_case_fail_rates.items():
                    if v != 0:
                        test_case_fail_rates[k] = v / float(len(check_indexes))
                        # end for
                feats = {
                    "commits_history": commits_history,
                    "files_changed": data_point["changed_files_num"],
                    "num_authors": data_point["distinct_authors_num"],
                    "test_fail_rates": test_case_fail_rates,
                    "num_test_cases": data_point["tests_cases_num"],
                    "num_common_tokens": test_case_common_tokens,
                    "extension_dict": extension_dict,
                    "changed_files": data_point["changed_files"],
                    "min_distance": test_case_min_dis,
                    "target_cardinality": cardinality
                }
                proj_name = proj_name.split(".")[0]
                processed_data = {
                    "proj_name": proj_name,
                    "raw_data": data_point,
                    "features": feats,
                    "labels": labels
                }
                processed_data_list.append(processed_data)
            # end for
            if processed_data_list:
                IOUtils.mk_dir(output_dir / proj_name)
                IOUtils.dump(output_dir / proj_name / "processed-repo-data.json", processed_data_list)
        # end for

    def split_project_data(self):
        """Split data in project level, i.e. extract the latest SHAs as the test data"""
        data_dir = Macros.data_dir / "proj-data"
        proj_list = listdir(data_dir)
        project_files = [Path(data_dir / proj / "processed-repo-data.json") for proj in listdir(data_dir)]
        merged_train_data = list()
        merged_test_data = list()
        for proj_file, proj_name in zip(project_files, proj_list):
            processed_data = IOUtils.load(proj_file)
            print(proj_file)
            total_data_num = len(processed_data)
            data_count = 0
            proj_test_data = list()
            print(total_data_num)
            while data_count / total_data_num <= self.TEST_RATIO:
                data_point = processed_data.pop(0)
                print(data_count)
                proj_test_data.append(data_point)
                data_count += 1
            # todo: add SHAs used per project as eval. add macros.
            # end while
            IOUtils.dump(data_dir / proj_name / "repo-test-data.json", proj_test_data)
            IOUtils.dump(data_dir / proj_name / "repo-train-data.json", processed_data)
            merged_train_data.extend(processed_data)
            merged_test_data.extend(proj_test_data)
        # end for
        model_data_dir = Macros.data_dir / "model-data"
        IOUtils.mk_dir(model_data_dir)
        IOUtils.dump(model_data_dir / "train.json", merged_train_data)
        IOUtils.dump(model_data_dir / "test.json", merged_test_data)
