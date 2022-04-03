from random import sample, shuffle
import random

from pts.processor.RankProcess import RankProcessor

random.seed(14)
from pathlib import Path
from typing import List, Dict
import ipdb
from collections import defaultdict
from seutil import LoggingUtils, IOUtils, BashUtils

from pts.Environment import Environment
from pts.Macros import Macros
from pts.data import diff_utils
from pts.processor.data_utils.SubTokenizer import SubTokenizer
from pts.data.diff_utils import extract_code_diffs, removeComments


class RankBM25Processor:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.repos_downloads_dir = Macros.repos_downloads_dir
        self.repos_result_dir = Macros.repos_results_dir

    def process_train_tools(self, proj_names: List[str]):
        """
        Create the dataset to train the rank model, this function uses the tests selected by tools as the positive
        examples.
        Ignore time-out mutants;
        """
        proj_dirs: List = [Path(self.repos_result_dir / proj) for proj in proj_names]  # a list of project dirs
        for proj, proj_dir in zip(proj_names, proj_dirs):
            # Get all test class names, with the methods in them
            # test_class_2_methods = collections.defaultdict(list)
            # test_class_file_list = BashUtils.run(f"find {Macros.repos_downloads_dir}/{proj}_ekstazi/.ekstazi/ "
            #                                      f"-name \"*Test*.clz\"").stdout.split("\n")
            collected_results_dir: Path = proj_dir / "collector"
            method_dict = IOUtils.load(collected_results_dir / "method-data.json")
            if (collected_results_dir / "method-data.json").exists():
                test_class_2_methods = IOUtils.load(collected_results_dir / "test2meth.json")
                print(f"{proj} has {len(test_class_2_methods.keys())} test classes in total.")
            else:
                raise FileNotFoundError("Can not find test2meth.json.")
            all_test_classes = list(test_class_2_methods.keys())
            test_class_num = len(all_test_classes)

            for data_type in ["Ekstazi"]:
                mutated_class_set = set()
                processed_train_data = list()
                processed_valid_data = list()
                output_dir = self.output_dir / proj.split('_')[1] / data_type
                IOUtils.mk_dir(output_dir)

                objs = IOUtils.load_json_stream(collected_results_dir / "mutant-data-rts-tool.json")
                for obj in objs:
                    if obj["status"] == "TIME_OUT":
                        continue
                    # Decide train or valid data
                    pb = 1
                    if obj["mutatedClass"] in mutated_class_set:
                        continue
                    else:
                        mutated_class_set.add(obj["mutatedClass"])
                    # get mutated class name
                    mutated_clss = SubTokenizer.sub_tokenize_java_like(obj["mutatedClass"].split('.')[-1])
                    mutated_method = SubTokenizer.sub_tokenize_java_like(obj["mutatedMethod"])

                    # Only use the new code as features
                    code_diffs = SubTokenizer.sub_tokenize_java_like(obj["new_code"])

                    killed_test_set = set(obj[data_type])
                    lived_test_class = set(all_test_classes).difference(killed_test_set)
                    for positive_test in killed_test_set:
                        pos_test_class = SubTokenizer.sub_tokenize_java_like(positive_test)  # + \
                        # SubTokenizer.sub_tokenize_java_like(
                        #     " ".join(test_class_2_methods[positive_test]))
                        # to avoid dataset becoming too large, do sampling for living test class
                        # sample_size = len(lived_test_class)
                        # sampled_lived_test_class = sample(lived_test_class, sample_size)
                        for negative_test in lived_test_class:
                            neg_test_class = SubTokenizer.sub_tokenize_java_like(negative_test)  # + \
                            # SubTokenizer.sub_tokenize_java_like(
                            #     " ".join(test_class_2_methods[negative_test]))
                            data_point = {
                                "changed_class_name": mutated_clss,
                                "changed_method_name": mutated_method,
                                "code_diff": code_diffs,
                                "abstract_code_diff": obj["mutator"],
                                "pos_test_class": pos_test_class,
                                "neg_test_class": neg_test_class
                            }
                            if pb > 0.1:
                                processed_train_data.append(data_point)
                            else:
                                processed_valid_data.append(data_point)
                            # end if
                        # end for
                    # end for
                # end for
                print(f"{data_type}")
                print(f"In total there are {len(processed_train_data)} data point for training")
                print(f"In total there are {len(processed_valid_data)} data point for validation")
                shuffle(processed_train_data)
                shuffle(processed_valid_data)
                IOUtils.dump(output_dir / "train.json", processed_train_data, IOUtils.Format.jsonNoSort)
                IOUtils.dump(output_dir / "valid.json", processed_valid_data, IOUtils.Format.jsonNoSort)
            pit_log_file = {}
            pit_log_file["project"] = proj
            pit_log_file["test_num"] = test_class_num
            IOUtils.dump(Macros.results_dir / "metrics" / f"stats-{proj}-pitlog.json", pit_log_file)

    def process_train_fail(self, proj_names: List[str]):
        """
        Create the dataset to train the rank model, this function uses the failed tests as the positive
        examples and will include the BM25 scores.
        """
        from pts.models.BM25Baseline import tokenize, build_bm25_model, pre_proecessing_for_training_sha, \
            get_BM25_score, max_abs_normalize
        proj_dirs: List = [Path(self.repos_result_dir / proj) for proj in proj_names]  # a list of project dirs
        for proj, proj_dir in zip(proj_names, proj_dirs):
            output_dir = self.output_dir / proj.split('_')[1]
            collected_results_dir = proj_dir / "collector"
            if (collected_results_dir / "method-data.json").exists():
                test_class_2_methods = IOUtils.load(collected_results_dir / "test2meth.json")
            else:
                raise FileNotFoundError("Can not find test2meth.json.")
            all_test_classes = list(test_class_2_methods.keys())
            test_class_num = len(all_test_classes)



            # initialize the list to store the data
            processed_train_data = []
            processed_valid_data = []

            objs = IOUtils.load(collected_results_dir / "mutant-data-rts-tool.json")
            changed_file_list = [d["sourceFile"] for d in objs]
            # Preprocessing for calculating BM25
            data_items = pre_proecessing_for_training_sha(proj, changed_file_list)
            trainDocs = [tokenize(data_items["data_objects"].get(i, "")) for i in all_test_classes]
            bm25model = build_bm25_model(trainDocs)


            for obj in objs:
                pb = random.uniform(0, 1)
                if obj["status"] == "TIME_OUT":
                    continue
                # get diff features: mutated_class_name, mutated_method_name, mutation_type
                changed_class_name = obj["mutatedClass"].split('.')[-1]
                changed_method_name = obj["mutatedMethod"]
                mutator = obj["mutator"]
                killed_test_set = set()
                if obj["succeedingTests"] == "All":
                    continue
                elif obj["status"] == "KILLED":
                    # extract killed test
                    for t in obj["killingTests"]:
                        if t[0] == t[1] and t[0] != "":
                            tc = t[0].split('.')[-1]
                            if tc in all_test_classes:
                                killed_test_set.add(tc)
                        elif t[0] == "":
                            try:
                                tc = t[1].split(".")[-2]
                            except IndexError:
                                tc = t[1]
                            if tc in all_test_classes:
                                killed_test_set.add(tc)
                        else:
                            try:
                                tc = t[0].split(".")[-1]
                            except IndexError:
                                tc = t[0]
                            if tc in all_test_classes:
                                killed_test_set.add(tc)
                        # end if
                    # end for
                    # extract pass tests
                    lived_test_class = set(all_test_classes).difference(killed_test_set)
                    # get mutated class name
                    mutated_clss = SubTokenizer.sub_tokenize_java_like(obj["mutatedClass"].split('.')[-1])
                    mutated_method = SubTokenizer.sub_tokenize_java_like(obj["mutatedMethod"])
                    mutated_file_path = obj["sourceFile"]
                    bm25_scores = max_abs_normalize(get_BM25_score(data_items, mutated_file_path, bm25_model=bm25model))
                    # extract code diff
                    code_diffs = SubTokenizer.sub_tokenize_java_like(obj["new_code"])
                    for positive_test in killed_test_set:
                        pos_test_input = SubTokenizer.sub_tokenize_java_like(positive_test)
                        pos_bm25_score = bm25_scores[all_test_classes.index(positive_test)]
                        sample_size = min(len(lived_test_class), 20)  # predefined maximum living test classes
                        sampled_lived_test_class = sample(lived_test_class, sample_size)
                        for negative_test in sampled_lived_test_class:
                            neg_test_input = SubTokenizer.sub_tokenize_java_like(negative_test)
                            neg_bm25_score = bm25_scores[all_test_classes.index(negative_test)]
                            data_point = {
                                "changed_class_name": mutated_clss,
                                "changed_method_name": mutated_method,
                                "code_diff": code_diffs,
                                "abstract_code_diff": mutator,
                                "pos_test_class": pos_test_input,
                                "pos_test_bm25": pos_bm25_score,
                                "neg_test_class": neg_test_input,
                                "neg_test_bm25": neg_bm25_score
                            }
                            if pb > 0.1:
                                processed_train_data.append(data_point)
                            else:
                                processed_valid_data.append(data_point)
                        # end for
                    # end for
                # end for
            print(f"In total there are {len(processed_train_data)} data point for training")
            print(f"In total there are {len(processed_valid_data)} data point for validation")
            shuffle(processed_train_data)
            shuffle(processed_valid_data)
            IOUtils.dump(output_dir / "fail-train.json", processed_train_data, IOUtils.Format.jsonNoSort)
            IOUtils.dump(output_dir / "fail-valid.json", processed_valid_data, IOUtils.Format.jsonNoSort)

    def process_eval(self, src_data_dir: Path, proj: str, train_sha=""):
        """
        Prepare eval data for the model from the real-world shas.
        for augumented data
        Note:
            * will filter those shas without any code diff
            * will filter those shas that ekstazi and starts select nothing
        """
        from pts.models.BM25Baseline import max_abs_normalize
        shas_data = IOUtils.load(src_data_dir)
        bm25_results = IOUtils.load(Macros.results_dir / "modelResults" / proj.split('_')[1] / "BM25Baseline" / "per-sha-result.json")
        data_list = list()
        pos_num = 0
        neg_num = 0
        discard_sha = 0
        ekstazi_selected_num = list()
        starts_selected_num = list()

        # Iterate data to process
        for sha, bm25_sha in zip(shas_data, bm25_results):
            # do sanity check
            assert sha["commit"] == bm25_sha["commit"]
            if "diff_per_file" not in sha:
                bad_sha = sha["commit"]
                self.logger.warning(f"This sha {bad_sha} does not have diff per file. Please fix the bug.")
                discard_sha += 1
                continue
            if sha["diff_code"] == "":
                bad_sha = sha["commit"]

            bm25_scores = max_abs_normalize(bm25_sha["prediction_scores"])
            failed_test_clss: List = sha["failed_test_list"]
            failed_test_num = len(failed_test_clss)
            failed_test_bm25 = bm25_scores[: failed_test_num]
            passed_test_clss: List = sha["passed_test_list"]
            passed_test_num = len(passed_test_clss)
            passed_test_bm25 = bm25_scores[failed_test_num: ]
            ekstazi_selected: List = sha["ekstazi_test_list"]
            starts_selected: List = sha["starts_test_list"]

            assert len(bm25_scores) == len(failed_test_bm25) + len(passed_test_bm25)

            # tuple of (ChangedFile.java, added_code)
            changed_files_code = [(c.split('/')[-1], sha["diff_per_file"][c]) for c in sha["diff_per_file"]]

            num_test_class = len(failed_test_clss) + len(passed_test_clss)
            num_changed_file = len(changed_files_code)

            collected_results_dir = Macros.repos_results_dir / proj / "collector"
            test_class_2_methods = IOUtils.load(collected_results_dir / "test2meth.json")

            ekstazi_selected_num.append(len(ekstazi_selected))
            starts_selected_num.append(len(starts_selected))

            if len(failed_test_clss) > 0:
                for cls, code in changed_files_code:
                    # Ignore non java files
                    if "java" not in cls:
                        continue
                    code_diffs = SubTokenizer.sub_tokenize_java_like(code)
                    # get mutator
                    mutator_type = RankProcessor.abstract_mutated_type(code)
                    for i, ftc in enumerate(failed_test_clss):
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
                            "sha": sha["commit"],
                            "label": 1,
                            "changed_class_name": mutated_clss,
                            "changed_method_name": mutated_method,
                            "code_diff": code_diffs,
                            "abstract_code_diff": mutator_type,
                            "pos_test_class": pos_test_input,
                            "pos_test_bm25": failed_test_bm25[i],
                            "neg_test_class": pos_test_input,
                            "neg_test_bm25": failed_test_bm25[i],
                            "ekstazi_label": ekstazi_label,
                            "starts_label": starts_label,
                            "num_test_class": num_test_class,
                            "num_changed_files": num_changed_file
                        }
                        data_list.append(data_point)
                        pos_num += 1
                    # end for
                    for i, ptc in enumerate(passed_test_clss):
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
                        # mutated_method = SubTokenizer.sub_tokenize_java_like(" ".join(sha["changed_method_names"]))
                        mutated_method = []
                        data_point = {
                            "sha": sha["commit"],
                            "label": 0,
                            "changed_method_name": mutated_method,
                            "changed_class_name": mutated_clss,
                            "code_diff": code_diffs,
                            "abstract_code_diff": mutator_type,
                            "pos_test_class": neg_test_input,
                            "pos_test_bm25": passed_test_bm25[i],
                            "neg_test_class": neg_test_input,
                            "neg_test_bm25": passed_test_bm25[i],
                            "ekstazi_label": ekstazi_label,
                            "starts_label": starts_label,
                            "num_test_class": num_test_class,
                            "num_changed_files": num_changed_file
                        }
                        data_list.append(data_point)
                        neg_num += 1
                    # end for
                # end for

        self.logger.info(
            f"In total there are {len(data_list)} number of data point for test. {pos_num} are positive and"
            f" {neg_num} are negative. In total there are {len(shas_data)} shas, {discard_sha} shas are kicked out.")
        IOUtils.dump(self.output_dir / proj.split('_')[1] / f"test.json", data_list, IOUtils.Format.jsonNoSort)

    @staticmethod
    def abstract_mutated_type(new_code: str) -> List[str]:
        new_code = new_code.split()
        math_mutators = {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", ">>>"}
        mutator_types_list = set()
        for code_piece in new_code:
            if "return" in code_piece and ("true" in code_piece or "false" in code_piece):
                mutator_types_list.add("BooleanFalseReturnValsMutator")
            if "<" in code_piece or "<=" in code_piece or ">" in code_piece or ">=" in code_piece:
                mutator_types_list.add("ConditionalsBoundaryMutator")
            if "==" in code_piece or "!=" in code_piece:
                mutator_types_list.add("NegateConditionalsMutator")
            for piece in code_piece:
                if '!' in piece:
                    mutator_types_list.add("NegateConditionalsMutator")
            if len(set(code_piece).intersection(math_mutators)) > 0:
                mutator_types_list.add("MathMutator")
            if "++" in code_piece or "--" in code_piece or "+=" in code_piece or "-=" in code_piece:
                mutator_types_list.add("IncrementsMutator")
            if "/*" in " ".join(code_piece):
                mutator_types_list.add("EmptyObjectReturnValsMutator")
        # for code_piece in deleted_code:
        #     if "return" in code_piece:
        #         mutator_types_list.add("NullReturnValsMutator")
        #     if "();" in code_piece:
        #         mutator_types_list.add("VoidMethodCallMutator")

        if len(mutator_types_list) == 0:
            # print(f"No mutator extracted!")
            # print(old_code + new_code)
            mutator_types_list.add("VoidMethodCallMutator")
        # print(len(mutator_types_list))
        return list(mutator_types_list)[:1]

    def process_triplet_labels(self, proj_names: List[str]):
        """
        Process the data to create three categories for the data:
        Ekstazi-selected-failed: positive
        Ekstazi-selected-passed: netural
        Ekstazi-not-selected: negative
        """
        proj_dirs: List = [Path(self.repos_result_dir / proj) for proj in proj_names]  # a list of project dirs
        for proj, proj_dir in zip(proj_names, proj_dirs):
            output_dir = self.output_dir / proj.split('_')[1]
            collected_results_dir = proj_dir / "collector"
            if (collected_results_dir / "method-data.json").exists():
                test_class_2_methods = IOUtils.load(collected_results_dir / "test2meth.json")
            else:
                raise FileNotFoundError("Can not find test2meth.json.")
            all_test_classes = list(test_class_2_methods.keys())
            test_class_num = len(all_test_classes)

            # initialize the list to store the data
            processed_train_data = []  # with large margin  fail v.s. passed & not selected
            processed_train_add_data = []  # with small margin  fail v.s. selected passed; selected passed v.s. not selected
            processed_valid_data = []
            data_type = "Ekstazi"
            objs = IOUtils.load_json_stream(collected_results_dir / "mutant-data-rts-tool.json")
            for obj in objs:
                pb = random.uniform(0, 1)
                if obj["status"] == "TIME_OUT":
                    continue
                # get diff features: mutated_class_name, mutated_method_name, mutation_type
                mutator = obj["mutator"]
                killed_test_set = set()
                if obj["succeedingTests"] == "All":
                    continue
                elif obj["status"] == "KILLED":
                    # extract killed test
                    for t in obj["killingTests"]:
                        if t[0] == t[1] and t[0] != "":
                            tc = t[0].split('.')[-1]
                            if tc in all_test_classes:
                                killed_test_set.add(tc)
                        elif t[0] == "":
                            try:
                                tc = t[1].split(".")[-2]
                            except IndexError:
                                tc = t[1]
                            if tc in all_test_classes:
                                killed_test_set.add(tc)
                        else:
                            try:
                                tc = t[0].split(".")[-1]
                            except IndexError:
                                tc = t[0]
                            if tc in all_test_classes:
                                killed_test_set.add(tc)
                        # end if
                    # end for
                    # extract tests select by tools but not fail
                    tool_selected_tests = set(obj[data_type])
                    tool_selected_fail_tests = tool_selected_tests.intersection(killed_test_set)
                    selected_pass_tests = tool_selected_tests.difference(tool_selected_fail_tests)
                    # extract passed tests and not selected
                    fail_plus_selected_test = tool_selected_tests.union(killed_test_set)
                    lived_test_class = set(all_test_classes).difference(fail_plus_selected_test)
                    # get mutated class name: fully qualified
                    mutated_clss = SubTokenizer.sub_tokenize_java_like(obj["mutatedClass"].split('.')[-1])
                    mutated_method = SubTokenizer.sub_tokenize_java_like(obj["mutatedMethod"])
                    # extract code diff
                    code_diffs = SubTokenizer.sub_tokenize_java_like(obj["new_code"])
                    for positive_test in killed_test_set:
                        pos_test_input = SubTokenizer.sub_tokenize_java_like(positive_test)
                        sample_size = min(len(lived_test_class), 20)  # predefined maximum living test classes
                        sampled_lived_test_class = sample(lived_test_class, sample_size)
                        for negative_test in sampled_lived_test_class:
                            neg_test_input = SubTokenizer.sub_tokenize_java_like(negative_test)
                            data_point = {
                                "changed_class_name": mutated_clss,
                                "changed_method_name": mutated_method,
                                "code_diff": code_diffs,
                                "abstract_code_diff": mutator,
                                "pos_test_class": pos_test_input,
                                "neg_test_class": neg_test_input
                            }
                            pb = random.uniform(0, 1)
                            if pb > 0.1:
                                processed_train_data.append(data_point)
                            else:
                                processed_valid_data.append(data_point)
                        # end for
                    # end for
                    # Additional training data for the small margin
                    for positive_test in killed_test_set:
                        pos_test_input = SubTokenizer.sub_tokenize_java_like(positive_test)
                        for negative_test in selected_pass_tests:
                            neg_test_input = SubTokenizer.sub_tokenize_java_like(negative_test)
                            data_point = {
                                "changed_class_name": mutated_clss,
                                "changed_method_name": mutated_method,
                                "code_diff": code_diffs,
                                "abstract_code_diff": mutator,
                                "pos_test_class": pos_test_input,
                                "neg_test_class": neg_test_input
                            }
                            processed_train_add_data.append(data_point)
                        # end for
                    # end for
                    for positive_test in selected_pass_tests:
                        pos_test_input = SubTokenizer.sub_tokenize_java_like(positive_test)
                        sample_size = min(len(lived_test_class), 20)  # predefined maximum living test classes
                        sampled_lived_test_class = sample(lived_test_class, sample_size)
                        for negative_test in sampled_lived_test_class:
                            neg_test_input = SubTokenizer.sub_tokenize_java_like(negative_test)
                            data_point = {
                                "changed_class_name": mutated_clss,
                                "changed_method_name": mutated_method,
                                "code_diff": code_diffs,
                                "abstract_code_diff": mutator,
                                "pos_test_class": pos_test_input,
                                "neg_test_class": neg_test_input
                            }
                            processed_train_add_data.append(data_point)
                        # end for
                    # end for
                # end for
            print(f"In total there are {len(processed_train_data)} data point for training")
            print(f"In total there are {len(processed_train_add_data)} data point for small margin training")
            print(f"In total there are {len(processed_valid_data)} data point for validation")
            shuffle(processed_train_data)
            shuffle(processed_valid_data)
            IOUtils.dump(output_dir / "triplet-train.json", processed_train_data, IOUtils.Format.jsonNoSort)
            IOUtils.dump(output_dir / "triplet-add-train.json", processed_train_add_data, IOUtils.Format.jsonNoSort)
            IOUtils.dump(output_dir / "triplet-valid.json", processed_valid_data, IOUtils.Format.jsonNoSort)
