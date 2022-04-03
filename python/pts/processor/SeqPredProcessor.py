import collections
import random
from pathlib import Path
from typing import List, Dict

from seutil import LoggingUtils, IOUtils, BashUtils

from pts.Environment import Environment
from pts.Macros import Macros
from pts.data import diff_utils

class Seq2PredProcessor:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.repos_downloads_dir = Macros.repos_downloads_dir
        self.repos_result_dir = Macros.repos_results_dir
        self.TRAIN_RATIO = 0.8
        self.VAL_RATIO = 0.1
        self.TEST_RATIO = 0.1

    def process(self, proj_names: List[str], data_type="A"):
        """Create dataset for the seq2pred model: the format is as follows:
        {
          label: 0;
          input: List[str], :str(change class name), str(method_body),  str(test class name)  Future: add test methods
        }
        """
        # proj_names: List = listdir(self.repos_result_dir)  # a list of project name
        proj_dirs: List = [Path(self.repos_result_dir / proj) for proj in proj_names]  # a list of project dirs
        data_count = 0
        processed_data = list()
        for proj, proj_dir in zip(proj_names, proj_dirs):
            output_dir = self.output_dir / proj.split('_')[1] / f"{data_type}"
            IOUtils.mk_dir(output_dir)
            collected_results_dir: Path = proj_dir / "collector"
            # build a test-class to test-method-name dict
            test_class_2_methods = IOUtils.load(collected_results_dir / "test2method.json")
            method_dict = IOUtils.load(collected_results_dir / "method-data.json")
            # test_class_2_method_names = dict()
            # for tc, m_l in test_class_2_methods.items():
            #     method_names = list()
            #     for i in set(m_l):
            #         method_names.append(method_dict[i]["name"])
            #     # end for
            #     test_class_2_method_names[tc] = tc + " " + " ".join(method_names)
            # end for
            # self.logger.info(f"Finish building the test-class to test-method-name dict for project {proj}.")
            print(proj)
            objs = IOUtils.load_json_stream(collected_results_dir / "mutant-data.json")
            for obj in objs:
                # get mutated class name
                mutated_clss = obj["mutatedClass"].split('.')[-1]
                # get changed code
                mutated_methd = method_dict[obj["mutatedMethodID"]]["code"].strip()
                old_code = obj["old_code"].strip()
                new_code = obj["new_code"].strip()
                # get the changed code
                if old_code in mutated_methd:
                    # mutated_methd.replace(old_code, new_code)
                    mutated_methd = old_code + " " + new_code
                else:
                    self.logger.info(f"[WARN]: Can not found the code {old_code} in the original code: {mutated_methd}.")
                    continue
                # end if
                all_test_classes = list(test_class_2_methods.keys())
                # test class
                if obj["succeedingTests"] == "All":
                    # all test passed
                    for tc, mn in test_class_2_methods.items():
                        data_point = {
                            "label": 0,
                            "input": [mutated_clss, mutated_methd, tc]
                        }
                        processed_data.append(data_point)
                        data_count += 1
                    # end for
                elif obj["succeedingTests"] == "Remaining":
                    # test killed
                    killed_test_set = set()
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
                    for tc in list(killed_test_set):
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 1,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            self.logger.warning(f"Can not find the killed test class: {tc} in project {proj}.")
                            continue
                    # end for
                    lived_test_class = list(set(all_test_classes).difference(set(killed_test_set)))
                    for tc in lived_test_class:
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 0,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            self.logger.warning(f"Can not find the lived test class: {tc} in project {proj}.")
                            continue
                    # end for
                elif obj["killingTests"] == "All":
                    killed_test_class = list(set(all_test_classes))
                    for tc in killed_test_class:
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 1,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            # self.logger.warning(f"Can not find the killed test class: {tc} in project {proj}.")
                            continue
                else:
                    self.logger.warning(f"Unrecognized status " + str(obj["succeedingTests"]))
                    raise RuntimeError
                # end if
            # end for
            print(f"In total there are {data_count} data point for training.")
            IOUtils.dump(output_dir / "A-mutant.json", processed_data, IOUtils.Format.jsonNoSort)
        # end for

        # By now not using this data for eval
        # data_split = self.split(processed_data, data_count)
        # # Collect metrics at the same time
        # seq2pred_data_stats = {}
        # seq2pred_data_stats["total_data"] = data_count
        # for typ in [Macros.train, Macros.test, Macros.val]:
        #     IOUtils.dump(self.output_dir / f"{typ}.json", data_split[typ], IOUtils.Format.jsonNoSort)
        #     seq2pred_data_stats[f"{typ}-num"] = len(data_split[typ])
        # # end for
        # IOUtils.dump(Macros.results_dir/"metrics"/"stats-seq2pred-dataset.json", seq2pred_data_stats, IOUtils.Format.jsonNoSort)

    def process_hybrid(self, proj_names: List[str]):
        """
        Create the dataset to train the hybrid model
        The format is as follows:
        """
        # proj_names: List = listdir(self.repos_result_dir)  # a list of project name
        proj_dirs: List = [Path(self.repos_result_dir / proj) for proj in proj_names]  # a list of project dirs
        for proj, proj_dir in zip(proj_names, proj_dirs):
            # Get all test class names, with the methods in them
            test_class_2_methods = collections.defaultdict(list)
            test_class_file_list = BashUtils.run(f"find {Macros.repos_downloads_dir}/{proj}_ekstazi/.ekstazi/ "
                                                 f"-name \"*Test*.clz\"").stdout.split("\n")
            collected_results_dir: Path = proj_dir / "collector"
            method_dict = IOUtils.load(collected_results_dir / "method-data.json")
            for cf in test_class_file_list:
                if cf != "":
                    class_name = cf.split('/')[-1].split('.')[-2]
                    for m in method_dict:
                        if m["class_name"] == class_name:
                            test_class_2_methods[class_name].append(m["code"])
                # end if
            # end for
            all_test_classes = list(test_class_2_methods.keys())
            processed_data = list()
            output_dir = self.output_dir / proj.split('_')[1] / "hybrid"
            IOUtils.mk_dir(output_dir)

            objs = IOUtils.load_json_stream(collected_results_dir / "mutant-data-rts-tool.json")
            positive = 0
            for obj in objs:
                if obj["status"] == "TIME_OUT":
                    continue
                # get diff sequence
                old_code = obj["old_code"].strip().split()
                new_code = obj["new_code"].strip().split()
                edit_seq, _, _ = diff_utils.compute_code_diffs(old_code, new_code)
                killed_test_set = set()
                if obj["succeedingTests"] == "All":
                    continue
                elif obj["succeedingTests"] == "Remaining":
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
                    for tc in list(killed_test_set):
                        data_point = {
                            "label": 1,
                            "code_diff": edit_seq,
                            "test_class": tc,
                            "test_class_methods": test_class_2_methods[tc]
                        }
                        processed_data.append(data_point)
                        positive += 1
                    # end for
                    starts_tests = set(obj["STARTS"])
                    lived_test_class = starts_tests.difference(killed_test_set)
                    for tc in list(lived_test_class):
                        data_point = {
                            "label": 0,
                            "code_diff": edit_seq,
                            "test_class": tc,
                            "test_class_methods": test_class_2_methods[tc]
                        }
                        processed_data.append(data_point)
                    # end for
                # end if
            # end for
            print(f"In total there are {len(processed_data)} data point for training, positive is {positive}")
            IOUtils.dump(output_dir / "hybrid-mutant.json", processed_data, IOUtils.Format.jsonNoSort)

    def process_tools(self, proj_names: List[str]):
        """
        Create the dataset to train tool-ml model, the labels are given by the tools
        (Outdated) Ignore TIME_OUT: Create dataset for the seq2pred model: the format is as follows:
        {
          label: 0;
          input: List[str], :str(change class name), str(method_body),  str(test class name)  Future: add test methods
        }
        """
        # proj_names: List = listdir(self.repos_result_dir)  # a list of project name
        proj_dirs: List = [Path(self.repos_result_dir / proj) for proj in proj_names]  # a list of project dirs
        for proj, proj_dir in zip(proj_names, proj_dirs):
            test_class_2_methods = collections.defaultdict(list)
            print(proj)
            test_class_file_list = BashUtils.run(f"find {Macros.repos_downloads_dir}/{proj}_ekstazi/.ekstazi/ "
                                                 f"-name \"*Test*.clz\"").stdout.split("\n")
            for cf in test_class_file_list:
                if cf != "":
                    class_name = cf.split('/')[-1].split('.')[-2]
                    test_class_2_methods[class_name].append("POS")
                # end if
            # end if
            for data_type in ["Ekstazi", "STARTS"]:
                data_count = 0
                processed_data = list()
                output_dir = self.output_dir / proj.split('_')[1] / f"{data_type}"
                IOUtils.mk_dir(output_dir)
                collected_results_dir: Path = proj_dir / "collector"
                # build a test-class to test-method-name dict
                method_dict = IOUtils.load(collected_results_dir / "method-data.json")
                pos = 0
                objs = IOUtils.load_json_stream(collected_results_dir / "mutant-data-rts-tool.json")
                for obj in objs:
                    # if obj["status"] == "TIME_OUT":
                    #     continue
                    # get mutated class name
                    mutated_clss = obj["mutatedClass"].split('.')[-1]
                    # get changed code
                    mutated_methd = method_dict[obj["mutatedMethodID"]]["code"].strip()
                    old_code = obj["old_code"].strip()
                    new_code = obj["new_code"].strip()
                    # get the changed code
                    if old_code in mutated_methd:
                        # mutated_methd.replace(old_code, new_code)
                        mutated_methd = old_code + " " + new_code
                    else:
                        # self.logger.warning(f"Can not found the code {old_code} in the original code: {mutated_methd}.")
                        continue
                    # end if
                    all_test_classes = list(test_class_2_methods.keys())

                    # test class
                    killed_test_set = set()
                    for tc in obj[data_type]:
                        data_point = {
                            "label": 1,
                            "input": [mutated_clss, mutated_methd, tc]
                        }
                        killed_test_set.add(tc)
                        processed_data.append(data_point)
                        data_count += 1
                        pos += 1
                    # end for
                    lived_test_class = list(set(all_test_classes).difference(killed_test_set))
                    if not killed_test_set.issubset(set(all_test_classes)):
                        print(killed_test_set)
                        print(all_test_classes)
                    for tc in lived_test_class:
                        data_point = {
                            "label": 0,
                            "input": [mutated_clss, mutated_methd, tc]
                        }
                        processed_data.append(data_point)
                        data_count += 1
                # assert len(lived_test_class) + len(killed_test_set) == len(all_test_classes)
                # end for
                print(f"In total there are {len(processed_data)} data point for training for {data_type}, positive is {pos}")
                IOUtils.dump(output_dir / f"{data_type}-mutant.json", processed_data, IOUtils.Format.jsonNoSort)
                IOUtils.rm_dir(Macros.repos_downloads_dir/f"{proj}_ekstazi")
                IOUtils.rm_dir(Macros.repos_downloads_dir/f"{proj}_starts")

    def process_tri(self, proj_names: List[str], data_type="A-tri"):
        """Create dataset for the seq2pred model: the format is as follows:
        {
          label: 0, 1, 2;  (2 means no coverage)
          input: List[str], :str(change class name), str(method_body),  str(test class name)  Future: add test methods
        }
        """
        # proj_names: List = listdir(self.repos_result_dir)  # a list of project name
        proj_dirs: List = [Path(self.repos_result_dir / proj) for proj in proj_names]  # a list of project dirs
        data_count = 0
        processed_data = list()
        for proj, proj_dir in zip(proj_names, proj_dirs):
            output_dir = self.output_dir / proj.split('_')[1] / f"{data_type}"
            IOUtils.mk_dir(output_dir)
            collected_results_dir: Path = proj_dir / "collector"
            # build a test-class to test-method-name dict
            test_class_2_methods = IOUtils.load(collected_results_dir / "test2method.json")
            method_dict = IOUtils.load(collected_results_dir / "method-data.json")
            # test_class_2_method_names = dict()
            # for tc, m_l in test_class_2_methods.items():
            #     method_names = list()
            #     for i in set(m_l):
            #         method_names.append(method_dict[i]["name"])
            #     # end for
            #     test_class_2_method_names[tc] = tc + " " + " ".join(method_names)
            # end for
            # self.logger.info(f"Finish building the test-class to test-method-name dict for project {proj}.")
            print(proj)
            objs = IOUtils.load_json_stream(collected_results_dir / "tri-mutant-data.json")
            for obj in objs:
                # get mutated class name
                mutated_clss = obj["mutatedClass"].split('.')[-1]
                # get changed code
                mutated_methd = method_dict[obj["mutatedMethodID"]]["code"].strip()
                old_code = obj["old_code"].strip()
                new_code = obj["new_code"].strip()
                # get the changed code
                if old_code in mutated_methd:
                    # mutated_methd.replace(old_code, new_code)
                    mutated_methd = old_code + " " + new_code
                else:
                    # self.logger.warning(f"Can not found the code {old_code} in the original code: {mutated_methd}.")
                    continue
                # end if
                all_test_classes = list(test_class_2_methods.keys())
                if obj["killingTests"] and len(obj["killingTests"]) > 0 and isinstance(obj["killingTests"], list):
                    # mutant killed
                    killed_test_set = set()
                    lived_test_set = set()
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
                    for tc in list(killed_test_set):
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 1,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            self.logger.warning(f"Can not find the killed test class: {tc} in project {proj}.")
                            continue
                    # end for
                    if obj["succeedingTests"] and len(obj["succeedingTests"]) > 0:
                        for t in obj["succeedingTests"]:
                            if t[0] == t[1] and t[0] != "":
                                tc = t[0].split('.')[-1]
                                if tc in all_test_classes:
                                    lived_test_set.add(tc)
                            elif t[0] == "":
                                try:
                                    tc = t[1].split(".")[-2]
                                except IndexError:
                                    tc = t[1]
                                if tc in all_test_classes:
                                    lived_test_set.add(tc)
                            else:
                                try:
                                    tc = t[0].split(".")[-1]
                                except IndexError:
                                    tc = t[0]
                                if tc in all_test_classes:
                                    lived_test_set.add(tc)
                        for tc in list(lived_test_set):
                            try:
                                # mn = test_class_2_method_names[tc]
                                data_point = {
                                    "label": 0,
                                    "input": [mutated_clss, mutated_methd, tc]
                                }
                                processed_data.append(data_point)
                                data_count += 1
                            except KeyError:
                                self.logger.warning(f"Can not find the lived test class: {tc} in project {proj}.")
                                continue
                    nocoverage_test_class = list(set(all_test_classes).difference(set(list(killed_test_set)+list(lived_test_set))))
                    for tc in nocoverage_test_class:
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 2,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            self.logger.warning(f"Can not find the no-coverage test class: {tc} in project {proj}.")
                            continue
                    # end for
                elif obj["killingTests"] == "All":
                    killed_test_class = list(set(all_test_classes))
                    for tc in killed_test_class:
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 1,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            # self.logger.warning(f"Can not find the killed test class: {tc} in project {proj}.")
                            continue
                elif obj["succeedingTests"] and len(obj["succeedingTests"]) > 0:
                    # survived mutants
                    lived_test_set = set()
                    for t in obj["succeedingTests"]:
                        if t[0] == t[1] and t[0] != "":
                            tc = t[0].split('.')[-1]
                            if tc in all_test_classes:
                                lived_test_set.add(tc)
                        elif t[0] == "":
                            try:
                                tc = t[1].split(".")[-2]
                            except IndexError:
                                tc = t[1]
                            if tc in all_test_classes:
                                lived_test_set.add(tc)
                        else:
                            try:
                                tc = t[0].split(".")[-1]
                            except IndexError:
                                tc = t[0]
                            if tc in all_test_classes:
                                lived_test_set.add(tc)
                    for tc in list(lived_test_set):
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 0,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            self.logger.warning(f"Can not find the lived test class: {tc} in project {proj}.")
                            continue
                    # end for
                    nocoverage_test_class = list(set(all_test_classes).difference(set((lived_test_set))))
                    for tc in nocoverage_test_class:
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 2,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            self.logger.warning(f"Can not find the no-coverage test class: {tc} in project {proj}.")
                            continue
                    # end for
                elif "nocoverageTests" in obj.keys() and obj["nocoverageTests"] == "All":
                    # mutants nocoverage
                    for tc in list(all_test_classes):
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 2,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            self.logger.warning(f"Can not find the no coverage test class: {tc} in project {proj}.")
                            continue
                    # end for
                else:
                    self.logger.warning(f"Unrecognized status " + str(obj))
                    raise RuntimeError
                # end if
            # end for
            print(f"In total there are {data_count} data point for training.")
            IOUtils.dump(output_dir / "A-tri-mutant.json", processed_data, IOUtils.Format.jsonNoSort)

    def process_killed(self, proj_names: List[str]):
        proj_dirs: List = [Path(self.repos_result_dir / proj) for proj in proj_names]  # a list of project dirs
        data_count = 0
        for proj, proj_dir in zip(proj_names, proj_dirs):
            eks_processed_data = list()
            sts_processed_data = list()
            output_dir = self.output_dir / proj.split('_')[1]
            IOUtils.mk_dir(output_dir)
            collected_results_dir: Path = proj_dir / "collector"
            # build a test-class to test-method-name dict
            test_class_2_methods = IOUtils.load(collected_results_dir / "test2method.json")
            method_dict = IOUtils.load(collected_results_dir / "method-data.json")
            objs = IOUtils.load_json_stream(collected_results_dir / "mutant-data-rts-tool.json")
            pos_count = 0
            neg_count = 0
            for obj in objs:
                # get mutated class name
                mutated_clss = obj["mutatedClass"].split('.')[-1]
                # get changed code
                mutated_methd = method_dict[obj["mutatedMethodID"]]["code"].strip()
                old_code = obj["old_code"].strip()
                new_code = obj["new_code"].strip()
                # get the changed code
                if old_code in mutated_methd:
                    # mutated_methd.replace(old_code, new_code)
                    mutated_methd = old_code + " " + new_code
                else:
                    # self.logger.warning(f"Can not found the code {old_code} in the original code: {mutated_methd}.")
                    continue
                # end if
                all_test_classes = list(test_class_2_methods.keys())

                # killed mutants
                if obj["killingTests"] and len(obj["killingTests"]) > 0 and isinstance(obj["killingTests"], list):
                    killed_test_set = set()
                    eks_test_set = set()
                    sts_test_set = set()
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
                    # end for
                    for t in obj["Ekstazi"]:
                        eks_test_set.add(t)
                    # end for
                    for t in obj["STARTS"]:
                        sts_test_set.add(t)
                    # end for
                    # Create positive labels
                    for tc in list(killed_test_set):
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 1,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            sts_processed_data.append(data_point)
                            eks_processed_data.append(data_point)
                            # processed_data.append(data_point)
                            pos_count += 1
                            data_count += 1
                        except KeyError:
                            self.logger.warning(f"Can not find the killed test class: {tc} in project {proj}.")
                            continue
                    # end for
                    # Create negative labels for tools
                    eks_neg_test = list(set(eks_test_set).difference(killed_test_set))
                    sts_neg_test = list(sts_test_set.difference(killed_test_set))
                    for tc in eks_neg_test:
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 0,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            eks_processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            self.logger.warning(f"Can not find the killed test class: {tc} in project {proj}.")
                            continue
                    # end for
                    for tc in sts_neg_test:
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 0,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            sts_processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            self.logger.warning(f"Can not find the killed test class: {tc} in project {proj}.")
                            continue
                    # end for
                elif obj["killingTests"] == "All":
                    # TIME_OUT mutants
                    for t in obj["Ekstazi"]:
                        eks_test_set.add(t)
                    # end for
                    for t in obj["STARTS"]:
                        sts_test_set.add(t)
                    # end for

                    for tc in eks_test_set:
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 1,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            eks_processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            self.logger.warning(f"Can not find the killed test class: {tc} in project {proj}.")
                            continue
                    # end for

                    for tc in sts_test_set:
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 1,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            sts_processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            self.logger.warning(f"Can not find the killed test class: {tc} in project {proj}.")
                            continue
                    # end for
                elif obj["succeedingTests"] and len(obj["succeedingTests"]) > 0:
                    for t in obj["Ekstazi"]:
                        eks_test_set.add(t)
                    # end for
                    for t in obj["STARTS"]:
                        sts_test_set.add(t)
                    # end for
                    for tc in eks_test_set:
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 0,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            eks_processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            self.logger.warning(f"Can not find the killed test class: {tc} in project {proj}.")
                            continue
                    # end for

                    for tc in sts_test_set:
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 0,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            sts_processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            self.logger.warning(f"Can not find the killed test class: {tc} in project {proj}.")
                            continue
                    # end for
                # end if
                IOUtils.dump(output_dir/"Ekstazi-killed"/"Ekstazi-killed-mutant.json", eks_processed_data, IOUtils.Format.jsonNoSort)
                IOUtils.dump(output_dir/"STARTS-killed"/"STARTS-killed-mutant.json", sts_processed_data, IOUtils.Format.jsonNoSort)

    def process_partially(self, proj_names: List[str], data_type="H"):
        """Create dataset for the seq2pred model: the format is as follows:
        { label: 0;
          input: List[str], :str(change class name), str(method_body),  str(test class name)  Future: add test methods
        }
        This function will ignore some mutators specified by the user. (currently hard-coded)
        """
        # proj_names: List = listdir(self.repos_result_dir)  # a list of project name
        proj_dirs: List = [Path(self.repos_result_dir / proj) for proj in proj_names]  # a list of project dirs
        data_count = 0
        processed_data = list()
        if data_type == "H":
            selected_mutators = {"BooleanTrueReturnValsMutator", "BooleanFalseReturnValsMutator",
                                 "ConditionalsBoundaryMutator", "MathMutator", "PrimitiveReturnsMutator"}
        elif data_type == "H-bar":
            selected_mutators = {"IncrementsMutator", "VoidMethodCallMutator", "NegateConditionalsMutator",
                                 "NullReturnValsMutator", "EmptyObjectReturnValsMutator"}
        for proj, proj_dir in zip(proj_names, proj_dirs):
            output_dir = self.output_dir / proj.split('_')[1] / data_type
            IOUtils.mk_dir(output_dir)
            collected_results_dir: Path = proj_dir / "collector"
            # build a test-class to test-method-name dict
            test_class_2_methods = IOUtils.load(collected_results_dir / "test2method.json")
            method_dict = IOUtils.load(collected_results_dir / "method-data.json")

            objs = IOUtils.load_json_stream(collected_results_dir / "mutant-data.json")
            for obj in objs:
                if obj["mutator"] not in selected_mutators:
                    continue
                # get mutated class name
                mutated_clss = obj["mutatedClass"].split('.')[-1]
                # get changed code
                mutated_methd = method_dict[obj["mutatedMethodID"]]["code"].strip()
                old_code = obj["old_code"].strip()
                new_code = obj["new_code"].strip()
                # get the changed code
                if old_code in mutated_methd:
                    mutated_methd = old_code + " " + new_code
                else:
                    # self.logger.warning(f"Can not found the code {old_code} in the original code: {mutated_methd}.")
                    continue
                # end if
                all_test_classes = list(test_class_2_methods.keys())
                # test class
                if obj["succeedingTests"] == "All":
                    # all test passed
                    for tc, mn in test_class_2_methods.items():
                        data_point = {
                            "label": 0,
                            "input": [mutated_clss, mutated_methd, tc]
                        }
                        processed_data.append(data_point)
                        data_count += 1
                    # end for
                elif obj["succeedingTests"] == "Remaining":
                    # test killed
                    killed_test_set = set()
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
                    for tc in list(killed_test_set):
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 1,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            self.logger.warning(f"Can not find the killed test class: {tc} in project {proj}.")
                            continue
                    # end for
                    lived_test_class = list(set(all_test_classes).difference(set(killed_test_set)))
                    for tc in lived_test_class:
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 0,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            self.logger.warning(f"Can not find the lived test class: {tc} in project {proj}.")
                            continue
                    # end for
                elif obj["killingTests"] == "All":
                    killed_test_class = list(set(all_test_classes))
                    for tc in killed_test_class:
                        try:
                            # mn = test_class_2_method_names[tc]
                            data_point = {
                                "label": 1,
                                "input": [mutated_clss, mutated_methd, tc]
                            }
                            processed_data.append(data_point)
                            data_count += 1
                        except KeyError:
                            self.logger.warning(f"Can not find the killed test class: {tc} in project {proj}.")
                            continue
                else:
                    self.logger.warning(f"Unrecognized status " + str(obj["succeedingTests"]))
                    raise RuntimeError
                # end if
            # end for
            print(f"In total there are {data_count} data point for training.")
            IOUtils.dump(output_dir / f"{data_type}-mutant.json", processed_data,
                         IOUtils.Format.jsonNoSort)
        # end for
        # By now not using this data for eval
        # data_split = self.split(processed_data, data_count)
        # # Collect metrics at the same time
        # seq2pred_data_stats = {}
        # seq2pred_data_stats["total_data"] = data_count
        # for typ in [Macros.train, Macros.test, Macros.val]:
        #     IOUtils.dump(self.output_dir / f"{typ}.json", data_split[typ], IOUtils.Format.jsonNoSort)
        #     seq2pred_data_stats[f"{typ}-num"] = len(data_split[typ])
        # # end for
        # IOUtils.dump(Macros.results_dir/"metrics"/"stats-seq2pred-dataset.json", seq2pred_dat

    def process_tools_simple(self, proj_names: List[str]):
        """This method create the dataset with the labels from Ekstazi and STARTS
        The input of the model will be the name of test class and changed files. (no code diff)
        """
        # proj_names: List = listdir(self.repos_result_dir)  # a list of project name
        proj_dirs: List = [Path(self.repos_result_dir / proj) for proj in proj_names]  # a list of project dirs
        data_count = 0
        for proj, proj_dir in zip(proj_names, proj_dirs):
            for data_type in ["Ekstazi", "STARTS"]:
                processed_data = list()
                output_dir = self.output_dir / proj.split('_')[1] / f"{data_type}-simple"
                IOUtils.mk_dir(output_dir)
                collected_results_dir: Path = proj_dir / "collector"
                # build a test-class to test-method-name dict
                test_class_2_methods = IOUtils.load(collected_results_dir / "test2method.json")
                method_dict = IOUtils.load(collected_results_dir / "method-data.json")
                # test_class_2_method_names = dict()
                # for tc, m_l in test_class_2_methods.items():
                #     method_names = list()
                #     for i in set(m_l):
                #         method_names.append(method_dict[i]["name"])
                #     # end for
                #     test_class_2_method_names[tc] = tc + " " + " ".join(method_names)
                # end for
                # self.logger.info(f"Finish building the test-class to test-method-name dict for project {proj}.")
                pos = 0
                objs = IOUtils.load_json_stream(collected_results_dir / "mutant-data-rts-tool.json")
                for obj in objs:
                    if obj["status"] == "TIME_OUT":
                        continue
                    # get mutated class name
                    mutated_clss = obj["mutatedClass"].split('.')[-1]
                    # get changed code
                    mutated_methd = method_dict[obj["mutatedMethodID"]]["code"].strip()
                    old_code = obj["old_code"].strip()
                    new_code = obj["new_code"].strip()
                    # get the changed code
                    if old_code in mutated_methd:
                        # mutated_methd.replace(old_code, new_code)
                        mutated_methd = old_code + " " + new_code
                    else:
                        self.logger.warning(f"Can not found the code {old_code} in the original code: {mutated_methd}.")
                        continue
                    # end if
                    all_test_classes = list(test_class_2_methods.keys())
                    # test class
                    killed_test_set = set()
                    for tc in obj[data_type]:
                        data_point = {
                            "label": 1,
                            "input": [mutated_clss, tc]
                        }
                        killed_test_set.add(tc)
                        processed_data.append(data_point)
                        data_count += 1
                        pos += 1
                    # end for
                    lived_test_class = list(set(all_test_classes).difference(set(killed_test_set)))
                    for tc in lived_test_class:
                        data_point = {
                            "label": 0,
                            "input": [mutated_clss, tc]
                        }
                        processed_data.append(data_point)
                        data_count += 1

                # end for
                print(f"In total there are {data_count} data point for training for {data_type}, positive is {pos}")
                IOUtils.dump(output_dir / f"{data_type}-mutant.json", processed_data, IOUtils.Format.jsonNoSort)

    def process_tools_labels(self, src_data_dir, proj, typ):
        """This method create the dataset with the labels from Ekstazi and STARTS
        The input of the model will be the name of test class and changed files. (no code diff)
        """
        output_eks_dir = self.output_dir / proj.split('_')[1] / "Ekstazi"
        IOUtils.mk_dir(output_eks_dir)
        output_sts_dir = self.output_dir / proj.split('_')[1] / "STARTS"
        IOUtils.mk_dir(output_sts_dir)
        objs = IOUtils.load_json_stream(src_data_dir)
        eks_data_list = list()
        sts_data_list = list()
        eks_pos_count = 0
        sts_pos_count = 0
        total_count = 0
        for obj in objs:
            test_class = obj["input"][2]
            if test_class in obj["Ekstazi"]:
                eks_data_list.append({
                    "label": 1,
                    "input": obj["input"][0] + obj["input"][2]
                })
                eks_pos_count += 1
            else:
                eks_data_list.append({
                    "label": 0,
                    "input": obj["input"][0] + obj["input"][2]
                })
            if test_class in obj["STARTS"]:
                sts_data_list.append({
                    "label": 1,
                    "input": obj["input"][0] + obj["input"][2]
                })
                sts_pos_count += 1
            else:
                sts_data_list.append({
                    "label": 0,
                    "input": obj["input"][0] + obj["input"][2]
                })
            # end if
            total_count += 1
        # end for
        # First dump the starts results
        IOUtils.dump(output_sts_dir / f"{typ}.json", sts_data_list, IOUtils.Format.jsonNoSort)
        # Then dump the eks results
        IOUtils.dump(output_eks_dir / f"{typ}.json", eks_data_list, IOUtils.Format.jsonNoSort)
        # Show the brief stats:
        print(
            f"In total {total_count} number of data, {eks_pos_count} of them are ekstazi positive, and {sts_pos_count}"
            f"are starts positive")
        # To merge train and valid data
        if typ == "valid":
            # first merge starts data
            tr_data = IOUtils.load(output_sts_dir / "train.json")
            vl_data = IOUtils.load(output_sts_dir / "valid.json")
            all_data = tr_data + vl_data
            IOUtils.dump(output_sts_dir / "STARTS-mutant.json", all_data, IOUtils.Format.jsonNoSort)
            # second merge ekstazi data
            tr_data = IOUtils.load(output_eks_dir / "train.json")
            vl_data = IOUtils.load(output_eks_dir / "valid.json")
            all_data = tr_data + vl_data
            IOUtils.dump(output_eks_dir / "Ekstazi-mutant.json", all_data, IOUtils.Format.jsonNoSort)
        # end if

    def process_eval_defects4J(self, src_data_dir: Path, proj: str):
        shas = IOUtils.load(src_data_dir)
        data_list = list()
        pos_num = 0
        discard_sha = 0
        empty_sha = 0
        ekstazi_selected_num = list()
        starts_selected_num = list()
        for sha in shas:
            if "diff_per_file" not in sha:
                discard_sha += 1
                bad_sha = sha["commit"]
                self.logger.warning(f"This sha {bad_sha} can not get diff per file!")
                continue
            if len(sha["ekstazi_test_list"]) == 0 or len(sha["starts_test_list"]) == 0:
                empty_sha += 1
                continue
            failed_test_clss: List = list(set([t.split('.')[-1] for t in sha["failed_test_list"]]))
            passed_test_clss: List = list(set([t.split('.')[-1] for t in sha["passed_test_list"]]))
            ekstazi_selected: List = list(set([t.split('.')[-1] for t in sha["ekstazi_test_list"]]))
            starts_selected: List = list(set([t.split('.')[-1] for t in sha["starts_test_list"]]))
            changed_clss: List = list(set([c.split('/')[-1] for c in sha["changed_files"]]))
            changed_code: List[str] = [sha["diff_per_file"][c] for c in sha["changed_files"]]
            assert len(changed_code) == len(changed_clss)
            if sha["diff_code"] == "":
                # if sha["diff_code"][1] == "":
                self.logger.warning("Discard this sha {}.".format(sha["commit"]))
                discard_sha += 1
                continue
            ekstazi_selected_num.append(len(ekstazi_selected))
            starts_selected_num.append(len(starts_selected))
            if failed_test_clss:
                # add positive examples first
                for ftc in failed_test_clss:
                    if ftc in ekstazi_selected:
                        ekstazi_label = 1
                    else:
                        ekstazi_label = 0
                    if ftc in starts_selected:
                        starts_label = 1
                    else:
                        starts_label = 0
                    for cls, code in zip(changed_clss, changed_code):
                        diff_list = code.split("\n")[:-1]
                        old_dif = " ".join([dif[1:].strip() for dif in diff_list if dif[0] == '-'])
                        new_dif = " ".join([dif[1:].strip() for dif in diff_list if dif[0] == '+'])
                        mutated_code: str = old_dif + ' ' + new_dif
                        if mutated_code == "":
                            self.logger.warning("Can not get diff: {}".format(sha["diff_code"]))
                        data_point = {
                            "label": 1,
                            "input": [cls, mutated_code, ftc],
                            "ekstazi_label": ekstazi_label,
                            "starts_label": starts_label,
                            "sha": sha["commit"]
                        }
                        data_list.append(data_point)
                        pos_num += 1
                    # end for
                # end for
            # add negative examples
            if passed_test_clss:
                for ptc in passed_test_clss:
                    if ptc in ekstazi_selected:
                        ekstazi_label = 1
                    else:
                        ekstazi_label = 0
                    if ptc in starts_selected:
                        starts_label = 1
                    else:
                        starts_label = 0
                    for cls, code in zip(changed_clss, changed_code):
                        diff_list = code.split("\n")[:-1]
                        old_dif = " ".join([dif[1:].strip() for dif in diff_list if dif[0] == '-'])
                        new_dif = " ".join([dif[1:].strip() for dif in diff_list if dif[0] == '+'])
                        mutated_code: str = old_dif + ' ' + new_dif
                        if mutated_code == "":
                            self.logger.warning("Can not get diff: {}".format(sha["diff_code"]))
                        data_point = {
                            "label": 0,
                            "input": [cls, mutated_code, ptc],
                            "ekstazi_label": ekstazi_label,
                            "starts_label": starts_label,
                            "sha": sha["commit"]
                        }
                        data_list.append(data_point)
                    # end for
                # end for
        self.logger.info(f"In total there are {len(data_list)} number of data point for eval.")
        print(f"In total {pos_num} positive examples among {len(shas)} shas.")
        print(f"Discard {discard_sha} SHAs in total, {empty_sha} SHAs have no bytecode changes.")
        remain_shas = len(shas) - 1 - discard_sha - empty_sha
        IOUtils.dump(self.output_dir / proj.split('_')[1] / f"val.json", data_list, IOUtils.Format.jsonNoSort)

    def process_eval(self, src_data_dir: Path, train_sha: str, proj: str):
        """
        Prepare eval data for the model from the real-world shas.

        Note:
            * will filter those shas without any code diff
            * will filter those shas that ekstazi and starts select nothing
        """
        shas = IOUtils.load(src_data_dir)[1:]
        data_list = list()
        pos_num = 0
        discard_sha = 0
        empty_sha = 0
        ekstazi_selected_num = list()
        starts_selected_num = list()
        for sha in shas:
            if sha["commit"] == train_sha:
                discard_sha += 1
                continue
            if "diff_per_file" not in sha:
                discard_sha += 1
                bad_sha = sha["commit"]
                self.logger.warning(f"This sha {bad_sha} can not get diff per file!")
                continue
            if len(sha["ekstazi_test_list"]) == 0 or len(sha["starts_test_list"]) == 0:
                empty_sha += 1
                continue
            failed_test_clss: List = list(set([t.split('.')[-1] for t in sha["failed_test_list"]]))
            passed_test_clss: List = list(set([t.split('.')[-1] for t in sha["passed_test_list"]]))
            ekstazi_selected: List = list(set([t.split('.')[-1] for t in sha["ekstazi_test_list"]]))
            starts_selected: List = list(set([t.split('.')[-1] for t in sha["starts_test_list"]]))
            changed_clss: List = list(set([c.split('/')[-1] for c in sha["changed_files"]]))
            changed_code: List[str] = [sha["diff_per_file"][c] for c in sha["changed_files"]]
            assert len(changed_code) == len(changed_clss)
            # changed_clss_str = " ".join(changed_clss)
            # Currently only use code diff bcause we do not know what context to useTODO: use new method body
            if sha["diff_code"] == "":
                # if sha["diff_code"][1] == "":
                self.logger.warning("Discard this sha {}.".format(sha["commit"]))
                discard_sha += 1
                continue
            ekstazi_selected_num.append(len(ekstazi_selected))
            starts_selected_num.append(len(starts_selected))
            if failed_test_clss:
                # add positive examples first
                for ftc in failed_test_clss:
                    if ftc in ekstazi_selected:
                        ekstazi_label = 1
                    else:
                        ekstazi_label = 0
                    if ftc in starts_selected:
                        starts_label = 1
                    else:
                        starts_label = 0
                    for cls, code in zip(changed_clss, changed_code):
                        diff_list = code.split("\n")[:-1]
                        old_dif = " ".join([dif[1:].strip() for dif in diff_list if dif[0] == '-'])
                        new_dif = " ".join([dif[1:].strip() for dif in diff_list if dif[0] == '+'])
                        mutated_code: str = old_dif + ' ' + new_dif
                        if mutated_code == "":
                            self.logger.warning("Can not get diff: {}".format(sha["diff_code"]))
                        data_point = {
                            "label": 1,
                            "input": [cls, mutated_code, ftc],
                            "ekstazi_label": ekstazi_label,
                            "starts_label": starts_label,
                            "sha": sha["commit"]
                        }
                        data_list.append(data_point)
                        pos_num += 1
                    # end for
                # end for
            # add negative examples
            if passed_test_clss:
                for ptc in passed_test_clss:
                    if ptc in ekstazi_selected:
                        ekstazi_label = 1
                    else:
                        ekstazi_label = 0
                    if ptc in starts_selected:
                        starts_label = 1
                    else:
                        starts_label = 0
                    for cls, code in zip(changed_clss, changed_code):
                        diff_list = code.split("\n")[:-1]
                        old_dif = " ".join([dif[1:].strip() for dif in diff_list if dif[0] == '-'])
                        new_dif = " ".join([dif[1:].strip() for dif in diff_list if dif[0] == '+'])
                        mutated_code: str = old_dif + ' ' + new_dif
                        if mutated_code == "":
                            self.logger.warning("Can not get diff: {}".format(sha["diff_code"]))
                        data_point = {
                            "label": 0,
                            "input": [cls, mutated_code, ptc],
                            "ekstazi_label": ekstazi_label,
                            "starts_label": starts_label,
                            "sha": sha["commit"]
                        }
                        data_list.append(data_point)
                    # end for
                # end for
        self.logger.info(f"In total there are {len(data_list)} number of data point for eval.")
        print(f"In total {pos_num} positive examples among {len(shas)} shas.")
        print(f"Discard {discard_sha} SHAs in total, {empty_sha} SHAs have no bytecode changes.")
        remain_shas = len(shas) - 1 - discard_sha - empty_sha
        IOUtils.dump(self.output_dir / proj.split('_')[1] / f"val.json", data_list, IOUtils.Format.jsonNoSort)

        print(f"min ekstazi selected {min(ekstazi_selected_num)}, max: {max(ekstazi_selected_num)}, average: "
              f"{sum(ekstazi_selected_num) / (len(shas) - discard_sha - empty_sha - 1)}")
        print(f"min starts selected {min(starts_selected_num)}, max: {max(starts_selected_num)}, average: "
              f"{sum(starts_selected_num) / (len(shas) - discard_sha - empty_sha - 1)}")

    def split(self, data_list: List[dict], data_count: int) -> Dict:
        """Split the seq2pred data according to the RATIO."""
        split_data: Dict[str, List[Dict]] = collections.defaultdict(list)
        # first shuffle the data
        random.seed(Environment.random_seed)
        random.Random(Environment.random_seed).shuffle(data_list)
        train_num = 0
        test_num = 0
        val_num = 0
        while train_num / data_count < self.TRAIN_RATIO:
            data_example = data_list.pop()
            split_data[Macros.train].append(data_example)
            train_num += 1
        # end while
        while test_num / data_count < self.TEST_RATIO:
            data_example = data_list.pop()
            split_data[Macros.test].append(data_example)
            test_num += 1
        # end while
        split_data[Macros.val] = data_list
        val_num = len(data_list)
        print(f"Number of training sample is {train_num}."
              f"Number of test sample is {test_num}."
              f"Number of val sample is {val_num}.")
        return split_data
