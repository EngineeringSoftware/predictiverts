import collections
import random
from pathlib import Path
from typing import List, Dict
import sys
from seutil import LoggingUtils, IOUtils, BashUtils, TimeUtils
from pts.Environment import Environment
from pts.Macros import Macros
import os
from pts.data import diff_utils
from pts.processor.data_utils.SubTokenizer import SubTokenizer
from pts.collector.mutation.rtstool_tests_collector import getTestsFromSTARTS
from pts.collector.eval_data_collection import test_list_from_surefile_reports
import pandas as pd
import time

class EALRTSProcessor:
    """
    Process the eval data for ELARTS model.
    """
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self):
        self.repos_downloads_dir = Macros.repos_downloads_dir
        self.repos_result_dir = Macros.repos_results_dir

    '''
    Get the dependency graph from STARTS
    '''
    # not using pts.collector.min_dis_graph because the key of the graph is changed file
    # But here we require the key to be the test
    def dep_graph(self, institution, project, prev_sha, sha):
        min_dis_dict = {}
        try:
            with TimeUtils.time_limit(1800):
                with IOUtils.cd(f"{Macros.repos_downloads_dir}"):
                    compile_time_start = time.time()
                    if not os.path.isdir(f"{institution}_{project}"):
                        BashUtils.run(
                            f"git clone https://github.com/{institution}/{project}.git {institution}_{project}")
                    if os.path.isdir(f"{Macros.repos_downloads_dir}/{institution}_{project}_{prev_sha}_{sha}"):
                        BashUtils.run(f"rm -rf {Macros.repos_downloads_dir}/{institution}_{project}")
                        BashUtils.run(f"cp -r {Macros.repos_downloads_dir}/{institution}_{project}_{prev_sha}_{sha} {Macros.repos_downloads_dir}/{institution}_{project}")
                    else:
                        with IOUtils.cd(f"{institution}_{project}"):
                            BashUtils.run(f"git checkout .")
                            BashUtils.run(f"git clean -xfd")
                            print("prev sha", prev_sha)
                            print("sha", sha)
                            BashUtils.run(f"git checkout {prev_sha}")
                            # BashUtils.run(f"mvn test-compile {Macros.SKIPS}")
                            # BashUtils.run(f"mvn starts:select -DupdateSelectChecksums -DstartsLogging=FINEST {Macros.SKIPS}")
                            BashUtils.run(f"mvn starts:starts {Macros.SKIPS}")

                            BashUtils.run(f"git checkout {sha}")
                            # build_res = BashUtils.run(f"mvn clean test-compile {Macros.SKIPS}").stdout
                            build_res = BashUtils.run(
                                    f"mvn starts:starts -DstartsLogging=FINEST {Macros.SKIPS}").stdout
                            if "BUILD FAILURE" in build_res:
                                print("project info: ", institution, project, sha)

                            BashUtils.run(
                                f"cp -r {Macros.repos_downloads_dir}/{institution}_{project} {Macros.repos_downloads_dir}/{institution}_{project}_{prev_sha}_{sha}")
                            # BashUtils.run(
                            #     f"mvn starts:select -DupdateSelectChecksums -DstartsLogging=FINEST {Macros.SKIPS}")
                    compile_time_end = time.time()
                    compile_time = compile_time_end - compile_time_start
                    with IOUtils.cd(f"{institution}_{project}"):
                        # collect the selected tests
                        selected_tests = set()
                        with open(".starts/selected-tests") as fp:
                            for line in fp.readlines():
                                selected_tests.add(line.strip())
                        # print("selected tests", selected_tests)
                        # collect how many files changed
                        changed_classes = set()
                        
                        with open(".starts/changed-classes") as fp:
                            for line in fp.readlines():
                                changed_class = ""
                                if "target/classes" in line:
                                    changed_class = line[line.index("target/classes")+len("target/classes/"):].strip()
                                elif "target/test-classes" in line:
                                    changed_class = line[line.index("target/test-classes")+len("target/test-classes/"):].strip()
                                qualified_changed_class = changed_class.replace("/", ".").replace(".class", "")
                                changed_classes.add(qualified_changed_class)
                        # print("changed classes", changed_classes)
                        # create the graph
                        adj_list = collections.defaultdict(set)
                        # deal with multi module
                        graph_files = BashUtils.run("find . -name 'graph'").stdout

                        if not graph_files or graph_files == "":
                            print("can not have .starts: ", institution, project, sha)
                            return min_dis_dict, selected_tests, changed_classes, {}, {}, 0

                        for graph_file in graph_files.split("\n"):
                            graph_file = graph_file.strip()
                            if graph_file.endswith("/.starts/graph"):
                                with open(graph_file) as dep_relation:
                                    for line in dep_relation.readlines():
                                        nodes = line.split(" ")
                                        node1 = nodes[0].strip()
                                        node2 = nodes[1].strip()
                                        adj_list[node1].add(node2)

                        for selected_test in selected_tests:
                            # bfs
                            dis_test_to_class = collections.defaultdict(set)
                            visited = set()
                            visited.add(selected_test)
                            dis = 0
                            queue = [selected_test]

                            while queue:
                                length = len(queue)
                                for i in range(length):
                                    cur = queue.pop(0)

                                    if cur in changed_classes:
                                        dis_test_to_class[cur] = dis
                                    else:
                                        if cur in adj_list:
                                            for depend_class in adj_list[cur]:
                                                if not depend_class in visited:
                                                    queue.append(depend_class)
                                                    visited.add(depend_class)
                                dis += 1
                            min_dis_dict[selected_test] = dis_test_to_class

                        _, _, test_cases_num_dict = test_list_from_surefile_reports()
                        # print("test cases num dict", test_cases_num_dict)

                        # collect historic change history
                        with IOUtils.cd("src/main/java"):
                            BashUtils.run(f"chmod +x {Macros.python_dir}/pts/processor/change_file.sh")
                            BashUtils.run(f"{Macros.python_dir}/pts/processor/change_file.sh")
                            change_history_dict = collections.Counter()
                            with open("log.log") as fp:
                                for line in fp.readlines():
                                    #print("log.log line", line)
                                    file_name = line.strip().replace(".java", "").split("/")[-1]
                                    change_history_dict[file_name] += 1
                            # print("change history dict", change_history_dict)
        except Exception as e:
            print(e)

        return min_dis_dict, selected_tests, changed_classes, change_history_dict, test_cases_num_dict, compile_time

    '''
    get the failure rate fo a test
    '''
    def get_failure_rate(self, test, failure_rate_dict):
        if test not in failure_rate_dict.keys():
            return 0
        else:
            return failure_rate_dict[test]


    def process_eval(self, src_data_dir: Path, project: str, train_sha=""):
        """Prepare eval data for the model from the real-world shas.
        for augmented data
        Note:
            * will filter those shas without any code diff
            * will filter those shas that ekstazi and starts select nothing
        """
        print(project)
        maven_home = os.getenv('M2_HOME')
        BashUtils.run(f"cp {Macros.tools_dir}/starts-extension-1.0-SNAPSHOT.jar {maven_home}/lib/ext")
        IOUtils.mk_dir(f"{Macros.model_data_dir}/EALRTS/{project}")
        file_name = open(f"{Macros.model_data_dir}/EALRTS/{project}/eval_reducedData.txt", "w+")

        shas_data = IOUtils.load(src_data_dir)
        discard_sha = 0

        failure_rate_dict = {}
        if not os.path.exists(f"{Macros.model_data_dir}/EALRTS/{project}/reducedData.txt"):
            # this project does not have training data
            return
        reduced_data = pd.read_csv(f"{Macros.model_data_dir}/EALRTS/{project}/reducedData.txt", sep=",", header=None, error_bad_lines=False)
        with open(f"{Macros.model_data_dir}/EALRTS/{project}/finalData.txt") as infile:
            for index, line in enumerate(infile):
                line = line.rstrip("\n")
                test_name = line.split(",")[8].strip()
                failure_rate = reduced_data.iat[index, 7]
                failure_rate_dict[test_name] = failure_rate

        sha_data_dict = {}
        # Iterate data to process
        for id, sha in enumerate(shas_data):
            if sha["commit"][:8] in sha_data_dict.keys():
                sha_data_list = sha_data_dict[sha["commit"][:8]]
                for test_data_list in sha_data_list:
                    test_data_list[0] = sha["commit"]
                    print(test_data_list)
                    stringToWrite = ','.join(map(str, test_data_list)) + "\n"
                    file_name.write(stringToWrite)
                continue

            sha_data_list, _ = self.process_eval_for_each_SHA(project, sha, failure_rate_dict)

            # write to file
            for test_data_list in sha_data_list:
                stringToWrite = ','.join(map(str, test_data_list)) + "\n"
                file_name.write(stringToWrite)
            # end for
            sha_data_dict[sha["commit"][:8]] = sha_data_list
        # end for

    def process_eval_for_each_SHA(self, project, sha, failure_rate_dict):
        """
        project: project name
        sha: mutated eval data item in mutated eval data
        """
        sha_data_list = []

        min_dis_dict, selected_tests, changed_classes, change_history_dict, test_cases_num_dict, compile_time = self.dep_graph(
            project.split("_")[0], project.split("_")[-1], sha["prev_commit"][:8], sha["commit"][:8])
        # print("min dis dict length", len(min_dis_dict))

        file_cardinality = len(changed_classes)
        target_cardinality = len(selected_tests)
        # test_cases_num_dict = sha["tests_cases_num"]
        # print("test cases num dict", test_cases_num_dict)

        for test in selected_tests:
            if test in sha["qualified_failed_test_list"]:
                label = 1
            else:
                label = 0

            test_methods = 0
            if test.split(".")[-1] in test_cases_num_dict.keys():
                test_methods = test_cases_num_dict[test.split(".")[-1]]
            # if test methods equal to 0, it means that it just have Test in class name but it is not a test
            if test_methods == 0:
                continue

            test_data_list = []

            test_to_class_dict = min_dis_dict[test]
            min_distance = 0
            if test_to_class_dict:
                min_distance = min(test_to_class_dict.values())
            # And connected file cardinality is how many files given a change transitively depend on a test target.
            connected_test = len(test_to_class_dict)
            # how many times have a file changed.
            changed_history = 0

            for connected_class in test_to_class_dict.keys():
                if "$" in connected_class:
                    connected_class = connected_class[:connected_class.index("$")]
                connected_class = connected_class.split(".")[-1]
                if connected_class in change_history_dict.keys():
                    changed_history += change_history_dict[connected_class]

            failure_rate = self.get_failure_rate(test, failure_rate_dict)  # TODO: get minimal distance, connected_test, changed_history for each test, add failure rate from the training data

            # test_data_list.append(id)
            # Question: Can I use commit here
            # Answer: Yes
            commit = sha["commit"]
            test_data_list.append(commit)
            test_data_list.append(test_methods)
            test_data_list.append(file_cardinality)
            test_data_list.append(target_cardinality)
            test_data_list.append(min_distance)
            test_data_list.append(connected_test)
            test_data_list.append(changed_history)
            test_data_list.append(failure_rate)
            test_data_list.append(label)
            test_data_list.append(test)
            # print(test_data_list)
            sha_data_list.append(test_data_list)
        return sha_data_list, compile_time



