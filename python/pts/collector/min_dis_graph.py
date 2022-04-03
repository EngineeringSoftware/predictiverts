import collections
import sys
import os
import re
from seutil import BashUtils, IOUtils, TimeUtils, TimeoutException
import json
from pts.Macros import Macros
import concurrent.futures

def dep_graph(institution, project, sha, changed_file_list, test_list, full_qualifed_name=True):
    project_folder = Macros.repos_downloads_dir

    num_target_dict = dict()
    # default_min_dis
    default_min_dis = dict()
    for t in test_list:
        default_min_dis[t] = -1
    min_dis_dict = {}
    target_set = set()
    try:
        with TimeUtils.time_limit(1800):
            with IOUtils.cd(f"{project_folder}"):
                if not os.path.isdir(f"{institution}_{project}"):
                    BashUtils.run(f"git clone https://github.com/{institution}/{project}.git {institution}_{project}")
                with IOUtils.cd(f"{institution}_{project}"):
                    BashUtils.run(f"git checkout .")
                    BashUtils.run(f"git clean -xfd")
                    BashUtils.run(f"git checkout {sha}")

                    build_res = BashUtils.run("mvn starts:starts -DstartsLogging=FINEST -DdepFormat=CLZ -Djacoco.skip "
                                              "-Dcheckstyle.skip -Drat.skip -Denforcer.skip -Danimal.sniffer.skip "
                                              "-Dmaven.javadoc.skip -Dfindbugs.skip -Dwarbucks.skip -Dmodernizer.skip "
                                              "-Dimpsort.skip -Dpmd.skip -Dxjc.skip --fail-at-end").stdout
                    if "BUILD FAILURE" in build_res:
                        print("project info: ", institution, project, sha)

                    # create the graph
                    adj_list = collections.defaultdict(set)
                    # deal with multi module
                    graph_files = BashUtils.run("find . -name 'graph'").stdout

                    if not graph_files or graph_files == "":
                        print("can not have .starts: ", institution, project, sha)
                        return min_dis_dict, num_target_dict

                    for graph_file in graph_files.split("\n"):
                        graph_file = graph_file.strip()
                        if graph_file.endswith("/.starts/graph"):
                            with open(graph_file) as dep_relation:
                                for line in dep_relation.readlines():
                                    nodes = line.split(" ")
                                    node1 = nodes[0].strip()
                                    node2 = nodes[1].strip()
                                    adj_list[node2].add(node1)
                    # if os.path.isdir(".starts"):
                    #     BashUtils.run(f"rm -rf .starts")
                    for changed_file in changed_file_list:
                        if not changed_file.endswith(".java"):
                            continue

                        changed_class_file = re.sub(".*src/test/java/|.*src/main/java/", "",changed_file)
                        changed_class_file = changed_class_file.replace(".java", "")
                        changed_class_file = changed_class_file.replace("/", ".")

                        if changed_class_file in test_list:
                            target_set.add(changed_class_file)

                        min_dis_dict[changed_file] = default_min_dis

                        # bfs
                        visited = []
                        dis = 1
                        queue = [changed_class_file]

                        for f in adj_list.keys():
                            if f.startswith(changed_class_file+'$'):
                                queue.append(f)

                        while queue:
                            length = len(queue)
                            for i in range(length):
                                cur = queue.pop(0)

                                next_list = adj_list[cur]
                                for next in next_list:
                                    if next in visited:
                                        continue
                                    if (full_qualifed_name and next in test_list) or (not full_qualifed_name and next.split(".")[-1] in test_list):
                                        target_set.add(next)
                                        min_dis_dict[changed_file][next] = dis
                                    visited.append(next)
                                    queue.append(next)
                            dis += 1
    except Exception as e:
        print(e)

    return min_dis_dict, len(target_set)

def process_graph(project):
    project = project.replace(".json", "")  # ignore the .json
    # for project in Macros.projects:
    try:
        res = []
        with open(f"{Macros.raw_data_dir}/{project}.json") as f:
            json_list = json.load(f)
            print()
            print(f"{project}", len(json_list))
            for json_item in json_list:
                if "num_target" in json_item and type(json_item["num_target"]) is int:
                    res.append(json_item)
                    continue
                # if "num_target" in json_item and json_item["num_target"]:
                #     zeroTarget = True
                #     for classname, number in json_item["num_target"].items():
                #         if number > 0:
                #             zeroTarget = False
                #     if not zeroTarget:
                #         res.append(json_item)
                #         continue
                min_dis_dict, num_target_dict = dep_graph(project.split("_")[0], project.split("_")[1],
                                                          json_item['commit'],
                                                          json_item['changed_files'],
                                                          json_item['passed_test_list'] + json_item[
                                                              'failed_test_list'])
                json_item['min_dis'] = min_dis_dict
                json_item['num_target'] = num_target_dict
                res.append(json_item)
        with open(f"{Macros.raw_data_dir}/{project}.json", 'w', encoding='utf-8') as json_file:
            json.dump(res, json_file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(e)


def main(function_name):
    if function_name == "min_dis":
        jsonfile_list = os.listdir(Macros.raw_data_dir)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(process_graph, jsonfile_list)
