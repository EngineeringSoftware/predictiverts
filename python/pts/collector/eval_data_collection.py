import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from os import listdir
from os.path import isfile, join
from typing import *
from seutil import BashUtils, IOUtils, LoggingUtils, TimeUtils
from pathlib import Path
import json
from pts.Macros import Macros
from pts.collector.FileParser import FileParser
import re
import git
from unidiff import PatchSet
from io import StringIO
import copy
from tqdm import tqdm
import time
import random
from pts.main import proj_logs

import ipdb

# projects = {'apache_commons-dbcp': '7b9fba0be5f9ae7daf55d995b980646838f8975e'}
NUM_SHA = 50
SUREFILE = "target/surefire-reports"
endTestRun = re.compile(
    ".*Tests run: ([0-9]+), Failures: ([0-9]+), Errors: ([0-9]+), Skipped: ([0-9]+).*")

logger = LoggingUtils.get_logger(__name__, LoggingUtils.INFO)
maven_home = os.getenv('M2_HOME')


def compile_project(projects: dict):
    for filename, start_SHA in projects.items():
        try:
            institution = filename.split("_")[0]
            project_name = filename.split("_")[1]
            with IOUtils.cd(Macros.repos_downloads_dir):
                if Path(f"{institution}_{project_name}").is_dir():
                    BashUtils.run(f"rm -rf {institution}_{project_name}")
                BashUtils.run(f"git clone https://github.com/{institution}/{project_name} {institution}_{project_name}")
                with IOUtils.cd(f"{institution}_{project_name}"):
                    compile_log = BashUtils.run(f"mvn test-compile {Macros.SKIPS}").stdout
                    if "BUILD FAILURE" in compile_log:
                        print(f"{project_name} has build failure")
                    else:
                        print(f"{project_name} can build")
        except:
            print(f"{project_name} has error")
            continue


def collect_training_sha_info(projs):
    res_path = Macros.data_dir / "train_tests.json"
    res = []
    for proj in projs:
        res_item = {}
        sha = proj_logs[proj][:8]
        with IOUtils.cd(Macros.repos_downloads_dir / proj):
            date = BashUtils.run("git show -s --format=%ci " + sha).stdout.strip()
            BashUtils.run(f"git checkout {sha}")
            BashUtils.run(f"mvn test {Macros.SKIPS}")
            failed_test_list, passed_test_list, test_cases_num_dict = test_list_from_surefile_reports()
            res_item["project"] = proj
            res_item["sha"] = sha
            res_item["date"] = date
            res_item["failed_test_list"] = failed_test_list
            res_item["passed_test_list"] = passed_test_list
            res_item["test_cases_num_dict"] = test_cases_num_dict
            res.append(res_item)
    IOUtils.dump(res_path, res, IOUtils.Format.jsonPretty)


def collect_date(projs):
    for proj, start_sha in projs.items():
        try:
            mutated_eval_path = Macros.data_dir / "mutated-eval-data" / f"{proj}.json"
            mutated_eval_json = IOUtils.load(mutated_eval_path)
            res = []
            for mutated_eval_item in mutated_eval_json:
                with IOUtils.cd(Macros.repos_downloads_dir / proj):
                    sha = mutated_eval_item["commit"][:8]
                    date = BashUtils.run("git show -s --format=%ci " + sha).stdout.strip()
                    # print("sha", sha, "date", date)
                    mutated_eval_item["date"] = date
                    res.append(mutated_eval_item)
            IOUtils.dump(Macros.data_dir / "mutated-eval-data" / f"{proj}-ag.json", res, IOUtils.Format.jsonPretty)
        except Exception as e:
            print(proj, e)


def get_test_runtime():
    """Parse mvn log file to extract the running time for 'mvn test'"""
    t = time.time()
    BashUtils.run(f"mvn test {Macros.SKIPS}")
    return time.time() - t


def test_list_from_surefile_reports(qualified_name=False):
    failed_test_list = []
    passed_test_list = []
    test_cases_num_dict = {}
    subfolders = sorted([x[0] for x in os.walk(".") if x[0].endswith(SUREFILE)])
    if not subfolders:
        return failed_test_list, passed_test_list, test_cases_num_dict
    # print(len(os.listdir()))
    for surefire_folder in subfolders:
        for test_log in sorted(os.listdir(surefire_folder)):
            if test_log.endswith(".txt"):
                with open(f"{surefire_folder}/{test_log}") as f:
                    for log_line in f.readlines():
                        res = endTestRun.match(log_line)
                        if res:
                            # print(log_line)
                            # print(res.group(2), res.group(3))
                            if not qualified_name:
                                test_name = test_log.split('.')[-2]
                            else:
                                test_name = test_log[0:-len(".txt")]
                            test_cases_num_dict[test_name] = int(res.group(1))
                            # print(res.group(1))
                            if int(res.group(2)) > 0 or int(res.group(3)) > 0:
                                failed_test_list.append(test_name)
                            else:
                                passed_test_list.append(test_name)

    return list(set(failed_test_list)), list(set(passed_test_list)), test_cases_num_dict


def collect_starts_tests_mut(proj: str, branch_id: int, output_dir: Path):
    BashUtils.run(f"cp -r {Macros.repos_downloads_dir}/{proj} {Macros.repos_downloads_dir}/tmp-{branch_id}")
    with IOUtils.cd(f"{Macros.repos_downloads_dir}/tmp-{branch_id}"):
        BashUtils.run(f"git checkout master")
        BashUtils.run(f"git checkout mutation-{branch_id}")
        BashUtils.run(f"mvn clean starts:starts {Macros.SKIPS} -DignoreClassPath -fn")
        starts_failed_test_list, starts_passed_test_list, _ = test_list_from_surefile_reports()
        starts_selected_tests = starts_passed_test_list + starts_failed_test_list
        res = {
            "starts-selected": starts_selected_tests
        }
        IOUtils.dump(output_dir / f"starts-{branch_id}.json", res)
    BashUtils.run(f"rm -rf {Macros.repos_downloads_dir}/tmp-{branch_id}")


comment_pattern = re.compile(r"^[+-]\s*(\*|//|/\*|\*+/)")


def remove_comment(s):
    res = ""
    for line in s.splitlines(keepends=True):
        if line.strip() and not comment_pattern.match(line):
            res += line
    return res


# input:
# pre_SHA: SHA of previous version
# change_file_path: path of changed file
#                   have to make sure that this method is used under the folder of project
#                   for example:
#                   with IOUtils.cd("f{Macros.repos_downloads_dir/apache_commons-lang}"):
#                       f_cd, f_cline = remove_comment_space("4b718f6e", "src/main/java/org/apache/commons/lang3/StringUtils.java")
# output:
# f_cd: code after filtering out comments and space
#        example: public static String[] splitByWholeSeparatorPreserveAllTokens(fina\
# l String str, final String separator) {\n    }\n
#
# f_cl_line_num_final: line number of the changed code after filtering out comments and space
#        example: [7630, 7632]

def remove_comment_space(pre_SHA, change_file_path):
    f_cl_line_num = []
    f_cl_line_num_final = []
    f_cd = ""

    try:
        # sample format of f_cl is
        # 246,0 +247
        # 260,2 +261,2
        f_cl = BashUtils.run(
            f"git diff {pre_SHA} --unified=0 -- {change_file_path} | grep -e '^@@'").stdout

        for f_cl_line_info in change_pattern.findall(f_cl):
            # print(f_cl_line_info)
            try:
                after_revision = f_cl_line_info.strip().split(" ")[1].replace("+", "")
                # print(after_revision)
                if "," in after_revision:
                    start_line_num = int(after_revision.split(",")[0])
                    continue_line_num = int(after_revision.split(",")[1])
                    for l in [start_line_num + i for i in range(continue_line_num)]:
                        f_cl_line_num.append(l)
                else:
                    f_cl_line_num.append(int(after_revision))
            except Exception as e:
                print(e)
                print(f"line: {f_cl_line_info}")
                continue
        if f_cl_line_num:
            if not os.path.exists(change_file_path):
                return f_cd, f_cl_line_num_final
            with open(change_file_path) as java_file:
                java_index = 0
                java_lines = java_file.readlines()
                while java_index < len(java_lines):
                    java_line = java_lines[java_index]
                    if java_line.strip().startswith("//") or java_line == "\n" or not java_line.strip():
                        java_index += 1
                        continue
                    if java_line.strip().startswith("/*"):
                        while not java_lines[java_index].strip().endswith("*/"):
                            java_index += 1
                    if java_index + 1 in f_cl_line_num:  # java_index + 1 because java_index starts with 0
                        # print(f"{change_file}: {java_index+1}")
                        f_cl_line_num_final.append(java_index + 1)
                        f_cd += java_lines[java_index]
                    java_index += 1
        # f_cd = BashUtils.run(
        #     f'git diff {pre_SHA}...{cur_SHA} --unified=0 -- {f} | egrep "^(\+)\s"').stdout
        # f_cd = remove_comment(f_cd)
        # if f_cd:
        #     code_diff_file[f] = f_cd
        # sample output of f_cl is
        # 246,0 +247
        # 260,2 +261,2
    except Exception as e:
        print(e)
    return f_cd, f_cl_line_num_final


def get_start_shas(projects: dict):
    for filename, end_SHA in projects.items():
        if not (Macros.eval_data_dir / "shalist" / f"{filename}.json").exists():
            continue
        with open(Macros.eval_data_dir / "shalist" / f"{filename}.json") as shalist_file:
            shalist = json.load(shalist_file)
            if shalist:
                print(f"{filename} : {shalist[0]}")
                print(len(shalist))


def filter_project_change_lines_and_ekstazi_tests():
    for filename in os.listdir(f'{Macros.raw_eval_data_dir}'):
        jsonlist = IOUtils.load(Macros.raw_eval_data_dir / filename,
                                IOUtils.Format.json)
        newjsonlist = []
        for json_item in jsonlist:
            total_line = 0
            diff_code_per_file = json_item["diff_line_number_list_per_file"]
            for file, line_list in diff_code_per_file.items():
                total_line += len(line_list)
            if total_line < 1000 and total_line > 0:
                if json_item["ekstazi_test_list"]:
                    newjsonlist.append(json_item)
                else:
                    print(filename, json_item["commit"], "ekstazi does not select any test")
            else:
                print(filename, json_item["commit"], total_line)
        if len(newjsonlist) > 0:
            IOUtils.dump(f'{Macros.raw_eval_data_dir}/{filename}', newjsonlist, IOUtils.Format.json)
        else:
            print(filename, "no data")

change_pattern = re.compile(r"@@(.*)@@")
def filter_shas(projects: dict):
    if not Macros.repos_downloads_dir.exists():
        Macros.repos_downloads_dir.mkdir()
    if not Macros.eval_data_dir.exists():
        Macros.eval_data_dir.mkdir()
    if not (Macros.eval_data_dir / "shalist").exists():
        (Macros.eval_data_dir / "shalist").mkdir()
    if not (Macros.tools_dir / "bytecodechecksum" / "target").exists():
        with IOUtils.cd(Macros.tools_dir / "bytecodechecksum"):
            BashUtils.run("mvn package")

    intermediate_result = []
    for filename, start_SHA in projects.items():
        try:
            start_time = time.time()
            institution = filename.split("_")[0]
            project_name = filename.split("_")[1]
            print("current project", project_name)
            shalist = []
            with IOUtils.cd(Macros.repos_downloads_dir):
                # download the project
                if Path(f"{institution}_{project_name}").is_dir():
                    BashUtils.run(f"rm -rf {institution}_{project_name}")
                BashUtils.run(f"git clone https://github.com/{institution}/{project_name} {institution}_{project_name}")

                with IOUtils.cd(f"{institution}_{project_name}"):
                    # the shas got from git log is in reversed chronological order
                    # used to be git log --color=never --first-parent --no-merges --pretty=format:'%H'
                    tempshalist = BashUtils.run(
                        f"git log --color=never --first-parent --pretty=format:'%H'").stdout.split("\n")
                    tempshalist = [sha[:8] for sha in tempshalist]
                    try:
                        start_index = tempshalist.index(start_SHA[:8])
                        if start_index == len(tempshalist)-1:
                            print("there is no more sha before current sha...")
                            continue
                    except ValueError:
                        print("start SHA does not exist...")
                        continue
                    print("start index:", start_index)

                    intermediate_data = {}
                    number_shas_checked = 0
                    number_shas_compile_failure = 0
                    number_shas_no_bytecode_change = 0
                    number_shas_1000_lines = 0
                    number_shas_only_delete_lines = 0

                    for index in range(start_index - 1, -1, -1):
                        try:
                            pre_sha = tempshalist[index+1]
                            cur_sha = tempshalist[index]
                            shalist_item = {}
                            shalist_item["pre_sha"] = pre_sha
                            shalist_item["cur_sha"] = cur_sha
                            if len(shalist) < 50:
                                number_shas_checked += 1

                                BashUtils.run(f"git checkout {pre_sha}")
                                pre_sha_compile_res = BashUtils.run(f"mvn clean test-compile {Macros.SKIPS}").stdout
                                BashUtils.run(f"java -cp {Macros.project_dir}/tools/bytecodechecksum/target/bytecodechecksum-1.0-SNAPSHOT.jar edu.utexas.checksum.Main")

                                BashUtils.run(f"git checkout {cur_sha}")
                                compile_res = BashUtils.run(f"mvn clean test-compile {Macros.SKIPS}").stdout
                                if ("BUILD FAILURE" not in pre_sha_compile_res) and ("BUILD FAILURE" not in compile_res):
                                    check_bytecode_res = BashUtils.run(f"java -cp {Macros.project_dir}/tools/bytecodechecksum/target/bytecodechecksum-1.0-SNAPSHOT.jar edu.utexas.checksum.Main").stdout
                                    print(f"bytecode changes: {check_bytecode_res}")
                                    # condition1: change bytecode
                                    if check_bytecode_res.strip() == "true":
                                        changed_files = BashUtils.run(f"git diff {pre_sha} {cur_sha} --name-only").stdout.split("\n")
                                        print(f"{pre_sha} {cur_sha}")
                                        if "" in changed_files:
                                            changed_files.remove("")
                                        # java file change
                                            num_java_code = 0
                                        for change_file in changed_files:
                                            if change_file.strip().endswith(".java"):
                                                # sample format of f_cl is
                                                # 246,0 +247
                                                # 260,2 +261,2
                                                f_cl = BashUtils.run(
                                                    f"git diff {pre_sha} {cur_sha} --unified=0 -- {change_file} | grep -e '^@@'").stdout

                                                f_cl_line_num = []
                                                for f_cl_line_info in change_pattern.findall(f_cl):
                                                    #print(f_cl_line_info)
                                                    try:
                                                        after_revision = f_cl_line_info.strip().split(" ")[1].replace("+", "")
                                                        #print(after_revision)
                                                        if "," in after_revision:
                                                            start_line_num = int(after_revision.split(",")[0])
                                                            continue_line_num = int(after_revision.split(",")[1])
                                                            for l in [start_line_num + i for i in range(continue_line_num)]:
                                                                f_cl_line_num.append(l)
                                                        else:
                                                            f_cl_line_num.append(int(after_revision))
                                                    except Exception as e:
                                                        print(e)
                                                        print(f"line: {f_cl_line_info}")
                                                        continue
                                                if f_cl_line_num:
                                                    if not os.path.exists(change_file):
                                                        continue
                                                    with open(change_file, errors='ignore') as java_file:
                                                        java_index = 0
                                                        java_lines = java_file.readlines()
                                                        while java_index < len(java_lines):
                                                            java_line = java_lines[java_index]
                                                            if java_line.strip().startswith("//") or java_line == "\n" or not java_line.strip():
                                                                java_index += 1
                                                                continue
                                                            if java_line.strip().startswith("/*"):
                                                                while not java_lines[java_index].strip().endswith("*/"):
                                                                    java_index += 1
                                                                java_index += 1
                                                                continue
                                                            if java_index + 1 in f_cl_line_num: # java_index + 1 because java_index starts with 0
                                                                #print(f"{change_file}: {java_index+1}")
                                                                num_java_code += 1
                                                            java_index += 1
                                        print("num of changed lines:", num_java_code)
                                        # condition 2: no big change
                                        # condition 3: not only delete code
                                        # num_java_code <= 1000 and
                                        if num_java_code > 0:
                                            shalist.append(shalist_item)
                                            if num_java_code > 1000:
                                                number_shas_1000_lines += 1
                                        else:
                                            number_shas_only_delete_lines += 1
                                            print(f"{cur_sha} changes too many lines or changes no line")
                                    else:
                                        number_shas_no_bytecode_change += 1
                                else:
                                    number_shas_compile_failure += 1
                                    print(f"{pre_sha} {cur_sha} build failure")
                            else:
                                print("collect enough data")
                                break
                        except Exception as e:
                            print(e)
                if len(shalist) == 0:
                    print("shalist is empty")
                    continue
                if len(shalist) < 50:
                    print("there are no enough shas")
                print("shalist: ", shalist)
                IOUtils.dump(Macros.eval_data_dir / "shalist" / f"{institution}_{project_name}.json", shalist,
                             IOUtils.Format.json)

            end_time = time.time()
            total_time = end_time - start_time
            intermediate_data["project-name"] = filename
            intermediate_data["time"] = total_time
            intermediate_data["num-sha-checked"] = number_shas_checked
            intermediate_data["num-sha-1000-lines"] = number_shas_1000_lines
            intermediate_data["num-sha-only-delete-lines"] = number_shas_only_delete_lines
            intermediate_data["num-sha-compile-failure"] = number_shas_compile_failure
            intermediate_data["num-sha-no-bytecode-change"] = number_shas_no_bytecode_change
            intermediate_data["num-sha-result"] = len(shalist)
            intermediate_result.append(intermediate_data)
            print(f"{project_name} total time: ", total_time)
        except Exception as e:
            print(e)
    IOUtils.dump(Macros.results_dir / "metrics" / f"intermediate-raw-eval-dataset.json", intermediate_result,
                 IOUtils.Format.json)


def qualified_test_name(projects: dict):
    maven_home = os.getenv('M2_HOME')
    BashUtils.run(f"rm {maven_home}/lib/ext/ekstazi-extension-1.0-SNAPSHOT.jar")
    cache = {}
    for filename, start_SHA in projects.items():
        print(filename)
        try:
            res = []
            institution = filename.split("_")[0]
            project_name = filename.split("_")[1]

            with IOUtils.cd(Macros.repos_downloads_dir):
                if Path(f"{institution}_{project_name}_all").is_dir():
                    BashUtils.run(f"rm -rf {institution}_{project_name}_all")
                BashUtils.run(
                    f"git clone https://github.com/{institution}/{project_name} {institution}_{project_name}_all")

                raw_eval_data_json = IOUtils.load(
                    f'{Macros.eval_data_dir}/mutated-eval-data/{institution}_{project_name}-ag.json',
                    IOUtils.Format.json)

                for raw_eval_data_item in raw_eval_data_json:
                    cur_SHA = raw_eval_data_item["commit"][:8]
                    if cur_SHA in cache.keys():
                        total_tests = cache[cur_SHA]
                    else:
                        with IOUtils.cd(f"{institution}_{project_name}_all"):
                            BashUtils.run(f"git checkout {cur_SHA}")
                            BashUtils.run(f"mvn test {Macros.SKIPS}")
                            origianl_failed_test_list_qualified, origianl_passed_test_list_qualified, _ = test_list_from_surefile_reports(
                                True)
                            total_tests = origianl_passed_test_list_qualified + origianl_failed_test_list_qualified
                            cache[cur_SHA] = total_tests

                    test_to_test_qualified = {}
                    for test in total_tests:
                        test_to_test_qualified[test.split(".")[-1]] = test
                    # print(test_to_test_qualified)
                    # print("passed test list:", len(total_tests))

                    failed_test_list = raw_eval_data_item["failed_test_list"]
                    passed_test_list = raw_eval_data_item["passed_test_list"]

                    failed_test_list_qualified = []
                    passed_test_list_qualified = []
                    for test in failed_test_list:
                        if test in test_to_test_qualified.keys():
                            test_qualified = test_to_test_qualified[test]
                            failed_test_list_qualified.append(test_qualified)
                            if test_qualified in origianl_failed_test_list_qualified:
                                print(cur_SHA, test, "failed without mutation")
                        else:
                            print(cur_SHA, test, "deos not exist")
                    for test in passed_test_list:
                        if test in test_to_test_qualified.keys():
                            passed_test_list_qualified.append(test_to_test_qualified[test])
                        else:
                            print(cur_SHA, test, "deos not exist")

                    raw_eval_data_item["qualified_failed_test_list"] = failed_test_list_qualified
                    raw_eval_data_item["qualified_passed_test_list"] = passed_test_list_qualified
                    # print("failed_test_list_qualified", failed_test_list_qualified)
                    # print("passed_test_list_qualified", passed_test_list_qualified)
                    res.append(raw_eval_data_item)
            if not Macros.raw_eval_data_adding_time_dir.exists():
                Macros.raw_eval_data_adding_time_dir.mkdir()
            with open(f'{Macros.eval_data_dir}/mutated-eval-data/{institution}_{project_name}-ag-qualifiedname.json',
                      'w') as res_file:
                json.dump(res, res_file, indent=4)
        except Exception as e:
            print(filename, e)
            continue


def diff_per_file_for_each_SHA(pre_SHA, cur_SHA, deleted_data = False):
    code_diff_file = {}
    diff_line_number_list_per_file = {}
    BashUtils.run(f'git checkout {cur_SHA}')
    # Extracted diff_code tends to be incorrect
    # For example,
    # project: Bukkit_Bukkit
    # git diff 4b0e6ba6 43d61f13 --unified=0 | egrep "^(\+)"
    # +++ b/src/main/java/org/bukkit/inventory/InventoryView.java
    # +import org.bukkit.GameMode;
    # +        if (getPlayer().getGameMode() == GameMode.CREATIVE && getType() == InventoryType.PLAYER) {
    # +            return slot;
    # +        }
    # However, git diff 4b0e6ba6 43d61f13 --unified=0 | egrep "^(\+)\s"
    # +        if (getPlayer().getGameMode() == GameMode.CREATIVE && getType() == InventoryType.PLAYER) {
    # +            return slot;
    # +        }
    diff_code = BashUtils.run(
        f'git diff {pre_SHA} {cur_SHA} --unified=0 | egrep "^(\+)\s"').stdout
    diff_code = remove_comment(diff_code)
    changed_java_file_list = []

    changed_file_list = BashUtils.run(
        f"git diff --name-only {pre_SHA} {cur_SHA}").stdout.split("\n")
    if "" in changed_file_list:
        changed_file_list.remove("")
    print("changed file list:", changed_file_list)
    # add dict to store code diff per file dict
    if deleted_data:
        BashUtils.run(f'git checkout {pre_SHA}')
    for change_file in changed_file_list:
        if change_file.strip().endswith(".java"):
            changed_java_file_list.append(change_file.strip())
            # sample format of f_cl is
            # 246,0 +247
            # 260,2 +261,2
            f_cl = BashUtils.run(
                f"git diff {pre_SHA} {cur_SHA} --unified=0 -- {change_file} | grep -e '^@@'").stdout
            f_cl_line_num = []
            f_cl_line_num_final = []
            f_cd = ""
            for f_cl_line_info in change_pattern.findall(f_cl):
                # print(f_cl_line_info)
                try:
                    if not deleted_data:
                        revised = f_cl_line_info.strip().split(" ")[1].replace("+", "")
                    else:
                        # before_revision
                        revised = f_cl_line_info.strip().split(" ")[0].replace("-", "")
                    # print(after_revision)
                    if "," in revised:
                        start_line_num = int(revised.split(",")[0])
                        continue_line_num = int(revised.split(",")[1])
                        for l in [start_line_num + i for i in range(continue_line_num)]:
                            f_cl_line_num.append(l)
                    else:
                        f_cl_line_num.append(int(revised))
                except Exception as e:
                    print(e)
                    print(f"line: {f_cl_line_info}")
                    continue
            if f_cl_line_num:
                if not os.path.exists(change_file):
                    continue
                with open(change_file, errors='ignore') as java_file:
                    java_index = 0
                    java_lines = java_file.readlines()
                    while java_index < len(java_lines):
                        java_line = java_lines[java_index]
                        if java_line.strip().startswith("//") or java_line == "\n" or not java_line.strip():
                            java_index += 1
                            continue
                        if java_line.strip().startswith("/*"):
                            while not java_lines[java_index].strip().endswith("*/"):
                                java_index += 1
                            java_index += 1
                            continue
                        if java_index + 1 in f_cl_line_num:  # java_index + 1 because java_index starts with 0
                            # print(f"{change_file}: {java_index+1}")
                            f_cl_line_num_final.append(java_index + 1)
                            f_cd += java_lines[java_index]
                        java_index += 1
            # f_cd = BashUtils.run(
            #     f'git diff {pre_SHA}...{cur_SHA} --unified=0 -- {f} | egrep "^(\+)\s"').stdout
            # f_cd = remove_comment(f_cd)
            # if f_cd:
            #     code_diff_file[f] = f_cd
            # sample output of f_cl is
            # 246,0 +247
            # 260,2 +261,2
            if f_cd != "":
                code_diff_file[change_file] = f_cd
            if f_cl_line_num_final:
                diff_line_number_list_per_file[change_file] = f_cl_line_num_final
            # print(change_file, f_cl_line_num_final)
            # print(f_cd)
    return changed_file_list, diff_code, code_diff_file, diff_line_number_list_per_file, changed_java_file_list


def eval_data_for_each_SHA(shalist, index, institution, project_name):
    try:
        with TimeUtils.time_limit(600):
            # init
            res_item = {}
            cur_SHA = shalist[index]["cur_sha"]
            pre_SHA = shalist[index]["pre_sha"]

            print("current SHA:", cur_SHA)

            with IOUtils.cd(f"{institution}_{project_name}"):
                # does not include deleted code
                changed_file_list, diff_code, code_diff_file, diff_line_number_list_per_file, changed_java_file_list = \
                    diff_per_file_for_each_SHA(pre_SHA, cur_SHA)
                _, _, deleted_code_diff_file, deleted_line_number_list_per_file, _ = diff_per_file_for_each_SHA(pre_SHA, cur_SHA, True)
                # print("code_diff", code_diff_file)
                # end for
                BashUtils.run(f"git checkout {cur_SHA}")
                BashUtils.run(f"mvn clean test {Macros.SKIPS}")
                failed_test_list, passed_test_list, test_cases_num_dict = test_list_from_surefile_reports()
                print("passed test list:", len(passed_test_list))

            print("going to run ekstazi...")

            BashUtils.run(f"cp {Macros.tools_dir}/ekstazi-extension-1.0-SNAPSHOT.jar {maven_home}/lib/ext")
            print(os.listdir(f"{maven_home}/lib/ext"))
            with IOUtils.cd(f"{institution}_{project_name}_ekstazi"):
                BashUtils.run(f"git checkout {pre_SHA}")
                BashUtils.run(f"mvn clean ekstazi:ekstazi {Macros.SKIPS}")
                BashUtils.run(f"git checkout {cur_SHA}")

                BashUtils.run(f"mvn clean test-compile {Macros.SKIPS}")

                ekstazi_select_start = time.time()
                BashUtils.run(f"mvn ekstazi:select {Macros.SKIPS}")
                ekstazi_select_end = time.time()
                ekstazi_select_time = ekstazi_select_end - ekstazi_select_start

                ekstazi_total_start = time.time()
                BashUtils.run(f"mvn clean ekstazi:ekstazi {Macros.SKIPS}")
                ekstazi_total_end = time.time()
                ekstazi_total_time = ekstazi_total_end - ekstazi_total_start
                ekstazi_failed_test_list, ekstazi_passed_test_list, _ = test_list_from_surefile_reports()
                print("ekstazi passed test list:", len(ekstazi_passed_test_list))
            BashUtils.run(f"rm {maven_home}/lib/ext/ekstazi-extension-1.0-SNAPSHOT.jar")

            print("going to run starts...")

            with IOUtils.cd(f"{institution}_{project_name}_starts"):
                BashUtils.run(f"git checkout {pre_SHA}")
                BashUtils.run(f"mvn clean starts:starts {Macros.SKIPS} -DignoreClassPath -fn")
                BashUtils.run(f"git checkout {cur_SHA}")

                BashUtils.run(f"mvn clean test-compile {Macros.SKIPS}")

                starts_select_start = time.time()
                BashUtils.run(f"mvn starts:select {Macros.SKIPS}")
                starts_select_end = time.time()
                starts_select_time = starts_select_end - starts_select_start

                starts_total_start = time.time()
                BashUtils.run(f"mvn clean starts:starts {Macros.SKIPS} -DignoreClassPath -fn")
                starts_total_end = time.time()
                starts_total_time = starts_total_end - starts_total_start
                starts_failed_test_list, starts_passed_test_list, _ = test_list_from_surefile_reports()
                print("starts passed test list:", len(starts_passed_test_list))

            res_item["commit"] = cur_SHA
            res_item["prev_commit"] = pre_SHA
            res_item["changed_files"] = changed_file_list
            res_item["changed_files_num"] = len(changed_file_list)
            res_item["changed_java_files"] = changed_java_file_list
            res_item["changed_java_files_num"] = len(changed_java_file_list)
            res_item["failed_test_list"] = failed_test_list
            res_item["passed_test_list"] = passed_test_list
            res_item["tests_cases_num"] = test_cases_num_dict
            res_item["ekstazi_test_list"] = ekstazi_passed_test_list
            res_item["ekstazi_failed_test_list"] = ekstazi_failed_test_list
            res_item["starts_test_list"] = starts_passed_test_list
            res_item["starts_failed_test_list"] = starts_failed_test_list
            res_item["diff_code"] = diff_code
            res_item["diff_per_file"] = code_diff_file
            res_item["diff_line_number_list_per_file"] = diff_line_number_list_per_file
            res_item["starts_total_time"] = starts_total_time
            res_item["ekstazi_total_time"] = ekstazi_total_time
            res_item["ekstazi_select_time"] = ekstazi_select_time
            res_item["starts_select_time"] = starts_select_time
            res_item["deleted_diff_per_file"] = deleted_code_diff_file
            res_item["deleted_line_number_list_per_file"] = deleted_line_number_list_per_file
            return res_item
    except Exception as e:
        print(e)
        return {}


def main(projects: dict):
    if not Macros.repos_downloads_dir.exists():
        Macros.repos_downloads_dir.mkdir()
    if not Macros.eval_data_dir.exists():
        Macros.eval_data_dir.mkdir()
    if not Macros.raw_eval_data_dir.exists():
        Macros.raw_eval_data_dir.mkdir()
    for filename, start_SHA in projects.items():
        try:
            start_time = time.time()
            res = []
            institution = filename.split("_")[0]
            project_name = filename.split("_")[1]

            print(project_name)
            with IOUtils.cd(Macros.repos_downloads_dir):
                if Path(f"{institution}_{project_name}").is_dir():
                    BashUtils.run(f"rm -rf {institution}_{project_name}")
                BashUtils.run(f"git clone https://github.com/{institution}/{project_name} {institution}_{project_name}")
                if Path(f"{institution}_{project_name}_ekstazi").is_dir():
                    BashUtils.run(f"rm -rf {institution}_{project_name}_ekstazi")
                BashUtils.run(
                    f"git clone https://github.com/{institution}/{project_name} {institution}_{project_name}_ekstazi")
                if Path(f"{institution}_{project_name}_starts").is_dir():
                    BashUtils.run(f"rm -rf {institution}_{project_name}_starts")
                BashUtils.run(
                    f"git clone https://github.com/{institution}/{project_name} {institution}_{project_name}_starts")

                if not (Macros.eval_data_dir / "shalist" / f"{institution}_{project_name}.json").exists():
                    # filter_shas(projects)
                    print(f"{institution}_{project_name}.json does not exist")
                else:
                    shalist = IOUtils.load(Macros.eval_data_dir / "shalist" / f"{institution}_{project_name}.json",
                                           IOUtils.Format.json)
                    BashUtils.run(f"rm {maven_home}/lib/ext/ekstazi-extension-1.0-SNAPSHOT.jar")
                    for index in range(0, len(shalist)):
                        res_item = eval_data_for_each_SHA(shalist, index, institution, project_name)
                        res.append(res_item)
                print(f"In total {len(res)} examples collected.")
                with open(f'{Macros.raw_eval_data_dir}/{institution}_{project_name}.json', 'w') as res_file:
                    json.dump(res, res_file, indent=4)
                end_time = time.time()
                total_time = end_time - start_time
                print(f"{project_name} total time: ", total_time)
        except Exception as e:
            print(e)


def collect_eval_data_adding_delete(projects: dict):
    if not Macros.repos_downloads_dir.exists():
        Macros.repos_downloads_dir.mkdir()
    if not Macros.eval_data_dir.exists():
        Macros.eval_data_dir.mkdir()
    if not Macros.raw_eval_data_no_dep_updated_dir.exists():
        Macros.raw_eval_data_no_dep_updated_dir.mkdir()

    for filename, start_SHA in projects.items():
        try:
            start_time = time.time()
            res = []
            original_res = []
            institution = filename.split("_")[0]
            project_name = filename.split("_")[1]
            if (Macros.raw_eval_data_no_dep_updated_dir / f'{institution}_{project_name}.json').exists():
                original_res = IOUtils.load(
                    f'{Macros.raw_eval_data_no_dep_updated_dir}/{institution}_{project_name}.json')
            print(project_name)
            with IOUtils.cd(Macros.repos_downloads_dir):
                if Path(f"{institution}_{project_name}").is_dir():
                    BashUtils.run(f"rm -rf {institution}_{project_name}")
                BashUtils.run(f"git clone https://github.com/{institution}/{project_name} {institution}_{project_name}")

                for res_item in original_res:
                    try:
                        with TimeUtils.time_limit(600):
                            # init
                            deleted_line_number_list_per_file = {}
                            cur_SHA = res_item["commit"]
                            pre_SHA = res_item["prev_commit"]
                            with IOUtils.cd(f"{institution}_{project_name}"):
                                changed_file_list = BashUtils.run(
                                    f"git diff --name-only {pre_SHA} {cur_SHA}").stdout.split("\n")
                                if "" in changed_file_list:
                                    changed_file_list.remove("")
                                print("changed file list:", changed_file_list)
                                # add dict to store code diff per file dict
                                for change_file in changed_file_list:
                                    if change_file.strip().endswith(".java"):
                                        # sample format of f_cl is
                                        # 246,0 +247
                                        # 260,2 +261,2
                                        f_cl = BashUtils.run(
                                            f"git diff {pre_SHA} {cur_SHA} --unified=0 -- {change_file} | grep -e '^@@'").stdout

                                        f_cl_line_num = []
                                        f_cl_line_num_final = []
                                        for f_cl_line_info in change_pattern.findall(f_cl):
                                            # print(f_cl_line_info)
                                            try:
                                                after_revision = f_cl_line_info.strip().split(" ")[0].replace("-", "")
                                                if "," in after_revision:
                                                    start_line_num = int(after_revision.split(",")[0])
                                                    continue_line_num = int(after_revision.split(",")[1])
                                                    for l in [start_line_num + i for i in range(continue_line_num)]:
                                                        f_cl_line_num.append(l)
                                                else:
                                                    f_cl_line_num.append(int(after_revision))
                                            except Exception as e:
                                                print(e)
                                                print(f"line: {f_cl_line_info}")
                                                continue

                                        if f_cl_line_num:
                                            BashUtils.run(f"git checkout {pre_SHA}")
                                            if os.path.exists(change_file):
                                                with open(change_file) as java_file:
                                                    java_index = 0
                                                    java_lines = java_file.readlines()
                                                    while java_index < len(java_lines):
                                                        java_line = java_lines[java_index]
                                                        if java_line.strip().startswith(
                                                                "//") or java_line == "\n" or not java_line.strip():
                                                            java_index += 1
                                                            continue
                                                        if java_line.strip().startswith("/*"):
                                                            while not java_lines[java_index].strip().endswith("*/"):
                                                                java_index += 1
                                                        if java_index + 1 in f_cl_line_num:  # java_index + 1 because java_index starts with 0
                                                            # print(f"{change_file}: {java_index+1}")
                                                            f_cl_line_num_final.append(java_index + 1)
                                                        java_index += 1
                                        if f_cl_line_num_final:
                                            deleted_line_number_list_per_file[change_file] = f_cl_line_num_final
                                            print("deleted lines", change_file, f_cl_line_num_final)

                            res_item["deleted_line_number_list_per_file"] = deleted_line_number_list_per_file
                            if "ekstazi_test_list" in res_item.keys():
                                res_item["ekstazi_test_list_no_dep_update"] = res_item.pop("ekstazi_test_list")
                            if "ekstazi_failed_test_list" in res_item.keys():
                                res_item["ekstazi_failed_test_list_no_dep_update"] = res_item.pop(
                                    "ekstazi_failed_test_list")
                            if "starts_test_list" in res_item.keys():
                                res_item["starts_test_list_no_dep_update"] = res_item.pop("starts_test_list")
                            if "starts_failed_test_list" in res_item.keys():
                                res_item["starts_failed_test_list_no_dep_update"] = res_item.pop(
                                    "starts_failed_test_list")

                            res.append(res_item)

                    except Exception as e:
                        print(e)
                        continue
                print(f"In total {len(res)} examples collected.")
                with open(f'{Macros.raw_eval_data_adding_deleted_dir}/{institution}_{project_name}.json',
                          'w') as res_file:
                    json.dump(res, res_file, indent=4)
                end_time = time.time()
                total_time = end_time - start_time
                print(f"{project_name} total time: ", total_time)
        except Exception as e:
            print(e)


def collect_eval_data_without_updating_dependencies(projects: dict, proj_logs: dict):
    if not Macros.repos_downloads_dir.exists():
        Macros.repos_downloads_dir.mkdir()
    if not Macros.eval_data_dir.exists():
        Macros.eval_data_dir.mkdir()
    if not Macros.raw_eval_data_no_dep_updated_dir.exists():
        Macros.raw_eval_data_no_dep_updated_dir.mkdir()
    maven_home = os.getenv('M2_HOME')

    for filename, start_SHA in projects.items():
        try:
            start_time = time.time()
            training_sha = proj_logs[filename]
            res = []
            original_res = []
            institution = filename.split("_")[0]
            project_name = filename.split("_")[1]
            if (Macros.raw_eval_data_dir / f'{institution}_{project_name}.json').exists():
                original_res = IOUtils.load(f'{Macros.raw_eval_data_dir}/{institution}_{project_name}.json')
            print(project_name)
            with IOUtils.cd(Macros.repos_downloads_dir):
                if Path(f"{institution}_{project_name}").is_dir():
                    BashUtils.run(f"rm -rf {institution}_{project_name}")
                BashUtils.run(f"git clone https://github.com/{institution}/{project_name} {institution}_{project_name}")
                if Path(f"{institution}_{project_name}_ekstazi").is_dir():
                    BashUtils.run(f"rm -rf {institution}_{project_name}_ekstazi")
                BashUtils.run(
                    f"git clone https://github.com/{institution}/{project_name} {institution}_{project_name}_ekstazi")
                with IOUtils.cd(f"{institution}_{project_name}_ekstazi"):
                    BashUtils.run(f"cp {Macros.tools_dir}/ekstazi-extension-1.0-SNAPSHOT.jar {maven_home}/lib/ext")
                    BashUtils.run(f"git checkout {training_sha}")
                    BashUtils.run(f"mvn clean ekstazi:ekstazi {Macros.SKIPS}")
                    BashUtils.run(f"cp -r .ekstazi .ekstazi_c")

                if Path(f"{institution}_{project_name}_starts").is_dir():
                    BashUtils.run(f"rm -rf {institution}_{project_name}_starts")
                BashUtils.run(
                    f"git clone https://github.com/{institution}/{project_name} {institution}_{project_name}_starts")
                with IOUtils.cd(f"{institution}_{project_name}_starts"):
                    BashUtils.run(f"rm {maven_home}/lib/ext/ekstazi-extension-1.0-SNAPSHOT.jar")
                    BashUtils.run(f"git checkout {training_sha}")
                    BashUtils.run(f"mvn clean starts:starts {Macros.SKIPS} -DignoreClassPath -fn")
                    BashUtils.run(f"cp -r .starts .starts_c")

                if not (Macros.eval_data_dir / "shalist" / f"{institution}_{project_name}.json").exists():
                    print(f"{institution}_{project_name}.json does not exist")
                    filter_shas({filename: start_SHA})

                shalist = IOUtils.load(Macros.eval_data_dir / "shalist" / f"{institution}_{project_name}.json",
                                       IOUtils.Format.json)
                BashUtils.run(f"rm {maven_home}/lib/ext/ekstazi-extension-1.0-SNAPSHOT.jar")
                for index in range(0, len(shalist)):
                    try:
                        with TimeUtils.time_limit(600):
                            # init
                            cur_SHA = shalist[index]["cur_sha"]

                            res_item = {}
                            for original_res_item in original_res:
                                if original_res_item["commit"] == cur_SHA:
                                    res_item = original_res_item
                                    break

                            print("current SHA:", cur_SHA)

                            print("going to run ekstazi...")

                            BashUtils.run(
                                f"cp {Macros.tools_dir}/ekstazi-extension-1.0-SNAPSHOT.jar {maven_home}/lib/ext")
                            print(os.listdir(f"{maven_home}/lib/ext"))
                            with IOUtils.cd(f"{institution}_{project_name}_ekstazi"):
                                BashUtils.run(f"rm -rf .ekstazi")
                                BashUtils.run(f"cp -r .ekstazi_c .ekstazi")
                                BashUtils.run(f"git checkout {cur_SHA}")
                                BashUtils.run(f"mvn clean ekstazi:ekstazi {Macros.SKIPS}")
                                ekstazi_failed_test_list, ekstazi_passed_test_list, _ = test_list_from_surefile_reports()
                                print("ekstazi passed test list:", len(ekstazi_passed_test_list))
                            BashUtils.run(f"rm {maven_home}/lib/ext/ekstazi-extension-1.0-SNAPSHOT.jar")

                            print("going to run starts...")

                            with IOUtils.cd(f"{institution}_{project_name}_starts"):
                                BashUtils.run(f"rm -rf .starts")
                                BashUtils.run(f"cp -r .starts_c .starts")
                                BashUtils.run(f"git checkout {cur_SHA}")
                                BashUtils.run(f"mvn clean starts:starts {Macros.SKIPS} -DignoreClassPath -fn")
                                starts_failed_test_list, starts_passed_test_list, _ = test_list_from_surefile_reports()
                                print("starts passed test list:", len(starts_passed_test_list))

                            res_item["ekstazi_test_list_no_dep_update"] = ekstazi_passed_test_list
                            res_item["ekstazi_failed_test_list_no_dep_update"] = ekstazi_failed_test_list
                            res_item["starts_test_list_no_dep_update"] = starts_passed_test_list
                            res_item["starts_failed_test_list_no_dep_update"] = starts_failed_test_list
                            res.append(res_item)

                    except Exception as e:
                        print(e)
                        continue
                print(f"In total {len(res)} examples collected.")
                with open(f'{Macros.raw_eval_data_no_dep_updated_dir}/{institution}_{project_name}.json',
                          'w') as res_file:
                    json.dump(res, res_file, indent=4)
                end_time = time.time()
                total_time = end_time - start_time
                print(f"{project_name} total time: ", total_time)
        except Exception as e:
            print(e)


def extract_changed_lines(project: str, old_sha: str, new_sha: str):
    """Given the project name, old sha and new sha, return the changed list with line numbers.
    Return: [(file_name, [row_number_of_deleted_line], [row_number_of_added_lines]), ... ]
    """
    repository = git.Repo(f"{Macros.repos_downloads_dir / project}")
    uni_diff_text = repository.git.diff(old_sha, new_sha,
                                        ignore_blank_lines=True,
                                        ignore_space_at_eol=True)
    patch_set = PatchSet(StringIO(uni_diff_text))

    change_list = []  # list of changes
    # [(file_name, [row_number_of_deleted_line],
    # [row_number_of_added_lines]), ... ]
    for patched_file in patch_set:
        file_path = patched_file.path  # file name
        # print('file name :' + file_path)
        ad_line_no = [line.target_line_no
                      for hunk in patched_file for line in hunk
                      if line.is_added and
                      line.value.strip() != '']  # the row number of deleted lines
        del_line_no = [line.source_line_no for hunk in patched_file
                       for line in hunk if line.is_removed and
                       line.value.strip() != '']  # the row number of added liens
        # print('added lines : ' + str(ad_line_no))
        # print('deleted lines : ' + str(del_line_no))
        change_list.append((file_path, del_line_no, ad_line_no))
    return change_list


def extract_fail_tests(project_dir: str, previous_sha: str, current_sha, indx):
    new_data = {}
    ppid = os.getpid()
    BashUtils.run(
        f"cp -r  {Macros.repos_downloads_dir}/{project_dir} {Macros.repos_downloads_dir}/{project_dir}_{ppid}",
        expected_return_code=0)
    with IOUtils.cd(f"{Macros.repos_downloads_dir}/{project_dir}_{ppid}"):
        # See whether it leads to failures
        test_return_code = BashUtils.run(f"mvn test {Macros.SKIPS}").return_code
        if test_return_code == 0:
            return new_data
        # Extract failed tests from surefire report
        failed_test_list, passed_test_list, test_cases_num_dict = test_list_from_surefile_reports()
        if len(failed_test_list) == 0:
            return new_data
        diff_code = BashUtils.run(f'git diff {previous_sha} --unified=0 | egrep "^(\+|-)\s"').stdout
        changed_file_list = BashUtils.run(
            f"git diff --name-only {previous_sha}").stdout.split("\n")
        if "" in changed_file_list:
            changed_file_list.remove("")
        # add dict to store code diff per file dict
        code_diff_file = {}
        for f in changed_file_list:
            f_cd = BashUtils.run(
                f'git diff {previous_sha} --unified=0 -- {f} | egrep "^(\+|-)\s"').stdout
            code_diff_file[f] = f_cd
        new_data["commit"] = f"{current_sha}-{indx}"
        new_data["changed_files"] = changed_file_list
        new_data["diff_code"] = diff_code
        new_data["diff_per_file"] = code_diff_file
        new_data["failed_test_list"] = failed_test_list
        new_data["passed_test_list"] = passed_test_list
        return [new_data]


def add_prev_sha(data_list):
    for i in tqdm(range(1, len(data_list))):
        data_list[i]["prev_commit"] = data_list[i - 1]["commit"]
    return data_list


def run_universal_mutator(project: str, sha_data: Dict) -> List:
    """Run UniversalMutator on one SHA.

    For each sha, run UniversalMutator on the each changed line of each changed file and get the failed tests.
    """
    results = []
    pid = os.getpid()
    new_data = copy.deepcopy(sha_data)
    # get info for this sha
    current_sha = sha_data["commit"]
    previous_sha = sha_data["prev_commit"]
    diff_per_file = sha_data["diff_per_file"]
    actual_changed_files = list(diff_per_file.keys())

    logger.info(f"[pid:{pid}] Current mutated sha is {current_sha}")

    # if changed lines are greater than 1000 in this sha, discard it
    changed_lines_per_sha = 0
    for filename, changed_code in diff_per_file.items():
        if filename.endswith(".java"):
            changed_lines_per_sha += len(changed_code.splitlines())
    if changed_lines_per_sha > 1000:
        logger.info(f"{project} in {current_sha} changes more than 1000 lines, skip it.")
        return results

    logger.info(f"[pid:{pid}] {current_sha} In total {len(actual_changed_files)} files changed.")

    BashUtils.run(f"cp -r {Macros.repos_downloads_dir}/{project} {Macros.repos_downloads_dir}/{project}_{pid}",
                  expected_return_code=0)

    indx = 0
    with IOUtils.cd(f"{Macros.repos_downloads_dir}/{project}_{pid}"):
        BashUtils.run(f"git clean -fd", expected_return_code=0)
        BashUtils.run(f"git checkout -f {current_sha}", expected_return_code=0)
        # get the time in seconds to run 'mvn test'
        try:
            test_run_time = get_test_runtime()
        except:
            logger.warning(f"Error found while running mvn test, exit!")
            return []
        random.shuffle(actual_changed_files)
        for i, ch_file in enumerate(actual_changed_files[:5]):
            BashUtils.run(f"git checkout -f {current_sha}", expected_return_code=0)
            # start to look at each changed file
            if not ch_file.endswith(".java") or "src/test" in ch_file or ch_file not in diff_per_file or ch_file not in \
                    sha_data["diff_line_number_list_per_file"]:
                continue
            logger.info(f"[pid:{pid}] {current_sha} Valid changed file is {ch_file}")

            mutated_lines = sha_data["diff_line_number_list_per_file"][ch_file]

            # write the lines going to be mutated to files
            IOUtils.rm_dir(Path("./mutatedFiles"), ignore_non_exist=True)
            mutated_path = "./mutatedFiles"
            IOUtils.mk_dir(mutated_path)
            # To ensure num of mutants not greater than 5
            random.shuffle(mutated_lines)
            for l in mutated_lines:
                with open("./mut_lines.txt", "w+") as f:
                    f.write(str(l) + " ")
                # end with
                # start to run the tool on one particular changed file
                try:
                    BashUtils.run(f"python3 {Macros.mutator_dir}/genmutants.py {ch_file} --cmd 'mvn test-compile' "
                                  f"--mutantDir ./mutatedFiles --lines ./mut_lines.txt", expected_return_code=0)
                except RuntimeError:
                    logger.warning("RuntimeError while running universalmutator!")
                    continue
                if len(listdir(mutated_path)) > 3:
                    break
            # end for
            # iterate all generated mutants and save them for one evaluation datap
            if len(listdir(mutated_path)) == 0:
                continue
            onlyfiles = [join(mutated_path, f) for f in listdir(mutated_path) if isfile(join(mutated_path, f))]
            logger.info(
                f"[pid:{pid}] {current_sha} {i + 1}/{len(actual_changed_files)} mutation finishes. {len(onlyfiles)} mutants are generated.")
            for mf in onlyfiles:
                # Make the mutation
                BashUtils.run(f"cp {mf} {ch_file}", expected_return_code=0)
                # See whether it leads to failures
                time_limit = int(test_run_time * 2 + 10)
                with TimeUtils.time_limit(time_limit):
                    try:
                        test_return_code = BashUtils.run(f"mvn clean test {Macros.SKIPS}").return_code
                    except Exception:
                        logger.warning(f"Timed out after {time_limit} seconds. Ignore this mutant {mf}.")
                        test_return_code = 0
                if test_return_code == 0:
                    continue
                # Extract failed tests from surefire report
                failed_test_list, passed_test_list, test_cases_num_dict = test_list_from_surefile_reports()
                real_failed_test_list = list(set(failed_test_list).intersection(sha_data["passed_test_list"]))
                if len(real_failed_test_list) == 0:
                    continue
                diff_code = remove_comment(
                    BashUtils.run(f'git diff {previous_sha} --unified=0 | egrep "^(\+|-)\s"').stdout)
                code_diff_file = {}
                for f in sha_data["diff_per_file"].keys():
                    if f != ch_file:
                        code_diff_file[f] = sha_data["diff_per_file"][f]
                    else:
                        f_cd, _ = remove_comment_space(previous_sha, ch_file)
                        code_diff_file[f] = f_cd
                new_data["commit"] = f"{current_sha}-{indx}"
                new_data["changed_files"] = sha_data["changed_files"]
                new_data["diff_code"] = diff_code
                new_data["diff_per_file"] = code_diff_file
                new_data["failed_test_list"] = real_failed_test_list
                new_data["passed_test_list"] = passed_test_list
                results.append(copy.deepcopy(new_data))
                indx += 1
                BashUtils.run(f"git checkout -- {ch_file}")
                # IOUtils.dump(Macros.eval_data_dir/f"{project}-ag-{pid}.json", results)
            # end for
            BashUtils.run("git clean -d -f")  # remove untracked changes
        # end for
    BashUtils.run(f"rm -rf {Macros.repos_downloads_dir}/{project}_{pid}", expected_return_code=0)
    return results


def mutate_git_patch_multiprocess(project: str):
    """Use https://github.com/agroce/universalmutator to mutate git patch.

    For each SHA, extract those changed files and the changed lines, then mutate those lines. 
    For each mutant, collect the results.
    """
    mutated_eval_data_dir = Macros.eval_data_dir / "mutated-eval-data"
    # set up logger
    logging_file = mutated_eval_data_dir / f"{project}-universal-mutator-log.txt"
    if logging_file.exists():
        BashUtils.run(f"rm {logging_file}")
    LoggingUtils.setup(filename=str(logging_file))
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.INFO)
    eval_data_list = IOUtils.load(Macros.eval_data_dir / "raw-eval-data" / f"{project}.json")
    final_results = []
    logger.info(f"Start mutating git diff for {project}, in total {len(eval_data_list)} shas for mutation...")
    with ProcessPoolExecutor(12) as executor:
        all_process = [executor.submit(run_universal_mutator, project, eval_data) for eval_data in eval_data_list]
        for p in tqdm(as_completed(all_process), total=len(all_process)):
            final_results += list(p.result())
    IOUtils.dump(mutated_eval_data_dir / f"{project}-ag-{len(final_results)}.json", final_results,
                 IOUtils.Format.jsonNoSort)
    logger.info(f"Finish mutating git diff for {project}...")


def collect_eval_data_defects4j(project: str):
    """This function is used for collecting data for projects in defects4j

    Collect eval data for one revision for one project.
    """

    res_item = {}
    proj_name = project.split('_')[1].split('-')[0]
    example_number = project.split('_')[1].split('-')[1]
    # Prepare directories to run experiments
    with IOUtils.cd(Macros.repos_downloads_dir):
        BashUtils.run(f"cp -r {project} {project}_ekstazi")
        BashUtils.run(f"cp -r {project} {project}_starts")
    # end with
    # first checkout to the buggy version
    with IOUtils.cd(Macros.repos_downloads_dir / project):
        pre_sha = f"D4J_{proj_name}_{example_number}_FIXED_VERSION"
        cur_sha = f"D4J_{proj_name}_{example_number}_BUGGY_VERSION"
        changed_file_list, diff_code, code_diff_file, diff_line_number_list_per_file = diff_per_file_for_each_SHA(
            pre_sha, cur_sha)
        BashUtils.run(f"git checkout -f {cur_sha}")
        BashUtils.run(f"mvn clean test {Macros.SKIPS}")
        failed_test_list, passed_test_list, test_cases_num_dict = test_list_from_surefile_reports()
        print("passed test list:", len(passed_test_list))
    # end with
    # Run Ekstazi
    print("going to run ekstazi...")
    maven_home = os.getenv('M2_HOME')
    BashUtils.run(f"cp {Macros.tools_dir}/ekstazi-extension-1.0-SNAPSHOT.jar {maven_home}/lib/ext")
    print(os.listdir(f"{maven_home}/lib/ext"))
    with IOUtils.cd(Macros.repos_downloads_dir / f"{project}_ekstazi"):
        BashUtils.run(f"git checkout -f {pre_sha}")
        BashUtils.run(f"mvn clean ekstazi:ekstazi {Macros.SKIPS}")
        BashUtils.run(f"git checkout {cur_sha}")
        BashUtils.run(f"mvn clean ekstazi:ekstazi {Macros.SKIPS}")
        ekstazi_failed_test_list, ekstazi_passed_test_list, _ = test_list_from_surefile_reports()
        print("ekstazi passed test list:", len(ekstazi_passed_test_list))
    BashUtils.run(f"rm {maven_home}/lib/ext/ekstazi-extension-1.0-SNAPSHOT.jar")

    print("going to run starts...")

    with IOUtils.cd(Macros.repos_downloads_dir / f"{project}_starts"):
        BashUtils.run(f"git checkout {pre_sha}")
        BashUtils.run(f"mvn clean starts:starts {Macros.SKIPS} -DignoreClassPath -fn")
        BashUtils.run(f"git checkout {cur_sha}")
        BashUtils.run(f"mvn clean starts:starts {Macros.SKIPS} -DignoreClassPath -fn")
        starts_failed_test_list, starts_passed_test_list, _ = test_list_from_surefile_reports()
        print("starts passed test list:", len(starts_passed_test_list))

    res_item["commit"] = cur_sha
    res_item["prev_commit"] = pre_sha
    res_item["changed_files"] = changed_file_list
    res_item["changed_files_num"] = len(changed_file_list)
    res_item["failed_test_list"] = failed_test_list
    res_item["passed_test_list"] = passed_test_list
    res_item["tests_cases_num"] = test_cases_num_dict
    res_item["ekstazi_test_list"] = ekstazi_passed_test_list
    res_item["ekstazi_failed_test_list"] = ekstazi_failed_test_list
    res_item["starts_test_list"] = starts_passed_test_list
    res_item["starts_failed_test_list"] = starts_failed_test_list
    res_item["diff_code"] = diff_code
    res_item["diff_per_file"] = code_diff_file
    res_item["diff_line_number_list_per_file"] = diff_line_number_list_per_file

    IOUtils.dump(Macros.eval_data_dir / f"{project}-ag.json", [res_item], IOUtils.Format.jsonNoSort)


def add_file_diff(src_data_dir: Path, proj_name: str):
    data_list = IOUtils.load(src_data_dir)
    with IOUtils.cd(Macros.repos_downloads_dir):
        if Path(proj_name).is_dir():
            BashUtils.run(f"rm -rf {proj_name}")
        BashUtils.run(f"git clone https://github.com/apache/commons-dbcp {proj_name}")
    with IOUtils.cd(Macros.repos_downloads_dir / proj_name):
        for i, d in enumerate(data_list):
            if i == 0:
                continue
            commit = d["commit"]
            pr_sha = data_list[i - 1]["commit"]
            changed_file_list = d["changed_files"]
            # add dict to store code diff per file dict
            code_diff_file = {}
            for f in changed_file_list:
                print(BashUtils.run(f'git diff {pr_sha}...{commit} --unified=0 {f} | egrep "^(\+|-)\s"').stdout)
                f_cd = BashUtils.run(f'git diff {pr_sha}...{commit} --unified=0 {f} | egrep "^(\+|-)\s"').stdout
                code_diff_file[f] = f_cd
            # end for
            d["diff_per_file"] = code_diff_file
    with open(f'{Macros.eval_data_dir}/{proj_name}.json', 'w') as res_file:
        json.dump(data_list, res_file, indent=4)


def check_sha_first_time_failed_tests(projects):
    res = []
    for project in projects:
        raw_eval_data_json = IOUtils.load(
            f'{Macros.raw_eval_data_dir}/{project}.json', IOUtils.Format.json)
        if not Path(f"{Macros.repos_downloads_dir}/{project}").is_dir():
            BashUtils.run(f"git clone https://github.com/{project.replace('_', '/')} {project}")
        with IOUtils.cd(f"{Macros.repos_downloads_dir}/{project}"):
            for raw_eval_data_item in raw_eval_data_json:
                commit = raw_eval_data_item["commit"]
                prev_commit = raw_eval_data_item["prev_commit"]

                BashUtils.run(f"git checkout {commit}")
                BashUtils.run(f"mvn clean test {Macros.SKIPS}")
                current_failed_test_list, current_passed_test_list, _ = test_list_from_surefile_reports()
                # print("current failed test", current_failed_test_list)
                # current_failed_test_list =raw_eval_data_item["failed_test_list"]
                if current_failed_test_list:
                    BashUtils.run(f"git checkout {prev_commit}")
                    BashUtils.run(f"mvn clean test {Macros.SKIPS}")

                    prev_failed_test_list, prev_passed_test_list, _ = test_list_from_surefile_reports()

                    first_time_failed_test_list = []
                    for t in current_failed_test_list:
                        if not t in prev_failed_test_list:
                            first_time_failed_test_list.append(t)
                    if first_time_failed_test_list:
                        raw_eval_data_item["project"] = project
                        raw_eval_data_item["first_time_failed_test_list"] = first_time_failed_test_list
                        print(project, "\t", commit, "\t", first_time_failed_test_list)
                        res.append(raw_eval_data_item)
    IOUtils.dump(f'{Macros.raw_eval_data_dir}/first_time_failed_tests.json', res, IOUtils.Format.jsonPretty)
