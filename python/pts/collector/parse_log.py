# command: python -m pts.main parse_travis_logs
import json
import os
import pathlib
from seutil import BashUtils, IOUtils
import re
from pts.Macros import Macros
import traceback

endTestRun = re.compile(".*Tests run: ([0-9]+), Failures: ([0-9]+), Errors: ([0-9]+), Skipped: ([0-9]+).*")

# collect information from git
def get_diff(institution, project_name, item):
    with IOUtils.cd(Macros.repos_downloads_dir):
        if not pathlib.Path(f"{institution}_{project_name}").is_dir():
            BashUtils.run(f"git clone https://github.com/{institution}/{project_name} {institution}_{project_name}")
        with IOUtils.cd(f"{institution}_{project_name}"):
            try:
                url = item["compare_url"]
                if "..." in url.split("/")[-1]:
                    sha1 = url.split("/")[-1].split("...")[0]
                    sha2 = url.split("/")[-1].split("...")[1]
                else:
                    cur_sha = item["commit"].strip()
                    # test if this is the sha exists in current project
                    checkout_res = BashUtils.run(f"git rev-list --parents -n 1 {cur_sha}").stdout
                    if not checkout_res or "fatal:" in checkout_res:
                        return [], []
                    else:
                        sha1 = checkout_res.split(" ")[0].strip()
                        sha2 = checkout_res.split(" ")[1].strip()
                ##  get code change
                # diff_code = BashUtils.run(f"git diff {sha1}...{sha2} | grep '^[+-]' | grep -Ev '^(--- a/|\+\+\+ b/)'").stdout

                changed_file_list = BashUtils.run(f"git log --name-only --format=tformat: {sha1}..{sha2}").stdout.split("\n")
                if "" in changed_file_list:
                    changed_file_list.remove("")

                num_of_distinct_authors = 0
                if changed_file_list:
                    distinct_authors_set = set()
                    for changed_file in changed_file_list:
                        distinct_authors_list = BashUtils.run(
                            f"git blame '{changed_file}' --porcelain | grep '^author ' | sort | uniq").stdout.split("\n")
                        distinct_authors_set.update(distinct_authors_list)
                    if "" in distinct_authors_set:
                        distinct_authors_set.remove("")
                    num_of_distinct_authors = len(distinct_authors_set)

                # return value: changed_file_list, number of authors who touched these files
                return changed_file_list, num_of_distinct_authors
            except Exception as e:
                traceback.print_exc()
                return [], []


def parse_log(log_string):
    log_list = re.split('\n|\r', log_string)
    failed_test_list = []
    passed_test_list = []
    test_cases_num_dict = {}
    try:
        for log_line in log_list:
            res = endTestRun.match(log_line)
            if res:
                if "- in" in log_line:
                    test_name = log_line.split("- in")[-1].strip()
                    test_cases_num_dict[test_name] = int(res.group(1))
                    if int(res.group(2)) > 0 or int(res.group(3)) > 0:
                        failed_test_list.append(test_name)
                    else:
                        passed_test_list.append(test_name)
        # print("test_cases_num_dict: ", test_cases_num_dict)
        return failed_test_list, passed_test_list, test_cases_num_dict
    except Exception as e:
        traceback.print_exc()
        return [], [], []


def main():
    for project_folder in os.listdir(Macros.build_logs_dir):
        # TODO: used for test
        # project_folder = "apache@commons-math"

        institution = project_folder.split("@")[0].strip()
        project_name = project_folder.split("@")[1].strip()

        flaky_list = []
        non_flaky_list = []
        try:
            with open(Macros.build_logs_dir / project_folder / "repo-data-travis.json") as json_file:
                travis_data = json.load(json_file)
                for travis_data_item in travis_data:
                    # only use master branch
                    if travis_data_item["branch"] != "master":
                        continue

                    if travis_data_item["status"] == "failed" or travis_data_item["status"] == "errored":
                        job_ids = travis_data_item["jobs"]
                        contain_passed = False
                        res_item = travis_data_item
                        failed_test_set = set()
                        passed_test_set = set()
                        test_cases_num_dict = {}
                        # check if the fail is flaky
                        for job_id in job_ids:
                            job_string = BashUtils.run(
                                f"curl -H 'Travis-API-Version: 3' -H 'User-Agent: API Explorer' 'https://api.travis-ci.org/job/{job_id}'").stdout
                            if not job_string:
                                continue
                            job_data = json.loads(job_string)
                            if "state" not in job_data:
                                continue
                            res_item["compare_url"] = job_data["commit"]["compare_url"]
                            # TODO: dependency graph info from starts
                            changed_file_list, num_of_distinct_authors = get_diff(institution, project_name, res_item)
                            if not changed_file_list:
                                continue

                            res_item["changed_files"] = changed_file_list
                            res_item["changed_files_num"] = len(changed_file_list)
                            res_item["distinct_authors_num"] = num_of_distinct_authors

                            res_item["message"] = job_data["commit"]["message"]

                            if job_data["state"] == "failed" or job_data["state"] == "errored":
                                # parse the log
                                log_string = BashUtils.run(f"curl -H 'Travis-API-Version: 3' -H 'User-Agent: API Explorer' 'https://api.travis-ci.org/v3/job/{job_id}/log.txt'").stdout
                                # TODO: test cases num
                                job_failed_test_list, job_passed_test_list, cur_job_test_cases_num_dict = parse_log(log_string)
                                failed_test_set.update(job_failed_test_list)
                                passed_test_set.update(job_passed_test_list)
                                test_cases_num_dict.update(cur_job_test_cases_num_dict)
                            if job_data["state"] == "passed":
                                contain_passed = True

                        if "" in failed_test_set:
                            failed_test_set.remove("")
                        if "" in passed_test_set:
                            passed_test_set.remove("")
                        if not failed_test_set or not test_cases_num_dict:
                            continue
                        res_item["failed_test_list"] = list(failed_test_set)
                        res_item["passed_test_list"] = list(passed_test_set)
                        res_item["tests_cases_num"] = test_cases_num_dict

                        if contain_passed:
                            flaky_list.append(res_item)
                        else:
                            non_flaky_list.append(res_item)
            if flaky_list:
                with open(f'{Macros.flaky_raw_data_dir}/{institution}_{project_name}.json', 'w') as flaky_file:
                    json.dump(flaky_list, flaky_file, indent=4)
            if non_flaky_list:
                print(f"{institution}_{project_name} number of builds: {len(non_flaky_list)}")
                with open(f'{Macros.raw_data_dir}/{institution}_{project_name}.json', 'w') as non_flaky_file:
                    json.dump(non_flaky_list, non_flaky_file, indent=4)

        except Exception as e:
            continue
