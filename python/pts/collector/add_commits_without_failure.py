import os
from seutil import BashUtils
import json

projects = ['apache@commons-io', 'apache@commons-validator', 'apache@commons-math', 'apache@commons-exec', 'jhy@jsoup',
 'apache@httpcomponents-core', 'apache@commons-net', 'apache@commons-jxpath', 'apache@commons-collections',
 'apache@commons-dbcp']

PROJECTPATH = "../_downloads"
NONFLAKYFOLDER = "../data/notflaky"
BUILD_LOGS = "/Users/liuyu/Documents/build_logs"
BUILD_LOGS_PARSED = "/Users/liuyu/Documents/build_logs_listener"


def get_failed_commit(sha, json_data):
    for json_item in json_data:
        if sha == json_item["commit"]:
            return json_item
    return None

def get_passed_test_list(sha, json_data):
    passed_test_list = []
    for json_item in json_data:
        if not sha or not json_item or "commit" not in json_item:
            continue
        if sha == json_item["commit"]:
            if "branch" in json_item and json_item["branch"] != "master":
                return None
            if "tests" in json_item:
                test_list = json_item["tests"]
                for test_case in test_list:
                    if "name" in test_case:
                        passed_test_list.append(test_case["name"])
                return passed_test_list
    return None


for project in projects:
    res = []

    with open(f"{NONFLAKYFOLDER}/{project}2.json") as file:
        failed_commit_json_data = json.load(file)

    with open(f"{BUILD_LOGS}/{project}/repo-data-travis.json") as file:
        all_commit_json_data = json.load(file)

    with open(f"{BUILD_LOGS_PARSED}/{project}.json") as file:
        all_commit_parsed_json_data = json.load(file)

    for each_commit in all_commit_json_data:
        if each_commit['status'] != 'passed':
            failed_commit = get_failed_commit(each_commit["commit"], failed_commit_json_data)
            if failed_commit:
                res.append(failed_commit)

        if each_commit["branch"] != "master":
            continue

        else:
            commit_res = each_commit

            job_ids = each_commit["jobs"]
            for job_id in job_ids:
                job_string = BashUtils.run(
                    f"curl -H 'Travis-API-Version: 3' -H 'User-Agent: API Explorer' 'https://api.travis-ci.org/job/{job_id}'").stdout
                if not job_string:
                    continue
                job_data = json.loads(job_string)
                if "state" not in job_data:
                    continue
                if job_data["commit"]["compare_url"]:
                    commit_res["compare_url"] = job_data["commit"]["compare_url"]
                    commit_res["message"] = job_data["commit"]["message"]
                    break

            if "compare_url" not in commit_res or "..." not in commit_res["compare_url"]:
                continue
            passed_test_list = get_passed_test_list(each_commit["commit"], all_commit_parsed_json_data)
            if passed_test_list:
                commit_res["failed_test_list"] = []
                commit_res["diff_code"] = ""
                commit_res["passed_test_list"] = passed_test_list
                res.append(commit_res)

    with open(f'{NONFLAKYFOLDER}/{project}3.json', 'w') as non_flaky_file:
        json.dump(res, non_flaky_file, indent=4)
