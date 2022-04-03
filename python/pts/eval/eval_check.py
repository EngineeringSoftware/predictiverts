from seutil import IOUtils, BashUtils
from pts.Macros import Macros
from pts.collector.eval_data_collection import eval_data_for_each_SHA
from pts.main import proj_logs
from pts.collector.eval_data_collection import diff_per_file_for_each_SHA


def check_selection_rate_outlier(projs, models):
    for proj in projs:
        proj_name = proj.split("_")[-1]
        for model in models:
            try:
                best_select_rate_per_sha = IOUtils.load(f"{Macros.results_dir}/modelResults/{proj_name}/{model}/best-select-rate-per-SHA.json")
                for sha, select_rate in best_select_rate_per_sha.items():
                    if select_rate > 0.75:
                        print(f"{proj}\t{model}\t{sha}\t{select_rate}")
            except Exception as e:
                print(e)
                continue

def check_not_safe_selection(projs):
    for proj in projs:
        ekstazi_not_safe_shas = set()
        starts_not_safe_shas = set()
       
        try:
            per_sha_result = IOUtils.load(f"{Macros.eval_data_dir}/mutated-eval-data/{proj}-ag.json")
            for per_sha_item in per_sha_result:
                failed_test_list = per_sha_item["failed_test_list"]
                ekstazi_test_list = per_sha_item["ekstazi_test_list"]
                starts_test_list = per_sha_item["starts_test_list"]
            
                for test in failed_test_list:
                    if test not in ekstazi_test_list:
                        ekstazi_not_safe_shas.add(per_sha_item['commit'])

                    if test not in starts_test_list:
                        starts_not_safe_shas.add(per_sha_item['commit'])
            for sha in ekstazi_not_safe_shas:
                print(f"{proj}\t{sha}\tekstazi\t")
            for sha in starts_not_safe_shas:
                print(f"{proj}\t{sha}\tstarts\t")
        except Exception as e:
            print(e)
            continue


def check_line_number(projs):
    num_of_shas = 0
    for proj in projs:
        try:
            per_sha_result = IOUtils.load(f"{Macros.data_dir}/mutated-eval-data/{proj}-ag.json")
            for per_sha_item in per_sha_result:
                num_of_shas += 1
                sha = per_sha_item["commit"]
                diff_line_num_per_file = per_sha_item["diff_line_number_list_per_file"]
                diff_per_file = per_sha_item["diff_per_file"]
            for file, line_num_list in diff_line_num_per_file.items():
                original_line_num = len(line_num_list)
                revised_line_num = len(diff_per_file[file].splitlines())
                if original_line_num != revised_line_num:
                    print(f"{proj}\t{sha}\t{file}\toriginal:{original_line_num}\trevised:{revised_line_num}")
        except Exception as e:
            print(e)
            continue
    print("There are", num_of_shas, "total shas")


def update_changed_file(projs):
    num_of_shas = 0
    for proj in projs:
        try:
            per_sha_result = IOUtils.load(f"{Macros.raw_eval_data_dir}/{proj}.json")
            updated = []
            if (Macros.repos_downloads_dir / proj).exists():
                BashUtils.run(f"rm -rf {Macros.repos_downloads_dir}/{proj}")
            BashUtils.run(
                    f"git clone https://github.com/{proj.replace('_', '/')} {Macros.repos_downloads_dir}/{proj}")
            with IOUtils.cd(f"{Macros.repos_downloads_dir}/{proj}"):
                for per_sha_item in per_sha_result:
                    commit = per_sha_item["commit"]
                    prev_commit = per_sha_item["prev_commit"]
                    original_diff_per_file = per_sha_item["diff_per_file"].keys()
                    changed_file_list, diff_code, code_diff_file, diff_line_number_list_per_file = diff_per_file_for_each_SHA(prev_commit, commit)
                    if len(original_diff_per_file) != len(diff_line_number_list_per_file.keys()):
                        print(f"{proj}\t{commit}\toriginal changed files:{len(original_diff_per_file)}\tcurrent changed files:{len(diff_line_number_list_per_file.keys())}")
                        num_of_shas += 1
                    per_sha_item["diff_code"] = diff_code
                    per_sha_item["diff_per_file"] = code_diff_file
                    per_sha_item["diff_line_number_list_per_file"] = diff_line_number_list_per_file
                    updated.append(per_sha_item)
                IOUtils.dump(f"{Macros.raw_eval_data_fixed_dir}/{proj}.json", updated, IOUtils.Format.jsonPretty)
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
    print("There are", num_of_shas, "total shas affected")

def eval_changed_file(projs):
    num_of_shas = 0
    for proj in projs:
        try:
            per_sha_result = IOUtils.load(f"{Macros.data_dir}/mutated-eval-data/{proj}-ag.json")
            for per_sha_item in per_sha_result:
                changed_files = per_sha_item["changed_files"]
                changed_java_files = [changed_file for changed_file in changed_files if ".java" in changed_file]
                num_of_shas += 1
                sha = per_sha_item["commit"]
                diff_per_file = per_sha_item["diff_per_file"].keys()
                if len(changed_java_files) != len(diff_per_file):
                    print(f"{proj}\t{sha}\tchanged_files:{len(changed_files)}\tdiff_per_file:{len(diff_per_file)}")
        except Exception as e:
            print(e)
            continue
    print("There are", num_of_shas, "total shas")

# def check_mutated_changed_file(projs):
#     num_of_shas = 0
#     num_of_shas_different_file = 0
#     num_of_shas_differnet_content = 0
#     for proj in projs:
#         try:
#             raw_eval_result = IOUtils.load(f"{Macros.raw_eval_data_dir}/{proj}.json")
#             per_sha_result = IOUtils.load(f"{Macros.eval_data_dir}/mutated-eval-data/{proj}-ag.json")
#             for per_sha_item in per_sha_result:
#                 diff_per_file = per_sha_item["diff_per_file"]
#                 commit = per_sha_item["commit"][:8]
#                 for raw_eval_item in raw_eval_result:
#                     if raw_eval_item["commit"] == commit:
#                         raw_diff_per_file = raw_eval_item["diff_per_file"]
#                         if len(raw_diff_per_file) != len(diff_per_file):
#                             num_of_shas += 1
#                             print(proj, per_sha_item["commit"], "different number of files")
#                         else:
#                             current_diff_num = 0
#                             for filename, content in raw_diff_per_file.items():
#                                 if filename not in diff_per_file:
#                                     num_of_shas_different_file += 1
#                                     print(proj, per_sha_item["commit"], "differnet file")
#                                     break
#                                 if content != diff_per_file[filename]:
#                                     current_diff_num += 1
#                                     if current_diff_num >= 2:
#                                         num_of_shas_differnet_content += 1
#                                     break
#                         break
#         except Exception as e:
#             print(e)
#             continue
#     print("There are", num_of_shas, "total shas that have different number of changed files")
#     print("There are", num_of_shas_different_file, "total shas that have different files")
#     print("There are", num_of_shas_differnet_content, "total shas that have different content")


def check_newly_added_tests(projs):
    for proj in projs:
        sha_with_failed_tests = 0
        sha_with_passed_tests = 0
        if (Macros.repos_downloads_dir / proj).exists():
            BashUtils.run(f"rm -rf {Macros.repos_downloads_dir}/{proj}")
            BashUtils.run(f"git clone https://github.com/{proj.replace('_', '/')} {Macros.repos_downloads_dir}/{proj}")
        try:
            per_sha_result = IOUtils.load(f"{Macros.eval_data_dir}/mutated-eval-data/{proj}-ag.json")
            for per_sha_item in per_sha_result:
                mutant_sha_pair = per_sha_item["commit"]
                cur_sha = per_sha_item["commit"][:8]
                failed_tests = per_sha_item["failed_test_list"]
                passed_tests = per_sha_item["passed_test_list"]
                training_sha = proj_logs[proj]
                with IOUtils.cd(f"{Macros.repos_downloads_dir}/{proj}"):
                    added_files = BashUtils.run(f"git diff {training_sha}...{cur_sha} --name-only --diff-filter=A").stdout.splitlines()
                    added_files = [added_file for added_file in added_files if ".java" in added_file]
                    for added_file in added_files:
                        if ".java" in added_file and added_file.split("/")[-1].replace(".java", "") in failed_tests:
                            sha_with_failed_tests += 1
                        if ".java" in added_file and added_file.split("/")[-1].replace(".java", "") in passed_tests:
                            sha_with_passed_tests += 1
                    if len(added_files) > 0:
                        print(f"{proj}\t{mutant_sha_pair}\tadded_files:{len(added_files)}\t")
            print(f"{proj}\tsha with newly added failed tests:{sha_with_failed_tests}\tsha with newly added passed tests:{sha_with_passed_tests}")
        except Exception as e:
            print(e)
            continue


def check_first_time_failed_test():
    res = []
    first_time_failed_tests = IOUtils.load(f"{Macros.raw_eval_data_dir}/first_time_failed_tests.json")
    for first_time_failed_tests_item in first_time_failed_tests:
        proj = first_time_failed_tests_item["project"]
        if (Macros.repos_downloads_dir / proj).exists():
            BashUtils.run(f"rm -rf {Macros.repos_downloads_dir}/{proj}")
        BashUtils.run(
                f"git clone https://github.com/{proj.replace('_', '/')} {Macros.repos_downloads_dir}/{proj}")
        with IOUtils.cd(f"{Macros.repos_downloads_dir}/{proj}"):
            commit = first_time_failed_tests_item["commit"]
            prev_commit = first_time_failed_tests_item["prev_commit"]
            changed_file_list, diff_code, code_diff_file, diff_line_number_list_per_file = diff_per_file_for_each_SHA(
                prev_commit, commit)
            first_time_failed_tests_item["diff_code"] = diff_code
            first_time_failed_tests_item["diff_per_file"] = code_diff_file
            first_time_failed_tests_item["diff_line_number_list_per_file"] = diff_line_number_list_per_file
            res.append(first_time_failed_tests_item)
    IOUtils.dump(f"{Macros.raw_eval_data_fixed_dir}/first_time_failed_tests.json", res, IOUtils.Format.jsonPretty)


def check_mutated_changed_file(projs):
    for proj in projs:
        try:
            result = []
            raw_eval_result_bug = IOUtils.load(f"{Macros.eval_data_dir}/raw-eval-data-original/{proj}.json")
            per_sha_result = IOUtils.load(f"{Macros.eval_data_dir}/mutated-eval-data-original/{proj}-ag.json")
            for per_sha_item in per_sha_result:
                diff_per_file = per_sha_item["diff_per_file"]
                commit = per_sha_item["commit"][:8]
                correct_raw_eval_item = {}

                with IOUtils.cd(f"{Macros.repos_downloads_dir}/{proj}"):
                    prev_commit = per_sha_item["prev_commit"]
                    changed_file_list, diff_code, code_diff_file, diff_line_number_list_per_file = diff_per_file_for_each_SHA(
                        prev_commit, commit)
                    correct_raw_eval_item["diff_code"] = diff_code
                    correct_raw_eval_item["diff_per_file"] = code_diff_file
                    correct_raw_eval_item["diff_line_number_list_per_file"] = diff_line_number_list_per_file

                for raw_eval_item in raw_eval_result_bug:
                    if raw_eval_item["commit"] == commit:
                        raw_diff_per_file = raw_eval_item["diff_per_file"]
                        if len(raw_diff_per_file) > 1:
                            changed_file = ""
                            for filename, changed_content in raw_diff_per_file.items():
                                if changed_content != diff_per_file[filename]:
                                    if changed_file != "":
                                        print(proj, per_sha_item["commit"], "change more than one file!")
                                    else:
                                        changed_file = filename
                                        diff_changed_file = per_sha_item["diff_per_file"][changed_file]
                                        diff_line_number_list = per_sha_item["diff_line_number_list_per_file"][changed_file]
                                        print(proj, per_sha_item["commit"], changed_file)
                            if changed_file == "":
                                print(proj, per_sha_item["commit"], "should not exist")
                            else:
                                per_sha_item["diff_per_file"] = correct_raw_eval_item["diff_per_file"]
                                per_sha_item["diff_line_number_list_per_file"] = correct_raw_eval_item["diff_line_number_list_per_file"]
                                per_sha_item["diff_per_file"][changed_file] = diff_changed_file
                                per_sha_item["diff_line_number_list_per_file"][changed_file] = diff_line_number_list
                        result.append(per_sha_item)
                        break
            IOUtils.dump(f"{Macros.eval_data_dir}/mutated-eval-data-fixed/{proj}-ag.json", result, IOUtils.Format.jsonPretty)
        except Exception as e:
            print(e)
            continue

def check_bm25_performs_well(projects, models):
    # for project in projects:
    #     proj = project.split("_")[-1]
    #     best_select_rate_per_sha = {}
    #     for model in models:
    #          best_select_rate_per_sha[model] = IOUtils.load(Macros.results_dir / "modelResults" / proj / model / "best-select-rate-per-SHA.json")
    #     for sha in best_select_rate_per_sha["BM25Baseline"].keys():
    #         bm25 = best_select_rate_per_sha["BM25Baseline"][sha]
    #         good = True
    #         for model in models:
    #             if best_select_rate_per_sha[model][sha] < bm25:
    #                 good = False
    #         if good:
    #             print(proj, sha, bm25)
    res = []
    for project in projects:
        proj = project.split("_")[-1]
        best_select_rate_per_sha = {}
        perfect_select_rate_per_sha = IOUtils.load(Macros.results_dir / "modelResults" / proj / "Fail-Code" / "perfect-select-rate-per-SHA.json")
        bm25_perfect = 0
        bm25_perfect_but_others_not = 0
        total = len(perfect_select_rate_per_sha)

        for model in models:
            best_select_rate_per_sha[model] = IOUtils.load(Macros.results_dir / "modelResults" / proj / model / "best-select-rate-per-SHA.json")

        for sha in best_select_rate_per_sha["BM25Baseline"].keys():
            bm25 = best_select_rate_per_sha["BM25Baseline"][sha]
            if bm25 == perfect_select_rate_per_sha[sha]:
                bm25_perfect += 1

                other_model_perfect = False
                for model in models:
                    if "BM25Baseline" in model:
                        continue
                    if best_select_rate_per_sha[model][sha] == perfect_select_rate_per_sha[sha]:
                        #print(project, model)
                        other_model_perfect = True
                        break

                if not other_model_perfect:
                    bm25_perfect_but_others_not += 1

        res_item = {}
        res_item["project"] = project
        res_item["bm25_perfect"] = bm25_perfect
        res_item["bm25_perfect_but_others_not"] = bm25_perfect_but_others_not
        res_item["bm25_perfect_rate"] = bm25_perfect / total
        res_item["bm25_perfect_but_others_not_rate"] = bm25_perfect_but_others_not / total
        res_item["total"] = total
        print(res_item)
        res.append(res_item)
    IOUtils.dump(Macros.eval_data_dir / "mutated-eval-data"/ "bm25_perfect.json", res, IOUtils.Format.jsonPretty)


def check_diff_between_models(projects, models):
    for project in projects:
        proj = project.split("_")[-1]
        BashUtils.run(f"echo '' > {Macros.docs_dir}/selection-rate-diff-{proj}.txt")
        best_select_rate_per_sha = {}
        mutated_eval_data = IOUtils.load(f"{Macros.eval_data_dir}/mutated-eval-data/{project}-ag.json")
        for model in models:
            best_select_rate_per_sha[model] = IOUtils.load(Macros.results_dir / "modelResults" / proj / model / "best-select-rate-per-SHA.json")

        for model in models:
            sha_mutant_to_diff = {}
            sha_mutant_to_content = {}
            
            if model == "BM25Baseline":
                continue
            
            res = f"SHA-mutant | #failed tests | #files_changed | {model.replace('BM25Baseline', 'bm25')} | bm25\n"
            print(res)
            BashUtils.run(f"echo '{res}' >> {Macros.docs_dir}/selection-rate-diff-{proj}.txt")
            
            for sha, bm25 in best_select_rate_per_sha["BM25Baseline"].items():
                for mutated_eval_data_item in mutated_eval_data:
                    if mutated_eval_data_item["commit"] == sha:
                        number_changed_files = len(mutated_eval_data_item["diff_per_file"].keys())
                        number_failed_tests = len(mutated_eval_data_item["failed_test_list"])
                sha_mutant_to_diff[sha] = best_select_rate_per_sha[model][sha] - bm25
                sha_mutant_to_content[sha] = f"{sha}\t{number_failed_tests}\t{number_changed_files}\t{best_select_rate_per_sha[model][sha]}\t{bm25}"
            # sort in decreasing order
            for key_sha, value_diff in sorted(sha_mutant_to_diff.items(), key=lambda item: -item[1]):
                res = sha_mutant_to_content[key_sha] + "\n"
                print(res)
                BashUtils.run(f"echo '{res}' >> {Macros.docs_dir}/selection-rate-diff-{proj}.txt")
            BashUtils.run(f"echo '\n\n' >> {Macros.docs_dir}/selection-rate-diff-{proj}.txt")
