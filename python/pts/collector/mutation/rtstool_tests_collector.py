import collections
import os
from pathlib import Path

from seutil import IOUtils, BashUtils
from pts.Macros import Macros
import json

def getTestsFromEkstazi(changedFile):
    # input format: org/apache/commons/codec/binary/BaseNCodec (no suffix and use "/" as separator)
    # output format: [org.apache.commons.codec.binary.Base32InputStreamTest, org.apache.commons.codec.binary.Base32InputStreamTest]
    # version2: output format [Base32InputStreamTest]
    # precondition:
    # 1. run "mvn ekstazi:ekstazi ${SKIPS}"
    # 2. ~/.ekstazi should contain following lines:
    # dependencies.format = txt
    # hash.without.debuginfo = true
    # x.log.runs = true
    # debug = true
    # debug.mode = everywhere
    affected_test_classes = set()
    subfolders = [x[0] for x in os.walk(".") if x[0].endswith(".ekstazi")]
    for subfolder in subfolders:
        with IOUtils.cd(f"{subfolder}"):
            grep_res = BashUtils.run(f"grep -rn {changedFile} *").stdout
            for test_class in grep_res.split("\n"):
                split_list = test_class.split(".clz")
                if len(split_list) <= 1:
                    continue
                affected_test_classes.add(split_list[0].strip().split(".")[-1])
    if not affected_test_classes:
        print(f"{changedFile} does not have affected test classes from Ekstazi")
    return list(affected_test_classes)

def getTestsFromSTARTS(changedFile):
    # input format: org/apache/commons/codec/binary/BaseNCodec (no suffix and use "/" as separator)
    # output format: [org.apache.commons.codec.binary.Base32InputStreamTest, org.apache.commons.codec.binary.Base32InputStreamTest]

    # precodition:
    # run "mvn starts:starts -DstartsLogging=FINEST ${SKIPS} -fn"
    affected_test_classes = []
    subfolders = [x[0] for x in os.walk(".") if x[0].endswith(".starts")]
    for subfolder in subfolders:
        with open(f"{subfolder}/deps.zlc") as f:
            lines = f.readlines()
            for line in lines:
                if changedFile in line:
                    split_list = line.split(" ")
                    if len(split_list) <= 1:
                        continue
                    affected_test_classes_full_name = split_list[-1].strip().split(",")
                    affected_test_classes = [i.split(".")[-1] for i in affected_test_classes_full_name]
    if not affected_test_classes:
        print(f"{changedFile} does not have affected test classes from STARTS")
    return affected_test_classes

# def collect_tools_mutants_data(proj: str):
#     """Outdated"
#     res = []
#     input_file_path = f"{Macros.repos_results_dir}/{proj}/collector/mutant-data.json"
#     output_file_path = input_file_path.replace(".json","-rts-tool.json")
#     with open(input_file_path) as f:
#         json_data = json.load(f)
#         for json_item in json_data:
#             changed_class = "/" + json_item["mutatedClass"].split(".")[-1] + ".class"
#             with IOUtils.cd(f"{Macros.repos_downloads_dir}/{proj}"):
#                 json_item['Ekstazi'] = getTestsFromEkstazi(changed_class)
#                 json_item['STARTS'] = getTestsFromSTARTS(changed_class)
#                 print("Ekstazi: ", json_item['Ekstazi'])
#                 print("STARTS: ", json_item["STARTS"])
#                 res.append(json_item)
#     IOUtils.dump(f"{output_file_path}", res, IOUtils.Format.jsonNoSort)
#                 #IOUtils.dump(f"{Macros.data_dir}/model-data/seq2pred-data/{tgt_f}-D-50-data-with-rtstool.json", res,  IOUtils.Format.jsonPretty)

def run_rts_mutant(project: str, target_sha: str):
    """
    Run RTS tool on the mutated sha, and get the .ekstazi, .starts dir.
    """
    with IOUtils.cd(Macros.repos_downloads_dir):
        BashUtils.run(f"cp -r {project} {project}_ekstazi")
        BashUtils.run(f"cp -r {project} {project}_starts")
    # end with
    # first run ekstazi
    with IOUtils.cd(Macros.repos_downloads_dir / f"{project}_ekstazi"):
        BashUtils.run(f"git checkout -f {target_sha}", expected_return_code=0)
        # Run Ekstazi
        print("going to run ekstazi...")
        config = "dependencies.format = txt\n" \
                 "hash.without.debuginfo = true\n" \
                 "x.log.runs = true\n" \
                 "debug = true\n" \
                 "debug.mode = everywhere"
        home_dir = os.path.expanduser('~')
        IOUtils.dump(Path(home_dir)/".ekstazirc", config, IOUtils.Format.txt)
        # BashUtils.run(f"echo {config} > {home_dir}/.ekstazirc", expected_return_code=0)
        maven_home = os.getenv('M2_HOME')
        BashUtils.run(f"cp {Macros.tools_dir}/ekstazi-extension-1.0-SNAPSHOT.jar {maven_home}/lib/ext")
        BashUtils.run(f"mvn clean ekstazi:ekstazi {Macros.SKIPS}", expected_return_code=0)
    # end with
    BashUtils.run(f"rm {maven_home}/lib/ext/ekstazi-extension-1.0-SNAPSHOT.jar")


def collect_rts_mutants(proj: str, target_sha: str):
    """
    Extract the rts results from the results got from RTS tools.
    Collect all the test-method pairs.
    """
    run_rts_mutant(proj, target_sha)
    res = []
    input_file_path = f"{Macros.repos_results_dir}/{proj}/collector/mutant-data.json"
    output_file_path = input_file_path.replace(".json","-rts-tool.json")
    with open(input_file_path) as f:
        json_data = json.load(f)
        for json_item in json_data:
            if "$" in json_item["mutatedClass"]:
                changed_class = "/" + json_item["mutatedClass"].split(".")[-1].split('$')[0] + ".class"
            else:
                changed_class = "/" + json_item["mutatedClass"].split(".")[-1] + ".class"
            # first collect ekstazi
            with IOUtils.cd(f"{Macros.repos_downloads_dir}/{proj}_ekstazi"):
                json_item['Ekstazi'] = getTestsFromEkstazi(changed_class)
            res.append(json_item)
    print("Saving the collected data...")
    test_class_2_methods = collections.defaultdict(list)
    test_class_file_list = BashUtils.run(f"find {Macros.repos_downloads_dir}/{proj}_ekstazi/.ekstazi/ "
                                         f"-name \"*Test*.clz\"").stdout.split("\n")
    collected_results_dir: Path = Macros.repos_results_dir / proj / "collector"
    method_dict = IOUtils.load(collected_results_dir / "method-data.json")
    for cf in test_class_file_list:
        if cf != "":
            class_name = cf.split('/')[-1].split('.')[-2]
            for m in method_dict:
                if m["class_name"] == class_name:
                    test_class_2_methods[class_name].append(m["name"])
        # end if
    # end for
    IOUtils.dump(collected_results_dir / "test2meth.json", test_class_2_methods)
    IOUtils.dump(f"{output_file_path}", res, IOUtils.Format.jsonNoSort)


def main1(proj: str):
    proj_name = proj.split('_')[1]
    res = []
    json_data = IOUtils.load_json_stream(f"{Macros.model_data_dir}/seq2pred-data/{proj_name}/A/A-mutant.json")
    for json_item in json_data:
        cls = json_item["input"][0]
        changed_class = "/" + cls + ".class"
        with IOUtils.cd(f"{Macros.repos_downloads_dir}/{proj}"):
            json_item["Ekstazi"] = getTestsFromEkstazi(changed_class)
            json_item["STARTS"] = getTestsFromSTARTS(changed_class)
            res.append(json_item)
    # end for
    IOUtils.dump(f"{Macros.model_data_dir}/seq2pred-data/{proj_name}/A-rtstool.json", res, IOUtils.Format.jsonNoSort)

                
def main2(proj: str):
    proj_name = proj.split('_')[1]
    for tgt_f in ["train", "valid"]:
        res = []
        print("processing " + tgt_f)
        json_data = IOUtils.load_json_stream(f"{Macros.model_data_dir}/seq2pred-data/{proj_name}/D-50/{tgt_f}.json")
        for json_item in json_data:
            cls = json_item["input"][0]
            changed_class = "/" + cls + ".class"
            with IOUtils.cd(f"{Macros.repos_downloads_dir}/{proj}"):
                json_item["Ekstazi"] = getTestsFromEkstazi(changed_class)
                json_item["STARTS"] = getTestsFromSTARTS(changed_class)
                res.append(json_item)
        # end for
        IOUtils.dump(f"{Macros.model_data_dir}/seq2pred-data/{proj_name}/{tgt_f}-D50-rtstool.json", res, IOUtils.Format.jsonNoSort)
    # end for
