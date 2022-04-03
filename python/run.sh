#!/bin/bash

# This script documents the exact procedures we use to get the
# dataset, get the models running, and collect the results.

# Each function is a group of commands and a later function usually
# requires the execution of all the proceeding functions.

# The commands within each function should always be executed one
# after one sequentially unless only partial functionality is wanted.

_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

readonly DATASET_PATH=${_DIR}/../data
readonly RESULTS_PATH=${_DIR}/../_results
readonly PROJECTS=(
    "asterisk-java_asterisk-java"
    "Bukkit_Bukkit"
    "apache_commons-configuration"
    "apache_commons-csv"
    "apache_commons-lang"
    "apache_commons-net"
    "apache_commons-validator"
    "frizbog_gedcom4j"
    "mikera_vectorz"
    "zeroturnaround_zt-exec"
)

readonly MODELS=(
        "Fail-Basic"
        "Fail-Code"
        "Fail-ABS"
        "Ekstazi-Basic"
        "Ekstazi-Code"
        "Ekstazi-ABS"
        "TFIDFBaseline"
        "BM25Baseline"
        "Fail-Code-BM25Baseline"
        "Ekstazi-Basic-BM25Baseline"
        "boosting"
)

# Commonly-used scripts
# Precondition
# Manually run Pitest, because it requires Junit4.
# The plugin for junit5: https://github.com/pitest/pitest-junit5-plugin

######################
# Download projects to _downloads
function downloads_repos() {
        python -m pts.main clone_checkout_projects
}

#################
# get raw training data from PIT generated mutants
function get_training_data() {
        projects=(
                "apache_commons-lang"
                "apache_commons-codec"
                "apache_commons-io"
                "apache_commons-net"
                "apache_commons-validator"
                "apache_commons-csv"
                "apache_commons-cli"
                "asterisk-java_asterisk-java"
        )
        for proj in "${projects[@]}"; do
                get_mutants $proj
                get_tools_data_for_mutants $proj
        done
}


function extract_test_methods_from_eval_data() {
        for proj in "${PROJECTS[@]}"; do
                collect_test_methods_in_eval_data $proj
        done
}

####
# Generate table for PIT mutants
function generate_table_PIT_mutants() {
        projects=(
                "apache_commons-lang"
                # "cloudfoundry_uaa"
                "logstash_logstash-logback-encoder"
                "apache_commons-codec"
                # "apache_commons-pool"
                "apache_commons-io"
                # "apache_commons-net"
                "apache_commons-validator"
                "apache_commons-dbutils"
                "apache_commons-csv"
                "apache_commons-cli"
                "asterisk-java_asterisk-java"
        )
        for proj in "${projects[@]}"; do
                pit_mutants_stats $proj
                project_pit_mutants_table $proj
        done
}

#######################################
# Parse the project to get all methods, parse pit-log to get mutants data.
# Globals:
# None
# Arguments:
#   proj: name of the project: {institution}_{projectName}
#######################################
function get_mutants(){
        proj=("$1")
        project_parser $proj
        echo "Extract mutants..."
        mutant_data_collect $proj
}

#######################################
# Get labels for PIT mutants from tools (Ekstazi and STARTS)
# Globals:
#   None
# Arguments:
#   proj: name of the project: {institution}_{projectName}
#######################################
function get_tools_data_for_mutants() {
        proj=("$1")
        echo "Run RTS tools..."
        python -m pts.main collect_tools_data_for_mutant --project=$proj
}


function train_rank_model() {
        python -m pts.main run_rank_model --project="$1" --config="hybrid.yaml"
}


# Step 1: download repos
function download_repos() {
        python -m pts.main download_repo
}

# Step 2: run PIT in the providec directory to get xml report
function run_pitest() {
        python -m pts.main run_pit --project_dir="$1"
}

# Step 3: run ProjectParser to collect all the methods in the repo in the particular SHA
#--- Parse project downloads dir to get methods, the specific project is passed through command line (the first positional arg)
function project_parser() {
        python -m pts.main parse_projects --project="$1"
}

# Step 4: parse the xml PIT report and recover the source code change based on the descriptions and line number
#--- code for get, collect, process mutation data, the name of the project is passed through command line (the first positional arg)
function mutant_data_collect() {
        python -m pts.main get_mutant_data --projects="$1" --binary
}

function parse_pit_report() {
        python -m pts.main parse_pit_report --project="$1"
}


function collect_eval_data() {
        python -m pts.main collect_eval_data --project="$1"
}

# Functions for creating eval data
function mutate_git_patch() {
        python -m pts.main augment_eval_data --project="$1"
}


#-----Processor--------------------------

#------------------------------------------------------------------
# Step 1 Run ekstazi and starts on the mutants
function get_tools_labels_mutants() {
        python -m pts.main collect_eval_data_from_ekstazi_starts\
               --project="$1"
}

# Step 1.0 Run ekstazi and starts on defects4J
function get_tools_labels_defects4j() {
        python -m pts.main collect_data_defects4j_ekstazi_starts --project="$1"
}

# Step 2: process tool data to be trainable by our model
function seq2pred_tools_data_process() {
#        python -m pts.main collect_tests_from_rtstool_changed_by_mutant --project=apache_commons-codec
        #        python -m pts.main process_data --which=Seq2pred\
                #               --type=tools --project=apache_commons-codec
        python -m pts.main process_data --which=Seq2pred\
               --type=tools --project="$1"
}


function index_mutant_data() {
        python -m pts.main index_mutant_data  # build 2 dict: (1) test2method, (2) mutated-method -> id
}


#------------Collect data------------------
# New
function collect_test_methods_in_eval_data(){
        python -m pts.main collect_methods_eval_shas --which=method-data-collect\
               --project="$1"
}

function collect_test_method_mappings() {
        python -m pts.main collect_data --which=test-method-data\
               --projects=apache_commons-lang
}


# ==========
# Main function
# This script can be executed as ./run.sh the_function_to_run

function main() {
        local action=${1:?Need Argument}; shift

        ( cd ${_DIR}
          $action "$@"
        )
}

main "$@"


# ==========
# Some notes of useful Bash commands

# Export Anaconda environment
# conda env export --from-history > env.yml

# Load Anaconda envrionment
# conda env create -n NAME -f env.yml