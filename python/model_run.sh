#!/bin/bash

# This script documents the exact procedures we use to process data
# for models, train the models, test the models and analyze the models.

_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

readonly DATASET_PATH=${_DIR}/../data
readonly RESULTS_PATH=${_DIR}/../_results

readonly defects4j_lang_id=(31 32 33 34 35)

readonly PROJECTS=(
         "Bukkit_Bukkit"
         "asterisk-java_asterisk-java"
         "zeroturnaround_zt-exec"
         "apache_commons-validator"
         "apache_commons-net"
         "apache_commons-csv"
         "frizbog_gedcom4j"
         "mikera_vectorz"
         "apache_commons-lang"
     "apache_commons-configuration"
#    "apache_Lang-36"
)

function train_all_models_in_parallel() {
        for proj in "${PROJECTS[@]}"; do
                train_rank_ekstazi_basic_model $proj &
                train_rank_ekstazi_code_model $proj &
                train_rank_ekstazi_abs_model $proj &
                wait
                train_rank_fail_basic_model $proj &
                train_rank_fail_code_model $proj &
                train_rank_fail_abs_model $proj &
                wait
        done
}

# Bagging model in the paper
function run_boosting_models_all() {
        for proj in "${PROJECTS[@]}"; do
                run_boosting_rank_models $proj
        done
}

function analyze_boosting_models_all() {
        for proj in "${PROJECTS[@]}"; do
                analyze_boosting_model $proj
        done
}

function get_eval_running_time_for_all() {
        for proj in "${PROJECTS[@]}"; do
                get_eval_running_time $proj &

        done
}

function eval_all_models() {
        for proj in "${PROJECTS[@]}"; do
                test_all_baselines_rank_models $proj
                analyze_rank_models $proj
        done
}

function analyze_baseline_models() {
        for proj in "${PROJECTS[@]}"; do
                analyze_BM25_model $proj
        done
}

function analyze_combo_models_all() {
        for proj in "${PROJECTS[@]}"; do
                analyze_ensemble_model $proj
        done
}

function analyze_ealrts_models(){
        for proj in "${PROJECTS[@]}"; do
                analyze_EALRTS_randomforest_model $proj
                analyze_EALRTS_xgboost_model $proj
        done
}

## Eval on real failures tests
function eval_models_on_real_failed_tests() {
       python -m pts.main eval_real_failed_tests --rule="True"
       python -m pts.main eval_real_failed_tests
}

###############
# Run all the combo models (called ensemble models before)
################
function run_combo_models_all() {
        for proj in "${PROJECTS[@]}"; do
                  run_ensemble_ekstazi_basic_bm25 $proj
                  run_ensemble_bm25_fail_basic $proj
        done
}

###############
# Run all the baseline models
################
function run_bm25_baseline_all(){
        for proj in "${PROJECTS[@]}"; do
                preprocess_bm25_baseline $proj
                run_bm25_baseline $proj
        done
}

function run_tfidf_content_baseline_all() {
        for proj in "${PROJECTS[@]}"; do
                run_TFIDF_content_baseline $proj
        done
}

function run_bm25_no_context_baseline_all() {
        for proj in "${PROJECTS[@]}"; do
                run_bm25_no_context_baseline $proj
        done
}

function run_test_one_by_one(){
        python -m pts.main run_test_one_by_one --projects="${PROJECTS[*]}"
}

########
# Process training data (tool, fail)
# local vars
function process_training_data() {
        for proj in "${PROJECTS[@]}"; do
                process_train_data_rank_model $proj
                process_train_data_rank_model_fail $proj
        done
}

# Process test data and collect metrics for test data.
function process_test_data() {
        for proj in "${PROJECTS[@]}"; do
                # First delete empty fail test sha:mutants
                python -m pts.main collect_metrics --which=mutated-eval-data --project=$proj
                process_test_data_rank_model $proj
        done
}

function make_eval_data_defects4j_lang() {
         for pid in ${defects4j_lang_id[@]}; do
             python -m pts.main collect_eval_data_defect4j --project=apache_Lang-${pid}
	 done
}

#######################################
# Data processor for rank model
#######################################
function process_train_data_rank_model() {
        python -m pts.main process_data --which=Rank\
               --label_type=Ekstazi --project="$1" --type=train
}

function process_train_data_rank_model_fail() {
        python -m pts.main process_data --which=Rank\
               --label_type=Fail\
               --project="$1" --type=train
}

function process_test_data_rank_model() {
        python -m pts.main process_data --which=Rank\
               --project="$1" --type=test
}

# caller function for rank_model_bm25
function process_train_data_rank_model_bm25_fail(){
        python -m pts.main process_data --which=Rank-bm25 --project="$1" --label_type=Fail --type=train
}

function process_test_data_rank_model_bm25() {
        python -m pts.main process_data --which=Rank-bm25 --project="$1" --type=test
}


########################
# Data Processor for the EALRTS
########################
function process_mutated_eval_data_ealrts_all() {
        for proj in "${PROJECTS[@]}"; do
            process_mutated_eval_data_ealrts $proj
        done
}

function process_mutated_eval_data_ealrts() {
        python -m pts.main process_data --which=EALRTS --project="$1"
}

############################
# Run EALRTS model
############################
function run_ealrts_baseline_all() {
        for proj in "${PROJECTS[@]}"; do
            run_ealrts_baseline $proj
        done
}

function run_ealrts_baseline() {
    python -m pts.main run_ealrts_baseline --project="$1" --model="randomforest"
    python -m pts.main run_ealrts_baseline --project="$1" --model="xgboost"
}

#####################################
# Random model runner
#########################################
function run_random_model() {
        python -m pts.main run_random_model --project="$1"
}

###########################################
# Line number baseline
##########################################
function run_linenumber_baseline() {
        python -m pts.main run_linenum_baseline --project="$1"
}

function run_linenumber_baseline_with_deleted_lines() {
        python -m pts.main run_linenum_baseline --project="$1" --use_deleted
}

function run_linenumber_baseline_with_all_covered_lines() {
        python -m pts.main run_linenum_baseline --project="$1" --all_covered
}

###################################
# Combine/ensemble models
###################################
function run_ensemble_bm25_fail_code(){
        python -m pts.main run_ensemble_models --project="$1" --models=Fail-Code --models=BM25Baseline
}

function run_ensemble_ekstazi_basic_bm25() {
        python -m pts.main run_ensemble_models --project="$1" --models=Ekstazi-Basic --models=BM25Baseline
}

function run_ensemble_boosting_bm25() {
        python -m pts.main run_ensemble_models --project="$1" --models=boosting --models=BM25Baseline
}

function run_ensemble_bm25_fail_basic() {
        python -m pts.main run_ensemble_models --project="$1" --models=Fail-Basic --models=BM25Baseline
}

###########################################
# IR baseline
##########################################
function run_tfidf_baseline() {
        python -m pts.main run_tfidf_baseline --project="$1"
}

function preprocess_bm25_baseline() {
        python -m pts.main preprocess_bm25_data --project="$1"
}

function run_bm25_baseline() {
        python -m pts.main run_bm25_baseline --project="$1"
}

function run_bm25_no_context_baseline() {
        python -m pts.main run_bm25_no_context_baseline --project="$1"
}

function run_TFIDF_content_baseline() {
        python -m pts.main run_TFIDF_content_baseline --project="$1"
}

#######################################
# Trainer function for CodeBert
#######################################

function train_codebert_fail_model() {
        python -m pts.main run_codebert_model --project="$1" --data_type=Fail --mode=train
}

#######################################
# Trainer function for the rank models
#######################################

function run_adamboosting_models() {
        python -m pts.main run_boosting --project="$1"
}

function train_rank_triplet_model() {
        python -m pts.main run_rank_model --project="$1" --config="triplet.yaml" --which=triplet
}

function train_rank_starts_basic_model() {
        python -m pts.main run_rank_model --project="$1" --config="rank-basic.yaml" --which=STARTS-Basic
}

function train_rank_starts_code_model() {
        python -m pts.main run_rank_model --project="$1" --config="rank-code.yaml"\
               --which=STARTS-Code
}

function train_rank_starts_method_code_model() {
        python -m pts.main run_rank_model --project="$1" --which=STARTS-mCode --config="rank-method-code.yaml" --mode=train
}

function train_rank_starts_abs_model() {
        python -m pts.main run_rank_model --project="$1" --config="rank-abs.yaml"\
               --which=STARTS-ABS
}

function train_rank_ekstazi_basic_model() {
        python -m pts.main run_rank_model --project="$1" --config="rank-basic.yaml" --which=Ekstazi-Basic
}

function train_rank_ekstazi_code_model() {
        python -m pts.main run_rank_model --project="$1" --config="rank-code.yaml" --which=Ekstazi-Code
}

function train_rank_ekstazi_method_code_model() {
        python -m pts.main run_rank_model --project="$1" --which=Ekstazi-mCode --config="rank-method-code.yaml" --mode=train
}

function train_rank_ekstazi_abs_model() {
        python -m pts.main run_rank_model --project="$1" --config="rank-abs.yaml"\
               --which=Ekstazi-ABS
}

function train_rank_fail_basic_model() {
        python -m pts.main run_rank_model --project="$1" --config="rank-basic.yaml"\
               --which=Fail-Basic
}

function train_rank_fail_code_model() {
        python -m pts.main run_rank_model --project="$1" --config="rank-code.yaml"\
               --which=Fail-Code
}

function train_rank_fail_method_code_model() {
        python -m pts.main run_rank_model --project="$1" --which=Fail-mCode --config=rank-method-code.yaml --mode=train
}

function train_rank_fail_abs_model() {
        python -m pts.main run_rank_model --project="$1" --config="rank-abs.yaml"\
               --which=Fail-ABS
}

function run_boosting_rank_models() {
        python -m pts.main run_boosting --project="$1"
}

## functions for the bm25-integrated model
function train_rank_bm25_basic_model() {
        python -m pts.main run_rank_model --project="$1" --config="rank-bm25.yaml" --which=Fail-Basic
}

####################################
# Get running time for the projects
####################################
function get_eval_running_time() {
        python -m pts.main get_eval_time --project="$1"
}

#######################################
# Tester function for the rank models
#######################################

function test_all_baselines_rank_models() {
        proj="$1"
        test_rank_ekstazi_basic_model $proj
        test_rank_ekstazi_code_model $proj
        test_rank_ekstazi_abs_model $proj
        test_rank_fail_basic_model $proj
        test_rank_fail_code_model $proj
        test_rank_fail_abs_model $proj
}

function test_rank_triplet_model() {
        python -m pts.main run_rank_model --project="$1" --which=triplet --config="triplet.yaml" --mode=test
}

function test_rank_ekstazi_basic_model() {
        python -m pts.main run_rank_model --project="$1" --which=Ekstazi-Basic --config="rank-basic.yaml" --mode=test
}

function test_rank_ekstazi_code_model() {
        python -m pts.main run_rank_model --project="$1" --which=Ekstazi-Code --config="rank-code.yaml" --mode=test
}

function test_rank_starts_method_code_model() {
        python -m pts.main run_rank_model --project="$1" --which=STARTS-mCode --config="rank-method-code.yaml" --mode=test
}

function test_rank_ekstazi_abs_model() {
        python -m pts.main run_rank_model --project="$1" --which=Ekstazi-ABS --config="rank-abs.yaml" --mode=test
}

function test_rank_fail_basic_model() {
        python -m pts.main run_rank_model --project="$1" --which=Fail-Basic --config="rank-basic.yaml" --mode=test
}

function test_rank_fail_code_model() {
        python -m pts.main run_rank_model --project="$1" --which=Fail-Code --config="rank-code.yaml" --mode=test
}

function test_rank_fail_abs_model() {
        python -m pts.main run_rank_model --project="$1" --which=Fail-ABS --config="rank-abs.yaml" --mode=test
}

function test_rank_fail_method_abs_model() {
        python -m pts.main run_rank_model --project="$1" --which=Fail-mABS --config="rank-method-abs.yaml" --mode=test
}

function test_fail_codebert_model() {
        python -m pts.main run_codebert_model --project="$1" --data_type=Fail --mode=test
}

function test_rank_bm25_basic_model() {
        python -m pts.main run_rank_model --project="$1" --which=Fail-Basic --config="rank-bm25.yaml" --mode=test
}

#######################################
# Analyze function for the rank models
#######################################

function get_best_safe_selection_rate_models() {
        python -m pts.main run_rank_model --project="$1" --which=Ekstazi-Basic --config="rank-basic.yaml" --mode=best-selection-rate
        python -m pts.main run_rank_model --project="$1" --which=Ekstazi-Code --config="rank-code.yaml" --mode=best-selection-rate
        python -m pts.main run_rank_model --project="$1" --which=Ekstazi-ABS --config="rank-abs.yaml" --mode=best-selection-rate
        python -m pts.main run_rank_model --project="$1" --which=Fail-Basic --config="rank-basic.yaml" --mode=best-selection-rate
        python -m pts.main run_rank_model --project="$1" --which=Fail-Code --config="rank-code.yaml" --mode=best-selection-rate
        python -m pts.main run_rank_model --project="$1" --which=Fail-ABS --config="rank-abs.yaml" --mode=best-selection-rate
}

function get_first_fail_test_selection_rate_models() {
        python -m pts.main run_rank_model --project="$1" --which=Ekstazi-Basic --config="rank-basic.yaml" --mode=first-fail-selection-rate
        python -m pts.main run_rank_model --project="$1" --which=Ekstazi-Code --config="rank-code.yaml" --mode=first-fail-selection-rate
        python -m pts.main run_rank_model --project="$1" --which=Ekstazi-ABS --config="rank-abs.yaml" --mode=first-fail-selection-rate
        python -m pts.main run_rank_model --project="$1" --which=Fail-Basic --config="rank-basic.yaml" --mode=first-fail-selection-rate
        python -m pts.main run_rank_model --project="$1" --which=Fail-Code --config="rank-code.yaml" --mode=first-fail-selection-rate
        python -m pts.main run_rank_model --project="$1" --which=Fail-ABS --config="rank-abs.yaml" --mode=first-fail-selection-rate
}

function analyze_rank_models() {
        python -m pts.main run_rank_model --project="$1" --which=Ekstazi-Basic --config="rank-basic.yaml" --mode=analysis
        python -m pts.main run_rank_model --project="$1" --which=Ekstazi-Code --config="rank-code.yaml" --mode=analysis
        python -m pts.main run_rank_model --project="$1" --which=Ekstazi-ABS --config="rank-abs.yaml" --mode=analysis
        python -m pts.main run_rank_model --project="$1" --which=Fail-Basic --config="rank-basic.yaml" --mode=analysis
        python -m pts.main run_rank_model --project="$1" --which=Fail-Code --config="rank-code.yaml" --mode=analysis
        python -m pts.main run_rank_model --project="$1" --which=Fail-ABS --config="rank-abs.yaml" --mode=analysis
}

# analyze bm25 model
function analyze_rank_bm25_basic_model() {
        python -m pts.main run_rank_model --project="$1" --which=Fail-Basic --config="rank-bm25.yaml" --mode=analysis
}

function analyze_triplet_model() {
        python -m pts.main run_rank_model --project="$1" --which=triplet --config="triplet.yaml" --mode=analysis
}

function analyze_ensemble_model() {
        # python -m pts.main run_rank_model --project="$1" --which=Ekstazi-Basic-BM25Baseline --config="triplet.yaml" --mode=analysis
    # python -m pts.main run_rank_model --project="$1" --which=Fail-Code-BM25Baseline --config="rank-abs.yaml" --mode=analysis
        python -m pts.main run_rank_model --project="$1" --which=Fail-Basic-BM25Baseline --config="rank-abs.yaml" --mode=analysis
}

function analyze_boosting_model() {
        python -m pts.main run_rank_model --project="$1" --which=boosting --config="triplet.yaml" --mode=analysis
}

function analyze_TFIDF_model(){
        python -m pts.main run_rank_model --project="$1" --which=TFIDFBaseline --config="rank-basic.yaml" --mode=analysis
}

function analyze_BM25_model(){
        python -m pts.main run_rank_model --project="$1" --which=BM25Baseline --config="rank-basic.yaml" --mode=analysis
}

function analyze_EALRTS_randomforest_model(){
        python -m pts.main run_rank_model --project="$1" --which=randomforest --config="rank-basic.yaml" --mode=analysis
}

function analyze_EALRTS_xgboost_model(){
        python -m pts.main run_rank_model --project="$1" --which=xgboost --config="rank-basic.yaml" --mode=analysis
}

function analyze_BM25_no_context_model(){
        python -m pts.main run_rank_model --project="$1" --which=BM25NoContext --config="rank-basic.yaml" --mode=analysis
}

function analyze_TFIDF_content_model(){
        python -m pts.main run_rank_model --project="$1" --which=TFIDFContent --config="rank-basic.yaml" --mode=analysis
}

function analyze_rank_fail_basic_models() {
        python -m pts.main run_rank_model --project="$1" --which=Fail-Basic --config="rank-basic.yaml" --mode=analysis
}

function analyze_rank_starts_mcode_models() {
        python -m pts.main run_rank_model --project="$1" --which=STARTS-mCode --config="rank-method-code.yaml" --mode=analysis
}

function analyze_rank_fail_mabs_models() {
        python -m pts.main run_rank_model --project="$1" --which=Fail-mABS --config="rank-method-code.yaml" --mode=analysis
}

# ==========
# Main function -- program entry point
# This script can be executed as ./run.sh the_function_to_run

function main() {
        local action=${1:?Need Argument}; shift

        ( cd ${_DIR}
          $action "$@"
        )
}

main "$@"
