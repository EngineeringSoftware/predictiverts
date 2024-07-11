# predictiverts

ML-based Regression Test Selection (RTS) models based on mutation
analysis and analysis-based RTS. [Link to the paper that describes the
technique](https://dl.acm.org/doi/10.1145/3524481.3527230).

## Installation

It is recommended to use [conda](https://docs.conda.io/en/latest) or
[virtual
environments](https://realpython.com/python-virtual-environments-a-primer)
to manage Python dependencies.

For conda users, run:

```bash
conda create -n prts python=3.7
conda activate prts
conda install pip
pip install -r requirements.txt # you might have to point to pip that is in your conda env
```

To run the experiments, you need Java 8 and Maven.  Also, you need a
machine with a GPU.

## Training Data Collection

We use [PIT](https://pitest.org) to get mutants for training data. To
generate and collect training data from an open-source project
(`${project}` in the rest of this text):

1. Create download directory `_downloads` and clone the repository
`${project}` to this directory.

The list of projects and SHAs we used are documented
[here](https://github.com/EngineeringSoftware/predictiverts/blob/43c65cc9bb2b7e1379e101457a734b7b2e01ad25/python/pts/main.py#L34).

To download all the projects with a single command, execute the
following:

```bash
mkdir -p _downloads
./python/run.sh downloads_repos
```

You will see 10 projects used in our paper downloaded to `_downloads`
and the corresponding results directories in `_results`; the
`_results` directory will be empty at this point.

2. Enter the project's directory. Make sure to checkout to the correct
   SHA of the `${project}` and the tests can be run without errors. We
   use 'apache_commons-validator' for demonstration in the rest of
   this document.

```bash
cd _downloads/apache_commons-validator
git checkout 97bb5737
mvn test
```

You should see the following as part of the output:
```
[INFO] Tests run: 527, Failures: 0, Errors: 0, Skipped: 0
```

3. Modifying the `pom.xml` file of the `${project}` by inserting the
   following plugin to the `pom.xml`.

```xml
<plugin>
<groupId>org.pitest</groupId>
<artifactId>pitest-maven</artifactId>
<version>1.7.1</version>
<configuration>
  <fullMutationMatrix>true</fullMutationMatrix>
  <outputFormats>
    <param>XML</param>
  </outputFormats>
</configuration>
</plugin>
```

4. Run PIT and get a report in xml format `mutations.xml`.

```bash
mvn org.pitest:pitest-maven:mutationCoverage
# You will find the report in _downloads/apache_commons-validator/target/pit-reports/$date/mutations.xml
```

You should see the following as part of the output:
```
================================================================================
- Statistics
================================================================================
>> Line Coverage: 2360/3030 (78%)
>> Generated 2008 mutations Killed 1498 (75%)
>> Mutations with no coverage 329. Test strength 89%
>> Ran 37400 tests (18.63 tests per mutation)
```

You can find more info about the `mutations.xml` on the PIT web
page. In short, it contains details about generated mutants.

Then copy the report (`mutations.xml`) to the result directory
`_results/${project}`.

```bash
cd ../../
cp _downloads/apache_commons-validator/target/pit-reports/*/mutations.xml _results/apache_commons-validator/
```

5. Parse the PIT report and the project's source code to collect the
   pit-generated mutants.

```bash
./python/run.sh get_mutants apache_commons-validator
```

If the script runs successfully, you will see `mutant-data.json` and
`method-data.json` in the `_results/${project}/collector` directory.

In the example project, the following should be a part of the output:
```
In total 1818 mutants recovered in project apache_commons-validator.
```

`mutant-data.json` contains information about mutants, e.g., list of
tests that kill each mutant and those that do not kill it, context
(i.e., mutated method).

`method-data.json` contains methods from the repository, their
argument types, etc.

6. We provide positive and negative labels to each mutant-test
   pair. For 'Ekstazi-\*' models, we label the mutant-test pairs based on
   RTS results, i.e., if the RTS tool (Ekstazi) select the test or not.

- In order to run Ekstazi, copy the `tools/ekstazi-extension-1.0-SNAPSHOT.jar` to `${MAVEN_HOME}/lib/ext/` (i.e., if not set, `MAVEN_HOME` is the maven installation directory). Please refer to [document](tools/xts-extension/README.md) for more details.

```bash
# Collect labels from Ekstazi results
./python/run.sh get_tools_data_for_mutants apache_commons-validator
```

If the script runs successfully, you will see
`mutant-data-rts-tool.json` file in
`_results/${project}/collector`. This file is similar to
`mutant-data.json`, but contains tests selected by Ekstazi.

7. Create training and validation dataset

- Dataset labeled by Ekstazi

```bash
./python/model_run.sh process_train_data_rank_model apache_commons-validator
```

You will see text to stdout similar to the following:
```
apache_commons-validator has 67 test classes in total.
Ekstazi
In total there are 19040 data point for training
In total there are 2838 data point for validation
```

The actual output depends on the seed, which changes from run to run.

The output will be in
`data/model-data/rank-model/commons-validator/Ekstazi/{train,valid}.json`.

- Dataset labeled by tests pass or fail

```bash
./python/model_run.sh process_train_data_rank_model_fail apache_commons-validator
```

You will see text to stdout similar to the following:
```
In total there are 85980 data point for training
In total there are 8240 data point for validation
```

The actual output depends on the seed, which changes from run to run.

The output will be in
`data/model-data/rank-model/commons-validator/Fail/{train,valid}.json`.

## Evaluation Data Preparation

[sec-downloads]: #data-downloads

Test data is hosted on UTBox via [a shared
folder](https://utexas.box.com/s/p0uvysksey7iz0l3fxxqo3k6p6xt78ji).

1. Download eval data and put in the `evaldata` directory

```bash
mkdir -p evaldata
unzip eval-data.zip -d evaldata
```

2. Process test data for a given project

```bash
./python/model_run.sh process_test_data_rank_model apache_commons-validator
```

The processed evaluation dataset will be store at
`data/model-data/rank-model/${project}/test.json`. This file is in a
similar format as `test.json` and `valid.json`.

## Model Training

1. Train model with data labeled by Ekstazi

```bash
# train Ekstazi-Basic Model
./python/model_run.sh train_rank_ekstazi_basic_model apache_commons-validator
# train Ekstazi-Code Model
./python/model_run.sh train_rank_ekstazi_code_model apache_commons-validator
# train Ekstazi-ABS model
./python/model_run.sh train_rank_ekstazi_abs_model apache_commons-validator
```

The model checkpoints will be saved to
`data/model-data/rank-model/${project}/Ekstazi-{Basic,Code,ABS}/saved_models`.

2. Train model with data labeled by tests results

```bash
# train Fail-Basic Model
./python/model_run.sh train_rank_fail_basic_model apache_commons-validator
# train Fail-Code Model
./python/model_run.sh train_rank_fail_code_model apache_commons-validator
# train Fail-ABS model
./python/model_run.sh train_rank_fail_abs_model apache_commons-validator
```

The model checkpoints will be saved to
`data/model-data/rank-model/${project}/Fail-{Basic,Code,ABS}/saved_models`.

## BM25 Baseline Results on Evaluation Dataset

1. Process evaluation data for BM25 baseline

```bash
./python/model_run.sh preprocess_bm25_baseline apache_commons-validator
```

The processed data will be written to
`evaldata/mutated-eval-data/f"{project}-ag-preprocessed.json"

2. Run BM25 on the evaluation data

```bash
./python/model_run.sh run_bm25_baseline apache_commons-validator
./python/model_run.sh analyze_BM25_model apache_commons-validator
```

The results will be written to
`results/modelResults/${project}/BM25Baseline/best-safe-selection-rate.json`.

The numbers (Baseline BM25) reported in the Table 4 ('best safe
selection rate of models that select from subset of Ekstazi')
correspond to the value of 'Ekstazi-subset-best-safe-selection-rate'
in the file 'best-safe-selection-rate.json'.

The numbers (Baseline BM25) reported in the Table 5 ('best safe
selection rate of models that select from subset of STARTS')
correspond to the value of 'STARTS-subset-best-safe-selection-rate' in
the file 'best-safe-selection-rate.json'.

## ML Models Evaluation

Run evaluation:

```bash
./python/model_run.sh test_rank_ekstazi_basic_model apache_commons-validator
./python/model_run.sh test_rank_ekstazi_code_model apache_commons-validator
./python/model_run.sh test_rank_ekstazi_abs_model apache_commons-validator
```

The eval results metrics will be written to
`results/modelResults/${project}/Ekstazi-{Basic,Code,ABS}/best-safe-selection-rate.json`
Same for 'Fail-\*' models.

The numbers reported in the Table 4 ('best safe selection rate of
models that select from subset of Ekstazi') correspond to the value of
'Ekstazi-subset-best-safe-selection-rate' in the file
'best-safe-selection-rate.json'.

The numbers reported in the Table 5 ('best safe selection rate of
models that select from subset of STARTS') correspond to the value of
'STARTS-subset-best-safe-selection-rate' in the file
'best-safe-selection-rate.json'.

Due to stochastic nature of models, the train-test loop should be
repeated a number of times and then averages, medians, and other
statistics should be considered.

## Research

If you have used our data or code in a research project, please cite:

```bibtex
@inproceedings{ZhangETAL22Comparing,
  author = {Zhang, Jiyang and Liu, Yu and Gligoric, Milos and Legunsen, Owolabi and Shi, August},
  booktitle = {International Conference on Automation of Software Test},
  title = {Comparing and Combining Analysis-Based and Learning-Based Regression Test Selection},
  year = {2022},
  pages = {17--28},
}
```
