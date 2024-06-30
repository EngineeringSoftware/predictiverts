# predictiverts

ML-based Regression test selection(RTS) models based on mutation analysis and analysis-based RTS.

## Installation

It is recommended to use [conda](https://docs.conda.io/en/latest/) or [virtual environments](https://realpython.com/python-virtual-environments-a-primer/) to manage Python dependencies.

For conda user, run:

```bash
conda create -n prts python=3.7
pip install -r requirements.txt
```

To run the experiments, you need Java 8 and Maven.

## Training Data Collection

We use [PIT](https://pitest.org/) to get mutants for training data. To generate and collect training data from an open-source project:

1. Create download directory `_downloads` and clone the repository `$project` to this directory.

The project list and SHAs we used are documented [here](https://github.com/EngineeringSoftware/predictiverts/blob/43c65cc9bb2b7e1379e101457a734b7b2e01ad25/python/pts/main.py#L34).
To download all the projects:

```bash
./python/run.sh downloads_repos
```

2. Modifying the `pom.xml` file of the `$project` by inserting the following plugin to the `pom.xml`.

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

3. Run pit and get a report in xml format `mutations.xml`

```bash
mvn org.pitest:pitest-maven:mutationCoverage -DoutputFormats=xml
# You will find the report in $repo_dir/target/pit-reports/$date/mutations.xml
```

Then create a directory `_results/$project` and move the report (`mutations.xml`) to this dirctory

4. Parse the pit report and the source code in the project to collect the pit-generated mutants.

```bash
./python/run.sh get_mutants $project
```

If the script runs successfully, you will see `mutant-data.json` and `method-data.json` in the `_results/$project/collector` directory.

5. We provide positive and negative labels to each mutant-test pair. For 'Ekstazi-\*' models, we label the mutant-test pairs based on if the RTS tool (Ekstazi) will select the test or not.

- In order to run Ekstazi, copy the `tools/ekstazi-extesnsion-1.0-SNAPSHOT.jar` to `${MAVEN_HOME}/lib/ext/`. Please refer to [document](tools/xts-extension/README.md) for detailed instructions.
- Note: We use java 1.8 when running Ekstazi

```bash
# Collect labels from Ekstazi results
./python/run.sh get_tools_data_for_mutants $project
```

If the script runs successfully, you will see `mutant-data-rts-tool.json` file in `_results/$project/collector`

6. Create training and validation dataset

- Dataset labeled by Ekstazi

```bash
./python/model_run.sh process_train_data_rank_model $project
```

- Dataset labeled by tests pass or fail

```bash
./python/model_run.sh process_train_data_rank_model_fail $project
```

The data files will be written in `data/model-data/rank-model/$project/`.

## Eval Data Preparation

[sec-downloads]: #data-downloads

All our data is hosted on UTBox via [a shared folder](https://utexas.box.com/s/p0uvysksey7iz0l3fxxqo3k6p6xt78ji).

1. Download eval data and put in the `evaldata/` directory

```bash
mkdir evaldata/
unzip eval-data.zip -d evaldata/
```

2. Process test data for a given project

```bash
./python/model_run.sh process_test_data_rank_model $project
```

The processed evaluation dataset will be store at `data/model-data/rank-model/$project/test.json`

## Model Training

1. Train model with data labeled by Ekstazi

```bash
# train Ekstazi-Basic Model
./python/model_run.sh train_rank_ekstazi_basic_model $project
# train Ekstazi-Code Model
./python/model_run.sh train_rank_ekstazi_code_model $project
# train Ekstazi-ABS model
./python/model_run.sh train_rank_ekstazi_abs_model $project
```

The model checkpoints will be saved to `data/model-data/rank-model/$project/Ekstazi-Basic(Code/ABS)/saved_models/`

2. Train model with data labeled by tests results

```bash
# train Fail-Basic Model
./python/model_run.sh train_rank_fail_basic_model $project
# train Fail-Code Model
./python/model_run.sh train_rank_fail_code_model $project
# train Fail-ABS model
./python/model_run.sh train_rank_fail_abs_model $project
```

The model checkpoints will be saved to `data/model-data/rank-model/$project/Fail-Basic(Code/ABS)/saved_models/`

## BM25 Baseline Results on Evaluation Dataset

1. process evaluaton data for BM25 baseline

```bash
./python/model_run.sh preprocess_bm25_dabaseline $project
```

2. Run BM25 on the evaluation data

```bash
./python/model_run.sh run_bm25_baseline $project
```

## ML Models Evaluation

1. Run evaluation

```bash
./python/model_run.sh test_rank_ekstazi_basic_model $project
./python/model_run.sh test_rank_ekstazi_code_model $project
./python/model_run.sh test_rank_ekstazi_abs_model $project
```

The eval results will be written to `data/model-data/rank-model/$project/Ekstazi-Basic(Code,ABS)/results`
Same for 'Fail-\*' models.

## Research

If you have used our data and code in a research project, please cite
the research paper in any related publication:

```bibtex
@inproceedings{ZhangETAL22Comparing,
  author={Zhang, Jiyang and Liu, Yu and Gligoric, Milos and Legunsen, Owolabi and Shi, August},
  booktitle={International Conference on Automation of Software Test},
  title={Comparing and Combining Analysis-Based and Learning-Based Regression Test Selection},
  year={2022},
  pages={17-28},
}
```
