from typing import *

import os
from pathlib import Path


class Macros:
    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    python_dir: Path = this_dir.parent
    project_dir: Path = python_dir.parent
    paper_dir: Path = project_dir / "papers" / "fase2022"

    data_dir: Path = project_dir / "data"
    tools_dir: Path = project_dir / "tools" 
    build_logs_dir: Path = project_dir / "tools" / "travistorrent-tools" / "build_logs"
    raw_data_dir: Path = data_dir / "raw-data"
    eval_data_dir: Path = project_dir / "evaldata"
    raw_eval_data_fixed_dir: Path = eval_data_dir / "raw-eval-data-fixed"
    raw_eval_data_dir: Path = eval_data_dir / "raw-eval-data"
    raw_eval_data_no_dep_updated_dir: Path = eval_data_dir / "raw-eval-data-updated"
    raw_eval_data_adding_deleted_dir: Path = eval_data_dir / "raw-eval-data-adding-deleted"
    raw_eval_data_adding_time_dir: Path = eval_data_dir / "raw-eval-data-adding-time"
    flaky_raw_data_dir: Path = data_dir / "raw-data-flaky"
    model_data_dir: Path = data_dir / "model-data"
    model_result_dir: Path = data_dir / "model-results"
    results_dir: Path = project_dir / "results"
    repos_downloads_dir: Path = project_dir / "_downloads"
    repos_results_dir: Path = project_dir / "_results"
    docs_dir: Path = project_dir / "docs"
    ml_logs_dir: Path = python_dir / "ml-logs"
    config_dir: Path = python_dir / "configs"
    metrics_dir: Path = results_dir / "metrics"
    plot_dir: Path = results_dir / "plots"

    collector_dir: Path = project_dir / "collector"
    collector_version = "0.1-dev"

    mutator_dir: Path = this_dir / "mutator" / "universalmutator"

    train = "train"
    val = "val"
    test = "test"

    multi_processing = 8
    trials = 3
    train_ratio = 0.8
    val_ration = 0.1
    test_ration = 0.1

    # projects = ['apache_commons-dbcp',
    #             'jhy_jsoup',
    #             'apache_commons-math',
    #             'apache_httpcomponents-core',
    #             'apache_commons-collections',
    #             'apache_commons-io',
    #             ]

    eval_projects = {
        # "vkostyukov_la4j": "e327ca6e", # latest SHA
        "apache_commons-lang": "b199af8d",
        # "assertj_assertj-core": "58b80881",
        # "msgpack_msgpack-java" : "103d1e1b",
        # # ------------- multi module project
        # "square_wire":"74f36cc3",
        # "caelum_vraptor4": "902ef634", # latest SHA
        # "cloudfoundry_uaa": "77a7df8e", # latest SHA

        #"logstash_logstash-logback-encoder": " 32ff8a6c",
        #"google_compile-testing": "77a7df8e",
        # "apache_commons-codec": "c4e813f8",
        #"apache_commons-dbcp": "d7a3d808",
        #"apache_commons-pool": "ef779fad",
        #"apache_commons-math": "859de3fcba23949bf610ff33e1efd66a9f9f0a82",
        #"apache_commons-io": "8af6781f",
        #"apache_commons-functor": "58b48273",
        "apache_commons-net": "67231c6d",
        #"apache_commons-text": "343b4a9c",
        "apache_commons-validator": "e15d8728",
        #"apache_commons-dbutils": "453f2fc7",
        #"apache_commons-scxml": "e1c43b8b", # latest sha
        "apache_commons-csv": "6aa17567",
        #"apache_commons-jexl": "720f3f76", # latest sha
        # "apache_commons-cli": "e093df2c",
        "asterisk-java_asterisk-java": "aca95a75",
        "Bukkit_Bukkit": "f210234e",
        #"apache_commons-compress": "1b7528fb",
        "apache_commons-configuration": "e2b0ff82",
        #"apache_commons-crypto": "2ed0f2bd",
        #"eclipse_eclipse-collections": "68fd7172",
        #"apache_empire-db": "7ec08fb7",
        "zeroturnaround_zt-exec": "f0526841",
        # "EsotericSoftware_yamlbeans": "26e445bf",
        "mikera_vectorz": "95831ecc",
        #"google_truth": "cc12e162",
        #"scribejava_scribejava": "cd20f57f",
        #"square_retrofit": "", gradle
        #"igniterealtime_Openfire": "f9e0cfdd",
        #"JodaOrg_joda-time": "", junit3
        #"graphhopper_graphhopper": "a05de4b1",
        #"jfree_jfreechart": "d03e68ac",
        #"google_google-java-format": "b9fd8d22",
        "frizbog_gedcom4j": "be310f27",
        #----------------------------------------
        #"mybatis_mybatis-3": "e7f59b62",
        #"opentripplanner_OpenTripPlanner": "a7f0ce9e"
    }

    projects = ['apache_commons-dbcp',
                'jhy_jsoup',
                'apache_commons-collections',
                'apache_commons-io'
                ]

    raw_eval_projects = ['commons-cli',
                         'commons-codec',
                         'commons-compress',
                         'commons-configuration',
                         'commons-validator',
                         'commons-csv',
                         'commons-dbcp',
                         'commons-lang',
                         'commons-net',
                         'empire-db',
                         'asterisk-java',
                         'Bukkit',
                         'vraptor4',
                         'gedcom4j',
                         'google-java-format',
                         'vectorz',
                         'zt-exec'
                        ]

    SKIPS = "-Djacoco.skip -Dcheckstyle.skip -Drat.skip -Denforcer.skip -Danimal.sniffer.skip -Dmaven.javadoc.skip " \
            "-Dfindbugs.skip -Dwarbucks.skip -Dmodernizer.skip -Dimpsort.skip -Dpmd.skip -Dxjc.skip"
