from typing import *

import collections
import copy
import random
from tqdm import tqdm
from pathlib import Path

from seutil import LoggingUtils, IOUtils

from pts.Macros import Macros
from pts.Environment import Environment

class DatasetSplitter:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    def __init__(self):
        self.output_dir = Macros.data_dir / "split"
        IOUtils.rm_dir(self.output_dir)
        IOUtils.mk_dir(self.output_dir)

        self.statistics = dict()
        return

    def split_dataset(self, src_data_dir: Path):
        """
        Splits the dataset to train/test by code changes
        """
        self.statistics.clear()
        processed_data = IOUtils.load(src_data_dir)
        # Do the splitting
        split_data_list = self.perform_splitting(processed_data)
        # Save the splitting data set
        IOUtils.dump(self.output_dir/"train-data.json", split_data_list[Macros.train])
        IOUtils.dump(self.output_dir/"test-data.json", split_data_list[Macros.test])
        # Save the split stats
        IOUtils.dump(self.output_dir/"statistics.json", self.statistics)

    def perform_splitting(self, data_list: List[Dict]):
        split_data: Dict[str, List[Dict]] = collections.defaultdict(list)
        random.seed(Environment.random_seed)
        random.Random(Environment.random_seed).shuffle(data_list)
        data_count = len(data_list)

        # Prepare statistics
        for data_type in [Macros.train, Macros.test]:
            self.statistics[f"num-code-change-{data_type}"] = 0
            self.statistics[f"num-test-case-{data_type}"] = 0
            self.statistics[f"num-failed-test-case-{data_type}"] = 0

        # Split
        while self.statistics[f"num-code-change-{Macros.train}"] / data_count < self.TRAIN_RATIO:
            data_example = data_list.pop()
            split_data[Macros.train].append(data_example)
            self.statistics[f"num-code-change-{Macros.train}"] += 1
            self.statistics[f"num-test-case-{Macros.train}"] += len(data_example["labels"])
            self.statistics[f"num-failed-test-case-{Macros.train}"] += len(data_example["raw_data"]["failed_test_list"])
        # end while
        

        while len(data_list) > 0:
            data_example = data_list.pop()
            split_data[Macros.test].append(data_example)
            self.statistics[f"num-code-change-{Macros.test}"] += 1
            self.statistics[f"num-test-case-{Macros.test}"] += len(data_example["labels"])
            self.statistics[f"num-failed-test-case-{Macros.test}"] += len(data_example["raw_data"]["failed_test_list"])
        # end while
        return split_data
