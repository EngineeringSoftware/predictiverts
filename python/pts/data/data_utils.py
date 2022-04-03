import random
from pathlib import Path

from seutil import IOUtils

from pts.Environment import Environment
from pts.Macros import *


def upsample_dataset(src_data_dir: Path, output_dir: Path, ratio=0.5, target=1, data_type="A"):
    """
    Upsample to make dataset balanced. Target is the arg to control downsampling which label (0 or 1).
    ratio: control what percent should target have.
    """
    # first read data
    objs = IOUtils.load_json_stream(src_data_dir)
    neg_data = list()
    pos_data = list()
    if ratio == 0:
        create_split_dataset(src_data_dir, output_dir)
        return
    for obj in objs:
        if obj["label"] == 0:
            neg_data.append(obj)
        else:
            pos_data.append(obj)
    # end for
    random.seed(Environment.random_seed)
    if target == 1:
        extra_sampling_num = round((1 - ratio) / ratio * len(neg_data)) - len(pos_data)
        upsample_data = random.choices(pos_data, k=extra_sampling_num)
        new_dataset = upsample_data + neg_data
    # end if
    print(
        f"Upsampled dataset has the size of {len(new_dataset)}, positive examples account for {ratio} of total data.")
    random.shuffle(new_dataset)
    if not output_dir.exists():
        IOUtils.mk_dir(output_dir)
    IOUtils.dump(output_dir / f"{data_type}-mutant.json", new_dataset, IOUtils.Format.jsonNoSort)
    # split the data
    valid_num = round(0.1 * len(new_dataset))
    valid_data = new_dataset[: valid_num]
    train_data = new_dataset[valid_num:]
    IOUtils.dump(output_dir / "train.json", train_data, IOUtils.Format.jsonNoSort)
    IOUtils.dump(output_dir / "valid.json", valid_data, IOUtils.Format.jsonNoSort)
    # Get metrics for dataset
    stats_balanced_dataset = {
        "train-num": len(train_data),
        "valid-num": len(valid_data)
    }
    print(stats_balanced_dataset)

def downsample_dataset(src_data_dir: Path, output_dir: Path, ratio=0.5, target=0, data_type="A"):
    """
    Downsample to make dataset balanced. Target is the arg to control downsampling which label (0 or 1).
    ratio: control what percent should target have.
    """
    # first read data
    objs = IOUtils.load_json_stream(src_data_dir)
    neg_data = list()
    pos_data = list()
    if ratio == 0:
        create_split_dataset(src_data_dir, output_dir)
        return
    for obj in objs:
        if obj["label"] == 0:
            neg_data.append(obj)
        else:
            pos_data.append(obj)
    # end for
    random.seed(Environment.random_seed)
    if target == 0:
        sampling_num = round((1 - ratio) / ratio * len(pos_data))
        downsample_data = random.choices(neg_data, k=sampling_num)
        new_dataset = downsample_data + pos_data
    else:
        sampling_num = round((1 - ratio) / ratio * len(pos_data))
        downsample_data = random.choices(pos_data, k=sampling_num)
        new_dataset = downsample_data + neg_data
    # end if
    print(
        f"Downsampled dataset has the size of {len(new_dataset)}, positive examples account for {ratio} of total data.")
    random.shuffle(new_dataset)
    if not output_dir.exists():
        IOUtils.mk_dir(output_dir)
    IOUtils.dump(output_dir / f"{data_type}-mutant.json", new_dataset, IOUtils.Format.jsonNoSort)
    # split the data
    valid_num = round(0.1 * len(new_dataset))
    valid_data = new_dataset[: valid_num]
    train_data = new_dataset[valid_num:]
    IOUtils.dump(output_dir / "train.json", train_data, IOUtils.Format.jsonNoSort)
    IOUtils.dump(output_dir / "valid.json", valid_data, IOUtils.Format.jsonNoSort)
    # Get metrics for dataset
    stats_balanced_dataset = {
        "train-num": len(train_data),
        "valid-num": len(valid_data)
    }
    print(stats_balanced_dataset)
    # IOUtils.dump(Macros.results_dir/"metrics"/f"stats-seq2pred-downsample-{ratio}-data.json", stats_balanced_dataset)


def create_split_dataset(src_data_dir: Path, output_dir: Path):
    # first read data
    objs = IOUtils.load_json_stream(src_data_dir)
    new_dataset = list()
    for obj in objs:
        new_dataset.append(obj)
    # end for
    print(f"Downsampled dataset has the size of {len(new_dataset)}")
    random.shuffle(new_dataset)
    if not output_dir.exists():
        IOUtils.mk_dir(output_dir)
    IOUtils.dump(output_dir / f"raw_dataset.json", new_dataset, IOUtils.Format.jsonNoSort)
    # split the data
    valid_num = round(0.1 * len(new_dataset))
    valid_data = new_dataset[: valid_num]
    train_data = new_dataset[valid_num:]
    IOUtils.dump(output_dir / "train.json", train_data, IOUtils.Format.jsonNoSort)
    IOUtils.dump(output_dir / "valid.json", valid_data, IOUtils.Format.jsonNoSort)
    # # Get metrics for dataset
    # stats_balanced_dataset = {
    #     "train-num": len(train_data),
    #     "valid-num": len(valid_data)
    # }
    # IOUtils.dump(Macros.results_dir/"metrics"/f"stats-seq2pred--{ratio}-data.json", stats_balanced_datas
    return


def filter_mutators(src_data_dir: Path, output_dir: Path):
    """
    Filter part of the mutators and dump the remaining mutants data
    :param src_data_dir:
    :param output_dir:
    :return:
    """
    # first read data
    objs = IOUtils.load_json_stream(src_data_dir)
    filtered_data = list()
    selected_mutators = {"BBooleanTrueReturnValsMutator", "BooleanFalseReturnValsMutator",
                         "ConditionalsBoundaryMutator", "MathMutator", "PrimitiveReturnsMutator"}
    # filter the unwanted data
    for obj in objs:
        if obj["mutator"] not in selected_mutators:
            continue
        filtered_data.append(obj)
    # end for
    # Dump the filtered data
    print(f"In total there are {len(filtered_data)} data point after removing half of the mutators.")
    IOUtils.dump(output_dir / "half-selected-mutant-processed-data.json", filtered_data,
                 IOUtils.Format.jsonNoSort)
    # Do downsampling
    downsample_dataset(src_data_dir, output_dir)


def filter_duplicates(src_data_dir: Path, output_dir: Path, tool: str):
    """
    Filter the duplicated data for Ekstazi and STARTS, because if we want to how tools predict, the only needed inputs are
    changed class name and test class name.
    :param src_data_dir:
    :param output_dir:
    :return:
    """
    objs = IOUtils.load_json_stream(src_data_dir)
    new_data_list = list()
    data_set = set()
    pos = 0
    neg = 0
    for obj in objs:
        hash_idx = obj["input"][0] + obj["input"][2]
        if hash_idx not in data_set:
            data_set.add(hash_idx)
            new_data_list.append(obj)
            if obj["label"] == 1:
                pos += 1
            elif obj["label"] == 0:
                neg += 1
    # end for
    IOUtils.dump(output_dir / f"{tool}-mutant.json", new_data_list, IOUtils.Format.jsonNoSort)
    print(f"After removing the duplicates, there are {pos} positive examples and {neg} negative examples.")

def data_augument(src_data_dir: Path, output_dir: Path, tool: str):
    """
    Data Augumentation: manually adding some data to let model select test file given changed test files.
    :param src_data_dir:
    :param output_dir:
    :return:
    """
    if not output_dir.exists():
        output_dir.mkdir()
    objs = IOUtils.load_json_stream(src_data_dir)
    new_data_list = list()
    test_class_set = []
    for obj in objs:
        test_class_set.append(obj["input"][2])
        new_data_list.append(obj)
    # end for
    for tc in test_class_set:
        data_point = {
            "label": 1,
            "input": [
                tc, "", tc
            ]
        }
        new_data_list.append(data_point)
        neg_tc = tc
        while neg_tc == tc:
            neg_tc = random.choice(list(test_class_set))
        # end while
        assert neg_tc != tc
        data_point = {
            "label": 0,
            "input": [
                tc, "", neg_tc
            ]
        }
        new_data_list.append(data_point)
    # end for
    random.shuffle(new_data_list)
    IOUtils.dump(output_dir / f"{tool}-mutant.json", new_data_list, IOUtils.Format.jsonNoSort)
