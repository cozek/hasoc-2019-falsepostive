"""
Utils for OLID dataset
"""
from argparse import Namespace
from tqdm import tqdm
import pandas as pd  # type: ignore
import numpy as np
import random
import torch
import mmap
import io
import os


def open_olid(file_path: str = None) -> dict:
    """Loads up OLID1.0 files
    Args:
        file_path: path of data file

    Returns:
        output_dict: dict of dataframes
    """
    if file_path == None:
        file_path = "../OLIDv1.0/"

    paths = {
        "train_data": file_path + "olid-training-v1.0.tsv",
        "test_task_a": file_path + "testset-levela.tsv",
        "test_task_b": file_path + "testset-levelb.tsv",
        "test_task_c": file_path + "testset-levelc.tsv",
        "labels_a": file_path + "labels-levela.csv",
        "labels_b": file_path + "labels-levelb.csv",
        "labels_c": file_path + "labels-levelc.csv",
    }

    output_dict = {}

    with open(paths["train_data"]) as file:
        f = [line.strip().split("\t") for line in file]
        cols = f.pop(0)
        train_df = pd.DataFrame(data=f, columns=cols)

    output_dict["train"] = train_df

    tasks = [
        ("test_task_a", "labels_a"),
        ("test_task_b", "labels_b"),
        ("test_task_c", "labels_c"),
    ]

    for tup in tasks:
        with open(paths[tup[0]]) as test_data:
            test_data = [line.strip().split("\t") for line in test_data]
            cols = test_data.pop(0)
            data_df = pd.DataFrame(data=test_data, columns=cols)
        with open(paths[tup[1]]) as labels:
            labels = [line.strip().split(",") for line in labels]
            cols = ["id", tup[1]]
            label_df = pd.DataFrame(data=labels, columns=cols)
        df = pd.merge(data_df, label_df, on="id")
        output_dict[tup[0]] = df
    return output_dict
