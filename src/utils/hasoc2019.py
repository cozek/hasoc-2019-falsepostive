"""
Utils for HASOC 2019 dataset
"""
from argparse import Namespace
from tqdm import tqdm
import pandas as pd  # type: ignore
import numpy as np
import random
import torch
import mmap
import csv
import io
import os


def open_data_as_df(file_location):
    data = []
    ids = []
    task_1 = []
    task_2 = []
    task_3 = []
    with open(file_location, "r") as tsvin:
        tsvin = csv.reader(tsvin, delimiter="\t")
        count = 0
        for row in tsvin:
            ids.append(row[0])
            data.append(row[1])
            task_1.append(row[2])
            task_2.append(row[3])
            task_3.append(row[4])

    # first entry is just the row name, hence removed
    data.pop(0)
    ids.pop(0)
    task_1.pop(0)
    task_2.pop(0)
    task_3.pop(0)

    d = {"id": ids, "text": data, "task_1": task_1, "task_2": task_2, "task_3": task_3}
    df = pd.DataFrame(data=d)
    return df


def labeller(df: pd.DataFrame, task: str, drop_cols: bool = True):
    """Adds a label to the samples in the given DataFrame
    Args:
        df (pandas.DataFrame): A dataframe containing the samples their given labels for each task
        task: one of 1,2,3
        drop_cols: drops some columns that are not necessary
    Returns:
        df (pandas.DataFrame): with a added column that labels of each sample respectively
        label_dict (dict): The labels and their corresponding integer values
    """
    assert task in [1, 2, 3]
    assert isinstance(drop_cols, bool)
    assert isinstance(df, pd.DataFrame)
    labels = {
        1: {"NOT": 0, "HOF": 1},
        2: {"HATE": 0, "PRFN": 1, "OFFN": 2},
        3: {"UNT": 0, "TIN": 1},
    }

    label_list = [
        labels[task][lbl] if lbl != "NONE" else -1 for lbl in df[f"task_{task}"]
    ]

    df["label"] = label_list

    if drop_cols:
        df = df[["text", "label"]]
        df = df.drop(df[df.label == -1].index)

    return df, labels[task]
