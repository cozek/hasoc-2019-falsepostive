"""
General Utilities
"""
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from argparse import Namespace
from tqdm import tqdm
import pandas as pd  # type: ignore
import numpy as np
import random
import torch
import mmap
import io
import os


def analyse_preds(y_pred, y_target, threshold=0.5):
    y_pred = (torch.sigmoid(y_pred) > threshold).cpu().long().numpy()
    y_target = y_target.cpu().numpy()

    conmat = confusion_matrix(y_pred=y_pred, y_true=y_target)
    confusion = pd.DataFrame(
        conmat, index=["NOT", "HS"], columns=["predicted_NOT", "predicted_HS"]
    )
    print("acc = ", accuracy_score(y_pred=y_pred, y_true=y_target))
    print(classification_report(y_pred=y_pred, y_true=y_target))
    print(confusion)


def make_train_state():
    return {
        "train_preds": [],
        "train_targets": [],
        "train_accuracies": [],
        "train_losses": [],
        "val_preds": [],
        "val_targets": [],
        "val_accuracies": [],
        "val_losses": [],
        "batch_preds": [],
        "batch_targets": [],
    }


def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def set_seed_everywhere(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe_tensor(x):
    """
    Prints information about a given tensor
    """
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))


FILE = "/Users/cozek/Documents/MTech/3rd Sem/Project/glove.twitter.27B/glove.twitter.27B.200d.txt"

if __name__ == "__main__":
    print(os.path.realpath(FILE))
    with open(FILE):
        pass
