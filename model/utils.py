#!/usr/bin/env python

__author__ = "Lorenz A. Kapsner"
__copyright__ = "Copyright 2020-2021, Universitätsklinikum Erlangen"


import torch.nn.functional as F
import logging
import pandas as pd


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def log_class_distribution(
    dataframe: pd.DataFrame,
    tag: str,
    class_column: str = "target_class"
):
    cls_count = dataframe.groupby(by=class_column).size()

    logging.info("Class distribution -- {0}:\n{1}".format(
        tag, cls_count
    ))
