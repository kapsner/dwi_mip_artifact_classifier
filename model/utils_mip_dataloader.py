#!/usr/bin/env python

__author__ = "Lorenz A. Kapsner"
__copyright__ = "Copyright 2020-2021, Universitätsklinikum Erlangen"


import os
import pandas as pd
import torch
import numpy as np

from pytorch_lightning.metrics.utils import to_onehot


class MipDatasetBinary(torch.utils.data.Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        base_dir: str,
        num_class: int,
        transform=None
    ):
        self.dataframe = dataframe
        self.base_dir = base_dir
        self.num_class = num_class
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(
            self.base_dir,
            str(self.dataframe.iloc[idx]["subject_id_pseudonymized"]),
            str(self.dataframe.iloc[idx]["study_date"]),
            self.dataframe.iloc[idx]["img_file"]
        )

        # one hot encode target class
        img_label = to_onehot(
            label_tensor=torch.tensor(
                [self.dataframe.iloc[idx]["target_class"]]  # list of integers
            ),
            num_classes=self.num_class
        )

        img = np.load(img_path)
        img = img[np.newaxis]  # add channel axis

        if self.transform:
            img = self.transform(img)

        _id = str(self.dataframe.iloc[idx]["subject_id_pseudonymized"]) + \
            "_" + \
            str(self.dataframe.iloc[idx]["study_date"]) + \
            "_" + \
            str(self.dataframe.iloc[idx]["side"])

        sample = {
            "image": torch.from_numpy(img).float(),
            # reduce dimension (remove first dimension)
            "target": torch.squeeze(img_label),
            "id": _id
        }

        return sample
