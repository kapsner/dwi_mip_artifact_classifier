#!/usr/bin/env python

__author__ = "Lorenz A. Kapsner"
__copyright__ = "Copyright 2020-2021, Universitätsklinikum Erlangen"

import pandas as pd

from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.metrics.utils import to_categorical

from utils_mip_dataloader import MipDatasetBinary
from utils import log_class_distribution

from utils_backbone import BackboneModel


class Net(BackboneModel):
    # https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # swith off some metrics
        self._metrics = {
            "average_precision": "avgpr",
            "auroc": "auroc",
            "accuracy": "acc",
            "f1": "f1",
            "fbeta": "fbeta",
            "precision": "precision",
            "recall": "recall",
            "auroc": "auroc"
        }

    @staticmethod
    def loss(preds, targets, pos_weight=None):
        loss = F.binary_cross_entropy_with_logits(
            input=preds,
            target=targets.type_as(preds),
            pos_weight=pos_weight.type_as(preds)
        )
        return loss

    @staticmethod
    def _y_to_cat(logits, targets):
        y_preds = to_categorical(
            tensor=torch.softmax(logits, dim=1),
            argmax_dim=1
        )
        y_trues = to_categorical(
            tensor=targets,
            argmax_dim=1
        )
        return y_preds, y_trues

    def _shared_step_end(self, outputs, prefix):

        # loss
        loss = self.loss(
            preds=outputs["logits"],
            targets=outputs["target"].type(torch.FloatTensor),
            pos_weight=torch.FloatTensor(self.hparams.weights)
        )

        # update metrics
        _preds, _targets = self._y_to_cat(
            logits=outputs["logits"],
            targets=outputs["target"]
        )

        for _metname, _metshort in self._metrics.items():
            if _metname not in ["average_precision", "auroc"]:
                _metric_preds = _preds.float()  # important for calculating metrics
            else:
                # probabilites, otherwise we will get wrong prauc, auc values
                _metric_preds = torch.softmax(
                    outputs["logits"], dim=1
                )[:, 1].float()  # take only probabilities of the positive class

            eval("self." + prefix + "_" + _metshort)(
                preds=_metric_preds,
                target=_targets
            )

        # confmat https://pytorch-lightning.readthedocs.io/en/latest/metrics.html#confusionmatrix
        # only validation + test
        if prefix in ["valid", "test"]:
            # (metrics class interface)
            eval("self." + prefix + "_confmat")(
                preds=_preds,
                target=_targets
            )

        return {"loss": loss, "batch_size": outputs["batch_size"].sum()}


class NetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        conf,
        num_classes,
        transforms=None
    ):
        super().__init__()

        self.train_data = train_data
        self.test_data = test_data

        self.transforms = transforms
        if self.transforms != None:
            self.transforms.set_random_state(seed=0)

        # get config
        self.batch_size = conf["batch_size"]
        self.base_dir = conf["base_dir"]
        self.num_classes = num_classes
        self.dl_workers = conf["dl_workers"]

    def setup(self, stage=None):

        train, valid = train_test_split(
            self.train_data,
            train_size=0.8,
            stratify=self.train_data["target_class"],
            random_state=0,
            shuffle=True
        )
        log_class_distribution(
            dataframe=train,
            tag="train_ds",
            class_column="target_class"
        )

        log_class_distribution(
            dataframe=valid,
            tag="valid_ds",
            class_column="target_class"
        )

        log_class_distribution(
            dataframe=self.test_data,
            tag="test_ds",
            class_column="target_class"
        )

        if stage == 'fit' or stage is None:
            self.train_ds = MipDatasetBinary(
                dataframe=train,
                base_dir=self.base_dir,
                num_class=self.num_classes,
                transform=self.transforms
            )
            self.valid_ds = MipDatasetBinary(
                dataframe=valid,
                base_dir=self.base_dir,
                num_class=self.num_classes
            )

        # Assign Test split(s) for use in Dataloaders
        if stage == 'test' or stage is None:
            self.test_ds = MipDatasetBinary(
                dataframe=self.test_data,
                base_dir=self.base_dir,
                num_class=self.num_classes
            )

    def train_dataloader(self):
        train_dl = DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dl_workers
        )
        return train_dl

    def val_dataloader(self):
        valid_dl = DataLoader(
            dataset=self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dl_workers
        )
        return valid_dl

    def test_dataloader(self):
        test_dl = DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dl_workers
        )
        return test_dl


class NetDataModuleTest(pl.LightningDataModule):
    def __init__(
        self,
        test_data: pd.DataFrame,
        conf,
        num_classes
    ):
        super().__init__()

        self.test_data = test_data

        # get config
        self.batch_size = conf["batch_size"]
        self.base_dir = conf["base_dir"]
        self.num_classes = num_classes
        self.dl_workers = conf["dl_workers"]

    def setup(self, stage=None):
        # Assign Test split(s) for use in Dataloaders
        if stage == 'test' or stage is None:
            self.test_ds = MipDatasetBinary(
                dataframe=self.test_data,
                base_dir=self.base_dir,
                num_class=self.num_classes
            )

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        test_dl = DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dl_workers
        )
        return test_dl
