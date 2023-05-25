#!/usr/bin/env python

__author__ = "Lorenz A. Kapsner"
__copyright__ = "Copyright 2020-2021, Universitätsklinikum Erlangen"


import os

import pandas as pd
from monai import transforms
import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint


class Config():

    def __init__(
        self,
        pp_data_path: str,
        experiment_dir: str,
        config_csv: str,
        mode: str
    ):
        """
        : param pp_data_path: (str) Path to the folder containing the preprocessed files.
        : param experiment_dir: (str) The directory, where experiment files/logs should be stored.
        : param config_csv: (str) The filename of the CSV file with the configurations. \
            This CSV file is expected to be in the same folder as this 'model_mip_classifier.py'.
        """
        # root directory of the experiment
        root_exp_dir = os.path.abspath(experiment_dir)

        # input path: path to the preprocessed files
        self.pp_data_path = pp_data_path

        # only if mode == "training"
        if mode == "training":
            self.exp_dir = root_exp_dir

            self.param_csv_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                config_csv
            )

            for _p in [self.pp_data_path, self.param_csv_file]:
                if not os.path.exists(_p):
                    raise ValueError(
                        "Path/File '{}' does not exist.".format(_p)
                    )

        elif mode == "inference":
            # put inference results into another folder
            self.exp_dir = os.path.join(
                root_exp_dir,
                "inference"
            )

        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        if mode == "inference":
            os.makedirs(
                os.path.join(
                    self.exp_dir,
                    "test_images"
                ),
                exist_ok=True
            )

        # logger directory:
        self.logger_dir = os.path.join(self.exp_dir, "logs")
        if not os.path.exists(self.logger_dir):
            os.makedirs(self.logger_dir)

        # filename of the csv file containing metadata of the preprocessed files
        # the csv file should be available in self.pp_data_path
        # the filename is defined during the preprocessing
        self.input_df_name = "train_info.csv"

        # load train_info_df
        self.full_df = pd.read_csv(
            filepath_or_buffer=os.path.join(
                self.pp_data_path,
                self.input_df_name
            )
        )
        self.train_info_df = \
            self.full_df[self.full_df.split_type == "train"].reset_index()
        self.inference_info_df = \
            self.full_df[self.full_df.split_type == "test"].reset_index()

    @ staticmethod
    def model_hyperparameters(param_dict):
        """
        Dictionary holding the model hyper parameters

        : optimizer: 'sgd', 'novograd', 'adam'
        : scheduler: 'None', 'plateau', 'onecycle', 'lambdalr', 'warmup_lin', 'warmup_exp'
        """
        model_hparams = {
            "num_classes": int(param_dict["num_classes"]),
            "model_name": str(param_dict["model"]).strip(),
            "weights": [int(param_dict["weight_class_0"]), int(param_dict["weight_class_1"])],
            "dropout_prob": float(param_dict["dropout"]),
            "epochs": int(param_dict["epochs"]),
            "bn_size": 4,
            "optim": str(param_dict["optimizer"]).strip(),
            "learning_rate": float(param_dict["learning_rate"]),
            "momentum": float(param_dict["momentum"]),
            "nesterov": bool(param_dict["nesterov"]),
            "weight_decay": float(param_dict["weight_decay"]),
            "scheduler": str(param_dict["scheduler"]).strip(),
            "gamma": float(param_dict["gamma"]),
            "warm_up_steps": float(param_dict["warm_up_steps"]),
            "monitor_metric": str(param_dict["monitor_metric"]).strip(),
            "pretrained": bool(param_dict["pretrained"]),
            "stochastic_weight_avg": bool(param_dict["stochastic_weight_avg"])
        }
        return model_hparams

    def datamodule_hyperparameters(self, batch_size):
        dm_params = {
            "batch_size": batch_size,  # 128 if os.cpu_count() > 32 else 8,
            "test_split": 0.15,
            "base_dir": self.pp_data_path.strip(),
            "dl_workers": 8 if os.cpu_count() > 32 else 4  # dataloader cpu workers
        }
        return dm_params

    @ staticmethod
    def augmentations(augment=True):

        if augment:
            # set augmentation transforms
            compose = transforms.Compose([
                transforms.RandRotate(range_x=180, prob=0.5, keep_size=True),
                transforms.RandFlip(spatial_axis=0, prob=0.5),
                transforms.RandFlip(spatial_axis=1, prob=0.5),
                transforms.RandZoom(min_zoom=0.5, max_zoom=1.5, prob=0.5)
            ])
        else:
            compose = None
        return compose

    def param_config(self):
        # load csv file
        param_df = pd.read_csv(
            filepath_or_buffer=self.param_csv_file
        )
        param_df = param_df[param_df["execute"].eq(1)]
        # transform to dictionary
        grid_list = param_df.to_dict(orient="records")

        if len(grid_list) < 1:
            raise RuntimeError("Empty parameter config.")

        return grid_list

    def logger_config(self, version=None):
        tensorboard_log = pl.loggers.TensorBoardLogger(
            save_dir=self.logger_dir,
            name="tensorboard",
            version=version,
            default_hp_metric=False
        )

        # csv_log = pl.loggers.CSVLogger(
        #     save_dir=self.logger_dir,
        #     version=version,
        #     name="csvlog"
        # )

        return [tensorboard_log]  # , csv_log]

    @ staticmethod
    def callbacks(monitor_metric: str = "loss/valid"):
        # https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.EarlyStopping.html#pytorch_lightning.callbacks.EarlyStopping
        # define early-stopping callback
        # earlystop_cb = EarlyStopping(
        #     monitor=monitor_metric,
        #     min_delta=0.00,
        #     patience=100,
        #     verbose=False,
        #     mode="min"
        # )

        # https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.LearningRateMonitor.html#pytorch_lightning.callbacks.LearningRateMonitor
        # define learning-rate monitor callback
        learning_rate_monitor_cb = LearningRateMonitor(
            logging_interval=None,  # default selection, based on scheduler
            log_momentum=True
        )

        # https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint
        # define model-checkpoint callback
        model_checkpoint_cb_loss = ModelCheckpoint(
            monitor="loss/valid",
            verbose=False,
            save_last=True,
            save_top_k=3,
            save_weights_only=False,
            mode='min',
            period=1,  # TODO replace with every_n_val_epochs
            dirpath=None,
            filename="{loss/valid:.4f}-{epoch}"
        )

        # @Lorenz monitoring two metrics seems currently not to work yet
        # model_checkpoint_cb_acc = ModelCheckpoint(
        #     monitor="auroc/valid",
        #     verbose=False,
        #     save_last=True,
        #     save_top_k=3,
        #     save_weights_only=False,
        #     mode='max',
        #     period=1,
        #     dirpath=None,
        #     filename="{auroc/valid:.4f}-{epoch}"
        # )

        # return [earlystop_cb, learning_rate_monitor_cb, model_checkpoint_cb]
        # model_checkpoint_cb_acc
        return [model_checkpoint_cb_loss, learning_rate_monitor_cb]
