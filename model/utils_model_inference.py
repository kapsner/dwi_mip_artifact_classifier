#!/usr/bin/env python

__author__ = "Lorenz A. Kapsner"
__copyright__ = "Copyright 2020-2021, Universitätsklinikum Erlangen"


import logging
import os
import torch

from datetime import datetime

from classifier_binary import Net, NetDataModuleTest
import pytorch_lightning as pl


def model_inference(
    inference_info,
    cf,
    use_gpu,
    accelerator,
    args,
    chk_pt_path
):

    # load model hyper parameters
    for _chkpath in chk_pt_path:
        if not os.path.exists(os.path.abspath(_chkpath)):
            raise ValueError(
                "Path/File '{}' does not exist.".format(_chkpath)
            )

        # load data module parameters
        dm_params = cf.datamodule_hyperparameters(
            batch_size=args.batch_size
        )

        # seed everything
        pl.seed_everything(0)

        # load the model from checkpoint
        model = Net.load_from_checkpoint(
            checkpoint_path=_chkpath,
            # args.hyperparameters (not needed since checkpoint saves hparams)
            hparams_file=None,
            map_location=None,
            inference_mode=True
        )

        # instantiate the data module
        test_dm = NetDataModuleTest(
            test_data=inference_info,
            conf=dm_params,
            num_classes=model.hparams.num_classes
        )

        # define name for file saving
        nameparts = _chkpath.split("/checkpoints")[0].split(os.path.sep)[-2:]
        run_name = nameparts[0] + "/" + nameparts[1]
        run_version = run_name + "_" + \
            str(datetime.now().strftime("%Y%m%d-%H%M"))
        logging.info("run_version: {}".format(run_version))

        # instantiate the trainer
        # test trainer
        trainer = pl.Trainer(
            accelerator=accelerator,
            logger=cf.logger_config(version=run_version),
            default_root_dir=cf.exp_dir,
            deterministic=True,
            gpus=use_gpu
        )

        # test the model
        trainer.test(
            model=model,
            datamodule=test_dm
        )
