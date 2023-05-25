#!/usr/bin/env python

__author__ = "Lorenz A. Kapsner"
__copyright__ = "Copyright 2020-2021, Universitätsklinikum Erlangen"


import logging

from classifier_binary import Net, NetDataModule
import pytorch_lightning as pl

from sklearn.model_selection import StratifiedKFold, train_test_split

import numpy as np
import torch


def model_training(
    train_info,
    param_settings,
    cf,
    n_gpus,
    use_gpu,
    accelerator,
    args,
    chk_pt_path,
    dev_run
):

    for param_dict in param_settings:

        logging.info(param_dict)

        # load model hyper parameters
        model_hparams = cf.model_hyperparameters(param_dict=param_dict)
        model_hparams["n_gpus"] = n_gpus

        if args.debugging == True:
            model_hparams["epochs"] = 2

        # load data module parameters
        dm_params = cf.datamodule_hyperparameters(
            batch_size=args.batch_size
        )

        logging.info("Weights: {0}".format(model_hparams["weights"]))

        # seed everything
        pl.seed_everything(0)

        # create folds here // split dataset into train-test here
        if args.kfolds is not None:
            logging.info("Experiment running with {}-fold cross-validation".format(
                args.kfolds
            ))

            skf = StratifiedKFold(
                n_splits=args.kfolds,
                shuffle=True,
                random_state=0
            )

            # split is performed by row-number
            splits = skf.split(
                X=np.zeros(len(train_info)),
                y=train_info["target_class"]
            )
        else:
            # we split on row numbers, therefor we create that
            # sequence of rows first
            sequence_of_rows = [rnum for rnum in range(0, len(train_info))]
            train_index, test_index = train_test_split(
                np.asarray(  # get and return indices
                    sequence_of_rows
                ),
                test_size=dm_params["test_split"],
                stratify=train_info["target_class"],
                random_state=0,
                shuffle=True
            )
            # list of tuple
            splits = [(train_index, test_index)]

        for _fold, split_generator in enumerate(splits):

            train_idx, test_idx = split_generator

            # shuffle indices again
            np.random.shuffle(train_idx)
            np.random.shuffle(test_idx)

            # instantiate the model
            model = Net(**model_hparams)

            # instantiate the data module
            dm = NetDataModule(
                train_data=train_info.iloc[train_idx],
                test_data=train_info.iloc[test_idx],
                conf=dm_params,
                num_classes=model_hparams["num_classes"],
                transforms=cf.augmentations(augment=bool(param_dict["augment"]))
            )

            # instantiate the trainer
            run_version = str(param_dict["model"]).strip() + "_" + \
                str(param_dict["optimizer"]).strip() + "_" + \
                str(param_dict["id"]) + "/fold_" + str(_fold)
            logging.info("run_version: {}".format(run_version))

            trainer = pl.Trainer(
                accelerator=accelerator,
                logger=cf.logger_config(version=run_version),
                max_epochs=model_hparams["epochs"],
                auto_scale_batch_size=None,
                auto_lr_find=False,
                callbacks=cf.callbacks(
                    monitor_metric=model_hparams["monitor_metric"]
                ),
                checkpoint_callback=True,
                fast_dev_run=dev_run,
                check_val_every_n_epoch=1,
                default_root_dir=cf.exp_dir,
                deterministic=True,
                gpus=use_gpu,
                resume_from_checkpoint=chk_pt_path,  # if existing, start from checkpoint
                stochastic_weight_avg=model_hparams["stochastic_weight_avg"]
            )

            # in any case, set chk_pt_path to None for next hparams iteration
            # at least the second hparams iteration will not resume from checkpoint
            chk_pt_path = None

            # fit the model
            trainer.fit(model=model, datamodule=dm)

            # Test multiple "best" checkpoints
            for _ckpt in range(len(trainer.checkpoint_callbacks)):
                logging.info("Testing: monitor metric: {}".format(
                    trainer.checkpoint_callbacks[_ckpt].monitor
                ))
                ckpt_path = trainer.checkpoint_callbacks[_ckpt].best_model_path
                logging.info("Best checkpoint path: {}".format(ckpt_path))

                # test the model using current checkpoint
                trainer.test(ckpt_path=ckpt_path)
