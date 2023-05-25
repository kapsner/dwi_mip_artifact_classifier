#!/usr/bin/env python

__author__ = "Lorenz A. Kapsner"
__copyright__ = "Copyright 2020-2021, Universitätsklinikum Erlangen"


import argparse

import logging

from sklearn.model_selection import train_test_split

# from classifier_multi import Net, NetDataModule

from utils import log_class_distribution
from model_config import Config
from utils_model_training import model_training
from utils_model_inference import model_inference


if __name__ == "__main__":

    """
    training:
    python model_execution.py -i /home/user/data/tech_artifacts/dwi_mips/pp_dwi_binary_tech_artifacts_221006/ \
        -e /home/user/development/trainings/dwi_mips/tech_artifacts_final_221006/ \
        -f param_config_dwi_ta_final_runs_2210.csv \
        -m training \
        -b 128 \
        -g 0 \
        -k 5

    inference:
    python model_execution.py -i /home/user/data/tech_artifacts/dwi_mips/pp_dwi_binary_tech_artifacts_221006/ \
        -e /home/user/development/trainings/dwi_mips/tech_artifacts_final_221006/ \
        -m inference \
        -b 128 \
        -g 0 \
        -c /home/user/development/trainings/dwi_mips/tech_artifacts_final_221006/tensorboard/densenet121_adam_1472/fold_0/checkpoints/loss/valid=0.2644-epoch=192.ckpt \
        /home/user/development/trainings/dwi_mips/tech_artifacts_final_221006/tensorboard/densenet121_adam_1472/fold_1/checkpoints/loss/valid=0.3037-epoch=157.ckpt \
        /home/user/development/trainings/dwi_mips/tech_artifacts_final_221006/tensorboard/densenet121_adam_1472/fold_2/checkpoints/loss/valid=0.3386-epoch=196.ckpt \
        /home/user/development/trainings/dwi_mips/tech_artifacts_final_221006/tensorboard/densenet121_adam_1472/fold_3/checkpoints/loss/valid=0.2459-epoch=187.ckpt \
        /home/user/development/trainings/dwi_mips/tech_artifacts_final_221006/tensorboard/densenet121_adam_1472/fold_4/checkpoints/loss/valid=0.3163-epoch=138.ckpt
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        dest="input_dir",
        type=str,
        default=None,
        help="The input directory containing the preprocessed files."
    )
    parser.add_argument(
        "-e",
        "--experiment_dir",
        dest="experiment_dir",
        type=str,
        default=None,
        help="The experiment directory containint experiment files, checkpoints, logs, etc."
    )
    parser.add_argument(
        "-f",
        "--config_csv",
        dest="config_csv",
        type=str,
        default=None,
        help="The name of the CSV file containing the configurations."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=None,
        help="The batch size."
    )
    parser.add_argument(
        "-k",
        "--kfolds",
        dest="kfolds",
        type=int,
        default=None,
        help="The number of folds for a cross validation."
    )
    parser.add_argument(
        "-g",
        "--gpu",
        dest="gpu",
        nargs="+",
        default=None,
        help="A list containing the GPU devices to use, e.g. '0' or '0 1 2'."
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        dest="checkpoint",
        nargs="+",
        default=None,
        help="Continue first learning cycle from this checkpoint path. \
            This is a list of len==1 for training and >= 1 for inference"
    )
    parser.add_argument(
        "-d",
        "--debugging",
        dest="debugging",
        default=False,
        action="store_true",
        help="A bool, to use a reduced dataset (5%) for debugging."
    )
    parser.add_argument(
        "-m",
        "--mode",
        dest="mode",
        type=str,
        default="training",
        help="A string, indicating the mode, either 'training' or 'inference'."
    )
    parser.add_argument(
        "-p",
        "--hyperparameters",
        dest="hyperparameters",
        type=str,
        default=None,
        help="The file path to the model hyperparameters ('hparams.yaml')."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.batch_size is None:
        raise Exception("Batch size argument is missing")

    # load config
    cf = Config(
        pp_data_path=args.input_dir,
        experiment_dir=args.experiment_dir,
        config_csv=args.config_csv,
        mode=args.mode
    )

    # extract dataset
    if args.mode == "training":
        # load train_info from preprocessed files
        train_info = cf.train_info_df

        # extract checkpoint, if exists
        if args.checkpoint is None:
            chk_pt_path = None
        else:
            chk_pt_path = args.checkpoint[0]

        # for debugging, use only 2.5% of the data
        if args.debugging == True:
            logging.info("Debugging mode: on")
            # dev_run = True
            train_info, _ = train_test_split(
                train_info,
                train_size=0.025,
                stratify=train_info["target_class"],
                random_state=0
            )
            dev_run = False
        else:
            # for CPU training: use only 40% of the data
            if not args.gpu:
                train_info, _ = train_test_split(
                    train_info,
                    train_size=0.4,
                    stratify=train_info["target_class"],
                    random_state=0
                )
            dev_run = False

        # count class labels
        log_class_distribution(
            dataframe=train_info,
            tag="overall",
            class_column="target_class"
        )

        # get parameter settings
        param_settings = cf.param_config()

    elif args.mode == "inference":
        inference_info = cf.inference_info_df

        # extract checkpoint, if exists
        if args.checkpoint is None:
            raise Exception("Checkpoints are missing")
        else:
            chk_pt_path = args.checkpoint

        # for debugging, use only 2.5% of the data
        if args.debugging == True:
            logging.info("Debugging mode: on")
            # dev_run = True
            inference_info, _ = train_test_split(
                inference_info,
                train_size=0.025,
                stratify=inference_info["target_class"],
                random_state=0
            )
            dev_run = False
        else:
            # for CPU training: use only 40% of the data
            if not args.gpu:
                inference_info, _ = train_test_split(
                    inference_info,
                    train_size=0.4,
                    stratify=inference_info["target_class"],
                    random_state=0
                )
            dev_run = False

        # count class labels
        log_class_distribution(
            dataframe=inference_info,
            tag="overall",
            class_column="target_class"
        )

    # gpu flag
    if args.gpu is None:
        use_gpu = None
        accelerator = None
        n_gpus = 0
    else:
        # create list with gpu devices
        use_gpu = [int(_gpu) for _gpu in args.gpu]
        # if len(use_gpu) > 2, set accelerator to "dp" (same machine)
        if len(use_gpu) > 1:
            accelerator = "ddp"
            n_gpus = len(use_gpu)
        else:
            accelerator = None
            n_gpus = 1

    if args.mode == "training":
        model_training(
            train_info=train_info,
            param_settings=param_settings,
            cf=cf,
            n_gpus=n_gpus,
            use_gpu=use_gpu,
            accelerator=accelerator,
            args=args,
            chk_pt_path=chk_pt_path,
            dev_run=dev_run
        )
    elif args.mode == "inference":
        model_inference(
            inference_info=inference_info,
            cf=cf,
            use_gpu=use_gpu,
            accelerator=accelerator,
            args=args,
            chk_pt_path=chk_pt_path
        )
