#!/usr/bin/env python

__author__ = "Lorenz A. Kapsner"
__copyright__ = "Copyright 2020-2021, Universitätsklinikum Erlangen"


from monai import transforms, visualize, networks
import os

import itertools

import pytorch_lightning as pl
import pandas as pd

import torch
from torch import optim

from matplotlib import pyplot as plt

from skimage import color
from skimage.io import imsave

import numpy as np

from utils_metrics import BackboneMetrics


class BackboneModel(pl.LightningModule, BackboneMetrics):
    # https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    def __init__(self, inference_mode=False, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        # is multiple GPU?
        self.multiple_gpus = True if self.hparams.n_gpus > 1 else False

        # len dataloader
        self.len_traindl = None

        # load model architecture:
        self.model = self._model_selection()

        # instantiate metrics (would be best pratice for distributed learning)
        self._init_metrics(
            hparams=self.hparams
        )

        # created inference dataframe
        self.inference_results = pd.DataFrame(
            columns=[
                "id",
                "y_pred_prob_positive",
                "y_pred_class",
                "y_true"
            ]
        )

        # instantiate monai transforms to log misclassifies test images
        self.img_resize = transforms.Compose([
            transforms.Resize(spatial_size=[256, 256])
        ])
        self.img_rescale = transforms.Compose([
            transforms.ScaleIntensity(minv=0, maxv=1)
        ])

        # transformation to save jpeg files
        self.img_jpeg_prep = transforms.Compose([
            transforms.ScaleIntensity(minv=0, maxv=255),
            transforms.CastToType(dtype=np.uint8)
        ])

        # set test_img num
        self.test_img_num = 0

        # set test iteration
        self.test_iteration = 0

        # instantiate class activation map function
        target_layer = "class_layers.relu"
        fc_layer = "class_layers.out"

        self.gradcampp = visualize.GradCAMpp(
            nn_module=self.model,
            target_layers=target_layer
        )

    def _model_selection(self):
        # https://docs.monai.io/en/latest/networks.html#monai.networks.nets.densenet121
        network = networks.nets.densenet.densenet121(
            pretrained=self.hparams.pretrained,
            spatial_dims=2,
            in_channels=1,
            out_channels=self.hparams.num_classes,
            bn_size=self.hparams.bn_size,
            dropout_prob=self.hparams.dropout_prob
        )

        return network

    def _optimizer_selection(self):
        if self.hparams.optim == "sgd":
            # https://pytorch.org/docs/stable/optim.html#torch.optim.SGD
            optimizer = optim.SGD(
                params=self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                nesterov=self.hparams.nesterov,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optim == "adam":
            # https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
            optimizer = optim.Adam(
                params=self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        return optimizer

    def _scheduler_selection(self, optimizer):

        if self.hparams.scheduler in ["lambdalr", "warmup_lin", "warmup_exp"]:
            # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.LambdaLR
            if self.hparams.scheduler == "lambdalr":
                def lmbda(epoch): return self.hparams.gamma ** epoch
                _lr_step = "epoch"
            else:
                lmbda = self._lr_warmup
                _lr_step = "step"

            scheduler = {
                "scheduler": optim.lr_scheduler.LambdaLR(
                    optimizer=optimizer,
                    lr_lambda=lmbda
                ),
                "interval": _lr_step,
                "frequency": 1
            }

        return scheduler

    def _lr_warmup(self, epoch):
        if self.len_traindl is None:
            self.len_traindl = len(self.train_dataloader())

        if self.trainer.current_epoch < self.hparams.warm_up_steps:
            # warm up lr
            if self.hparams.scheduler == "warmup_lin":
                lr_scale = min(
                    1.,
                    float(self.trainer.global_step + 1) /
                    float(self.hparams.warm_up_steps * self.len_traindl)
                )
            elif self.hparams.scheduler == "warmup_exp":
                lr_scale = self.hparams.learning_rate ** (
                    (self.hparams.warm_up_steps * self.len_traindl) -
                    self.trainer.global_step
                )
        else:
            # update lr only on beginning of new epoch
            # (global_step has offset = -1 --> global_step == len(dataloader),
            # is an update of lr on first batch of new epoch. Otherwise, lr_scale remains = 1
            if float(self.trainer.global_step + 1) % float(len(self.train_dataloader())) == 0:
                lr_scale = self.hparams.gamma ** self.trainer.current_epoch
            else:
                lr_scale = 1.

        return lr_scale

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):

        optimizer = self._optimizer_selection()

        if self.hparams.scheduler != "None":
            scheduler = self._scheduler_selection(optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def training_step_end(self, batch_parts):
        step_results = self._shared_step_end(batch_parts, "train")
        self.log(
            name="train_loss",
            value=step_results["loss"],
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.multiple_gpus
        )
        return step_results

    def training_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "valid")

    def validation_step_end(self, batch_parts):
        step_results = self._shared_step_end(batch_parts, "valid")
        self.log(
            name="valid_loss",
            value=step_results["loss"],
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.multiple_gpus
        )
        return step_results

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def test_step_end(self, batch_parts):
        return self._shared_step_end(batch_parts, "test")

    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "test")

    def _shared_step(self, batch, batch_idx, prefix):
        x = batch["image"]
        y = batch["target"]

        logits = self(x)

        bs = torch.tensor([len(x)], dtype=torch.int16).type_as(x)

        if prefix == "test":
            # log misclassified images
            # https://docs.neptune.ai/integrations/pytorch_lightning.html#log-misclassified-images-for-the-test-set

            # prepare targets
            y_preds, y_trues = self._y_to_cat(
                logits=logits,
                targets=y
            )

            # append inference data
            append_row = pd.DataFrame(
                data={
                    "id": batch["id"],
                    "y_pred_prob_positive": list(
                        torch.softmax(logits, dim=1).cpu(
                        ).detach().numpy()[:, 1]
                    ),
                    "y_pred_class": list(
                        y_preds.cpu().detach().numpy()
                    ),
                    "y_true": list(
                        y_trues.cpu().detach().numpy()
                    )
                }
            )
            self.inference_results = self.inference_results.append(
                other=append_row,
                ignore_index=True
            )

            self._log_images(
                data=x,
                y_preds=y_preds,
                y_trues=y_trues,
                ids=batch["id"]
            )

        return {"target": y, "logits": logits, "batch_size": bs}

    def _shared_epoch_end(self, outputs, prefix):
        # concat batch sizes
        batch_sizes = torch.stack(
            [x["batch_size"] for x in outputs]
        ).type_as(outputs[0]["loss"])

        # concat losses
        losses = torch.stack(
            [x["loss"] for x in outputs]
        ).type_as(outputs[0]["loss"])

        # calculating weighted mean loss
        avg_loss = torch.sum(losses * batch_sizes) / torch.sum(batch_sizes)

        log_prefix = prefix if prefix != "test" else prefix + "/" + \
            str(self.test_iteration)

        self.log(
            name="loss/" + log_prefix,
            value=avg_loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.multiple_gpus
        )

        # compute metrics
        for _metname, _metshort in self._metrics.items():
            self.log(
                name=_metname + "/" + log_prefix,
                value=eval("self." + prefix + "_" +
                           _metshort + ".compute()"),
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=self.multiple_gpus
            )
            eval("self." + prefix + "_" +
                 _metshort + ".reset()")

        # # confmat https://pytorch-lightning.readthedocs.io/en/latest/metrics.html#confusionmatrix
        # # only validation + test; plot as figure
        if prefix in ["valid", "test"]:
            # (metrics class interface)
            cm = eval("self." + prefix + "_confmat" + ".compute()")
            # from 1.2 onward:
            # From v1.2 onward compute() will no longer automatically call reset(), and it is up to the user to reset metrics between epochs,
            # except in the case where the metric is directly passed to LightningModule`s self.log.
            eval("self." + prefix + "_confmat" + ".reset()")
            figure = self.plot_confusion_matrix(
                cm=cm,
                class_names=[i for i in range(0, self.hparams.num_classes)]
            )

            if prefix == "valid":
                cur_step = self.current_epoch
            elif prefix == "test":
                cur_step = self.test_iteration
                self.test_iteration += 1

            # assuming, tensorboard is first logger in list
            self.logger[0].experiment.add_figure(
                tag="Confusion Matrix: " + prefix,
                figure=figure,
                global_step=cur_step
            )

            # save inference results as csv
            self.inference_results.to_csv(
                path_or_buf=os.path.join(
                    self.trainer.default_root_dir,
                    "logs",
                    "tensorboard",
                    self.logger.version,
                    "inference_results.csv"
                ),
                index=False
            )

    def _log_images(self, data, y_preds, y_trues, ids):

        y_preds = y_preds.cpu().detach().numpy()
        y_trues = y_trues.cpu().detach().numpy()

        for raw_image, _y_pred, _y_true, _id in zip(data, y_preds, y_trues, ids):

            if _y_pred == 1 and _y_true == 1:
                performance_indicator = "TP"
            elif _y_pred == 1 and _y_true == 0:
                performance_indicator = "FP"
            elif _y_pred == 0 and _y_true == 0:
                performance_indicator = "TN"
            elif _y_pred == 0 and _y_true == 1:
                performance_indicator = "FN"

            # create output-text
            tag = "{} (test iteration {}) / {} - ID {}".format(
                performance_indicator,
                self.test_iteration,
                self.test_img_num,
                _id
            )

            cm_images = {}

            # transform class activation map
            # cam needs 4-dimensional array
            # cam_img = self.cam(x=raw_image[None], class_idx=None)

            with torch.set_grad_enabled(True):

                # GradCAM++
                # without class_idx
                cm_images["None"] = self.gradcampp(
                    x=raw_image[None],
                    class_idx=None
                )

                # with class_idx
                for _cls in range(self.hparams.num_classes):
                    cm_images[_cls] = self.gradcampp(
                        x=raw_image[None],
                        class_idx=_cls
                    )

            # dict of images to plot later
            img2plot = {}

            # get colormap
            cm = plt.get_cmap("jet_r")

            # iterate over images that require color map
            for _img_key, _img_val in cm_images.items():
                # apply colormap
                _img = cm(np.squeeze(_img_val))
                # create 3-channel image for displaying in tensorboard
                img2plot[_img_key] = np.transpose(
                    color.rgba2rgb(_img),
                    (2, 0, 1)
                )

            # process original image
            # add axis, required for transformations (c, h, w)
            img = np.squeeze(raw_image.cpu().detach().numpy())[None]
            # rescale original image
            img = self.img_rescale(img)  # intensity=0-1, c, h, w
            # create 3-channel grayscale image for displaying in tensorboard
            img2plot["original_image"] = np.transpose(
                color.gray2rgb(np.squeeze(img)),
                (2, 0, 1)
            )

            img_keys = ["None"] + \
                [i for i in range(self.hparams.num_classes)] + \
                ["original_image"]
            for _index, _img in enumerate(img_keys):
                _img2tb = self.img_resize(img2plot[_img])  # c, h, w
                _step = _index + 1
                visualize.img2tensorboard.plot_2d_or_3d_image(
                    data=_img2tb[None],
                    step=_step,
                    writer=self.logger[0].experiment,
                    max_channels=3,
                    max_frames=1,
                    tag=tag
                )

                if self.hparams.inference_mode:
                    # tag our cam images
                    if _img != "original_image":
                        suffix = "_cam"
                    else:
                        suffix = ""

                    # create output dir
                    test_img_basedir = os.path.join(
                        self.trainer.default_root_dir,
                        "logs",
                        "tensorboard",
                        self.logger.version,
                        "test_images",
                        performance_indicator
                    )

                    if not os.path.exists(test_img_basedir):
                        os.makedirs(test_img_basedir)

                    # build filename
                    _fname = os.path.join(
                        test_img_basedir,
                        _id + suffix + "_" + str(_img) + ".jpeg"
                    )

                    # convert image before saving
                    _img2tb = self.img_jpeg_prep(_img2tb)  # c, h, w

                    # save image
                    imsave(
                        fname=_fname,
                        arr=np.transpose(  # h, w, c
                            _img2tb,
                            (1, 2, 0)
                        )
                    )

            # increment image number
            self.test_img_num += 1

    @ staticmethod
    def plot_confusion_matrix(cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
            cm (array, shape = [n, n]): a confusion matrix of integer classes
            class_names (array, shape = [n]): String names of the integer classes
        """
        cm = cm.cpu().detach().numpy()

        figure = plt.figure(figsize=(5, 5))
        plt.imshow(cm, interpolation='nearest', cmap="Blues")
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Compute the labels from the normalized confusion matrix.
        # (relative proportion of predicted labels per true class)
        # labels = np.around(cm.astype('float') / cm.sum(axis=1)
        #                    [:, np.newaxis], decimals=2)
        labels = cm.astype(np.uint16)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j],
                     horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

    @ staticmethod
    def _y_to_cat(logits, targets):
        """
        This method must be defined by the user!
        """
        raise Exception("Please customize the method '_y_to_cat' \
            to fit your data.")
