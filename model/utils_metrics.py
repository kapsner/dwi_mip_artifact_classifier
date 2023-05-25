#!/usr/bin/env python

__author__ = "Lorenz A. Kapsner"
__copyright__ = "Copyright 2020-2021, Universitätsklinikum Erlangen"


import torchmetrics as metrics


class BackboneMetrics():

    def __init__(self):
        pass

    def _init_metrics(self, hparams):
        # see also https://pytorch-lightning.readthedocs.io/en/latest/metrics.html#class-vs-functional-metrics
        num_cls = None if hparams.num_classes == 2 else hparams.num_classes
        is_multiclass = False if num_cls is None else True

        # Average Precision (num_classes not necessary for binary problems)
        self.train_avgpr = metrics.classification.AveragePrecision(
            num_classes=num_cls,
            compute_on_step=False
        )
        self.valid_avgpr = metrics.classification.AveragePrecision(
            num_classes=num_cls,
            compute_on_step=False
        )
        self.test_avgpr = metrics.classification.AveragePrecision(
            num_classes=num_cls,
            compute_on_step=False
        )

        # ROC (not nessesary to provide for binary problems)
        self.train_auroc = metrics.classification.AUROC(
            num_classes=num_cls,
            compute_on_step=False
        )
        self.valid_auroc = metrics.classification.AUROC(
            num_classes=num_cls,
            compute_on_step=False
        )
        self.test_auroc = metrics.classification.AUROC(
            num_classes=num_cls,
            compute_on_step=False
        )

        # Accuracy
        self.train_acc = metrics.classification.Accuracy(
            compute_on_step=False
        )
        self.valid_acc = metrics.classification.Accuracy(
            compute_on_step=False
        )
        self.test_acc = metrics.classification.Accuracy(
            compute_on_step=False
        )

        # F1
        self.train_f1 = metrics.classification.F1(
            num_classes=num_cls,
            compute_on_step=False,
            multiclass=is_multiclass
        )
        self.valid_f1 = metrics.classification.F1(
            num_classes=num_cls,
            compute_on_step=False,
            multiclass=is_multiclass
        )
        self.test_f1 = metrics.classification.F1(
            num_classes=num_cls,
            compute_on_step=False,
            multiclass=is_multiclass
        )

        # FBeta
        self.train_fbeta = metrics.classification.FBeta(
            num_classes=num_cls,
            beta=2,
            compute_on_step=False,
            multiclass=is_multiclass
        )
        self.valid_fbeta = metrics.classification.FBeta(
            num_classes=num_cls,
            beta=2,
            compute_on_step=False,
            multiclass=is_multiclass
        )
        self.test_fbeta = metrics.classification.FBeta(
            num_classes=num_cls,
            beta=2,
            compute_on_step=False,
            multiclass=is_multiclass
        )

        # Precision (Necessary for 'macro', 'weighted' and None average methods.)
        self.train_precision = metrics.classification.Precision(
            num_classes=num_cls,
            compute_on_step=False,
            is_multiclass=is_multiclass
        )
        self.valid_precision = metrics.classification.Precision(
            num_classes=num_cls,
            compute_on_step=False,
            is_multiclass=is_multiclass
        )
        self.test_precision = metrics.classification.Precision(
            num_classes=num_cls,
            compute_on_step=False,
            is_multiclass=is_multiclass
        )

        # Recall (Necessary for 'macro', 'weighted' and None average methods.)
        self.train_recall = metrics.classification.Recall(
            num_classes=num_cls,
            compute_on_step=False,
            is_multiclass=is_multiclass
        )
        self.valid_recall = metrics.classification.Recall(
            num_classes=num_cls,
            compute_on_step=False,
            is_multiclass=is_multiclass
        )
        self.test_recall = metrics.classification.Recall(
            num_classes=num_cls,
            compute_on_step=False,
            is_multiclass=is_multiclass
        )

        # Confusion Matrix
        self.valid_confmat = metrics.classification.ConfusionMatrix(
            num_classes=hparams.num_classes,
            compute_on_step=False
        )
        self.test_confmat = metrics.classification.ConfusionMatrix(
            num_classes=hparams.num_classes,
            compute_on_step=False
        )

        self._metrics = {
            "average_precision": "avgpr",
            "auroc": "auroc",
            "accuracy": "acc",
            "f1": "f1",
            "fbeta": "fbeta",
            "precision": "precision",
            "recall": "recall"
        }
