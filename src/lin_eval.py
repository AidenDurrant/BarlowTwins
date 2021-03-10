import os
from configargparse import ArgumentParser
from typing import Any
from copy import deepcopy
import math
from tqdm import tqdm
import logging
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import seed_everything
import pytorch_lightning.metrics.functional as plm
from torch.optim import Adam, SGD, LBFGS

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import network as models
from optimiser import LARSSGD, collect_params


class SSLLinearEval(pl.LightningModule):
    def __init__(self,
                 encoder,
                 num_classes,
                 model,
                 batch_size,
                 ft_learning_rate: float = 0.2,
                 ft_weight_decay: float = 1.5e-6,
                 ft_epochs: int = 1,
                 ft_optimiser: str = 'sgd',
                 effective_bsz: int = 256,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = encoder

        self.encoder.fc = Identity()
        #self.encoder.fc = models.Sup_Head(model, num_classes)

        print("\n Num Classes: {}".format(num_classes))

        self.lin_head = models.Sup_Head(model, num_classes)

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.lin_head.parameters():
            param.requires_grad = True

        self.batch_size = batch_size
        self.ft_learning_rate = ft_learning_rate
        self.ft_weight_decay = ft_weight_decay
        self.ft_optimiser = ft_optimiser
        self.ft_epochs = ft_epochs
        self.num_classes = num_classes

        self.effective_bsz = effective_bsz
        #
        print("\n\n\n effective_bsz:{} \n\n\n".format(self.effective_bsz))

        self.train_acc = []
        self.valid_acc = []
        self.test_acc = []

        self.train_t5 = []
        self.valid_t5 = []
        self.test_t5 = []

        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []

        self.criterion = torch.nn.CrossEntropyLoss().cuda()

    def on_train_epoch_start(self) -> None:
        self.encoder.eval()
        self.lin_head.train()

    def training_step(self, batch, batch_idx):
        loss, acc, t5 = self.shared_step(batch)

        self.log_dict({'train_acc': acc, 'train_loss': loss},
                      prog_bar=True, on_epoch=True)

        self.train_loss.append(loss.item())
        self.train_acc.append(acc.item())
        self.train_t5.append(t5)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, t5 = self.shared_step(batch)
        # result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        self.log_dict({'val_acc': acc, 'val_loss': loss},
                      prog_bar=True, on_epoch=True)

        self.valid_loss.append(loss.item())
        self.valid_acc.append(acc.item())
        self.valid_t5.append(t5)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc, t5 = self.shared_step(batch)

        print("Test loss: {}".format(loss))
        print("Test Acc: {}".format(acc))
        print("Test t5: {}".format(t5))

        # result = pl.EvalResult()
        self.log_dict({'test_acc': acc, 'test_loss': loss, 'test_t5': t5})

        self.test_loss.append(loss.item())
        self.test_acc.append(acc.item())
        self.test_t5.append(t5)

        return loss

    def shared_step(self, batch):
        x, y = batch

        with torch.no_grad():

            feats = self.encoder(x)

        feats = feats.view(feats.size(0), -1)

        logits = self.lin_head(feats)

        loss = self.criterion(logits, y)

        acc = plm.accuracy(logits, y, num_classes=self.num_classes)

        t5 = self.top5(logits, y)

        return loss, acc, t5

    def configure_optimizers(self):

        lr = (self.ft_learning_rate * (self.effective_bsz / 256))

        params = self.lin_head.parameters()

        print("\n OPTIM :{} \n".format(self.ft_optimiser))

        if self.ft_optimiser == 'lars':

            models = [self.lin_head]

            param_list = collect_params(models, exclude_bias_and_bn=True)

            optimizer = LARSSGD(
                param_list, lr=lr, weight_decay=self.hparams.weight_decay, eta=0.001, nesterov=False)

        elif self.ft_optimiser == 'adam':
            optimizer = Adam(params, lr=lr,
                             weight_decay=self.ft_weight_decay)
        elif self.ft_optimiser == 'sgd':
            optimizer = SGD(params, lr=lr,
                            weight_decay=self.ft_weight_decay, momentum=0.9, nesterov=False)
        elif self.ft_optimiser == 'lbfgs':
            optimizer = LBFGS(params, lr=lr)
        else:
            raise NotImplementedError('{} not setup.'.format(self.ft_optimiser))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.ft_epochs, last_epoch=-1)

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        (args, _) = parser.parse_known_args()

        # optim
        parser.add_argument('--ft_epochs', type=int, default=2)
        parser.add_argument('--ft_batch_size', type=int, default=128)
        parser.add_argument('--ft_learning_rate', type=float, default=0.02)
        parser.add_argument('--ft_weight_decay', type=float, default=1.5e-6)
        parser.add_argument('--ft_optimiser', default='sgd',
                            help='Optimiser, (Options: sgd, adam, lars).')

        return parser

    def top5(self, x, y):
        _, output_topk = x.topk(5, 1, True, True)

        acc_top5 = (output_topk == y.view(-1, 1).expand_as(output_topk)
                    ).sum().item() / y.size(0)

        return acc_top5


class Identity(torch.nn.Module):
    """
    An identity class to replace arbitrary layers in pretrained models
    Example::
        from pl_bolts.utils import Identity
        model = resnet18()
        model.fc = Identity()
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
