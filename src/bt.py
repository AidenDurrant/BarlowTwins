import os
import numpy as np
from configargparse import ArgumentParser
from typing import Any
from copy import deepcopy
import math
from tqdm import tqdm
import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.optim import Adam, SGD

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import network as models

from optimiser import LARSSGD, collect_params

class BT(pl.LightningModule):
    def __init__(self,
                 num_classes,
                 learning_rate: float = 0.2,
                 weight_decay: float = 1.5e-6,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 warmup_epochs: int = 0,
                 max_epochs: int = 1,
                 o_units: int = 256,
                 h_units: int = 4096,
                 model: str = 'resnet18',
                 tau: float = 0.996,
                 optimiser: str = 'sgd',
                 effective_bsz: int = 256,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.norm_layer == 'GN':

            self.hparams.norm_l = nn.GroupNorm
        else:
            self.hparams.norm_l = nn.BatchNorm2d

        self.encoder_online = getattr(models, self.hparams.model)(
            dataset=self.hparams.dataset, norm_layer=self.hparams.norm_l)

        self.encoder_online.fc = Identity()

        self.proj_head_online = models.projection_MLP(model=self.hparams.model,
                                                      output_dim=self.hparams.o_units, hidden_dim=self.hparams.h_units, norm_layer=self.hparams.norm_l)

        # self.init_trainlog()
        self.effective_bsz = effective_bsz
        
        print("\n\n\n effective_bsz:{} \n\n\n".format(self.effective_bsz))

        self.train_loss = []
        self.valid_loss = []

        # Loss constants
        self.idnt = torch.eye(self.hparams.o_units).cuda()
        self.off_diag = torch.mul(torch.ones((self.hparams.o_units, self.hparams.o_units), dtype=bool).cuda(), self.hparams.lambd).fill_diagonal_(1)

    def init_trainlog(self):
        print("log_path: {}".format(self.hparams.log_path))
        os.makedirs(self.hparams.log_path, exist_ok=True)

        # reset root logger
        [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
        # info logger for saving command line outputs during training
        logging.basicConfig(level=logging.INFO, format='%(message)s',
                            handlers=[logging.FileHandler(os.path.join(self.hparams.log_path, 'trainlogs.txt')),
                                      logging.StreamHandler()])

    def shared_step(self, batch, batch_idx):

        (img_i, img_j), y = batch

        z_i = self.encoder_online(img_i)
        z_j = self.encoder_online(img_j)

        o_z_i = self.proj_head_online(z_i)
        o_z_j = self.proj_head_online(z_j)

        loss_1_2 = self.compute_loss(o_z_i, o_z_j)

        loss = (loss_1_2).mean()

        # Testing Printing loss functions
        # print("loss :{}".format((loss_1_2 + loss_2_1).mean()))
        # print("i_on_loss :{}".format(i_on_loss))
        # print("j_on_loss :{}".format(j_on_loss))

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict({'loss': loss}, prog_bar=True, on_epoch=True)
        self.train_loss.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict({'val_loss': loss}, prog_bar=True, on_epoch=True)

        self.valid_loss.append(loss.item())

        # return loss

    def configure_optimizers(self):

        lr = (self.hparams.learning_rate * (self.effective_bsz / 256))

        params = list(self.encoder_online.parameters()) + \
            list(self.proj_head_online.parameters())

        if self.hparams.optimiser == 'lars':

            models = [self.encoder_online, self.proj_head_online]

            param_list = collect_params(models, exclude_bias_and_bn=True)

            # print(params)

            optimizer = LARSSGD(
                param_list, lr=lr, weight_decay=self.hparams.weight_decay, eta=0.001, nesterov=False)

        elif self.hparams.optimiser == 'adam':
            optimizer = Adam(params, lr=lr,
                             weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimiser == 'sgd':
            optimizer = SGD(params, lr=lr,
                            weight_decay=self.hparams.weight_decay, momentum=0.9, nesterov=True)
        else:
            raise NotImplementedError('{} not setup.'.format(self.ft_optimiser))

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=1e-3 * lr
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        (args, _) = parser.parse_known_args()

        # Data
        parser.add_argument('--dataset', type=str, default='cifar10',
                            help='cifar10, imagenet')
        parser.add_argument('--data_dir', type=str, default=None)
        parser.add_argument('--num_workers', default=0, type=int)
        parser.add_argument('--jitter_d', type=float, default=0.5)
        parser.add_argument('--jitter_p', type=float, default=0.8)
        parser.add_argument('--blur_p', type=float, default=0.5)
        parser.add_argument('--grey_p', type=float, default=0.2)
        parser.add_argument('--blur_sigma', nargs=2, type=float, default=[0.1, 2.0])

        # optim
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--learning_rate', type=float, default=0.02)
        parser.add_argument('--weight_decay', type=float, default=1.5e-6)
        parser.add_argument('--warmup_epochs', type=float, default=10)
        parser.add_argument('--optimiser', default='sgd',
                            help='Optimiser, (Options: sgd, adam, lars).')

        # Model
        parser.add_argument('--model', default='resnet18',
                            help='Model, (Options: resnet18, resnet34, resnet50, resnet101, resnet152).')
        parser.add_argument('--h_units', type=int, default=4096)
        parser.add_argument('--o_units', type=int, default=256)
        parser.add_argument('--lambd', type=float, default=5E-3)
        parser.add_argument('--norm_layer', default=nn.BatchNorm2d)

        parser.add_argument('--save_checkpoint', dest='save_checkpoint', action='store_true',
                            help='Save the checkpoints to Neptune (Default: False)')
        parser.set_defaults(save_checkpoint=False)
        parser.add_argument('--print_freq', type=int, default=1)

        return parser

    def compute_loss(self, x, y):
        
        N = x.size(0)

        x = (x-x.mean(0))/ x.std(0)
        y = (y-y.mean(0))/ y.std(0)

        c = torch.mm(x.T, y) / N

        c_diff = (c - self.idnt).pow(2)

        c_diff_w = torch.mul(c_diff, self.off_diag)

        return c_diff_w.sum()


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
