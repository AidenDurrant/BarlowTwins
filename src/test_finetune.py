#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import socket
import logging
import random
import warnings
import numpy as np

import neptune

from configargparse import ArgumentParser

from copy import deepcopy
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.optim import Adam

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from bt import BT
from utils import get_dm, PTPrintingCallback, FTPrintingCallback, TestNeptuneCallback
from lin_eval import SSLLinearEval


def cli_main():
    # Arguments
    default_config = os.path.join(os.path.split(os.getcwd())[0], 'config.conf')

    print(default_config)

    parser = ArgumentParser(
        description='Pytorch BT', default_config_files=[default_config])
    parser.add_argument('-c', '--my-config', required=False,
                        is_config_file=True, help='config file path')
    parser.add_argument('--finetune', dest='finetune', action='store_true',
                        help='Perform only finetuning (Default: False)')
    parser.set_defaults(finetune=False)
    parser.add_argument('--transfer', dest='transfer', action='store_true',
                        help='Perform transfer learning on linear eval (Default: False)')
    parser.set_defaults(transfer=False)
    parser.add_argument('--offline_log', dest='offline_log', action='store_true',
                        help='Do not log online (Default:  False)')
    parser.set_defaults(offline_log=False)
    parser.add_argument('--pt_checkpoint', type=str, default=None)
    parser.add_argument('--val_every_n', type=int, default=1)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=222)
    parser.add_argument('--project_name', type=str, default=None)

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = BT.add_model_specific_args(parser)

    parser = SSLLinearEval.add_model_specific_args(parser)

    args = parser.parse_args()

    seed_everything(args.seed)

    args.status = 'Test'

    args.batch_size = args.ft_batch_size

    # Get DataModule
    dm, ft_dm, args = get_dm(args)

    # Define model
    BT(**args.__dict__)

    load_log_file = save_dir = os.path.join(os.getcwd(), 'log_files.txt')

    log_dirs = np.genfromtxt(load_log_file, delimiter=" ", dtype='str')

    print("\n\n Log Dir: {}\n\n".format(log_dirs))

    ft_model_dir = log_dirs[1]
    checkpoint_path = log_dirs[2]

    if not args.offline_log:

        exp_num = log_dirs[3]

        print("Loading checkpoint: {}".format(os.path.join(ft_model_dir,
                                                           os.listdir(ft_model_dir+'/')[-1])))

        print("Experiment Num: {}".format(exp_num))

        project = neptune.init(args.project_name)

        experiment = project.get_experiments(id=exp_num)[0]

        print(experiment)

        callback_list = [TestNeptuneCallback(experiment)]

    else:
        callback_list = [TestNeptuneCallback(None)]

    encoder = BT.load_from_checkpoint(checkpoint_path, strict=False)

    SSLLinearEval(encoder.encoder_online, **args.__dict__)

    path = os.path.join(ft_model_dir, os.listdir(ft_model_dir+'/')[-1])

    ft_model = SSLLinearEval.load_from_checkpoint(
        path, strict=False, encoder=encoder.encoder_online, **args.__dict__)

    if args.accelerator == 'ddp' or args.accelerator == 'ddp2':
        replace_sampler = True  # False
    else:
        replace_sampler = True

    trainer_ft = pl.Trainer.from_argparse_args(
        args,
        logger=None,
        checkpoint_callback=False,
        callbacks=callback_list,
        deterministic=False,
        fast_dev_run=False,
        sync_batchnorm=True,
        replace_sampler_ddp=replace_sampler)

    # Fit
    trainer_ft.test(ft_model, datamodule=ft_dm)


if __name__ == '__main__':
    cli_main()
