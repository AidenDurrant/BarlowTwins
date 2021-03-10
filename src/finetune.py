#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import socket
import logging
import random
import warnings
import numpy as np

from configargparse import ArgumentParser
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from bt import BT
from utils import get_dm, PTPrintingCallback, FTPrintingCallback
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

    args.status = 'Finetune'

    args.batch_size = args.ft_batch_size

    # Get DataModule
    dm, ft_dm, args = get_dm(args)

    neptune_logger = NeptuneLogger(
        offline_mode=args.offline_log,
        api_key=None,
        project_name=args.project_name,
        experiment_name='Testing',  # Optional,
        params=vars(args),  # Optional,
        tags=["Test", args.tag],  # Optional,
        upload_source_files=['src/*.py'],
        close_after_fit=False
    )

    # Define model
    model = BT(**args.__dict__)

    load_log_file = os.path.join(os.getcwd(), 'log_files.txt')

    log_dirs = np.genfromtxt(load_log_file, delimiter=" ", dtype='str')

    print("\n\n Log Dir: {}\n\n".format(log_dirs))

    ft_model_dir = log_dirs[1]
    checkpoint_path = log_dirs[2]

    print("Loading checkpoint: {}".format(checkpoint_path))

    ft_model_checkpoint = pl.callbacks.ModelCheckpoint(
        filepath=(ft_model_dir+'/'), save_top_k=1, monitor='val_loss')

    encoder = BT.load_from_checkpoint(checkpoint_path, strict=False)

    if args.accelerator == 'ddp' or args.accelerator == 'ddp2':
        replace_sampler = True  # False
        if args.accelerator == 'ddp':
            args.effective_bsz = args.ft_batch_size * args.num_nodes * args.gpus

        elif args.accelerator == 'ddp2':
            args.effective_bsz = args.ft_batch_size * args.num_nodes

    else:
        replace_sampler = True
        args.effective_bsz = args.ft_batch_size

    ft_model = SSLLinearEval(encoder.encoder_online, **args.__dict__)

    trainer_ft = pl.Trainer.from_argparse_args(
        args, max_epochs=args.ft_epochs,
        logger=neptune_logger,
        callbacks=[FTPrintingCallback(ft_model_dir, args)],
        deterministic=True,
        checkpoint_callback=False,
        fast_dev_run=False,
        sync_batchnorm=True,
        track_grad_norm=-1,
        replace_sampler_ddp=replace_sampler,
        progress_bar_refresh_rate=args.print_freq)

    if trainer_ft.local_rank == 0:

        if not args.offline_log:

            print("Experiment: {}".format(str(trainer_ft.logger.experiment)))

            log_dirs = np.append(log_dirs, str(trainer_ft.logger.experiment).split('(')[1][:-1])

            save_log_file = os.path.join(os.getcwd(), 'log_files.txt')

            np.savetxt(save_log_file, log_dirs, delimiter=" ", fmt="%s")

    # Fit
    trainer_ft.fit(ft_model, ft_dm)

    if args.save_checkpoint:

        neptune_logger.experiment.log_artifact(os.path.join(ft_model_dir,
                                                            os.listdir(ft_model_dir+'/')[-1]), os.path.join('finetune/', os.listdir(ft_model_dir+'/')[-1]))

    neptune_logger.experiment.stop()

if __name__ == '__main__':
    cli_main()
