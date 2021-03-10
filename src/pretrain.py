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
from utils import get_dm, PTPrintingCallback, FTPrintingCallback, CheckpointSave
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

    args.status = 'Pretrain'

    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")

    save_dir = os.path.join(os.getcwd(), 'checkpoints')

    pt_model_dir = os.path.join(save_dir, ("BT_" + run_name + '/pretrain'))
    ft_model_dir = os.path.join(save_dir, ("BT_" + run_name + '/finetune'))
    reps_model_dir = os.path.join(save_dir, ("BT_" + run_name + '/reps'))

    os.makedirs(pt_model_dir, exist_ok=True)
    os.makedirs(ft_model_dir, exist_ok=True)
    os.makedirs(reps_model_dir, exist_ok=True)

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

    pt_model_checkpoint = pl.callbacks.ModelCheckpoint(
        filepath=pt_model_dir, save_top_k=1, monitor='loss')

    if args.accelerator == 'ddp' or args.accelerator == 'ddp2':
        replace_sampler = True  # False
        if args.accelerator == 'ddp':
            args.effective_bsz = args.batch_size * args.num_nodes * args.gpus

        elif args.accelerator == 'ddp2':
            args.effective_bsz = args.batch_size * args.num_nodes
    else:
        replace_sampler = True
        args.effective_bsz = args.batch_size

    # Define trainer
    trainer = pl.Trainer.from_argparse_args(
        args, max_epochs=args.max_epochs,
        logger=neptune_logger,
        callbacks=[PTPrintingCallback(pt_model_dir, args), CheckpointSave(
            pt_model_dir)],
        deterministic=True,
        fast_dev_run=False,
        sync_batchnorm=True,
        checkpoint_callback=False,
        replace_sampler_ddp=replace_sampler,
        resume_from_checkpoint=args.resume_ckpt,
        progress_bar_refresh_rate=args.print_freq,
        check_val_every_n_epoch=args.val_every_n)

    # Define model
    model = BT(**args.__dict__)

    # Fit
    trainer.fit(model, dm)

    # time.sleep(15)

    if trainer.local_rank == 0:

        print("os.listdir(pt_model_dir) :{}".format(os.listdir(pt_model_dir)))

        checkpoint_path = os.path.join(pt_model_dir,
                                       os.listdir(pt_model_dir)[-1])

        if args.save_checkpoint:

            neptune_logger.experiment.log_artifact(os.path.join(pt_model_dir,
                                                                os.listdir(pt_model_dir)[-1]), os.path.join('pretrain/', os.listdir(pt_model_dir)[-1]))

        log_files = [pt_model_dir, ft_model_dir, checkpoint_path]

        save_log_file = os.path.join(os.getcwd(), 'log_files.txt')

        np.savetxt(save_log_file, log_files, delimiter=" ", fmt="%s")

    neptune_logger.experiment.stop()


if __name__ == '__main__':
    cli_main()
