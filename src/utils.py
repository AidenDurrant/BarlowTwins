import os
import logging
import numpy as np
import math

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
from pl_bolts.datamodules import STL10DataModule

from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from dataloaders import *

import h5py
from PIL import Image, ImageFilter
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


def get_dm(args):

     # init default datamodule
    if args.dataset == 'cifar10':
        dm = CIFAR10_DataModule.from_argparse_args(args)
        dm.train_transforms = CifarTrainDataTransform(args)
        dm.val_transforms = CifarEvalDataTransform(args)
        args.num_classes = dm.num_classes

        dm_ft = CIFAR10_DataModule.from_argparse_args(args)
        dm_ft.train_transforms = CifarTrainLinTransform(args)
        dm_ft.val_transforms = CifarEvalLinTransform(args)
        dm_ft.test_transforms = CifarTestLinTransform(args)

    elif args.dataset == 'cifar100':
        dm = CIFAR100_DataModule.from_argparse_args(args)
        dm.train_transforms = CifarTrainDataTransform(args)
        dm.val_transforms = CifarEvalDataTransform(args)
        args.num_classes = dm.num_classes

        dm_ft = CIFAR100_DataModule.from_argparse_args(args)
        dm_ft.train_transforms = CifarTrainLinTransform(args)
        dm_ft.val_transforms = CifarEvalLinTransform(args)
        dm_ft.test_transforms = CifarTestLinTransform(args)

    elif args.dataset == 'imagenet':
        dm = Imagenet_DataModule.from_argparse_args(args)
        dm.train_transforms = INTrainDataTransform(args)
        dm.val_transforms = INEvalDataTransform(args)
        args.num_classes = dm.num_classes

        dm_ft = Imagenet_DataModule.from_argparse_args(args)
        dm_ft.train_transforms = INTrainLinTransform(args)
        dm_ft.val_transforms = INEvalLinTransform(args)
        dm_ft.test_transforms = INTestLinTransform(args)

    elif args.dataset == 'imagenette':

        if '-160' in args.data_dir:
            args.img_dim = 128

        else:
            args.img_dim = 224

        dm = Imagenette_DataModule.from_argparse_args(args)
        dm.train_transforms = INtteTrainDataTransform(args)
        dm.val_transforms = INtteEvalDataTransform(args)
        args.num_classes = dm.num_classes

        dm_ft = Imagenette_DataModule.from_argparse_args(args)
        dm_ft.train_transforms = INtteTrainLinTransform(args)
        dm_ft.val_transforms = INtteEvalLinTransform(args)
        dm_ft.test_transforms = INtteTestLinTransform(args)

    elif args.dataset == 'stl10':

        args.img_dim = 64

        dm = STL10_DataModule_PT.from_argparse_args(args)

        dm.train_transforms = STL10TrainDataTransform(args)
        dm.val_transforms = STL10EvalDataTransform(args)
        args.num_classes = dm.num_classes

        dm_ft = STL10_DataModule_FT.from_argparse_args(args)
        dm_ft.train_transforms = STL10TrainLinTransform(args)
        dm_ft.val_transforms = STL10EvalLinTransform(args)
        dm_ft.test_transforms = STL10TestLinTransform(args)

    # elif args.dataset == 'imagenet':
    #     dm = Imagenet_DataModule.from_argparse_args(args, image_size=196)
    #     (c, h, w) = dm.size()
    #     dm.train_transforms = ImagenetTrainDataTransform(h)
    #     dm.val_transforms = ImagenetEvalDataTransform(h)
    #     args.num_classes = dm.num_classes

    return dm, dm_ft, args


class HDF5_Dataset(Dataset):
    """ Creates a custom pytorch dataset.

        - Creates two views of the same input used for unsupervised visual
        representational learning. (SimCLR, Moco, MocoV2)

    Args:
        data (array): Array / List of datasamples

        labels (array): Array / List of labels corresponding to the datasamples

        transforms (Dictionary, optional): The torchvision transformations
            to make to the datasamples. (Default: None)

        target_transform (Dictionary, optional): The torchvision transformations
            to make to the labels. (Default: None)

        two_crop (bool, optional): Whether to perform and return two views
            of the data input. (Default: False)

    Returns:
        img (Tensor): Datasamples to feed to the model.

        labels (Tensor): Corresponding lables to the datasamples.
    """

    def __init__(self, root):

        self.root = root
        self.num_imgs = len(h5py.File(root, 'r')['labels'])

        with h5py.File(root, 'r') as f:
            self.targets = f['labels'][:]

        self.samples = np.arange(self.num_imgs)

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):

        with h5py.File(self.root, 'r') as f:
            img = f['imgs'][index]
            target = f['labels'][index]

        return img, int(target)


class CustomDatasetHDF5(Dataset):
    """ Creates a custom pytorch dataset.

        - Creates two views of the same input used for unsupervised visual
        representational learning. (SimCLR, Moco, MocoV2)

    Args:
        data (array): Array / List of datasamples

        labels (array): Array / List of labels corresponding to the datasamples

        transforms (Dictionary, optional): The torchvision transformations
            to make to the datasamples. (Default: None)

        target_transform (Dictionary, optional): The torchvision transformations
            to make to the labels. (Default: None)

        two_crop (bool, optional): Whether to perform and return two views
            of the data input. (Default: False)

    Returns:
        img (Tensor): Datasamples to feed to the model.

        labels (Tensor): Corresponding lables to the datasamples.
    """

    def __init__(self, root, data, transform=None, target_transform=None):

        # shuffle the dataset
        idx = np.random.permutation(data.shape[0])

        self.data = data[idx]

        self.transform = transform
        self.target_transform = target_transform
        self.root = root

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        hdf_idx = self.data[index]

        with h5py.File(self.root, 'r') as f:
            image = f['imgs'][index]
            target = f['labels'][index]

        image = np.transpose(image, (1, 2, 0))

        image = Image.fromarray(image)

        if self.transform is not None:

            # Data augmentation and normalisation
            img = self.transform(image)

        if self.target_transform is not None:

            # Transforms the target, i.e. object detection, segmentation
            target = self.target_transform(target)

        target = torch.LongTensor(np.asarray(target, dtype=float))

        return img, target


def class_random_split(data, labels, n_classes, n_samples_per_class):
    """
    Creates a class-balanced validation set from a training set.

    args:
        data (Array / List): Array of data values or list of paths to data.

        labels (Array, int): Array of each data samples semantic label.

        n_classes (int): Number of Classes.

        n_samples_per_class (int): Quantity of data samples to be placed
                                    per class into the validation set.

    return:
        train / valid (dict): New Train and Valid splits of the dataset.
    """

    train_x, train_y, valid_x, valid_y = [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    for i in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        train_x.extend(data[train_samples])
        train_y.extend(labels[train_samples])
        valid_x.extend(data[valid_samples])
        valid_y.extend(labels[valid_samples])

    if isinstance(data, torch.Tensor):
        # torch.stack transforms list of tensors to tensor
        return {'train': torch.stack(train_x), 'valid': torch.stack(valid_x)}, \
            {'train': torch.stack(train_y), 'valid': torch.stack(valid_y)}
    # transforms list of np arrays to tensor
    return {'train': torch.from_numpy(np.stack(train_x)),
            'valid': torch.from_numpy(np.stack(valid_x))}, \
        {'train': torch.from_numpy(np.stack(train_y)),
         'valid': torch.from_numpy(np.stack(valid_y))}


def random_split_image_folder(data, labels, n_classes, n_samples_per_class):
    """
    Creates a class-balanced validation set from a training set.

    Specifically for the image folder class.

    args:
        data (Array / List): Array of data values or list of paths to data.

        labels (Array, int): Array of each data samples semantic label.

        n_classes (int): Number of Classes.

        n_samples_per_class (int): Quantity of data samples to be placed
                                    per class into the validation set.

    return:
        train / valid (dict): New Train and Valid splits of the dataset.
    """

    train_x, train_y, valid_x, valid_y = [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    for i in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        train_x.extend(data[train_samples])
        train_y.extend(labels[train_samples])
        valid_x.extend(data[valid_samples])
        valid_y.extend(labels[valid_samples])

    # torch.from_numpy(np.stack(labels)) this takes the list of class ids and turns them to tensor.long

    return {'train': train_x, 'valid': valid_x}, \
        {'train': torch.from_numpy(np.stack(train_y)), 'valid': torch.from_numpy(np.stack(valid_y))}


class PTPrintingCallback(pl.Callback):
    def __init__(self, path, args):
        self.args = args

        if self.args.num_nodes > 1:
            self.num_gpus = self.args.num_nodes * self.args.gpus
        else:
            self.num_gpus = 1

        self.cur_loss = 1E+12
        self.path = path

    def on_init_start(self, trainer):
        print('Starting pre train print callback!')

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        loss = np.mean(pl_module.train_loss)/self.num_gpus

        epoch = trainer.current_epoch

        if self.cur_loss > loss:
            self.cur_loss = loss

            save_path = os.path.join(self.path, ('best_epoch.ckpt'))

            trainer.save_checkpoint(save_path)

        pl_module.train_loss = []

        print("\n [Train] Avg Loss: {:.4f}".format(loss))

        for param_group in trainer.optimizers[0].param_groups:
            lr = param_group['lr']

        trainer.logger.experiment.log_metric("learning_rate", lr)

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = np.mean(pl_module.valid_loss)/self.num_gpus
        epoch = trainer.current_epoch

        pl_module.valid_loss = []

        print("\n Epoch: {}".format(epoch))
        print("\n [Valid] Avg Loss: {:.4f}".format(loss))


class FTPrintingCallback(pl.Callback):
    def __init__(self, path, args):
        self.args = args
        if self.args.num_nodes > 1:
            self.num_gpus = self.args.num_nodes * self.args.gpus
        else:
            self.num_gpus = 1

        self.cur_loss = 10000000.0
        self.path = path

    def on_init_start(self, trainer):
        print('Starting finetune print callback!')

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        loss = np.mean(pl_module.train_loss)/self.num_gpus
        acc = np.mean(pl_module.train_acc)/self.num_gpus
        t5 = np.mean(pl_module.train_t5)/self.num_gpus

        pl_module.train_loss = []
        pl_module.train_acc = []
        pl_module.train_t5 = []

        print("\n [Train] Avg Loss: {:.4f}, Avg Acc: {:.4f}, Avg top5: {:.4f}".format(
            loss, acc, t5))

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = np.mean(pl_module.valid_loss)/self.num_gpus
        acc = np.mean(pl_module.valid_acc)/self.num_gpus
        t5 = np.mean(pl_module.valid_t5)/self.num_gpus

        epoch = trainer.current_epoch

        if self.cur_loss > loss:
            self.cur_loss = loss

            save_path = os.path.join(self.path, ('best_epoch.ckpt'))

            trainer.save_checkpoint(save_path)

        pl_module.valid_loss = []
        pl_module.valid_acc = []
        pl_module.valid_t5 = []

        print("\n Epoch: {}".format(epoch))
        print("\n [Valid] Avg Loss: {:.4f}, Avg Acc: {:.4f}, Avg tckptop5: {:.4f}".format(
            loss, acc, t5))

    def on_test_epoch_end(self, trainer, pl_module):
        loss = np.mean(pl_module.test_loss)/self.num_gpus
        acc = np.mean(pl_module.test_acc)/self.num_gpus
        t5 = np.mean(pl_module.test_t5)/self.num_gpus

        epoch = trainer.current_epoch

        print("\n [Test] Avg Loss: {:.4f}, Avg Acc: {:.4f}, Avg top5: {:.4f}".format(loss, acc, t5))


class TestNeptuneCallback(pl.Callback):
    def __init__(self, experiment):
        self.experiment = experiment

    def on_test_epoch_end(self, trainer, pl_module):
        loss = np.array(pl_module.test_loss).mean()
        acc = np.array(pl_module.test_acc).mean()
        t5 = np.array(pl_module.test_t5).mean()

        epoch = trainer.current_epoch

        if self.experiment is not None:
            self.experiment.log_metric('test_loss', loss)
            self.experiment.log_metric('test_acc', acc)
            self.experiment.log_metric('test_top5', t5)

        print("\n [Test] Avg Loss: {:.4f}, Avg Acc: {:.4f}, Avg top5: {:.4f}".format(loss, acc, t5))


class CheckpointSave(pl.Callback):
    def __init__(self, path):
        self.path = os.path.join(path, 'latest_checkpoint.ckpt')

    def on_init_start(self, trainer):
        print('Starting saving callback!')
        print('\n saving at: {}\n'.format(self.path))

    def on_train_epoch_end(self, trainer, pl_module, output):
        print('... Saving checkpoint! ...')

        trainer.save_checkpoint(self.path)

def sample_weights(labels):
    """ Calculates per sample weights. """
    class_sample_count = np.unique(labels, return_counts=True)[1]
    class_weights = 1. / torch.Tensor(class_sample_count)
    return class_weights[list(map(int, labels))]