import os
import numpy as np
from PIL import Image, ImageFilter
import random

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, STL10

import utils

__all__ = ['CIFAR10_DataModule', 'CIFAR100_DataModule', 'CifarTrainDataTransform', 'CifarEvalDataTransform',
           'CifarTrainLinTransform', 'CifarEvalLinTransform', 'CifarTestLinTransform',
           'Imagenet_DataModule', 'INTrainDataTransform', 'INEvalDataTransform',
           'INTrainLinTransform', 'INEvalLinTransform', 'INTestLinTransform',
           'Imagenette_DataModule', 'INtteTrainDataTransform', 'INtteEvalDataTransform',
           'INtteTrainLinTransform', 'INtteEvalLinTransform', 'INtteTestLinTransform',
           'STL10_DataModule_PT', 'STL10_DataModule_FT', 'STL10TrainDataTransform', 'STL10EvalDataTransform',
           'STL10TrainLinTransform', 'STL10EvalLinTransform', 'STL10TestLinTransform',
           ]


class CIFAR10_DataModule(LightningDataModule):
    name = 'cifar10'
    extra_args = {}

    def __init__(
            self,
            data_dir: str = None,
            val_split: int = 5000,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            accelerator: str = 'ddp',
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.dims = (3, 32, 32)
        self.DATASET = CIFAR10
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_samples = 60000 - val_split
        self.accelerator = accelerator

        print("\n\n batch_size in dataloader:{}".format(self.batch_size))

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        self.DATASET(self.data_dir, train=True, download=True,
                     transform=transforms.ToTensor(), **self.extra_args)
        self.DATASET(self.data_dir, train=False, download=True,
                     transform=transforms.ToTensor(), **self.extra_args)

    def train_dataloader(self):

        transf = self.default_transforms() if self.train_transforms is None else self.train_transforms

        dataset = self.DATASET(self.data_dir, train=True, download=False,
                               transform=transf, **self.extra_args)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        s_weights = utils.sample_weights(np.asarray(
            dataset_train.dataset.targets)[dataset_train.indices])

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset_train)
            shuffle = False
        else:
            sampler = WeightedRandomSampler(s_weights,
                                            num_samples=len(s_weights), replacement=True)
            shuffle = False

        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):

        transf = self.default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = self.DATASET(self.data_dir, train=True, download=False,
                               transform=transf, **self.extra_args)
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset_val)
        else:
            sampler = None

        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=sampler
        )
        return loader

    def test_dataloader(self):

        transf = self.default_transforms() if self.test_transforms is None else self.test_transforms

        dataset = self.DATASET(self.data_dir, train=False, download=False,
                               transform=transf, **self.extra_args)

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset)
        else:
            sampler = None

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=sampler
        )
        return loader

    def default_transforms(self):
        cf10_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
            #                      std=[0.24703223, 0.24348513, 0.26158784])
        ])
        return cf10_transforms


class CIFAR100_DataModule(LightningDataModule):
    name = 'cifar100'
    extra_args = {}

    def __init__(
            self,
            data_dir: str = None,
            val_split: int = 5000,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            accelerator: str = 'ddp',
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.dims = (3, 32, 32)
        self.DATASET = CIFAR100
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_samples = 60000 - val_split
        self.accelerator = accelerator

        print("\n\n batch_size in dataloader:{}".format(self.batch_size))

    @property
    def num_classes(self):
        return 100

    def prepare_data(self):
        self.DATASET(self.data_dir, train=True, download=True,
                     transform=transforms.ToTensor(), **self.extra_args)
        self.DATASET(self.data_dir, train=False, download=True,
                     transform=transforms.ToTensor(), **self.extra_args)

    def train_dataloader(self):

        transf = self.default_transforms() if self.train_transforms is None else self.train_transforms

        dataset = self.DATASET(self.data_dir, train=True, download=False,
                               transform=transf, **self.extra_args)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        s_weights = utils.sample_weights(np.asarray(
            dataset_train.dataset.targets)[dataset_train.indices])

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset_train)
            shuffle = False
        else:
            sampler = WeightedRandomSampler(s_weights,
                                            num_samples=len(s_weights), replacement=True)
            shuffle = False

        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):

        transf = self.default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = self.DATASET(self.data_dir, train=True, download=False,
                               transform=transf, **self.extra_args)
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset_val)
        else:
            sampler = None

        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=sampler
        )
        return loader

    def test_dataloader(self):

        transf = self.default_transforms() if self.test_transforms is None else self.test_transforms

        dataset = self.DATASET(self.data_dir, train=False, download=False,
                               transform=transf, **self.extra_args)

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset)
        else:
            sampler = None

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=sampler
        )
        return loader

    def default_transforms(self):
        cf10_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                 std=[0.24703223, 0.24348513, 0.26158784])
        ])
        return cf10_transforms


class CifarTrainDataTransform(object):
    def __init__(self, args):

        color_jitter = transforms.ColorJitter(
            0.8*args.jitter_d, 0.8*args.jitter_d, 0.4*args.jitter_d, 0.2*args.jitter_d)
        data_transforms = transforms.Compose([  # transforms.ToPILImage(),
            transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
            transforms.RandomApply([color_jitter], p=args.jitter_p),
            transforms.RandomGrayscale(p=args.grey_p),
            # transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                 std=[0.24703223, 0.24348513, 0.26158784])
        ])
        self.train_transform = data_transforms

    def __call__(self, sample):
        transform = self.train_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class CifarEvalDataTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([  # transforms.ToPILImage(),
            transforms.CenterCrop((32 * 0.875, 32 * 0.875)),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                 std=[0.24703223, 0.24348513, 0.26158784])
        ])
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class CifarTrainLinTransform(object):
    def __init__(self, args):

        data_transforms = transforms.Compose([  # transforms.ToPILImage(),
            transforms.RandomResizedCrop((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                 std=[0.24703223, 0.24348513, 0.26158784])
        ])
        self.train_transform = data_transforms

    def __call__(self, sample):
        transform = self.train_transform
        x = transform(sample)
        return x


class CifarEvalLinTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([  # transforms.ToPILImage(),
            transforms.CenterCrop((32 * 0.875, 32 * 0.875)),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                 std=[0.24703223, 0.24348513, 0.26158784])
        ])
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        x = transform(sample)
        return x


class CifarTestLinTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([
            transforms.CenterCrop((32 * 0.875, 32 * 0.875)),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                 std=[0.24703223, 0.24348513, 0.26158784])
        ])
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        x = transform(sample)
        return x


class Imagenet_DataModule(LightningDataModule):
    name = 'imagenet'
    extra_args = {}

    def __init__(
            self,
            data_dir: str = None,
            val_split: int = 50000,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            accelerator: str = 'ddp',
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.dims = (3, 224, 224)
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.accelerator = accelerator

    @property
    def num_classes(self):
        return 1000

    def prepare_data(self):
        self.train_imagenet = utils.HDF5_Dataset(
            root=os.path.join(self.data_dir, 'trainILSVRC256.hdf5'))
        self.test_imagenet = utils.HDF5_Dataset(
            root=os.path.join(self.data_dir, 'valILSVRC256.hdf5'))

    def train_dataloader(self):

        self.train_imagenet = utils.HDF5_Dataset(
            root=os.path.join(self.data_dir, 'trainILSVRC256.hdf5'))

        self.data, self.labels = utils.random_split_image_folder(data=self.train_imagenet.samples,
                                                                 labels=self.train_imagenet.targets,
                                                                 n_classes=1000,
                                                                 n_samples_per_class=np.repeat(10, 1000).reshape(-1))

        transf = self.default_transforms() if self.train_transforms is None else self.train_transforms

        dataset_train = utils.CustomDatasetHDF5(root=os.path.join(self.data_dir, 'trainILSVRC256.hdf5'), data=np.asarray(
            self.train_imagenet.samples), transform=transf)  # self.data['train']), transform=transf)

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset_train)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=sampler
        )
        return loader

    def val_dataloader(self):

        self.train_imagenet = utils.HDF5_Dataset(
            root=os.path.join(self.data_dir, 'trainILSVRC256.hdf5'))

        self.data, self.labels = utils.random_split_image_folder(data=self.train_imagenet.samples,
                                                                 labels=self.train_imagenet.targets,
                                                                 n_classes=1000,
                                                                 n_samples_per_class=np.repeat(10, 1000).reshape(-1))

        transf = self.default_transforms() if self.val_transforms is None else self.val_transforms

        dataset_valid = utils.CustomDatasetHDF5(root=os.path.join(self.data_dir, 'trainILSVRC256.hdf5'), data=np.asarray(
            self.data['valid']), transform=transf)

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset_valid)
        else:
            sampler = None

        loader = DataLoader(
            dataset_valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=sampler
        )
        return loader

    def test_dataloader(self):

        self.test_imagenet = utils.HDF5_Dataset(
            root=os.path.join(self.data_dir, 'valILSVRC256.hdf5'))

        transf = self.default_transforms() if self.test_transforms is None else self.test_transforms

        dataset_test = utils.CustomDatasetHDF5(root=os.path.join(
            self.data_dir, 'valILSVRC256.hdf5'), data=self.test_imagenet.samples, transform=transf)

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset_test)
        else:
            sampler = None

        loader = DataLoader(
            dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=sampler
        )
        return loader

    def default_transforms(self):
        cf10_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return cf10_transforms


class INTrainDataTransform(object):
    def __init__(self, args):

        color_jitter = transforms.ColorJitter(
            0.8*args.jitter_d, 0.8*args.jitter_d, 0.4*args.jitter_d, 0.2*args.jitter_d)
        data_transforms = transforms.Compose([  # transforms.ToPILImage(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.RandomApply([color_jitter], p=args.jitter_p),
            transforms.RandomGrayscale(p=args.grey_p),
            transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.train_transform = data_transforms

    def __call__(self, sample):
        transform = self.train_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class INEvalDataTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([  # transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class INTrainLinTransform(object):
    def __init__(self, args):

        data_transforms = transforms.Compose([  # transforms.ToPILImage(),
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.train_transform = data_transforms

    def __call__(self, sample):
        transform = self.train_transform
        x = transform(sample)
        return x


class INEvalLinTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([  # transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        x = transform(sample)
        return x


class INTestLinTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        x = transform(sample)
        return x


class Imagenette_DataModule(LightningDataModule):
    name = 'imagenette'
    extra_args = {}

    '''
    https://github.com/fastai/imagenette
    '''

    def __init__(
            self,
            data_dir: str = None,
            val_split: int = 1000,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            accelerator: str = 'ddp',
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.dims = (3, 224, 224)
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.accelerator = accelerator

    @property
    def num_classes(self):
        return 10

    def train_dataloader(self):

        transf = self.default_transforms() if self.train_transforms is None else self.train_transforms

        self.train_imagenet = ImageFolder(
            root=os.path.join(self.data_dir, 'train'), transform=transf)

        dataset_train = self.train_imagenet

        # dataset_train, _ = torch.utils.data.random_split(self.train_imagenet, [8469, 1000])

        # self.data, self.labels = utils.random_split_image_folder(data=self.train_imagenet.samples,
        #                                                          labels=self.train_imagenet.targets,
        #                                                          n_classes=10,
        #                                                          n_samples_per_class=np.repeat(100, 10).reshape(-1))

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset_train)
            shuffle = False
        else:
            sampler = None
            shuffle = False

        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):

        transf = self.default_transforms() if self.val_transforms is None else self.val_transforms

        self.train_imagenet = ImageFolder(
            root=os.path.join(self.data_dir, 'val'), transform=transf)

        dataset_val = self.train_imagenet

        # _, dataset_val = torch.utils.data.random_split(self.train_imagenet, [8469, 1000])

        # self.data, self.labels = utils.random_split_image_folder(data=self.train_imagenet.samples,
        #                                                          labels=self.train_imagenet.targets,
        #                                                          n_classes=10,
        #                                                          n_samples_per_class=np.repeat(100, 10).reshape(-1))

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset_train)
            shuffle = False
        else:
            sampler = None
            shuffle = False

        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def test_dataloader(self):

        transf = self.default_transforms() if self.test_transforms is None else self.test_transforms

        self.test_imagenet = ImageFolder(
            root=os.path.join(self.data_dir, 'val'), transform=transf)

        # self.data, self.labels = utils.random_split_image_folder(data=self.train_imagenet.samples,
        #                                                          labels=self.train_imagenet.targets,
        #                                                          n_classes=10,
        #                                                          n_samples_per_class=np.repeat(100, 10).reshape(-1))

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset_train)
            shuffle = False
        else:
            sampler = None
            shuffle = False

        loader = DataLoader(
            self.test_imagenet,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def default_transforms(self):
        cf10_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return cf10_transforms


class INtteTrainDataTransform(object):
    def __init__(self, args):

        color_jitter = transforms.ColorJitter(
            0.8*args.jitter_d, 0.8*args.jitter_d, 0.4*args.jitter_d, 0.2*args.jitter_d)
        data_transforms = transforms.Compose([  # transforms.ToPILImage(),
            transforms.RandomResizedCrop((args.img_dim, args.img_dim), scale=(0.08, 1.0)),
            transforms.RandomApply([color_jitter], p=args.jitter_p),
            transforms.RandomGrayscale(p=args.grey_p),
            transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.train_transform = data_transforms

    def __call__(self, sample):
        transform = self.train_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class INtteEvalDataTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([  # transforms.ToPILImage(),
            transforms.Resize((int(args.img_dim//0.8), int(args.img_dim//0.8))),
            transforms.CenterCrop((args.img_dim, args.img_dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class INtteTrainLinTransform(object):
    def __init__(self, args):

        data_transforms = transforms.Compose([  # transforms.ToPILImage(),
            transforms.RandomResizedCrop((args.img_dim, args.img_dim)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.train_transform = data_transforms

    def __call__(self, sample):
        transform = self.train_transform
        x = transform(sample)
        return x


class INtteEvalLinTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([  # transforms.ToPILImage(),
            transforms.Resize((int(args.img_dim//0.8), int(args.img_dim//0.8))),
            transforms.CenterCrop((args.img_dim, args.img_dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        x = transform(sample)
        return x


class INtteTestLinTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([
            transforms.Resize((int(args.img_dim//0.8), int(args.img_dim//0.8))),
            transforms.CenterCrop((args.img_dim, args.img_dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        x = transform(sample)
        return x


class STL10_DataModule_PT(LightningDataModule):
    name = 'stl10'
    extra_args = {}

    def __init__(
            self,
            data_dir: str = None,
            val_split: int = 5000,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            accelerator: str = 'ddp',
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.dims = (3, 32, 32)
        self.DATASET = STL10
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_samples = 100000 - val_split
        self.accelerator = accelerator

        print("\n\n batch_size in dataloader:{}".format(self.batch_size))

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):

        self.DATASET(self.data_dir, split='unlabeled', download=True,
                     transform=transforms.ToTensor(), **self.extra_args)
        self.DATASET(self.data_dir, split='train', download=True,
                     transform=transforms.ToTensor(), **self.extra_args)
        self.DATASET(self.data_dir, split='test', download=True,
                     transform=transforms.ToTensor(), **self.extra_args)

    def train_dataloader(self):

        transf = self._default_transforms() if self.train_transforms is None else self.train_transforms

        dataset = self.DATASET(self.data_dir, split='unlabeled', download=False,
                               transform=transf, **self.extra_args)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset, [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset_train)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):

        transf = self.default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = self.DATASET(self.data_dir, split='unlabeled', download=False,
                               transform=transf, **self.extra_args)
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset_val)
        else:
            sampler = None

        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=sampler
        )
        return loader

    def test_dataloader(self):

        transf = self.default_transforms() if self.test_transforms is None else self.test_transforms

        dataset = self.DATASET(self.data_dir, split='test', download=False,
                               transform=transf, **self.extra_args)

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset)
        else:
            sampler = None

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=sampler
        )
        return loader

    def default_transforms(self):
        stl10_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43, 0.42, 0.39],
                                 std=[0.27, 0.26, 0.27])
        ])
        return stl10_transforms


class STL10_DataModule_FT(LightningDataModule):
    name = 'stl10'
    extra_args = {}

    def __init__(
            self,
            data_dir: str = None,
            val_split: int = 500,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            accelerator: str = 'ddp',
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.dims = (3, 32, 32)
        self.DATASET = STL10
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_samples = 100000 - val_split
        self.accelerator = accelerator

        print("\n\n batch_size in dataloader:{}".format(self.batch_size))

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):

        self.DATASET(self.data_dir, split='unlabeled', download=True,
                     transform=transforms.ToTensor(), **self.extra_args)
        self.DATASET(self.data_dir, split='train', download=True,
                     transform=transforms.ToTensor(), **self.extra_args)
        self.DATASET(self.data_dir, split='test', download=True,
                     transform=transforms.ToTensor(), **self.extra_args)

    def train_dataloader(self):

        transf = self._default_transforms() if self.train_transforms is None else self.train_transforms

        dataset = self.DATASET(self.data_dir, split='train', download=False,
                               transform=transf, **self.extra_args)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset, [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        s_weights = utils.sample_weights(np.asarray(
            dataset_train.dataset.labels)[dataset_train.indices])

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset_train)
            shuffle = False
        else:
            sampler = WeightedRandomSampler(s_weights,
                                            num_samples=len(s_weights), replacement=True)
            shuffle = False

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):

        transf = self.default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = self.DATASET(self.data_dir, split='train', download=False,
                               transform=transf, **self.extra_args)
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset_val)
        else:
            sampler = None

        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=sampler
        )
        return loader

    def test_dataloader(self):

        transf = self.default_transforms() if self.test_transforms is None else self.test_transforms

        dataset = self.DATASET(self.data_dir, split='test', download=False,
                               transform=transf, **self.extra_args)

        if self.accelerator == 'ddp' or self.accelerator == 'ddp2':
            sampler = None  # DistributedSampler(dataset)
        else:
            sampler = None

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=sampler
        )
        return loader

    def default_transforms(self):
        stl10_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43, 0.42, 0.39],
                                 std=[0.27, 0.26, 0.27])
        ])
        return stl10_transforms


class STL10TrainDataTransform(object):
    def __init__(self, args):

        color_jitter = transforms.ColorJitter(
            0.8*args.jitter_d, 0.8*args.jitter_d, 0.4*args.jitter_d, 0.2*args.jitter_d)
        data_transforms = transforms.Compose([  # transforms.ToPILImage(),
            transforms.RandomResizedCrop((64, 64), scale=(0.2, 1.0)),
            transforms.RandomApply([color_jitter], p=args.jitter_p),
            transforms.RandomGrayscale(p=args.grey_p),
            # transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43, 0.42, 0.39],
                                 std=[0.27, 0.26, 0.27])
        ])
        self.train_transform = data_transforms

    def __call__(self, sample):
        transform = self.train_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class STL10EvalDataTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([  # transforms.ToPILImage(),
            transforms.CenterCrop((96 * 0.875, 96 * 0.875)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43, 0.42, 0.39],
                                 std=[0.27, 0.26, 0.27])
        ])
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class STL10TrainLinTransform(object):
    def __init__(self, args):

        data_transforms = transforms.Compose([  # transforms.ToPILImage(),
            transforms.RandomResizedCrop((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43, 0.42, 0.39],
                                 std=[0.27, 0.26, 0.27])
        ])
        self.train_transform = data_transforms

    def __call__(self, sample):
        transform = self.train_transform
        x = transform(sample)
        return x


class STL10EvalLinTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([  # transforms.ToPILImage(),
            transforms.CenterCrop((96 * 0.875, 96 * 0.875)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43, 0.42, 0.39],
                                 std=[0.27, 0.26, 0.27])
        ])
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        x = transform(sample)
        return x


class STL10TestLinTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([
            transforms.CenterCrop((96 * 0.875, 96 * 0.875)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43, 0.42, 0.39],
                                 std=[0.27, 0.26, 0.27])
        ])
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        x = transform(sample)
        return x


class GaussianBlur(object):
    """
    Gaussian blur augmentation:
        https://github.com/facebookresearch/moco/
    """

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
