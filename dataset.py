import os
import pickle
from abc import ABC

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from omegaconf import DictConfig
from torch.utils.data import Dataset, Subset
from torchvision import transforms, datasets
from torchvision.transforms import transforms, Compose

from data_aug.style_transforms import test_transform
from exceptions.exceptions import InvalidDatasetSelection


class StyleCLRPLDataset(pl.LightningDataModule, ABC):
    def __init__(self, cfg: DictConfig) -> None:
        """
        Main PL dataset module. Prepares data and dataloaders.
        See PL documentation.
        :param cfg: Hydra format dataset configuration. [omegaconf.dictconfig.DictConfig]
        """
        self.cfg = cfg
        self.train_dataset = None
        self.stylized_dataset = None

        self.style_transform = None
        super().__init__()

    def setup(self, **kwargs):
        self.content_dataset = ContentImageDataset(self.cfg).get_dataset_for_simclr()
        self.stylized_dataset = StylizedDatasetOnGPU(content_dataset=self.content_dataset,
                                                     style_data_path=self.cfg.dataset.style.data_path,
                                                     style_pickle_filename=self.cfg.dataset.style.pickle_filename)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.stylized_dataset,
                                           # sampler=SubsetRandomSampler(torch.randperm(len(self.train_dataset))[:1000]), # in case need to test with less data
                                           batch_size=self.cfg.dataset.batch_size,
                                           shuffle=True,
                                           num_workers=self.cfg.dataset.num_workers,
                                           pin_memory=True,
                                           drop_last=True)


class ContentImageDataset:
    def __init__(self, cfg: DictConfig) -> None:
        """
        Base dataset class. Only downloads data and applies only ToTensor transform.
        Other augmentations to run on GPU.
        :param cfg: Hydra format dataset configuration. [omegaconf.dictconfig.DictConfig]
        """
        self.cfg = cfg

    @staticmethod
    def get_no_transforms():
        data_transforms = transforms.Compose([transforms.ToTensor()])
        return data_transforms

    @staticmethod
    def get_probe_transforms(size):
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.ToTensor()])
        return data_transforms

    @staticmethod
    def get_resize_transforms():
        data_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        return data_transforms

    def split_train_valid_dataset(self, dataset):
        dataset_len = len(dataset)
        # TODO: why are we taking the test split part? they are loaded separatelly.
        val_split = self.cfg.dataset.val_split / (1 - self.cfg.dataset.test_split)
        val_len = int(dataset_len * val_split)
        train_len = dataset_len - val_len
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, (train_len, val_len))
        return train_dataset, val_dataset

    def get_cifar10(self):

        train_val_dataset = datasets.CIFAR10(self.cfg.dataset.content.path,
                                             train=True,
                                             transform=self.get_no_transforms(),
                                             download=True)

        train_dataset, val_dataset = self.split_train_valid_dataset(train_val_dataset)

        test_dataset = datasets.CIFAR10(self.cfg.dataset.content.path,
                                        train=False,
                                        transform=self.get_no_transforms(),
                                        download=True)

        return train_dataset, val_dataset, test_dataset

    def get_stl10_labeled(self):
        # need to add transformations used in paper: random cropping
        indices = list(range(5000))
        split = int(np.floor(self.cfg.dataset.val_split * 5000))

        train_idx, val_idx = indices[split:], indices[:split]

        # need to load two distinc datasets train and val because they have different transforms.
        train_dataset = datasets.STL10(self.cfg.dataset.content.path, split='train',
                                       transform=self.get_probe_transforms(96),
                                       download=True)

        val_dataset = datasets.STL10(self.cfg.dataset.content.path, split='train',
                                     transform=self.get_no_transforms(),
                                     download=False)

        test_dataset = datasets.STL10(self.cfg.dataset.content.path, split='test',
                                      transform=self.get_no_transforms(),
                                      download=False)

        return Subset(train_dataset, train_idx), Subset(val_dataset, val_idx), test_dataset

    def get_stl10_unlabeled(self):

        return datasets.STL10(self.cfg.dataset.content.path, split='unlabeled',
                              transform=self.get_no_transforms(),
                              download=True)

    def get_stl10_bg(self):
        class TransparencyDataset:
            def __init__(self, rgb, rgba):
                assert len(rgb) == len(rgba)
                self.rgb = rgb
                self.rgba = rgba

            def __len__(self):
                return len(self.rgb)

            def __getitem__(self, index):
                return torch.cat([self.rgb[index][0], transforms.ToTensor()(self.rgba[index][0])[[3], ...]], dim=0), -1

        with open(self.cfg.dataset.content.bg_path, "rb") as fd:
            return TransparencyDataset(datasets.STL10(self.cfg.dataset.content.path,
                                                      split='unlabeled', download=True,
                                                      transform=self.get_no_transforms()),
                                       pickle.load(fd)['unlabeled'])

    def get_imagenet(self):

        train_dataset = datasets.ImageNet(self.cfg.dataset.content.path,
                                          split='train',
                                          transform=self.get_resize_transforms())

        val_dataset = datasets.ImageNet(self.cfg.dataset.content.path,
                                        split='val',
                                        transform=self.get_no_transforms())

        test_dataset = datasets.ImageNet(self.cfg.dataset.content.path,
                                         split='test',
                                         transform=self.get_no_transforms())

        return train_dataset, val_dataset, test_dataset

    def get_dataset_for_linear_probe(self):
        """
        Get datasets for the linear probe. We need a training, a validation and a test dataset.
        Everything should be labeled.
        """
        val_datasets = {'cifar10': lambda: self.get_cifar10(),
                        'stl10': lambda: self.get_stl10_labeled(),
                        'stl10_bg': lambda: self.get_stl10_labeled(),
                        'imagenet': lambda: self.get_imagenet()}

        try:
            dataset_fn = val_datasets[self.cfg.dataset.content.name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()

    def get_dataset_for_simclr(self):
        """
        Get datasets for the simclr. We only need the training dataset, and unlabeled is fine.
        Returns one dataset
        """
        if self.cfg.augment.background_replacer:
            valid_datasets = {  
                'stl10': lambda: self.get_stl10_bg(),
            }
        else:            
            valid_datasets = {
                'cifar10': lambda: self.get_cifar10()[0],
                'stl10': lambda: self.get_stl10_unlabeled(),   
                'imagenet': lambda: self.get_imagenet()[0]
            }
        try:
            dataset_fn = val_datasets[self.cfg.dataset.content.name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()


class StylizedDatasetOnGPU:
    def __init__(self,
                 content_dataset: datasets.vision.VisionDataset,
                 style_data_path: str,
                 style_pickle_filename: str) -> None:
        """
        Dataset for style image featurization. Uses a frozen VGG network to precalculate style feature vectors
        and avoid loading style images to GPU on each batch.
        :param content_dataset: Content images torchvision dataset. [torchvision.datasets.vision.VisionDataset]
        :param style_data_path: Style metadata and images main path. [str]
        :param style_pickle_filename: Precalculated features pickle filename. [str]
        """

        self.content_dataset = content_dataset

        # TODO: abstract this values to config file
        self.content_size = 96

        self.content_tf = test_transform(self.content_size)
        self.toPIL = transforms.ToPILImage()

        # load style images embeddings
        precalculated_style_features_path = f'{style_data_path}{style_pickle_filename}'
        if os.path.exists(precalculated_style_features_path):
            self.style_embeddings = pickle.load(open(precalculated_style_features_path, 'rb'))
        else:
            print('Precalculated style image features pickle file does not exist. Needs to calculate features')
            self.style_embeddings = None

    def __len__(self):
        return len(self.content_dataset)

    def __getitem__(self, idx):
        # sample to style images
        # TODO: does it matter that 2 random images have the same style in one batch? Maybe need to implement a more efficient sampling method per epoch.
        sampled_styles = torch.randint(low=0, high=self.style_embeddings.shape[0], size=(2,))

        # get content image
        content_image = self.content_dataset[idx][0]
        if content_image.shape[0] == 4:  # RGBA
            transparancy = content_image[3, None, ...]
            content_image = content_image[:3, ...]
        else:
            transparancy = []
        content = self.content_tf(self.toPIL(content_image))

        return [content,
                transparancy,
                self.style_embeddings[sampled_styles[0]],
                self.style_embeddings[sampled_styles[1]]], -1


class PrecalculateSyleEmbeddings:
    def __init__(self, cfg: DictConfig, style_transform: Compose) -> None:
        """
        Dataset for style image featurization. Uses a frozen VGG network to precalculate style feature vectors
        and avoid loading style images to GPU on each batch.
        :param cfg: Hydra format style CFG configuration. [omegaconf.dictconfig.DictConfig]
        :param style_transform: torchvision compose of transformations to apply on style image. [torchvision.transforms.Compose]
        """
        # get style metadata
        self.style_transform = style_transform
        self.style_image_folder_path = f'{cfg.data_path}{cfg.image_path}'
        self.style_csv = pd.read_csv(f'{cfg.data_path}{cfg.metadata_filename}')

        # style transforms
        self.style_tf = test_transform(cfg.image_size)
        self.toPIL = transforms.ToPILImage()

    def __len__(self):
        return len(self.style_csv)

    def __getitem__(self, idx):
        # # sample to style images
        style_filename = self.style_csv.loc[idx].filename

        # load and transform style images
        style_image = self.style_transform(Image.open(f'{self.style_image_folder_path}{style_filename}'))
        style_image = self.style_tf(self.toPIL(style_image))

        return style_image
