import glob
import os
from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import DictConfig
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from chewbacca.datamodules.components.imagenet_dataset import build_imagenet_dataset
import numpy as np

class ImageDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        cfg: DictConfig,
        train: bool = True,
    ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val:   Optional[Dataset] = None
        
    @property
    def num_classes(self):
        return 80

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            if self.hparams.cfg.dataset_type=="imagenet":
                if(self.hparams.train):
                    self.data_train = build_imagenet_dataset(True, self.hparams.cfg)
                self.data_val = build_imagenet_dataset(False, self.hparams.cfg)
            elif self.hparams.cfg.dataset_type == "cifar100":
                from torchvision.datasets import CIFAR100
                from torchvision import transforms

                transform = transforms.Compose([
                    transforms.Resize((self.hparams.cfg.input_size, self.hparams.cfg.input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])

                if self.hparams.train:
                    self.data_train = CIFAR100(root='./data', train=True, download=True, transform=transform)
                self.data_val = CIFAR100(root='./data', train=False, download=True, transform=transform)
                
    def train_dataloader(self):
        
        
        dataloader = DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.cfg.train_batch_size,
            num_workers=self.hparams.cfg.train_num_workers,
            pin_memory=self.hparams.cfg.pin_memory,
            shuffle=True,        
            drop_last=True if self.hparams.cfg.task=="finetune" else False,
        )
    
        return dataloader

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.cfg.test_batch_size,
            num_workers=self.hparams.cfg.test_num_workers,
            pin_memory=self.hparams.cfg.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    pass
