import glob
import random
from typing import Any, Dict, Optional

import numpy as np
import webdataset as wds
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from chewbacca.datamodules.components.video_dataset import VideoDataset

class VideoDataModule(LightningDataModule):
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

        # for evals
        if "k400-vjepa" in self.hparams.cfg.training_type:
            from chewbacca.datamodules.components.video_datasets_vjepa import VideoDataset as VideoDatasetVJepa
            self.data_train = VideoDatasetVJepa(self.hparams.cfg, train=True, datasets_weights=None, frames_per_clip=self.hparams.cfg.seq_length, 
                                            frame_step=4, num_clips=1, transform=None, shared_transform=None, random_clip_sampling=True, 
                                            allow_clip_overlap=True, filter_short_videos=False, filter_long_videos=int(10**9), duration=None)
            self.data_val = VideoDatasetVJepa(self.hparams.cfg, train=False, datasets_weights=None, frames_per_clip=self.hparams.cfg.seq_length, frame_step=4, num_clips=1, 
                                            transform=None, shared_transform=None, random_clip_sampling=True, allow_clip_overlap=True, filter_short_videos=False, 
                                            filter_long_videos=int(10**9), duration=None)

        elif "k400" in self.hparams.cfg.training_type:
            from chewbacca.datamodules.components.video_dataset import VideoDataset
            video_paths_train = np.load("data/kinetics_400_train_label.npy")
            video_paths_val = np.load("data/kinetics_400_val_label.npy")
            self.data_train = VideoDataset(self.hparams.cfg, video_paths_train, True)
            self.data_val = VideoDataset(self.hparams.cfg, video_paths_val, False)

        elif "ego4d" in self.hparams.cfg.training_type:
            video_paths_train = np.load("data/ego4d_train_label.npy")
            video_paths_val = np.load("data/ego4d_val_label.npy")
            self.data_train = VideoDataset(self.hparams.cfg, video_paths_train, True, flip_rgb=True)
            self.data_val = VideoDataset(self.hparams.cfg, video_paths_val, False, flip_rgb=True)


        elif "kinetics" in self.hparams.cfg.training_type:
            from chewbacca.datamodules.components.video_dataset import VideoDataset
            video_paths_train = np.load("data/kinetics_train_label.npy")
            video_paths_val = np.load("data/kinetics_val_label.npy")
            self.data_train = VideoDataset(self.hparams.cfg, video_paths_train, True)
            self.data_val = VideoDataset(self.hparams.cfg, video_paths_val, False)
        elif "kubric" in self.hparams.cfg.training_type:
            from chewbacca.datamodules.components.video_dataset import VideoDataset
            video_paths_train = np.concatenate([
                np.load("data/kubric_train_label.npy"),
                np.load("data/kinetics_train_subset.npy")
            ])
            video_paths_val = np.load("data/kubric_val_label.npy")
            self.data_train = VideoDataset(self.hparams.cfg, video_paths_train, True)
            self.data_val = VideoDataset(self.hparams.cfg, video_paths_val, False)
        elif "zeroshot" in self.hparams.cfg.training_type:
            from chewbacca.datamodules.components.video_dataset import VideoDataset
            video_paths_train = np.concatenate([
                np.load("data/kubric_train_label.npy"),
                np.load("data/kinetics_train_subset.npy"),
                np.load("data/davis_label.npy"),
            ])
            video_paths_val = np.load("data/kubric_val_label.npy")
            self.data_train = VideoDataset(self.hparams.cfg, video_paths_train, True)
            self.data_val = VideoDataset(self.hparams.cfg, video_paths_val, False)
            
        elif "testdata" in self.hparams.cfg.training_type:
            from chewbacca.datamodules.components.video_dataset import VideoDataset
            video_paths_train = np.load("data/test_data_label.npy")
            video_paths_val = np.load("data/test_data_label.npy")
            self.data_train = VideoDataset(self.hparams.cfg, video_paths_train, True)
            self.data_val = VideoDataset(self.hparams.cfg, video_paths_val, False)

        elif "cater" in self.hparams.cfg.training_type:
            video_paths_train = np.load("data/cater_train_label.npy")
            video_paths_val = np.load("data/cater_val_label.npy")
            self.data_train = VideoDataset(self.hparams.cfg, video_paths_train, True)
            self.data_val = VideoDataset(self.hparams.cfg, video_paths_val, False)
        elif "ucf101" in self.hparams.cfg.training_type:
            from chewbacca.datamodules.components.video_dataset import VideoDataset
            video_paths_train = np.load("data/ucf101_train_label.npy")
            video_paths_val = np.load("data/ucf101_val_label.npy")
            self.data_train = VideoDataset(self.hparams.cfg, video_paths_train, True)
            self.data_val = VideoDataset(self.hparams.cfg, video_paths_val, False)
        elif "ssv2" in self.hparams.cfg.training_type:
            video_paths_train = np.load("data/ssv2_train_label.npy")
            video_paths_val = np.load("data/ssv2_val_label.npy")
            self.data_train = VideoDataset(self.hparams.cfg, video_paths_train, True)
            self.data_val = VideoDataset(self.hparams.cfg, video_paths_val, False)
        else:
            raise ValueError("Invalid Dataset Specification")

    def train_dataloader(self):
        dataloader = DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.cfg.train_batch_size,
            num_workers=self.hparams.cfg.train_num_workers,
            pin_memory=self.hparams.cfg.pin_memory,
            shuffle=True,
            drop_last=True,
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
