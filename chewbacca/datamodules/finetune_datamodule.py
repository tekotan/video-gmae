import glob
import random
from typing import Any, Dict, Optional

import numpy as np
import webdataset as wds
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

class FinetuneDataModule(LightningDataModule):
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
        if "point-tracking" in self.hparams.cfg.training_type:
            if "eval" in self.hparams.cfg.training_type:
                from chewbacca.datamodules.components.point_tracking_eval_dataset import PointTrackingEvalDataset

                pickle_files = [
                    "/home/ubuntu/Chewbacca_test/data-download/tapvid_davis/tapvid_davis.pkl",
                    # "/home/tekotan/Chewbacca_test/data-download/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl",
                    # "/home/tekotan/Chewbacca_test/data-download/tapvid_kinetics/0000_of_0010.pkl",
                    # "/home/tekotan/Chewbacca_test/data-download/tapvid_kinetics/0001_of_0010.pkl",
                    # "/home/tekotan/Chewbacca_test/data-download/tapvid_kinetics/0002_of_0010.pkl",
                    # "/home/tekotan/Chewbacca_test/data-download/tapvid_kinetics/0003_of_0010.pkl",
                    # "/home/tekotan/Chewbacca_test/data-download/tapvid_kinetics/0004_of_0010.pkl",
                    # "/home/tekotan/Chewbacca_test/data-download/tapvid_kinetics/0005_of_0010.pkl",
                    # "/home/tekotan/Chewbacca_test/data-download/tapvid_kinetics/0006_of_0010.pkl",
                    # "/home/tekotan/Chewbacca_test/data-download/tapvid_kinetics/0007_of_0010.pkl",
                    # "/home/tekotan/Chewbacca_test/data-download/tapvid_kinetics/0008_of_0010.pkl",
                    # "/home/tekotan/Chewbacca_test/data-download/tapvid_kinetics/0009_of_0010.pkl",
                ]
                self.data_train = PointTrackingEvalDataset(pickle_files, self.hparams.cfg)
                self.data_val = PointTrackingEvalDataset(pickle_files, self.hparams.cfg)
            elif "train" in self.hparams.cfg.training_type:
                from chewbacca.datamodules.components.kubric_dataset import KubricPointTrackingDataset
                self.data_train = KubricPointTrackingDataset(self.hparams.cfg, True)
                self.data_val = KubricPointTrackingDataset(self.hparams.cfg, False)
        elif "object-tracking" in self.hparams.cfg.training_type:
            from chewbacca.datamodules.components.mot_dataset import VideoAnnotationDataset
            video_paths_train = np.load("data/mot_train_label.npy")
            if self.hparams.cfg.finetune_params.test:
                video_paths_val = np.load("data/mot_train_label.npy")
            else:
                video_paths_val = np.load("data/mot_val_label.npy")
            self.data_train = VideoAnnotationDataset(self.hparams.cfg, video_paths_train, "/datasets/motsynth/current/MOTSynth_mot_annotations/mot_annotations/", train=True)
            self.data_val = VideoAnnotationDataset(self.hparams.cfg, video_paths_val, "/datasets/motsynth/current/MOTSynth_mot_annotations/mot_annotations/", train=False)
        else:
            raise ValueError("Invalid Dataset Specification")

    def train_dataloader(self):
        dataloader = DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.cfg.train_batch_size,
            num_workers=self.hparams.cfg.train_num_workers,
            pin_memory=self.hparams.cfg.pin_memory,
            shuffle=False,
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
