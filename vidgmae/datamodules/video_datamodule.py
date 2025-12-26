import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from vidgmae.datamodules.components.video_dataset import VideoDataset
from vidgmae.utils import get_pylogger

log = get_pylogger(__name__)


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
        self.data_val: Optional[Dataset] = None

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
        if self.data_train or self.data_val:
            return

        self._select_datasets()

    def _select_datasets(self) -> None:
        builders: list[tuple[str, Callable[[], None]]] = [
            ("k400-vjepa", self._build_k400_vjepa),
            (
                "k400",
                lambda: self._build_from_labels(
                    "kinetics_400_train_label.npy", "kinetics_400_val_label.npy"
                ),
            ),
            (
                "ego4d",
                lambda: self._build_from_labels(
                    "ego4d_train_label.npy", "ego4d_val_label.npy", flip_rgb=True
                ),
            ),
            (
                "kinetics",
                lambda: self._build_from_labels(
                    "kinetics_train_label.npy", "kinetics_val_label.npy"
                ),
            ),
            (
                "kubric",
                lambda: self._build_from_labels(
                    ["kubric_train_label.npy", "kinetics_train_subset.npy"],
                    "kubric_val_label.npy",
                ),
            ),
            (
                "zeroshot",
                lambda: self._build_from_labels(
                    [
                        "kubric_train_label.npy",
                        "kinetics_train_subset.npy",
                        "davis_label.npy",
                    ],
                    "kubric_val_label.npy",
                ),
            ),
            (
                "testdata",
                lambda: self._build_from_labels(
                    "test_data_label.npy", "test_data_label.npy"
                ),
            ),
            (
                "cater",
                lambda: self._build_from_labels(
                    "cater_train_label.npy", "cater_val_label.npy"
                ),
            ),
            (
                "ucf101",
                lambda: self._build_from_labels(
                    "ucf101_train_label.npy", "ucf101_val_label.npy"
                ),
            ),
            (
                "ssv2",
                lambda: self._build_from_labels(
                    "ssv2_train_label.npy", "ssv2_val_label.npy"
                ),
            ),
        ]

        for key, builder in builders:
            if key in self.hparams.cfg.training_type:
                log.info(
                    f"Using dataset key '{key}' for training_type '{self.hparams.cfg.training_type}'"
                )
                builder()
                return
        raise ValueError(
            f"Invalid Dataset Specification: {self.hparams.cfg.training_type}"
        )

    def _build_k400_vjepa(self) -> None:
        from vidgmae.datamodules.components.video_datasets_vjepa import (
            VideoDataset as VideoDatasetVJepa,
        )

        self.data_train = VideoDatasetVJepa(
            self.hparams.cfg,
            train=True,
            datasets_weights=None,
            frames_per_clip=self.hparams.cfg.seq_length,
            frame_step=4,
            num_clips=1,
            transform=None,
            shared_transform=None,
            random_clip_sampling=True,
            allow_clip_overlap=True,
            filter_short_videos=False,
            filter_long_videos=int(10**9),
            duration=None,
        )
        self.data_val = VideoDatasetVJepa(
            self.hparams.cfg,
            train=False,
            datasets_weights=None,
            frames_per_clip=self.hparams.cfg.seq_length,
            frame_step=4,
            num_clips=1,
            transform=None,
            shared_transform=None,
            random_clip_sampling=True,
            allow_clip_overlap=True,
            filter_short_videos=False,
            filter_long_videos=int(10**9),
            duration=None,
        )

    def _build_from_labels(
        self, train_files, val_files, flip_rgb: bool = False
    ) -> None:
        train_labels = self._load_labels(train_files)
        val_labels = self._load_labels(val_files)
        log.info(f"Loaded {len(train_labels)} train and {len(val_labels)} val videos.")
        self.data_train = VideoDataset(
            self.hparams.cfg, train_labels, True, flip_rgb=flip_rgb
        )
        self.data_val = VideoDataset(
            self.hparams.cfg, val_labels, False, flip_rgb=flip_rgb
        )

    def _load_labels(self, files) -> np.ndarray:
        if isinstance(files, (list, tuple)):
            arrays = [np.load(self._label_path(f)) for f in files]
            return np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
        return np.load(self._label_path(files))

    def _label_path(self, filename: str) -> Path:
        path = self._project_root() / "data" / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected dataset labels at {path}")
        return path

    def _project_root(self) -> Path:
        return Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

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
