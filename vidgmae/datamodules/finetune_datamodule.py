import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from vidgmae.utils import get_pylogger

log = get_pylogger(__name__)


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
        self.data_val: Optional[Dataset] = None
        self._project_root = Path(
            os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2])
        )

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

        data_root = self._project_root / "data-download"

        if "point-tracking" in self.hparams.cfg.training_type:
            self._setup_point_tracking(data_root)
            return
        if "object-tracking" in self.hparams.cfg.training_type:
            self._setup_object_tracking()
            return
        raise ValueError(
            f"Invalid Dataset Specification: {self.hparams.cfg.training_type}"
        )

    def _setup_point_tracking(self, data_root: Path) -> None:
        from vidgmae.datamodules.components.point_tracking_eval_dataset import (
            PointTrackingEvalDataset,
        )

        tapvid_paths = {
            "eval-davis": [data_root / "tapvid_davis" / "tapvid_davis.pkl"],
            "eval-rgb": [data_root / "tapvid_rgb_stacking" / "tapvid_rgb_stacking.pkl"],
            "eval-kinetics": [
                data_root / "tapvid_kinetics" / "kinetics_10percent_sample.pkl"
            ],
        }

        for key, paths in tapvid_paths.items():
            if key in self.hparams.cfg.training_type:
                files = [str(self._ensure_file(p, key)) for p in paths]
                dataset = PointTrackingEvalDataset(files, self.hparams.cfg)
                self.data_val = dataset
                self.data_train = dataset
                log.info(f"Loaded point-tracking eval dataset '{key}' from {files}")
                return

        if "train" in self.hparams.cfg.training_type:
            import importlib.util

            missing = []
            for pkg in ("tensorflow", "tensorflow_datasets", "tensorflow_graphics"):
                if importlib.util.find_spec(pkg) is None:
                    missing.append(pkg)
            if missing:
                extras_cmd = "pip install '.[kubric]'"
                raise RuntimeError(
                    f"Kubric dataset selected but missing dependencies: {missing}. "
                    f"Install Kubric extras first via `{extras_cmd}`."
                )
            
            from vidgmae.datamodules.components.kubric_dataset import (
                KubricPointTrackingDataset,
            )

            self.data_train = KubricPointTrackingDataset(self.hparams.cfg, True)
            self.data_val = KubricPointTrackingDataset(self.hparams.cfg, False)
            log.info("Loaded Kubric point-tracking train/val splits.")
            return

        raise ValueError(
            f"Invalid point-tracking training_type: {self.hparams.cfg.training_type}"
        )

    def _setup_object_tracking(self) -> None:
        from vidgmae.datamodules.components.mot_dataset import VideoAnnotationDataset

        video_paths_train = self._load_label("mot_train_label.npy")
        val_filename = (
            "mot_train_label.npy"
            if self.hparams.cfg.finetune_params.test
            else "mot_val_label.npy"
        )
        video_paths_val = self._load_label(val_filename)
        mot_root = Path(
            "/datasets/motsynth/current/MOTSynth_mot_annotations/mot_annotations/"
        )
        self._ensure_path(mot_root, "MOTSynth annotations root")
        self.data_train = VideoAnnotationDataset(
            self.hparams.cfg, video_paths_train, str(mot_root), train=True
        )
        self.data_val = VideoAnnotationDataset(
            self.hparams.cfg, video_paths_val, str(mot_root), train=False
        )
        log.info(
            f"Loaded MOT datasets with {len(video_paths_train)} train and {len(video_paths_val)} val items."
        )

    def _load_label(self, filename: str) -> np.ndarray:
        path = self._project_root / "data" / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected label file at {path}")
        return np.load(path)

    def _ensure_file(self, path: Path, description: str) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"Missing {description} file: {path}")
        return path

    def _ensure_path(self, path: Path, description: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")

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
