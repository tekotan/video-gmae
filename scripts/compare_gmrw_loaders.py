#!/usr/bin/env python3
"""Compare GMRW and Chewbacca loaders to ensure frame parity."""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chewbacca.datamodules.finetune_datamodule import FinetuneDataModule
from chewbacca.datamodules.components.gmrw_kinetics_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def build_cfg(
    training_type: str,
    input_size: int,
    seq_length: int,
    batch_size: int,
    pickle_path: Path,
) -> OmegaConf:
    cfg = OmegaConf.create(
        {
            "training_type": training_type,
            "input_size": input_size,
            "seq_length": seq_length,
            "train_batch_size": batch_size,
            "train_num_workers": 0,
            "test_batch_size": batch_size,
            "test_num_workers": 0,
            "pin_memory": False,
            "finetune_params": {
                "tracks_to_sample": 32,
                "test": False,
                "gmrw_chunk_size": seq_length,
                "gmrw_kinetics_path": str(pickle_path),
                "gmrw_max_samples": -1,
            },
        }
    )
    return cfg


def denormalize_frames(frames: torch.Tensor) -> np.ndarray:
    frames_np = frames.cpu().numpy()
    mean = IMAGENET_MEAN.reshape(1, 1, 3, 1, 1)
    std = IMAGENET_STD.reshape(1, 1, 3, 1, 1)
    frames_np = frames_np * std + mean
    frames_np = np.clip(frames_np, 0.0, 1.0)
    return frames_np.transpose(0, 1, 3, 4, 2)


def compare_batches(dm: FinetuneDataModule, num_batches: int, tolerance: float) -> None:
    dataloader = dm.train_dataloader()
    dataset = dm.data_train

    logger = logging.getLogger("compare")
    total_samples = 0
    max_diff_overall = 0.0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        frames = batch[0]
        frames_01 = denormalize_frames(frames)
        B = frames_01.shape[0]
        batch_max = 0.0

        for b in range(B):
            sample_idx = total_samples + b
            gmrw_frames = dataset.get_raw_video(sample_idx)
            diff = np.abs(frames_01[b] - gmrw_frames)
            sample_max = diff.max()
            sample_mean = diff.mean()
            batch_max = max(batch_max, sample_max)
            max_diff_overall = max(max_diff_overall, sample_max)

            logger.info(
                "batch=%03d sample=%06d max_diff=%.6f mean_diff=%.6f",
                batch_idx,
                sample_idx,
                sample_max,
                sample_mean,
            )

            if sample_max > tolerance:
                raise RuntimeError(
                    f"Frames diverged beyond tolerance at sample {sample_idx}: {sample_max:.6f}"
                )

        total_samples += B
        logger.info(
            "Completed batch %03d with batch_max=%.6f (cumulative samples=%d)",
            batch_idx,
            batch_max,
            total_samples,
        )

    logger.info(
        "Comparison finished for %d batches (%d samples). "
        "Overall max difference: %.6f",
        min(num_batches, batch_idx + 1),
        total_samples,
        max_diff_overall,
    )


def main():
    parser = argparse.ArgumentParser(description="Compare GMRW and Chewbacca loaders.")
    parser.add_argument(
        "--kinetics_pkl",
        type=Path,
        default=Path(
            "/home/tekotan/Chewbacca_test/data-download/tapvid_kinetics/kinetics_10percent_sample.pkl"
        ),
    )
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    cfg = build_cfg(
        training_type="point-tracking+gmrw-kinetics",
        input_size=args.input_size,
        seq_length=args.chunk_size,
        batch_size=args.batch_size,
        pickle_path=args.kinetics_pkl,
    )

    dm = FinetuneDataModule(cfg=cfg, train=True)
    dm.setup()

    compare_batches(dm, num_batches=args.num_batches, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
