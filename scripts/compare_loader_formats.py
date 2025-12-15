#!/usr/bin/env python3
"""Compare eval-kinetics and gmrw-kinetics loader outputs for shape/range parity."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chewbacca.datamodules.components.gmrw_kinetics_dataset import (
    GMRWKineticsDataset,
)
from chewbacca.datamodules.components.point_tracking_eval_dataset import (
    PointTrackingEvalDataset,
)


def build_cfg(
    training_type: str,
    input_size: int,
    seq_length: int,
    pickle_path: Path,
    tracks_to_sample: int,
) -> OmegaConf:
    return OmegaConf.create(
        {
            "training_type": training_type,
            "input_size": input_size,
            "seq_length": seq_length,
            "finetune_params": {
                "tracks_to_sample": tracks_to_sample,
                "test": False,
                "gmrw_chunk_size": seq_length,
                "gmrw_kinetics_path": str(pickle_path),
                "gmrw_max_samples": -1,
            },
        }
    )


def tensor_stats(t: torch.Tensor) -> Tuple[Tuple[int, ...], float, float]:
    arr = t.detach().cpu().numpy()
    return tuple(arr.shape), float(arr.min()), float(arr.max())


def compare_datasets(
    pickle_path: Path,
    input_size: int,
    seq_length: int,
    num_samples: int,
    tracks_to_sample: int,
) -> None:
    cfg_eval = build_cfg(
        "point-tracking+eval-kinetics",
        input_size,
        seq_length,
        pickle_path,
        tracks_to_sample,
    )
    cfg_gmrw = build_cfg(
        "point-tracking+gmrw-kinetics",
        input_size,
        seq_length,
        pickle_path,
        tracks_to_sample,
    )

    eval_dataset = PointTrackingEvalDataset([str(pickle_path)], cfg_eval)
    gmrw_dataset = GMRWKineticsDataset(cfg_gmrw)

    eval_iter = iter(eval_dataset)

    for idx in range(num_samples):
        try:
            frames_eval, query_eval, target_eval, occ_eval = next(eval_iter)
        except StopIteration:
            break
        frames_gmrw, query_gmrw, target_gmrw, occ_gmrw = gmrw_dataset[idx]

        print(f"Sample {idx:03d}")
        se, me, xe = tensor_stats(frames_eval)
        sg, mg, xg = tensor_stats(frames_gmrw)
        print(f"  frames eval shape {se} min {me:.4f} max {xe:.4f}")
        print(f"  frames gmrw shape {sg} min {mg:.4f} max {xg:.4f}")

        s_eval, m_eval, x_eval = tensor_stats(query_eval)
        s_gmrw, m_gmrw, x_gmrw = tensor_stats(query_gmrw)
        print(f"  query eval shape {s_eval} min {m_eval:.4f} max {x_eval:.4f}")
        print(f"  query gmrw shape {s_gmrw} min {m_gmrw:.4f} max {x_gmrw:.4f}")

        st_eval, mt_eval, xt_eval = tensor_stats(target_eval)
        st_gmrw, mt_gmrw, xt_gmrw = tensor_stats(target_gmrw)
        print(
            f"  target eval shape {st_eval} min {mt_eval:.4f} max {xt_eval:.4f}"
        )
        print(
            f"  target gmrw shape {st_gmrw} min {mt_gmrw:.4f} max {xt_gmrw:.4f}"
        )

        so_eval, mo_eval, xo_eval = tensor_stats(occ_eval)
        so_gmrw, mo_gmrw, xo_gmrw = tensor_stats(occ_gmrw)
        print(f"  occ eval shape {so_eval} min {mo_eval:.4f} max {xo_eval:.4f}")
        print(f"  occ gmrw shape {so_gmrw} min {mo_gmrw:.4f} max {xo_gmrw:.4f}")

        print("")


def main():
    parser = argparse.ArgumentParser(description="Compare loader formats.")
    parser.add_argument(
        "--pickle_path",
        type=Path,
        default=Path(
            "/home/tekotan/Chewbacca_test/data-download/tapvid_kinetics/dummy_small.pkl"
        ),
    )
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--seq_length", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--tracks_to_sample", type=int, default=32)
    args = parser.parse_args()

    compare_datasets(
        pickle_path=args.pickle_path,
        input_size=args.input_size,
        seq_length=args.seq_length,
        num_samples=args.num_samples,
        tracks_to_sample=args.tracks_to_sample,
    )


if __name__ == "__main__":
    main()
