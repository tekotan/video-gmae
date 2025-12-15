import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .gmrw_kinetics_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    chunk_example,
    prepare_gmrw_sample,
)


def load_davis_entries(pickle_path: Path) -> List[Dict[str, Any]]:
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported DAVIS pickle structure.")


class GMRWDavisDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_size = getattr(cfg, "input_size")
        self.chunk_size = int(
            getattr(cfg.finetune_params, "gmrw_chunk_size", getattr(cfg, "seq_length", 16))
        )
        default_path = Path(
            "/home/tekotan/Chewbacca_test/data-download/tapvid_davis/tapvid_davis.pkl"
        )
        self.pickle_path = Path(
            getattr(cfg.finetune_params, "gmrw_davis_path", default_path)
        )
        self.max_samples = getattr(cfg.finetune_params, "gmrw_max_samples", -1)
        self.tracks_to_sample = int(
            getattr(cfg.finetune_params, "tracks_to_sample", 32)
        )

        entries = load_davis_entries(self.pickle_path)
        self.samples: List[Dict[str, Any]] = []
        for entry in entries:
            chunks = chunk_example(entry, self.chunk_size)
            for chunk in chunks:
                self.samples.append(chunk)
                if 0 < self.max_samples == len(self.samples):
                    break
            if 0 < self.max_samples == len(self.samples):
                break

    def __len__(self) -> int:
        return len(self.samples)

    def _prepare_chunk(self, index: int) -> Dict[str, np.ndarray]:
        chunk = self.samples[index]
        return prepare_gmrw_sample(chunk, self.input_size)

    def get_raw_video(self, index: int) -> np.ndarray:
        converted = self._prepare_chunk(index)
        video = converted["video"][0]
        return (video + 1.0) / 2.0

    def __getitem__(self, index: int):
        converted = self._prepare_chunk(index)

        video = converted["video"][0]
        video_01 = (video + 1.0) / 2.0
        frames_norm = (video_01 - IMAGENET_MEAN) / IMAGENET_STD
        frames_torch = torch.from_numpy(frames_norm).permute(3, 0, 1, 2).contiguous()

        target_points = converted["target_points"][0] / self.input_size
        query_points = converted["query_points"][0].copy()
        if query_points.size == 0:
            query_points = np.zeros((0, 3), dtype=np.float32)
        else:
            query_points[:, 0] = 0.0
            query_points[:, 1:] /= self.input_size
        occluded = converted["occluded"][0].astype(np.float32)

        return (
            frames_torch.float(),
            self._pad_or_trim(
                torch.from_numpy(query_points).float(),
                (self.tracks_to_sample, 3),
                fill_value=-1.0,
            ),
            self._pad_or_trim(
                torch.from_numpy(target_points).float(),
                (self.tracks_to_sample, target_points.shape[1], 2),
            ),
            self._pad_or_trim(
                torch.from_numpy(occluded).float(),
                (self.tracks_to_sample, occluded.shape[1]),
                fill_value=1.0,
            ),
        )

    def _pad_or_trim(
        self,
        tensor: torch.Tensor,
        target_shape: Tuple[int, ...],
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        current_shape = tensor.shape
        if list(current_shape) == list(target_shape):
            return tensor

        num_current = current_shape[0] if tensor.ndim > 0 else 0
        target_tracks = target_shape[0]

        if num_current >= target_tracks:
            indices = torch.arange(num_current, device=tensor.device)[:target_tracks]
            return tensor.index_select(0, indices)

        padding_shape = (target_tracks - num_current, *current_shape[1:])
        pad_tensor = torch.full(padding_shape, fill_value, dtype=tensor.dtype)
        return torch.cat([tensor, pad_tensor], dim=0)