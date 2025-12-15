import io
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def decode_video_frames(raw_frames: Sequence[Any]) -> np.ndarray:
    if isinstance(raw_frames, np.ndarray):
        frames = raw_frames
    elif isinstance(raw_frames, list) and raw_frames:
        if isinstance(raw_frames[0], bytes):
            decoded = []
            for frame_bytes in raw_frames:
                with Image.open(io.BytesIO(frame_bytes)) as img:
                    decoded.append(np.array(img))
            frames = np.stack(decoded, axis=0)
        else:
            frames = np.stack(raw_frames, axis=0)
    else:
        raise ValueError("Unsupported video container.")
    return frames


def resize_video_square(frames: np.ndarray, output_size: int) -> np.ndarray:
    frames = frames.astype(np.float32)
    frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
    resized = F.interpolate(
        frames_tensor,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.permute(0, 2, 3, 1).numpy()


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Dict[str, np.ndarray]:
    tracks = []
    occs = []
    queries = []
    trackgroups = []
    trackgroup = np.arange(target_occluded.shape[0])

    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        if not np.any(mask):
            continue
        query = np.stack(
            [
                i * np.ones(target_occluded.shape[0]),
                target_points[:, i, 1],
                target_points[:, i, 0],
            ],
            axis=-1,
        )
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
        trackgroups.append(trackgroup[mask])

    if not queries:
        # Fallback to a dummy entry to keep alignment
        queries.append(np.zeros((0, 3), dtype=np.float32))
        tracks.append(np.zeros((0, target_points.shape[1], 2), dtype=np.float32))
        occs.append(np.zeros((0, target_points.shape[1]), dtype=bool))
        trackgroups.append(np.zeros((0,), dtype=np.int32))

    return {
        "video": frames[np.newaxis, ...],
        "query_points": np.concatenate(queries, axis=0)[np.newaxis, ...],
        "target_points": np.concatenate(tracks, axis=0)[np.newaxis, ...],
        "occluded": np.concatenate(occs, axis=0)[np.newaxis, ...],
        "trackgroup": np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
    }


def chunk_example(example: Dict[str, Any], chunk_size: int) -> List[Dict[str, Any]]:
    if chunk_size <= 0:
        return [example]

    total_frames = len(example["video"])
    if total_frames <= chunk_size:
        return [example]

    starts = list(range(0, total_frames - chunk_size + 1, chunk_size))
    last_start = total_frames - chunk_size
    if starts[-1] != last_start:
        starts.append(last_start)

    chunks = []
    for start in starts:
        end = start + chunk_size
        chunk = {
            "video": example["video"][start:end],
            "points": example["points"][:, start:end],
            "occluded": example["occluded"][:, start:end],
            "chunk_start": start,
            "chunk_end": end,
        }
        chunks.append(chunk)
    return chunks


def load_and_chunk_kinetics(
    pickle_path: Path,
    chunk_size: int,
    max_chunks: int = -1,
) -> List[Dict[str, Any]]:
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        entries = list(data.values())
    else:
        entries = list(data)

    all_chunks: List[Dict[str, Any]] = []
    for example in entries:
        all_chunks.extend(chunk_example(example, chunk_size))
        if max_chunks > 0 and len(all_chunks) >= max_chunks:
            return all_chunks[:max_chunks]
    return all_chunks


def prepare_gmrw_sample(
    chunk: Dict[str, Any],
    input_size: int,
) -> Dict[str, np.ndarray]:
    frames = decode_video_frames(chunk["video"])
    frames = resize_video_square(frames, input_size)
    if frames.max() > 1.0:
        frames = frames / 255.0
    frames = frames.astype(np.float32)
    frames = frames * 2.0 - 1.0

    target_points = chunk["points"].astype(np.float32)
    target_points *= np.array([input_size, input_size], dtype=np.float32)
    occluded = chunk["occluded"].astype(bool)

    converted = sample_queries_strided(
        target_occluded=occluded,
        target_points=target_points,
        frames=frames,
    )
    return converted


class GMRWKineticsDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_size = getattr(cfg, "input_size")
        self.chunk_size = int(
            getattr(cfg.finetune_params, "gmrw_chunk_size", getattr(cfg, "seq_length", 16))
        )
        default_path = Path("/home/tekotan/Chewbacca_test/data-download/tapvid_kinetics/kinetics_10percent_sample.pkl")
        self.pickle_path = Path(
            getattr(cfg.finetune_params, "gmrw_kinetics_path", default_path)
        )
        self.max_samples = getattr(cfg.finetune_params, "gmrw_max_samples", -1)
        self.tracks_to_sample = int(
            getattr(cfg.finetune_params, "tracks_to_sample", 32)
        )

        self.samples = load_and_chunk_kinetics(
            self.pickle_path, self.chunk_size, max_chunks=self.max_samples
        )

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
            self._pad_or_trim(torch.from_numpy(query_points).float(), (self.tracks_to_sample, 3), fill_value=-1.0),
            self._pad_or_trim(torch.from_numpy(target_points).float(), (self.tracks_to_sample, target_points.shape[1], 2)),
            self._pad_or_trim(torch.from_numpy(occluded).float(), (self.tracks_to_sample, occluded.shape[1]), fill_value=1.0),
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
