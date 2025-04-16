import pickle
import torch
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms

import numpy as np
import os

def random_crop_4d(
    frames_4d: torch.Tensor,
    points_3d: np.ndarray,
    occlusions: np.ndarray,
    crop_size: int
):
    """
    Randomly crop a [length, 3, H, W] tensor of frames to [length, 3, crop_size, crop_size]
    and shift/clamp the corresponding points [N, length, 2].

    Args:
        frames_4d:  Tensor of shape (3, length, H, W)
        points_3d:  Numpy array of shape (N, length, 2)
        crop_size:  Desired output spatial size
    Returns:
        cropped_frames:  (3, length, crop_size, crop_size)
        adjusted_points: (N, length, 2) - shifted/clamped coordinates
    """
    _, _, H, W = frames_4d.shape

    # If the original frames are smaller than crop_size, you have options:
    # (a) skip cropping,
    # (b) center-crop or
    # (c) upsample to meet the size requirement, etc.
    # Below we just do a guard so we don't break if H < crop_size or W < crop_size.
    if H <= crop_size or W <= crop_size:
        # E.g., do a center-crop as fallback (or just return frames_4d, points_3d)
        y_off = max(0, (H - crop_size) // 2)
        x_off = max(0, (W - crop_size) // 2)
    else:
        # Random offsets in [0, H - crop_size], [0, W - crop_size]
        y_off = np.random.randint(0, H - crop_size + 1)
        x_off = np.random.randint(0, W - crop_size + 1)

    # 1) Crop the frames
    cropped_frames = frames_4d[:, :, y_off : y_off + crop_size, x_off : x_off + crop_size]

    # 2) Shift and clamp points
    adjusted_points = points_3d.copy()  # shape (N, length, 2)
    scale_factor = np.array([W, H])[np.newaxis, np.newaxis, :]
    adjusted_points = adjusted_points * scale_factor

    # Subtract the offset
    adjusted_points[:, :, 0] -= x_off
    adjusted_points[:, :, 1] -= y_off

    # Now clamp them to [0, crop_size-1]
    external_x = (adjusted_points[:, :, 0] < 0) | (adjusted_points[:, :, 0] >= crop_size)
    external_y = (adjusted_points[:, :, 1] < 0) | (adjusted_points[:, :, 1] >= crop_size)

    adjusted_occlusions = occlusions.copy()
    adjusted_occlusions[external_x | external_y] = 1.0

    adjusted_points[:, :, 0] = np.clip(adjusted_points[:, :, 0], 0, crop_size)
    adjusted_points[:, :, 1] = np.clip(adjusted_points[:, :, 1], 0, crop_size)

    adjusted_points = adjusted_points / crop_size

    return cropped_frames, adjusted_points, adjusted_occlusions


class PointTrackingEvalDataset(IterableDataset):
    def __init__(
        self,
        pickle_files,  # List[str]
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.bin_size = min(24, self.cfg.seq_length)
        self.all_items = []

        for pkl_file in pickle_files:
            if not os.path.isfile(pkl_file):
                raise FileNotFoundError(f"Pickle file not found: {pkl_file}")

            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict):
                # DAVIS-like => {video_name: content_dict, ...}
                for video_name, content in data.items():
                    self.all_items.append((video_name, content))
            elif isinstance(data, list):
                # RGB-Stacking => [ content_dict, ... ]
                for content in data:
                    self.all_items.append((None, content))
            else:
                raise ValueError(
                    f"Unsupported pickle structure in {pkl_file}. Must be dict or list."
                )
        nearest_power_of_two = 2 ** int(np.ceil(np.log2(self.cfg.input_size)))
        self.resize_transform = transforms.Resize(nearest_power_of_two)
        # Debug break if needed
        print("Loaded {} video items.".format(len(self.all_items)))

        # debugging:
        if self.cfg.finetune_params.test:
            self.all_items = self.all_items[7:8] * 100

    def __iter__(self):
        for (video_name, content) in self.all_items:
            frames_np = content["video"]   # shape (T, H, W, 3), uint8
            points_np = content["points"]  # shape (N, T, 2), float32
            occluded_np = content["occluded"].astype(float)  # shape (N, T), float

            T = frames_np.shape[0]
            N = points_np.shape[0]
            H = frames_np.shape[1]
            W = frames_np.shape[2]

            # 2) Convert frames to float [0..1] and normalize
            frames_np = frames_np / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            frames_np = (frames_np - mean) / std

            # Convert to torch (T, H, W, 3)
            frames_torch = torch.from_numpy(frames_np)

            # 3) For each bin, yield a chunk
            start_idx = 0
            while start_idx < T:
                end_idx = start_idx + self.bin_size
                if end_idx > T:
                    # break or skip if partial bins are not desired
                    # For eval, you can decide to allow partial
                    break

                length = end_idx - start_idx
                if length == 0:
                    break

                # Slice frames: shape (length, H, W, 3)
                frames_chunk = frames_torch[start_idx:end_idx]  # (length, H, W, 3)
                # Now permute to (length, 3, H, W)
                frames_chunk = frames_chunk.permute(3, 0, 1, 2).contiguous()
                frames_chunk = self.resize_transform(frames_chunk)

                # Slice points in time => (N, length, 2)
                points_chunk_np = points_np[:, start_idx:end_idx, :]
                occluded_chunk_np = occluded_np[:, start_idx:end_idx]
                # -------------------------------------------------
                # Perform random cropping to cfg.input_size here
                # -------------------------------------------------
                frames_chunk, points_chunk_np, occluded_chunk_np = random_crop_4d(
                    frames_chunk,         # shape (3, length, H, W)
                    points_chunk_np,      # shape (N, length, 2)
                    occluded_chunk_np,    # shape (N, length)
                    self.cfg.input_size
                )
                # frames_chunk:  (length, 3, crop_size, crop_size)
                # points_chunk_np: (N, length, 2) adjusted

                # Remove points not visible in first frame
                visible_in_first = ~(occluded_chunk_np[:, 0].astype(bool))
                points_chunk_np = points_chunk_np[visible_in_first]
                occluded_chunk_np = occluded_chunk_np[visible_in_first]
                N = np.sum(visible_in_first)

                # 4) target_points => torch, shape (N, length, 2)
                target_points_torch = torch.from_numpy(points_chunk_np)

                # 5) query_points => from time=0 in THIS chunk => shape (N, 2)
                # Convert (x, y) -> (0, y, x)
                first_time_np = points_chunk_np[:, 0, :]  # (N, 2)
                zeros = np.zeros((N, 1), dtype=first_time_np.dtype)
                y_vals = first_time_np[:, 1:2]
                x_vals = first_time_np[:, 0:1]
                query_np = np.concatenate([zeros, y_vals, x_vals], axis=1)  # (N, 3)
                query_points_torch = torch.from_numpy(query_np)

                occluded_points_torch = torch.from_numpy(occluded_chunk_np)

                # 6) Pad if N < vocab_size
                if N < self.cfg.finetune_params.tracks_to_sample:
                    diff = self.cfg.finetune_params.tracks_to_sample - N
                    # target_points => pad with 0
                    pad_target = torch.zeros(diff, length, 2, dtype=target_points_torch.dtype)
                    target_points_torch = torch.cat([target_points_torch, pad_target], dim=0)
                    # query_points => pad with -1
                    pad_query = -1.0 * torch.ones(diff, 3, dtype=query_points_torch.dtype)
                    query_points_torch = torch.cat([query_points_torch, pad_query], dim=0)
                    # occluded_points => pad with 1
                    pad_occluded = torch.ones(diff, length, dtype=occluded_points_torch.dtype)
                    occluded_points_torch = torch.cat([occluded_points_torch, pad_occluded], dim=0)
                if N > self.cfg.finetune_params.tracks_to_sample:
                    random_idx = np.random.choice(N, self.cfg.finetune_params.tracks_to_sample, replace=False)
                    target_points_torch = target_points_torch[random_idx]
                    query_points_torch = query_points_torch[random_idx]
                    occluded_points_torch = occluded_points_torch[random_idx]
                # 7) Yield
                yield (frames_chunk.float(), query_points_torch.float(), target_points_torch.float(), occluded_points_torch.float())

                start_idx += self.bin_size
