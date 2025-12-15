#!/usr/bin/env python3
"""Create supplemental comparison grids from TAP-Vid PT outputs."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm


@dataclasses.dataclass
class TrackSample:
    frames_rgb: np.ndarray  # (T, H, W, 3) uint8
    tracks_px: np.ndarray  # (P, T, 2) float32 pixels
    occluded: np.ndarray  # (P, T) bool


def ensure_uint8_video(video: np.ndarray) -> np.ndarray:
    video = np.asarray(video)
    if video.dtype != np.uint8:
        video = np.clip(video, 0.0, 255.0)
        if video.max() <= 1.0:
            video = video * 255.0
        video = video.astype(np.uint8)
    return video


def rescale_tracks_if_normalized(tracks: np.ndarray, video: np.ndarray) -> np.ndarray:
    """If tracks lie in [0, 1], expand them to the pixel grid using the video size."""
    if tracks.size == 0:
        return tracks
    min_val = float(tracks.min())
    max_val = float(tracks.max())
    if 0.0 <= min_val and max_val <= 1.0:
        height = video.shape[1] - 1
        width = video.shape[2] - 1
        tracks = tracks.copy()
        tracks[..., 0] *= width
        tracks[..., 1] *= height
    return tracks


def occlusion_mask_to_bool(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert numeric occlusion probabilities to boolean visibility."""
    mask = np.asarray(mask)
    if mask.dtype == bool:
        return mask
    if np.isfinite(mask).all() and mask.min() >= 0.0 and mask.max() <= 1.0:
        return mask >= threshold
    return mask.astype(bool)


def load_method_sample(path: Path, sample_idx: int) -> Tuple[TrackSample, TrackSample]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    video = data["video"][sample_idx]
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    video = ensure_uint8_video(video)

    tracks = data["points_pred"][sample_idx]
    if isinstance(tracks, torch.Tensor):
        tracks = tracks.cpu().numpy()
    tracks = tracks.astype(np.float32)
    tracks = rescale_tracks_if_normalized(tracks, video)

    occ = data["occlusions_pred"][sample_idx]
    if isinstance(occ, torch.Tensor):
        occ = occ.cpu().numpy()
    occluded = occlusion_mask_to_bool(occ)

    gt_tracks = data.get("points_gt")
    gt_occluded = data.get("occlusions_gt")
    if gt_tracks is None or gt_occluded is None:
        raise KeyError("Ground-truth tracks/occlusions missing in method PT file.")
    gt_tracks = gt_tracks[sample_idx]
    if isinstance(gt_tracks, torch.Tensor):
        gt_tracks = gt_tracks.cpu().numpy()
    gt_tracks = gt_tracks.astype(np.float32)
    gt_tracks[..., 0] *= (video.shape[2] - 1)
    gt_tracks[..., 1] *= (video.shape[1] - 1)

    gt_occ = gt_occluded[sample_idx]
    if isinstance(gt_occ, torch.Tensor):
        gt_occ = gt_occ.cpu().numpy()
    gt_occ = gt_occ.astype(bool)

    method_sample = TrackSample(
        frames_rgb=video,
        tracks_px=tracks,
        occluded=occluded,
    )
    gt_sample = TrackSample(
        frames_rgb=video,
        tracks_px=gt_tracks,
        occluded=gt_occ,
    )
    return method_sample, gt_sample


def load_gmrw_sample(path: Path, sample_idx: int) -> TrackSample:
    batch_payload: List[Dict] = torch.load(path, map_location="cpu", weights_only=False)
    sample = batch_payload[sample_idx]

    video = sample["video_pixels"]
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    if video.ndim != 4:
        raise ValueError("Expected video_pixels with 4 dimensions")
    if video.shape[-1] == 3:
        pass
    elif video.shape[1] == 3:
        video = np.transpose(video, (0, 2, 3, 1))
    else:
        raise ValueError(f"Unrecognized video tensor shape {video.shape}")
    video = ensure_uint8_video(video)

    tracks = sample["predicted_points"]
    if isinstance(tracks, torch.Tensor):
        tracks = tracks.cpu().numpy()
    if tracks.ndim != 3:
        raise ValueError("Expected predicted_points with 3 dims")
    if tracks.shape[0] == video.shape[0]:
        tracks = np.transpose(tracks, (1, 0, 2))
    tracks = tracks.astype(np.float32)
    tracks = rescale_tracks_if_normalized(tracks, video)

    occ = sample["pred_occluded"]
    if isinstance(occ, torch.Tensor):
        occ = occ.cpu().numpy()
    occluded = occlusion_mask_to_bool(occ)

    query_points = sample.get("query_points")
    if query_points is not None:
        if isinstance(query_points, torch.Tensor):
            query_points = query_points.cpu().numpy()
        query_times = np.round(query_points[:, 0]).astype(int)
        mask = query_times == 0
        if mask.any():
            tracks = tracks[mask]
            occluded = occluded[mask]
        else:
            order = np.argsort(query_times)
            tracks = tracks[order]
            occluded = occluded[order]

    return TrackSample(frames_rgb=video, tracks_px=tracks, occluded=occluded)


def match_tracks(reference: TrackSample, other: TrackSample, threshold: float) -> Dict[int, int]:
    ref_first = reference.tracks_px[:, 0]
    other_first = other.tracks_px[:, 0]
    available = list(range(other_first.shape[0]))
    mapping: Dict[int, int] = {}

    for ref_idx, ref_pt in enumerate(ref_first):
        if not available:
            break
        candidates = other_first[available]
        dists = np.linalg.norm(candidates - ref_pt[None], axis=1)
        best_pos = int(np.argmin(dists))
        best_idx = available[best_pos]
        if dists[best_pos] <= threshold:
            mapping[ref_idx] = best_idx
            available.pop(best_pos)
    return mapping


def slice_tracks(sample: TrackSample, indices: List[int]) -> TrackSample:
    tracks = sample.tracks_px[indices]
    occluded = sample.occluded[indices]
    return TrackSample(sample.frames_rgb, tracks, occluded)


def overlay_point_tracks(
    frames_rgb: Sequence[np.ndarray],
    tracks_px: np.ndarray,
    occluded: np.ndarray | None = None,
    *,
    radius: int = 2,
    thickness: int = 1,
    tail: bool = True,
    colormap: int = cv2.COLORMAP_RAINBOW,
) -> np.ndarray:
    frames_rgb = [np.asarray(f) for f in frames_rgb]
    T = len(frames_rgb)
    P = tracks_px.shape[0]

    if occluded is None:
        occluded = np.zeros((P, T), dtype=bool)
    else:
        occluded = occluded.astype(bool)

    colours = (
        cv2.applyColorMap(np.linspace(0, 255, P, dtype=np.uint8), colormap)[:, 0, ::-1]
        .astype(int)
        .tolist()
    )

    rendered: List[np.ndarray] = []
    for t in range(T):
        frame = frames_rgb[t].copy()
        for p in range(P):
            if occluded[p, t]:
                continue
            if tail:
                vis_idx = np.where(~occluded[p, : t + 1])[0]
                pts = tracks_px[p, vis_idx].astype(int)
                for i in range(1, len(pts)):
                    cv2.line(
                        frame,
                        tuple(pts[i - 1]),
                        tuple(pts[i]),
                        colours[p],
                        thickness,
                        cv2.LINE_AA,
                    )
            x, y = tracks_px[p, t].astype(int)
            cv2.circle(frame, (x, y), radius, colours[p], -1, cv2.LINE_AA)
        rendered.append(frame)
    return np.stack(rendered, axis=0)


def resolve_template_path(template: str, batch: int) -> Path:
    return Path(template.format(batch=batch, batch03=f"{batch:03d}"))


def determine_common_length(lengths: Sequence[int], max_frames: int | None) -> int:
    if not lengths:
        raise ValueError("No videos were loaded.")
    frames = min(lengths)
    if max_frames is not None:
        frames = min(frames, max_frames)
    if frames <= 0:
        raise ValueError("Resolved zero frames for rendering.")
    return frames


def render_tracks(sample: TrackSample, num_frames: int) -> np.ndarray:
    frames = sample.frames_rgb[:num_frames]
    tracks = sample.tracks_px[:, :num_frames]
    occluded = sample.occluded[:, :num_frames]
    return overlay_point_tracks(frames, tracks, occluded)


def pad_video(video: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    height, width = video.shape[1:3]
    if height > target_height or width > target_width:
        raise ValueError(
            f"Cannot pad video of shape {(height, width)} to {(target_height, target_width)}."
        )
    pad_top = (target_height - height) // 2
    pad_bottom = target_height - height - pad_top
    pad_left = (target_width - width) // 2
    pad_right = target_width - width - pad_left
    if pad_top == pad_bottom == pad_left == pad_right == 0:
        return video
    return np.pad(
        video,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def resize_video(video: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    if video.shape[1] == target_height and video.shape[2] == target_width:
        return video
    resized = [
        cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        for frame in video
    ]
    return np.stack(resized, axis=0)


def make_grid_video(videos: Sequence[np.ndarray], rows: int, cols: int) -> np.ndarray:
    if len(videos) != rows * cols:
        raise ValueError(f"Expected {rows * cols} videos, received {len(videos)}.")

    lengths = [video.shape[0] for video in videos]
    frames = min(lengths)
    videos = [video[:frames] for video in videos]

    max_height = max(video.shape[1] for video in videos)
    max_width = max(video.shape[2] for video in videos)
    padded = [pad_video(video, max_height, max_width) for video in videos]

    grid_frames: List[np.ndarray] = []
    for t in range(frames):
        rows_frames: List[np.ndarray] = []
        for r in range(rows):
            start = r * cols
            row_frames = [padded[start + c][t] for c in range(cols)]
            rows_frames.append(np.concatenate(row_frames, axis=1))
        grid_frames.append(np.concatenate(rows_frames, axis=0))
    return np.stack(grid_frames, axis=0)


def draw_border(frames: np.ndarray, fraction: float = 0.025, color: Tuple[int, int, int] = (0, 255, 0)) -> None:
    height, width = frames.shape[1:3]
    thickness = max(1, int(round(min(height, width) * fraction)))
    for frame in frames:
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), color, thickness, cv2.LINE_AA)


def create_three_by_three_grid(
    pt_template: str,
    samples: Sequence[Tuple[int, int]],
    output: Path,
    fps: float,
    max_frames: int | None,
    upsample_factor: int = 2,
) -> None:
    sample_list = list(samples)
    if len(sample_list) != 9:
        raise ValueError("The 3x3 grid requires exactly 9 samples.")

    rendered_videos: List[np.ndarray] = []
    lengths: List[int] = []
    loaded_samples: List[TrackSample] = []

    for batch, sample_idx in tqdm(sample_list, desc="Loading 3x3 samples"):
        pt_path = resolve_template_path(pt_template, batch)
        method_sample, _ = load_method_sample(pt_path, sample_idx)
        loaded_samples.append(method_sample)
        lengths.append(method_sample.frames_rgb.shape[0])

    frames = determine_common_length(lengths, max_frames)
    for sample in tqdm(loaded_samples, desc="Rendering 3x3 samples"):
        rendered_videos.append(render_tracks(sample, frames))

    grid_video = make_grid_video(rendered_videos, rows=3, cols=3)
    write_video(grid_video, output, fps, upsample_factor=upsample_factor)


def create_comparison_grid(
    method_template: str,
    gmrw_template: str,
    samples: Sequence[Tuple[int, int]],
    output: Path,
    fps: float,
    max_frames: int | None,
    matching_threshold: float,
    upsample_factor: int = 2,
) -> None:
    sample_list = list(samples)
    if len(sample_list) != 8:
        raise ValueError("The comparison grid requires exactly 8 samples.")

    paired_samples: List[Tuple[TrackSample, TrackSample]] = []
    pair_lengths: List[int] = []

    for batch, sample_idx in tqdm(sample_list, desc="Loading comparison samples"):
        method_path = resolve_template_path(method_template, batch)
        gmrw_path = resolve_template_path(gmrw_template, batch)

        method_sample, _ = load_method_sample(method_path, sample_idx)
        gmrw_sample = load_gmrw_sample(gmrw_path, sample_idx)

        mapping = match_tracks(method_sample, gmrw_sample, matching_threshold)
        if not mapping:
            raise ValueError(
                f"No GMRW tracks matched the method sample for batch {batch}, idx {sample_idx}."
            )

        keep_indices = sorted(mapping.keys())
        method_slice = slice_tracks(method_sample, keep_indices)
        gmrw_slice = slice_tracks(gmrw_sample, [mapping[i] for i in keep_indices])

        pair_lengths.append(
            min(method_slice.frames_rgb.shape[0], gmrw_slice.frames_rgb.shape[0])
        )
        paired_samples.append((method_slice, gmrw_slice))

    frames = determine_common_length(pair_lengths, max_frames)
    example_videos: List[np.ndarray] = []

    for method_slice, gmrw_slice in tqdm(paired_samples, desc="Rendering comparison pairs"):
        method_frames = render_tracks(method_slice, frames)
        gmrw_frames = render_tracks(gmrw_slice, frames)

        target_height = gmrw_frames.shape[1]
        target_width = gmrw_frames.shape[2]
        method_frames = resize_video(method_frames, target_height, target_width)
        gmrw_frames = resize_video(gmrw_frames, target_height, target_width)

        draw_border(method_frames, fraction=0.025)
        combined = np.concatenate([method_frames, gmrw_frames], axis=2)
        example_videos.append(combined)

    grid_video = make_grid_video(example_videos, rows=4, cols=2)
    write_video(grid_video, output, fps, upsample_factor=upsample_factor)


def write_video(
    frames: np.ndarray,
    output_path: Path,
    fps: float,
    upsample_factor: int = 2,
) -> None:
    if frames.dtype != np.uint8:
        frames = ensure_uint8_video(frames)

    if upsample_factor != 1:
        height, width = frames.shape[1:3]
        target_size = (width * upsample_factor, height * upsample_factor)
        resized_frames = [
            cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR) for frame in frames
        ]
        frames = np.stack(resized_frames, axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_videofile(str(output_path), codec="libx264", audio=False)


def run_from_config() -> None:
    """Edit the parameters below (like compare_tracks.py) and set the toggles to run."""

    # RUN_GRID3X3 = True
    # GRID_PT_TEMPLATE = "/home/tekotan/Chewbacca_test/logs/vgmae_large_seq16_fullfinetune_point_new/0/tests/eval_1_0_{batch}.pt"
    # GRID_SAMPLES: List[Tuple[int, int]] = [
    #     # new
    #     (10, 5),
    #     (12, 2),
    #     (30, 0),
    #     (34, 5),
    #     (38, 1),
    #     (46, 6),
    #     (53, 3),
    #     (56, 6),
    #     (65, 2),
    # ]

    RUN_GRID3X3 = True
    GRID_PT_TEMPLATE = "/home/tekotan/Chewbacca_test/logs/vgmae_large_seq16_fullfinetune_point_new/0/tests-davis/eval_1_0_{batch}.pt"
    GRID_SAMPLES: List[Tuple[int, int]] = [
        # new
        (7, 2),
        (10, 2),
        (13, 3),
        (17, 4),
        (22, 7),
        (25, 6),
        (31, 3),
        (34, 0),
        (41, 7),
    ]

    # RUN_GRID3X3 = False
    # GRID_PT_TEMPLATE = "/home/tekotan/Chewbacca_test/logs/vgmae_base_zeroshot_midtrain_v2/0/tests/gmrw_1_0_{batch}.pt"
    # GRID_SAMPLES: List[Tuple[int, int]] = [
    #     # new
    #     (14, 0),
    #     (16, 2),
    #     (24, 3),
    #     (49, 2),
    #     (78, 3),
    #     (78, 3),
    #     (84, 1),
    #     (91, 1),
    #     (94, 0),
    # ]

    # RUN_GRID3X3 = True
    # GRID_PT_TEMPLATE = "/home/tekotan/Chewbacca_test/logs/vgmae_base_zeroshot_midtrain_v2/0/tests/gmrw-davis_1_0_{batch}.pt"
    # GRID_SAMPLES: List[Tuple[int, int]] = [
    #     # new
    #     (17, 0),
    #     (10, 2),
    #     (8, 1),
    #     (21, 0),
    #     (23, 3),
    #     (24, 2),
    #     (30, 0),
    #     (31, 2),
    #     (33, 0),
    # ]
    GRID_OUTPUT = Path("finetune-davis2.mp4")
    GRID_FPS = 10.0
    GRID_MAX_FRAMES = None
    GRID_UPSAMPLE = 4

    RUN_COMPARISON_GRID = False
    METHOD_PT_TEMPLATE = "/home/tekotan/Chewbacca_test/logs/vgmae_base_zeroshot_midtrain_v2/0/tests/gmrw_1_0_{batch}.pt"
    GMRW_PT_TEMPLATE = "/home/tekotan/gmrw/results/kinetics_full_run/batch_{batch03}.pt"
    COMPARISON_SAMPLES: List[Tuple[int, int]] = [
        (56, 1),
        (77, 1),
        (39, 2),
        (27, 3),
        (12, 2),
        (16, 1),
        (22, 1),
        (26, 2),
    ]
    # METHOD_PT_TEMPLATE = "/home/tekotan/Chewbacca_test/logs/vgmae_base_zeroshot_midtrain_v2/0/tests/gmrw-davis_1_0_{batch}.pt"
    # GMRW_PT_TEMPLATE = "/home/tekotan/gmrw/results/davis_batches/batch_{batch03}.pt"
    # COMPARISON_SAMPLES: List[Tuple[int, int]] = [
    #     (0, 1), #
    #     (6, 1), #
    #     (9, 1), #
    #     (13, 1), #
    #     (16, 0), #
    #     (18, 2), #
    #     (24, 1), #
    #     (33, 1), #
    # ]

    COMPARISON_OUTPUT = Path("comparison_grid1.mp4")
    COMPARISON_FPS = 10.0
    COMPARISON_MAX_FRAMES = None
    MATCHING_THRESHOLD = 30.0
    COMPARISON_UPSAMPLE = 4

    if RUN_GRID3X3:
        create_three_by_three_grid(
            pt_template=GRID_PT_TEMPLATE,
            samples=GRID_SAMPLES,
            output=GRID_OUTPUT,
            fps=GRID_FPS,
            max_frames=GRID_MAX_FRAMES,
            upsample_factor=GRID_UPSAMPLE,
        )

    if RUN_COMPARISON_GRID:
        create_comparison_grid(
            method_template=METHOD_PT_TEMPLATE,
            gmrw_template=GMRW_PT_TEMPLATE,
            samples=COMPARISON_SAMPLES,
            output=COMPARISON_OUTPUT,
            fps=COMPARISON_FPS,
            max_frames=COMPARISON_MAX_FRAMES,
            matching_threshold=MATCHING_THRESHOLD,
            upsample_factor=COMPARISON_UPSAMPLE,
        )


if __name__ == "__main__":
    run_from_config()
