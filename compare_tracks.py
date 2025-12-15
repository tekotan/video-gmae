#!/usr/bin/env python3
"""Match TAP-Vid point tracks between two PT files and visualize the comparison."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


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


def load_method_sample(path: Path, sample_idx: int) -> Tuple[TrackSample, TrackSample]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    video = data["video"][sample_idx]
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    video = ensure_uint8_video(video)

    tracks = data["points_pred"][sample_idx]
    if isinstance(tracks, torch.Tensor):
        tracks = tracks.cpu().numpy()
    # expected shape (P, T, 2); enforce dtype float32
    tracks = tracks.astype(np.float32)

    occ = data["occlusions_pred"][sample_idx]
    if isinstance(occ, torch.Tensor):
        occ = occ.cpu().numpy()
    occluded = occ.astype(bool)

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
        # (T, P, 2) -> (P, T, 2)
        tracks = np.transpose(tracks, (1, 0, 2))
    tracks = tracks.astype(np.float32)

    occ = sample["pred_occluded"]
    if isinstance(occ, torch.Tensor):
        occ = occ.cpu().numpy()
    # stored as (P, T)
    occluded = occ.astype(bool)

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


def match_tracks(
    reference: TrackSample,
    other: TrackSample,
    threshold: float,
) -> Dict[int, int]:
    """Greedy nearest-neighbour matching with a distance threshold."""
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


def resolve_template_path(template: str, batch: int) -> Path:
    """Resolve a PT template like create_supp_tracks.py does."""
    return Path(template.format(batch=batch, batch03=f"{batch:03d}"))


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
    """Draw tracks with optional visibility masks."""
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


def make_strip(rendered_frames: np.ndarray) -> np.ndarray:
    """Create a horizontal strip using every-other frame for readability."""
    subset = rendered_frames[::2]
    if len(subset) == 0:
        subset = rendered_frames
    return np.concatenate(subset, axis=1)


def draw_top_strip_border(
    image: np.ndarray,
    top_height: int,
    color: Tuple[int, int, int] = (0, 255, 0),
    fraction: float = 0.015,
) -> None:
    """Highlight the top strip of an example with a colored border."""
    thickness = max(1, int(round(top_height * fraction)))
    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1] - 1, top_height - 1),
        color,
        thickness,
        cv2.LINE_AA,
    )


def render_comparison_strip(
    method_pt_path: Path,
    method_sample_idx: int,
    gmrw_pt_path: Path,
    gmrw_sample_idx: int,
    show_gt: bool = True,
    matching_threshold: float = 7.5,
) -> np.ndarray:
    method_sample, gt_sample = load_method_sample(method_pt_path, method_sample_idx)
    gmrw_sample = load_gmrw_sample(gmrw_pt_path, gmrw_sample_idx)

    mapping_gmrw = match_tracks(method_sample, gmrw_sample, matching_threshold)
    if not mapping_gmrw:
        raise ValueError("No GMRW tracks matched within the threshold.")

    if show_gt:
        mapping_gt = match_tracks(method_sample, gt_sample, matching_threshold)
        keep_indices = sorted(set(mapping_gmrw.keys()) & set(mapping_gt.keys()))
        if not keep_indices:
            raise ValueError("No tracks matched across all three sources within threshold.")
    else:
        mapping_gt = {}
        keep_indices = sorted(mapping_gmrw.keys())

    method_sample = slice_tracks(method_sample, keep_indices)
    gmrw_sample = slice_tracks(gmrw_sample, [mapping_gmrw[i] for i in keep_indices])
    if show_gt:
        gt_sample = slice_tracks(gt_sample, [mapping_gt[i] for i in keep_indices])

    print(
        f"Matched {len(keep_indices)} tracks between the method and GMRW samples "
        f"for batch path {method_pt_path} idx {method_sample_idx}."
    )

    strips: List[np.ndarray] = []

    strip_samples: List[TrackSample] = []
    if show_gt:
        strip_samples.append(gt_sample)
    strip_samples.extend([method_sample, gmrw_sample])
    min_frames = min(sample.frames_rgb.shape[0] for sample in strip_samples)

    def render_strip(sample: TrackSample) -> np.ndarray:
        frames = sample.frames_rgb[:min_frames]
        tracks = sample.tracks_px[:, :min_frames]
        occluded = sample.occluded[:, :min_frames]
        rendered = overlay_point_tracks(
            frames, tracks, occluded
        )
        return make_strip(rendered)

    if show_gt:
        strips.append(render_strip(gt_sample))

    strips.append(render_strip(method_sample))
    strips.append(render_strip(gmrw_sample))

    max_height = max(strip.shape[0] for strip in strips)
    max_width = max(strip.shape[1] for strip in strips)

    def resize_to_shape(image: np.ndarray, height: int, width: int) -> np.ndarray:
        if image.shape[0] == height and image.shape[1] == width:
            return image
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    resized = [resize_to_shape(strip, max_height, max_width) for strip in strips]
    combined = np.concatenate(resized, axis=0)
    draw_top_strip_border(combined, top_height=resized[0].shape[0])
    return combined


def compare_pt_files(
    method_pt_path: Path,
    method_sample_idx: int,
    gmrw_pt_path: Path,
    gmrw_sample_idx: int,
    output_path: Path,
    show_gt: bool = True,
    matching_threshold: float = 7.5,
) -> None:
    combined = render_comparison_strip(
        method_pt_path=method_pt_path,
        method_sample_idx=method_sample_idx,
        gmrw_pt_path=gmrw_pt_path,
        gmrw_sample_idx=gmrw_sample_idx,
        show_gt=show_gt,
        matching_threshold=matching_threshold,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"Saved comparison strip to {output_path}")

    plt.figure(figsize=(16, 6))
    plt.imshow(combined)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def stack_example_strips(
    strips: Sequence[np.ndarray],
    border_fraction: float = 0.025,
    border_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    if not strips:
        raise ValueError("No comparison strips were provided.")

    max_width = max(strip.shape[1] for strip in strips)

    def pad_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
        """Pad the strip horizontally with the border color so widths match."""
        if image.shape[1] == target_width:
            return image
        total_pad = target_width - image.shape[1]
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        height, _, channels = image.shape
        pads: List[np.ndarray] = []
        if pad_left:
            pads.append(
                np.full((height, pad_left, channels), border_color, dtype=image.dtype)
            )
        pads.append(image)
        if pad_right:
            pads.append(
                np.full((height, pad_right, channels), border_color, dtype=image.dtype)
            )
        return np.concatenate(pads, axis=1)

    padded = [pad_to_width(strip, max_width) for strip in strips]
    layers: List[np.ndarray] = []
    for index, strip in enumerate(padded):
        layers.append(strip)
        if index < len(padded) - 1:
            border_height = max(1, int(round(strip.shape[0] * border_fraction)))
            border = np.full(
                (border_height, max_width, strip.shape[2]),
                border_color,
                dtype=strip.dtype,
            )
            layers.append(border)
    return np.concatenate(layers, axis=0)


def create_comparison_figure(
    method_template: str,
    gmrw_template: str,
    samples: Sequence[Tuple[int, int]],
    output_path: Path,
    *,
    show_gt: bool = True,
    matching_threshold: float = 7.5,
    border_fraction: float = 0.025,
) -> np.ndarray:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("No samples were provided for rendering.")

    strips: List[np.ndarray] = []
    for batch, sample_idx in sample_list:
        method_path = resolve_template_path(method_template, batch)
        gmrw_path = resolve_template_path(gmrw_template, batch)
        strip = render_comparison_strip(
            method_pt_path=method_path,
            method_sample_idx=sample_idx,
            gmrw_pt_path=gmrw_path,
            gmrw_sample_idx=sample_idx,
            show_gt=show_gt,
            matching_threshold=matching_threshold,
        )
        strips.append(strip)

    combined = stack_example_strips(
        strips,
        border_fraction=border_fraction,
        border_color=(255, 255, 255),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"Saved comparison figure with {len(strips)} examples to {output_path}")

    plt.figure(figsize=(16, 4 * len(strips)))
    plt.imshow(combined)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return combined


if __name__ == "__main__":
    # Configure the samples you want to stack into one figure.
    METHOD_PT_TEMPLATE = "/home/tekotan/Chewbacca_test/logs/vgmae_base_zeroshot_midtrain_v2/0/tests/gmrw_1_0_{batch}.pt"
    GMRW_PT_TEMPLATE = "/home/tekotan/gmrw/results/kinetics_full_run/batch_{batch03}.pt"
    SAMPLES: List[Tuple[int, int]] = [
        (56, 1),
        (77, 1),
        (39, 2),
        (27, 3),
        # (12, 2),
        # (16, 1),
        # (22, 1),
        # (26, 2),
    ]
    SHOW_GT = False
    MATCHING_THRESHOLD = 30
    OUTPUT_IMAGE = Path("comparison_strip_kinetics1.png")

    create_comparison_figure(
        method_template=METHOD_PT_TEMPLATE,
        gmrw_template=GMRW_PT_TEMPLATE,
        samples=SAMPLES,
        output_path=OUTPUT_IMAGE,
        show_gt=SHOW_GT,
        matching_threshold=MATCHING_THRESHOLD,
    )
    METHOD_PT_TEMPLATE = "/home/tekotan/Chewbacca_test/logs/vgmae_base_zeroshot_midtrain_v2/0/tests/gmrw_1_0_{batch}.pt"
    GMRW_PT_TEMPLATE = "/home/tekotan/gmrw/results/kinetics_full_run/batch_{batch03}.pt"
    SAMPLES: List[Tuple[int, int]] = [
        # (56, 1),
        # (77, 1),
        # (39, 2),
        # (27, 3),
        (12, 2),
        (16, 1),
        (22, 1),
        (26, 2),
    ]
    SHOW_GT = False
    MATCHING_THRESHOLD = 30
    OUTPUT_IMAGE = Path("comparison_strip_kinetics2.png")

    create_comparison_figure(
        method_template=METHOD_PT_TEMPLATE,
        gmrw_template=GMRW_PT_TEMPLATE,
        samples=SAMPLES,
        output_path=OUTPUT_IMAGE,
        show_gt=SHOW_GT,
        matching_threshold=MATCHING_THRESHOLD,
    )

    METHOD_PT_TEMPLATE = "/home/tekotan/Chewbacca_test/logs/vgmae_base_zeroshot_midtrain_v2/0/tests/gmrw-davis_1_0_{batch}.pt"
    GMRW_PT_TEMPLATE = "/home/tekotan/gmrw/results/davis_batches/batch_{batch03}.pt"
    SAMPLES: List[Tuple[int, int]] = [
        (0, 1), #
        (6, 1), #
        (9, 1), #
        (13, 1), #
        # (16, 0), #
        # (18, 2), #
        # (24, 1), #
        # (33, 1), #
    ]

    SHOW_GT = False
    MATCHING_THRESHOLD = 30
    OUTPUT_IMAGE = Path("comparison_strip_davis1.png")

    create_comparison_figure(
        method_template=METHOD_PT_TEMPLATE,
        gmrw_template=GMRW_PT_TEMPLATE,
        samples=SAMPLES,
        output_path=OUTPUT_IMAGE,
        show_gt=SHOW_GT,
        matching_threshold=MATCHING_THRESHOLD,
    )

    METHOD_PT_TEMPLATE = "/home/tekotan/Chewbacca_test/logs/vgmae_base_zeroshot_midtrain_v2/0/tests/gmrw-davis_1_0_{batch}.pt"
    GMRW_PT_TEMPLATE = "/home/tekotan/gmrw/results/davis_batches/batch_{batch03}.pt"
    SAMPLES: List[Tuple[int, int]] = [
        # (0, 1), #
        # (6, 1), #
        # (9, 1), #
        # (13, 1), #
        (16, 0), #
        (18, 2), #
        (24, 1), #
        (33, 1), #
    ]

    SHOW_GT = False
    MATCHING_THRESHOLD = 30
    OUTPUT_IMAGE = Path("comparison_strip_davis2.png")

    create_comparison_figure(
        method_template=METHOD_PT_TEMPLATE,
        gmrw_template=GMRW_PT_TEMPLATE,
        samples=SAMPLES,
        output_path=OUTPUT_IMAGE,
        show_gt=SHOW_GT,
        matching_threshold=MATCHING_THRESHOLD,
    )
