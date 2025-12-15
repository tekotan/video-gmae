#!/usr/bin/env python3
"""Export TAP-Vid test pickles to mp4 files.

This script converts the TAP-Vid DAVIS and Kinetics pickles into standalone
mp4s so they can be consumed by the VideoDataset pipeline (and labelled via
`zeroshot-test.py`). Note that the Kinetics pickles are very large; consider
using ``--limit`` while testing to avoid loading the full file.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAVIS_PKL = REPO_ROOT / "data-download" / "tapvid_davis" / "tapvid_davis.pkl"
DEFAULT_KINETICS_PKL = (
    REPO_ROOT / "data-download" / "tapvid_kinetics" / "kinetics_10percent_sample.pkl"
)
DEFAULT_DAVIS_OUT = REPO_ROOT / "data-download" / "zeroshot-test" / "davis"
DEFAULT_KINETICS_OUT = REPO_ROOT / "data-download" / "zeroshot-test" / "kinetics"


def _sanitize_name(raw: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in raw)
    return safe or "video"


def _open_writer(output_path: Path, fps: float):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        output_path,
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        output_params=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
    )


def _write_video(frames: np.ndarray, output_path: Path, fps: float) -> None:
    with _open_writer(output_path, fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def _iter_davis_entries(pkl_path: Path) -> Iterator[Tuple[str, dict]]:
    with pkl_path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"DAVIS pickle at {pkl_path} must be a dict.")
    for name, example in data.items():
        yield str(name), example


def _iter_kinetics_entries(pkl_path: Path) -> Iterator[Tuple[str, dict]]:
    with pkl_path.open("rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict):
        for name, example in data.items():
            yield str(name), example
        return
    if isinstance(data, list):
        for idx, example in enumerate(data):
            name = example.get("_video_name") if isinstance(example, dict) else None
            yield name or f"kinetics_{idx:06d}", example
        return
    raise ValueError(f"Unsupported structure in kinetics pickle {pkl_path}")


def export_split(
    entries: Iterable[Tuple[str, dict]],
    output_dir: Path,
    fps: float,
    *,
    limit: int | None,
    overwrite: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, example in entries:
        if limit is not None and written >= limit:
            break
        video = example.get("video")
        if video is None:
            logging.warning("Skipping %s because no 'video' key was found.", name)
            continue
        video_path = output_dir / f"{_sanitize_name(name)}.mp4"
        if video_path.exists() and not overwrite:
            logging.debug("Skipping existing video %s", video_path)
        else:
            logging.info("Writing %s", video_path)
            _write_video(video, video_path, fps)
        written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TAP-Vid test pickles into mp4 files."
    )
    parser.add_argument(
        "--davis-pkl",
        type=Path,
        default=DEFAULT_DAVIS_PKL,
        help="Path to tapvid_davis.pkl.",
    )
    parser.add_argument(
        "--kinetics-pkl",
        type=Path,
        default=DEFAULT_KINETICS_PKL,
        help="Path to kinetics_10percent_sample.pkl (or shard).",
    )
    parser.add_argument(
        "--davis-output",
        type=Path,
        default=DEFAULT_DAVIS_OUT,
        help="Destination directory for DAVIS test mp4s.",
    )
    parser.add_argument(
        "--kinetics-output",
        type=Path,
        default=DEFAULT_KINETICS_OUT,
        help="Destination directory for Kinetics test mp4s.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Frame rate used when encoding the videos.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of videos to export per split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mp4 files.",
    )
    parser.add_argument(
        "--skip-davis",
        action="store_true",
        help="Skip DAVIS conversion.",
    )
    parser.add_argument(
        "--skip-kinetics",
        action="store_true",
        help="Skip Kinetics conversion.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.skip_davis and args.skip_kinetics:
        logging.error("Both DAVIS and Kinetics conversions are skipped; nothing to do.")
        return 1

    if not args.skip_davis:
        count = export_split(
            _iter_davis_entries(args.davis_pkl),
            args.davis_output,
            args.fps,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        logging.info("Exported %d DAVIS videos to %s", count, args.davis_output)

    if not args.skip_kinetics:
        count = export_split(
            _iter_kinetics_entries(args.kinetics_pkl),
            args.kinetics_output,
            args.fps,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        logging.info("Exported %d Kinetics videos to %s", count, args.kinetics_output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
