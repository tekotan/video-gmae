#!/usr/bin/env python3
"""Download Kubric MOVi_E and DAVIS videos as mp4 files.

This script fetches the Kubric MOVi_E/256x256 dataset via TensorFlow Datasets
and exports the raw RGB frames as mp4 files. It also downloads DAVIS archives,
converts extracted image sequences to mp4, and stores the results in the
expected zeroshot mid-training directory structure.

Examples
--------
Download both datasets into the default locations::

    python scripts/download_midtraining_videos.py

Only download DAVIS videos and keep the extracted frame folders::

    python scripts/download_midtraining_videos.py \\
        --skip-kubric \\
        --davis-splits 2017-trainval 2017-test-dev \\
        --davis-keep-frames
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import imageio
import numpy as np
import tensorflow_datasets as tfds

try:
    import tensorflow as tf  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError(
        "TensorFlow is required for loading the Kubric MOVi_E dataset. "
        "Install it with `pip install tensorflow` before running this script."
    ) from exc


# Disable GPU usage to avoid unnecessary TF warnings when only using the CPU.
try:  # pragma: no cover - depends on system GPU availability
    tf.config.set_visible_devices([], "GPU")
except Exception:  # pylint: disable=broad-except
    pass


DEFAULT_KUBRIC_OUTPUT = Path("data-download/zeroshot-midtraining/kubric")
DEFAULT_DAVIS_OUTPUT = Path("data-download/zeroshot-midtraining/davis")


DAVIS_URLS = {
    # "2016-trainval": "https://data.vision.ee.ethz.ch/cvl/DAVIS/DAVIS-2016-trainval-480p.zip",
    "2017-trainval": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-480p.zip",
    # "2017-test-dev": "https://data.vision.ee.ethz.ch/cvl/DAVIS/DAVIS-2017-test-dev-480p.zip",
    # "2017-test-challenge": "https://data.vision.ee.ethz.ch/cvl/DAVIS/DAVIS-2017-test-challenge-480p.zip",
}


def _sanitize_name(raw: bytes | str) -> str:
    """Return a filesystem-safe identifier derived from the provided name."""
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return safe or "video"


def _open_ffmpeg_writer(output_path: Path, fps: float):
    """Return an imageio ffmpeg writer with sane defaults for odd frame sizes."""
    return imageio.get_writer(
        output_path,
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        output_params=[
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # ensure even frame dimensions
        ],
    )


def _write_video_mp4(frames: np.ndarray, output_path: Path, fps: float) -> None:
    """Write a sequence of frames to `output_path` as an mp4 file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # imageio automatically downloads the ffmpeg binary on first use.
    with _open_ffmpeg_writer(
        output_path,
        fps,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def download_movi_e(
    output_dir: Path,
    *,
    splits: Iterable[str],
    tfds_data_dir: Optional[str],
    limit: Optional[int],
    overwrite: bool,
    fps: float,
) -> None:
    """Download MOVi_E videos and save them as mp4 files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tfds_dir = tfds_data_dir

    builder_name = "kubric:movi_e/256x256"
    read_config = tfds.ReadConfig(try_autocache=False)

    for split in splits:
        logging.info("Processing MOVi_E split '%s'", split)
        load_kwargs = {
            "split": split,
            "shuffle_files": False,
            "read_config": read_config,
        }
        if tfds_dir is not None:
            if "://" not in tfds_dir:
                Path(tfds_dir).mkdir(parents=True, exist_ok=True)
            load_kwargs["data_dir"] = tfds_dir

        ds = tfds.load(builder_name, **load_kwargs)
        counter = 0
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for example in tfds.as_numpy(ds):
            video = example["video"]
            meta = example["metadata"]
            video_name = _sanitize_name(meta["video_name"])
            video_path = split_dir / f"{video_name}.mp4"

            if video_path.exists() and not overwrite:
                logging.debug("Skipping existing MOVi_E video %s", video_path)
            else:
                logging.info("Writing MOVi_E video %s", video_path)
                _write_video_mp4(video, video_path, fps=fps)

            counter += 1
            if limit is not None and counter >= limit:
                break

        logging.info("Finished split '%s' with %d videos.", split, counter)


def _download_file(url: str, destination: Path, overwrite: bool) -> None:
    """Download `url` into `destination` if needed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        logging.info("Archive already present at %s; skipping download.", destination)
        return

    logging.info("Downloading %s -> %s", url, destination)
    with urllib.request.urlopen(url) as response, open(destination, "wb") as output:
        shutil.copyfileobj(response, output)


def _find_frames_root(extracted_root: Path) -> Path:
    """Locate the DAVIS `JPEGImages/480p` directory within the extracted tree."""
    candidates = list(extracted_root.rglob("JPEGImages/480p"))
    if not candidates:
        raise RuntimeError(
            "Could not locate 'JPEGImages/480p' in the extracted DAVIS archive."
        )
    return candidates[0]


def download_davis(
    output_dir: Path,
    *,
    splits: Iterable[str],
    fps: float,
    overwrite: bool,
    keep_frames: bool,
) -> None:
    """Download DAVIS archives and convert frame sequences to mp4 files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        if split not in DAVIS_URLS:
            raise ValueError(
                f"Unknown DAVIS split '{split}'. Available: {sorted(DAVIS_URLS)}"
            )

        archive_url = DAVIS_URLS[split]
        archive_name = archive_url.split("/")[-1]
        archive_path = output_dir / archive_name
        _download_file(archive_url, archive_path, overwrite=overwrite)

        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            logging.info("Extracting %s", archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(tmp_dir)

            extracted_root = Path(tmp_dir)
            frames_root = _find_frames_root(extracted_root)

            for sequence_dir in sorted(frames_root.iterdir()):
                if not sequence_dir.is_dir():
                    continue

                video_path = split_dir / f"{sequence_dir.name}.mp4"
                if video_path.exists() and not overwrite:
                    logging.debug("Skipping existing DAVIS video %s", video_path)
                    continue

                frame_paths = sorted(sequence_dir.glob("*.jpg"))
                if not frame_paths:
                    logging.warning("No frames found in %s; skipping.", sequence_dir)
                    continue

                logging.info(
                    "Writing DAVIS video %s (%d frames)",
                    video_path,
                    len(frame_paths),
                )
                with _open_ffmpeg_writer(video_path, fps) as writer:
                    for frame_path in frame_paths:
                        frame = imageio.imread(frame_path)
                        writer.append_data(frame)

            if keep_frames:
                destination = split_dir / "frames"
                if destination.exists():
                    if overwrite:
                        shutil.rmtree(destination)
                    else:
                        logging.info(
                            "DAVIS frames directory already exists at %s; leaving as-is.",
                            destination,
                        )
                        continue
                logging.info("Preserving extracted DAVIS frames at %s", destination)
                shutil.copytree(frames_root, destination)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Kubric MOVi_E and DAVIS videos as mp4 files."
    )
    parser.add_argument(
        "--kubric-output",
        type=Path,
        default=DEFAULT_KUBRIC_OUTPUT,
        help="Destination directory for MOVi_E videos.",
    )
    parser.add_argument(
        "--kubric-splits",
        nargs="+",
        default=("train", "validation", "test"),
        help="MOVi_E splits to download.",
    )
    parser.add_argument(
        "--kubric-limit",
        type=int,
        default=None,
        help="Optional limit on number of MOVi_E videos per split.",
    )
    parser.add_argument(
        "--kubric-data-dir",
        type=str,
        default=None,
        help=(
            "Optional TFDS data_dir to pass to tfds.load. Leave unset to use the "
            "default (reads directly from the Kubric public GCS bucket)."
        ),
    )
    parser.add_argument(
        "--kubric-overwrite",
        action="store_true",
        help="Overwrite existing MOVi_E mp4 files.",
    )
    parser.add_argument(
        "--kubric-fps",
        type=float,
        default=12.0,
        help="Frame rate to encode MOVi_E videos with (dataset renders at 12fps).",
    )

    parser.add_argument(
        "--davis-output",
        type=Path,
        default=DEFAULT_DAVIS_OUTPUT,
        help="Destination directory for DAVIS videos.",
    )
    parser.add_argument(
        "--davis-splits",
        nargs="+",
        default=("2017-trainval",),
        help=f"DAVIS splits to download. Choices: {sorted(DAVIS_URLS)}",
    )
    parser.add_argument(
        "--davis-fps",
        type=float,
        default=25.0,
        help="Frame rate used when encoding DAVIS videos.",
    )
    parser.add_argument(
        "--davis-overwrite",
        action="store_true",
        help="Overwrite existing DAVIS mp4 files and frame folders.",
    )
    parser.add_argument(
        "--davis-keep-frames",
        action="store_true",
        help="Preserve extracted DAVIS frame directories alongside the mp4 files.",
    )

    parser.add_argument(
        "--skip-kubric",
        action="store_true",
        help="Skip Kubric MOVi_E download.",
    )
    parser.add_argument(
        "--skip-davis",
        action="store_true",
        help="Skip DAVIS download.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.skip_kubric and args.skip_davis:
        logging.error("Both Kubric and DAVIS downloads are skipped; nothing to do.")
        return 1

    if not args.skip_kubric:
        download_movi_e(
            args.kubric_output,
            splits=args.kubric_splits,
            tfds_data_dir=args.kubric_data_dir,
            limit=args.kubric_limit,
            overwrite=args.kubric_overwrite,
            fps=args.kubric_fps,
        )

    if not args.skip_davis:
        download_davis(
            args.davis_output,
            splits=args.davis_splits,
            fps=args.davis_fps,
            overwrite=args.davis_overwrite,
            keep_frames=args.davis_keep_frames,
        )

    logging.info("All downloads complete.")
    return 0


if __name__ == "__main__":  # pragma: no branch
    raise SystemExit(main())
