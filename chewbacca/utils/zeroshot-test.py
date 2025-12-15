from pathlib import Path

import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(".").resolve()
DATA_DIR = REPO_ROOT / "data"
KINETICS_DIR = REPO_ROOT / "data-download" / "zeroshot-test" / "kinetics"
DAVIS_DIR = REPO_ROOT / "data-download" / "zeroshot-test" / "davis"


def build_labels(video_dir: Path, save_path: Path) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    videos = sorted(video_dir.glob("*.mp4"))
    print(f"Found {len(videos)} videos in {video_dir}")
    labels = np.array([[str(path), -1] for path in tqdm(videos)], dtype=object)
    np.save(save_path, labels)
    print(f"Saved labels to {save_path}")


def main() -> None:
    build_labels(KINETICS_DIR, DATA_DIR / "kinetics_test_label.npy")
    build_labels(DAVIS_DIR, DATA_DIR / "davis_test_label.npy")


if __name__ == "__main__":
    main()
