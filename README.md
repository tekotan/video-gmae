<!-- import the image.webp -->

<p align="center">
  <img src="assets/chewbacca.webp" alt="Chewbacca" width="300" />
</p>

# Tracking by Predicting 3-D Gaussians Over Time

Code release for the video masked autoencoder models used for point tracking experiments on TAP-Vid (Kinetics and DAVIS). The repo ships the evaluation configs and checkpoints used in the paper, plus small helpers for working with the TAP-Vid pickles.

## Installation

Tested with Python 3.10 and CUDA 12.1.

Pixi (recommended)
```bash
# install pixi: https://pixi.sh
git clone https://github.com/tekotan/video-gmae.git && cd video-gmae
pixi install       # resolve the env (Linux, CUDA 12.1, torch 2.1 + xformers)
pixi run install   # editable install of this repo
pixi shell         # drop into the environment
# optional extras (inside pixi shell), e.g. kubric data utils:
# pip install '.[kubric,wandb]'
```

Conda (fallback)
```bash
conda create -n chewbacca python=3.10
conda activate chewbacca
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
# optional extras:
# pip install '.[kubric,wandb]'
```

`pyrootutils` will set `PROJECT_ROOT` for you when running the scripts, so everything works from the repo root.

## Checkpoints and data

The evaluation scripts expect checkpoints under `checkpoints/` and TAP-Vid pickles under a `data-download/` directory at the repo root (Hydra sets `PROJECT_ROOT`, so relative paths work even when the working directory changes).

### Checkpoints

Place the released weights in `checkpoints/` with these filenames (matching the scripts):
- `checkpoints/zeroshot_checkpoint.ckpt` – zero-shot GMAE
- `checkpoints/finetune_checkpoint.ckpt` – finetuned GMAE

### TAP-Vid data

Follow the official TAP-Vid download instructions (DAVIS and Kinetics) in the TAPNet repository: https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid/README.md. If you already have the raw videos, you can also generate pickles with `chewbacca/utils/generate_tapvid.py`.

Layout expected by the eval scripts:

```
data-download/
├── tapvid_davis/
│   └── tapvid_davis.pkl
├── tapvid_kinetics/
│   ├── kinetics_10percent_sample.pkl  # quick eval split
│   └── 0000_of_0010.pkl ... 0009_of_0010.pkl # optional full shards
```

If you store the data elsewhere, update the paths in `chewbacca/datamodules/finetune_datamodule.py` (training types `eval-davis` and `eval-kinetics`).

## Reproducing evaluations

All runs use `configs/gmae_ema.yaml` and default to a single GPU (`trainer.devices=1`). Activate your environment, ensure the checkpoints and pickles are in place, then run from the repo root:

```bash
bash scripts/launch_zeroshot_davis_eval.sh      # zero-shot TAP-Vid-DAVIS
bash scripts/launch_zeroshot_kinetics_eval.sh   # zero-shot TAP-Vid-Kinetics
bash scripts/launch_finetune_davis_eval.sh      # finetuned TAP-Vid-DAVIS
bash scripts/launch_finetune_kinetics_eval.sh   # finetuned TAP-Vid-Kinetics
```

Hydra writes logs and predictions under `logs/<task_name>/`. Adjust batch sizes, device counts, or `configs.weights_path` inside the scripts if you move things around.
