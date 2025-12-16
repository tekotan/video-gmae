#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="chewbacca",
    version="0.0.0",
    description="Large scale training code for anything.",
    author="Jathushan Rajasegaran",
    author_email="jathushan@berkeley.edu",
    url="https://github.com/brjathu/chewbacca",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    packages=find_packages(),
    install_requires=[
        "torch>=2.1,<2.2",
        "torchvision>=0.16,<0.17",
        "xformers==0.0.22.post7",
        "lightning==2.4.0",
        "torchmetrics==1.4.2",
        "hydra-core==1.3.2",
        "hydra-submitit-launcher==1.2.0",
        "submitit>=1.5.0",
        "pyrootutils==1.0.4",
        "einops==0.8.0",
        "timm==1.0.9",
        "tqdm==4.66.5",
        "rich==13.8.1",
        "absl-py==2.1.0",
        "joblib==1.4.2",
        "numpy>=1.26,<2.0",
        "tensorboard==2.18.0",
        "scikit-image==0.25.0",
        "opencv-python==4.10.0.84",
        "pillow>=9.4.0",
        "matplotlib>=3.8,<3.9",
        "mediapy>=1.2.0",
        "decord==0.6.0",
        "webdataset==0.2.100",
        "ffmpeg-python==0.2.0",
        "moviepy==1.0.3",
        "lpips==0.1.4",
        "transformers==4.44.2",
        "gsplat>=1.5.3",
    ],
    extras_require={
        "kubric": [
            "tensorflow>=2.13",
            "tensorflow-datasets>=4.9",
            "tensorflow-graphics==2021.12.3",
        ],
        "wandb": ["wandb>=0.17"],
    },
)
