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
    # install_requires=[
    #     "pytorch-lightning", 
    #     "lightning",
    #     "submitit",
    #     "pyrootutils",
    #     "opencv-python",
    #     "joblib",
    #     "rich",
    #     "einops",
    #     "hydra-core",
    #     "hydra-submitit-launcher",
    #     "timm",
    # ],
    # extras_require={
    #     'demo': [
    #         "phalp[all,blur] @ git+https://github.com/brjathu/PHALP.git",
    #         "pytorchvideo @ git+https://github.com/facebookresearch/pytorchvideo.git",
    #         "slowfast @ git+https://github.com/brjathu/SlowFast.git",
    #     ],
    # },
)
