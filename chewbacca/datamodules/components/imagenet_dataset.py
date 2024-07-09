

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import glob
import math
import os
from typing import Any

import joblib
import numpy as np
import PIL
import torch
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import create_transform
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from chewbacca.utils import get_pylogger
import joblib

log = get_pylogger(__name__)

MAX_CACHE_SIZE = 1000



class RandomResizedCrop(transforms.RandomResizedCrop):
    """
    https://github.com/facebookresearch/mae/blob/main/util/crop.py#L15
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """
    @staticmethod
    def get_params(img, scale, ratio):
        width, height = F.get_image_size(img)
        area = height * width

        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()

        return i, j, h, w
    
def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray

def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
    
def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def build_transform(is_train, cfg):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:

        if(cfg.task=="finetune"):
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=cfg.input_size,
                is_training=True,
                color_jitter=cfg.color_jitter,
                auto_augment=cfg.aa,
                interpolation='bicubic',
                re_prob=cfg.reprob,
                re_mode=cfg.remode,
                re_count=cfg.recount,
                mean=mean,
                std=std,
            )
            return transform
        else:
            if cfg.input_size == 480 or cfg.input_size == 481:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop((480, 640), interpolation=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
                return transform
            else:
                transform = transforms.Compose([
                    RandomResizedCrop(cfg.input_size, interpolation=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
                return transform
    
    # eval transform
    t = []
    if cfg.input_size == 480 or cfg.input_size == 481:
        crop_pct = 0.875
        t.append(
            transforms.Resize((int(480/crop_pct), int(640/crop_pct)), interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop((480, 640)))
    else:
        crop_pct = 0.875
        size = int(cfg.input_size / crop_pct)
        t.append(
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(cfg.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_transform_probe(is_train, cfg):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = transforms.Compose([
            RandomResizedCrop(cfg.input_size, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    else:
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(cfg.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    return transform




class PKL_dataset(Dataset):
    def __init__(self, train=True, transform=None):

        self.train = train
        self.transform = transform

        if(self.train):
            self.data = np.load("data/imagenet_train_label.npy", allow_pickle=True)
            # seed and, shuffle the list
            np.random.seed(1234)
            np.random.shuffle(self.data)
        else:
            self.data = np.load("data/imagenet_val_label.npy", allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_ = self.data[idx]
        image     = default_loader(data_[0])
        image_ = self.transform(image)
        label     = data_[1]

        return image_, int(label)
    

def build_imagenet_dataset(is_train, cfg):
    transform = build_transform(is_train, cfg)

    print("Building ImageNet dataset {}".format('train' if is_train else 'val'))
    dataset = PKL_dataset(train=is_train, transform=transform)
    print("Done building ImageNet dataset {}".format('train' if is_train else 'val'))

    print(dataset)

    return dataset

