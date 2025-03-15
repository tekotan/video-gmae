# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pathlib
import warnings

from logging import getLogger

import numpy as np
import pandas as pd

from decord import VideoReader, cpu

import torch
from lart.datamodules.components.augmentation.vjepa import make_transforms

class VideoDataset(torch.utils.data.Dataset):
    """ Video classification dataset. """

    def __init__(
        self,
        cfg,
        train=True,
        datasets_weights=None,
        frames_per_clip=16,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=True,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
    ):
        self.cfg = cfg
        self.train = train
        self.datasets_weights = datasets_weights
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Load video paths and labels
        # samples, labels = [], []
        self.num_samples_per_dataset = []
        
        if self.train:
            video_paths = np.load("data/k400_train_with_labels.npy")
        else:
            video_paths = np.load("data/k400_val_with_labels.npy")

        samples = list(video_paths[:, 0])
        labels = list(video_paths[:, 1])
        num_samples = len(samples)
        self.num_samples_per_dataset.append(num_samples)

        self.samples = samples
        self.labels = labels


        if "32k6" in self.cfg.storage_folder:
            num_views_per_clip = 1
            self.num_clips = 8
            if "32k6_test" in self.cfg.storage_folder:
                num_views_per_clip = 1
                self.num_clips = 8
        if "32k7" in self.cfg.storage_folder:
            num_views_per_clip = 1
            self.num_clips = 8
        else:
            num_views_per_clip = 1
            self.num_clips = 1

        self.transform = make_transforms(
                training=self.train,
                num_views_per_clip=num_views_per_clip,
                random_horizontal_flip=False,
                random_resize_aspect_ratio=(0.75, 4/3),
                random_resize_scale=(0.08, 1.0),
                reprob=0.25,
                auto_augment=True,
                motion_shift=False,
                crop_size=self.cfg.input_size,
            )

        self.good_index = []

    def __getitem__(self, index):
        sample = self.samples[index]
        # Keep trying to load videos until you find a valid sample
        loaded_video = False

        if "32k6" in self.cfg.storage_folder:
            loaded_video = False
            buffer, clip_indices = self.loadvideo_decord(sample)  # [T H W 3]
            loaded_video = len(buffer) > 0
            if not loaded_video:
                index = 0 #np.random.randint(self.__len__())
                sample = self.samples[index]
                buffer, clip_indices = self.loadvideo_decord(sample)  # [T H W 3]
        else:
            while not loaded_video:
                buffer, clip_indices = self.loadvideo_decord(sample)  # [T H W 3]
                loaded_video = len(buffer) > 0
                if not loaded_video and len(self.good_index)>0:
                    index = np.random.randint(len(self.good_index))
                    sample = self.good_index[index]
                else:
                    index = np.random.randint(self.__len__())
                    sample = self.samples[index]

            if sample not in self.good_index:
                self.good_index.append(sample)

        # Label/annotations for video
        label = int(self.labels[index])

        def split_into_clips(video):
            """ Split video into a list of clips """
            fpc = self.frames_per_clip
            nc = self.num_clips
            return [video[i*fpc:(i+1)*fpc] for i in range(nc)]

        # Parse video into frames & apply data augmentations
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        return buffer, label, clip_indices

    def loadvideo_decord(self, sample):
        """ Load video content using Decord """

        fname = sample
        if not os.path.exists(fname):
            warnings.warn(f'video path not found {fname=}')
            return [], None

        _fsize = os.path.getsize(fname)
        if _fsize < 1 * 1024:  # avoid hanging issue
            warnings.warn(f'video too short {fname=}')
            return [], None
        if _fsize > self.filter_long_videos:
            warnings.warn(f'skipping long video of size {_fsize=} (bytes)')
            return [], None

        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception:
            return [], None

        fpc = self.frames_per_clip
        fstp = self.frame_step
        if self.duration is not None:
            try:
                fps = vr.get_avg_fps()
                fstp = int(self.duration * fps / fpc)
            except Exception as e:
                warnings.warn(e)
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f'skipping video of length {len(vr)}')
            return [], None

        vr.seek(0)  # Go to start of video before sampling frames

        # Partition video into equal sized segments and sample each clip
        # from a different segment
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):

            if partition_len > clip_len:
                # If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx-1).astype(np.int64)
                # --
                indices = indices + i * partition_len
            else:
                # If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - partition_len // fstp) * partition_len,))
                    indices = np.clip(indices, 0, partition_len-1).astype(np.int64)
                    # --
                    indices = indices + i * partition_len

                # If partition overlap is allowed and partition_len < clip_len
                # then start_indx of segment i+1 will lie within segment i
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - sample_len // fstp) * sample_len,))
                    indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
                    # --
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices

    def __len__(self):
        return len(self.samples)
