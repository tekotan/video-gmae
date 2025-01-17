import glob
import random

import numpy as np
import webdataset as wds
from decord import VideoReader, cpu, gpu
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import joblib
from chewbacca.utils import get_pylogger
from typing import Any



logger = get_pylogger(__name__)


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



class VideoDataset(Dataset):
    def __init__(self, cfg, video_paths, train=True, flip_rgb=False):
        super().__init__()
        self.cfg = cfg
        self.train = train
        # video paths is list of videos and labels
        self.video_paths = video_paths
        self.good_videos = []
        self.flip_rgb = flip_rgb

        logger.info(f"Number of videos: {len(self.video_paths)}")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):

        # video
        data = self.video_paths[idx]
        video_path = data[0]
        video_label = int(data[1])
        video_dataset = 0
        video_idx = idx

        arr_list = []
        try:
            video_container = VideoReader(video_path, ctx=cpu(0))
            num_frames = len(video_container)
            self.good_videos.append(data)
        except:
            logger.info("error in reading a video frames." + video_path)
            if len(self.good_videos) > 0:
                idx = random.sample(list(range(len(self.good_videos))), 1)[0]
                data = self.good_videos[idx]
                video_path = data[0]
                video_label = int(data[1])
                video_container = VideoReader(video_path, ctx=cpu(0))
                num_frames = len(video_container)
                video_idx = idx
            else:
                try:
                    video_container = VideoReader(self.video_paths[0], ctx=cpu(0))
                    video_label = int(self.video_paths[0][1])
                    num_frames = len(video_container)
                    video_idx = 0
                except:
                    logger.info("error in reading a video frames.")
                    num_frames = 0
                    return_0 = True


        frames = [i for i in range(num_frames)]
        if num_frames > self.cfg.seq_length * self.cfg.sample_rate:
            start = np.random.randint(0, num_frames-self.cfg.seq_length * self.cfg.sample_rate)
            frames = frames[start:start + self.cfg.seq_length * self.cfg.sample_rate:self.cfg.sample_rate]

        # GET THE RGB FRAMES
        try:
            frames_ = video_container.get_batch(frames)
            frames_ = frames_.asnumpy()
        except:
            logger.info("error in reading a video frames.")
            # return a black image
            frames_ = np.random.randint(0, 255, (self.cfg.seq_length, self.cfg.input_size, self.cfg.input_size, 3), dtype=np.uint8)

        for i, frame in enumerate(frames_):
            frame = Image.fromarray(frame)
            while min(*frame.size) >= 2 * self.cfg.input_size:
                frame = frame.resize(
                    tuple(x // 2 for x in frame.size), resample=Image.BOX
                )
            scale = self.cfg.input_size / min(*frame.size)
            frame = frame.resize(
                tuple(round(x * scale) for x in frame.size), resample=Image.BICUBIC
            )

            arr = np.array(frame.convert("RGB"))
            crop_y = (arr.shape[0] - self.cfg.input_size) // 2
            crop_x = (arr.shape[1] - self.cfg.input_size) // 2
            arr = arr[crop_y : crop_y + self.cfg.input_size, crop_x : crop_x + self.cfg.input_size]
            # save frames as images
            # img = Image.fromarray(arr)
            # img.save(f"frame2_{i}.png")
            # convert to float32 and normalize imagenet style
            arr = arr.astype(np.float32)/255.0
            arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr_list.append(arr)

        arr_seq = np.array(arr_list)
        arr_seq = np.transpose(arr_seq, [3, 0, 1, 2])
        # fill in missing frames with 0s
        if arr_seq.shape[1] < self.cfg.seq_length:
            required_dim = self.cfg.seq_length - arr_seq.shape[1]
            fill = np.zeros((3, required_dim, self.cfg.input_size, self.cfg.input_size), dtype=np.float32)
            arr_seq = np.concatenate((arr_seq, fill), axis=1)

        arr_seq = arr_seq[:, :self.cfg.seq_length, :, :]

        if(self.cfg.seq_length == 1):
            arr_seq = arr_seq[:, 0, :, :]

        if(self.flip_rgb):
            arr_seq = arr_seq[:, :, :, :].flip(dims=(0))

        return arr_seq, video_label, -1, video_idx
