import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

from decord import VideoReader, cpu, gpu

logger = logging.getLogger(__name__)

class VideoAnnotationDataset(Dataset):
    def __init__(self, cfg, video_infos, root_dir, train=True, flip_rgb=False):
        """
        Args:
            cfg: A configuration object/dict containing at least:
                 - seq_length: number of frames per sample (int)
                 - sample_rate: spacing between sampled frames (int)
                 - input_size: desired (square) output image size (int)
                 - finetune_params.tracks_to_sample: maximum number of tracks (int)
            video_infos: list of paths to mp4 files. For example:
                         ["/path/to/000.mp4", "/path/to/001.mp4", ...]
            root_dir: base directory containing one folder per sequence.
                      Each sequence folder should be named by its video id (e.g. "000")
                      and contain a subfolder "gt" with the file "gt.txt".
            train: training mode flag (unused here, but preserved for compatibility)
            flip_rgb: if True, flip the channel order of the returned video frames.
        """
        super().__init__()
        self.cfg = cfg
        self.train = train
        self.flip_rgb = flip_rgb
        self.root_dir = root_dir
        self.video_infos = video_infos  # list of paths to mp4 files
        self.good_videos = []  # will hold tuples (video_path, label) for successfully read videos
        self.retry_counter = 0
        
        # Preload MOT annotations for each video.
        # We assume that the MOT annotation file for video "000.mp4" is at:
        #   {root_dir}/000/gt/gt.txt
        if self.cfg.finetune_params.test:
            self.video_infos = self.video_infos[5:6]

        self.annotations = {}
        for video_path in self.video_infos:
            video_id = os.path.splitext(os.path.basename(video_path[0]))[0]
            annot_path = os.path.join(self.root_dir, video_id, "gt", "gt.txt")
            if os.path.exists(annot_path):
                self.annotations[video_id] = self._load_annotations(annot_path)
            else:
                logger.warning(f"Annotation file not found: {annot_path}")
                self.annotations[video_id] = {}
        
        if self.cfg.finetune_params.test:
            self.video_infos = self.video_infos.repeat(100, 0)

    def _load_annotations(self, annot_path):
        """
        Parses a MOT annotation file and returns a dictionary mapping
        frame ids (int) to a list of annotations. Each annotation is a dict:
            {'object_id': int, 'bbox': [x, y, w, h]}
        """
        annotations = {}
        try:
            with open(annot_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue
                    # MOT format: frame_id, object_id, x, y, w, h, ... (ignore the rest)
                    frame_id = int(float(parts[0]))
                    object_id = int(float(parts[1]))
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    if frame_id not in annotations:
                        annotations[frame_id] = []
                    annotations[frame_id].append({
                        'object_id': object_id,
                        'bbox': [x, y, w, h]
                    })
        except Exception as e:
            logger.error(f"Error reading annotation file {annot_path}: {e}")
        return annotations

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        # Get video path and (optional) label. Here, we assume no label so we set it to -1.
        video_path = self.video_infos[idx][0]
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        video_label = -1

        # Try to open the video using Decord.
        try:
            video_container = VideoReader(video_path, ctx=cpu(0))
            num_frames = len(video_container)
            # Save this video info if successful.
            self.good_videos.append((video_path, video_label))
        except Exception as e:
            logger.info("Error reading video frames: " + video_path)
            # Fallback: if we have any good videos already, sample one.
            if len(self.good_videos) > 0:
                idx_sample = random.choice(range(len(self.good_videos)))
                video_path, video_label = self.good_videos[idx_sample]
                video_container = VideoReader(video_path, ctx=cpu(0))
                num_frames = len(video_container)
            else:
                # Last resort: try the very first video.
                try:
                    video_container = VideoReader(self.video_infos[0], ctx=cpu(0))
                    video_label = -1
                    num_frames = len(video_container)
                except Exception as e2:
                    logger.info("Error reading video frames from fallback.")
                    num_frames = 0
                    black_frames = torch.zeros(3, self.cfg.seq_length, self.cfg.input_size, self.cfg.input_size, dtype=torch.float32)
                    return black_frames, {}, video_label, idx

        # Determine which frames to sample.
        desired = self.cfg.seq_length * self.cfg.sample_rate
        if num_frames > desired:
            start = np.random.randint(0, num_frames - desired + 1)
            selected_indices = list(range(start, start + desired, self.cfg.sample_rate))
        else:
            selected_indices = list(range(num_frames))

        # Try to get the frames via Decord.
        try:
            frames_ = video_container.get_batch(selected_indices).asnumpy()  # shape: (T, H, W, C)
        except Exception as e:
            logger.info("Error getting batch from video: " + video_path)
            frames_ = np.random.randint(0, 255, (self.cfg.seq_length, self.cfg.input_size, self.cfg.input_size, 3), dtype=np.uint8)

        proc_frames = []   # list to hold processed (normalized) frames
        bbox_frames = {}   # dict mapping object_id -> {"frames": [], "bboxes": [], "occluded": []}
        invalid_object_ids = set()
        orig_h, orig_w = None, None

        # Process each sampled frame.
        for i, frame_idx in enumerate(selected_indices):
            # Convert frame (numpy array) to a PIL Image.
            frame = frames_[i]  # shape (H, W, C)
            im = Image.fromarray(frame)

            orig_w, orig_h = im.size  # (width, height)
            h_scale = self.cfg.input_size / orig_w
            w_scale = self.cfg.input_size / orig_h

            # Resize image to input_size and normalize (ImageNet style).
            im = im.resize((self.cfg.input_size, self.cfg.input_size), resample=Image.LANCZOS)
            arr = np.array(im).astype(np.float32) / 255.0
            arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
                  np.array([0.229, 0.224, 0.225], dtype=np.float32)
            proc_frames.append(arr)

            # Get annotations for this frame.
            # Note: MOT frame_ids are 1-indexed so we use (actual frame index + 1)
            annot_frame_id = selected_indices[i] + 1
            annotations = self.annotations.get(video_id, {}).get(annot_frame_id, [])
            for ann in annotations:
                object_id = ann["object_id"]
                bbox = ann["bbox"]  # [x, y, w, h]
                # Rearrange to [y, x, h, w]
                # bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
                new_t_y = bbox[0] * h_scale
                new_t_x = bbox[1] * w_scale
                b_h = bbox[2] * h_scale
                b_w = bbox[3] * w_scale

                if object_id not in bbox_frames:
                    if i == 0:
                        bbox_frames[object_id] = {"frames": [], "bboxes": [], "occluded": []}
                    else:
                        continue
                # Check if the bbox is fully inside the resized image.
                if (new_t_y < 0 or new_t_x < 0 or 
                    (new_t_y + b_h) > self.cfg.input_size or 
                    (new_t_x + b_w) > self.cfg.input_size):
                    # if i == 0:
                    #     invalid_object_ids.add(object_id)
                    # else:
                    bbox_frames[object_id]["occluded"].append(1)
                else:
                    bbox_frames[object_id]["occluded"].append(0)
                bbox_frames[object_id]["frames"].append(i)
                bbox_frames[object_id]["bboxes"].append([
                    new_t_y / self.cfg.input_size, new_t_x / self.cfg.input_size,
                    (new_t_y + b_h) / self.cfg.input_size, (new_t_x + b_w) / self.cfg.input_size
                ])

        # Remove any object_ids that became invalid (either missing in some frames or cropped out on the first frame)
        for object_id in list(bbox_frames.keys()):
            if len(bbox_frames[object_id]["frames"]) < len(selected_indices):
                pad_length = len(selected_indices) - len(bbox_frames[object_id]["frames"])
                bbox_frames[object_id]["frames"].extend([-1] * pad_length)
                bbox_frames[object_id]["bboxes"].extend([[-1, -1, -1, -1]] * pad_length)
                bbox_frames[object_id]["occluded"].extend([1] * pad_length)
            if object_id in invalid_object_ids:
                del bbox_frames[object_id]

        # Convert the list of processed frames into a tensor.
        proc_frames = np.array(proc_frames)  # shape: (T, H, W, C)
        proc_frames = np.transpose(proc_frames, (3, 0, 1, 2))  # shape: (3, T, H, W)
        T = proc_frames.shape[1]
        # If there are not enough frames, pad with zeros.

        if self.cfg.seq_length == 1:
            proc_frames = proc_frames[:, 0, :, :]

        if self.flip_rgb:
            proc_frames = proc_frames[::-1, ...]  # flip channel order if needed

        video_torch = torch.from_numpy(proc_frames)

        # Convert bbox lists into tensors.
        target_bboxes_torch = torch.Tensor([v["bboxes"] for v in bbox_frames.values()])
        occluded_bboxes_torch = torch.Tensor([v["occluded"] for v in bbox_frames.values()])
        if len(target_bboxes_torch.shape) == 1:
            self.retry_counter += 1
            if self.retry_counter > 10:
                raise ValueError("Too many retries")
            else:
                return self.__getitem__(idx)
        query_bboxes_torch = target_bboxes_torch[:, 0]

        if T < self.cfg.seq_length:
            pad = np.zeros((proc_frames.shape[0], self.cfg.seq_length - T, self.cfg.input_size, self.cfg.input_size), dtype=np.float32)
            proc_frames = np.concatenate((proc_frames, pad), axis=1)
            proc_frames = proc_frames[:, :self.cfg.seq_length, :, :]
            
            target_bboxes_torch = torch.cat([target_bboxes_torch, torch.ones(target_bboxes_torch.shape[0], self.cfg.seq_length - T, 4) * -1], dim=1)
            occluded_bboxes_torch = torch.cat([occluded_bboxes_torch, torch.zeros(occluded_bboxes_torch.shape[0], self.cfg.seq_length - T)], dim=1)
        
        N = target_bboxes_torch.shape[0]
        if N > self.cfg.finetune_params.tracks_to_sample:
            random_idx = np.random.choice(N, self.cfg.finetune_params.tracks_to_sample, replace=False)
            target_bboxes_torch = target_bboxes_torch[random_idx]
            query_bboxes_torch = query_bboxes_torch[random_idx]
            occluded_bboxes_torch = occluded_bboxes_torch[random_idx]
        if N < self.cfg.finetune_params.tracks_to_sample:
            target_bboxes_torch = torch.cat([target_bboxes_torch, torch.zeros(self.cfg.finetune_params.tracks_to_sample - N, self.cfg.seq_length, 4)], dim=0)
            query_bboxes_torch = torch.cat([query_bboxes_torch, torch.ones(self.cfg.finetune_params.tracks_to_sample - N, 4) * -1], dim=0)
            occluded_bboxes_torch = torch.cat([occluded_bboxes_torch, torch.zeros(self.cfg.finetune_params.tracks_to_sample - N, self.cfg.seq_length)], dim=0)
        self.retry_counter = 0
        return video_torch, query_bboxes_torch, target_bboxes_torch, occluded_bboxes_torch