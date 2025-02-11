import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset


def _downsample_and_fps_select(video_numpy, new_h=256, new_w=384, fps=5, original_fps=25):
    """
    Example stub for downsampling frames to (new_h, new_w) and sampling
    frames at new 'fps' from an original 'original_fps'.

    Args:
        video_numpy: A float or uint8 array of shape [T, H, W, 3].
        new_h: Height after spatial downsampling (256).
        new_w: Width after spatial downsampling (384).
        fps: Target temporal sampling rate (5).
        original_fps: Original frame rate of the data (25), for example.

    Returns:
        A [T_new, new_h, new_w, 3] float32 array.
    """
    # 1) Temporal sub-sampling
    #    Example: Keep every (original_fps // fps)-th frame.
    #    You can adapt or refine depending on actual FPS in the real dataset
    stride = max(1, int(round(original_fps / float(fps))))
    frames = video_numpy[::stride, ...]

    # 2) Spatial downsampling
    #    Use tf.image.resize or similar
    frames_tf = tf.cast(frames, tf.float32)
    frames_tf = tf.image.resize(frames_tf, [new_h, new_w])
    return frames_tf.numpy().astype(np.float32)


def _central_crop_256x256(frames_np):
    """
    Central spatial crop from frames [T, H, W, 3] -> [T, 256, 256, 3].
    Assumes frames_np is already [T, 256, 384, 3]. We take the central 256×256.

    Args:
        frames_np: float array [T, 256, 384, 3]

    Returns:
        A float array [T, 256, 256, 3]
    """
    # frames_np.shape is (T, 256, 384, 3) after previous step
    # center crop to (256, 256)
    # horizontally, 384 -> 256 => offset = (384 - 256) // 2 = 64
    offset_w = (frames_np.shape[2] - 256) // 2
    frames_cropped = frames_np[:, :, offset_w : offset_w + 256, :]
    return frames_cropped


def _random_temporal_subclip(frames_np, boxes_np, clip_len=16):
    """
    Randomly choose a consecutive subclip of length clip_len from frames,
    and filter the boxes accordingly.

    Args:
        frames_np: float array [T, H, W, 3]
        boxes_np: float array [T, N, 4] or [T, N, 5] (depending on your scheme)
                  You might store boxes as [xmin, xmax, ymin, ymax, <other info>].
                  Or keep them in a dictionary. Adapt as needed.
        clip_len: integer, number of frames for the subclip

    Returns:
        frames_subclip: [clip_len, H, W, 3]
        boxes_subclip:  [clip_len, N, 4] (or your choice)
    """
    T = frames_np.shape[0]
    if T <= clip_len:
        # If not enough frames, just return from the start or skip
        return frames_np, boxes_np
    # random start
    start_idx = np.random.randint(0, T - clip_len + 1)
    end_idx = start_idx + clip_len
    return frames_np[start_idx:end_idx], boxes_np[start_idx:end_idx]


def _filter_small_boxes(boxes_first_frame, min_area_ratio=0.005, frame_h=256, frame_w=256):
    """
    Keep only boxes occupying >= min_area_ratio of the first frame area.
    E.g., min_area_ratio=0.005 means at least 0.5% of the 256x256 = 65536 => area >= 327.68 px

    Args:
        boxes_first_frame: shape [N, 4], each row is [xmin, xmax, ymin, ymax] in [0..1]
                           or in pixel coordinates. Adjust accordingly.
        min_area_ratio: float, the fraction of the entire frame the box must occupy
        frame_h, frame_w: used for computing absolute area if boxes are in [0..1].

    Returns:
        Indices of boxes that pass the filter.
    """
    # Suppose boxes_first_frame is in normalized coordinates [0..1].
    # Convert to pixel coords:
    w = (boxes_first_frame[:, 1] - boxes_first_frame[:, 0]) * frame_w
    h = (boxes_first_frame[:, 3] - boxes_first_frame[:, 2]) * frame_h
    area = w * h
    area_threshold = min_area_ratio * frame_h * frame_w
    keep_mask = (area >= area_threshold)
    keep_idx = np.where(keep_mask)[0]
    return keep_idx


def _limit_num_boxes(boxes, max_boxes=25):
    """
    Keep at most max_boxes boxes along the 2nd dim (N dimension).
    We assume boxes has shape [T, N, 4].
    """
    T, N = boxes.shape[0], boxes.shape[1]
    if N <= max_boxes:
        return boxes
    # random subset
    chosen = np.random.choice(N, max_boxes, replace=False)
    chosen = np.sort(chosen)
    return boxes[:, chosen, :]


def process_waymo_sample(sample, max_boxes=25, clip_len=16):
    """
    Entry point: process an individual dataset sample from the Waymo tf.data pipeline
    to produce:
      1) A 16-frame subclip of shape (16, 256, 256, 3)
      2) A set of boxes for each frame => shape (16, M, 4)
      3) Possibly a "query_boxes" = boxes on the first subclip frame => shape (M, 4)

    Returns:
        A dictionary with {
            'video': (16, 256, 256, 3) float32,
            'query_boxes': (M, 4),
            'target_boxes': (16, M, 4),
        }
    """
    # --- Read out raw data from sample (this can vary widely on how your TFRecord is structured) ---
    # Example placeholders below:
    video_raw = sample["video"]  # shape [T, H, W, 3], maybe uint8
    # Suppose sample["boxes"] has shape [T, NumBoxes, 4], in normalized coords [0..1].
    boxes_raw = sample["boxes"]  # shape [T, N, 4]
    # Suppose we know original_fps and do a certain step:
    original_fps = 25

    # 1) Downsample & uniform fps
    frames_ds = _downsample_and_fps_select(video_raw, new_h=256, new_w=384, fps=5, original_fps=original_fps)
    # NOTE you would also want to correspondingly keep the same subsampled timesteps for the boxes
    # i.e., if frames go from the original T to T_new, you must do the same indexing on boxes.
    # For brevity, let's assume boxes_raw was also mapped to the selected frames in sync.

    # 2) Central crop from [T, 256, 384, 3] to [T, 256, 256, 3]
    frames_cropped = _central_crop_256x256(frames_ds)

    # 3) Random temporal subclip of length 16
    frames_subclip, boxes_subclip = _random_temporal_subclip(frames_cropped, boxes_raw, clip_len=clip_len)

    # 4) Filter out boxes that occupy < 0.5% in the first subclip frame
    #    In normalized coords, we can pass frame_h=1, frame_w=1 but here we used 256x256 as the final size.
    keep_idx = _filter_small_boxes(boxes_subclip[0], min_area_ratio=0.005, frame_h=256, frame_w=256)
    boxes_subclip = boxes_subclip[:, keep_idx, :]

    # 5) Limit the number of boxes to at most 25
    boxes_subclip = _limit_num_boxes(boxes_subclip, max_boxes=max_boxes)

    # Now we have frames_subclip: shape [16, 256, 256, 3], boxes_subclip: [16, M, 4].
    # Our "query" boxes presumably come from the first frame of the subclip.
    query_boxes = boxes_subclip[0]   # shape [M, 4]
    target_boxes = boxes_subclip     # shape [16, M, 4]

    return {
        "video": frames_subclip,        # (16, 256, 256, 3) float32
        "query_boxes": query_boxes,     # (M, 4)  float32
        "target_boxes": target_boxes,   # (16, M, 4)
    }


def create_waymo_open_dataset(
    split="train",
    shuffle_buffer_size=256,
    repeat=True,
    max_boxes=25,
    clip_len=16,
    **tfds_kwargs
):
    """
    Builds a tf.data.Dataset from the Waymo Open Dataset
    (assuming a TFDS wrapper is available.)

    Args:
        split: which TFDS split to use
        shuffle_buffer_size: buffer size for shuffling
        repeat: whether to repeat the dataset
        max_boxes: maximum number of boxes to keep
        clip_len: how many frames in the final subclip
        **tfds_kwargs: extra arguments passed to tfds.load

    Returns:
        A tf.data.Dataset that yields pre-processed samples (dicts).
    """

    # 1) Load the dataset from TFDS (this is a placeholder dataset name;
    #    you might need the correct TFDS name or your custom pipeline).
    ds = tfds.load("waymo_open_dataset", split=split, data_dir="/scratch/one_month/current/tekotan/waymo_open_tfds", **tfds_kwargs)
    # ds = tfds.load('waymo_open_dataset/v1.0', data_dir='gs://waymo_open_dataset_v_1_0_0_individual_files/tensorflow_datasets')

    # 2) Possibly shuffle & repeat
    if repeat:
        ds = ds.repeat()
    if shuffle_buffer_size > 0:
        ds = ds.shuffle(shuffle_buffer_size)

    # 3) Preprocess each sample
    def _preproc_fn(sample):
        # Turn it into the final dictionary we want
        outputs = process_waymo_sample(sample, max_boxes=max_boxes, clip_len=clip_len)
        return outputs

    ds = ds.map(_preproc_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return ds


class WaymoOpenBoxTrackingDataset(IterableDataset):
    """
    A PyTorch IterableDataset that wraps around a TensorFlow Dataset of
    { "video": (16, 256, 256, 3), "query_boxes": (M,4), "target_boxes": (16,M,4) }
    samples from the Waymo Open Dataset.
    """
    def __init__(
        self,
        cfg,
        split="train",
        repeat=True,
        shuffle_buffer_size=256,
        max_boxes=25,
        clip_len=16,
        **tfds_kwargs
    ):
        super().__init__()
        self.cfg = cfg

        # Build the TF dataset pipeline on CPU
        with tf.device("/CPU:0"):
            self.tf_dataset = create_waymo_open_dataset(
                split=split,
                shuffle_buffer_size=shuffle_buffer_size,
                repeat=repeat,
                max_boxes=max_boxes,
                clip_len=clip_len,
                **tfds_kwargs
            )
            # Convert tf.data.Dataset -> Python generator of NumPy arrays
            self.generator = tfds.as_numpy(self.tf_dataset)

    def __iter__(self):
        for sample in self.generator:
            # sample is a dictionary with keys: "video", "query_boxes", "target_boxes"
            video_np = sample["video"]           # shape: (16, 256, 256, 3) float32
            query_boxes_np = sample["query_boxes"]    # shape: (M, 4)
            target_boxes_np = sample["target_boxes"]  # shape: (16, M, 4)

            # Suppose we want to feed the frames into a model expecting (C, T, H, W).
            # Convert (16, 256, 256, 3) -> (3, 16, 256, 256)
            video_np = np.transpose(video_np, (3, 0, 1, 2))

            # Convert to torch
            video_torch = torch.from_numpy(video_np)  # (3,16,256,256)

            # Also convert boxes to torch. Depending on your usage, you might keep them
            # in normalized coords, or transform them to (xmin, xmax, ymin, ymax) in absolute px.
            query_boxes_torch = torch.from_numpy(query_boxes_np)       # (M, 4)
            target_boxes_torch = torch.from_numpy(target_boxes_np)      # (16, M, 4)

            # Possibly pad boxes if you want a fixed shape. E.g. M <= 25 in your pipeline.
            # Or you can pass them as is (variable shapes).
            # E.g. code snippet for padding to exactly 25 boxes:
            M = query_boxes_torch.shape[0]
            if M < 25:
                diff = 25 - M
                # For query_boxes => shape (M,4)
                pad_q = torch.zeros(diff, 4, dtype=query_boxes_torch.dtype)
                # put a sentinel to indicate no box, e.g. -1
                pad_q[:, :] = -1.0
                query_boxes_torch = torch.cat([query_boxes_torch, pad_q], dim=0)

                # For target_boxes => shape (16, M, 4)
                pad_t = torch.zeros(16, diff, 4, dtype=target_boxes_torch.dtype)
                pad_t[:, :] = -1.0
                target_boxes_torch = torch.cat([target_boxes_torch, pad_t], dim=1)

            # yield data
            yield (video_torch.float(), query_boxes_torch.float(), target_boxes_torch.float())

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--repeat", default=False, action="store_true")
    parser.add_argument("--shuffle_buffer_size", default=256, type=int)
    parser.add_argument("--max_boxes", default=25, type=int)
    parser.add_argument("--clip_len", default=16, type=int)
    args = parser.parse_args()

    print("Testing create_waymo_open_dataset with arguments:", args)
    ds = create_waymo_open_dataset(
        split=args.split,
        shuffle_buffer_size=args.shuffle_buffer_size,
        repeat=args.repeat,
        max_boxes=args.max_boxes,
        clip_len=args.clip_len
    )

    sample_ds = next(iter(ds))
    print("Got a sample from the tf.data.Dataset:")
    for k, v in sample_ds.items():
        if hasattr(v, 'shape'):
            print(k, v.shape)
        else:
            print(k, v)

    print("\nTesting WaymoOpenBoxTrackingDataset (PyTorch) with arguments:", args)
    pt_ds = WaymoOpenBoxTrackingDataset(
        cfg={},
        split=args.split,
        repeat=args.repeat,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_boxes=args.max_boxes,
        clip_len=args.clip_len
    )

    pt_it = iter(pt_ds)
    pt_sample = next(pt_it)
    video_shape = pt_sample[0].shape
    query_boxes_shape = pt_sample[1].shape
    target_boxes_shape = pt_sample[2].shape
    print("First PyTorch sample shapes:", video_shape, query_boxes_shape, target_boxes_shape)
