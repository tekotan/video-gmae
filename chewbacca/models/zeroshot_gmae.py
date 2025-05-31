import math
import os
from typing import Any, Optional, Tuple

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import transformers
# import xformers.ops as xops
from einops import rearrange
# from lart.models.components.tokeizers.tokenizer import Tokenizer
from lightning import LightningModule
from moviepy.editor import ImageSequenceClip
from omegaconf import DictConfig
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch import nn
from torchmetrics import MeanMetric, MeanSquaredError, SumMetric
from torchmetrics.aggregation import CatMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import lpips
from tqdm import tqdm

from chewbacca.utils import get_pylogger
from chewbacca.utils.lamb import LARS, Lamb
from gsplat import rasterization
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
)

from chewbacca.utils import tapvid_viz_utils


log = get_pylogger(__name__)

def compute_tapvid_metrics(
    query_points,
    gt_occluded,
    gt_tracks,
    pred_occluded,
    pred_tracks,
    query_mode="first",
):
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)

    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.

    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.

    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.

    Returns:
        A dict with the following keys:

        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """
    assert gt_occluded.shape == pred_occluded.shape

    metrics = {}

    # Don't evaluate the query point.  Numpy doesn't have one_hot, so we
    # replicate it by indexing into an identity matrix.
    one_hot_eye = np.eye(gt_tracks.shape[2])
    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = one_hot_eye[query_frame] == 0

    # If we're using the first point on the track as a query, don't evaluate the
    # other points.

    if query_mode == "first":
        assert gt_occluded.shape[0] == 1, "Expected batch size 1 gt_occluded"
        for i in range(gt_occluded.shape[1]):
            index = np.where(gt_occluded[0, i] == 0)[0]
            index = index[0] if len(index) > 0 else gt_occluded.shape[2]
            evaluation_points[0, i, :index] = False
    elif query_mode != "strided":
        raise ValueError("Unknown query mode " + query_mode)

    # breakpoint()
    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy"] = occ_acc

    # Added by DW based on code by SK
    metrics["occ_tp"] = np.sum(np.equal(pred_occluded, gt_occluded) & gt_occluded & evaluation_points, axis=(1, 2))
    metrics["occ_fp"] = np.sum(
        np.logical_not(np.equal(pred_occluded, gt_occluded)) & pred_occluded & evaluation_points, axis=(1, 2)
    )
    metrics["occ_fn"] = np.sum(
        np.logical_not(np.equal(pred_occluded, gt_occluded)) & np.logical_not(pred_occluded) & evaluation_points,
        axis=(1, 2),
    )

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    # breakpoint()
    L2_error = np.sqrt(
        np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        )
    )
    masked_L2_error = L2_error * (1 - gt_occluded)
    avg_distance = np.sum(masked_L2_error) / np.sum(1 - gt_occluded)
    metrics["avg_distance"] = np.array([avg_distance])
    nonzero_masked_error = L2_error[(1 - gt_occluded).astype(bool)]
    assert np.allclose(masked_L2_error.sum(), nonzero_masked_error.sum()), (
        masked_L2_error.sum(),
        nonzero_masked_error.sum(),
        set((1 - gt_occluded).flatten().tolist()),
    )
    assert nonzero_masked_error.size == np.sum(1 - gt_occluded)
    assert np.allclose(avg_distance, nonzero_masked_error.mean()), (
        avg_distance,
        nonzero_masked_error.mean(),
        avg_distance - nonzero_masked_error.mean(),
        set((1 - gt_occluded).flatten().tolist()),
    )
    metrics["median_distance"] = np.array([np.median(nonzero_masked_error)])

    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct

        metrics["num_visible"] = count_visible_points
        metrics["num_pts_within_" + str(thresh)] = count_correct

        all_frac_within.append(frac_correct)

        true_positives = np.sum(is_correct & pred_visible & evaluation_points, axis=(1, 2))

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (MB)."""
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 / 1024

def dense_flow_one_frame(means_t,    # (G,3)
                         mean_delta, # (G,3)   μ(t+1)−μ(t)
                         quats, scales, opacities,
                         view, K, W, H, idx=None, device="cuda"):
    _, xy_t, _, _, _ = fully_fused_projection(means_t, None, quats, scales, view[None], K[None], W, H)
    _, xy_t1, _, _, _ = fully_fused_projection(means_t + mean_delta, None, quats, scales, view[None], K[None], W, H)
    d_pix = (xy_t1 - xy_t).squeeze()
    d_pix = d_pix.to(torch.float32)
    rgbs = torch.zeros(means_t.shape[0], 3).to(device)
    rgbs[:, :2] = d_pix


    dels, alpha, _ = rasterization(
        means_t, quats, scales, opacities.squeeze(-1), rgbs,
        view[None], K[None], width=W, height=H,
        packed=False, render_mode="RGB")
    flow = dels[0, :, :, :2] * alpha[0, :, :]  # (H,W,2)

    return flow.detach()

def get_zeroshot_flow(data):
    device = data["x_points"].device
    T, H, W = 16, 224, 224
    f = 0.5 * W / math.tan(math.pi / 4)
    K = torch.tensor([[f, 0, W / 2],
                      [0, W / H * f, H / 2],
                      [0, 0, 1]], device=device).float()
    view = torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 8],
                         [0, 0, 0, 1]], device=device).float()

    H, W = data["video"].shape[2:4]
    batch_flows = []
    for i in range(data["x_points"].shape[0]):
        x = data["x_points"][i]
        
        allmeans = 5 * torch.tanh(x[:, :3])
        mean_deltas = allmeans[256:].reshape(-1, 256, 3) / 10
        means = allmeans[:256]
        scales = torch.sigmoid(x[:, 3:6])[:256]
        q_raw = torch.sigmoid(x[:, 6:10])[:256]

        a, b, c = q_raw[..., 0:1], q_raw[..., 1:2], q_raw[..., 2:3]
        quats_all = torch.cat(
            [
                torch.sqrt(1 - a) * torch.sin(2 * torch.pi * b),
                torch.sqrt(1 - a) * torch.cos(2 * torch.pi * b),
                torch.sqrt(a) * torch.sin(2 * torch.pi * c),
                torch.sqrt(a) * torch.cos(2 * torch.pi * c),
            ],
            -1,
        )

        opacities_all = torch.sigmoid(x[:, 13:14])[:256]
        rgb = x[:, 10:13]

        flows = []
        means_ = means.clone()
        for t in tqdm(range(T - 1), desc="compute flow"):
            flow = dense_flow_one_frame(
                means_,
                mean_deltas[t],
                quats_all,
                scales,
                opacities_all,
                view,
                K,
                W, H,
                device=device)
            flows.append(flow)
            means_ = means_ + mean_deltas[t]
        
        flows = torch.stack(flows, dim=0)
        batch_flows.append(flows)
    batch_flows = torch.stack(batch_flows, dim=0)

    return batch_flows

def track_points(points_xy, flows):
    """
    Track points through a sequence of optical flow fields.
    :param points_xy: (N,2) array of points to track
    :param flows: (T,H,W,2) array of optical flow fields
    :return: (T,N,2) array of tracked points
    """
    T = len(flows) + 1
    N = len(points_xy)
    tracked_points = np.zeros((T, N, 2), dtype=np.float32)
    tracked_points[0] = points_xy[:, 1:][:, ::-1]
    for t in range(1, T):
        u = cv2.remap(flows[t-1][..., 0], tracked_points[t-1, :, 0], tracked_points[t-1, :, 1], cv2.INTER_LINEAR)
        v = cv2.remap(flows[t-1][..., 1], tracked_points[t-1, :, 0], tracked_points[t-1, :, 1], cv2.INTER_LINEAR)
        tracked_points[t] = (tracked_points[t-1] + np.concatenate([u, v], axis=-1)).clip(0, 224)

    return tracked_points

class ZeroshotLitModule(LightningModule):
    """
    Lightning module for training GPT models.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False,)
        self.cfg = self.hparams.cfg

        self.num_frames = self.cfg.seq_length if (self.cfg.dataset_type=="video" or self.cfg.dataset_type=="video-vjepa") else 1
        import chewbacca.models.components.mae.models_mae_video as models_mae
        self.encoder = models_mae.__dict__[self.cfg.model_name](
                                                                norm_pix_loss=True,
                                                                img_size=self.cfg.input_size,
                                                                number_of_frames=self.num_frames,
                                                                num_gaussian=self.cfg.vocab_size,
                                                                scale_factor=self.cfg.scale_factor,
                                                                scale_vocab=self.cfg.scale_vocab,
                                                                deltas_reg_weight=self.cfg.deltas_reg_weight,
                                                                random_frames=self.cfg.random_frames,
                                                                mean_deltas=self.cfg.mean_deltas,
                                                                rgb_deltas=self.cfg.rgb_deltas,
                                                                rgb_deltas_scale=self.cfg.rgb_deltas_scale,
                                                                upsample_gaussians=self.cfg.upsample_gaussians,
                                                                spawning=self.cfg.spawning,
                                                                frame_zero=self.cfg.frame_zero,
                                                                pairwise_random_frames=self.cfg.pairwise_random_frames,
                                                                videomae=self.cfg.videomae,
                                                                mae_st=self.cfg.mae_st,
                                                                training_type=self.cfg.training_type,
                                                                freeze_encoder=self.cfg.finetune_params.freeze_encoder,
                                                            )
            
        
        num_hidden_layers = len(self.encoder.blocks)
            
        hsize = self.encoder.patch_embed.proj.weight.shape[0]

        # create folders for storing results
        os.makedirs(self.cfg.storage_folder + "/results/", exist_ok=True)
        os.makedirs(self.cfg.storage_folder + "/tests/", exist_ok=True)
        os.makedirs(self.cfg.storage_folder + "/videos/", exist_ok=True)
        log.info("Storage folder : " + self.cfg.storage_folder)

        self.average_jaccard = MeanMetric()
        self.avg_distance = MeanMetric()
        self.average_pts_within_thresh = MeanMetric()
        self.occlusion_accuracy = MeanMetric()
        self.occ_tp = SumMetric()
        self.occ_fp = SumMetric()
        self.occ_fn = SumMetric()
  
    def forward_loss(self, imgs, pred, mask, frame_num=None, deltas=None, additional_data=None):
        return torch.tensor(0).to(imgs.device).to(imgs.dtype)

    def step(self, batch: Any, batch_idx: int, return_images=False):
        video = batch[0] # bs, 3, t, h, w
        init_queries = batch[1]
        finetune_target = batch[2] # dataset index
        occluded_points = batch[3]

        device = video.device
        dtype = video.dtype

        latent, mask, ids_restore, latent_layers = self.encoder.forward_encoder(video, mask_ratio=0.0)

        imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device=device).to(dtype=dtype)
        imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device=device).to(dtype=dtype)
        imgs = (video * imagenet_std[None, :, None, None, None]) + imagenet_mean[None, :, None, None, None]
        imgs = (imgs.permute(0, 2, 3, 4, 1).cpu().numpy() * 255).astype(np.uint8)

        preds_all = []

        with torch.no_grad():
            x_points = self.encoder.forward_decoder(latent, ids_restore)
                
            pred_, primitives = self.encoder.forward_render(x_points, limit_gaussian_z=-1, return_primitives=True)
            primitives["video"] = imgs
            primitives["renders"] = (pred_.detach().cpu().numpy() * 255).astype(np.uint8)
        
        # zeroshot tracking
        scale_factor = np.array([1, self.cfg.input_size, self.cfg.input_size])

        batch_flows = get_zeroshot_flow(primitives)
        x_points = []
        occlusions_pred = []
        
        for i in range(batch_flows.shape[0]):
            queries = init_queries[i].cpu().numpy()
            valid_queries = queries[queries[:, 0] != -1] * scale_factor

            if valid_queries.shape[0] == 0:
                x_points.append(np.zeros((batch_flows.shape[1]+1, queries.shape[0], 2)))
                occlusions_pred.append(np.zeros((batch_flows.shape[1]+1, queries.shape[0])))
                continue
            
            tracks = track_points(valid_queries, batch_flows[i].cpu().numpy())
            tracks = np.concatenate([
                tracks, 
                np.zeros((tracks.shape[0], queries.shape[0] - valid_queries.shape[0], 2))
            ], axis=1)
            tracks = tracks.clip(0, self.cfg.input_size)

            x_points.append(tracks)
            occlusions_pred.append(np.zeros_like(tracks[:, :, 0]))

        x_points = np.stack(x_points, axis=0)
        x_points = x_points.transpose(0, 2, 1, 3)
        occlusions_pred = np.stack(occlusions_pred, axis=0)
        occlusions_pred = occlusions_pred.transpose(0, 2, 1)

        # visualize tracking
        final_image_ = self.visualize_point_tracking(video, x_points, init_queries, finetune_target, occluded_points, occlusions_pred)
        final_image_ = np.concatenate(final_image_, axis=1)
        final_image_ = [i for i in final_image_]

        train_ = "train" if self.training else "val"
        global_rank = self.trainer.global_rank
        if global_rank==0 and not self.cfg.inference.testing:

            clip = ImageSequenceClip(final_image_, fps=1).resize(2)  # fps can be adjusted to your need
            clip.write_videofile(f"{self.cfg.storage_folder}/results/{train_}_{self.current_epoch}_{batch_idx}_video.mp4", codec='libx264')

        if self.cfg.inference.testing:
            if self.cfg.inference.save_predictions:
                imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device=video.device).to(dtype=video.dtype)
                imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device=video.device).to(dtype=video.dtype)
                imgs = (video * imagenet_std[None, :, None, None, None]) + imagenet_mean[None, :, None, None, None]
                imgs = (imgs.permute(0, 2, 3, 4, 1).cpu().numpy() * 255).astype(np.uint8)
                data = {"video": imgs, "init_queries": init_queries, "points_pred": x_points, "occlusions_pred": occlusions_pred, "points_gt": finetune_target, "occlusions_gt": occluded_points}
                torch.save(data, f"{self.cfg.storage_folder}/tests/eval_{self.cfg.inference.context_length}_{self.current_epoch}_{batch_idx}.pt")
                
            title = f"eval_{self.cfg.inference.context_length}" if "eval" in self.cfg.training_type else "kubric"
            clip = ImageSequenceClip(final_image_, fps=1).resize(2)  # fps can be adjusted to your need
            clip.write_videofile(f"{self.cfg.storage_folder}/tests/{title}_{self.current_epoch}_{batch_idx}_video.mp4", codec='libx264')

        # default return values
        # check if loss_index.mean is nan
        loss_dict  = {}
        logits_    = {}

        extras_    = {"x_points": x_points, "occluded_pred": occlusions_pred}

        return loss_dict, extras_, logits_

    def on_train_epoch_start(self) -> None:
        torch.cuda.empty_cache()

    def training_step(self, batch: Any, batch_idx: int):
        return {}

    def on_train_epoch_end(self):
        log.info("\n " + self.cfg.storage_folder +  " : Training epoch " + str(self.current_epoch) + " ended.")

    def on_validation_epoch_start(self) -> None:
        torch.cuda.empty_cache()

    def validation_step(self, batch: Any, batch_idx: int):
        metric_prefix = "test" if self.cfg.inference.testing else "val"
        loss_dict, extra, logits_cls = self.step(batch, batch_idx, return_images=True)
        
        query_points = batch[1].cpu().numpy()
        target_points = batch[2].cpu().numpy()
        occluded = batch[3].cpu().numpy()
        filtered_x_points = extra["x_points"]
        filtered_x_points[query_points[:, :, 0] == -1] = 0
        query_points[query_points[:, :, 0] == -1] = 0

        # Update Jaccard metric
        pred_points = filtered_x_points
        pred_visible = ~(extra.get("occluded_pred", np.zeros_like(query_points[:,:,0])) > 0.5)
        true_points = target_points
        true_visible = ~(occluded>0.5)  # Assuming batch[3] contains occlusion information
                    
        for i in range(query_points.shape[0]):
            first_invalid_index = np.where(query_points[i, :, 0] == -1)[0]

            if len(first_invalid_index) == 0:
                first_invalid_index = true_points.shape[1]
            else:
                first_invalid_index = first_invalid_index[0]
            first_n_frames = self.cfg.seq_length # 5
            query_points_ = query_points[i:i+1, :first_invalid_index]
            query_points_[:, :, 1:] *= self.cfg.input_size

            pred_points_ = pred_points[i:i+1, :first_invalid_index, :first_n_frames]
            pred_visible_ = pred_visible[i:i+1, :first_invalid_index, :first_n_frames]

            true_points_ = true_points[i:i+1, :first_invalid_index, :first_n_frames] * self.cfg.input_size
            true_visible_ = true_visible[i:i+1, :first_invalid_index, :first_n_frames]

            if set((1 - (~true_visible_)).flatten().tolist()) == {0}:
                log.info(f"Skipping batch {batch_idx} for point tracking, all points are occluded")
                continue
            metrics = compute_tapvid_metrics(query_points_, (~true_visible_), true_points_, 
                                            (~pred_visible_), pred_points_)
            
            self.average_jaccard.update(metrics["average_jaccard"].mean())
            self.avg_distance.update(metrics["avg_distance"].mean())
            self.average_pts_within_thresh.update(metrics["average_pts_within_thresh"].mean())
            self.occlusion_accuracy.update(metrics["occlusion_accuracy"].mean())

            self.occ_fp.update(metrics["occ_fp"].sum())
            self.occ_tp.update(metrics["occ_tp"].sum())
            self.occ_fn.update(metrics["occ_fn"].sum())

        occ_tp = self.occ_tp.compute()
        occ_fp = self.occ_fp.compute()
        occ_fn = self.occ_fn.compute()
        precision = occ_tp / (occ_tp + occ_fp)
        recall = occ_tp / (occ_tp + occ_fn)
        occ_f1 = 2 * ((precision * recall) / (precision + recall))
        self.log(f"{metric_prefix}/occ_f1", occ_f1, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        jaccard_score = self.average_jaccard.compute()
        self.log(f"{metric_prefix}/jaccard", jaccard_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        avg_distance = self.avg_distance.compute()
        self.log(f"{metric_prefix}/avg_distance", avg_distance, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        avg_pts_within_thresh = self.average_pts_within_thresh.compute()
        self.log(f"{metric_prefix}/avg_pts_within_thresh", avg_pts_within_thresh, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        occlusion_accuracy = self.occlusion_accuracy.compute()
        self.log(f"{metric_prefix}/occlusion_accuracy", occlusion_accuracy, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # for terminal logging
        self.log_iter_stats(batch_idx)

        # deleting tensors
        del loss_dict, batch

        return {}

    def on_validation_epoch_end(self):
        log.info("\n " + self.cfg.storage_folder +  " : Validation epoch " + str(self.current_epoch) + " ended.")
        metric_prefix = "test" if self.cfg.inference.testing else "val"

        log.info("Jaccard Score : " + str(self.average_jaccard.compute().item()))
        log.info("Avg Distance : " + str(self.avg_distance.compute().item()))
        log.info("Avg Points within Thresh : " + str(self.average_pts_within_thresh.compute().item()))
        log.info("Occlusion Accuracy : " + str(self.occlusion_accuracy.compute().item()))

        occ_tp = self.occ_tp.compute()
        occ_fp = self.occ_fp.compute()
        occ_fn = self.occ_fn.compute()
        precision = occ_tp / (occ_tp + occ_fp)
        recall = occ_tp / (occ_tp + occ_fn)
        occ_f1 = 2 * ((precision * recall) / (precision + recall))
        log.info("Occlusion F1 Score : " + str(occ_f1.item()))

        # reset the metric
        self.average_jaccard.reset()
        self.avg_distance.reset()
        self.average_pts_within_thresh.reset()
        self.occlusion_accuracy.reset()
        self.occ_fp.reset()
        self.occ_tp.reset()
        self.occ_fn.reset()
            
    def visualize_point_tracking(self, video, x_points, query_points, target_points, occluded_points, occlusions_pred):
        # renormalize to 0,1
        imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device=video.device).to(dtype=video.dtype)
        imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device=video.device).to(dtype=video.dtype)
        imgs = (video * imagenet_std[None, :, None, None, None]) + imagenet_mean[None, :, None, None, None]
        imgs = (imgs.permute(0, 2, 3, 4, 1).cpu().numpy() * 255).astype(np.uint8)

        scale_factor = np.array([self.cfg.input_size, self.cfg.input_size])[np.newaxis, np.newaxis, :]

        final_image_ = []
        for i in range(len(x_points)):
            first_invalid_index = torch.where(query_points[i, :, 0] == -1)[0]
            if len(first_invalid_index) == 0:
                first_invalid_index = x_points.shape[1]
            else:
                first_invalid_index = first_invalid_index[0]

            active_tracks_pred = x_points[i, :first_invalid_index]
            active_tracks_true = target_points[i, :first_invalid_index]

            active_occluded_points = occluded_points[i, :first_invalid_index]
            active_occlusions_pred = occlusions_pred[i, :first_invalid_index]

            painted_frames_pred = tapvid_viz_utils.paint_point_track(
                imgs[i],
                active_tracks_pred,
                ~(active_occlusions_pred > 0.5),
            )

            painted_frames_true = tapvid_viz_utils.paint_point_track(
                imgs[i],
                active_tracks_true.cpu().numpy() * scale_factor,
                ~(active_occluded_points.cpu().numpy() > 0.5),
            )

            painted_frames_blank = tapvid_viz_utils.paint_point_track(
                np.ones_like(imgs[i]) * 255,
                active_tracks_pred,
                ~(active_occlusions_pred > 0.5),
            )

            final_image = np.concatenate([painted_frames_true, painted_frames_pred, painted_frames_blank], axis=2)
            final_image_.append(final_image)
        return final_image_
    def log_iter_stats(self, cur_iter):

        if(cur_iter%self.cfg.log_frequency != 0):
            return 0

        mem_usage = gpu_mem_usage()
        try:
            stats = {
                "epoch": "{}/{}".format(self.current_epoch, self.trainer.max_epochs),
                "iter": "{}/{}".format(cur_iter + 1, self.trainer.num_training_batches if self.training else self.trainer.num_val_batches),
                "time": "%.4f"%(self.timer.time_elapsed()-self.timer_last_iter),
                # "eta": "%.4f"%(self.timer.time_remaining()),
                "lr": self.trainer.optimizers[0].param_groups[0]['lr'] if self.training else 0,
                "mem": int(np.ceil(mem_usage)),
            }
            self.timer_last_iter = self.timer.time_elapsed()
        except:
            # add self.timer from trainer.callbacks
            for cb_ in self.trainer.callbacks:
                if(cb_.__class__.__name__ == "Timer"):
                    self.timer = cb_
            self.timer_last_iter = self.timer.time_elapsed()
            stats = {}

        log.info(stats)



    def get_param_groups(self):
        
        def _get_layer_decay(name):
            layer_id = None
            depth = len(self.encoder.blocks)
            if "class_token" in name or "pos_embed" in name or "patch_embed.proj" in name:
                layer_id = 0
            elif ("_head" in name):
                layer_id = depth + 1
            elif ".blocks." in name:
                layer_id = int(name.split(".blocks.")[1].split(".")[0]) + 1
            else:
                layer_id = depth + 1
            layer_decay = self.cfg.layer_decay ** (depth + 1 - layer_id)
            return layer_id, layer_decay

        non_bn_parameters_count = 0
        zero_parameters_count = 0
        no_grad_parameters_count = 0
        parameter_group_names = {}
        parameter_group_vars = {}

        for name, p in self.named_parameters():
            print(name)
            if not p.requires_grad:
                group_name = "no_grad"
                no_grad_parameters_count += 1
                continue
            name = name[len("module."):] if name.startswith("module.") else name
            if ((len(p.shape) == 1 or name.endswith(".bias")) and self.cfg.ZERO_WD_1D_PARAM):
                layer_id, layer_decay = _get_layer_decay(name)
                group_name = "layer_%d_%s" % (layer_id, "zero")
                weight_decay = 0.0
                zero_parameters_count += 1
            else:
                layer_id, layer_decay = _get_layer_decay(name)
                group_name = "layer_%d_%s" % (layer_id, "non_bn")
                weight_decay = self.cfg.weight_decay
                non_bn_parameters_count += 1

            if group_name not in parameter_group_names:
                parameter_group_names[group_name] = {
                    "weight_decay": weight_decay,
                    "params": [],
                    "lr": self.cfg.lr * self.lr_scaler * layer_decay,
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": weight_decay,
                    "params": [],
                    "lr": self.cfg.lr * self.lr_scaler * layer_decay,
                }
            parameter_group_names[group_name]["params"].append(name)
            parameter_group_vars[group_name]["params"].append(p)
        
        import json
        print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        optim_params = list(parameter_group_vars.values())
        return optim_params

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        # linear learning rate scaling for multi-gpu
        if(self.trainer.num_devices * self.trainer.num_nodes>1 and self.cfg.apply_linear_scaling):
            self.lr_scaler = self.trainer.num_devices * self.trainer.num_nodes * self.trainer.accumulate_grad_batches * self.cfg.train_batch_size / 256
        else:
            self.lr_scaler = 1

        # print devices, nodes, batchsize, and total number of parameters
        log.info("num_devices: {}, num_nodes: {}, accumulate_grad_batches: {}, train_batch: {}".format(self.trainer.num_devices, self.trainer.num_nodes, self.trainer.accumulate_grad_batches, self.cfg.train_batch_size))
        log.info("Linear LR scaling factor: {}".format(self.lr_scaler))
        log.info(f"Total number of trainable parameters : {sum(p.numel() / 1024 / 1024 for p in self.trainer.model.parameters() if p.requires_grad):.6f} million")

        if(self.cfg.layer_decay is not None):
            optim_params = self.get_param_groups()
        else:
            optim_params = [{'params': filter(lambda p: p.requires_grad, self.trainer.model.parameters()), 'lr': self.cfg.lr * self.lr_scaler}]

        if(self.cfg.solver=="LARS"):
            optim_params = [{'params': filter(lambda p: p.requires_grad, self.linear_layer.parameters()), 'lr': self.cfg.lr * self.lr_scaler}]
            optimizer = LARS(params=optim_params, weight_decay=self.cfg.weight_decay)
        elif(self.cfg.solver=="LAMB"):
            optim_params = [{'params': filter(lambda p: p.requires_grad, self.linear_layer.parameters()), 'lr': self.cfg.lr * self.lr_scaler}]
            optimizer = Lamb(params=optim_params, weight_decay=self.cfg.weight_decay)
        elif(self.cfg.solver=="AdamW"):
            optim_params = add_weight_decay(self.trainer.model, self.cfg.weight_decay)
            optimizer = optim.AdamW(params=optim_params, lr=self.cfg.lr * self.lr_scaler, betas=(0.9, 0.95))
        elif(self.cfg.solver=="AdamW-layer"):
            optim_params = add_weight_decay(self.trainer.model, self.cfg.weight_decay)
            optimizer = optim.AdamW(params=optim_params, lr=self.cfg.lr * self.lr_scaler, betas=(0.9, 0.95))
        else:
            raise NotImplementedError("Unknown solver : " + self.cfg.solver)

        def warm_start_and_cosine_annealing(epoch):
            if epoch < self.cfg.warmup_epochs:
                lr = (epoch+1) / self.cfg.warmup_epochs
            else:
                lr = 0.5 * (1. + math.cos(math.pi * ((epoch+1) - self.cfg.warmup_epochs) / (self.trainer.max_epochs - self.cfg.warmup_epochs )))
            return lr

        def warm_start_and_cosine_annealing_step(step):
            if step < self.cfg.warmup_steps: # here we use step instead of epoch
                lr = (step+1) / self.cfg.warmup_steps
            else:
                lr = 0.5 * (1. + math.cos(math.pi * ((step+1) - self.cfg.warmup_steps) / (self.trainer.max_epochs*self.trainer.num_training_batches - self.cfg.warmup_steps )))
            return lr

        def inverse_sqrt_annealing(epoch):
            if epoch < self.cfg.warmup_epochs:
                lr = (epoch+1) / self.cfg.warmup_epochs
            else:
                lr = 1/math.sqrt(epoch+1 - self.cfg.warmup_epochs)
            return lr

        def linear_annealing(epoch):
            if epoch < self.cfg.warmup_epochs:
                lr = (epoch+1) / self.cfg.warmup_epochs
            else:
                lr = 1 - (epoch+1 - self.cfg.warmup_epochs)/(self.trainer.max_epochs - self.cfg.warmup_epochs)
            return lr

        if(self.cfg.scheduler == "cosine"):
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[warm_start_and_cosine_annealing for _ in range(len(optim_params))], verbose=False)
        elif(self.cfg.scheduler == "cosine_step"):
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[warm_start_and_cosine_annealing_step for _ in range(len(optim_params))], verbose=False)
        elif(self.cfg.scheduler == "inv_sqrt"):
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[inverse_sqrt_annealing for _ in range(len(optim_params))], verbose=False)
        elif(self.cfg.scheduler == "linear"):
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[linear_annealing for _ in range(len(optim_params))], verbose=False)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.cfg.decay_steps, gamma=self.cfg.decay_gamma, verbose=False)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval" : self.cfg.lr_interval,
                'frequency': 1,
            }
        }


if __name__ == "__main__":
    pass
