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
class attntion_probe(nn.Module):
    def __init__(self, num_heads, in_dim, out_dim):
        super().__init__()
        self.num_heads = num_heads
        self.q_ = nn.Parameter(torch.randn(1, out_dim//num_heads))
        self.wk = nn.Linear(in_dim, out_dim)
        self.wv = nn.Linear(in_dim, out_dim)
        self.scale = 1 / (out_dim // num_heads) ** 0.5

    def forward(self, x):
        # implements multi-head attention with fixed query
        # x: (N, L, D)
        k = self.wk(x).view(x.shape[0], x.shape[1], self.num_heads, -1).transpose(1, 2)
        v = self.wv(x).view(x.shape[0], x.shape[1], self.num_heads, -1).transpose(1, 2)
        q = self.q_.unsqueeze(0)[:, :, None, :].repeat(1, 1, self.num_heads, 1) * self.scale
        attn = torch.einsum('nlhd,nhkd->nhlk', q, k)
        attn = attn.softmax(dim=-1)
        attn_output = torch.einsum('nhlk,nhkd->nhld', attn, v)
        attn_output = attn_output.transpose(1, 2).view(x.shape[0], -1)
        return attn_output

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

def render_soft_assignments(means, quats, scales, opacities,
                            view, K, W, H, chunk=32, device="cuda"):
    """
    Render per-pixel per-Gaussian contribution weights.
    Returns: (H, W, N) torch.float32 on CPU.
    """
    N = means.shape[0]
    if N == 0:
        return torch.zeros(H, W, 0, dtype=torch.float32)

    W_full = torch.zeros(H, W, N, device=device, dtype=torch.float32)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        D = end - start
        if D <= 0:
            continue
        colors = torch.zeros(N, D, device=device, dtype=torch.float32)
        colors[start:end, torch.arange(D, device=device)] = 1.0

        cols, alphas, _ = rasterization(
            means, quats, scales, opacities.squeeze(-1), colors,
            view[None], K[None], width=W, height=H,
            packed=False, render_mode="RGB"
        )
        W_full[:, :, start:end] = cols[0]

    return W_full.detach().cpu()

def project_centers(means, quats, scales, view, K, W, H, device="cuda"):
    """
    Project Gaussian centers to pixel space and depth.
    Returns:
        xy: (N, 2) torch.float32 on CPU
        z:  (N,)  torch.float32 on CPU
    """
    _, xy, _, depths, _ = fully_fused_projection(
        means, None, quats, scales, view[None], K[None], W, H
    )
    return xy.squeeze(0).detach().cpu(), depths.squeeze(0).detach().cpu()

def _bilinear_at(img, x, y):
    """
    Bilinear sample from a 2D map (numpy array).
    """
    H, W = img.shape[:2]
    x = float(np.clip(x, 0, W - 1))
    y = float(np.clip(y, 0, H - 1))
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, W - 1)
    y1 = min(y0 + 1, H - 1)
    wx = x - x0
    wy = y - y0
    if img.ndim == 3:
        return ((1 - wx) * (1 - wy) * img[y0, x0] +
                wx * (1 - wy) * img[y0, x1] +
                (1 - wx) * wy * img[y1, x0] +
                wx * wy * img[y1, x1])
    return ((1 - wx) * (1 - wy) * img[y0, x0] +
            wx * (1 - wy) * img[y0, x1] +
            (1 - wx) * wy * img[y1, x0] +
            wx * wy * img[y1, x1])

def track_points_occlusion(points_xy, flows, W_seq, XY_seq,
                           topk=8, tau_vis=0.05, tau_re=0.08,
                           visible_mode="hybrid", beta_hybrid=0.3,
                           beta_scale_with_mass=True, eps=1e-8):
    """
    Occlusion-aware tracking. Returns trajectories, fixed owner ids, mixture weights, and visibility flags.
    """
    flows_np = np.asarray(flows, dtype=np.float32)
    W_seq_np = np.asarray(W_seq, dtype=np.float32)
    XY_seq_np = np.asarray(XY_seq, dtype=np.float32)

    Tm1 = flows_np.shape[0] if flows_np.ndim >= 4 else 0
    H = flows_np.shape[1] if Tm1 > 0 else (W_seq_np.shape[1] if W_seq_np.ndim == 4 else 0)
    W = flows_np.shape[2] if Tm1 > 0 else (W_seq_np.shape[2] if W_seq_np.ndim == 4 else 0)
    T = Tm1 + 1
    G = W_seq_np.shape[-1] if W_seq_np.ndim == 4 else 0
    N = points_xy.shape[0]

    max_g = max(G, 1)
    topk_eff = max(1, min(topk, max_g))

    traj = np.zeros((T, N, 2), dtype=np.float32)
    vis = np.zeros((T, N), dtype=np.float32)
    pis = np.zeros((T, N, topk_eff), dtype=np.float32)
    ids_fixed = np.zeros((N, topk_eff), dtype=np.int32)
    offs = np.zeros((N, topk_eff, 2), dtype=np.float32)

    if N == 0:
        return traj, ids_fixed, pis, vis

    traj[0] = points_xy.astype(np.float32)

    def topk_indices_at(Wimg, x, y, k):
        if Wimg.ndim == 0 or Wimg.size == 0:
            return np.zeros((k,), dtype=np.int32), np.zeros((k,), dtype=np.float32)
        w_vec = _bilinear_at(Wimg, x, y)
        if np.isscalar(w_vec):
            w_vec = np.array([float(w_vec)], dtype=np.float32)
        if w_vec.ndim == 0:
            w_vec = w_vec[None]
        idx = np.argpartition(-w_vec, kth=min(k - 1, len(w_vec) - 1))[:k]
        idx = idx[np.argsort(-w_vec[idx])]
        w_k = w_vec[idx]
        if len(idx) < k:
            pad_idx = idx[-1] if len(idx) else 0
            idx = np.pad(idx, (0, k - len(idx)), constant_values=pad_idx)
            w_k = np.pad(w_k, (0, k - len(w_k)), constant_values=0.0)
        return idx.astype(np.int32), w_k.astype(np.float32)

    if Tm1 > 0 and W_seq_np.ndim == 4 and W_seq_np.shape[0] > 0:
        W0 = W_seq_np[0]
    else:
        W0 = np.zeros((H, W, G), dtype=np.float32)

    for j in range(N):
        x0, y0 = traj[0, j]
        idx_k, w_k = topk_indices_at(W0, x0, y0, topk_eff)
        ids_fixed[j] = idx_k
        mass0 = float(np.sum(w_k))
        vis[0, j] = 1.0 if mass0 >= tau_vis else 0.0

    for t in range(1, T):
        F_tm1 = flows_np[t - 1] if Tm1 > 0 else np.zeros((H, W, 2), dtype=np.float32)
        W_tm1 = W_seq_np[t - 1] if W_seq_np.shape[0] >= t else np.zeros((H, W, G), dtype=np.float32)
        XY_t = XY_seq_np[t - 1] if XY_seq_np.shape[0] >= t else np.zeros((G, 2), dtype=np.float32)

        for j in range(N):
            x_prev, y_prev = traj[t - 1, j]
            owners = ids_fixed[j]
            cxcy = XY_t[owners] if XY_t.size else np.zeros((topk_eff, 2), dtype=np.float32)

            w_vec_full = _bilinear_at(W_tm1, x_prev, y_prev) if W_tm1.size else np.zeros((G,), dtype=np.float32)
            if np.isscalar(w_vec_full):
                w_vec_full = np.array([float(w_vec_full)], dtype=np.float32)
            if w_vec_full.ndim == 0:
                w_vec_full = w_vec_full[None]
            if w_vec_full.size == 0:
                w_k = np.zeros((topk_eff,), dtype=np.float32)
            else:
                w_k = w_vec_full[owners]
            mass = float(np.sum(w_k))
            if mass > 0:
                pi_k = (w_k / (mass + eps)).astype(np.float32)
            else:
                pi_k = np.full((topk_eff,), 1.0 / topk_eff, dtype=np.float32)
            pis[t - 1, j] = pi_k

            if t == 1 and not np.any(offs[j]):
                p_prim_next = np.sum(cxcy * pi_k[:, None], axis=0)
            else:
                p_prim_next = np.sum((cxcy + offs[j]) * pi_k[:, None], axis=0)

            if mass >= tau_vis:
                if visible_mode == "flow":
                    u, v = _bilinear_at(F_tm1, x_prev, y_prev)
                    p_flow_next = np.array([x_prev + u, y_prev + v], dtype=np.float32)
                    p_next = p_flow_next
                elif visible_mode == "mixture":
                    p_next = p_prim_next.astype(np.float32)
                else:
                    u, v = _bilinear_at(F_tm1, x_prev, y_prev)
                    p_flow_next = np.array([x_prev + u, y_prev + v], dtype=np.float32)
                    if beta_scale_with_mass:
                        beta = float(beta_hybrid) * max(0.0, 1.0 - min(1.0, mass))
                    else:
                        beta = float(beta_hybrid)
                    p_next = ((1.0 - beta) * p_flow_next + beta * p_prim_next).astype(np.float32)
                vis[t, j] = 1.0
            else:
                p_next = p_prim_next.astype(np.float32)
                vis[t, j] = 0.0

            if W and H:
                x_next = float(p_next[0])
                y_next = float(p_next[1])
                if (x_next < 0) or (x_next >= W) or (y_next < 0) or (y_next >= H):
                    vis[t, j] = 0.0
            traj[t, j] = p_next
            offs[j] = p_next[None, :] - cxcy

    if T >= 2:
        pis[-1] = pis[-2]
    return traj, ids_fixed, pis, vis

def get_zeroshot_flow(data, chunk=32):
    device = data["x_points"].device
    video = data["video"]
    if torch.is_tensor(video):
        T = video.shape[1]
        H, W = video.shape[2:4]
    else:
        video_np = np.asarray(video)
        T = video_np.shape[1]
        H, W = video_np.shape[2:4]

    if T <= 0:
        raise ValueError("Video must contain at least one frame.")

    f = 0.5 * W / math.tan(math.pi / 4)
    K = torch.tensor([[f, 0, W / 2],
                      [0, W / H * f, H / 2],
                      [0, 0, 1]], device=device).float()
    view = torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 8],
                         [0, 0, 0, 1]], device=device).float()

    batch_flows = []
    batch_weights = []
    batch_centers = []
    for i in range(data["x_points"].shape[0]):
        x = data["x_points"][i]

        allmeans = 5 * torch.tanh(x[:, :3])
        total_gaussians = allmeans.shape[0]
        if total_gaussians % T != 0:
            raise ValueError(
                f"Expected total gaussians ({total_gaussians}) to be divisible by number of frames ({T})."
            )
        num_gaussians = total_gaussians // T

        means = allmeans[:num_gaussians]
        if T > 1:
            mean_deltas = allmeans[num_gaussians:].reshape(T - 1, num_gaussians, 3) / 10
        else:
            mean_deltas = allmeans.new_zeros((0, num_gaussians, 3))
        scales = torch.sigmoid(x[:, 3:6])[:num_gaussians]
        q_raw = torch.sigmoid(x[:, 6:10])[:num_gaussians]

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

        opacities_all = torch.sigmoid(x[:, 13:14])[:num_gaussians]

        flows = []
        weights = []
        centers = []
        means_ = means.clone()
        for t in tqdm(range(max(T - 1, 0)), desc="compute flow", leave=False):
            flow_t = dense_flow_one_frame(
                means_,
                mean_deltas[t],
                quats_all,
                scales,
                opacities_all,
                view,
                K,
                W, H,
                device=device)
            flows.append(flow_t.detach().cpu())

            weights_t = render_soft_assignments(
                means_, quats_all, scales, opacities_all,
                view, K, W, H, chunk=chunk, device=device
            ).float()
            weights.append(weights_t)

            means_next = means_ + mean_deltas[t]
            xy_next, _ = project_centers(
                means_next, quats_all, scales, view, K, W, H, device=device
            )
            centers.append(xy_next.float())

            means_ = means_next
        if flows:
            flows_tensor = torch.stack(flows, dim=0)
        else:
            flows_tensor = torch.zeros((0, H, W, 2), dtype=torch.float32)
        if weights:
            weights_tensor = torch.stack(weights, dim=0)
        else:
            weights_tensor = torch.zeros((0, H, W, num_gaussians), dtype=torch.float32)
        if centers:
            centers_tensor = torch.stack(centers, dim=0)
        else:
            centers_tensor = torch.zeros((0, num_gaussians, 2), dtype=torch.float32)

        batch_flows.append(flows_tensor)
        batch_weights.append(weights_tensor)
        batch_centers.append(centers_tensor)

    batch_flows = torch.stack(batch_flows, dim=0)
    batch_weights = torch.stack(batch_weights, dim=0)
    batch_centers = torch.stack(batch_centers, dim=0)

    return batch_flows, batch_weights, batch_centers

def track_points_with_occlusions(points_xyz, flows, weights, centers, image_size,
                                 topk=8, tau_vis=0.05, tau_re=0.08,
                                 visible_mode="hybrid", beta_hybrid=0.3,
                                 beta_scale_with_mass=True):
    """
    Wrapper that converts query points and runs occlusion-aware tracking.
    Returns:
        tracks: (T, N, 2)
        occlusions: (T, N) with 1 indicating occluded.
    """
    flows_np = np.asarray(flows, dtype=np.float32)
    weights_np = np.asarray(weights, dtype=np.float32)
    centers_np = np.asarray(centers, dtype=np.float32)
    if flows_np.ndim >= 4:
        T = flows_np.shape[0] + 1
    else:
        T = 1

    points_arr = np.asarray(points_xyz, dtype=np.float32)
    if points_arr.size == 0:
        return (
            np.zeros((T, 0, 2), dtype=np.float32),
            np.zeros((T, 0), dtype=np.float32),
        )

    points_xy = points_arr[:, [2, 1]]  # (x, y)
    traj, _, _, vis = track_points_occlusion(
        points_xy,
        flows_np,
        weights_np,
        centers_np,
        topk=topk,
        tau_vis=tau_vis,
        tau_re=tau_re,
        visible_mode=visible_mode,
        beta_hybrid=beta_hybrid,
        beta_scale_with_mass=beta_scale_with_mass,
    )
    traj = traj.astype(np.float32)
    if image_size is not None:
        traj = np.clip(traj, 0, float(image_size))
    occlusions = (vis < 0.5).astype(np.float32)
    return traj, occlusions

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

        # just to match weights from training:

        if self.cfg.videomae:
            num_hidden_layers = len(self.encoder.videomae_model.encoder.layer) + 1
        elif self.cfg.mae_st:
            num_hidden_layers = len(self.encoder.mae_st_model.blocks)
        else:
            num_hidden_layers = len(self.encoder.blocks)
            
        hsize = self.encoder.patch_embed.proj.weight.shape[0]

        if "attn" in self.cfg.training_type:
            self.linear_layer = nn.ModuleList([torch.nn.Sequential(attntion_probe(8, hsize, hsize), nn.Linear(hsize,self.cfg.num_classes)) for i in range(num_hidden_layers)])
        else:
            self.linear_layer = nn.ModuleList([torch.nn.Sequential(torch.nn.BatchNorm1d(hsize, affine=False, eps=1e-6), nn.Linear(hsize,self.cfg.num_classes)) for i in range(num_hidden_layers)])
        self.test_rfid = FrechetInceptionDistance()
  
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

        inference_cfg = getattr(self.cfg, "inference", None)
        tau_vis = 0.5
        tau_re = 0.08
        tracking_topk = 8
        visible_mode = "hybrid"
        beta_hybrid = 0.3
        beta_scale_with_mass = True
        if inference_cfg is not None:
            tau_vis_val = getattr(inference_cfg, "tau_vis", None)
            if tau_vis_val is not None:
                tau_vis = float(tau_vis_val)
            tau_re_val = getattr(inference_cfg, "tau_re", None)
            if tau_re_val is not None:
                tau_re = float(tau_re_val)
            topk_val = getattr(inference_cfg, "tracking_topk", None)
            if topk_val is not None:
                tracking_topk = int(topk_val)
            visible_mode_val = getattr(inference_cfg, "visible_mode", None)
            if visible_mode_val is not None:
                visible_mode = str(visible_mode_val).lower()
            beta_hybrid_val = getattr(inference_cfg, "beta_hybrid", None)
            if beta_hybrid_val is not None:
                beta_hybrid = float(beta_hybrid_val)
            beta_scale_val = getattr(inference_cfg, "beta_scale_with_mass", None)
            if beta_scale_val is not None:
                if isinstance(beta_scale_val, str):
                    beta_scale_with_mass = beta_scale_val.lower() in {"1", "true", "yes", "y"}
                else:
                    beta_scale_with_mass = bool(beta_scale_val)
        if visible_mode not in {"flow", "mixture", "hybrid"}:
            visible_mode = "hybrid"

        if torch.is_tensor(init_queries):
            init_queries_np = init_queries.detach().cpu().numpy()
        else:
            init_queries_np = np.asarray(init_queries)
        if torch.is_tensor(occluded_points):
            occluded_np = occluded_points.detach().cpu().numpy()
        else:
            occluded_np = np.asarray(occluded_points)

        batch_flows, batch_weights, batch_centers = get_zeroshot_flow(primitives)
        x_points = []
        occlusions_pred = []
        
        for i in range(batch_flows.shape[0]):
            queries = init_queries_np[i]
            valid_mask = queries[:, 0] != -1
            valid_queries = (queries[valid_mask] * scale_factor) if np.any(valid_mask) else np.empty((0, 3))
            valid_queries = valid_queries.astype(np.float32, copy=False)

            flows_np = batch_flows[i].numpy()
            weights_np = batch_weights[i].numpy()
            centers_np = batch_centers[i].numpy()
            num_frames = flows_np.shape[0] + 1 if flows_np.ndim >= 4 else 1

            if valid_queries.shape[0] == 0:
                x_points.append(np.zeros((num_frames, queries.shape[0], 2), dtype=np.float32))
                occlusions_pred.append(np.zeros((num_frames, queries.shape[0]), dtype=np.float32))
                continue

            tracks, occlusion = track_points_with_occlusions(
                valid_queries,
                flows_np,
                weights_np,
                centers_np,
                self.cfg.input_size,
                topk=tracking_topk,
                tau_vis=tau_vis,
                tau_re=tau_re,
                visible_mode=visible_mode,
                beta_hybrid=beta_hybrid,
                beta_scale_with_mass=beta_scale_with_mass,
            )

            tracks_full = np.zeros((tracks.shape[0], queries.shape[0], 2), dtype=np.float32)
            occlusion_full = np.zeros((occlusion.shape[0], queries.shape[0]), dtype=np.float32)
            tracks_full[:, valid_mask, :] = tracks
            occlusion_full[:, valid_mask] = occlusion

            x_points.append(tracks_full)
            occlusions_pred.append(occlusion_full)

        x_points = np.stack(x_points, axis=0)
        x_points = x_points.transpose(0, 2, 1, 3)
        occlusions_pred = np.stack(occlusions_pred, axis=0)
        occlusions_pred = occlusions_pred.transpose(0, 2, 1)

        for vid_idx in range(init_queries_np.shape[0]):
            valid_mask = init_queries_np[vid_idx, :, 0] != -1
            total_valid = int(valid_mask.sum())
            if total_valid == 0:
                gt_pct = 0.0
                pred_pct = 0.0
            else:
                gt_vals = np.asarray(occluded_np[vid_idx, valid_mask], dtype=np.float32).reshape(-1)
                pred_vals = np.asarray(occlusions_pred[vid_idx, valid_mask], dtype=np.float32).reshape(-1)
                gt_pct = float(gt_vals.mean() * 100.0) if gt_vals.size else 0.0
                pred_pct = float(pred_vals.mean() * 100.0) if pred_vals.size else 0.0
            print(f"[Occlusion] batch {batch_idx} video {vid_idx}: GT {gt_pct:.2f}% occluded | Pred {pred_pct:.2f}% occluded")

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
                torch.save(data, f"{self.cfg.storage_folder}/tests/gmrw-davis_{self.cfg.inference.context_length}_{self.current_epoch}_{batch_idx}.pt")
                
            # title = f"eval_{self.cfg.inference.context_length}" if "eval" in self.cfg.training_type else "kubric"
            title = f"gmrw-davis_{self.cfg.inference.context_length}"
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
            first_n_frames = 5 # self.cfg.seq_length
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
