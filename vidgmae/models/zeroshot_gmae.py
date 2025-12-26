import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from moviepy.editor import ImageSequenceClip
from omegaconf import DictConfig
from torchmetrics import MeanMetric, SumMetric
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from vidgmae.utils import get_pylogger
from vidgmae.utils.model_utils import compute_tapvid_metrics, gpu_mem_usage
from gsplat import rasterization
from gsplat.cuda._wrapper import (
    fully_fused_projection,
)

from vidgmae.utils import tapvid_viz_utils


log = get_pylogger(__name__)


def dense_flow_one_frame(
    means_t,  # (G,3)
    mean_delta,  # (G,3)   μ(t+1)−μ(t)
    quats,
    scales,
    opacities,
    view,
    K,
    W,
    H,
    idx=None,
    device="cuda",
):
    _, xy_t, _, _, _ = fully_fused_projection(
        means_t, None, quats, scales, view[None], K[None], W, H
    )
    _, xy_t1, _, _, _ = fully_fused_projection(
        means_t + mean_delta, None, quats, scales, view[None], K[None], W, H
    )
    d_pix = (xy_t1 - xy_t).squeeze()
    d_pix = d_pix.to(torch.float32)
    rgbs = torch.zeros(means_t.shape[0], 3).to(device)
    rgbs[:, :2] = d_pix

    dels, alpha, _ = rasterization(
        means_t,
        quats,
        scales,
        opacities.squeeze(-1),
        rgbs,
        view[None],
        K[None],
        width=W,
        height=H,
        packed=False,
        render_mode="RGB",
    )
    flow = dels[0, :, :, :2] * alpha[0, :, :]  # (H,W,2)

    return flow.detach()


def render_soft_assignments(
    means, quats, scales, opacities, view, K, W, H, chunk=32, device="cuda"
):
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
            means,
            quats,
            scales,
            opacities.squeeze(-1),
            colors,
            view[None],
            K[None],
            width=W,
            height=H,
            packed=False,
            render_mode="RGB",
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
        return (
            (1 - wx) * (1 - wy) * img[y0, x0]
            + wx * (1 - wy) * img[y0, x1]
            + (1 - wx) * wy * img[y1, x0]
            + wx * wy * img[y1, x1]
        )
    return (
        (1 - wx) * (1 - wy) * img[y0, x0]
        + wx * (1 - wy) * img[y0, x1]
        + (1 - wx) * wy * img[y1, x0]
        + wx * wy * img[y1, x1]
    )


def track_points_occlusion(
    points_xy, flows, W_seq, XY_seq, topk=8, tau_vis=0.05, beta_hybrid=0.3, eps=1e-8
):
    """
    Occlusion-aware tracking. Returns trajectories, fixed owner ids, mixture weights, and visibility flags.
    """
    flows_np = np.asarray(flows, dtype=np.float32)
    W_seq_np = np.asarray(W_seq, dtype=np.float32)
    XY_seq_np = np.asarray(XY_seq, dtype=np.float32)

    Tm1 = flows_np.shape[0] if flows_np.ndim >= 4 else 0
    H = (
        flows_np.shape[1]
        if Tm1 > 0
        else (W_seq_np.shape[1] if W_seq_np.ndim == 4 else 0)
    )
    W = (
        flows_np.shape[2]
        if Tm1 > 0
        else (W_seq_np.shape[2] if W_seq_np.ndim == 4 else 0)
    )
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
        W_tm1 = (
            W_seq_np[t - 1]
            if W_seq_np.shape[0] >= t
            else np.zeros((H, W, G), dtype=np.float32)
        )
        XY_t = (
            XY_seq_np[t - 1]
            if XY_seq_np.shape[0] >= t
            else np.zeros((G, 2), dtype=np.float32)
        )

        for j in range(N):
            x_prev, y_prev = traj[t - 1, j]
            owners = ids_fixed[j]
            cxcy = (
                XY_t[owners] if XY_t.size else np.zeros((topk_eff, 2), dtype=np.float32)
            )

            w_vec_full = (
                _bilinear_at(W_tm1, x_prev, y_prev)
                if W_tm1.size
                else np.zeros((G,), dtype=np.float32)
            )
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
                u, v = _bilinear_at(F_tm1, x_prev, y_prev)
                p_flow_next = np.array([x_prev + u, y_prev + v], dtype=np.float32)
                beta = float(beta_hybrid) * max(0.0, 1.0 - min(1.0, mass))
                p_next = ((1.0 - beta) * p_flow_next + beta * p_prim_next).astype(
                    np.float32
                )
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
    K = torch.tensor(
        [[f, 0, W / 2], [0, W / H * f, H / 2], [0, 0, 1]], device=device
    ).float()
    view = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 8], [0, 0, 0, 1]], device=device
    ).float()

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
                W,
                H,
                device=device,
            )
            flows.append(flow_t.detach().cpu())

            weights_t = render_soft_assignments(
                means_,
                quats_all,
                scales,
                opacities_all,
                view,
                K,
                W,
                H,
                chunk=chunk,
                device=device,
            ).float()
            weights.append(weights_t)

            means_next = means_ + mean_deltas[t]
            xy_next, _ = project_centers(
                means_next, quats_all, scales, view, K, W, H, device=device
            )
            centers.append(xy_next.float())

            means_ = means_next
        flows_tensor = torch.stack(flows, dim=0)
        weights_tensor = torch.stack(weights, dim=0)
        centers_tensor = torch.stack(centers, dim=0)

        batch_flows.append(flows_tensor)
        batch_weights.append(weights_tensor)
        batch_centers.append(centers_tensor)

    batch_flows = torch.stack(batch_flows, dim=0)
    batch_weights = torch.stack(batch_weights, dim=0)
    batch_centers = torch.stack(batch_centers, dim=0)

    return batch_flows, batch_weights, batch_centers


def track_points_with_occlusions(
    points_xyz,
    flows,
    weights,
    centers,
    image_size,
    topk=8,
    tau_vis=0.05,
    beta_hybrid=0.3,
):
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
        beta_hybrid=beta_hybrid,
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
        self.save_hyperparameters(
            logger=False,
        )
        self.cfg = self.hparams.cfg

        self.num_frames = (
            self.cfg.seq_length
            if (
                self.cfg.dataset_type == "video"
                or self.cfg.dataset_type == "video-vjepa"
            )
            else 1
        )
        import vidgmae.models.components.mae.models_mae_video as models_mae

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

        self.linear_layer = nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.BatchNorm1d(hsize, affine=False, eps=1e-6),
                    nn.Linear(hsize, self.cfg.num_classes),
                )
                for i in range(num_hidden_layers)
            ]
        )
        self.test_rfid = FrechetInceptionDistance()

    def forward_loss(
        self, imgs, pred, mask, frame_num=None, deltas=None, additional_data=None
    ):
        return torch.tensor(0).to(imgs.device).to(imgs.dtype)

    def step(self, batch: Any, batch_idx: int, return_images=False):
        video = batch[0]  # bs, 3, t, h, w
        init_queries = batch[1]
        finetune_target = batch[2]  # dataset index
        occluded_points = batch[3]

        device = video.device
        dtype = video.dtype

        latent, mask, ids_restore, _ = self.encoder.forward_encoder(
            video, mask_ratio=0.0
        )

        imagenet_mean = (
            torch.from_numpy(np.array([0.485, 0.456, 0.406]))
            .to(device=device)
            .to(dtype=dtype)
        )
        imagenet_std = (
            torch.from_numpy(np.array([0.229, 0.224, 0.225]))
            .to(device=device)
            .to(dtype=dtype)
        )
        imgs = (video * imagenet_std[None, :, None, None, None]) + imagenet_mean[
            None, :, None, None, None
        ]
        imgs = (imgs.permute(0, 2, 3, 4, 1).cpu().numpy() * 255).astype(np.uint8)

        with torch.no_grad():
            x_points = self.encoder.forward_decoder(latent, ids_restore)

            pred_, primitives = self.encoder.forward_render(
                x_points, limit_gaussian_z=-1, return_primitives=True
            )
            primitives["video"] = imgs
            primitives["renders"] = (pred_.detach().cpu().numpy() * 255).astype(
                np.uint8
            )

        # zeroshot tracking
        scale_factor = np.array([1, self.cfg.input_size, self.cfg.input_size])

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
            valid_queries = (
                (queries[valid_mask] * scale_factor)
                if np.any(valid_mask)
                else np.empty((0, 3))
            )
            valid_queries = valid_queries.astype(np.float32, copy=False)

            flows_np = batch_flows[i].numpy()
            weights_np = batch_weights[i].numpy()
            centers_np = batch_centers[i].numpy()
            num_frames = flows_np.shape[0] + 1 if flows_np.ndim >= 4 else 1

            if valid_queries.shape[0] == 0:
                x_points.append(
                    np.zeros((num_frames, queries.shape[0], 2), dtype=np.float32)
                )
                occlusions_pred.append(
                    np.zeros((num_frames, queries.shape[0]), dtype=np.float32)
                )
                continue

            tracks, occlusion = track_points_with_occlusions(
                valid_queries,
                flows_np,
                weights_np,
                centers_np,
                self.cfg.input_size,
                topk=self.cfg.inference.topk,
                tau_vis=self.cfg.inference.tau_vis,
                beta_hybrid=self.cfg.inference.beta_hybrid,
            )

            tracks_full = np.zeros(
                (tracks.shape[0], queries.shape[0], 2), dtype=np.float32
            )
            occlusion_full = np.zeros(
                (occlusion.shape[0], queries.shape[0]), dtype=np.float32
            )
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
                gt_vals = np.asarray(
                    occluded_np[vid_idx, valid_mask], dtype=np.float32
                ).reshape(-1)
                pred_vals = np.asarray(
                    occlusions_pred[vid_idx, valid_mask], dtype=np.float32
                ).reshape(-1)
                gt_pct = float(gt_vals.mean() * 100.0) if gt_vals.size else 0.0
                pred_pct = float(pred_vals.mean() * 100.0) if pred_vals.size else 0.0
            print(
                f"[Occlusion] batch {batch_idx} video {vid_idx}: GT {gt_pct:.2f}% occluded | Pred {pred_pct:.2f}% occluded"
            )

        # visualize tracking
        final_image_ = self.visualize_point_tracking(
            video,
            x_points,
            init_queries,
            finetune_target,
            occluded_points,
            occlusions_pred,
        )
        final_image_ = np.concatenate(final_image_, axis=1)
        final_image_ = [i for i in final_image_]

        train_ = "train" if self.training else "val"
        global_rank = self.trainer.global_rank
        if global_rank == 0 and not self.cfg.inference.testing:
            clip = ImageSequenceClip(final_image_, fps=1).resize(
                2
            )  # fps can be adjusted to your need
            clip.write_videofile(
                f"{self.cfg.storage_folder}/results/{train_}_{self.current_epoch}_{batch_idx}_video.mp4",
                codec="libx264",
            )

        if self.cfg.inference.testing:
            if self.cfg.inference.save_predictions:
                imagenet_mean = (
                    torch.from_numpy(np.array([0.485, 0.456, 0.406]))
                    .to(device=video.device)
                    .to(dtype=video.dtype)
                )
                imagenet_std = (
                    torch.from_numpy(np.array([0.229, 0.224, 0.225]))
                    .to(device=video.device)
                    .to(dtype=video.dtype)
                )
                imgs = (
                    video * imagenet_std[None, :, None, None, None]
                ) + imagenet_mean[None, :, None, None, None]
                imgs = (imgs.permute(0, 2, 3, 4, 1).cpu().numpy() * 255).astype(
                    np.uint8
                )
                data = {
                    "video": imgs,
                    "init_queries": init_queries,
                    "points_pred": x_points,
                    "occlusions_pred": occlusions_pred,
                    "points_gt": finetune_target,
                    "occlusions_gt": occluded_points,
                }
                torch.save(
                    data,
                    f"{self.cfg.storage_folder}/tests/zeroshot_{self.cfg.inference.context_length}_{self.current_epoch}_{batch_idx}.pt",
                )

            # title = f"eval_{self.cfg.inference.context_length}" if "eval" in self.cfg.training_type else "kubric"
            title = f"zeroshot_{self.cfg.inference.context_length}"
            clip = ImageSequenceClip(final_image_, fps=1).resize(
                2
            )  # fps can be adjusted to your need
            clip.write_videofile(
                f"{self.cfg.storage_folder}/tests/{title}_{self.current_epoch}_{batch_idx}_video.mp4",
                codec="libx264",
            )

        # default return values
        # check if loss_index.mean is nan
        loss_dict = {}
        logits_ = {}

        extras_ = {"x_points": x_points, "occluded_pred": occlusions_pred}

        return loss_dict, extras_, logits_

    def on_train_epoch_start(self) -> None:
        torch.cuda.empty_cache()

    def training_step(self, batch: Any, batch_idx: int):
        return {}

    def on_train_epoch_end(self):
        log.info(
            "\n "
            + self.cfg.storage_folder
            + " : Training epoch "
            + str(self.current_epoch)
            + " ended."
        )

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
        pred_visible = ~(
            extra.get("occluded_pred", np.zeros_like(query_points[:, :, 0])) > 0.5
        )
        true_points = target_points
        true_visible = ~(
            occluded > 0.5
        )  # Assuming batch[3] contains occlusion information

        for i in range(query_points.shape[0]):
            first_invalid_index = np.where(query_points[i, :, 0] == -1)[0]

            if len(first_invalid_index) == 0:
                first_invalid_index = true_points.shape[1]
            else:
                first_invalid_index = first_invalid_index[0]
            first_n_frames = 5  # self.cfg.seq_length
            query_points_ = query_points[i : i + 1, :first_invalid_index]
            query_points_[:, :, 1:] *= self.cfg.input_size

            pred_points_ = pred_points[i : i + 1, :first_invalid_index, :first_n_frames]
            pred_visible_ = pred_visible[
                i : i + 1, :first_invalid_index, :first_n_frames
            ]

            true_points_ = (
                true_points[i : i + 1, :first_invalid_index, :first_n_frames]
                * self.cfg.input_size
            )
            true_visible_ = true_visible[
                i : i + 1, :first_invalid_index, :first_n_frames
            ]

            if set((1 - (~true_visible_)).flatten().tolist()) == {0}:
                log.info(
                    f"Skipping batch {batch_idx} for point tracking, all points are occluded"
                )
                continue
            metrics = compute_tapvid_metrics(
                query_points_,
                (~true_visible_),
                true_points_,
                (~pred_visible_),
                pred_points_,
            )

            self.average_jaccard.update(metrics["average_jaccard"].mean())
            self.avg_distance.update(metrics["avg_distance"].mean())
            self.average_pts_within_thresh.update(
                metrics["average_pts_within_thresh"].mean()
            )
            self.occlusion_accuracy.update(metrics["occlusion_accuracy"].mean())

            self.occ_fp.update(metrics["occ_fp"].sum())
            self.occ_tp.update(metrics["occ_tp"].sum())
            self.occ_fn.update(metrics["occ_fn"].sum())

        occ_tp = self.occ_tp.compute()
        occ_fp = self.occ_fp.compute()
        occ_fn = self.occ_fn.compute()
        precision = occ_tp / (occ_tp + occ_fp)
        recall = occ_tp / (occ_tp + occ_fn)

        jaccard_score = self.average_jaccard.compute()
        self.log(
            f"{metric_prefix}/jaccard",
            jaccard_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        avg_distance = self.avg_distance.compute()
        self.log(
            f"{metric_prefix}/avg_distance",
            avg_distance,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        avg_pts_within_thresh = self.average_pts_within_thresh.compute()
        self.log(
            f"{metric_prefix}/avg_pts_within_thresh",
            avg_pts_within_thresh,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        occlusion_accuracy = self.occlusion_accuracy.compute()
        self.log(
            f"{metric_prefix}/occlusion_accuracy",
            occlusion_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        # for terminal logging
        self.log_iter_stats(batch_idx)

        # deleting tensors
        del loss_dict, batch

        return {}

    def on_validation_epoch_end(self):
        log.info(
            "\n "
            + self.cfg.storage_folder
            + " : Validation epoch "
            + str(self.current_epoch)
            + " ended."
        )
        metric_prefix = "test" if self.cfg.inference.testing else "val"

        occ_tp = self.occ_tp.compute()
        occ_fp = self.occ_fp.compute()
        occ_fn = self.occ_fn.compute()
        precision = occ_tp / (occ_tp + occ_fp)
        recall = occ_tp / (occ_tp + occ_fn)

        # reset the metric
        self.average_jaccard.reset()
        self.avg_distance.reset()
        self.average_pts_within_thresh.reset()
        self.occlusion_accuracy.reset()
        self.occ_fp.reset()
        self.occ_tp.reset()
        self.occ_fn.reset()

    def visualize_point_tracking(
        self,
        video,
        x_points,
        query_points,
        target_points,
        occluded_points,
        occlusions_pred,
    ):
        # renormalize to 0,1
        imagenet_mean = (
            torch.from_numpy(np.array([0.485, 0.456, 0.406]))
            .to(device=video.device)
            .to(dtype=video.dtype)
        )
        imagenet_std = (
            torch.from_numpy(np.array([0.229, 0.224, 0.225]))
            .to(device=video.device)
            .to(dtype=video.dtype)
        )
        imgs = (video * imagenet_std[None, :, None, None, None]) + imagenet_mean[
            None, :, None, None, None
        ]
        imgs = (imgs.permute(0, 2, 3, 4, 1).cpu().numpy() * 255).astype(np.uint8)

        scale_factor = np.array([self.cfg.input_size, self.cfg.input_size])[
            np.newaxis, np.newaxis, :
        ]

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

            final_image = np.concatenate(
                [painted_frames_true, painted_frames_pred, painted_frames_blank], axis=2
            )
            final_image_.append(final_image)
        return final_image_

    def log_iter_stats(self, cur_iter):
        if cur_iter % self.cfg.log_frequency != 0:
            return 0

        mem_usage = gpu_mem_usage()
        try:
            stats = {
                "epoch": "{}/{}".format(self.current_epoch, self.trainer.max_epochs),
                "iter": "{}/{}".format(
                    cur_iter + 1,
                    (
                        self.trainer.num_training_batches
                        if self.training
                        else self.trainer.num_val_batches
                    ),
                ),
                "time": "%.4f" % (self.timer.time_elapsed() - self.timer_last_iter),
                # "eta": "%.4f"%(self.timer.time_remaining()),
                "lr": (
                    self.trainer.optimizers[0].param_groups[0]["lr"]
                    if self.training
                    else 0
                ),
                "mem": int(np.ceil(mem_usage)),
            }
            self.timer_last_iter = self.timer.time_elapsed()
        except:
            # add self.timer from trainer.callbacks
            for cb_ in self.trainer.callbacks:
                if cb_.__class__.__name__ == "Timer":
                    self.timer = cb_
            self.timer_last_iter = self.timer.time_elapsed()
            stats = {}

        log.info(stats)

    def get_param_groups(self):
        def _get_layer_decay(name):
            layer_id = None
            depth = len(self.encoder.blocks)
            if (
                "class_token" in name
                or "pos_embed" in name
                or "patch_embed.proj" in name
            ):
                layer_id = 0
            elif "_head" in name:
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
            name = name[len("module.") :] if name.startswith("module.") else name
            if (
                len(p.shape) == 1 or name.endswith(".bias")
            ) and self.cfg.ZERO_WD_1D_PARAM:
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


if __name__ == "__main__":
    pass
