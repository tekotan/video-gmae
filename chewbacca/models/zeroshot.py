"""
dense_gaussian_flow_decord.py
--------------------------------
Batch optical-flow for dynamic Gaussian Splatting + occlusion-aware tracking
(using decord for video IO and cumulative mean-deltas).
"""
import cv2, torch, gsplat, numpy as np
from gsplat import rasterization
from decord import VideoReader, cpu
from pathlib import Path
from tqdm import tqdm
import math
from moviepy.editor import ImageSequenceClip
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
)

# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def dense_flow_one_frame(means_t,    # (G,3)
                         mean_delta, # (G,3)   μ(t+1)−μ(t)
                         quats, scales, opacities,
                         view, K, W, H, idx=None, device="cuda"):
    """
    Returns:
      flow: (H, W, 2) torch.float32 on CPU
      d_pix: (G, 2) per-primitive displacements (torch.float32, CPU)
    """
    _, xy_t, _, _, _  = fully_fused_projection(means_t, None, quats, scales, view[None], K[None], W, H)
    _, xy_t1, _, _, _ = fully_fused_projection(means_t + mean_delta, None, quats, scales, view[None], K[None], W, H)
    d_pix = (xy_t1 - xy_t).squeeze().to(torch.float32)  # (G,2)

    rgbs = torch.zeros(means_t.shape[0], 3, device=device, dtype=torch.float32)
    rgbs[:, :2] = d_pix  # encode displacement in RGB.xy

    cols, alpha, _ = gsplat.rasterization(
        means_t, quats, scales, opacities.squeeze(-1), rgbs,
        view[None], K[None], width=W, height=H,
        packed=False, render_mode="RGB"
    )
    # cols: [1,H,W,3], alpha: [1,H,W,1]
    flow = cols[0, :, :, :2] * alpha[0, :, :, 0:1]  # (H,W,2)
    return flow.detach().cpu(), d_pix.detach().cpu()


@torch.no_grad()
def render_soft_assignments(means, quats, scales, opacities,
                            view, K, W, H, chunk=32, device="cuda"):
    """
    Per-pixel per-Gaussian contribution weights W[y,x,i].
    Implemented by rendering one-hot features in chunks of size `chunk`.
    Returns: W  (H, W, N)  float32 CPU
    """
    N = means.shape[0]
    W_full = torch.zeros(H, W, N, device=device, dtype=torch.float32)

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        D = end - start
        colors = torch.zeros(N, D, device=device, dtype=torch.float32)
        colors[start:end, torch.arange(D, device=device)] = 1.0

        cols, alphas, _ = gsplat.rasterization(
            means, quats, scales, opacities.squeeze(-1), colors,
            view[None], K[None], width=W, height=H,
            packed=False, render_mode="RGB"
        )
        # cols: [1, H, W, D] already alpha-weighted; write into full tensor:
        W_full[:, :, start:end] = cols[0]

    return W_full.detach().cpu()


@torch.no_grad()
def project_centers(means, quats, scales, view, K, W, H, device="cuda"):
    """
    Project Gaussian centers -> pixel centers and depths.
    Returns:
      xy: (N,2) torch.float32 CPU
      z:  (N,)  torch.float32 CPU
    """
    _, xy, _, depths, _ = fully_fused_projection(
        means, None, quats, scales, view[None], K[None], W, H
    )
    return xy.squeeze(0).detach().cpu(), depths.squeeze(0).detach().cpu()


def _bilinear_at(img, x, y):
    """
    img: (H,W,C) or (H,W) numpy, float32
    x,y: scalars in pixel coords
    returns: vector (C,) or scalar
    """
    H, W = img.shape[:2]
    x = float(np.clip(x, 0, W-1)); y = float(np.clip(y, 0, H-1))
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0+1, W-1), min(y0+1, H-1)
    wx, wy = x - x0, y - y0
    if img.ndim == 3:
        return ((1-wx)*(1-wy)*img[y0, x0] +
                wx*(1-wy)*img[y0, x1] +
                (1-wx)*wy*img[y1, x0] +
                wx*wy*img[y1, x1])
    else:
        return ((1-wx)*(1-wy)*img[y0, x0] +
                wx*(1-wy)*img[y0, x1] +
                (1-wx)*wy*img[y1, x0] +
                wx*wy*img[y1, x1])


# ---------------------------------------------------------------------------
#  Tracking (classic flow advection vs occlusion-aware)
# ---------------------------------------------------------------------------

def track_points_flow_only(points_xy, flows):
    """
    Classic flow advection (no occlusion awareness).
    points_xy: (N,2) float32, in (x,y)
    flows: (T-1, H, W, 2) numpy float32
    Returns: (T, N, 2) numpy float32 trajectories
    """
    Tm1, H, W, _ = flows.shape
    T = Tm1 + 1
    N = points_xy.shape[0]
    traj = np.zeros((T, N, 2), dtype=np.float32)
    traj[0] = points_xy
    for t in range(T-1):
        F = flows[t]
        for j in range(N):
            x, y = traj[t, j]
            u, v = _bilinear_at(F, x, y)
            traj[t+1, j] = np.array([np.clip(x+u, 0, W-1),
                                     np.clip(y+v, 0, H-1)], dtype=np.float32)
    return traj


def track_points_occlusion(points_xy, flows, W_seq, XY_seq,
                                      topk=8, tau_vis=0.05, tau_re=0.08,
                                      visible_mode="hybrid",   # "flow" | "mixture" | "hybrid"
                                      beta_hybrid=0.3,         # weight toward primitive mixture
                                      beta_scale_with_mass=True,
                                      eps=1e-8):
    """
    Occlusion-aware tracking with FIXED top-k owner set per point (chosen at t=0).

    points_xy: (N,2) initial (x,y) at frame 0
    flows:     (T-1,H,W,2) float32 numpy
    W_seq:     (T-1,H,W,G) float32 numpy  (weights at time t)
    XY_seq:    (T-1,G,2)   float32 numpy  (centers at time t+1)
    Returns:
      traj: (T,N,2)
      ids_fixed: (N,topk)   fixed owner ids per point
      pis: (T,N,topk)       mixture weights over the fixed owners (for logging)
      vis: (T,N)            visibility flags
    """
    Tm1, H, W, _ = flows.shape
    T = Tm1 + 1
    G = W_seq.shape[-1]
    N = points_xy.shape[0]
    topk = max(1, min(topk, G))

    traj = np.zeros((T, N, 2), dtype=np.float32)
    vis  = np.zeros((T, N), dtype=np.float32)
    pis  = np.zeros((T, N, topk), dtype=np.float32)
    offs = np.zeros((N, topk, 2), dtype=np.float32)  # per-point per-owner offsets
    ids_fixed = -np.ones((N, topk), dtype=np.int32)  # FIXED owner set per point

    traj[0] = points_xy

    def topk_indices_at(Wimg, x, y, k):
        """Return top-k indices and their (unnormalized) weights at (x,y)."""
        w_vec = _bilinear_at(Wimg, x, y)  # (G,)
        if w_vec.ndim == 0:
            w_vec = np.array([float(w_vec)], dtype=np.float32)
        idx = np.argpartition(-w_vec, kth=min(k-1, len(w_vec)-1))[:k]
        idx = idx[np.argsort(-w_vec[idx])]
        w_k = w_vec[idx]
        return idx.astype(np.int32), w_k.astype(np.float32)

    # 1) FIXED owners from W^{(0)}
    W0 = W_seq[0]  # (H,W,G)
    for j in range(N):
        x0, y0 = traj[0, j]
        idx_k, w_k = topk_indices_at(W0, x0, y0, topk)
        ids_fixed[j, :len(idx_k)] = idx_k
        mass0 = float(np.sum(w_k))
        vis[0, j] = 1.0 if mass0 >= tau_vis else 0.0
        # offsets need XY at t=0; we don't have it—will initialize at first step.

    # 2) Time loop
    for t in range(1, T):
        F_tm1 = flows[t-1]    # (H,W,2)
        W_tm1 = W_seq[t-1]    # (H,W,G) weights at frame t-1
        XY_t  = XY_seq[t-1]   # (G,2)   centers at frame t

        for j in range(N):
            x_prev, y_prev = traj[t-1, j]
            owners = ids_fixed[j]                     # (topk,)
            cxcy   = XY_t[owners]                     # (topk,2)

            # weights restricted to fixed owners at the CURRENT pixel
            w_vec_full = _bilinear_at(W_tm1, x_prev, y_prev)  # (G,)
            w_k = w_vec_full[owners] if w_vec_full.ndim > 0 else np.zeros((topk,), dtype=np.float32)
            mass = float(np.sum(w_k))
            if mass > 0:
                pi_k = (w_k / (mass + eps)).astype(np.float32)
            else:
                pi_k = np.full((topk,), 1.0/topk, dtype=np.float32)
            pis[t-1, j] = pi_k  # store mixture used this step

            # primitive mixture proposal
            if t == 1 and not np.any(offs[j]):  # no offsets yet
                p_prim_next = np.sum(cxcy * pi_k[:, None], axis=0)
            else:
                p_prim_next = np.sum((cxcy + offs[j]) * pi_k[:, None], axis=0)

            # visible / occluded decision using ONLY fixed owners' mass
            if mass >= tau_vis:
                if visible_mode == "flow":
                    u, v = _bilinear_at(F_tm1, x_prev, y_prev)
                    p_flow_next = np.array([x_prev + u, y_prev + v], dtype=np.float32)
                    p_next = p_flow_next
                elif visible_mode == "mixture":
                    p_next = p_prim_next.astype(np.float32)
                else:  # "hybrid"
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

            # clamp, write, and refresh offsets w.r.t fixed owners
            p_next[0] = float(np.clip(p_next[0], 0, W-1))
            p_next[1] = float(np.clip(p_next[1], 0, H-1))
            traj[t, j] = p_next
            offs[j] = p_next[None, :] - cxcy  # (topk,2)

    # final pis slice: reuse previous
    pis[-1] = pis[-2]
    return traj, ids_fixed, pis, vis


def overlay_tracks_from_trajs(frames_rgb, traj, vis=None, radius=3):
    """
    Draw precomputed trajectories onto frames.
    traj: (T,N,2) in (x,y)
    vis:  (T,N)   optional visibility flags to modulate brightness
    Returns: list of RGB frames
    """
    T, H, W, _ = len(frames_rgb), frames_rgb[0].shape[0], frames_rgb[0].shape[1], frames_rgb[0].shape[2]
    N = traj.shape[1]

    # distinct colors per point
    hsv = np.zeros((N, 1, 3), np.uint8)
    hsv[:, 0, 0] = np.linspace(0, 179, N).astype(np.uint8)
    hsv[:, 0, 1:] = 255
    base_colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0].astype(np.int32)

    out = []
    for t in range(T):
        frame = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2BGR).copy()
        for j in range(N):
            x, y = traj[t, j]
            c = base_colors[j].copy()
            if vis is not None:
                # dim color when occluded
                alpha = 1.0 if vis[t, j] > 0.5 else 0.35
                c = (c * alpha).astype(np.int32)
            cv2.circle(frame, (int(x), int(y)), radius, (int(c[0]), int(c[1]), int(c[2])), -1)
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return out


def grid_points(W, H, step=32):
    yy, xx = np.mgrid[step//2:H:step, step//2:W:step]
    pts = np.stack([xx, yy], -1).astype(np.float32).reshape(-1, 2)
    return pts


def overlay_tracks(frames_rgb, flows, step=32, radius=3, fps=30, outfile="tracked_.mp4"):
    """
    Original flow-only overlay (kept for backwards-compat).
    """
    H, W, _ = frames_rgb[0].shape
    yy, xx  = np.mgrid[step//2:H:step, step//2:W:step]
    pts     = np.stack([xx, yy], -1).astype(np.float32).reshape(-1, 2)

    hsv         = np.zeros((len(pts), 1, 3), np.uint8)
    hsv[:, 0, 0] = np.linspace(0, 179, len(pts)).astype(np.uint8)
    hsv[:, 0, 1:] = 255
    colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0].tolist()

    rendered = []
    for t in range(len(flows)):
        frame = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2BGR).copy()
        for (x, y), c in zip(pts.astype(int), colors):
            cv2.circle(frame, (x, y), radius, c, -1)
        rendered.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        flow = flows[t] if isinstance(flows[t], np.ndarray) else flows[t].cpu().numpy()
        u = cv2.remap(flow[..., 0], pts[:, 0], pts[:, 1], cv2.INTER_LINEAR)
        v = cv2.remap(flow[..., 1], pts[:, 0], pts[:, 1], cv2.INTER_LINEAR)
        pts += np.concatenate([u, v], -1)

    frame = cv2.cvtColor(frames_rgb[-1], cv2.COLOR_RGB2BGR).copy()
    for (x, y), c in zip(pts.astype(int), colors):
        cv2.circle(frame, (x, y), radius, c, -1)
    rendered.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    ImageSequenceClip(rendered, fps=fps).write_videofile(outfile, codec="libx264", audio=False)
    return rendered


@torch.no_grad()
def test_combined(path, device="cuda", outfile="t_comparison.mp4"):
    # (unchanged except minor tensor->numpy moves in dense_flow_one_frame)
    T, H, W = 16, 224, 224
    video   = torch.ones(T, H, W, 3, device=device)
    frames_rgb = (video.cpu().numpy() * 255).astype(np.uint8)

    f = 0.5 * W / math.tan(math.pi / 4)
    K = torch.tensor([[f, 0, W / 2],
                      [0, W / H * f, H / 2],
                      [0, 0, 1]], device=device).float()
    view = torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 8],
                         [0, 0, 0, 1]], device=device).float()

    means        = torch.tensor([[-0.5, 0., 0.], [0.5, 0., 0.]], device=device)
    mean_deltas  = torch.tensor([[0, 1, 0] for _ in range(1, 16)],
                                device=device).reshape(-1, 1, 3).repeat(1, 2, 1)
    mean_deltas[:, 1, :] *= -1
    scales       = torch.tensor([[1., 1., 1.]], device=device).repeat(2, 1) / 2
    quats_all    = torch.tensor([[1, 0, 0., 0.]], device=device).repeat(2, 1)
    opacities_all= torch.ones(1, 1, device=device).repeat(2, 1)

    flows = []
    means_ = means.clone()
    for t in tqdm(range(T - 1), desc="compute flow"):
        flow, _ = dense_flow_one_frame(
            means_, mean_deltas[t], quats_all, scales, opacities_all,
            view, K, W, H, device=device)
        flows.append(flow.numpy())
        means_ = means_ + mean_deltas[t]

    frames_flow = overlay_tracks(list(frames_rgb), flows)          # len = 15

    limit = 2
    tile = 16
    tw, th = math.ceil(W / tile), math.ceil(H / tile)
    imgs = []
    rgb = torch.tensor([[1, 0, 0], [0, 1, 0]] * 16).to(device)
    means = torch.cat([means, mean_deltas.reshape(-1, 3)], dim=0)  # (G,3)

    m = means[:limit].float()
    s = scales[:limit].float()
    q0 = quats_all[:limit].float()
    rg0 = rgb[:limit].unsqueeze(0)
    op = opacities_all[:limit].float().view(1, -1)

    chunks = means.shape[0] // limit
    for j in range(chunks):
        if j:
            m += means[j * limit : (j + 1) * limit]
            rg0 = rg0 + rgb[j * limit : (j + 1) * limit].unsqueeze(0)
        radii, xy, depth, conic, _ = fully_fused_projection(m, None, q0, s, view[None], K[None], W, H)
        _, ids, flat = isect_tiles(xy, radii, depth, tile, tw, th)
        off = isect_offset_encode(ids, 1, tw, th)
        col, alpha = rasterize_to_pixels(xy, conic, torch.sigmoid(rg0), op, W, H, tile, off, flat, absgrad=True)
        imgs.append((col * alpha + (1 - alpha)).squeeze())

    out = (torch.stack(imgs).cpu().numpy() * 255).astype(np.uint8)
    gt = (video.detach().cpu().numpy()*255).astype(np.uint8)

    N = 16
    stitched = [np.concatenate([frames_flow[i], out[i]], axis=1)
                for i in range(N)]

    ImageSequenceClip(stitched, fps=30).write_videofile(outfile, codec="libx264", audio=False)
    print(f"✓ saved {outfile}")


def _video_to_uint8_numpy(video):
    if isinstance(video, torch.Tensor):
        video_np = video.detach().cpu().numpy()
    else:
        video_np = np.asarray(video)
    return video_np.astype(np.uint8)


def _camera_matrices(W, H, device):
    f = 0.5 * W / math.tan(math.pi / 4)
    K = torch.tensor([[f, 0, W / 2],
                      [0, W / H * f, H / 2],
                      [0, 0, 1]], device=device, dtype=torch.float32)
    view = torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 8],
                         [0, 0, 0, 1]], device=device, dtype=torch.float32)
    return view, K


def _track_single_video(video, x_points, device, step, tau_vis, tau_re, chunk):
    frames_rgb = _video_to_uint8_numpy(video)
    T = frames_rgb.shape[0]
    if T == 0:
        raise ValueError("Video has zero frames; cannot perform tracking.")
    H, W = frames_rgb.shape[1:3]

    view, K = _camera_matrices(W, H, device)

    if not torch.is_tensor(x_points):
        x = torch.tensor(x_points, device=device)
    else:
        x = x_points.to(device)

    if x.ndim != 2:
        raise ValueError(f"x_points must be a 2D tensor, got shape {tuple(x.shape)}")

    total_gaussians = x.shape[0]
    if total_gaussians % T != 0:
        raise ValueError(
            f"Expected total gaussians ({total_gaussians}) to be divisible by number of frames ({T})."
        )
    n = total_gaussians // T

    allmeans    = 5 * torch.tanh(x[:, :3])
    mean_deltas = allmeans[n:].reshape(T - 1, n, 3) / 10 if T > 1 else allmeans.new_zeros((0, n, 3))
    means       = allmeans[:n]
    scales      = torch.sigmoid(x[:, 3:6])[:n]
    q_raw       = torch.sigmoid(x[:, 6:10])[:n]

    a, b, c = q_raw[..., 0:1], q_raw[..., 1:2], q_raw[..., 2:3]
    quats_all = torch.cat(
        [
            torch.sqrt(1 - a) * torch.sin(2 * torch.pi * b),
            torch.sqrt(1 - a) * torch.cos(2 * torch.pi * b),
            torch.sqrt(a)     * torch.sin(2 * torch.pi * c),
            torch.sqrt(a)     * torch.cos(2 * torch.pi * c),
        ],
        -1,
    )

    opacities_all = torch.sigmoid(x[:, 13:14])[:n]

    flows_np, W_seq_list, XY_seq_list = [], [], []
    means_ = means.clone()
    for t in tqdm(range(T - 1), desc="compute flow + assignments + centers", leave=False):
        flow_t, _ = dense_flow_one_frame(
            means_, mean_deltas[t], quats_all, scales, opacities_all,
            view, K, W, H, device=device
        )
        flows_np.append(flow_t.numpy())

        W_t = render_soft_assignments(
            means_, quats_all, scales, opacities_all,
            view, K, W, H, chunk=chunk, device=device
        )
        W_seq_list.append(W_t.numpy())

        means_next = means_ + mean_deltas[t]
        xy_next, _ = project_centers(means_next, quats_all, scales, view, K, W, H, device=device)
        XY_seq_list.append(xy_next.numpy())

        means_ = means_next

    if flows_np:
        flows_np = np.stack(flows_np, axis=0)
        W_seq    = np.stack(W_seq_list, axis=0)
        XY_seq   = np.stack(XY_seq_list, axis=0)
    else:
        flows_np = np.zeros((0, H, W, 2), dtype=np.float32)
        W_seq    = np.zeros((0, H, W, n), dtype=np.float32)
        XY_seq   = np.zeros((0, n, 2), dtype=np.float32)

    pts0 = grid_points(W, H, step=step)
    traj_flow = track_points_flow_only(pts0.copy(), flows_np)
    frames_flow_only = overlay_tracks_from_trajs(list(frames_rgb), traj_flow, vis=None, radius=3)

    traj_occ, ids, pis, vis = track_points_occlusion(
        pts0.copy(), flows_np, W_seq, XY_seq,
        tau_vis=tau_vis, tau_re=tau_re
    )
    occluded_once = np.count_nonzero(np.any(vis < 0.5, axis=0))
    print(f"{occluded_once} / {vis.shape[1]} points were occluded at least once across {vis.shape[0]} frames.")
    frames_occ = overlay_tracks_from_trajs(list(frames_rgb), traj_occ, vis=vis, radius=3)

    stitched = [np.concatenate([frames_flow_only[i], frames_occ[i]], axis=1) for i in range(len(frames_occ))]
    return stitched


@torch.no_grad()
def demo_combined(path, device="cuda", outfile="t_comparison.mp4",
                  step=32, tau_vis=0.3, tau_re=0.08, chunk=32, fps=30):
    data = torch.load(path, map_location=device)
    videos = data["video"]
    x_points = data["x_points"]

    if isinstance(videos, torch.Tensor):
        batch_size = videos.shape[0]
    elif isinstance(videos, np.ndarray):
        batch_size = videos.shape[0]
    else:
        batch_size = len(videos)

    if isinstance(x_points, torch.Tensor):
        x_batch = x_points
        x_size = x_batch.shape[0]
    else:
        x_batch = x_points
        x_size = len(x_batch)

    if batch_size == 0:
        raise ValueError("No videos found in the provided file.")
    if x_size != batch_size:
        raise ValueError(
            f"Mismatched batch sizes between video ({batch_size}) and x_points ({x_size})."
        )

    stitched_per_video = []
    iterator = tqdm(range(batch_size), desc="track batch", leave=False)
    for idx in iterator:
        video_i = videos[idx]
        x_i = x_batch[idx]
        stitched = _track_single_video(
            video_i, x_i, device=device,
            step=step, tau_vis=tau_vis, tau_re=tau_re, chunk=chunk
        )
        if stitched_per_video and len(stitched) != len(stitched_per_video[0]):
            raise ValueError("All videos must share the same number of frames to stack vertically.")
        stitched_per_video.append(stitched)

    num_frames = len(stitched_per_video[0])
    stacked_frames = [
        np.concatenate([stitched_per_video[b][t] for b in range(batch_size)], axis=0)
        for t in range(num_frames)
    ]

    ImageSequenceClip(stacked_frames, fps=fps).write_videofile(outfile, codec="libx264", audio=False)
    print(f"✓ saved {outfile}  (top-to-bottom stacks {batch_size} videos; left/right show flow vs occlusion)")


if __name__ == "__main__":
    demo_combined("logs/vgmae_base_zeroshot_midtrain/0/tests/0_1_gaussians.pt", device="cuda")
