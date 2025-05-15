"""
dense_gaussian_flow_decord.py
--------------------------------
Batch optical-flow for dynamic Gaussian Splatting + visualisation
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


    dels, alpha, _ = gsplat.rasterization(
        means_t, quats, scales, opacities.squeeze(-1), rgbs,
        view[None], K[None], width=W, height=H,
        packed=False, render_mode="RGB")
    flow = dels[0, :, :, :2] * alpha[0, :, :]  # (H,W,2)

    return flow.detach().cpu().numpy()

def overlay_tracks(frames_rgb, flows, step=32, radius=3, fps=30, outfile="tracked_.mp4"):
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

        flow = flows[t].cpu().numpy() if torch.is_tensor(flows[t]) else flows[t]
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
    # quats_all    = torch.tensor([[-0.6182, 0.4361, -0.6299, 0.1756]], device=device).repeat(2, 1)
    quats_all    = torch.tensor([[1, 0, 0., 0.]], device=device).repeat(2, 1)
    opacities_all= torch.ones(1, 1, device=device).repeat(2, 1)

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
    stitched = [np.concatenate([frames_flow[i], out[i]], axis=1)  # align indices
                for i in range(N)]

    ImageSequenceClip(stitched, fps=30) \
        .write_videofile(outfile, codec="libx264", audio=False)
    print(f"✓ saved {outfile}")


@torch.no_grad()
def demo_combined(path, device="cuda", outfile="t_comparison.mp4"):
    T, H, W = 16, 224, 224
    f = 0.5 * W / math.tan(math.pi / 4)
    K = torch.tensor([[f, 0, W / 2],
                      [0, W / H * f, H / 2],
                      [0, 0, 1]], device=device).float()
    view = torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 8],
                         [0, 0, 0, 1]], device=device).float()

    data = torch.load(path, map_location=device)
    video = data["video"][0]
    frames_rgb = (video).astype(np.uint8)
    H, W = video.shape[1:3]

    x = data["x_points"][0]
    
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
    frames_flow = overlay_tracks(list(frames_rgb), flows)          # len = 15

    limit = means.shape[0]
    tile = 16
    tw, th = math.ceil(W / tile), math.ceil(H / tile)
    imgs = []
    means = allmeans

    m = means[:limit].float()
    s = scales[:limit].float()
    q0 = quats_all[:limit].float()
    rg0 = rgb[:limit].unsqueeze(0)
    op = opacities_all[:limit].float().view(1, -1)

    chunks = means.shape[0] // limit
    for j in range(chunks):
        if j:
            m += means[j * limit : (j + 1) * limit]/10.0
            rg0 = rg0 + rgb[j * limit : (j + 1) * limit].unsqueeze(0)
        radii, xy, depth, conic, _ = fully_fused_projection(m, None, q0, s, view[None], K[None], W, H)
        _, ids, flat = isect_tiles(xy, radii, depth, tile, tw, th)
        off = isect_offset_encode(ids, 1, tw, th)
        col, alpha = rasterize_to_pixels(xy, conic, torch.sigmoid(rg0), op, W, H, tile, off, flat, absgrad=True)
        imgs.append((col * alpha + (1 - alpha)).squeeze())

    out = (torch.stack(imgs).cpu().numpy() * 255).astype(np.uint8)
    gt = (video).astype(np.uint8)


    N = 16
    stitched = [np.concatenate([frames_flow[i], out[i]], axis=1)  # align indices
                for i in range(N)]

    ImageSequenceClip(stitched, fps=30) \
        .write_videofile(outfile, codec="libx264", audio=False)
    print(f"✓ saved {outfile}")




if __name__ == "__main__":
    demo_combined("logs/vgmae_base_kinetics_gaussians/0/tests/0_45_gaussians.pt", device="cuda")