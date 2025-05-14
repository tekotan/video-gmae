import torch
import cv2
from gsplat import rasterization
import math
import numpy as np
from moviepy.editor import ImageSequenceClip

from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
)

def demo(path, device="cuda"):
    data = torch.load(path, map_location=device)
    video = data["video"][0]
    H, W = video.shape[1:3]

    x = data["x_points"]
    means = 5 * torch.tanh(x[:, :, :3])
    scales = torch.sigmoid(x[:, :, 3:6])
    q = torch.sigmoid(x[:, :, 6:10])

    a, b, c = q[..., 0:1], q[..., 1:2], q[..., 2:3]
    q = torch.cat(
        [
            torch.sqrt(1 - a) * torch.sin(2 * torch.pi * b),
            torch.sqrt(1 - a) * torch.cos(2 * torch.pi * b),
            torch.sqrt(a) * torch.sin(2 * torch.pi * c),
            torch.sqrt(a) * torch.cos(2 * torch.pi * c),
        ],
        -1,
    )

    rgb = x[:, :, 10:13]
    opa = torch.sigmoid(x[:, :, 13:14])

    limit = 256
    view = (
        torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 8], [0, 0, 0, 1]], device=device)
        .float()
        .unsqueeze(0)
    )
    f = 0.5 * W / math.tan(math.pi / 4)
    K = (
        torch.tensor([[f, 0, W / 2], [0, W / H * f, H / 2], [0, 0, 1]], device=device)
        .float()
        .unsqueeze(0)
    )

    tile = 16
    tw, th = math.ceil(W / tile), math.ceil(H / tile)

    imgs = []
    m = means[0, :limit].float()
    s = scales[0, :limit].float()
    q0 = q[0, :limit].float()
    rg0 = rgb[0, :limit].unsqueeze(0)
    op = opa[0, :limit].float().view(1, -1)

    chunks = means.shape[1] // limit
    for j in range(chunks):
        if j:
            m += means[0, j * limit : (j + 1) * limit] / 10
            rg0 = rg0 + rgb[0, j * limit : (j + 1) * limit].unsqueeze(0)
        radii, xy, depth, conic, _ = fully_fused_projection(m, None, q0, s, view, K, W, H)
        _, ids, flat = isect_tiles(xy, radii, depth, tile, tw, th)
        off = isect_offset_encode(ids, 1, tw, th)
        col, alpha = rasterize_to_pixels(xy, conic, torch.sigmoid(rg0), op, W, H, tile, off, flat, absgrad=True)
        imgs.append((col * alpha + (1 - alpha)).squeeze())

    out = (torch.stack(imgs).cpu().numpy() * 255).astype(np.uint8)
    gt = (video).astype(np.uint8)
    ImageSequenceClip([i for i in out], fps=30).write_videofile("t_rendered.mp4", codec="libx264", audio=False)
    cv2.imwrite("image.png", out[0, :, :, ::-1])
    cv2.imwrite("image1.png", gt[0, :, :, ::-1])

def test(path, device="cuda"):
    
    video = torch.ones(16, 224, 224, 3).to(device)
    H, W = video.shape[1:3]

    means = torch.tensor([[0, 3, 0] for i in range(0, 16)]).to(device)
    scales = torch.tensor([[1, 1, 1]]).to(device)

    q = torch.tensor([[-0.6182,  0.4361, -0.6299,  0.1756]]).to(device)

    rgb = torch.tensor([[1, 0, 0]] * 16).to(device)
    opa = torch.tensor([[1]]).to(device)

    limit = 1
    view = (
        torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 8], [0, 0, 0, 1]], device=device)
        .float()
        .unsqueeze(0)
    )
    f = 0.5 * W / math.tan(math.pi / 4)
    K = (
        torch.tensor([[f, 0, W / 2], [0, W / H * f, H / 2], [0, 0, 1]], device=device)
        .float()
        .unsqueeze(0)
    )

    tile = 16
    tw, th = math.ceil(W / tile), math.ceil(H / tile)
    imgs = []
    m = means[:limit].float()
    s = scales[:limit].float()
    q0 = q[:limit].float()
    rg0 = rgb[:limit].unsqueeze(0)
    op = opa[:limit].float().view(1, -1)
    

    chunks = means.shape[1] // limit
    for j in range(chunks):
        if j:
            m += means[j * limit : (j + 1) * limit]
            rg0 = rg0 + rgb[j * limit : (j + 1) * limit].unsqueeze(0)
        radii, xy, depth, conic, _ = fully_fused_projection(m, None, q0, s, view, K, W, H)
        _, ids, flat = isect_tiles(xy, radii, depth, tile, tw, th)
        off = isect_offset_encode(ids, 1, tw, th)
        col, alpha = rasterize_to_pixels(xy, conic, torch.sigmoid(rg0), op, W, H, tile, off, flat, absgrad=True)
        imgs.append((col * alpha + (1 - alpha)).squeeze())

    out = (torch.stack(imgs).cpu().numpy() * 255).astype(np.uint8)
    gt = (video.detach().cpu().numpy()*255).astype(np.uint8)
    ImageSequenceClip([i for i in out], fps=30).write_videofile("t_rendered.mp4", codec="libx264", audio=False)
    cv2.imwrite("image.png", out[0, :, :, ::-1])
    cv2.imwrite("image1.png", gt[0, :, :, ::-1])

if __name__ == "__main__":
    demo("./logs/vgmae_base_kinetics_gaussians/0/tests/0_0_gaussians.pt")