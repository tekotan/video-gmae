import math
import os
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tyro
from PIL import Image
from torch import Tensor, optim

from gsplat import rasterization, rasterization_2dgs
from models.components.mae.models_mae_video import mae_vit_base_patch16 

class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        
        self.model = mae_vit_base_patch16(number_of_frames=2, num_gaussian=num_points, img_size=gt_image.shape[1]).cuda()


    def train(
        self,
        iterations: int = 5000,
        lr: float = 0.01,
        mask_ratio: float = 0.0,
        save_imgs: bool = False,
        model_type: Literal["3dgs", "2dgs"] = "3dgs",
    ):
        optimizer = optim.Adam(
            self.model.parameters(), lr
        )
        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 2  # rasterization, backward

        for iter in range(iterations):
            gt = self.gt_image.permute(2, 0, 1).unsqueeze(1).unsqueeze(0).repeat(2, 1, 2, 1, 1)
            start = time.time()
            loss, pred, mask, latent, latents_layer = self.model.forward(gt, mask_ratio=mask_ratio)
            torch.cuda.synchronize()
            times[0] += time.time() - start
            loss = mse_loss(pred[0,0], self.gt_image)
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 5 == 0:
                loss, pred, mask, latent, latents_layer = self.model.forward(gt, mask_ratio=0.0)
                frames.append((pred[0, 0].detach().cpu().numpy() * 255).astype(np.uint8))
        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
        print(f"Total(s):\nRasterization: {times[0]:.3f}, Backward: {times[1]:.3f}")
        print(
            f"Per step(s):\nRasterization: {times[0]/iterations:.5f}, Backward: {times[1]/iterations:.5f}"
        )


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    height: int = 112,
    width: int = 112,
    num_points: int = 1024,
    mask_ratio: float = 0.0,
    save_imgs: bool = True,
    img_path: Optional[Path] = None,
    iterations: int = 300,
    lr: float = 1e-6,
    model_type: Literal["3dgs", "2dgs"] = "3dgs",
) -> None:
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
    trainer.train(
        iterations=iterations,
        lr=lr,
        mask_ratio=mask_ratio,
        save_imgs=save_imgs,
        model_type=model_type,
    )
    # print(trainer.means.min(), trainer.means.max())
    # print(trainer.scales.min(), trainer.scales.max())
    # print(trainer.quats.min(), trainer.quats.max())
    # print(trainer.opacities.min(), trainer.opacities.max())
    # print(trainer.rgbs.min(), trainer.rgbs.max())


if __name__ == "__main__":
    tyro.cli(main)