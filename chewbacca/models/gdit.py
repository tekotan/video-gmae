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
import transformers
import xformers.ops as xops
from einops import rearrange
from lart.models.components.tokeizers.tokenizer import Tokenizer
from lightning import LightningModule
from moviepy.editor import ImageSequenceClip
from omegaconf import DictConfig
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.vision_transformer import PatchEmbed
from torch import nn
from torchmetrics import MeanMetric, MeanSquaredError
from torchmetrics.aggregation import CatMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

from chewbacca.models.components.dit import DitGaussian
from chewbacca.models.components.mae.models_mae import get_2d_sincos_pos_embed
from chewbacca.utils import get_pylogger
from chewbacca.utils.lamb import LARS, Lamb
from gsplat.project_gaussians import ProjectGaussians
from gsplat.rasterize import RasterizeGaussians

log = get_pylogger(__name__)


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



class attntion_pool(nn.Module):
    def __init__(self, num_heads, in_dim, out_dim):
        super().__init__()
        self.num_heads = num_heads
        self.q_ = nn.Parameter(torch.randn(200, out_dim//num_heads))
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
        attn_output = rearrange(attn_output, 'n h l d -> n l (h d)')
        # attn_output = attn_output.transpose(1, 2).view(x.shape[0], -1)
        return attn_output



class GDITLitModule(LightningModule):
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

        self.num_frames = self.cfg.seq_length if self.cfg.dataset_type=="video" else 1

        if self.cfg.dataset_type=="imagenet":
            embed_dim = 256
            self.patch_emb = PatchEmbed(self.cfg.input_size, self.cfg.patch_size, 3, embed_dim)
            self.patch_emb2 = PatchEmbed(self.cfg.input_size, self.cfg.patch_size, 3, embed_dim)
            self.dit = DitGaussian(self.cfg.input_size, patch_size=4, hidden_size=embed_dim, depth=16, num_heads=16, mlp_ratio=4.0)
            
            self.g_proj_in = nn.Linear(14, embed_dim)
            self.g_proj_out = nn.Linear(embed_dim, 14)
            self.attn_pool = attntion_pool(2, embed_dim, 14)

            # pos embedding for pixels
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_emb.num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_emb.num_patches**.5), cls_token=False)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            # --------------------------------------------------------------------------
            # Gaussian specifics
            self.num_points = 784 #1000
            self.BLOCK_X, self.BLOCK_Y = 16, 16
            self.fov_x = torch.pi / 2.0
            self.H, self.W = self.cfg.input_size, self.cfg.input_size
            self.focal = 0.5 * float(self.W) / math.tan(0.5 * self.fov_x)
            self.tile_bounds = (
                        (self.W + self.BLOCK_X - 1) // self.BLOCK_X,
                        (self.H + self.BLOCK_Y - 1) // self.BLOCK_Y,
                        1,
                    )
            self.block = torch.tensor([self.BLOCK_X, self.BLOCK_Y, 1])
            self.viewmat = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 8.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],)
            # --------------------------------------------------------------------------

        # setup meters
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.test_mse = MeanSquaredError()
        self.test_ssim = StructuralSimilarityIndexMeasure()
        self.test_psnr = PeakSignalNoiseRatio()
        self.test_rfid = FrechetInceptionDistance()

        # create folders for storing results
        os.makedirs(self.cfg.storage_folder + "/results/", exist_ok=True)
        os.makedirs(self.cfg.storage_folder + "/videos/", exist_ok=True)
        log.info("Storage folder : " + self.cfg.storage_folder)

    def forward_loss(self, imgs, pred, mask, additional_data=None):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        device = imgs.device
        dtype = imgs.dtype
        imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device=device).to(dtype=dtype)
        imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device=device).to(dtype=dtype)
        imgs_ = (imgs * imagenet_std[None, :, None, None, None]) + imagenet_mean[None, :, None, None, None]

        loss = 0
        count = 0

        for i in range(imgs_.shape[0]):
            if(torch.sum(pred[i])>0):
                if "loss-masked" in self.cfg.training_type:
                    # loss on only the masked patches
                    target = self.encoder.patchify(imgs_[i][None, :, 0])
                    if "norm-patch" in self.cfg.training_type:
                        mean = target.mean(dim=-1, keepdim=True)
                        var = target.var(dim=-1, keepdim=True)
                        target = (target - mean) / (var + 1.e-6)**.5
                    pred_target = self.encoder.patchify(pred[None, i, 0].permute(0, 3, 1, 2))
                    loss_ = (pred_target - target) ** 2
                    loss_ = loss_.mean(dim=-1)  # [N, L], mean loss per patch
                    if "all-patch" in self.cfg.training_type:
                        loss += loss_.mean()
                    else:
                        loss += (loss_ * mask[i]).sum() / mask.sum()  # mean loss on removed patches
                else:
                    loss += torch.nn.functional.mse_loss(imgs_[i][:, :self.num_frames], pred[i].permute(3, 0, 1, 2))
                count += 1

        return loss

    def step(self, batch: Any, batch_idx: int, return_images=False):
        x = batch[0]
        px = self.patch_emb(x) + self.pos_embed

        bs = x.size(0)
        nt = torch.randn((bs,)).to(x.device)
        t = torch.sigmoid(nt)
        texp = t.view([bs, *([1] * len(x.shape[1:]))])


        z1 = torch.randn((px.shape[0], 200, 14), device=px.device)
        img_t = self.render(z1) * texp + (1 - texp) * x
        zt = self.attn_pool(self.patch_emb2(img_t))
        z0 = self.attn_pool(self.patch_emb2(x))
        

        px = self.patch_emb(x) + self.pos_embed
        zt_ = self.g_proj_in(zt)
        x_ = torch.cat([px, zt_], dim=1)
        vt = self.dit(x_, t, cond_length=px.shape[1])
        vt = vt[:, px.shape[1]:]
        vtheta = self.g_proj_out(vt)


        mse_loss = (((z1 - z0) - vtheta) ** 2).mean()    

        z0_ = z0 + vtheta
        loss_1 = torch.nn.functional.mse_loss(self.attn_pool(self.patch_emb2((self.render(z1)))), z1)
        loss_2 = torch.nn.functional.mse_loss(self.attn_pool(self.patch_emb2((self.render(z0_)))), z0_)
        
        loss = mse_loss + loss_1 + loss_2

        return {"loss": loss}

    def render(self, points):
        
        dtype = points.dtype
        device = points.device
        limit_gaussian = -1
        return_depth = False

        imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device=device).to(dtype=dtype)
        imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device=device).to(dtype=dtype)

        means = 2*(torch.sigmoid(points[:, :, :3])-0.5)
        scales = torch.sigmoid(points[:, :, 3:6])
        quats = torch.sigmoid(points[:, :, 6:10])
        au = quats[:, :, :1]
        av = quats[:, :, 1:2]
        aw = quats[:, :, 2:3]
        quats = torch.stack([
            torch.sqrt(1.0 - au) * torch.sin(2.0 * torch.pi * av),
            torch.sqrt(1.0 - au) * torch.cos(2.0 * torch.pi * av),
            torch.sqrt(au) * torch.sin(2.0 * torch.pi * aw),
            torch.sqrt(au) * torch.cos(2.0 * torch.pi * aw),
        ], dim=-1)
        quats = quats[:, :, 0, :]

        rgbs = points[:, :, 10:13]
        opacities = points[:, :, 13:14]

        self.viewmat = self.viewmat.to(dtype=dtype).to(device=device)
        imgs_ = []
        xys_ = []
        for i in range(means.shape[0]):
            images_per_video = []
            # render first image
            means_ = means[i].contiguous().view(-1, 3)[:limit_gaussian]
            scales_ = scales[i].contiguous().view(-1, 3)[:limit_gaussian]
            quats_ = quats[i].contiguous().view(-1, 4)[:limit_gaussian]
            means_ = means_.float()
            scales_ = scales_.float()
            quats_ = quats_.float()
            self.viewmat = self.viewmat.float()
            xys, depths, radii, conics, num_tiles_hit, cov3d = ProjectGaussians.apply(means_, scales_, 1, quats_, self.viewmat, self.viewmat, self.focal, self.focal, self.W / 2, self.H / 2, self.H, self.W, self.tile_bounds,)

            rgbs_ = rgbs[i].view(-1, 3).float()[:limit_gaussian]
            opacities_ = opacities[i].view(-1, 1).float()[:limit_gaussian]
            try:
                xys_.append([xys, opacities_])
                out_img = RasterizeGaussians.apply(xys, depths, radii, conics, num_tiles_hit, torch.sigmoid(rgbs_), torch.sigmoid(opacities_), self.H, self.W,)
            except Exception as e:
                print(e)
                out_img = torch.zeros((self.H, self.W, 3), dtype=dtype, device=device)
            imgs_.append(out_img)
        imgs_ = torch.stack(imgs_, dim=0)
        imgs_ = rearrange(imgs_, 'b h w c -> b c h w')

        # normalize the image
        imgs_ = (imgs_ - imagenet_mean[None, :, None, None]) / imagenet_std[None, :, None, None]

        return imgs_

    @torch.no_grad()
    def sample(self, batch, batch_idx):
        self.eval()
        # sampling
        x = batch[0]
        px = self.patch_emb(x) + self.pos_embed
        steps = 10

        z1 = torch.randn((px.shape[0], 200, 14), device=px.device)
        bs = x.size(0)
        for i in range(steps):
            dt = 1/(steps)
            t = i / steps
            t = torch.tensor([t] * bs).to(x.device)
            texp = t.view([bs, *([1] * len(z1.shape[1:]))])

            x_ = torch.cat([px, self.g_proj_in(z1)], dim=1)
            vt = self.dit(x_, t, cond_length=px.shape[1])
            vt = vt[:, px.shape[1]:]
            vtheta = self.g_proj_out(vt)

            z1 = z1 + vtheta * dt

        img_recon = self.render(z1)
        img_real  = x

        device = x.device
        dtype = x.dtype
        imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device=device).to(dtype=dtype)
        imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device=device).to(dtype=dtype)
        final_image = torch.cat([img_real, img_recon], dim=2)
        final_image = final_image * imagenet_std[None, :, None, None] + imagenet_mean[None, :, None, None] 
        final_image = (final_image.detach().cpu().numpy() * 255).astype(np.uint8)

        final_image = final_image[:, ::-1]
        final_image = rearrange(final_image, 'b c h w -> h (b w) c')
        # save final image with epoch and batch index
        train_ = "train" if self.training else "val"
        global_rank = self.trainer.global_rank
        if global_rank==0:
            cv2.imwrite(self.cfg.storage_folder + f"/results/{train_}_{self.current_epoch}_{batch_idx}.png", final_image)

        self.train()
        return final_image

    def on_train_epoch_start(self) -> None:
        torch.cuda.empty_cache()

    def training_step(self, batch: Any, batch_idx: int):

        if "overfit" in self.cfg.training_type:
            self.batch = batch if batch_idx==0 else self.batch
            batch = self.batch

        if batch_idx%10==0:
            _ = self.sample(batch, batch_idx)

        loss_dict = self.step(batch, batch_idx)
        loss = sum([v for k,v in loss_dict.items()])

        # update and log metrics
        self.train_loss(loss.item())
        for key in loss_dict.keys():
            self.log("train/loss/" + key, loss_dict[key].item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
        # for terminal logging
        self.log_iter_stats(batch_idx)

        # deleting tensors
        del loss_dict, batch

        return {"loss": loss}

    def on_train_epoch_end(self):
        log.info("\n " + self.cfg.storage_folder +  " : Training epoch " + str(self.current_epoch) + " ended.")

    def on_validation_epoch_start(self) -> None:
        torch.cuda.empty_cache()

    def validation_step(self, batch: Any, batch_idx: int):
        
        if "overfit" in self.cfg.training_type:
            batch = self.batch

        loss_dict = self.step(batch, batch_idx)


        # bs = x.size(0)
        # nt = torch.randn((bs,)).to(x.device)
        # t = torch.sigmoid(nt)
        # texp = t.view([bs, *([1] * len(x.shape[1:]))])


        # z1 = torch.randn((px.shape[0], 200, 14), device=px.device)
        # img_t = self.render(z1) * texp + (1 - texp) * x
        # zt = self.attn_pool(self.patch_emb2(img_t))
        # z0 = self.attn_pool(self.patch_emb2(x))
        

        # zt_ = self.g_proj_in(zt)
        # x_ = torch.cat([px, zt_], dim=1)
        # vt = self.dit(x_, t, cond_length=px.shape[1])
        # vt = vt[:, px.shape[1]:]
        # vtheta = self.g_proj_out(vt)


        # mse_loss = (((z1 - z0) - vtheta) ** 2).mean()    

        # z0_ = z0 + vtheta
        # loss_1 = torch.nn.functional.mse_loss(self.attn_pool(self.patch_emb2((self.render(z1)))), z1)
        # loss_2 = torch.nn.functional.mse_loss(self.attn_pool(self.patch_emb2((self.render(z0_)))), z0_)



        # # update metrics
        # # # computed Metrics:
        # device = pred.device
        # dtype = pred.dtype
        # imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device=device).to(dtype=dtype)
        # imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device=device).to(dtype=dtype)
        # x_in = batch[0]
        # x_in_ = (x_in * imagenet_std[None, :, None, None]) + imagenet_mean[None, :, None, None]
        # x_rgb_ = pred[:, 0].permute(0, 3, 1, 2).contiguous()

        # # MSE
        # self.test_mse.update(x_rgb_, x_in_)
        # # SSIM
        # self.test_ssim.update(x_rgb_, x_in_)
        # # PSNR
        # self.test_psnr.update(x_rgb_, x_in_)
        # # rFID take uint 8
        # x_rgb_2 = (torch.clamp(x_rgb_, 0, 1) * 255).type(torch.uint8)
        # x_in_2 = (torch.clamp(x_in_, 0, 1) * 255).type(torch.uint8)
        # # rFID
        # self.test_rfid.update(x_rgb_2, real=False)
        # self.test_rfid.update(x_in_2, real=True)

        # loss = sum([v for k,v in loss_dict.items()])

        # # update and log metrics
        # self.val_loss(loss.item())
        # for key in loss_dict.keys():
        #     self.log("val/loss/" + key, loss_dict[key].item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # # for terminal logging
        # self.log_iter_stats(batch_idx)

        # # deleting tensors
        # del loss_dict, batch


        return 0 #{"loss": loss}

    def on_validation_epoch_end(self):
        log.info("\n " + self.cfg.storage_folder +  " : Validation epoch " + str(self.current_epoch) + " ended.")

        # if "mae" not in self.cfg.training_type and "vit" not in self.cfg.training_type and self.cfg.dataset_type!="video":

        #     total_mse = self.test_mse.compute()
        #     self.log(f"mse", total_mse, sync_dist=True)

        #     total_ssim = self.test_ssim.compute()
        #     self.log(f"ssim", total_ssim, sync_dist=True)

        #     total_psnr = self.test_psnr.compute()
        #     self.log(f"psnr", total_psnr, sync_dist=True)

        #     total_fid = self.test_rfid.compute()
        #     self.log(f"rfid", total_fid, sync_dist=True)

        #     # reset the metric
        #     self.val_loss.reset()
        #     self.train_loss.reset()
        #     self.test_mse.reset()
        #     self.test_ssim.reset()
        #     self.test_psnr.reset()
        #     self.test_rfid.reset()

        # # reset the metric
        # self.val_loss.reset()
        # self.train_loss.reset()


    def log_iter_stats(self, cur_iter):

        if(cur_iter%self.cfg.log_frequency != 0):
            return 0

        mem_usage = gpu_mem_usage()
        try:
            stats = {
                "epoch": "{}/{}".format(self.current_epoch, self.trainer.max_epochs),
                "iter": "{}/{}".format(cur_iter + 1, self.trainer.num_training_batches if self.training else self.trainer.num_val_batches),
                "train_loss": "%.4f"%(self.train_loss.compute().item()),
                # "val_loss": "%.4f"%(self.val_loss.compute().item()),
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

        self.train_loss.reset()
        self.val_loss.reset()

        log.info(stats)


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

        optim_params = [{'params': filter(lambda p: p.requires_grad, self.trainer.model.parameters()), 'lr': self.cfg.lr * self.lr_scaler}]

        if(self.cfg.solver=="LARS"):
            optim_params = [{'params': filter(lambda p: p.requires_grad, self.linear_layer.parameters()), 'lr': self.cfg.lr * self.lr_scaler}]
            optimizer = LARS(params=optim_params, weight_decay=self.cfg.weight_decay)
        elif(self.cfg.solver=="LAMB"):
            optim_params = [{'params': filter(lambda p: p.requires_grad, self.linear_layer.parameters()), 'lr': self.cfg.lr * self.lr_scaler}]
            optimizer = Lamb(params=optim_params, weight_decay=self.cfg.weight_decay)
        elif(self.cfg.solver=="AdamW"):
            optim_params = add_weight_decay(self.trainer.model, self.cfg.weight_decay)
            optimizer = optim.AdamW(params=optim_params, lr=self.cfg.lr * self.lr_scaler, betas=(0.9, 0.999))
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
            return 1

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
