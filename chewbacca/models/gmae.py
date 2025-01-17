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
# from lart.models.components.tokeizers.tokenizer import Tokenizer
from lightning import LightningModule
from moviepy.editor import ImageSequenceClip
from omegaconf import DictConfig
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch import nn
from torchmetrics import MeanMetric, MeanSquaredError
from torchmetrics.aggregation import CatMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

from chewbacca.utils import get_pylogger
from chewbacca.utils.lamb import LARS, Lamb

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


class GMAELitModule(LightningModule):
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
            if "mae" in self.cfg.training_type:
                import chewbacca.models.components.mae.models_mae as models_mae
                self.encoder = models_mae.__dict__[self.cfg.model_name](
                                                                        norm_pix_loss=True,
                                                                        img_size=self.cfg.input_size,
                                                                        number_of_frames=self.num_frames,
                                                                        num_gaussian=self.cfg.vocab_size,
                                                                        scale_factor=self.cfg.scale_factor,
                                                                        scale_vocab=self.cfg.scale_vocab,
                                                                    )
            if "vit" in self.cfg.training_type:
                import chewbacca.models.components.mae.models_vit as models_vit
                self.encoder = models_vit.__dict__[self.cfg.model_name](
                                                                        global_pool="avg",
                                                                        img_size = self.cfg.input_size,
                                                                        class_token=True,
                                                                        num_classes=self.cfg.num_classes,
                                                                        drop_path_rate=self.cfg.drop_path,
                                                                    )
                
        else:
            import chewbacca.models.components.mae.models_mae_finetune as models_mae
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
                                                                    frame_zero=self.cfg.frame_zero
                                                                )
            
        
        num_hidden_layers = len(self.encoder.blocks)
        hsize = self.encoder.patch_embed.proj.weight.shape[0]

        self.linear_layer = nn.ModuleList([torch.nn.Sequential(torch.nn.BatchNorm1d(hsize, affine=False, eps=1e-6), nn.Linear(hsize,self.cfg.num_classes)) for i in range(num_hidden_layers)])

        # setup meters
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # for imagenet_classification accuracy, also make it nn.ModuleList
        self.train_acc = nn.ModuleList([Accuracy(task="multiclass", num_classes=self.cfg.num_classes) for i in range(num_hidden_layers)])
        self.val_acc = nn.ModuleList([Accuracy(task="multiclass", num_classes=self.cfg.num_classes) for i in range(num_hidden_layers)])
        self.test_mse = MeanSquaredError()
        self.test_ssim = StructuralSimilarityIndexMeasure()
        self.test_psnr = PeakSignalNoiseRatio()
        self.test_rfid = FrechetInceptionDistance()

        # create folders for storing results
        os.makedirs(self.cfg.storage_folder + "/results/", exist_ok=True)
        os.makedirs(self.cfg.storage_folder + "/tests/", exist_ok=True)
        os.makedirs(self.cfg.storage_folder + "/videos/", exist_ok=True)
        log.info("Storage folder : " + self.cfg.storage_folder)

        if(self.cfg.solver=="LARS" or self.cfg.solver=="LAMB"):
            # freeze all layers except fc
            for _, p in self.encoder.named_parameters():
                p.requires_grad = False

        self.mixup_fn = None
        if self.cfg.task=="finetune" and not "no-mixup" in self.cfg.training_type:
            mixup_active = self.cfg.mixup > 0 or self.cfg.cutmix > 0. or self.cfg.cutmix_minmax is not None
            if mixup_active:
                log.info("Mixup is activated!")
                self.mixup_fn = Mixup(
                    mixup_alpha=self.cfg.mixup, cutmix_alpha=self.cfg.cutmix, cutmix_minmax=self.cfg.cutmix_minmax,
                    prob=self.cfg.mixup_prob, switch_prob=self.cfg.mixup_switch_prob, mode=self.cfg.mixup_mode,
                    label_smoothing=0.1, num_classes=self.cfg.num_classes)
                
            if self.mixup_fn is not None:
                # smoothing is handled with mixup label transform
                self.criterion = SoftTargetCrossEntropy()

  
    def forward_loss(self, imgs, pred, mask, frame_num=None, deltas=None, additional_data=None):
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
            # if(torch.sum(pred[i])>0):
            if True:
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
                    if self.cfg.random_frames and frame_num is not None:
                        loss += torch.nn.functional.mse_loss(torch.cat([imgs_[i][:, 0:1], imgs_[i][:, frame_num:frame_num+1]], axis=1), 
                                                        pred[i].permute(3, 0, 1, 2)) # 0, t
                        # loss += torch.nn.functional.mse_loss(imgs_[i][:, frame_num-1:frame_num+1], pred[i].permute(3, 0, 1, 2)) # t, t+1

                    else:
                        loss += torch.nn.functional.mse_loss(imgs_[i][:, :self.num_frames], pred[i].permute(3, 0, 1, 2))
                count += 1
        if self.cfg.deltas_reg_weight > 0:
            means, scales, quats = deltas
            loss += torch.linalg.norm(means) * self.cfg.deltas_reg_weight
            loss += torch.linalg.norm(scales) * self.cfg.deltas_reg_weight
            loss += torch.linalg.norm(quats) * self.cfg.deltas_reg_weight
        return loss

    def step(self, batch: Any, batch_idx: int, return_images=False):

        # for image based pretraining
        if(self.cfg.dataset_type=="imagenet"):

            image = batch[0] # bs, 3, h, w
            labels = batch[1]
            device = image.device
            dtype = image.dtype

            logits_cls = []
            extras_    = {}
            logits_    = {}

            # if "mae" in self.cfg.training_type:
            #     latent, mask, ids_restore, latent_layers = self.encoder.forward_encoder(image, mask_ratio=0.75)
            #     pred = self.encoder.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            #     loss = self.encoder.forward_loss(image, pred, mask)
            #     loss_dict  = {"loss": loss}

            if "vit" in self.cfg.training_type:
                if self.mixup_fn is not None and self.training:
                    image, labels = self.mixup_fn(image, labels)

                latent_layers = self.encoder.forward_features(image)
                loss_dict = {"loss": torch.tensor(0).to(device=device).to(dtype=dtype)}
                for i in range(len(self.encoder.blocks)):
                    cls_token = latent_layers
                    if "full-finetuning" in self.cfg.training_type  and i==len(self.encoder.blocks)-1:
                        logits__ = self.linear_layer[i](cls_token)
                    else:
                        logits__ = self.linear_layer[i](cls_token.detach())
                    if torch.sum(labels!=-1)>0:

                        if self.mixup_fn is not None and self.training:
                            # smoothing is handled with mixup label transform
                            outputs = logits__[labels!=-1].to(device, non_blocking=True)
                            targets = labels[labels!=-1].to(device, non_blocking=True)
                            loss_cls = self.criterion(outputs, targets)
                        else:
                            loss_cls = F.cross_entropy(logits__[labels!=-1], labels[labels!=-1].to(device=device))
                        loss_dict["loss_cls_" + str(i)] = loss_cls
                        logits_cls.append(logits__)
                    else:
                        loss_dict["loss_cls_" + str(i)] = torch.tensor(0).to(device=device).to(dtype=dtype)
                        logits_cls.append([])
                logits_["logits_cls"] = logits_cls

                return loss_dict, extras_, logits_

            if "gaussian" in self.cfg.training_type:
                latent, mask, ids_restore, latent_layers = self.encoder.forward_encoder(image, mask_ratio=self.cfg.mask_ratio)
                x_points_main = self.encoder.forward_decoder(latent, ids_restore)
                pred_main = self.encoder.forward_render(x_points_main)  # [N, L, p*p*3]
                loss = self.forward_loss(image[:, :, None, :, :], pred_main, mask, additional_data=None)

                loss_dict  = {"loss": loss}
                if "mythostra" in self.cfg.training_type:
                    for j in [32,64,128,256,512]:
                        x_points_ = self.encoder.forward_decoder(latent, ids_restore, limit_gaussian=j)
                        pred_ = self.encoder.forward_render(x_points_, limit_gaussian=j) # [N, L, p*p*3]

                        loss_ = self.forward_loss(image[:, :, None, :, :], pred_, mask, additional_data=None)
                        loss_dict[f"loss_{j}"] = loss_ / (self.cfg.vocab_size - j)

                # save the masked images and reconstructed images
                if "save-images" in self.cfg.training_type and batch_idx<1000000:
                    if "no-mask" in self.cfg.training_type:
                        latent, mask, ids_restore, latent_layers = self.encoder.forward_encoder(image, mask_ratio=0.0)
                    # renormalize to 0,1
                    imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device=device).to(dtype=dtype)
                    imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device=device).to(dtype=dtype)
                    imgs = (image * imagenet_std[None, :, None, None]) + imagenet_mean[None, :, None, None]
                    imgs = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

                    preds_all = []
                    if "save-images-z" in self.cfg.training_type:
                        x_points_ = self.encoder.forward_decoder(latent, ids_restore)
                        for j in [32,64,128,256,512,-1]:
                            pred_ = self.encoder.forward_render(x_points_, limit_gaussian_z=j)
                            preds_all.append(pred_)
                        preds_2 = torch.stack(preds_all, dim=0)
                        preds = preds_2[:, :, 0].permute(1, 2, 0, 3, 4).reshape(preds_2.shape[1], preds_2.shape[3], preds_2.shape[0]*preds_2.shape[4], 3)
                        preds = (preds.detach().cpu().numpy() * 255).astype(np.uint8)
                    elif "save-images-m" in self.cfg.training_type:
                        for j in [32,64,128,256,512,-1]:
                            x_points_m = self.encoder.forward_decoder(latent, ids_restore, limit_gaussian=j)
                            pred_ = self.encoder.forward_render(x_points_m, limit_gaussian=j)
                            preds_all.append(pred_)
                        preds_2 = torch.stack(preds_all, dim=0)
                        preds = preds_2[:, :, 0].permute(1, 2, 0, 3, 4).reshape(preds_2.shape[1], preds_2.shape[3], preds_2.shape[0]*preds_2.shape[4], 3)
                        preds = (preds.detach().cpu().numpy() * 255).astype(np.uint8)
                    else:
                        preds = (pred[:, 0].detach().cpu().numpy() * 255).astype(np.uint8)


                    # get points on xy density map
                    x_points = self.encoder.forward_decoder(latent, ids_restore)
                    _, xys = self.encoder.forward_render(x_points, return_gaussians=True, limit_gaussian_z=-1)
                    plane_ = torch.zeros(len(xys), self.cfg.input_size, self.cfg.input_size, 3).to(device=device).to(dtype=dtype)
                    # try:
                    #     for i in range(len(xys)):
                    #         for j in range(len(xys[i][0])):
                    #             if xys[i][1][j]>0:
                    #                 plane_[i, int(xys[i][0][j][1]), int(xys[i][0][j][0])] = 1
                    # except:
                    #     pass
                    plane_ = (plane_.detach().cpu().numpy() * 255).astype(np.uint8)

                    # get depth maps
                    pred_depth = self.encoder.forward_render(x_points, return_depth=True)
                    pred_depth = pred_depth[:, 0]
                    pred_depth = (pred_depth.detach().cpu().numpy() * 255).astype(np.uint8)

                    final_image = np.concatenate([imgs, preds, plane_, pred_depth], axis=2).reshape(-1, (imgs.shape[2]+preds.shape[2]+plane_.shape[2]+pred_depth.shape[2]), 3)
                    final_image = final_image[:, :, ::-1]
                    # save final image with epoch and batch index
                    train_ = "train" if self.training else "val"
                    global_rank = self.trainer.global_rank
                    if global_rank==0:
                        cv2.imwrite(self.cfg.storage_folder + f"/results/{train_}_{self.current_epoch}_{batch_idx}.png", final_image)
                        joblib.dump(xys, self.cfg.storage_folder + f"/results/{train_}_{self.current_epoch}_{batch_idx}.pkl")


            # for linear probe
            for i in range(len(latent_layers)):
                cls_token = latent_layers[i].mean(1)
                logits__ = self.linear_layer[i](cls_token.detach())
                if torch.sum(labels!=-1)>0:
                    if "cls-label-smooth" in self.cfg.training_type:
                        value_ = float(self.cfg.training_type.split("cls-label-smooth")[1].split("_")[0])
                        loss_cls = F.cross_entropy(logits__[labels!=-1], labels[labels!=-1].to(device=device), label_smoothing=value_)
                    else:
                        loss_cls = F.cross_entropy(logits__[labels!=-1], labels[labels!=-1].to(device=device))
                    loss_dict["loss_cls_" + str(i)] = loss_cls
                    logits_cls.append(logits__)
                else:
                    loss_dict["loss_cls_" + str(i)] = torch.tensor(0).to(device=device).to(dtype=dtype)
                    logits_cls.append([])
            logits_["logits_cls"] = logits_cls

            if return_images:
                return loss_dict, extras_, logits_, pred_main

            return loss_dict, extras_, logits_

        # for video based pretraining
        elif(self.cfg.dataset_type=="video"):

            video = batch[0] # bs, 3, t, h, w
            labels = batch[1]
            dataset = batch[2] # dataset index
            device = video.device
            dtype = video.dtype
            if "gaussian" in self.cfg.training_type:
                # loss, pred, mask, latent, latent_layers = self.encoder(video, mask_ratio=0.95)
                if "rand-mask" in self.cfg.training_type:
                    p_ = np.random.rand()
                    if p_ < 0.7:
                        mask_ = 0.9
                    else:
                        mask_ = (np.random.rand() + 1)/2
                else:
                    mask_ = self.cfg.mask_ratio
                latent, mask, ids_restore, latent_layers = self.encoder.forward_encoder(video, mask_ratio=mask_)
                if self.cfg.random_frames:
                    x_points, random_frame = self.encoder.forward_decoder(latent, ids_restore)
                    pred, deltas = self.encoder.forward_render(x_points, return_deltas=True)  # [N, L, p*p*3]
                    loss = self.forward_loss(video, pred, mask, frame_num=random_frame, deltas=deltas, additional_data=None)
                else:
                    x_points = self.encoder.forward_decoder(latent, ids_restore)
                    pred, deltas = self.encoder.forward_render(x_points, return_deltas=True)  # [N, L, p*p*3]
                    loss = self.forward_loss(video, pred, mask, deltas=deltas, additional_data=None)


                # save the masked images and reconstructed images
                if "save-images" in self.cfg.training_type and batch_idx<1 or self.cfg.inference.testing and batch_idx % 50 == 0:
                    if "no-mask" in self.cfg.training_type:
                        latent, mask, ids_restore, latent_layers = self.encoder.forward_encoder(video, mask_ratio=0.0)
                    # renormalize to 0,1
                    imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device=device).to(dtype=dtype)
                    imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device=device).to(dtype=dtype)
                    imgs = (video * imagenet_std[None, :, None, None, None]) + imagenet_mean[None, :, None, None, None]
                    imgs = (imgs.permute(0, 2, 3, 4, 1).cpu().numpy() * 255).astype(np.uint8)

                    preds_all = []
                    if "save-images-z" in self.cfg.training_type:
                        with torch.no_grad():
                            if not self.cfg.random_frames:
                                x_points = self.encoder.forward_decoder(latent, ids_restore)
                                
                            for j in [32,64,128,256,512,-1]:
                                if not self.cfg.random_frames:
                                    pred_ = self.encoder.forward_render(x_points, limit_gaussian_z=j, return_depth=self.cfg.inference.depth, return_corres=self.cfg.inference.correspondences)
                                else:
                                    pred_ = self.encoder.forward_render_all_frames(latent, ids_restore, limit_gaussian_z=j, return_depth=self.cfg.inference.depth, return_corres=self.cfg.inference.correspondences)
                                preds_all.append(pred_)
                        preds_2 = torch.stack(preds_all, dim=0)
                        pred_ab = []
                        for i in range(preds_2.shape[2]):
                            TMP_ = preds_2[:, :, i].permute(1, 2, 0, 3, 4).reshape(preds_2.shape[1], preds_2.shape[3], preds_2.shape[0]*preds_2.shape[4], 3)
                            TMP_ = (TMP_.detach().cpu().numpy() * 255).astype(np.uint8)
                            pred_ab.append(TMP_)

                    final_image_ = []
                    for i in range(len(pred_ab)):
                        final_image = np.concatenate([imgs[:, i], pred_ab[i]], axis=2).reshape(-1, (imgs[:, 0].shape[2]+pred_ab[0].shape[2]), 3)
                        if "k400" in self.cfg.training_type or "ucf101" in self.cfg.training_type:
                            final_image_.append(final_image)
                        else:
                            final_image_.append(final_image[:, :, ::-1])

                    # save final image with epoch and batch index
                    train_ = "train" if self.training else "val"
                    global_rank = self.trainer.global_rank
                    if global_rank==0 and not self.cfg.inference.testing:
                        # cv2.imwrite(self.cfg.storage_folder + f"/results/{train_}_{self.current_epoch}_{batch_idx}.png", final_image)
                        # make video with final_image_a, final_image_b using moviepy
                        clip = ImageSequenceClip(final_image_, fps=1)  # fps can be adjusted to your need
                        clip.write_videofile(f"{self.cfg.storage_folder}/results/{train_}_{self.current_epoch}_{batch_idx}_video.mp4", codec='libx264')
                    
                    if self.cfg.inference.testing:
                        title = "depth" if self.cfg.inference.depth else "corr" if  self.cfg.inference.correspondences else "test"
                        clip = ImageSequenceClip(final_image_, fps=1)  # fps can be adjusted to your need
                        clip.write_videofile(f"{self.cfg.storage_folder}/tests/{title}_{self.current_epoch}_{batch_idx}_video.mp4", codec='libx264')


            # default return values
            # check if loss_index.mean is nan
            loss_dict  = {"loss": loss}

            # track loss at each token position
            extras_    = {} #{"loss_index": loss_index.mean(dim=0)}
            logits_    = {}

            # for linear probeing: for k400
            logits_cls = []
            # for linear probe
            labels_ = batch[1]
            for i in range(len(latent_layers)):
                cls_token = latent_layers[i].mean(1)
                logits__ = self.linear_layer[i](cls_token.detach())
                if torch.sum(labels!=-1)>0:
                    if "cls-label-smooth" in self.cfg.training_type:
                        value_ = float(self.cfg.training_type.split("cls-label-smooth")[1].split("_")[0])
                        loss_cls = F.cross_entropy(logits__[labels!=-1], labels[labels!=-1].to(device=device), label_smoothing=value_)
                    else:
                        loss_cls = F.cross_entropy(logits__[labels!=-1], labels[labels!=-1].to(device=device))
                    loss_dict["loss_cls_" + str(i)] = loss_cls
                    logits_cls.append(logits__)
                else:
                    loss_dict["loss_cls_" + str(i)] = torch.tensor(0).to(device=device).to(dtype=dtype)
                    logits_cls.append([])
            logits_["logits_cls"] = logits_cls

            return loss_dict, extras_, logits_

    def on_train_epoch_start(self) -> None:
        torch.cuda.empty_cache()

    def training_step(self, batch: Any, batch_idx: int):

        loss_dict, extra, logits_cls = self.step(batch, batch_idx)
        loss = sum([v for k,v in loss_dict.items()])

        # update and log metrics
        self.train_loss(loss.item())
        for key in loss_dict.keys():
            self.log("train/loss/" + key, loss_dict[key].item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            

        # for linear probe accuracy on imagenet and k400
        if("logits_cls" in logits_cls.keys()):
            for i, layer_ in enumerate(self.linear_layer):
                if len(logits_cls['logits_cls'][i])>0:
                    self.train_acc[i](logits_cls['logits_cls'][i], batch[1])


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
        if(self.cfg.promt):
            self.generate(batch, batch_idx)
            return {"loss": 0}

        if "mae" in self.cfg.training_type or "vit" in self.cfg.training_type or self.cfg.dataset_type=="video":
            loss_dict, extra, logits_cls = self.step(batch, batch_idx, return_images=False)
        else:
            loss_dict, extra, logits_cls, pred = self.step(batch, batch_idx, return_images=True)

            # update metrics
            # # computed Metrics:
            device = pred.device
            dtype = pred.dtype
            imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device=device).to(dtype=dtype)
            imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device=device).to(dtype=dtype)
            x_in = batch[0]
            x_in_ = (x_in * imagenet_std[None, :, None, None]) + imagenet_mean[None, :, None, None]
            x_rgb_ = pred[:, 0].permute(0, 3, 1, 2).contiguous()

            # MSE
            self.test_mse.update(x_rgb_, x_in_)
            # SSIM
            self.test_ssim.update(x_rgb_, x_in_)
            # PSNR
            self.test_psnr.update(x_rgb_, x_in_)
            # rFID take uint 8
            x_rgb_2 = (torch.clamp(x_rgb_, 0, 1) * 255).type(torch.uint8)
            x_in_2 = (torch.clamp(x_in_, 0, 1) * 255).type(torch.uint8)
            # rFID
            self.test_rfid.update(x_rgb_2, real=False)
            self.test_rfid.update(x_in_2, real=True)
            
        loss = sum([v for k,v in loss_dict.items()])

        # update and log metrics
        self.val_loss(loss.item())
        for key in loss_dict.keys():
            self.log("val/loss/" + key, loss_dict[key].item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # for linear probe accuracy on imagenet and k400
        if("logits_cls" in logits_cls.keys()):
            for i, layer_ in enumerate(self.linear_layer):
                if len(logits_cls['logits_cls'][i])>0:
                    self.val_acc[i](logits_cls['logits_cls'][i], batch[1])

        # for terminal logging
        self.log_iter_stats(batch_idx)

        # deleting tensors
        del loss_dict, batch


        return {"loss": loss}

    def on_validation_epoch_end(self):
        log.info("\n " + self.cfg.storage_folder +  " : Validation epoch " + str(self.current_epoch) + " ended.")

        if "mae" not in self.cfg.training_type and "vit" not in self.cfg.training_type and self.cfg.dataset_type!="video":

            total_mse = self.test_mse.compute()
            self.log(f"mse", total_mse, sync_dist=True)

            total_ssim = self.test_ssim.compute()
            self.log(f"ssim", total_ssim, sync_dist=True)

            total_psnr = self.test_psnr.compute()
            self.log(f"psnr", total_psnr, sync_dist=True)

            total_fid = self.test_rfid.compute()
            self.log(f"rfid", total_fid, sync_dist=True)

            # reset the metric
            self.val_loss.reset()
            self.train_loss.reset()
            self.test_mse.reset()
            self.test_ssim.reset()
            self.test_psnr.reset()
            self.test_rfid.reset()

        # for linear probe accuracy on imagenet and k400, compute and log the accuracy
        if("imagenet" in self.cfg.training_type or "k400" in self.cfg.training_type or "ucf101" in self.cfg.training_type):
            for i, layer_ in enumerate(self.linear_layer):
                self.log("val/linear_probe_acc_" + str(i), self.val_acc[i].compute().item(), prog_bar=True, sync_dist=True)
                log.info("Accuracy : " + str(self.val_acc[i].compute().item()))
                # reset the metric
                self.val_acc[i].reset()
                self.train_acc[i].reset()

        # reset the metric
        self.val_loss.reset()
        self.train_loss.reset()


    def log_iter_stats(self, cur_iter):

        if(cur_iter%self.cfg.log_frequency != 0):
            return 0

        mem_usage = gpu_mem_usage()
        try:
            stats = {
                "epoch": "{}/{}".format(self.current_epoch, self.trainer.max_epochs),
                "iter": "{}/{}".format(cur_iter + 1, self.trainer.num_training_batches if self.training else self.trainer.num_val_batches),
                "train_loss": "%.4f"%(self.train_loss.compute().item()),
                "val_loss": "%.4f"%(self.val_loss.compute().item()),
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
