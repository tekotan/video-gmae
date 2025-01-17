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

from chewbacca.utils import tapvid_viz_utils

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


class FinetuneLitModule(LightningModule):
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

        self.num_frames = self.cfg.seq_length

        if "point-tracking" in self.cfg.training_type:
            self.mode = "tracking"
        else:
            raise ValueError("Must specify finetuning task")
            
        import chewbacca.models.components.mae.models_mae_finetune as models_mae
        self.encoder = models_mae.__dict__[self.cfg.model_name](
                                                                mode=self.mode,
                                                                norm_pix_loss=True,
                                                                num_gaussian=self.cfg.vocab_size,
                                                                img_size=self.cfg.input_size,
                                                                number_of_frames=self.num_frames,
                                                                reuse_decoder=self.cfg.finetune_params.reuse_decoder,
                                                                num_fourier_features=self.cfg.finetune_params.num_fourier_features,
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

        # create folders for storing results
        os.makedirs(self.cfg.storage_folder + "/results/", exist_ok=True)
        os.makedirs(self.cfg.storage_folder + "/tests/", exist_ok=True)
        os.makedirs(self.cfg.storage_folder + "/videos/", exist_ok=True)
        log.info("Storage folder : " + self.cfg.storage_folder)

  
    def forward_loss(self, true, pred, additional_data=None):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        loss = 0
        count = 0
        for i in range(pred[0].shape[0]):
            # if(torch.sum(pred[i])>0):
            if self.mode == "tracking":
                tracks_true, occluded_true = true
                tracks_pred, occluded_pred = pred
                
                first_invalid_index = torch.where(additional_data[i, :, 0] == -1)[0]
                if len(first_invalid_index) == 0:
                    first_invalid_index = tracks_pred.shape[1]
                else:
                    first_invalid_index = first_invalid_index[0]

                active_tracks_pred = tracks_pred[i, :first_invalid_index]
                active_occluded_pred = occluded_pred[i, :first_invalid_index]

                active_tracks_true = tracks_true[i, :first_invalid_index]
                active_occluded_true = occluded_true[i, :first_invalid_index]

                loss += torch.nn.functional.mse_loss(active_tracks_true, active_tracks_pred)
                loss += torch.nn.functional.binary_cross_entropy(active_occluded_pred.float(), active_occluded_true.float())
            count += 1
        return loss

    def step(self, batch: Any, batch_idx: int, return_images=False):
        video = batch[0] # bs, 3, t, h, w
        query_points = batch[1]
        target_points = batch[2] # dataset index
        occluded_points = batch[3]
        device = video.device
        dtype = video.dtype

        if "rand-mask" in self.cfg.training_type:
            p_ = np.random.rand()
            if p_ < 0.7:
                mask_ = 0.9
            else:
                mask_ = (np.random.rand() + 1)/2
        else:
            mask_ = self.cfg.mask_ratio

        latent, mask, ids_restore, latent_layers = self.encoder.forward_encoder(video, mask_ratio=mask_)

        x_points, occluded = self.encoder.forward_decoder(latent, ids_restore, query_points) # B x 256 x T x 2
        
        loss = self.forward_loss((target_points, occluded_points), (x_points, occluded), additional_data=query_points)
        # save the masked images and reconstructed images
        if "save-images" in self.cfg.training_type and batch_idx<1 or self.cfg.inference.testing:
            if "no-mask" in self.cfg.training_type:
                latent, mask, ids_restore, latent_layers = self.encoder.forward_encoder(video, mask_ratio=0.0)
            x_points, occlusions_pred = self.encoder.forward_decoder(latent, ids_restore, query_points) # B x 256 x T x 2
            # renormalize to 0,1
            imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device=device).to(dtype=dtype)
            imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device=device).to(dtype=dtype)
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

                active_tracks_pred = x_points[i, :first_invalid_index].detach().cpu().numpy()
                active_tracks_true = target_points[i, :first_invalid_index].detach().cpu().numpy()

                active_occluded_points = occluded_points[i, :first_invalid_index].detach().cpu().numpy()
                active_occlusions_pred = occlusions_pred[i, :first_invalid_index].detach().cpu().numpy()
                
                painted_frames_pred = tapvid_viz_utils.paint_point_track(
                    imgs[i],
                    active_tracks_pred * scale_factor,
                    ~(active_occlusions_pred > 0.5),
                )
                
                painted_frames_true = tapvid_viz_utils.paint_point_track(
                    imgs[i],
                    active_tracks_true * scale_factor,
                    ~(active_occluded_points > 0.5),
                )

                final_image = np.concatenate([painted_frames_true, painted_frames_pred], axis=2)
                final_image_.append(final_image)
            final_image_ = np.concatenate(final_image_, axis=1)
            final_image_ = [frame for frame in final_image_]
            # save final image with epoch and batch index
            train_ = "train" if self.training else "val"
            global_rank = self.trainer.global_rank
            if global_rank==0 and not self.cfg.inference.testing:
                # cv2.imwrite(self.cfg.storage_folder + f"/results/{train_}_{self.current_epoch}_{batch_idx}.png", final_image)
                # make video with final_image_a, final_image_b using moviepy
                clip = ImageSequenceClip(final_image_, fps=1).resize(2)  # fps can be adjusted to your need
                clip.write_videofile(f"{self.cfg.storage_folder}/results/{train_}_{self.current_epoch}_{batch_idx}_video.mp4", codec='libx264')
            
            if self.cfg.inference.testing:
                title = "depth" if self.cfg.inference.depth else "corr" if  self.cfg.inference.correspondences else "test"
                clip = ImageSequenceClip(final_image_, fps=1).resize(2)  # fps can be adjusted to your need
                clip.write_videofile(f"{self.cfg.storage_folder}/tests/{title}_{self.current_epoch}_{batch_idx}_video.mp4", codec='libx264')


        # default return values
        # check if loss_index.mean is nan
        loss_dict  = {"loss": loss}

        # track loss at each token position
        extras_    = {"x_points": x_points} #{"loss_index": loss_index.mean(dim=0)}
        logits_    = {}

        # for linear probeing: for k400
        # logits_cls = []
        # # for linear probe
        # labels_ = batch[1]
        # for i in range(len(latent_layers)):
        #     cls_token = latent_layers[i].mean(1)
        #     logits__ = self.linear_layer[i](cls_token.detach())
        #     if torch.sum(labels!=-1)>0:
        #         if "cls-label-smooth" in self.cfg.training_type:
        #             value_ = float(self.cfg.training_type.split("cls-label-smooth")[1].split("_")[0])
        #             loss_cls = F.cross_entropy(logits__[labels!=-1], labels[labels!=-1].to(device=device), label_smoothing=value_)
        #         else:
        #             loss_cls = F.cross_entropy(logits__[labels!=-1], labels[labels!=-1].to(device=device))
        #         loss_dict["loss_cls_" + str(i)] = loss_cls
        #         logits_cls.append(logits__)
        #     else:
        #         loss_dict["loss_cls_" + str(i)] = torch.tensor(0).to(device=device).to(dtype=dtype)
        #         logits_cls.append([])
        # logits_["logits_cls"] = logits_cls

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
        if self.mode == "tracking":
            loss_dict, extra, logits_cls = self.step(batch, batch_idx, return_images=False)
            
            query_points = batch[1]
            target_points = batch[2]
            filtered_x_points = extra["x_points"]
            filtered_x_points[query_points[:, :, 0] == -1] = 0
            self.test_mse.update(target_points.contiguous(), filtered_x_points.contiguous())
            
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

        if self.mode == "tracking":

            total_mse = self.test_mse.compute()
            self.log(f"mse", total_mse, sync_dist=True)

            # reset the metric
            self.val_loss.reset()
            self.train_loss.reset()
            self.test_mse.reset()

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
