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

from torchvision.ops import complete_box_iou_loss

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


def calculate_iou(boxes1, boxes2):
    """Calculate IoU between two sets of boxes"""
    # boxes are in format [xmin, xmax, ymin, ymax]
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 2], boxes2[..., 2])
    x2 = torch.min(boxes1[..., 1], boxes2[..., 1])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    boxes1_area = (boxes1[..., 1] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 2])
    boxes2_area = (boxes2[..., 1] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 2])
    union = boxes1_area + boxes2_area - intersection

    iou = intersection / (union + 1e-6)
    return iou

class AverageJaccard(torch.nn.Module):
    def __init__(self, thresholds=[1, 2, 4, 8, 16]):
        super().__init__()
        self.thresholds = thresholds
        self.reset()

    def reset(self):
        self.total_jaccard = 0
        self.count = 0

    def compute(self):
        return self.total_jaccard / (self.count if self.count > 0 else 1)

    def update(self, pred_points, pred_visible, true_points, true_visible):
        batch_size = pred_points.shape[0]

        for batch_idx in range(batch_size):
            jaccard_sum = 0

            for threshold in self.thresholds:
                # Calculate distances between predicted and ground truth points
                distances = torch.norm(
                    pred_points[batch_idx].unsqueeze(1) - true_points[batch_idx].unsqueeze(0),
                    dim=-1
                )

                # Points within threshold distance
                within_threshold = distances <= threshold

                # True positives: predicted visible points that are within threshold of visible ground truth points
                true_positives = torch.sum(
                    (pred_visible[batch_idx].unsqueeze(1) & true_visible[batch_idx].unsqueeze(0) & within_threshold).float()
                )

                # False positives: predicted visible points that are either farther than threshold or ground truth is occluded
                false_positives = torch.sum(
                    pred_visible[batch_idx] & ~torch.any(within_threshold & true_visible[batch_idx].unsqueeze(0), dim=1)
                )

                # False negatives: ground truth visible points that are predicted occluded or farther than threshold
                false_negatives = torch.sum(
                    true_visible[batch_idx] & ~torch.any(within_threshold & pred_visible[batch_idx].unsqueeze(1), dim=0)
                )

                # Calculate Jaccard
                denominator = true_positives + false_positives + false_negatives
                jaccard = true_positives / (denominator + 1e-6)
                jaccard_sum += jaccard

            # Average over thresholds
            self.total_jaccard += jaccard_sum / len(self.thresholds)
            self.count += 1



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
            self.mode = "point-tracking"
        elif "object-tracking" in self.cfg.training_type:
            self.mode = "object-tracking"
        else:
            raise ValueError("Must specify finetuning task")

        import chewbacca.models.components.mae.models_mae_finetune as models_mae
        self.encoder = models_mae.__dict__[self.cfg.model_name](
                                                                mode=self.mode,
                                                                norm_pix_loss=True,
                                                                num_tracks=self.cfg.finetune_params.tracks_to_sample,
                                                                img_size=self.cfg.input_size,
                                                                number_of_frames=self.num_frames,
                                                                reuse_decoder=self.cfg.finetune_params.reuse_decoder,
                                                                num_fourier_features=self.cfg.finetune_params.num_fourier_features,
                                                                batch_prediction=self.cfg.finetune_params.batch_prediction,
                                                                new_readout_mode=self.cfg.finetune_params.new_readout_mode,
                                                                zero_t_prediction = self.cfg.finetune_params.zero_t_prediction,
                                                                autoregressive=self.cfg.finetune_params.autoregressive,
                                                                quantize_output_bins=self.cfg.finetune_params.quantize_output_bins,
                                                                dit_head=self.cfg.finetune_params.dit_head,
                                                            )
        if self.cfg.inference.testing:
            self.encoder.requires_grad = False


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

        # Add Average Jaccard metric
        if self.mode == "object-tracking":
            self.box_mse = MeanSquaredError()
            self.box_iou = MeanMetric()  # For tracking mean IoU
        elif self.mode == "point-tracking":
            self.average_jaccard = AverageJaccard()

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
        
        if self.mode == "point-tracking":
            B = pred[0].shape[0]
        else:
            B = pred.shape[0]
        
        if self.cfg.finetune_params.zero_t_prediction:
            frame_num = additional_data[1]
            additional_data = additional_data[0]
        
        for i in range(B):
            # if(torch.sum(pred[i])>0):
            if self.mode == "point-tracking":
                if self.cfg.finetune_params.zero_t_prediction:
                    tracks_true = torch.cat([true[0][:, :, 0:1], true[0][:, :, frame_num:frame_num+1]], dim=2)
                    occluded_true = torch.cat([true[1][:, :, 0:1], true[1][:, :, frame_num:frame_num+1]], dim=2)
                else:
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

                # Position loss using Huber loss instead of MSE
                if self.cfg.finetune_params.quantize_output_bins is not None:
                    quantized_targets = torch.round(active_tracks_true * self.cfg.finetune_params.quantize_output_bins)\
                                        .type(torch.int64).clip(min=0, max=self.cfg.finetune_params.quantize_output_bins-1)

                    for j in range(active_tracks_true.shape[1]):
                        for k in range(active_tracks_true.shape[2]):
                            loss += torch.nn.functional.cross_entropy(active_tracks_pred[:, j, k], quantized_targets[:, j, k]) * 100

                    loss += torch.nn.functional.binary_cross_entropy(active_occluded_pred.float(), active_occluded_true.float()) * 0.1
                elif self.cfg.finetune_params.dit_head:
                    loss += torch.nn.functional.mse_loss(active_tracks_true, active_tracks_pred) * 100
                    loss += torch.nn.functional.binary_cross_entropy(active_occluded_pred.float(), active_occluded_true.float()) * 0.1
                else:
                    loss += torch.nn.functional.huber_loss(active_tracks_true, active_tracks_pred) * 100
                    loss += torch.nn.functional.binary_cross_entropy(active_occluded_pred.float(), active_occluded_true.float()) * 0.1
            elif self.mode == "object-tracking":
                if self.cfg.finetune_params.zero_t_prediction:
                    target_boxes = torch.cat([true[:, :, 0:1], true[:, :, frame_num:frame_num+1]], dim=2)
                    pred_boxes = pred
                else:
                    target_boxes, pred_boxes = true, pred

                first_invalid_index = torch.where(additional_data[i, :, 0] == -1)[0]
                if len(first_invalid_index) == 0:
                    first_invalid_index = target_boxes.shape[1]
                else:
                    first_invalid_index = first_invalid_index[0]

                target_boxes = target_boxes[i, :first_invalid_index]
                pred_boxes = pred_boxes[i, :first_invalid_index]


                for j in range(len(target_boxes)):
                    pred_boxes_clone = pred_boxes[j].clone()
                    target_boxes_clone = target_boxes[j].clone()

                    first_invalid_index = torch.where(target_boxes_clone[:, 0] == -1)[0]
                    if len(first_invalid_index) == 0:
                        first_invalid_index = target_boxes_clone.shape[0]
                    else:
                        first_invalid_index = first_invalid_index[0]

                    pred_boxes_clone = pred_boxes_clone[:first_invalid_index]
                    target_boxes_clone = target_boxes_clone[:first_invalid_index]

                    if self.cfg.finetune_params.quantize_output_bins is not None:
                        quantized_targets = torch.round(target_boxes_clone * self.cfg.finetune_params.quantize_output_bins)\
                                            .type(torch.int64).clip(min=0, max=self.cfg.finetune_params.quantize_output_bins-1)
                        for j in range(target_boxes_clone.shape[1]):
                            loss += torch.nn.functional.cross_entropy(pred_boxes_clone[:, j], quantized_targets[:, j]) * 100
                    elif self.cfg.finetune_params.dit_head:
                        loss += torch.nn.functional.mse_loss(pred_boxes_clone, target_boxes_clone)
                    else:
                        iou_loss = complete_box_iou_loss(pred_boxes_clone, target_boxes_clone, reduction="mean")
                        iou_loss = torch.nan_to_num(iou_loss, nan=0.0)
                        loss += iou_loss
            count += 1
        return loss / B

    def step(self, batch: Any, batch_idx: int, return_images=False):
        video = batch[0] # bs, 3, t, h, w
        init_queries = batch[1]
        finetune_target = batch[2] # dataset index
        
        if self.mode == "point-tracking":
            occluded_points = batch[3]

        if "rand-mask" in self.cfg.training_type:
            p_ = np.random.rand()
            if p_ < 0.7:
                mask_ = 0.9
            else:
                mask_ = (np.random.rand() + 1)/2
        else:
            mask_ = self.cfg.mask_ratio
        latent, mask, ids_restore, latent_layers = self.encoder.forward_encoder(video, mask_ratio=mask_)
        if self.mode == "point-tracking":
            if self.cfg.finetune_params.batch_prediction:
                x_points, occlusions_pred, frame_num = self.encoder.forward_decoder(latent, ids_restore, init_queries) # B x 256 x T x 2
            else:
                batch_x_points = []
                batch_occluded = []
                if self.cfg.finetune_params.dit_head:
                    batch_dit_z_diff = []
                if self.cfg.finetune_params.zero_t_prediction:
                    frame_num = torch.randint(low=1, high=self.encoder.input_frames, size=())
                else:
                    frame_num = None
                for i in range(latent.shape[0]):
                    x_points_ = []
                    occluded_ = []
                    if self.cfg.finetune_params.dit_head:
                        dit_z_diff_ = []
                    for j in range(0, init_queries.shape[1], latent.shape[0]):
                        flat_init_queries = init_queries[i, j:j+latent.shape[0]].view(-1, 1, 3)
                        num_points = flat_init_queries.shape[0]
                        if self.cfg.finetune_params.autoregressive:
                            if self.cfg.finetune_params.quantize_output_bins is not None:
                                quantized_cond_target = torch.round(finetune_target[i, j:j+latent.shape[0], :-1] * self.cfg.finetune_params.quantize_output_bins) / self.cfg.finetune_params.quantize_output_bins
                                cond = torch.cat([occluded_points[i, j:j+latent.shape[0], :-1].unsqueeze(-1), quantized_cond_target], dim=-1)
                            else:
                                cond = torch.cat([occluded_points[i, j:j+latent.shape[0], :-1].unsqueeze(-1), finetune_target[i, j:j+latent.shape[0], :-1]], dim=-1)
                        else:
                            cond = flat_init_queries
                        if self.cfg.finetune_params.dit_head:
                            z1 = finetune_target[i, j:j+latent.shape[0], 1:]
                            z1 = z1.reshape(z1.shape[0]*z1.shape[1], z1.shape[2])
                            z0 = torch.randn_like(z1)
                            dit_z = (z0, z1)
                            dit_z_diff_.append((z1-z0).view(cond.shape[0], cond.shape[1], -1))
                        else:
                            dit_z = None
                            
                        if True:
                        # if np.random.rand() < 1.1:
                            x_points, occluded, frame_num = self.encoder.forward_decoder(latent[i:i+1].repeat(num_points, 1, 1), ids_restore[i:i+1].repeat(num_points, 1), cond, frame_num=frame_num, dit_training=True, dit_z=dit_z) # B x 256 x T x 2
                        else:
                            x_points, occluded = self.encoder.forward_decoder_autoreg(latent[i:i+1].repeat(num_points, 1, 1), ids_restore[i:i+1].repeat(num_points, 1), cond[:, 0:1], return_logits=True, dit_training=False) # B x 256 x T x 2
                            frame_num = None
                            occluded = occluded.squeeze(-1).squeeze(-1)
                        x_points_.append(x_points)
                        occluded_.append(occluded)

                    x_points = torch.cat(x_points_, dim=0)
                    occluded = torch.cat(occluded_, dim=0)

                    if self.cfg.finetune_params.dit_head:
                        dit_z_diff = torch.cat(dit_z_diff_, dim=0)
                        batch_dit_z_diff.append(dit_z_diff)

                    batch_x_points.append(x_points)
                    batch_occluded.append(occluded)                        


                x_points = torch.stack(batch_x_points, dim=0).squeeze(2)
                occlusions_pred = torch.stack(batch_occluded, dim=0).squeeze(2)
                if self.cfg.finetune_params.dit_head:
                    dit_z_diff = torch.stack(batch_dit_z_diff, dim=0).squeeze(2)
            if self.cfg.finetune_params.zero_t_prediction:
                loss = self.forward_loss((finetune_target, occluded_points), (x_points, occlusions_pred), additional_data=(init_queries, frame_num))
            elif self.cfg.finetune_params.autoregressive:
                if self.cfg.finetune_params.dit_head:
                    loss = self.forward_loss((dit_z_diff, occluded_points[:, :, 1:]), (x_points, occlusions_pred), additional_data=init_queries)
                else:
                    loss = self.forward_loss((finetune_target[:, :, 1:], occluded_points[:, :, 1:]), (x_points, occlusions_pred), additional_data=init_queries)
            else:
                loss = self.forward_loss((finetune_target, occluded_points), (x_points, occlusions_pred), additional_data=init_queries)
        elif self.mode == "object-tracking":
            if self.cfg.finetune_params.batch_prediction:
                box_pred, frame_num = self.encoder.forward_decoder(latent, ids_restore, init_queries) # B x 256 x T x 2
            else:
                batch_box_pred = []
                if self.cfg.finetune_params.dit_head:
                    batch_dit_z_diff = []
                if self.cfg.finetune_params.zero_t_prediction:
                    frame_num = torch.randint(low=1, high=self.encoder.input_frames, size=())
                else:
                    frame_num = None
                for i in range(latent.shape[0]):
                    box_pred_ = []
                    if self.cfg.finetune_params.dit_head:
                        dit_z_diff_ = []
                    for j in range(0, init_queries.shape[1], latent.shape[0]):
                        flat_init_queries = init_queries[i, j:j+latent.shape[0]].view(-1, 1, 4)
                        num_points = flat_init_queries.shape[0]
                        if self.cfg.finetune_params.autoregressive:
                            if self.cfg.finetune_params.quantize_output_bins is not None:
                                cond = torch.round(finetune_target[i, j:j+latent.shape[0], :-1] * self.cfg.finetune_params.quantize_output_bins) / self.cfg.finetune_params.quantize_output_bins
                            else:
                                cond = finetune_target[i, j:j+latent.shape[0], :-1]
                        else:
                            cond = flat_init_queries

                        if self.cfg.finetune_params.dit_head:
                            z1 = finetune_target[i, j:j+latent.shape[0], 1:]
                            z1 = z1.reshape(z1.shape[0]*z1.shape[1], z1.shape[2])
                            z0 = torch.randn_like(z1)
                            dit_z = (z0, z1)
                            dit_z_diff_.append((z1-z0).view(cond.shape[0], cond.shape[1], -1))
                        else:
                            dit_z = None

                        box_pred, frame_num = self.encoder.forward_decoder(latent[i:i+1].repeat(num_points, 1, 1), ids_restore[i:i+1].repeat(num_points, 1), cond, frame_num=frame_num, dit_training=True, dit_z=dit_z) # B x 256 x T x 
                        box_pred_.append(box_pred)

                    if self.cfg.finetune_params.dit_head:
                        dit_z_diff = torch.cat(dit_z_diff_, dim=0)
                        batch_dit_z_diff.append(dit_z_diff)

                    box_pred = torch.cat(box_pred_, dim=0)
                    batch_box_pred.append(box_pred)

                box_pred = torch.stack(batch_box_pred, dim=0).squeeze(2)
                if self.cfg.finetune_params.dit_head:
                    dit_z_diff = torch.stack(batch_dit_z_diff, dim=0).squeeze(2)
            if self.cfg.finetune_params.zero_t_prediction:
                loss = self.forward_loss(finetune_target, box_pred, additional_data=(init_queries, frame_num))
            elif self.cfg.finetune_params.autoregressive:
                if self.cfg.finetune_params.dit_head:
                    loss = self.forward_loss(dit_z_diff, box_pred, additional_data=init_queries)
                else:
                    loss = self.forward_loss(finetune_target[:, :, 1:], box_pred, additional_data=init_queries)
            else:
                loss = self.forward_loss(finetune_target, box_pred, additional_data=init_queries)

        # save the masked images and reconstructed images
        if "save-images" in self.cfg.training_type and batch_idx<1 or self.cfg.inference.testing:
            with torch.no_grad():
                if "no-mask" in self.cfg.training_type:
                    latent, mask, ids_restore, latent_layers = self.encoder.forward_encoder(video, mask_ratio=0.0)
                if self.mode == "point-tracking":
                    if self.cfg.finetune_params.quantize_output_bins is not None:
                        x_points_train = torch.cat([torch.round(finetune_target[:, :, 0:1] * self.cfg.finetune_params.quantize_output_bins) / self.cfg.finetune_params.quantize_output_bins, 
                            torch.argmax(x_points, dim=-1) / self.cfg.finetune_params.quantize_output_bins], dim=2)
                        occlusions_pred_train = torch.cat([occluded_points[:, :, 0:1], occlusions_pred.clone()], dim=2)
                    else:
                        x_points_train = torch.cat([finetune_target[:, :, 0:1], x_points.clone()], dim=2)
                        occlusions_pred_train = torch.cat([occluded_points[:, :, 0:1], occlusions_pred.clone()], dim=2)

                    if self.cfg.finetune_params.batch_prediction:
                        if self.cfg.finetune_params.zero_t_prediction:
                            x_points, occlusions_pred = self.encoder.forward_decoder_zero_t(latent, ids_restore, init_queries)
                        else:
                            x_points, occlusions_pred, frame_num = self.encoder.forward_decoder(latent, ids_restore, init_queries) # B x 256 x T x 2
                    else:
                        batch_x_points = []
                        batch_occlusions_pred = []
                        for i in range(latent.shape[0]):
                            x_points_ = []
                            occlusions_pred_ = []
                            for j in range(0, init_queries.shape[1], latent.shape[0]):
                                if self.cfg.finetune_params.autoregressive:
                                    if self.cfg.finetune_params.quantize_output_bins is not None:
                                        quantized_cond_target = torch.round(finetune_target[i, j:j+latent.shape[0], 0:1] * self.cfg.finetune_params.quantize_output_bins) / self.cfg.finetune_params.quantize_output_bins
                                        flat_init_queries = torch.cat([occluded_points[i, j:j+latent.shape[0], 0:1].unsqueeze(-1), quantized_cond_target], dim=-1)
                                    else:
                                        flat_init_queries = torch.cat([occluded_points[i, j:j+latent.shape[0], 0:1].unsqueeze(-1), finetune_target[i, j:j+latent.shape[0], 0:1]], dim=-1)
                                else:
                                    flat_init_queries = init_queries[i, j:j+latent.shape[0]].view(-1, 1, 3)
                                num_points = flat_init_queries.shape[0]
                                if self.cfg.finetune_params.zero_t_prediction:
                                    x_points, occlusions_pred = self.encoder.forward_decoder_zero_t(latent[i:i+1].repeat(num_points, 1, 1), ids_restore[i:i+1].repeat(num_points, 1), flat_init_queries)
                                elif self.cfg.finetune_params.autoregressive:
                                    x_points, occlusions_pred = self.encoder.forward_decoder_autoreg(latent[i:i+1].repeat(num_points, 1, 1), ids_restore[i:i+1].repeat(num_points, 1), flat_init_queries)
                                else:
                                    x_points, occlusions_pred, frame_num = self.encoder.forward_decoder(latent[i:i+1].repeat(num_points, 1, 1), ids_restore[i:i+1].repeat(num_points, 1), flat_init_queries) # B x 256 x T x 2
                                x_points_.append(x_points)
                                occlusions_pred_.append(occlusions_pred)

                            x_points = torch.cat(x_points_, dim=0)
                            occlusions_pred = torch.cat(occlusions_pred_, dim=0)
                            batch_x_points.append(x_points)
                            batch_occlusions_pred.append(occlusions_pred)

                        x_points = torch.stack(batch_x_points, dim=0).squeeze(2)
                        occlusions_pred = torch.stack(batch_occlusions_pred, dim=0).squeeze(2).squeeze(-1)
                    final_image_ = self.visualize_point_tracking(video, x_points, init_queries, finetune_target, occluded_points, occlusions_pred)
                    final_image_train_ = self.visualize_point_tracking(video, x_points_train, init_queries, finetune_target, occluded_points, occlusions_pred_train)

                    final_image_ = np.concatenate(final_image_, axis=1)
                    final_image_train_ = np.concatenate(final_image_train_, axis=1)
                    final_image_ = np.concatenate([final_image_train_, final_image_], axis=2)
                    final_image_ = [frame for frame in final_image_]
                elif self.mode == "object-tracking":
                    if self.cfg.finetune_params.quantize_output_bins is not None:
                        box_pred_train = torch.cat([torch.round(finetune_target[:, :, 0:1] * self.cfg.finetune_params.quantize_output_bins) / self.cfg.finetune_params.quantize_output_bins, 
                                torch.argmax(box_pred, dim=-1) / self.cfg.finetune_params.quantize_output_bins], dim=2)
                    else:
                        box_pred_train = torch.cat([finetune_target[:, :, 0:1], box_pred.clone()], dim=2)
                    if self.cfg.finetune_params.batch_prediction:
                        if self.cfg.finetune_params.zero_t_prediction:
                            box_pred = self.encoder.forward_decoder_zero_t(latent, ids_restore, init_queries)
                        else:
                            box_pred, frame_num = self.encoder.forward_decoder(latent, ids_restore, init_queries) # B x 256 x T x 2
                        final_image_ = self.visualize_object_tracking(video, box_pred, finetune_target, batch_idx)
                    else:
                        batch_box_pred = []
                        for i in range(latent.shape[0]):
                            box_pred_ = []
                            occlusions_pred_ = []
                            for j in range(0, init_queries.shape[1], latent.shape[0]):
                                flat_init_queries = init_queries[i, j:j+latent.shape[0]].view(-1, 1, 4)
                                num_points = flat_init_queries.shape[0]
                                if self.cfg.finetune_params.zero_t_prediction:
                                    box_pred = self.encoder.forward_decoder_zero_t(latent[i:i+1].repeat(num_points, 1, 1), ids_restore[i:i+1].repeat(num_points, 1), flat_init_queries)
                                elif self.cfg.finetune_params.autoregressive:
                                    if self.cfg.finetune_params.quantize_output_bins is not None:
                                        flat_init_queries = torch.round(finetune_target[i, j:j+latent.shape[0], 0:1] * self.cfg.finetune_params.quantize_output_bins) / self.cfg.finetune_params.quantize_output_bins

                                    box_pred = self.encoder.forward_decoder_autoreg(latent[i:i+1].repeat(num_points, 1, 1), ids_restore[i:i+1].repeat(num_points, 1), flat_init_queries)
                                else:
                                    box_pred, frame_num = self.encoder.forward_decoder(latent[i:i+1].repeat(num_points, 1, 1), ids_restore[i:i+1].repeat(num_points, 1), flat_init_queries) # B x 256 x T x 2
                                box_pred_.append(box_pred)

                            box_pred = torch.cat(box_pred_, dim=0)
                            batch_box_pred.append(box_pred)

                        box_pred = torch.stack(batch_box_pred, dim=0).squeeze(2)
                        final_image_ = self.visualize_object_tracking(video, box_pred, finetune_target, batch_idx)
                    final_image_train_ = self.visualize_object_tracking(video, box_pred_train, finetune_target, batch_idx)
                    final_image_ = np.concatenate(final_image_, axis=2)
                    final_image_train_ = np.concatenate(final_image_train_, axis=2)
                    final_image_ = np.concatenate([final_image_train_, final_image_], axis=1)
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
                title = "tapvid_typ" if "eval" in self.cfg.training_type else "kubric"
                clip = ImageSequenceClip(final_image_, fps=1).resize(2)  # fps can be adjusted to your need
                clip.write_videofile(f"{self.cfg.storage_folder}/tests/{title}_{self.current_epoch}_{batch_idx}_video.mp4", codec='libx264')


        # default return values
        # check if loss_index.mean is nan
        loss_dict  = {"loss": loss}

        # track loss at each token position
        if self.mode == "point-tracking":
            extras_    = {"x_points": x_points, "occluded_pred": occlusions_pred, "frame_num": frame_num} #{"loss_index": loss_index.mean(dim=0)}
        elif self.mode == "object-tracking":
            extras_    = {"box_pred": box_pred, "frame_num": frame_num}
        else:
            extras_    = {}
        logits_    = {}

        return loss_dict, extras_, logits_

    def visualize_object_tracking(self, video, pred_boxes, target_boxes, batch_idx):
        """Visualize object tracking predictions"""
        # Denormalize video
        imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device=video.device).to(dtype=video.dtype)
        imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device=video.device).to(dtype=video.dtype)
        imgs = (video * imagenet_std[None, :, None, None, None]) + imagenet_mean[None, :, None, None, None]
        imgs = (imgs.permute(0, 2, 3, 4, 1).cpu().numpy() * 255).astype(np.uint8)

        final_images = []
        for b in range(len(imgs)):
            sequence = []
            for t in range(len(imgs[b])):
                frame = imgs[b][t].copy()

                # Draw predicted boxes
                for pred_box, target_box in zip(pred_boxes[b, :, t].cpu().numpy(), target_boxes[b, :, t].cpu().numpy()):
                    # Draw ground truth box
                    ymin, xmin, ymax, xmax = target_box * self.cfg.input_size  # Scale to image size
                    if ymin < 0 or xmin < 0:
                        continue
                    cv2.rectangle(frame,
                                  (int(ymin), int(xmin)),
                                  (int(ymax), int(xmax)),
                                  (255, 0, 0), 2)  # Red for ground truth

                    ymin, xmin, ymax, xmax = pred_box * self.cfg.input_size  # Scale to image size

                    cv2.rectangle(frame,
                                  (int(ymin), int(xmin)),
                                  (int(ymax), int(xmax)),
                                  (0, 255, 0), 2)  # Green for predictions

                sequence.append(frame)
            sequence = np.stack(sequence)
            final_images.append(sequence)
        return final_images
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

            painted_frames_blank = tapvid_viz_utils.paint_point_track(
                np.ones_like(imgs[i]) * 255,
                active_tracks_pred * scale_factor,
                ~(active_occlusions_pred > 0.5),
            )

            final_image = np.concatenate([painted_frames_true, painted_frames_pred, painted_frames_blank], axis=2)
            final_image_.append(final_image)
        return final_image_
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
        del loss_dict, batch, extra

        return {"loss": loss}

    def on_train_epoch_end(self):
        log.info("\n " + self.cfg.storage_folder +  " : Training epoch " + str(self.current_epoch) + " ended.")

    def on_validation_epoch_start(self) -> None:
        torch.cuda.empty_cache()

    def validation_step(self, batch: Any, batch_idx: int):
        if(self.cfg.promt):
            self.generate(batch, batch_idx)
            return {"loss": 0}
        if self.mode == "point-tracking":
            loss_dict, extra, logits_cls = self.step(batch, batch_idx, return_images=False)
            
            query_points = batch[1]
            target_points = batch[2]
            occluded = batch[3]
            filtered_x_points = extra["x_points"]
            filtered_x_points[query_points[:, :, 0] == -1] = 0
            if extra["x_points"].shape[2] == 2:
                frame_num = extra["frame_num"]
                target_points = torch.cat([target_points[:, :, 0:1], target_points[:, :, frame_num:frame_num+1]], dim=2)
                occluded = torch.cat([occluded[:, :, 0:1], occluded[:, :, frame_num:frame_num+1]], dim=2)
            if extra["x_points"].shape[2] == target_points.shape[2] - 1:
                target_points = target_points[:, :, 1:]
                occluded = occluded[:, :, 1:]
            if self.cfg.finetune_params.quantize_output_bins is not None and \
                filtered_x_points.shape[-1] == self.cfg.finetune_params.quantize_output_bins:

                filtered_x_points = torch.argmax(filtered_x_points, dim=-1) / self.cfg.finetune_params.quantize_output_bins

            self.test_mse.update(target_points.contiguous(), filtered_x_points.contiguous())

            # Update Jaccard metric
            pred_points = filtered_x_points
            pred_visible = ~(extra.get("occluded_pred", torch.zeros_like(query_points[:,:,0])) > 0.5).squeeze(-1)
            true_points = target_points
            true_visible = ~(occluded>0.5)  # Assuming batch[3] contains occlusion information
            self.average_jaccard.update(pred_points, pred_visible, true_points, true_visible)
        if self.mode == "object-tracking":
            loss_dict, extra, logits_cls = self.step(batch, batch_idx, return_images=False)

            target_boxes = batch[2]
            pred_boxes = extra["box_pred"]

            if extra["box_pred"].shape[2] == 2:
                frame_num = extra["frame_num"]
                target_boxes = torch.cat([target_boxes[:, :, 0:1], target_boxes[:, :, frame_num:frame_num+1]], dim=2)
            if extra["box_pred"].shape[2] == target_boxes.shape[2] - 1:
                target_boxes = target_boxes[:, :, 1:]
            if self.cfg.finetune_params.quantize_output_bins is not None and \
                extra["box_pred"].shape[-1] == self.cfg.finetune_params.quantize_output_bins:
                
                pred_boxes = torch.argmax(pred_boxes, dim=-1) / self.cfg.finetune_params.quantize_output_bins
            self.box_mse.update(target_boxes.contiguous(), pred_boxes.contiguous())

            # Update IoU metric
            iou = calculate_iou(target_boxes, pred_boxes)
            self.box_iou.update(iou)

        loss = sum([v for k,v in loss_dict.items()])

        # update and log metrics
        self.val_loss(loss.item())
        for key in loss_dict.keys():
            self.log("val/loss/" + key, loss_dict[key].item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log Jaccard metric
        if self.mode == "point-tracking":
            jaccard_score = self.average_jaccard.compute()
            self.log("val/jaccard", jaccard_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        elif self.mode == "object-tracking":
            iou_score = self.box_iou.compute()
            self.log("val/iou", iou_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            box_mse = self.box_mse.compute()
            self.log("val/box_mse", box_mse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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

        if self.mode == "point-tracking":

            total_mse = self.test_mse.compute()
            self.log(f"mse", total_mse, sync_dist=True)

            # reset the metric
            self.val_loss.reset()
            self.train_loss.reset()
            self.test_mse.reset()
        elif self.mode == "object-tracking":
            total_mse = self.box_mse.compute()
            self.log(f"mse", total_mse, sync_dist=True)

            total_iou = self.box_iou.compute()
            self.log(f"iou", total_iou, sync_dist=True)

            # reset the metric
            self.val_loss.reset()
            self.train_loss.reset()
            self.box_mse.reset()
            self.box_iou.reset()

        # for linear probe accuracy on imagenet and k400, compute and log the accuracy
        if("imagenet" in self.cfg.training_type or "k400" in self.cfg.training_type or "ucf101" in self.cfg.training_type):
            for i, layer_ in enumerate(self.linear_layer):
                self.log("val/linear_probe_acc_" + str(i), self.val_acc[i].compute().item(), prog_bar=True, sync_dist=True)
                log.info("Accuracy : " + str(self.val_acc[i].compute().item()))
                # reset the metric
                self.val_acc[i].reset()
                self.train_acc[i].reset()
            if self.mode == "point-tracking":
                log.info("Jaccard Score : " + str(self.average_jaccard.compute().item()))
            elif self.mode == "object-tracking":
                log.info("IoU : " + str(self.box_iou.compute().item()))
                log.info("Box MSE : " + str(self.box_mse.compute().item()))

        # reset the metric
        if self.mode == "point-tracking":
            self.average_jaccard.reset()
        elif self.mode == "object-tracking":
            self.box_iou.reset()
            self.box_mse.reset()

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
