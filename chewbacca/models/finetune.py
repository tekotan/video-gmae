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
from torchmetrics import MeanMetric, MeanSquaredError, SumMetric
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
                                                                quantized_prediction=self.cfg.finetune_params.quantized_prediction,
                                                                quantize_output_bins=self.cfg.finetune_params.quantize_output_bins,
                                                                dit_head=self.cfg.finetune_params.dit_head,
                                                                use_dit_decoder=self.cfg.finetune_params.use_dit_decoder,
                                                                videomae=self.cfg.videomae,
                                                                mae_st=self.cfg.mae_st,
                                                                training_type=self.cfg.training_type,
                                                                freeze_encoder=self.cfg.finetune_params.freeze_encoder,
                                                            )
        if self.cfg.inference.testing:
            self.encoder.requires_grad = False

        if self.cfg.videomae:
            num_hidden_layers = len(self.encoder.videomae_model.encoder.layer) + 1
        elif self.cfg.mae_st:
            num_hidden_layers = len(self.encoder.mae_st_model.blocks)
        else:
            num_hidden_layers = len(self.encoder.blocks)
        hsize = self.encoder.patch_embed.proj.weight.shape[0]

        self.linear_layer = nn.ModuleList([torch.nn.Sequential(torch.nn.BatchNorm1d(hsize, affine=False, eps=1e-6), nn.Linear(hsize,self.cfg.num_classes)) for i in range(num_hidden_layers)])
        
        if self.cfg.finetune_params.use_dit_decoder:
            import chewbacca.models.components.dit.gaussian_diffusion as gd
            betas = gd.get_named_beta_schedule("linear", 1000)
            
            self.gaussian_diffusion = gd.GaussianDiffusion(
                betas=betas, model_mean_type=gd.ModelMeanType.EPSILON,
                model_var_type=gd.ModelVarType.FIXED_LARGE, loss_type=gd.LossType.MSE
            )
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
            self.average_jaccard = MeanMetric()
            self.avg_distance = MeanMetric()
            self.average_pts_within_thresh = MeanMetric()
            self.occlusion_accuracy = MeanMetric()
            self.occ_tp = SumMetric()
            self.occ_fp = SumMetric()
            self.occ_fn = SumMetric()

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
                if self.cfg.finetune_params.quantized_prediction:
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

                    if self.cfg.finetune_params.quantized_prediction:
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
            x_points, occlusions_pred = self.encoder.forward_decoder(latent, ids_restore, init_queries) # B x 256 x T x 2
            loss = self.forward_loss((finetune_target, occluded_points), (x_points, occlusions_pred), additional_data=init_queries)

        elif self.mode == "object-tracking":
            box_pred = self.encoder.forward_decoder(latent, ids_restore, init_queries) # B x 256 x T x 2
            loss = self.forward_loss(finetune_target, box_pred, additional_data=init_queries)

        # save the masked images and reconstructed images
        if "save-images" in self.cfg.training_type and batch_idx<1 or self.cfg.inference.testing:
            with torch.no_grad():
                if "no-mask" in self.cfg.training_type:
                    latent, mask, ids_restore, latent_layers = self.encoder.forward_encoder(video, mask_ratio=0.0)

                if self.mode == "point-tracking":
                    x_points, occlusions_pred = self.encoder.forward_decoder(latent, ids_restore, init_queries) # B x 256 x T x 2
                    
                    if self.cfg.inference.save_predictions:
                        data = (init_queries, x_points, occlusions_pred, finetune_target, occluded_points)
                        torch.save(data, f"{self.cfg.storage_folder}/tests/eval_{self.cfg.inference.context_length}_{self.current_epoch}_{batch_idx}.pt")

                    final_image_ = self.visualize_point_tracking(video, x_points, init_queries, finetune_target, occluded_points, occlusions_pred)
                    final_image_ = np.concatenate(final_image_, axis=1)

                    final_image_ = [frame for frame in final_image_]
                elif self.mode == "object-tracking":
                    box_pred = self.encoder.forward_decoder(latent, ids_restore, init_queries) # B x 256 x T x 2
                    final_image_ = self.visualize_object_tracking(video, box_pred, finetune_target, batch_idx)
                    
                    if self.cfg.inference.save_predictions:
                        data = (init_queries, box_pred, finetune_target)
                        torch.save(data, f"{self.cfg.storage_folder}/tests/eval_{self.cfg.inference.context_length}_{self.current_epoch}_{batch_idx}.pt")

                    final_image_ = np.concatenate(final_image_, axis=2)
                    
                    final_image_ = [frame for frame in final_image_]

            train_ = "train" if self.training else "val"
            global_rank = self.trainer.global_rank
            if global_rank==0 and not self.cfg.inference.testing:

                clip = ImageSequenceClip(final_image_, fps=1).resize(2)  # fps can be adjusted to your need
                clip.write_videofile(f"{self.cfg.storage_folder}/results/{train_}_{self.current_epoch}_{batch_idx}_video.mp4", codec='libx264')

            if self.cfg.inference.testing:
                title = f"eval_{self.cfg.inference.context_length}" if "eval" in self.cfg.training_type else "kubric"
                clip = ImageSequenceClip(final_image_, fps=1).resize(2)  # fps can be adjusted to your need
                clip.write_videofile(f"{self.cfg.storage_folder}/tests/{title}_{self.current_epoch}_{batch_idx}_video.mp4", codec='libx264')


        # default return values
        # check if loss_index.mean is nan
        loss_dict  = {"loss": loss}

        # track loss at each token position
        if self.mode == "point-tracking":
            extras_    = {"x_points": x_points, "occluded_pred": occlusions_pred} #{"loss_index": loss_index.mean(dim=0)}
        elif self.mode == "object-tracking":
            extras_    = {"box_pred": box_pred}
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
        metric_prefix = "test" if self.cfg.inference.testing else "val"
        if(self.cfg.promt):
            self.generate(batch, batch_idx)
            return {"loss": 0}
        if self.mode == "point-tracking":
            loss_dict, extra, logits_cls = self.step(batch, batch_idx, return_images=True)
            
            query_points = batch[1]
            target_points = batch[2]
            occluded = batch[3]
            filtered_x_points = extra["x_points"]
            filtered_x_points[query_points[:, :, 0] == -1] = 0
            query_points[query_points[:, :, 0] == -1] = 0

            self.test_mse.update(target_points.contiguous(), filtered_x_points.contiguous())

            # Update Jaccard metric
            pred_points = filtered_x_points
            pred_visible = ~(extra.get("occluded_pred", torch.zeros_like(query_points[:,:,0])) > 0.5).squeeze(-1)
            true_points = target_points
            true_visible = ~(occluded>0.5)  # Assuming batch[3] contains occlusion information
                        
            for i in range(query_points.shape[0]):
                first_invalid_index = torch.where(query_points[i, :, 0] == -1)[0]

                if len(first_invalid_index) == 0:
                    first_invalid_index = true_points.shape[1]
                else:
                    first_invalid_index = first_invalid_index[0]

                query_points_ = query_points[i:i+1, :first_invalid_index]
                query_points_[:, :, 1:] *= self.cfg.input_size

                pred_points_ = pred_points[i:i+1, :first_invalid_index] * self.cfg.input_size
                pred_visible_ = pred_visible[i:i+1, :first_invalid_index]

                true_points_ = true_points[i:i+1, :first_invalid_index] * self.cfg.input_size
                true_visible_ = true_visible[i:i+1, :first_invalid_index]

                if set((1 - (~true_visible_).detach().cpu().numpy()).flatten().tolist()) == {0}:
                    log.info(f"Skipping batch {batch_idx} for point tracking, all points are occluded")
                    continue

                metrics = compute_tapvid_metrics(query_points_.detach().cpu().numpy(), (~true_visible_).detach().cpu().numpy(), true_points_.detach().cpu().numpy(), 
                                                (~pred_visible_).detach().cpu().numpy(), pred_points_.detach().cpu().numpy())
                
                self.average_jaccard.update(metrics["average_jaccard"].mean())
                self.avg_distance.update(metrics["avg_distance"].mean())
                self.average_pts_within_thresh.update(metrics["average_pts_within_thresh"].mean())
                self.occlusion_accuracy.update(metrics["occlusion_accuracy"].mean())

                self.occ_fp.update(metrics["occ_fp"].sum())
                self.occ_tp.update(metrics["occ_tp"].sum())
                self.occ_fn.update(metrics["occ_fn"].sum())

        if self.mode == "object-tracking":
            loss_dict, extra, logits_cls = self.step(batch, batch_idx, return_images=True)
            
            query_points = batch[1]
            target_boxes = batch[2]
            pred_boxes = extra["box_pred"]

            for i in range(query_points.shape[0]):
                first_invalid_index = torch.where(query_points[i, :, 0] == -1)[0]

                if len(first_invalid_index) == 0:
                    first_invalid_index = target_boxes.shape[1]
                else:
                    first_invalid_index = first_invalid_index[0]

                pred_boxes_ = pred_boxes[i:i+1, :first_invalid_index] * self.cfg.input_size
                # pred_visible_ = pred_visible[i:i+1, :first_invalid_index]

                true_boxes_ = target_boxes[i:i+1, :first_invalid_index] * self.cfg.input_size
                # true_visible_ = true_visible[i:i+1, :first_invalid_index]

                # if set((1 - (~true_visible_).detach().cpu().numpy()).flatten().tolist()) == {0}:
                #     log.info(f"Skipping batch {batch_idx} for point tracking, all points are occluded")
                #     continue

                self.box_mse.update(true_boxes_.contiguous(), pred_boxes_.contiguous())

                # Update IoU metric
                iou = calculate_iou(true_boxes_, pred_boxes_)
                self.box_iou.update(iou)

        loss = sum([v for k,v in loss_dict.items()])

        # update and log metrics
        self.val_loss(loss.item())
        for key in loss_dict.keys():
            self.log(f"{metric_prefix}/loss/" + key, loss_dict[key].item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log Jaccard metric
        if self.mode == "point-tracking":
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
        elif self.mode == "object-tracking":
            iou_score = self.box_iou.compute()
            self.log(f"{metric_prefix}/iou", iou_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            box_mse = self.box_mse.compute()
            self.log(f"{metric_prefix}/box_mse", box_mse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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
        metric_prefix = "test" if self.cfg.inference.testing else "val"
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
        if(self.cfg.task == "finetune" or "imagenet" in self.cfg.training_type or "k400" in self.cfg.training_type or "ucf101" in self.cfg.training_type):
            for i, layer_ in enumerate(self.linear_layer):
                self.log(f"{metric_prefix}/linear_probe_acc_" + str(i), self.val_acc[i].compute().item(), prog_bar=True, sync_dist=True)
                log.info("Accuracy : " + str(self.val_acc[i].compute().item()))
                # reset the metric
                self.val_acc[i].reset()
                self.train_acc[i].reset()
            if self.mode == "point-tracking":
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

            elif self.mode == "object-tracking":
                log.info("IoU : " + str(self.box_iou.compute().item()))
                log.info("Box MSE : " + str(self.box_mse.compute().item()))

        # reset the metric
        if self.mode == "point-tracking":
            self.average_jaccard.reset()
            self.avg_distance.reset()
            self.average_pts_within_thresh.reset()
            self.occlusion_accuracy.reset()
            self.occ_fp.reset()
            self.occ_tp.reset()
            self.occ_fn.reset()
            
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
