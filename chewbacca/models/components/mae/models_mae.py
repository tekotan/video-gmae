# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.special
from einops import rearrange
from gsplat.project_gaussians import ProjectGaussians
from gsplat.rasterize import RasterizeGaussians

from timm.models.vision_transformer import PatchEmbed, Block



# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

def get_3d_sincos_pos_embed(embed_dim, grid_size, t_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(
        embed_dim_spatial, grid
    )

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(
        embed_dim_temporal, grid_t
    )

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(
        pos_embed_temporal, grid_size**2, axis=1
    )  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(
        pos_embed_spatial, t_size, axis=0
    )  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'encoder.pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['encoder.pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_extra_tokens_old = 0

        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2]-num_extra_tokens_old) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = model.pos_embed[:, :1]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens_old:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            new_pos_embed
        
            return new_pos_embed



def interpolate_pos_embed2(model, checkpoint_model):
    if 'encoder.decoder_pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['encoder.decoder_pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_extra_tokens_old = 0

        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.decoder_pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2]-num_extra_tokens_old) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = model.decoder_pos_embed[:, :1]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens_old:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            new_pos_embed
        
            return new_pos_embed



class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, num_gaussian=1000, number_of_frames=2, scale_factor=1.0, scale_vocab=1.0):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.number_of_frames = number_of_frames
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.scale_factor = scale_factor
        self.scale_vocab = int(scale_vocab)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.num_layers = depth
        self.embed_dim = embed_dim

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Gaussian specifics
        self.num_points = num_gaussian #1000
        self.BLOCK_X, self.BLOCK_Y = 16, 16
        self.fov_x = torch.pi / 2.0
        self.H, self.W = img_size, img_size
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
        self.decoder_pos_embed_gaussian = nn.Parameter(torch.rand(1, self.num_points*self.number_of_frames, decoder_embed_dim), requires_grad=True)  # learnable parameters
        self.linear_gaussian = nn.Linear(decoder_embed_dim, int(14 * self.scale_vocab), bias=False)
        # --------------------------------------------------------------------------

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        latents = []
        for blk in self.blocks:
            x = blk(x)
            latents.append(x)
        x = self.norm(x)

        return x, mask, ids_restore, latents
        
    # def forward_decoder(self, x, ids_restore):
    #     # embed tokens
    #     x = self.decoder_embed(x)

    #     # append mask tokens to sequence
    #     mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    #     x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    #     x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    #     x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

    #     # add pos embed
    #     x = x + self.decoder_pos_embed

    #     # apply Transformer blocks
    #     for blk in self.decoder_blocks:
    #         x = blk(x)
    #     x = self.decoder_norm(x)

    #     # predictor projection
    #     x = self.decoder_pred(x)

    #     # remove cls token
    #     x = x[:, 1:, :]

    #     return x

    def forward_decoder(self, x, ids_restore, limit_gaussian=-1, limit_gaussian_z=-1, return_gaussians=False, return_depth=False, select_range_z=-1):
        
        if limit_gaussian > 0:
            limit_gaussian = min(limit_gaussian, int(self.num_points * self.scale_vocab))
        else:
            limit_gaussian = int(self.num_points * self.scale_vocab)

        x_ = self.decoder_embed(x)
        # choose pos embed for only the encoded patches
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        ids_shuffle = ids_shuffle[:, :x_.shape[1]-1]  # remove cls token
        p_ = torch.gather(self.decoder_pos_embed[:, :, :].repeat(x_.shape[0], 1, 1), dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, self.decoder_pos_embed.shape[2]))  # unshuffle
        x__ = x_[:, 1:] + p_
        x_ = torch.cat([x_[:, :1], x__], dim=1)  # append cls token

        mask_tokens = self.mask_token.repeat(x_.shape[0], self.num_points*self.number_of_frames, 1) + self.decoder_pos_embed_gaussian

        x_ = torch.cat([x_, mask_tokens], dim=1)  # no cls token
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x_ = blk(x_)
        x_ = self.decoder_norm(x_)
        x_points = self.linear_gaussian(x_[:, -self.number_of_frames*self.num_points:])
        if self.scale_vocab > 1:
            x_points = x_points.reshape(x_points.shape[0], self.number_of_frames, self.num_points, self.scale_vocab, 14)
            x_points = x_points.permute(0, 1, 3, 2, 4)
            x_points = x_points.reshape(x_points.shape[0], self.number_of_frames * self.scale_vocab * self.num_points, 14)

        device = x_points.device
        dtype = x_points.dtype

        means = 2*(torch.sigmoid(x_points[:, :, :3])-0.5)
        scales = self.scale_factor * torch.sigmoid(x_points[:, :, 3:6])
        quats = torch.sigmoid(x_points[:, :, 6:10])
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

        rgbs = x_points[:, :, 10:13]
        opacities = x_points[:, :, 13:14]

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
            if return_depth:
                depth_ = (depths[:limit_gaussian] - torch.min(depths)) / (torch.max(depths) - torch.min(depths))
                # log scale
                depth_ = torch.log(depth_)
                rgbs_[:, 0] = depth_
                rgbs_[:, 1] = depth_
                rgbs_[:, 2] = depth_

            opacities_ = opacities[i].view(-1, 1).float()[:limit_gaussian]
            try:
                if limit_gaussian_z > 0:
                    # sort over depths and take the first limit_gaussian_z
                    _, indices = torch.sort(depths)
                    xys = xys[indices[:limit_gaussian_z]]
                    depths = depths[indices[:limit_gaussian_z]]
                    radii = radii[indices[:limit_gaussian_z]]
                    conics = conics[indices[:limit_gaussian_z]]
                    rgbs_ = rgbs_[indices[:limit_gaussian_z]]
                    opacities_ = opacities_[indices[:limit_gaussian_z]]
                    num_tiles_hit = num_tiles_hit[indices[:limit_gaussian_z]]

                if select_range_z > 0:
                    _, indices = torch.sort(depths)
                    start_ = select_range_z[0]
                    end_ = select_range_z[1]
                    xys = xys[indices[start_:end_]]
                    depths = depths[indices[start_:end_]]
                    radii = radii[indices[start_:end_]]
                    conics = conics[indices[start_:end_]]
                    rgbs_ = rgbs_[indices[start_:end_]]
                    opacities_ = opacities_[indices[start_:end_]]
                    num_tiles_hit = num_tiles_hit[indices[start_:end_]]

                xys_.append([xys, opacities_, scales_, means_, depths])
                out_img = RasterizeGaussians.apply(xys, depths, radii, conics, num_tiles_hit, torch.sigmoid(rgbs_), torch.sigmoid(opacities_), self.H, self.W,)

            except:
                out_img = torch.zeros(3, self.H, self.W).to(dtype=dtype).to(device=device)
            # out_img = out_img.half()
            # imgs_.append(out_img)
            images_per_video.append(out_img)

            # render the rest of the images
            # for j in range(1, means.shape[1]//self.num_points):
            #     means_delta = means[i].contiguous().view(-1, 3)[j*self.num_points:(j+1)*self.num_points][:limit_gaussian]/10.0
            #     means_delta[:, -1] *= 0
            #     # TODO: only do x,y
            #     # TODO: l1,l2 regualization to deltas
            #     # TODO: add d-ssim loss
            #     # TODO: get color avg from all frames.
            #     # TODO: add a contraint for color, opacity and scale

            #     # TODO: gaussian tokizer, 3d masking and next depth prediction
            #     # TODO: masking
            #     # scales_delta = scales[i].contiguous().view(-1, 3)[j*self.num_points:(j+1)*self.num_points][:limit_gaussian]
            #     # quats_delta = quats[i].contiguous().view(-1, 4)[j*self.num_points:(j+1)*self.num_points][:limit_gaussian]
            #     means_ = means_ + means_delta
            #     # scales_ = scales_.float()
            #     # quats_ = quats_.float()
            #     xys, depths, radii, conics, num_tiles_hit, cov3d = ProjectGaussians.apply(means_, scales_, 1, quats_, self.viewmat, self.viewmat, self.focal, self.focal, self.W / 2, self.H / 2, self.H, self.W, self.tile_bounds,)
            #     # rgbs_ = rgbs[i].contiguous().view(-1, 3).float()
            #     # opacities_ = opacities[i].contiguous().view(-1, 1).float()
            #     try:
            #         if limit_gaussian_z > 0:
            #             # sort over xys[2] and take the first limit_gaussian_z
            #             _, indices = torch.sort(depths)
            #             xys = xys[indices[:limit_gaussian_z]]
            #             depths = depths[indices[:limit_gaussian_z]]
            #             radii = radii[indices[:limit_gaussian_z]]
            #             conics = conics[indices[:limit_gaussian_z]]
            #             # rgbs_ = rgbs_[indices[:limit_gaussian_z]]
            #             # opacities_ = opacities_[indices[:limit_gaussian_z]]
            #             num_tiles_hit = num_tiles_hit[indices[:limit_gaussian_z]]

            #         out_img = RasterizeGaussians.apply(xys, depths, radii, conics, num_tiles_hit, torch.sigmoid(rgbs_), torch.sigmoid(opacities_), self.H, self.W,)
            #     except:
            #         out_img = torch.zeros(3, self.H, self.W).to(dtype=dtype).to(device=device)
            #     # out_img = out_img.half()
            #     # imgs_.append(out_img)
            #     images_per_video.append(out_img)
            imgs_.append(torch.stack(images_per_video, dim=0))

        imgs_ = torch.stack(imgs_, dim=0)

        if return_gaussians:
            return 0, xys_

        return imgs_


    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore, latents_layer = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
