# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.special
from einops import rearrange
# from gsplat.project_gaussians import ProjectGaussians
# from gsplat.rasterize import RasterizeGaussians
from timm.layers import DropPath, Mlp, PatchEmbed
from timm.models.vision_transformer import (Attention, Block, LayerScale,
                                            PatchEmbed)

from chewbacca.models.components.mae.pos_embed import (get_2d_sincos_pos_embed,
                                                  get_3d_sincos_pos_embed,
                                                  get_1d_sincos_pos_embed_from_grid)

from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
)

import cv2

from gsplat import rasterization
import matplotlib.pyplot as plt

from transformers import VideoMAEModel




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




class CausalAttention(Attention):
    def forward(self, x):

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                is_causal=True
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            causal_mask = torch.tril(torch.ones(N, N, device=x.device)).unsqueeze(0).unsqueeze(0)
            attn = attn.masked_fill(causal_mask == 0, float('-inf'))

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CausalBlock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the attention module with the causal version
        self.attn = CausalAttention(
            dim=self.attn.qkv.in_features,
            num_heads=self.attn.num_heads,
            qkv_bias=self.attn.qkv.bias is not None,
            attn_drop=self.attn.attn_drop.p,
            proj_drop=self.attn.proj_drop.p
        )

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, num_gaussian=1000, 
                 number_of_frames=2, scale_factor=1.0, scale_vocab=1.0,
                 deltas_reg_weight=0.0, random_frames=False, mean_deltas=False, rgb_deltas=False,
                 rgb_deltas_scale=1, upsample_gaussians=None, spawning=False, frame_zero=False,
                 pairwise_random_frames=False, videomae=False, training_type=None):
        super().__init__()
        # --------------------------------------------------------------------------
        # V1 Upgrades
        self.deltas_reg_weight = deltas_reg_weight
        self.random_frames = random_frames
        self.pairwise_random_frames = pairwise_random_frames
        self.mean_deltas = mean_deltas
        self.rgb_deltas = rgb_deltas
        self.rgb_deltas_scale = rgb_deltas_scale
        self.upsample_gaussians = upsample_gaussians
        self.spawning = spawning
        self.frame_zero = frame_zero
        self.videomae = videomae
        self.training_type = training_type
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.number_of_frames = number_of_frames if not self.random_frames else 2
        self.total_frames = number_of_frames
        
        if self.videomae:
            self.total_frames = 16

        self.input_frames = 1 if self.frame_zero else self.total_frames
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.scale_factor = scale_factor
        self.scale_vocab = int(scale_vocab)
        if self.videomae:
            self.videomae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches*self.input_frames, embed_dim), requires_grad=False)  # fixed sin-cos embedding

            self.blocks = nn.ModuleList([
                    Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                    for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.num_layers = depth
        self.embed_dim = embed_dim
        # --------------------------------------------------------------------------


        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        

        if random_frames:
            self.frame_pos_embed = nn.Parameter(torch.zeros(1, self.total_frames, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches*self.total_frames, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        if not random_frames:
            DecoderBlock = CausalBlock
        else:
            DecoderBlock = Block

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.norm_pix_loss = norm_pix_loss

        self.num_points = num_gaussian #1000
        self.tile_size = 16
        self.fov_x = torch.pi / 2.0
        self.H, self.W = img_size, img_size
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * self.fov_x)
        self.Ks = torch.tensor(
            [[
                [self.focal, 0, self.W/2],
                [0.0, self.W/self.H * self.focal, self.H/2],
                [0.0, 0.0, 1.0],
            ], ],)
        self.block = torch.tensor([self.tile_size, self.tile_size, 1])
        self.viewmat = torch.tensor(
            [[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ], ],)
        self.decoder_pos_embed_gaussian = nn.Parameter(torch.rand(1, self.num_points*self.number_of_frames, decoder_embed_dim), requires_grad=True)  # learnable parameters
        self.params_per_gaussian = 30 if self.spawning else 14

        self.linear_gaussian = nn.Linear(decoder_embed_dim, int(self.params_per_gaussian * self.scale_vocab), bias=False)

        self.linear_deltas = nn.ModuleList([nn.Linear(decoder_embed_dim + self.params_per_gaussian, \
                                            int(self.params_per_gaussian * self.scale_vocab * upsample), bias=False) for upsample in self.upsample_gaussians])

        if "interpolate" in self.training_type:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
            self.pos_embed_t = nn.Parameter(torch.zeros(1, self.number_of_frames, embed_dim), requires_grad=False)  # fixed sin-cos embedding

            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
            
        
        
        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        if "interpolate" in self.training_type:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

            pos_embed_t = get_1d_sincos_pos_embed_from_grid(self.pos_embed_t.shape[-1], np.array(list(range(self.number_of_frames))))
            self.pos_embed_t.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0))
            self.pos_embed_t[:, :1, :] = 0

        else:
            if not self.videomae:
                pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), self.input_frames, cls_token=False)
                self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

                # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
                w = self.patch_embed.proj.weight.data
                torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

                # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
                torch.nn.init.normal_(self.cls_token, std=.02)


            decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), self.total_frames, cls_token=False)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
            
            if self.random_frames:
                frame_pos_embed = get_1d_sincos_pos_embed_from_grid(self.frame_pos_embed.shape[-1], np.array(list(range(self.total_frames))))
                self.frame_pos_embed.data.copy_(torch.from_numpy(frame_pos_embed).float().unsqueeze(0))

            

        # initialize gaussian deltas to 0
        for delta in self.linear_deltas:
            torch.nn.init.constant_(delta.weight, 0)

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
        if self.videomae:
            out = self.videomae_model(pixel_values=x.permute(0,2,1,3,4), output_hidden_states=True)
            x = out["last_hidden_state"]
            latents = out["hidden_states"]
            ids_restore = torch.arange(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1).to(x.device)
            return x, None, ids_restore, latents
        else:
            if self.frame_zero:
                # only take in frame 0
                # normal behavior
                x_ = self.patch_embed(x[:, :, 0, :, :])
            else:
                x_ = []
                for a in range(x.shape[2]):
                    x_.append(self.patch_embed(x[:, :, a, :, :]))
                x_ = torch.stack(x_, dim=1)[:, :self.input_frames]
                

            # add pos embed w/o cls token
            if "interpolate" in self.training_type:
                # spatial pos embed
                x_ = x_ + self.pos_embed[:, None, 1:, :]
                # temporal pos embed
                x_ = x_ + self.pos_embed_t[:, :, None, :]
                #
                x = rearrange(x_, 'n t c d -> n (t c) d')
            else:
                x_ = rearrange(x_, 'n t c d -> n (t c) d')
                x = x_ + self.pos_embed

            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
            
            if "interpolate" in self.training_type:
                # add cls token
                cls_token = self.cls_token + self.pos_embed[:, :1, :]
                cls_tokens = cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
            
            # apply Transformer blocks
            latents = []
            for blk in self.blocks:
                x = blk(x)
                latents.append(x)
            x = self.norm(x)

            if "interpolate" in self.training_type:
                # remove cls token
                x = x[:, 1:, :]
                latents = [latent[:, 1:, :] for latent in latents]
            return x, mask, ids_restore, latents
    def decoder_step(self, x, mask_tokens, frame_token=None):
        if self.random_frames:
            assert frame_token is not None, "Need to provide frame number to use the random_frames functionality"
            x_ = torch.cat([x, frame_token, mask_tokens], dim=1)  # no cls token
        else:
            x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x_ = blk(x_)
        x_ = self.decoder_norm(x_)
        x_points = self.linear_gaussian(x_[:, -self.number_of_frames*self.num_points:])
        for linear_delta in self.linear_deltas:
            up_factor_feats = x_points.shape[1] // (self.number_of_frames*self.num_points)
            delta_feats = torch.cat([x_[:, -self.number_of_frames*self.num_points:].repeat_interleave(up_factor_feats, axis=1), x_points], axis=-1)
            x_delta = linear_delta(delta_feats)
            x_points_list = []
            for j in range(x_delta.shape[-1]//x_points.shape[-1]):
                x_points_ = x_points + x_delta[:, :, j*x_points.shape[-1]:(j+1)*x_points.shape[-1]]
                x_points_list.append(x_points_)
            x_points = torch.cat(x_points_list, dim=1)
        
        return x_points

    def forward_decoder(self, x, ids_restore, limit_gaussian=-1, frame_num=None):
        x_ = self.decoder_embed(x)
        # choose pos embed for only the encoded patches
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        ids_shuffle = ids_shuffle[:, :x_.shape[1]]  # remove cls token
        if "interpolate" in self.training_type:
            x_ = x_ 
        else:
            p_ = torch.gather(self.decoder_pos_embed[:, :, :].repeat(x_.shape[0], 1, 1), dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, self.decoder_pos_embed.shape[2]))  # unshuffle
            x_ = x_ + p_
        mask_tokens = self.mask_token.repeat(1, self.num_points*self.number_of_frames, 1) + self.decoder_pos_embed_gaussian
        mask_tokens = mask_tokens.repeat(x_.shape[0], 1, 1)
        if self.random_frames:
            if self.pairwise_random_frames:
                rf_1 = torch.randint(low=0, high=self.total_frames, size=()) if not frame_num else frame_num[0]
                rf_2 = torch.randint(low=0, high=self.total_frames, size=()) if not frame_num else frame_num[1]
                ft_1 = self.frame_pos_embed[:, rf_1:rf_1+1, :].repeat(x_.shape[0], 1, 1)
                ft_2 = self.frame_pos_embed[:, rf_2:rf_2+1, :].repeat(x_.shape[0], 1, 1)
                frame_token = torch.cat([ft_1, ft_2], dim=1)
                random_frame = (rf_1, rf_2)
            else:
                random_frame = torch.randint(low=1, high=self.total_frames, size=()) if not frame_num else frame_num
                frame_token = self.frame_pos_embed[:, random_frame:random_frame+1, :].repeat(x_.shape[0], 1, 1)
            x_points = self.decoder_step(x_, mask_tokens, frame_token=frame_token)                
        else:
            x_points = self.decoder_step(x_, mask_tokens, frame_token=None)

        if self.scale_vocab > 1:
            reshape_frames = self.number_of_frames
            x_points = x_points.reshape(x_points.shape[0], reshape_frames, self.num_points * np.prod(self.upsample_gaussians), self.scale_vocab, 14)
            x_points = x_points.permute(0, 1, 3, 2, 4)
            x_points = x_points.reshape(x_points.shape[0], reshape_frames * self.scale_vocab * self.num_points * np.prod(self.upsample_gaussians), 14)
        if self.random_frames and not frame_num:
            return x_points, random_frame
        return x_points
    
    def forward_render(self, x_points, limit_gaussian=-1, limit_gaussian_z=-1, return_gaussians=False, return_depth=False, select_range_z=-1, return_deltas=False, return_corres=False, camera_jitter=False):
        if limit_gaussian > 0:
            limit_gaussian = min(limit_gaussian, int(self.num_points * self.scale_vocab * np.prod(self.upsample_gaussians)))
        else:
            limit_gaussian = int(self.num_points * self.scale_vocab * np.prod(self.upsample_gaussians))

        device = x_points.device
        dtype = x_points.dtype

        means = 5 * (torch.tanh(x_points[:, :, :3]))
        scales = self.scale_factor * torch.sigmoid(x_points[:, :, 3:6])
        quats = torch.sigmoid(x_points[:, :, 6:10])

        au = quats[:, :, :1]
        av = quats[:, :, 1:2]
        aw = quats[:, :, 2:3]

        quats = torch.cat([
            torch.sqrt(1.0 - au) * torch.sin(2.0 * torch.pi * av),
            torch.sqrt(1.0 - au) * torch.cos(2.0 * torch.pi * av),
            torch.sqrt(au) * torch.sin(2.0 * torch.pi * aw),
            torch.sqrt(au) * torch.cos(2.0 * torch.pi * aw),
        ], dim=-1)

        rgbs = x_points[:, :, 10:13]
        opacities = x_points[:, :, 13:14]
        if camera_jitter:
            viewmat = self.viewmat.clone().to(dtype=dtype).to(device=device)
            random_r = torch.rand(())
            random_theta = torch.rand(()) * 2 * torch.pi
            random_vector = torch.tensor([random_r * torch.cos(random_theta), random_r * torch.sin(random_theta)]).to(dtype=dtype).to(device=device)
            viewmat[:, :2, 3] = random_vector
        else:
            viewmat = self.viewmat.to(dtype=dtype).to(device=device)
        
        self.Ks = self.Ks.to(dtype=dtype).to(device=device)


        imgs_ = []
        xys_ = []

        for i in range(means.shape[0]):
            images_per_video = []
            means_ = means[i].contiguous().view(-1, 3)[:limit_gaussian]
            scales_ = scales[i].contiguous().view(-1, 3)[:limit_gaussian]
            quats_ = quats[i].contiguous().view(-1, 4)[:limit_gaussian]

            means_ = means_.float()
            scales_ = scales_.float()
            quats_ = quats_.float()
            viewmat = viewmat.float()

            radii, xys, depths, conics, _ = fully_fused_projection(means_, None, quats_, scales_, viewmat, self.Ks, self.W, self.H)
            
            rgbs_ = rgbs[i][:limit_gaussian].unsqueeze(0)

            if return_depth:
                depth_ = (depths[:, :limit_gaussian] - torch.min(depths)) / (torch.max(depths) - torch.min(depths))
                # log scale
                depth_ = torch.log(depth_)
                rgbs_[:, :, 0] = depth_
                rgbs_[:, :, 1] = depth_
                rgbs_[:, :, 2] = depth_

            opacities_ = opacities[i].view(-1, 1).float()[:limit_gaussian].view(1, -1)
            if limit_gaussian_z > 0:
                # sort over depths and take the first limit_gaussian_z
                _, indices = torch.sort(depths[0])
                xys = xys[:, indices[:limit_gaussian_z]]
                depths = depths[:, indices[:limit_gaussian_z]]
                radii = radii[:, indices[:limit_gaussian_z]]
                conics = conics[:, indices[:limit_gaussian_z]]
                rgbs_ = rgbs_[:, indices[:limit_gaussian_z]]
                opacities_ = opacities_[:, indices[:limit_gaussian_z]]

            if select_range_z > 0:
                _, indices = torch.sort(depths[0])
                start_ = select_range_z[0]
                end_ = select_range_z[1]
                xys = xys[:, indices[start_:end_]]
                depths = depths[:, indices[start_:end_]]
                radii = radii[:, indices[start_:end_]]
                conics = conics[:, indices[start_:end_]]
                rgbs_ = rgbs_[:, indices[start_:end_]]
                opacities_ = opacities_[:, indices[start_:end_]]

            xys_.append([xys, opacities_])

            tile_width = math.ceil(self.W / float(self.tile_size))
            tile_height = math.ceil(self.H / float(self.tile_size))
            tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(xys, radii, depths, self.tile_size, tile_width, tile_height)
            isect_offsets = isect_offset_encode(isect_ids, viewmat.shape[0], tile_width, tile_height)
            render_colors, render_alphas = rasterize_to_pixels(xys, conics, torch.sigmoid(rgbs_), torch.sigmoid(opacities_), self.W, self.H, self.tile_size, isect_offsets, flatten_ids, absgrad=True)
            out_img = render_colors * render_alphas + (1.0 - render_alphas)
            out_img = out_img.squeeze()

            images_per_video.append(out_img)

            # render the rest of the images
            for j in range(1, means.shape[1]//limit_gaussian):
                means_delta = means[i][j*limit_gaussian:(j+1)*limit_gaussian]/10.0
                # scales_delta = scales[i][j*limit_gaussian:(j+1)*limit_gaussian]
                # quats_delta = quats[i][j*limit_gaussian:(j+1)*limit_gaussian]
                if self.mean_deltas:
                    means_ = means_ + means_delta
                # scales_ = scales_ + scales_delta
                # quats_ = quats_ + quats_delta
                radii, xys, depths, conics, compensations = fully_fused_projection(means_, None, quats_, scales_, viewmat, self.Ks, self.W, self.H)

                if self.rgb_deltas:
                    rgbs_delta = rgbs[i][j*limit_gaussian:(j+1)*limit_gaussian].unsqueeze(0)
                    if self.rgb_deltas_scale != 1:
                        rgbs_ = torch.sigmoid(rgbs[i][:limit_gaussian].unsqueeze(0)) + torch.sigmoid(rgbs_delta) * self.rgb_deltas_scale
                        rgbs_ = torch.clamp(rgbs_, 0, 1)
                    else:
                        rgbs_ = rgbs[i][:limit_gaussian].unsqueeze(0) + rgbs_delta
                        rgbs_ = torch.sigmoid(rgbs_)
                else:
                    rgbs_ = torch.sigmoid(rgbs_)
                
                # opacities_ = opacities[i][j*limit_gaussian:(j+1)*limit_gaussian]
                # opacities_ = opacities_.view(1, -1)

                if limit_gaussian_z > 0:
                    # sort over xys[2] and take the first limit_gaussian_z
                    _, indices = torch.sort(depths[0])
                    xys = xys[:, indices[:limit_gaussian_z]]
                    depths = depths[:, indices[:limit_gaussian_z]]
                    radii = radii[:, indices[:limit_gaussian_z]]
                    conics = conics[:, indices[:limit_gaussian_z]]
                    if self.rgb_deltas:
                        rgbs_ = rgbs_[:, indices[:limit_gaussian_z]]
                    # opacities_ = opacities_[:, indices[:limit_gaussian_z]]
                
                xys_.append([xys, opacities_])

                tile_width = math.ceil(self.W / float(self.tile_size))
                tile_height = math.ceil(self.H / float(self.tile_size))
                tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(xys, radii, depths, self.tile_size, tile_width, tile_height)
                isect_offsets = isect_offset_encode(isect_ids, viewmat.shape[0], tile_width, tile_height)
                render_colors, render_alphas = rasterize_to_pixels(xys, conics, rgbs_, torch.sigmoid(opacities_), self.W, self.H, self.tile_size, isect_offsets, flatten_ids, absgrad=True)
                out_img = render_colors * render_alphas + (1.0 - render_alphas)
                out_img = out_img.squeeze()
                images_per_video.append(out_img)
                
            imgs_.append(torch.stack(images_per_video, dim=0))

        imgs_ = torch.stack(imgs_, dim=0)

        if return_gaussians:
            return 0, xys_
        if return_deltas:
            mean_deltas = means.contiguous().view(means.shape[0], -1, 3)[limit_gaussian:]/10.0
            scales_delta = scales[i].contiguous().view(-1, 3)[limit_gaussian:]
            quats_delta = quats[i].contiguous().view(-1, 4)[limit_gaussian:]
            return imgs_, (mean_deltas, scales_delta, quats_delta)

        return imgs_
        
    def forward_render_all_frames(self, x, ids_restore, limit_gaussian=-1, limit_gaussian_z=-1, return_depth=False, return_corres=False, camera_jitter=False):
        init_imgs = []
        next_imgs = []
        gaussian_centers = []

        for i in range(1, self.total_frames):
            with torch.no_grad():
                if self.pairwise_random_frames:
                    random_first_frame = torch.randint(low=0, high=i, size=())
                    frame_num = (random_first_frame.item(), i)
                else:
                    frame_num = i

                x_points = self.forward_decoder(x, ids_restore, limit_gaussian=limit_gaussian, frame_num=frame_num)
                imgs_ = self.forward_render(x_points, limit_gaussian_z=limit_gaussian_z, return_depth=return_depth, return_corres=return_corres, camera_jitter=camera_jitter).cpu()

                if return_corres:
                    gaussians = self.forward_render(x_points, limit_gaussian_z=limit_gaussian_z, return_gaussians=True)[1]
                    gaussians = torch.cat(list(map(lambda x: x[0], gaussians))).clip(0, self.H).cpu()
                    gaussian_centers.append(gaussians)

                init_imgs.append(imgs_[:, 0:1])
                next_imgs.append(imgs_[:, 1:2])

        init_imgs = torch.cat(init_imgs, axis=1).mean(axis=1, keepdim=True) # 0, t
        # init_imgs = torch.cat(init_imgs, axis=1)[:, 0:1] # t, t+1
        next_imgs = torch.cat(next_imgs, axis=1)
        imgs_ = torch.cat([init_imgs, next_imgs], axis=1)

        if return_corres and limit_gaussian_z < 0:

            gaussian_centers = torch.stack(gaussian_centers)
            init_gaussians = gaussian_centers[:, ::2].mean(axis=0, keepdim=True)
            next_gaussians = gaussian_centers[:, 1::2]

            gaussians = torch.cat([init_gaussians, next_gaussians], axis=0).permute(1, 0, 2, 3)
            diffs = gaussians[:, 1:] - gaussians[:, :-1] # batch x seq_len-1 x n_gauss x 2
            dists = diffs.norm(dim=-1) # batch x seq_len-1 x n_gauss
            for batch in range(imgs_.shape[0]):
                gaussian_mapping = None
                img_hist = []
                gauss_indices = torch.logical_and(dists[batch] < 20, (dists[batch] > 2).sum(dim=0) > 15).all(dim=0).nonzero(as_tuple=True)
                # gauss_indices = ((gaussians[batch] > 25) & (gaussians[batch] < 75)).all(dim=0).nonzero(as_tuple=True)
                # gauss_indices = ((dists[batch] > 1).sum(dim=0) > 5).nonzero(as_tuple=True)
                for idx in range(imgs_.shape[1]):
                    img, mapping = self.plot_correspondences(gaussians[batch, idx, gauss_indices[0]], imgs_[batch, idx], mapping=gaussian_mapping, history=img_hist)
                    imgs_[batch, idx] = img

                    gaussian_mapping = mapping if gaussian_mapping is None and limit_gaussian_z < 0 else gaussian_mapping

                    if limit_gaussian_z < 0:
                        img_hist.append(gaussians[batch, idx, gauss_indices[0]])

        return imgs_
    
    def plot_correspondences(self, gaussians, image, mapping=None, history=[]):
        gaussians_mapping = torch.zeros((gaussians.shape[0], 3))
        img = (image.cpu().numpy() * 255).astype(np.uint8)
        max_distance = np.sqrt(self.H**2 + self.W**2)

        for idx, point in enumerate(gaussians):
            x, y = point
            distance = np.sqrt(x**2 + y**2)
            normalized_distance = min(distance / max_distance, 1.0)

            cmap = plt.get_cmap('plasma')
            if mapping is not None:
                color = mapping[idx].tolist()
            else:
                color = cmap(normalized_distance)
            
            rgb_color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

            gaussians_mapping[idx] = torch.from_numpy(np.array(rgb_color) / 255)

            cv2.circle(img, (int(x), int(y)), radius=2, color=rgb_color, thickness=-1)
            to_tuple = lambda x: (int(x[0].item()), int(x[1].item()))

            for j in range(1, len(history)):
                cv2.line(img, to_tuple(history[j-1][idx]), to_tuple(history[j][idx]), rgb_color, 1)
            if len(history) > 0:
                cv2.line(img, to_tuple(history[-1][idx]), (int(x), int(y)), rgb_color, 1)
        
        return torch.from_numpy(img).to(dtype=image.dtype) / 255, gaussians_mapping
            
    def forward_loss(self, imgs, pred, mask, additional_data=None, deltas=None, frame_num=None):
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

        for i in range(imgs_.shape[0]):
            if(torch.sum(pred[i])>0):
                if frame_num is not None:
                    loss += torch.nn.functional.mse_loss(torch.cat([imgs_[i][:, 0:1], imgs_[i][:, frame_num:frame_num+1]], axis=1), 
                                                        pred[i].permute(3, 0, 1, 2))
                else:
                    loss += torch.nn.functional.mse_loss(imgs_[i][:, :self.number_of_frames], pred[i].permute(3, 0, 1, 2))
        if self.deltas_reg_weight > 0:
            means, scales, quats = deltas
            loss += torch.linalg.norm(means) * self.deltas_reg_weight
            loss += torch.linalg.norm(scales) * self.deltas_reg_weight
            loss += torch.linalg.norm(quats) * self.deltas_reg_weight

        return loss

    def forward(self, imgs, mask_ratio=0.75, additional_data=None, all_frames=False):
        """
        imgs: [N, 3, t, H, W]
        mask_ratio: masking ratio
        """

        latent, mask, ids_restore, latents_layer = self.forward_encoder(imgs, mask_ratio)
        if self.random_frames:
            x_points, random_frame = self.forward_decoder(latent, ids_restore, all_frames=all_frames)
            pred, deltas = self.forward_render(x_points, return_deltas=True)
            loss = self.forward_loss(imgs, pred, mask, additional_data, deltas, frame_num=random_frame)
        else:
            x_points = self.forward_decoder(latent, ids_restore)
            pred, deltas = self.forward_render(x_points, return_deltas=True)  # [N, L, p*p*3]
            loss = self.forward_loss(imgs, pred, mask, additional_data, deltas)
        
        # if(self.use_gaussian):
        return loss, pred, mask, latent, latents_layer


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

def mae_vit_huge_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_huge_patch14_dec512d8b_JM(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=28, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_huge_patch16_dec512d8b_JM(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=28, embed_dim=2048, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_giga_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1664, depth=48, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_j_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1920, depth=64, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch16_dec512d8b_GAUSSIAN(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_small_patch16_dec512d8b_GAUSSIAN(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b_GAUSSIAN(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b_GAUSSIAN(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_dec256d8b_GAUSSIAN(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch16 = mae_vit_huge_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14_JM = mae_vit_huge_patch14_dec512d8b_JM  # decoder: 512 dim, 8 blocks


mae_vit_giga_patch16 = mae_vit_giga_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_j_patch16 = mae_vit_j_patch16_dec512d8b  # decoder: 512 dim, 8 blocks


mae_vit_huge_patch16_GAUSSIAN = mae_vit_huge_patch16_dec512d8b_GAUSSIAN
mae_vit_small_patch16_GAUSSIAN = mae_vit_small_patch16_dec512d8b_GAUSSIAN
mae_vit_base_patch16_GAUSSIAN = mae_vit_base_patch16_dec512d8b_GAUSSIAN
mae_vit_large_patch16_GAUSSIAN = mae_vit_large_patch16_dec512d8b_GAUSSIAN
mae_vit_large_patch16_GAUSSIAN2 = mae_vit_large_patch16_dec256d8b_GAUSSIAN
