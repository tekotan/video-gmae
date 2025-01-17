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
from chewbacca.models.components.mae.models_mae_video import MaskedAutoencoderViT

import cv2
import matplotlib.pyplot as plt

def fourier_encode_3d(x: torch.Tensor, num_fourier_features: int) -> torch.Tensor:
    """
    Convert a [B, N, 3] tensor into a [B, N, 3 * num_fourier_features] tensor
    by applying sinusoidal Fourier features to each of the 3 input dimensions.
    
    Args:
        x (torch.Tensor): Input tensor of shape [B, N, 3].
        num_fourier_features (int): Total number of Fourier features per
                                    input dimension. For standard sin/cos,
                                    this should be an even number, where half
                                    come from sin() and half from cos().
    
    Returns:
        torch.Tensor: Output tensor of shape [B, N, 3 * num_fourier_features].
    """
    B, N, C = x.shape
    assert C == 3, "Input must have exactly 3 channels in the last dimension."

    L = num_fourier_features // 2
    freq_bands = (2.0 ** torch.arange(L, device=x.device, dtype=x.dtype)) * math.pi
    
    x_expanded = x.unsqueeze(-1) * freq_bands  # shape: [B, N, 3, L]

    x_sin = torch.sin(x_expanded)  # shape: [B, N, 3, L]
    x_cos = torch.cos(x_expanded)  # shape: [B, N, 3, L]

    x_encoded = torch.cat([x_sin, x_cos], dim=-1)  # shape: [B, N, 3, 2L]

    x_encoded = x_encoded.view(B, N, 3 * 2 * L)  # shape: [B, N, 3 * num_fourier_features]

    return x_encoded

class FinetuneMaskedAutoencoderViT(MaskedAutoencoderViT):
    def __init__(self, mode=None, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, num_gaussian=1000, 
                 number_of_frames=2, reuse_decoder=False, num_fourier_features=0):
        nn.Module.__init__(self)
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.number_of_frames = number_of_frames
        self.total_frames = number_of_frames
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches*self.total_frames, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(depth)])

        self.blocks.requires_grad = False
        self.patch_embed.requires_grad = False

        self.norm = norm_layer(embed_dim)
        self.num_layers = depth
        self.embed_dim = embed_dim
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.reuse_decoder = reuse_decoder
        if not self.reuse_decoder:
            self.decoder_embed_fine = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.mask_token_fine = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches*self.total_frames, decoder_embed_dim), requires_grad=False)
            self.decoder_blocks_fine = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])

            self.decoder_norm_fine = norm_layer(decoder_embed_dim)
        else:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches*self.total_frames, decoder_embed_dim), requires_grad=False)
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])

            self.decoder_norm = norm_layer(decoder_embed_dim)
        self.norm_pix_loss = norm_pix_loss

        self.num_points = num_gaussian
        # --------------------------------------------------------------------------
        # Fine Tune specifics
        self.num_fourier_features = num_fourier_features
        self.linear_cond = None
        self.mode = mode
        if self.mode == "tracking":
            self.input_frames = 24
            self.params_per_out = self.input_frames * 3
            self.cond_size = self.num_fourier_features * 3 if self.num_fourier_features > 0 else 3
            self.linear_cond = nn.Linear(self.cond_size, decoder_embed_dim)
        else:
            self.input_frames = self.total_frames

        self.number_of_frames = 1
        self.decoder_pos_embed_cond = nn.Parameter(torch.rand(1, self.num_points, decoder_embed_dim), requires_grad=True)  # learnable parameters
        self.linear_out = nn.Linear(decoder_embed_dim, self.params_per_out, bias=True)
        
        
    
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), self.total_frames, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), self.total_frames, cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        if not self.reuse_decoder:
            torch.nn.init.normal_(self.mask_token_fine, std=.02)
        else:
            torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x_ = []
        for a in range(x.shape[2]):
            x_.append(self.patch_embed(x[:, :, a, :, :]))
        x_ = torch.stack(x_, dim=1)[:, :self.input_frames]
        x_ = rearrange(x_, 'n t c d -> n (t c) d')

        # add pos embed w/o cls token
        x = x_ + self.pos_embed[:, :self.input_frames*self.patch_embed.num_patches]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        

        # apply Transformer blocks
        latents = []
        for blk in self.blocks:
            x = blk(x)
            latents.append(x)
        x = self.norm(x)

        return x, mask, ids_restore, latents

    def decoder_step(self, x, mask_tokens):
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        # apply Transformer blocks
        if not self.reuse_decoder:
            for blk in self.decoder_blocks_fine:
                x_ = blk(x_)
            x_ = self.decoder_norm_fine(x_)
        else:
            for blk in self.decoder_blocks:
                x_ = blk(x_)
            x_ = self.decoder_norm(x_)

        x_points = self.linear_out(x_[:, -self.num_points:])        
        return x_points

    def forward_decoder(self, x, ids_restore, cond):
        if not self.reuse_decoder:
            x_ = self.decoder_embed_fine(x)
            mask_tokens = self.mask_token_fine.repeat(x_.shape[0], self.num_points, 1)
        else:
            x_ = self.decoder_embed(x)
            mask_tokens = self.mask_token.repeat(x_.shape[0], self.num_points, 1)

        # choose pos embed for only the encoded patches
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        ids_shuffle = ids_shuffle[:, :x_.shape[1]]  # remove cls token
        p_ = torch.gather(self.decoder_pos_embed[:, :, :].repeat(x_.shape[0], 1, 1), dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, self.decoder_pos_embed.shape[2]))  # unshuffle
        x_ = x_ + p_
        if self.num_fourier_features > 0:
            cond = fourier_encode_3d(cond, self.num_fourier_features)
        if self.linear_cond is not None:
            decoder_cond_embedding = self.linear_cond(cond)
            mask_tokens += decoder_cond_embedding
        
        mask_tokens += self.decoder_pos_embed_cond
        
        x_points = self.decoder_step(x_, mask_tokens) # B x 256 x 48

        x_points = x_points.reshape(x_points.shape[0], self.num_points, self.input_frames, -1) # B x 256 x 48 -> B x 256 x 24 x 2
        if self.mode == "tracking":
            occluded = torch.nn.functional.sigmoid(x_points[:, :, :, 2])
            x_points = x_points[:, :, :, :2]
            return x_points, occluded

        return x_points



def finetune_vit_base_patch16_dec512d8b(**kwargs):
    model = FinetuneMaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def finetune_vit_large_patch16_dec512d8b(**kwargs):
    model = FinetuneMaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def finetune_vit_huge_patch14_dec512d8b(**kwargs):
    model = FinetuneMaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def finetune_vit_huge_patch16_dec512d8b(**kwargs):
    model = FinetuneMaskedAutoencoderViT(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def finetune_vit_huge_patch14_dec512d8b_JM(**kwargs):
    model = FinetuneMaskedAutoencoderViT(
        patch_size=28, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def finetune_vit_huge_patch16_dec512d8b_JM(**kwargs):
    model = FinetuneMaskedAutoencoderViT(
        patch_size=28, embed_dim=2048, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def finetune_vit_giga_patch16_dec512d8b(**kwargs):
    model = FinetuneMaskedAutoencoderViT(
        patch_size=16, embed_dim=1664, depth=48, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def finetune_vit_j_patch16_dec512d8b(**kwargs):
    model = FinetuneMaskedAutoencoderViT(
        patch_size=16, embed_dim=1920, depth=64, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def finetune_vit_huge_patch16_dec512d8b_GAUSSIAN(**kwargs):
    model = FinetuneMaskedAutoencoderViT(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def finetune_vit_small_patch16_dec512d8b_GAUSSIAN(**kwargs):
    model = FinetuneMaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def finetune_vit_base_patch16_dec512d8b_GAUSSIAN(**kwargs):
    model = FinetuneMaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def finetune_vit_large_patch16_dec512d8b_GAUSSIAN(**kwargs):
    model = FinetuneMaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def finetune_vit_large_patch16_dec256d8b_GAUSSIAN(**kwargs):
    model = FinetuneMaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
finetune_vit_base_patch16 = finetune_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
finetune_vit_large_patch16 = finetune_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
finetune_vit_huge_patch14 = finetune_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
finetune_vit_huge_patch16 = finetune_vit_huge_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
finetune_vit_huge_patch14_JM = finetune_vit_huge_patch14_dec512d8b_JM  # decoder: 512 dim, 8 blocks


finetune_vit_giga_patch16 = finetune_vit_giga_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
finetune_vit_j_patch16 = finetune_vit_j_patch16_dec512d8b  # decoder: 512 dim, 8 blocks


finetune_vit_huge_patch16_GAUSSIAN = finetune_vit_huge_patch16_dec512d8b_GAUSSIAN
finetune_vit_small_patch16_GAUSSIAN = finetune_vit_small_patch16_dec512d8b_GAUSSIAN
finetune_vit_base_patch16_GAUSSIAN = finetune_vit_base_patch16_dec512d8b_GAUSSIAN
finetune_vit_large_patch16_GAUSSIAN = finetune_vit_large_patch16_dec512d8b_GAUSSIAN
finetune_vit_large_patch16_GAUSSIAN2 = finetune_vit_large_patch16_dec256d8b_GAUSSIAN
