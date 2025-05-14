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
from chewbacca.models.components.dit.dit_gaussian import TimestepEmbedder, DiT, FinalLayer

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from transformers import VideoMAEModel

from chewbacca.models.components.mae_st.models_vit import vit_large_patch16 as mae_st_vit_large_patch16

class CrossAttentionReadout(nn.Module):
    def __init__(self, embed_dim, cond_size, output_size, num_heads=8, num_enc_tokens=8, num_input_frames=24, num_fourier_features=16):
        super().__init__()
        self.num_fourier_features = num_fourier_features

        # Layer norms
        self.norm = nn.LayerNorm(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Temporal embeddings
        self.enc_temporal_embed = nn.Parameter(torch.zeros(1, num_enc_tokens, embed_dim))
        self.query_temporal_embed = nn.Parameter(torch.zeros(1, num_input_frames, embed_dim))

        # Cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Query point embedding
        fourier_dim = self.num_fourier_features * cond_size  # 3 for xyz coords, *2 for sin/cos
        self.fourier_embedding = nn.Linear(fourier_dim, 512)
        self.query_mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim)
        )

        # self.mlp = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim * 4),
        #     nn.GELU(),
        #     nn.Linear(embed_dim * 4, embed_dim)
        # )

        # Output projection
        self.output_proj = nn.Linear(embed_dim, output_size) # x,y,visibility

def fourier_encode_3d(x: torch.Tensor, num_fourier_features: int) -> torch.Tensor:
    B, N, C = x.shape
    L = num_fourier_features // 2
    freq_bands = (2.0 ** torch.arange(L, device=x.device, dtype=x.dtype)) * math.pi
    x_expanded = x.unsqueeze(-1) * freq_bands
    x_sin = torch.sin(x_expanded)
    x_cos = torch.cos(x_expanded)
    x_encoded = torch.cat([x_sin, x_cos], dim=-1)
    x_encoded = x_encoded.view(B, N, C * 2 * L)
    return x_encoded

class FinetuneMaskedAutoencoderViT(MaskedAutoencoderViT):
    def __init__(self, mode=None, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, num_tracks=1000,
                 number_of_frames=2, reuse_decoder=False, num_fourier_features=16,
                 batch_prediction=True, new_readout_mode=True, zero_t_prediction=False,
                 autoregressive=False, quantized_prediction=False, quantize_output_bins=None, dit_head=False,
                 use_dit_decoder=False, videomae=False, mae_st=False, training_type=None, freeze_encoder=True):
        # Original initialization code preserved
        nn.Module.__init__(self)
        self.number_of_frames = number_of_frames
        self.total_frames = number_of_frames
        self.videomae = videomae
        self.training_type = training_type
        self.mae_st = mae_st

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_embed.requires_grad = not freeze_encoder

        if self.videomae == "large":
            self.videomae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-large")
            for name, param in self.videomae_model.named_parameters():
                param.requires_grad = not freeze_encoder
        elif self.videomae == "base":
            self.videomae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
            for name, param in self.videomae_model.named_parameters():
                param.requires_grad = not freeze_encoder
        elif self.mae_st == "large":
            self.mae_st_model = mae_st_vit_large_patch16(num_frames=16, t_patch_size=2)
            checkpoint = torch.load("./logs/checkpoints/mae_pretrain_vit_large_k400.pth", map_location="cpu")
            checkpoint_model = checkpoint["model_state"]
            state_dict = self.mae_st_model.state_dict()
            for k in ["head.weight", "head.bias"]:
                if (
                    k in checkpoint_model
                    and checkpoint_model[k].shape != state_dict[k].shape
                ):
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            msg = self.mae_st_model.load_state_dict(checkpoint_model, strict=False)
            for name, param in self.mae_st_model.named_parameters():
                param.requires_grad = not freeze_encoder

        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches*self.total_frames, embed_dim), requires_grad=False)

            self.blocks = nn.ModuleList([
                    Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                    for i in range(depth)])

            for name, param in self.blocks.named_parameters():
                param.requires_grad = not freeze_encoder 

        self.norm = norm_layer(embed_dim)
        self.num_layers = depth
        self.embed_dim = embed_dim

        # New cross-attention method flag
        self.new_readout_mode = new_readout_mode
        self.mode = mode
        self.use_dit_decoder = use_dit_decoder

        if self.mode == "object-tracking":
            self.cond_size = 4
            self.output_size = 4
            self.input_frames = min(32, self.total_frames)

        elif self.mode == "point-tracking":
            self.cond_size = 3
            self.output_size = 3
            self.input_frames = min(24, self.total_frames)

        if self.videomae or self.mae_st:
            self.input_frames = 16

        enc_tokens = num_patches * self.input_frames // 2 if self.videomae or self.mae_st else num_patches * self.input_frames
        self.readout = CrossAttentionReadout(
            embed_dim=embed_dim,
            cond_size=self.cond_size,
            output_size=self.output_size,
            num_heads=8,
            num_enc_tokens=enc_tokens,
            num_input_frames=self.input_frames,
            num_fourier_features=num_fourier_features
        )

        self.decoder_embed_dim = decoder_embed_dim

        self.norm_pix_loss = norm_pix_loss
        self.num_decoder_queries = num_tracks
        self.batch_prediction = batch_prediction
        self.num_fourier_features = num_fourier_features
        self.zero_t_prediction = zero_t_prediction
        self.autoregressive = autoregressive
        self.quantize_output_bins = quantize_output_bins
        self.quantized_prediction = quantized_prediction
        self.dit_head = dit_head

        if self.mode == "point-tracking":
            self.params_per_out = self.input_frames * self.output_size
            self.cond_size = self.num_fourier_features * self.output_size if self.num_fourier_features > 0 else self.output_size
        elif self.mode == "object-tracking":
            self.params_per_out = self.input_frames * self.output_size
            self.cond_size = self.num_fourier_features * self.output_size if self.num_fourier_features > 0 else self.output_size
        else:
            self.input_frames = self.total_frames

        if "interpolate" in self.training_type:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
            self.pos_embed_t = nn.Parameter(torch.zeros(1, self.number_of_frames, embed_dim), requires_grad=False)  # fixed sin-cos embedding

            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.initialize_weights()

    def initialize_weights(self):
        # Original initialization code preserved
        if "interpolate" in self.training_type:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

            pos_embed_t = get_1d_sincos_pos_embed_from_grid(self.pos_embed_t.shape[-1], np.array(list(range(self.number_of_frames))))
            self.pos_embed_t.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0))
            self.pos_embed_t[:, :1, :] = 0

        else:
            if not self.videomae and not self.mae_st:
                pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), self.input_frames, cls_token=False)
                self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

                # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
                w = self.patch_embed.proj.weight.data
                torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

                # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
                torch.nn.init.normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def forward_encoder(self, x, mask_ratio):
        # Original encoder code preserved
        if self.videomae:
            x = self.videomae_model.embeddings.patch_embeddings(x.permute(0,2,1,3,4))
            x = x + self.videomae_model.embeddings.position_embeddings.type_as(x).to(device=x.device, copy=True)
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
            out = self.videomae_model.encoder(x, output_hidden_states=True)

            x = out["last_hidden_state"]
            latents = out["hidden_states"]

            # out = self.videomae_model(pixel_values=x.permute(0,2,1,3,4), output_hidden_states=True)
            # x = out["last_hidden_state"]
            # latents = out["hidden_states"]
            # ids_restore = torch.arange(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1).to(x.device)
            return x, mask, ids_restore, latents
        elif self.mae_st:
            x = self.mae_st_model.patch_embed(x)
            N, T, L, C = x.shape
            x = x.view([N, T * L, C])
            x = x + self.mae_st_model.pos_embed[:, :T*self.mae_st_model.patch_embed.num_patches]

            x, mask, ids_restore = self.random_masking(x, mask_ratio)

            latents = []
            for blk in self.mae_st_model.blocks:
                x = blk(x)
                latents.append(x)
            x = self.mae_st_model.norm(x)

            return x, mask, ids_restore, latents
        else:
            x_ = []
            for a in range(x.shape[2]):
                x_.append(self.patch_embed(x[:, :, a, :, :]))
            x_ = torch.stack(x_, dim=1)[:, :self.input_frames]
            
            if "interpolate" in self.training_type:
                # spatial pos embed
                x_ = x_ + self.pos_embed[:, None, 1:, :]
                # temporal pos embed
                x_ = x_ + self.pos_embed_t[:, :self.input_frames*self.patch_embed.num_patches, None, :]
                #
                x = rearrange(x_, 'n t c d -> n (t c) d')
            else:
                x_ = rearrange(x_, 'n t c d -> n (t c) d')
                x = x_ + self.pos_embed[:, :self.input_frames*self.patch_embed.num_patches]

            x, mask, ids_restore = self.random_masking(x, mask_ratio)

            if "interpolate" in self.training_type:
                # add cls token
                cls_token = self.cls_token + self.pos_embed[:, :1, :]
                cls_tokens = cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)

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

    def forward_decoder(self, x, ids_restore, cond=None, limit_gaussian=-1, frame_num=None, dit_training=True, dit_z=None):
        query_points = fourier_encode_3d(cond, self.num_fourier_features)

        x = self.readout.norm(x)
        x = x + self.readout.enc_temporal_embed

        queries = self.readout.fourier_embedding(query_points)
        queries = self.readout.query_mlp(queries)

        random_frame = None

        queries = queries.unsqueeze(2).repeat(1, 1, self.input_frames, 1)
        queries = queries + self.readout.query_temporal_embed.unsqueeze(1)
        batch_size = queries.shape[0]
        queries = queries.view(batch_size, -1, self.embed_dim)

        queries = self.readout.norm1(queries)
        attn_output, _ = self.readout.cross_attn(queries, x, x)
        queries = queries + attn_output

        # queries = self.readout.mlp(queries)

        outputs = self.readout.output_proj(queries)
        outputs = outputs.view(batch_size, -1, self.input_frames, self.output_size)

        if self.mode == "point-tracking":
            positions = torch.sigmoid(outputs[..., :2])
            visibility = torch.sigmoid(outputs[..., 2])
            return positions, visibility
        elif self.mode == "object-tracking":
            bboxes = torch.sigmoid(outputs[..., :4])
            return bboxes

        return outputs
        
# Rest of the code preserved as-is
# Model instantiation functions
def finetune_vit_base_patch16_dec512d8b(**kwargs):
    model = FinetuneMaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def finetune_vit_base_patch16_dec512d2b(**kwargs):
    model = FinetuneMaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
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
finetune_vit_base_patch16_small = finetune_vit_base_patch16_dec512d2b # decoder: 512 dim, 4 blocks
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
