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

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class DITHead(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(out_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(out_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * out_channels, bias=True)
        )

    def forward(self, x, t, c):
        t = t + c
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiTConditional(DiT):
    def __init__(self, denoise_feats, enc_feats, output_frames, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_frames = output_frames
        if denoise_feats == 3:
            self.x_embedder = TimestepEmbedder(self.hidden_size//2, frequency_embedding_size=256)
            self.y_embedder = TimestepEmbedder(self.hidden_size//2, frequency_embedding_size=256)
        elif denoise_feats == 4:
            self.x_embedder = TimestepEmbedder(self.hidden_size//4, frequency_embedding_size=256)
            self.y_embedder = TimestepEmbedder(self.hidden_size//4, frequency_embedding_size=256)

        self.input_mlp = nn.Sequential(
            nn.Linear(denoise_feats, self.hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
        )
        self.enc_tokens_proj = nn.Linear(enc_feats, self.hidden_size, bias=True)
        self.out_channels = denoise_feats if not self.learn_sigma else denoise_feats * 2
        self.final_layer = FinalLayer(self.hidden_size, 1, self.out_channels)

    def forward(self, enc_tokens, x, t, c):
        """
        enc_tokens: (N, enc_T, enc_D)
        x: (N, T, denoise_D)
        t: (N)
        c: (N, 2) for point_track, (N, 4) for object_track
        """
        t = self.t_embedder(t)                   # (N, D)
        if c.shape[1] == 2:
            x_ = self.x_embedder(c[:, 0])                   # (N, D)
            y_ = self.y_embedder(c[:, 1])                   # (N, D)
            c = torch.cat([x_, y_], dim=-1)
        elif c.shape[1] == 4:
            x_ = self.x_embedder(c[:, 0])                   # (N, D)
            y_ = self.y_embedder(c[:, 1])                   # (N, D)
            w_ = self.x_embedder(c[:, 2])                   # (N, D)
            h_ = self.y_embedder(c[:, 3])                   # (N, D)
            c = torch.cat([x_, y_, w_, h_], dim=-1)
        c = t + c                    # (N, D)

        enc_tokens = self.enc_tokens_proj(enc_tokens)  # (N, T, D)
        x = self.input_mlp(x.float())                          # (N, T, D)

        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)               # (N, enc_T + T, denoise_D)
        x = x[:, -self.output_frames:]           # (N, T, denoise_D) or (N, T, denoise_D*2)
        if self.learn_sigma:
            x = x.view(x.shape[0], x.shape[1]*2, x.shape[2]//2) # (N, T*2, denoise_D)
        return x

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

class CrossAttentionReadout(nn.Module):
    def __init__(self, embed_dim, cond_size, output_size, num_heads=8, num_enc_tokens=8, num_input_frames=24, num_fourier_features=16):
        super().__init__()
        self.num_fourier_features = num_fourier_features

        # Layer norms
        self.norm = nn.LayerNorm(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

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

        # Residual MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

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
                 use_dit_decoder=False, videomae=False):
        # Original initialization code preserved
        nn.Module.__init__(self)
        self.number_of_frames = number_of_frames
        self.total_frames = number_of_frames
        self.videomae = videomae
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_embed.requires_grad = False

        if self.videomae:
            self.videomae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-large")
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches*self.total_frames, embed_dim), requires_grad=False)

            self.blocks = nn.ModuleList([
                    Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                    for i in range(depth)])
                    
            self.blocks.requires_grad = False

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
            self.input_frames = 32

        elif self.mode == "point-tracking":
            self.cond_size = 3
            self.output_size = 3
            self.input_frames = 24
        
        if self.videomae:
            self.input_frames = 16

        if new_readout_mode:
            enc_tokens = 49 * self.input_frames * 2 if self.videomae else 64 * 32 * self.cond_size #49 * self.input_frames
            self.readout = CrossAttentionReadout(
                embed_dim=embed_dim,
                cond_size=self.cond_size,
                output_size=self.output_size,
                num_heads=8,
                num_enc_tokens=enc_tokens,
                num_input_frames=self.input_frames,
                num_fourier_features=num_fourier_features
            )
        elif self.use_dit_decoder:
            self.dit_decoder = DiTConditional(denoise_feats=self.output_size, enc_feats=embed_dim, output_frames=self.input_frames, \
                                                depth=12, hidden_size=384, patch_size=2, num_heads=6, learn_sigma=False)
        else:
            # Original decoder components preserved
            self.reuse_decoder = reuse_decoder
            if autoregressive:
                assert batch_prediction == False
                DecoderBlock = CausalBlock
            else:
                DecoderBlock = Block

            if not self.reuse_decoder:
                self.decoder_embed_fine = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
                self.mask_token_fine = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
                self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches*self.total_frames, decoder_embed_dim), requires_grad=False)
                self.decoder_blocks_fine = nn.ModuleList([
                    DecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                    for i in range(decoder_depth)])
                self.decoder_norm_fine = norm_layer(decoder_embed_dim)
            else:
                self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
                self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
                self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches*self.total_frames, decoder_embed_dim), requires_grad=False)
                self.decoder_blocks = nn.ModuleList([
                    DecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                    for i in range(decoder_depth)])
                self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_embed_dim = decoder_embed_dim

        self.norm_pix_loss = norm_pix_loss
        self.num_decoder_queries = num_tracks
        self.batch_prediction = batch_prediction
        self.num_fourier_features = num_fourier_features
        self.linear_cond = None
        self.zero_t_prediction = zero_t_prediction
        self.autoregressive = autoregressive
        self.quantize_output_bins = quantize_output_bins
        self.quantized_prediction = quantized_prediction
        self.dit_head = dit_head
        

        if self.mode == "point-tracking":
            if self.batch_prediction:
                self.params_per_out = self.input_frames * self.output_size
                self.cond_size = self.num_fourier_features * self.output_size if self.num_fourier_features > 0 else self.output_size
                self.linear_cond = nn.Linear(self.cond_size, decoder_embed_dim)
            elif self.quantized_prediction:
                self.params_per_out = (self.output_size - 1) * self.quantize_output_bins + 1
                self.num_decoder_queries = self.input_frames
                self.cond_size = self.num_fourier_features * self.output_size if self.num_fourier_features > 0 else self.output_size
            else:
                self.params_per_out = self.output_size
                self.num_decoder_queries = self.input_frames
                self.cond_size = self.num_fourier_features * self.output_size if self.num_fourier_features > 0 else self.output_size
        elif self.mode == "object-tracking":
            if self.batch_prediction:
                self.params_per_out = self.input_frames * self.output_size
                self.cond_size = self.num_fourier_features * self.output_size if self.num_fourier_features > 0 else self.output_size
                self.linear_cond = nn.Linear(self.cond_size, decoder_embed_dim)
            elif self.quantized_prediction:
                self.params_per_out = self.output_size * self.quantize_output_bins
                self.num_decoder_queries = self.input_frames
                self.cond_size = self.num_fourier_features * self.output_size if self.num_fourier_features > 0 else self.output_size
            else:
                self.params_per_out = self.output_size
                self.num_decoder_queries = self.input_frames
                self.cond_size = self.num_fourier_features * self.output_size if self.num_fourier_features > 0 else self.output_size
        else:
            self.input_frames = self.total_frames

        if self.zero_t_prediction:
            if self.batch_prediction:
                self.params_per_out = int(self.params_per_out / self.input_frames)
                self.num_decoder_queries = num_tracks * 2
            else:
                self.num_decoder_queries = 2

            self.frame_pos_embed = nn.Parameter(torch.zeros(1, self.total_frames, decoder_embed_dim))

        self.number_of_frames = 1
        self.decoder_pos_embed_cond = nn.Parameter(torch.rand(1, self.num_decoder_queries, decoder_embed_dim), requires_grad=True)
        if self.dit_head:
            if self.mode == "point-tracking":
                self.linear_out = nn.Linear(decoder_embed_dim, 1, bias=True)
                self.dit_out = DITHead(decoder_embed_dim, self.params_per_out-1)
            else:
                self.dit_out = DITHead(decoder_embed_dim, self.params_per_out)
            self.timestep_embed = TimestepEmbedder(decoder_embed_dim, frequency_embedding_size=256)
        else:
            self.linear_out = nn.Linear(decoder_embed_dim, self.params_per_out, bias=True)

    def initialize_weights(self):
        # Original initialization code preserved
        if not self.videomae:
            pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), self.total_frames, cls_token=False)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            w = self.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if not self.new_readout_mode:
            decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), self.total_frames, cls_token=False)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        
        torch.nn.init.normal_(self.cls_token, std=.02)
        if not self.new_readout_mode:
            if not self.reuse_decoder:
                torch.nn.init.normal_(self.mask_token_fine, std=.02)
            else:
                torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def forward_encoder(self, x, mask_ratio):
        # Original encoder code preserved
        if self.videomae:
            out = self.videomae_model(pixel_values=x.permute(0,2,1,3,4), output_hidden_states=True)
            x = out["last_hidden_state"]
            latents = out["hidden_states"]
            ids_restore = torch.arange(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1).to(x.device)
            return x, None, ids_restore, latents
        else:
            x_ = []
            for a in range(x.shape[2]):
                x_.append(self.patch_embed(x[:, :, a, :, :]))
            x_ = torch.stack(x_, dim=1)[:, :self.input_frames]
            x_ = rearrange(x_, 'n t c d -> n (t c) d')

            x = x_ + self.pos_embed[:, :self.input_frames*self.patch_embed.num_patches]

            x, mask, ids_restore = self.random_masking(x, mask_ratio)

            latents = []
            for blk in self.blocks:
                x = blk(x)
                latents.append(x)
            x = self.norm(x)

            return x, mask, ids_restore, latents
    def dit_step(self, x, dit_z, dit_training=True):
        B = x.shape[0]
        T = x.shape[1]
        if dit_training:
            z0, z1 = dit_z

            t = torch.rand(B, device=x.device).repeat_interleave(T)
            t_emb = self.timestep_embed(t)
            zt = t.unsqueeze(-1) * z1 + (1 - t.unsqueeze(-1)) * z0
            out_z = self.dit_out(zt, t_emb, x.reshape(B * T, self.decoder_embed_dim))
            return out_z.view(B, T, -1)
        else:
            z0, _ = dit_z
            steps = 10
            for i in range(steps):
                t = torch.rand(B*T, device=x.device) * 0 + i / steps
                t_emb = self.timestep_embed(t)

                out_z = self.dit_out(z0, t_emb, x.reshape(B * T, self.decoder_embed_dim))
                z0 = z0 + out_z * (1.0 / steps)
            out_z = z0.view(B, T, -1)

        return out_z
    def decoder_step(self, x, mask_tokens, frame_token=None, dit_training=True, dit_z=None):
        if self.new_readout_mode:
            # Process through cross-attention readout
            x_ = self.readout(x)
            return x_
        else:
            if frame_token is not None:
                x_ = torch.cat([x, frame_token, mask_tokens], dim=1)
            else:
                # Original decoder step preserved
                x_ = torch.cat([x, mask_tokens], dim=1)

            if not self.reuse_decoder:
                for blk in self.decoder_blocks_fine:
                    x_ = blk(x_)
                x_ = self.decoder_norm_fine(x_)
            else:
                for blk in self.decoder_blocks:
                    x_ = blk(x_)
                x_ = self.decoder_norm(x_)
            if self.dit_head:
                linear_in = x_[:, -mask_tokens.shape[1]:]
                x_points = self.dit_step(linear_in, dit_training=dit_training, dit_z=dit_z)
                
                if self.mode == "point-tracking":
                    occlusions = []
                    for idx in range(linear_in.shape[1]):
                        occlusions.append(self.linear_out(linear_in[:, idx]))
                    occlusions = torch.stack(occlusions, dim=1)
                    x_points = torch.cat([x_points, occlusions], dim=-1)
            else:
                linear_in = x_[:, -mask_tokens.shape[1]:]
                x_points = []
                for idx in range(linear_in.shape[1]):
                    x_points.append(self.linear_out(linear_in[:, idx]))
                x_points = torch.stack(x_points, dim=1)
            return x_points

    def forward_decoder(self, x, ids_restore, cond=None, limit_gaussian=-1, frame_num=None, dit_training=True, dit_z=None):
        if self.new_readout_mode:
            # Fourier encode the query points
            query_points = fourier_encode_3d(cond, self.num_fourier_features)

            # Process through cross-attention readout
            x = self.readout.norm(x)
            x = x + self.readout.enc_temporal_embed

            # Embed query points
            queries = self.readout.fourier_embedding(query_points)
            queries = self.readout.query_mlp(queries)

            if self.zero_t_prediction:
                random_frame = torch.randint(low=1, high=self.input_frames, size=()) if not frame_num else frame_num
                queries_0 = queries.unsqueeze(2) + self.readout.query_temporal_embed[:, 0:1].unsqueeze(1)
                queries_t = queries.unsqueeze(2) + self.readout.query_temporal_embed[:, random_frame:random_frame+1].unsqueeze(1)
                queries_2frames = torch.cat([queries_0, queries_t], dim=2)
                B, N, F, D = queries_2frames.shape
                queries_2frames = queries_2frames.view(B, N*F, D)

                queries_2frames = self.readout.norm1(queries_2frames)
                attn_output, _ = self.readout.cross_attn(queries_2frames, x, x)
                queries_2frames = queries_2frames + attn_output

                # MLP
                queries_2frames = queries_2frames + self.readout.mlp(
                    self.readout.norm2(queries_2frames)
                )

                # Output projection
                outputs = self.readout.output_proj(queries_2frames)
                outputs = outputs.view(B, N, F, self.output_size)

                if self.mode == "point-tracking":
                    positions = torch.sigmoid(outputs[..., :2])
                    visibility = torch.sigmoid(outputs[..., 2])
                    return positions, visibility, random_frame
                elif self.mode == "object-tracking":
                    bboxes = torch.sigmoid(outputs[..., :4])
                    return bboxes, random_frame
                return outputs
            else:
                random_frame = None
                # Replicate queries with temporal embeddings
                queries = queries.unsqueeze(2).repeat(1, 1, self.input_frames, 1)
                queries = queries + self.readout.query_temporal_embed.unsqueeze(1)
                batch_size = queries.shape[0]
                queries = queries.view(batch_size, -1, self.embed_dim)

                # Cross attention
                queries = self.readout.norm1(queries)
                attn_output, _ = self.readout.cross_attn(queries, x, x)
                queries = queries + attn_output

                # MLP
                queries = queries + self.readout.mlp(self.readout.norm2(queries))

                # Output projection
                outputs = self.readout.output_proj(queries)
                outputs = outputs.view(batch_size, -1, self.input_frames, self.output_size)

                if self.mode == "point-tracking":
                    positions = torch.sigmoid(outputs[..., :2])
                    visibility = torch.sigmoid(outputs[..., 2])
                    return positions, visibility, random_frame
                elif self.mode == "object-tracking":
                    bboxes = torch.sigmoid(outputs[..., :4])
                    return bboxes, random_frame

                return outputs, random_frame
        elif self.use_dit_decoder:
            pass
        else:
            if self.autoregressive:
                num_mask_repeat = cond.shape[1]
            else:
                num_mask_repeat = self.num_decoder_queries
            if not self.reuse_decoder:
                x_ = self.decoder_embed_fine(x)
                mask_tokens = self.mask_token_fine.repeat(x_.shape[0], num_mask_repeat, 1)
            else:
                x_ = self.decoder_embed(x)
                mask_tokens = self.mask_token.repeat(x_.shape[0], num_mask_repeat, 1)
                
            ids_shuffle = torch.argsort(ids_restore, dim=1)
            ids_shuffle = ids_shuffle[:, :x_.shape[1]]
            p_ = torch.gather(self.decoder_pos_embed[:, :, :].repeat(x_.shape[0], 1, 1), dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, self.decoder_pos_embed.shape[2]))
            x_ = x_ + p_

            if self.batch_prediction:
                if self.num_fourier_features > 0:
                    cond = fourier_encode_3d(cond, self.num_fourier_features)
                if self.linear_cond is not None:
                    decoder_cond_embedding = self.linear_cond(cond)
                    mask_tokens += decoder_cond_embedding
            else:
                frames = torch.arange(num_mask_repeat, device=x_.device).repeat(x_.shape[0], 1).unsqueeze(-1)
                if self.autoregressive:
                    t_cond = torch.arange(cond.shape[1], device=x_.device)
                else:
                    t_cond = cond[:, 0, 0].int()

                num_feats = max(self.num_fourier_features, 1)
                if self.num_fourier_features > 0:
                    frames = fourier_encode_3d(frames, self.num_fourier_features)
                    cond = fourier_encode_3d(cond, self.num_fourier_features)
                    
                mask_tokens[:, :, :num_feats] = frames
                mask_tokens[:, t_cond, :self.cond_size] = cond.float()

            mask_tokens += self.decoder_pos_embed_cond[:, :num_mask_repeat, :]
            if self.zero_t_prediction:
                random_frame = torch.randint(low=1, high=self.input_frames, size=()) if not frame_num else frame_num
                frame_token = self.frame_pos_embed[:, random_frame:random_frame+1, :].repeat(x_.shape[0], 1, 1)
                mask_tokens = torch.cat([mask_tokens[:, 0:1], mask_tokens[:, random_frame:random_frame+1]], dim=1)
            else:
                random_frame = 0
                frame_token = None

            x_points = self.decoder_step(x_, mask_tokens, frame_token=frame_token, dit_training=dit_training, dit_z=dit_z)
            if self.batch_prediction:
                if self.zero_t_prediction:
                    x_points = x_points.reshape(x_points.shape[0], num_mask_repeat, 2, -1)
                else:
                    x_points = x_points.reshape(x_points.shape[0], num_mask_repeat, self.input_frames, -1)
            else:
                x_points = x_points.reshape(x_points.shape[0], 1, num_mask_repeat, -1)

            if self.mode == "point-tracking":
                if self.quantized_prediction:
                    occluded = torch.nn.functional.sigmoid(x_points[:, :, :, -1])
                    x_points_x = x_points[:, :, :, :self.quantize_output_bins]
                    x_points_y = x_points[:, :, :, self.quantize_output_bins:self.quantize_output_bins*2]
                    x_points = torch.stack([x_points_x, x_points_y], dim=-2)
                else:
                    occluded = torch.nn.functional.sigmoid(x_points[:, :, :, 2])
                    x_points = x_points[:, :, :, :2]
                return x_points, occluded, random_frame
            elif self.mode == "object-tracking":
                if self.quantized_prediction:
                    bboxes_xmin = x_points[:, :, :, :self.quantize_output_bins]
                    bboxes_ymin = x_points[:, :, :, self.quantize_output_bins:self.quantize_output_bins*2]
                    bboxes_xmax = x_points[:, :, :, self.quantize_output_bins*2:self.quantize_output_bins*3]
                    bboxes_ymax = x_points[:, :, :, self.quantize_output_bins*3:self.quantize_output_bins*4]
                    bboxes = torch.stack([bboxes_xmin, bboxes_ymin, bboxes_xmax, bboxes_ymax], dim=-2)
                else:
                    bboxes = torch.sigmoid(x_points[..., :4])
                return bboxes, random_frame

            return x_points, random_frame
    def forward_decoder_zero_t(self, x, ids_restore, cond=None, limit_gaussian=-1):
        preds = []
        visibility = []
        for i in range(1, self.input_frames):
            pred, vis, _ = self.forward_decoder(x, ids_restore, cond, limit_gaussian, frame_num=i)
            if len(preds) == 0:
                preds.append(pred)
                visibility.append(vis)
            else:
                preds.append(pred[:, :, 1:])
                visibility.append(vis[:, :, 1:])
        return torch.cat(preds, dim=2), torch.cat(visibility, dim=2)
    def forward_decoder_autoreg(self, x, ids_restore, cond=None, limit_gaussian=-1, return_logits=False, dit_training=True, dit_z=None, context_length=1):
        if self.mode == "point-tracking":
            if return_logits:
                visibility = []
                preds = []
            else:
                visibility = [(cond[:, :, 0:1] > 0.5).float().unsqueeze(1)]
                preds = [cond[:, :, 1:].unsqueeze(1)]
        elif self.mode == "object-tracking":
            if return_logits:
                preds = []
            else:
                preds = [cond.unsqueeze(1)]

        cond_ = cond.clone()
        for i in range(self.input_frames-context_length):
            if self.dit_head:
                if self.mode == "point-tracking":
                    z0 = torch.randn(cond_.shape[0] * cond_.shape[1], cond_.shape[2]-1, device=cond_.device)
                else:
                    z0 = torch.randn(cond_.shape[0] * cond_.shape[1], cond_.shape[2], device=cond_.device)
            else:
                z0 = None
            dit_z = (z0, None)

            if self.mode == "point-tracking":
                pred, vis, _ = self.forward_decoder(x, ids_restore, cond_, limit_gaussian, frame_num=i, dit_training=False, dit_z=dit_z)
                visibility.append(vis[:, :, -1:].unsqueeze(-1))
            elif self.mode == "object-tracking":
                pred, _ = self.forward_decoder(x, ids_restore, cond_, limit_gaussian, frame_num=i, dit_training=False, dit_z=dit_z)
            if self.quantized_prediction and not return_logits:
                pred = (torch.argmax(pred, dim=-1) / self.quantize_output_bins)
            if self.quantize_output_bins is not None:
                pred = torch.round(pred * self.quantize_output_bins) / self.quantize_output_bins

            preds.append(pred[:, :, -1:])

            if return_logits:
                pred = (torch.argmax(pred, dim=-1) / self.quantize_output_bins)

            if self.mode == "point-tracking":
                cond_ = torch.cat([cond_, torch.cat([(vis[:, :, -1] > 0.5).unsqueeze(-1).float(), pred[:, :, -1]], dim=-1)], dim=1)
            elif self.mode == "object-tracking":
                cond_ = torch.cat([cond_, pred[:, :, -1]], dim=1)
        if self.mode == "point-tracking":
            return torch.cat(preds, dim=2), torch.cat(visibility, dim=2).unsqueeze(-1)
        elif self.mode == "object-tracking":
            return torch.cat(preds, dim=2)
        
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
