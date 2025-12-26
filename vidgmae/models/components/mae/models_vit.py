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

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, global_pool=False, full_image=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            # self.fc_norm = norm_layer(embed_dim)

            # del self.norm  # remove the original norm

        if full_image:
            self.full_img_emb = nn.Parameter(
                torch.randn(
                    embed_dim,
                )
            )
            self.detect_encoder = nn.Linear(1500, embed_dim)
            self.pose_encoder = nn.Linear(229, embed_dim)

    def forward_features(self, x):
        # check x is a list or not
        if isinstance(x, list):
            B = x[0].shape[0]

            x_0 = self.patch_embed(x[0])
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x_0 = torch.cat((cls_tokens, x_0), dim=1)
            x_0 = x_0 + self.pos_embed

            x_1 = self.patch_embed(x[1])
            x_1 = torch.cat((cls_tokens, x_1), dim=1)
            x_1 = x_1 + self.pos_embed + self.full_img_emb
            x_token = torch.cat((x_0, x_1[:, 1:, :]), dim=1)

            # additonal vector containing detection bins
            if len(x) == 3:
                det_vector = x[2]
                det_vector = self.detect_encoder(det_vector)
                # det_vector = det_vector.unsqueeze(1)
                # x_token = torch.cat((x_token, det_vector), dim=1)
                x_token[:, 0, :] += det_vector

            if len(x) == 4:
                pose_vector = x[3].to(self.pose_encoder.weight.dtype)
                pose_vector = self.pose_encoder(pose_vector)
                # pose_vector = pose_vector.unsqueeze(1)
                # x_token = torch.cat((x_token, pose_vector), dim=1)
                x_token[:, 0, :] += pose_vector

            x = self.pos_drop(x_token)
        else:
            B = x.shape[0]
            x = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = self.norm(x)
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = x  # self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward_head(self, x, pre_logits: bool = False):
        # if self.global_pool:
        #     x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        # x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14_JM(**kwargs):
    model = VisionTransformer(
        patch_size=28,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
