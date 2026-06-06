"""
Added get selfattention from all layer

Mostly copy-paster from DINO (https://github.com/facebookresearch/dino/blob/main/vision_transformer.py)
and timm library (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)

"""
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from torch.nn import functional as F
import torch
import torch.nn as nn




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Prototype_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.scale = qk_scale or head_dim ** -0.5
        self.learn_scale = nn.Parameter(torch.ones(num_heads, 1, 1), requires_grad=True)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim *2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prototype_token):
        B, N, C = x.shape
        prototype_num = prototype_token.shape[1]
        # prototype_token = prototype_token.repeat((B, 1, 1))
        # q, k, b [B, head_num, num_token, C]
        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.kv(prototype_token).reshape(B, prototype_num, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.learn_scale
        attn = F.relu(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Aggregation_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, T, C = x.shape
        _, N, _ = y.shape
        q = self.q(x).reshape(B, T, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attnmap = attn.softmax(dim=-1)
        attn = self.attn_drop(attnmap)
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Aggregation_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Aggregation_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(y)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Prototype_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Prototype_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, prototype, return_attention=False):
        # if attn_mask is not None:
        #     y, attn = self.attn(self.norm1(x))
        # else:
        y, attn = self.attn(self.norm1(x), self.norm1(prototype))
        x = self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        else:
            return x
        

class ResidualCalibrationModule(nn.Module):
    """
    Calibrate prototype contribution based on residual map.
    Prototype tokens that help reconstruct high-residual (anomaly) regions
    are penalized, pushing the model to focus on normal regions.
    """
    def __init__(self, dim, num_prototypes, num_heads=8):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_heads = num_heads

        # Lightweight MLP to compute calibration weights per prototype
        self.calibration_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_prototypes),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, en_fused, de_fused, prototype):
        """
        en_fused: [B, C, H, W] — fused encoder features
        de_fused: [B, C, H, W] — fused decoder features
        prototype: [B, INP_num, C] — current prototype

        Returns:
            calibrated_prototype: [B, INP_num, C]
            calibration_weights:  [B, INP_num] for logging/loss
        """
        B, C, H, W = en_fused.shape

        # Residual map: [B, H, W]
        residual = 1.0 - F.cosine_similarity(en_fused, de_fused, dim=1)  # [B, H, W]

        # Normalize residual to [0, 1] per sample
        r_flat = residual.flatten(1)  # [B, H*W]
        r_min  = r_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)  # [B,1,1]
        r_max  = r_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        residual_norm = (residual - r_min) / (r_max - r_min + 1e-8)  # [B, H, W]

        # Normal mask: high residual → low weight (we want prototype to focus on normal)
        normal_mask = 1.0 - residual_norm  # [B, H, W]

        # Weighted average of encoder features over normal regions → normal summary
        normal_mask_flat = normal_mask.flatten(1).unsqueeze(-1)         # [B, H*W, 1]
        en_flat = en_fused.flatten(2).permute(0, 2, 1)                  # [B, H*W, C]
        normal_context = (en_flat * normal_mask_flat).sum(dim=1) / \
                         (normal_mask_flat.sum(dim=1) + 1e-8)           # [B, C]

        # Compute calibration weight per prototype token
        # How much each prototype aligns with normal context
        normal_context_norm = self.norm(normal_context)                 # [B, C]
        calibration_weights = self.calibration_mlp(normal_context_norm) # [B, INP_num]

        # Apply calibration: scale prototype by normality weight
        calibrated_prototype = prototype * calibration_weights.unsqueeze(-1)  # [B, INP_num, C]

        return calibrated_prototype, calibration_weights





