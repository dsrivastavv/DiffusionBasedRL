# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
# import xformers.ops
import torch.nn as nn

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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

#################################################################################
#               Embedding Layers for Timesteps and Captions                     #
#################################################################################

class ScalarEmbedder(nn.Module):
    """
    Embeds scalar values into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def scalar_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.scalar_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

#################################################################################
#                                 DiT Block Utils                               #
#################################################################################

# class MultiHeadCrossAttention(nn.Module):
#     def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0., **block_kwargs):
#         super(MultiHeadCrossAttention, self).__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads

#         self.q_linear = nn.Linear(d_model, d_model)
#         self.kv_linear = nn.Linear(d_model, d_model*2)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(d_model, d_model)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x, y, y_padding_mask=None):
#         # query: img tokens; key/value: condition; mask: if padding tokens
#         B, N, C = x.shape

#         q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
#         kv = self.kv_linear(y).view(1, -1, 2, self.num_heads, self.head_dim)
#         k, v = kv.unbind(2)
#         attn_bias = None
#         if y_padding_mask is not None:
#             y_padding_len = y_padding_mask.sum(dim=1).cpu().tolist()
#             attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, y_padding_len)
#         x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
#         x = x.view(B, -1, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, x_padding_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        modulate_sa = modulate(self.norm1(x), shift_msa, scale_msa)
        x = (
            x
            + gate_msa.unsqueeze(1)
            * self.attn(modulate_sa, modulate_sa, modulate_sa, key_padding_mask=x_padding_mask)[0]
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class InputEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, **kwargs) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(in_features=input_dim, out_features=hidden_dim)


    def forward(self, x):
        '''
        Embed inputs to hidden dimension space
        '''
        return self.proj(x)


class FinalLayer(nn.Module):
    """
    The final layer of DiT to map from hidden dim to input dimension
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        max_in_len=100,
        learn_sigma=False,
    ):
        '''
        Args:
        - in_channels: Number of input channels (or embedding of tokenized input)
        - hidden_size: Hidden dimension size
        - depth: Number of transformer blocks
        - num_heads: Number of attention heads
        - mlp_ratio: Ratio of hidden dimension to mlp hidden dimension
        - class_dropout_prob: Probability of dropping y
        - max_in_len: Maximum input length
        - learn_sigma: Whether to learn sigma
        '''
        super().__init__()
        self.learn_sigma = learn_sigma
        self.num_heads = num_heads
        self.max_len = max_in_len

        self.x_embedder = InputEmbedder(in_channels, hidden_size)
        self.t_embedder = ScalarEmbedder(hidden_size)

        # num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, max_in_len, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        if learn_sigma:
            self.final_layer = FinalLayer(hidden_size, 2 * in_channels)
        else:
            self.final_layer = FinalLayer(hidden_size, in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], max_len=self.max_len)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        nn.init.xavier_uniform_(self.x_embedder.proj.weight)
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, cond, t, x_padding_mask=None):
        """
        Forward pass of DiT.
        x: (N, max_in_len, in_channel) tensor of spatial inputs 
        t: (N,) tensor of diffusion timesteps
        ar: (N,) tensor of aspect ratios
        x_enc: (N, hidden_size') encoding of token labels
        """
        # token level conditions
        x = self.x_embedder(x) + self.pos_embed[:, :x.shape[1]] # (N, max_in_len, hidden_size)

        # sample level conditions
        t = self.t_embedder(t)                   # (N, hidden_size)
        c = t                                    # (N, hidden_size)

        # transformer blocks
        for block in self.blocks:
            x = block(x, c, x_padding_mask)   # (N, max_in_len, hidden_size)

        # final layer
        x = self.final_layer(x, c)               # (N, max_in_len, (2 or 1) x in_channel)

        if self.learn_sigma:
            x = x.chunk(2, dim=-1)                 # (N, max_in_len, in_channel), (N, max_in_len, in_channel)
            x = torch.concat([x[0], x[1]], dim=1)        # (N, 2 x max_in_len, in_channel

        # return output
        return x                                 # (N, 2 x max_in_len, in_channel)

    def forward_with_cfg(self, x, x_padding_mask, t, ar, x_enc, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, x_padding_mask, t, ar, x_enc)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

# def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
#     """
#     grid_size: int of the grid height and width
#     return:
#     pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
#     """
#     grid_h = np.arange(grid_size, dtype=np.float32)
#     grid_w = np.arange(grid_size, dtype=np.float32)
#     grid = np.meshgrid(grid_w, grid_h)  # here w goes first
#     grid = np.stack(grid, axis=0)

#     grid = grid.reshape([2, 1, grid_size, grid_size])
#     pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
#     if cls_token and extra_tokens > 0:
#         pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
#     return pos_embed


# def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
#     assert embed_dim % 2 == 0

#     # use half of dimensions to encode grid_h
#     emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
#     emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

#     emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
#     return emb

def get_1d_sincos_pos_embed(embed_dim, max_len, cls_token=False, extra_tokens=0):
    """
    max_len: maximum length of sequence
    return:
    pos_embed: [max_len, embed_dim] or [1+max_len, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(max_len, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)  # (H*W, D/2)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)

    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL(**kwargs):
    return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)

def DiT_S(**kwargs):
    return DiT(depth=9, hidden_size=320, num_heads=8, **kwargs)

def DiT_XS(**kwargs):
    return DiT(depth=6, hidden_size=256, num_heads=4, **kwargs)

def DiT_XXS(**kwargs):
    # return DiT(depth=8, hidden_size=160, num_heads=4, **kwargs)
    return DiT(depth=8, hidden_size=80, num_heads=4, **kwargs)
    # return DiT(depth=3, hidden_size=128, num_heads=4, **kwargs)

LDiT_models = {
    'DiT-XL': DiT_XL,
    'DiT-L':  DiT_L,
    'DiT-B':  DiT_B,
    'DiT-S':  DiT_S,
    'DiT-XS': DiT_XS,
    'DiT-XXS': DiT_XXS,
}
