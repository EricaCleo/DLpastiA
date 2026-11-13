"""U-ViT noise predictor (Bao et al., 2023, "All are Worth Words: A ViT Backbone for Diffusion Models").
All inputs (time step, image patches) are tokens; transformer blocks are connected by LONG SKIP connections
(U-Net style: early blocks feed late blocks by concatenation + linear). Same interface as EpsUNet: forward(x, t) -> noise.
Plain PyTorch (scaled_dot_product_attention); no extra dependencies.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, skip=False):
        super().__init__()
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.norm1, self.attn, self.norm2 = nn.LayerNorm(dim), Attention(dim, heads), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))      # long skip connection
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class EpsUViT(nn.Module):
    def __init__(self, image_size=128, patch=4, dim=512, depth=13, heads=8, mlp_ratio=4):
        super().__init__()
        assert image_size % patch == 0 and depth % 2 == 1, "image_size must divide by patch; depth must be odd (in + 1 mid + out)"
        self.cfg = dict(arch="uvit", image_size=image_size, patch=patch, dim=dim, depth=depth, heads=heads, mlp_ratio=mlp_ratio)
        self.patch, self.dim, self.grid = patch, dim, image_size // patch
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.time_embed = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.grid ** 2, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        n = depth // 2
        self.in_blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(n)])
        self.mid_block = Block(dim, heads, mlp_ratio)
        self.out_blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, skip=True) for _ in range(n)])
        self.norm = nn.LayerNorm(dim)
        self.to_patch = nn.Linear(dim, patch * patch * 3)
        self.final_conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x, t):
        B = x.shape[0]
        tok = self.patch_embed(x).flatten(2).transpose(1, 2)                       # (B, grid^2, dim)
        t_tok = self.time_embed(timestep_embedding(t, self.dim)).unsqueeze(1)      # time as a token
        h = torch.cat([t_tok, tok], 1) + self.pos_embed
        skips = []
        for b in self.in_blocks:
            h = b(h); skips.append(h)
        h = self.mid_block(h)
        for b in self.out_blocks:
            h = b(h, skips.pop())
        h = self.to_patch(self.norm(h))[:, 1:]                                      # drop the time token
        p, g = self.patch, self.grid
        h = h.view(B, g, g, p, p, 3).permute(0, 5, 1, 3, 2, 4).reshape(B, 3, g * p, g * p)
        return self.final_conv(h)
