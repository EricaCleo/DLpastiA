"""Generators, encoder and discriminator.

CoModGenerator (Methods 3 and 4):  encoder reads (masked image + mask [+ mirrored copies]); its global vector is
co-modulated with the style from z, its 4x4 features start the synthesis and its multi-scale features are added as skips.
UncondGenerator (Method 2, inversion): ordinary StyleGAN2-style generator, z -> image (never sees holes).
"""
import math

import torch
import torch.nn as nn

from .layers import DownBlock, EqConv2d, EqLinear, StyledConv, ToRGB, minibatch_std


def chan(res, base=16384, cap=512):
    return min(base // res, cap)


class Mapping(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, n_layers=4):
        super().__init__()
        dims = [z_dim] + [w_dim] * n_layers
        self.net = nn.Sequential(*[EqLinear(dims[i], dims[i + 1], lr_mul=0.01, act=True) for i in range(n_layers)])

    def forward(self, z):
        return self.net(z * torch.rsqrt(z.square().mean(1, keepdim=True) + 1e-8))


class Synthesis(nn.Module):
    """styles: (B, w) broadcast to every layer, or (B, n_styles, w) for per-layer styles (W+)."""

    def __init__(self, size, w_dim=512, base=16384, cap=512, use_enc=False):
        super().__init__()
        self.size, self.res = size, [2 ** i for i in range(2, int(math.log2(size)) + 1)]
        ch = {r: chan(r, base, cap) for r in self.res}
        self.ch = ch
        self.const = nn.Parameter(torch.randn(1, ch[4], 4, 4))
        self.conv4, self.rgb4 = StyledConv(ch[4], ch[4], w_dim), ToRGB(ch[4], w_dim)
        self.ups = nn.ModuleList([StyledConv(ch[r // 2], ch[r], w_dim, upsample=True) for r in self.res[1:]])
        self.convs = nn.ModuleList([StyledConv(ch[r], ch[r], w_dim) for r in self.res[1:]])
        self.rgbs = nn.ModuleList([ToRGB(ch[r], w_dim) for r in self.res[1:]])
        self.skips = nn.ModuleList([EqConv2d(ch[r], ch[r], 1) for r in self.res[1:]]) if use_enc else None
        self.n_styles = 2 + 3 * (len(self.res) - 1)

    def set_noise(self, enabled):
        for m in self.modules():
            if hasattr(m, "enabled") and isinstance(m, nn.Module) and m.__class__.__name__ == "NoiseInject":
                m.enabled = enabled

    def forward(self, styles, enc=None):
        get = (lambda i: styles[:, i]) if styles.dim() == 3 else (lambda i: styles)
        x = enc[4] if enc is not None else self.const.expand(styles.shape[0], -1, -1, -1)
        i = 0
        x = self.conv4(x, get(i)); i += 1
        rgb = self.rgb4(x, get(i)); i += 1
        for k, r in enumerate(self.res[1:]):
            x = self.ups[k](x, get(i)); i += 1
            if enc is not None:
                x = x + self.skips[k](enc[r])
            x = self.convs[k](x, get(i)); i += 1
            rgb = self.rgbs[k](x, get(i), skip=rgb); i += 1
        return rgb


class Encoder(nn.Module):
    def __init__(self, size, in_ch, w_dim=512, base=16384, cap=512):
        super().__init__()
        res = [2 ** i for i in range(2, int(math.log2(size)) + 1)]
        ch = {r: chan(r, base, cap) for r in res}
        self.size, self.inp = size, EqConv2d(in_ch, ch[size], 1, act=True)
        self.blocks = nn.ModuleList([DownBlock(ch[r], ch[r // 2]) for r in res[:0:-1]])   # size -> 8 -> 4
        self.fc = EqLinear(ch[4] * 16, w_dim, act=True)

    def forward(self, x):
        h = self.inp(x)
        feats, r = {self.size: h}, self.size
        for b in self.blocks:
            h, r = b(h), r // 2
            feats[r] = h
        return feats, self.fc(h.flatten(1))


class CoModGenerator(nn.Module):
    def __init__(self, size=128, in_ch=4, z_dim=512, w_dim=512, base=16384, cap=512, map_layers=4):
        super().__init__()
        self.size, self.in_ch, self.z_dim = size, in_ch, z_dim
        self.mapping, self.enc = Mapping(z_dim, w_dim, map_layers), Encoder(size, in_ch, w_dim, base, cap)
        self.comod = EqLinear(2 * w_dim, w_dim, act=True)
        self.synth = Synthesis(size, w_dim, base, cap, use_enc=True)

    def forward(self, x_in, z):
        feats, vec = self.enc(x_in)
        return self.synth(self.comod(torch.cat([self.mapping(z), vec], 1)), feats)


class UncondGenerator(nn.Module):
    def __init__(self, size=128, z_dim=512, w_dim=512, base=16384, cap=512, map_layers=4):
        super().__init__()
        self.size, self.z_dim, self.w_dim = size, z_dim, w_dim
        self.mapping, self.synth = Mapping(z_dim, w_dim, map_layers), Synthesis(size, w_dim, base, cap)

    def forward(self, z):
        return self.synth(self.mapping(z))


class Discriminator(nn.Module):
    def __init__(self, size=128, in_ch=4, base=16384, cap=512):
        super().__init__()
        res = [2 ** i for i in range(2, int(math.log2(size)) + 1)]
        ch = {r: chan(r, base, cap) for r in res}
        self.inp = EqConv2d(in_ch, ch[size], 1, act=True)
        self.blocks = nn.Sequential(*[DownBlock(ch[r], ch[r // 2]) for r in res[:0:-1]])
        self.conv = EqConv2d(ch[4] + 1, ch[4], 3, 1, act=True)
        self.fc1, self.fc2 = EqLinear(ch[4] * 16, ch[4], act=True), EqLinear(ch[4], 1)

    def forward(self, x):
        h = minibatch_std(self.blocks(self.inp(x)))
        return self.fc2(self.fc1(self.conv(h).flatten(1))).squeeze(1)
