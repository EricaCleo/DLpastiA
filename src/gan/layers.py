"""StyleGAN2-style building blocks in plain PyTorch (no custom CUDA ops: runs on any GPU, incl. RTX 5090).
Equalised learning rate, modulated/demodulated convolutions, noise injection. Resampling uses bilinear
interpolation / average pooling instead of StyleGAN2's FIR filters (simpler, slightly less exact)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SQRT2 = math.sqrt(2.0)


def lrelu(x):
    return F.leaky_relu(x, 0.2) * SQRT2


class EqLinear(nn.Module):
    def __init__(self, i, o, bias_init=0.0, lr_mul=1.0, act=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(o, i) / lr_mul)
        self.bias = nn.Parameter(torch.full((o,), float(bias_init)))
        self.scale, self.lr_mul, self.act = lr_mul / math.sqrt(i), lr_mul, act

    def forward(self, x):
        x = F.linear(x, self.weight * self.scale, self.bias * self.lr_mul)
        return lrelu(x) if self.act else x


class EqConv2d(nn.Module):
    def __init__(self, i, o, k, padding=0, bias=True, act=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(o, i, k, k))
        self.bias = nn.Parameter(torch.zeros(o)) if bias else None
        self.scale, self.padding, self.act = 1 / math.sqrt(i * k * k), padding, act

    def forward(self, x):
        x = F.conv2d(x, self.weight * self.scale, self.bias, padding=self.padding)
        return lrelu(x) if self.act else x


class ModConv(nn.Module):
    """Modulated convolution (StyleGAN2): per-sample style scales the weights; optional demodulation."""

    def __init__(self, i, o, k, style_dim, demodulate=True, upsample=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, o, i, k, k))
        self.affine = EqLinear(style_dim, i, bias_init=1.0)
        self.scale, self.k, self.o, self.i = 1 / math.sqrt(i * k * k), k, o, i
        self.demodulate, self.upsample = demodulate, upsample

    def forward(self, x, style):
        B, C, H, W = x.shape
        w = self.scale * self.weight * self.affine(style).view(B, 1, C, 1, 1)         # (B, o, i, k, k)
        if self.demodulate:
            w = w * torch.rsqrt(w.square().sum([2, 3, 4], keepdim=True) + 1e-8)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            H, W = H * 2, W * 2
        out = F.conv2d(x.reshape(1, B * C, H, W), w.reshape(B * self.o, C, self.k, self.k),
                       padding=self.k // 2, groups=B)
        return out.view(B, self.o, H, W)


class NoiseInject(nn.Module):
    enabled = True

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if not self.enabled:
            return x
        return x + self.weight * torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)


class StyledConv(nn.Module):
    def __init__(self, i, o, style_dim, upsample=False):
        super().__init__()
        self.conv, self.noise = ModConv(i, o, 3, style_dim, upsample=upsample), NoiseInject()
        self.bias = nn.Parameter(torch.zeros(1, o, 1, 1))

    def forward(self, x, style):
        return lrelu(self.noise(self.conv(x, style)) + self.bias)


class ToRGB(nn.Module):
    def __init__(self, i, style_dim):
        super().__init__()
        self.conv = ModConv(i, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        out = self.conv(x, style) + self.bias
        if skip is not None:
            out = out + F.interpolate(skip, scale_factor=2, mode="bilinear", align_corners=False)
        return out


class DownBlock(nn.Module):
    """Residual block that halves the resolution (encoder and discriminator)."""

    def __init__(self, i, o):
        super().__init__()
        self.c1, self.c2 = EqConv2d(i, i, 3, 1, act=True), EqConv2d(i, o, 3, 1, act=True)
        self.skip = EqConv2d(i, o, 1, bias=False)

    def forward(self, x):
        y = F.avg_pool2d(self.c2(self.c1(x)), 2)
        return (y + F.avg_pool2d(self.skip(x), 2)) / SQRT2


def minibatch_std(x, group=4):
    B, C, H, W = x.shape
    G = group if B % group == 0 else 1
    y = x.view(G, B // G, C, H, W)
    y = (y - y.mean(0)).square().mean(0).add(1e-8).sqrt().mean([1, 2, 3])        # (B//G,)
    return torch.cat([x, y.view(-1, 1, 1, 1).repeat(G, 1, H, W)], 1)
