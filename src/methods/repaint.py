"""Method 5: RePaint (unsupervised, unconditional DDPM, zero-shot). Slow, stochastic.

Training: standard DDPM noise-prediction loss on clean AFHQ (no masks). Plan: use a pretrained
AFHQ diffusion model if available, else train at 128x128 (or 64x64).
Sampling at each step t:  unknown <- one reverse step; known <- q(x_{t-1}|x_0) noised real pixels;
x_{t-1} = m*unknown + (1-m)*known; plus resampling jumps (jump length ~10, ~10 resamples).
Backbone options: U-Net (baseline) or U-ViT (Bao et al., 2023) for the ViT-vs-U-Net question.
"""
from .base import Inpainter


class RePaint(Inpainter):
    name = "repaint"
    stochastic = True

    def __init__(self, model=None, T=250, jump_length=10, resamples=10, device="cuda"):
        self.model, self.T, self.jump_length, self.resamples = model, T, jump_length, resamples
        # TODO: load diffusion model + noise schedule

    def fill(self, masked, mask, K=1):
        raise NotImplementedError
