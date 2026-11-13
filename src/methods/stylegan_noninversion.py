"""Method 3: StyleGAN non-inversion, CoModGAN-style (supervised). Fast at test time, stochastic.

Encoder reads (masked image + mask = 4 channels); its global vector is co-modulated with the style from z ~ N(0,I);
a StyleGAN2-style generator (modulated convolutions, encoder skips) outputs the full image. Trained on
(masked, original) pairs: adversarial + R1 on D, plus L1 (hole) and LPIPS. See src/gan/ and scripts/m3_stylegan_noninversion.py.
Written in plain PyTorch (no custom CUDA ops), trained from scratch at 128x128 under a wall-clock budget.
"""
import torch

from src.gan import load_generator
from src.gan.inputs import INPUTS
from .base import Inpainter


class CoModGANInpainter(Inpainter):
    name = "stylegan_noninversion"
    stochastic = True
    mode = "comod"

    def __init__(self, G, device="cuda"):
        self.G, self.device = G.to(device).eval(), device
        self.build_input = INPUTS[self.mode][0]

    @classmethod
    def from_ckpt(cls, path, device="cuda"):
        G, cfg = load_generator(path, device)
        assert cfg["mode"] == cls.mode, f"checkpoint is mode '{cfg['mode']}', expected '{cls.mode}'"
        return cls(G, device)

    @torch.no_grad()
    def fill(self, masked, mask, K=1):
        masked, mask = masked.to(self.device), mask.to(self.device)
        inp = self.build_input(masked, mask)
        return [self.G(inp, torch.randn(masked.shape[0], self.G.z_dim, device=self.device)).clamp(-1, 1) for _ in range(K)]
