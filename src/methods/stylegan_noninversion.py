"""Method 3: StyleGAN non-inversion, CoModGAN-style (supervised). Fast at test time, stochastic.

Plan: encoder reads (masked image + mask = 4 channels); its output is combined with a style code
from z ~ N(0,I) (512-d); a StyleGAN-style generator produces the full image.
Losses: non-saturating adversarial + R1 on D (+ optional LPIPS). Trained on (masked, original) pairs.
`in_channels` is a parameter so Method 4 reuses this exact model with 8 channels.
"""
from .base import Inpainter


class CoModGANInpainter(Inpainter):
    name = "stylegan_noninversion"
    stochastic = True

    def __init__(self, in_channels=4, z_dim=512, device="cuda"):
        self.in_channels, self.z_dim = in_channels, z_dim
        # TODO: encoder + StyleGAN generator + discriminator

    def build_input(self, masked, mask):
        import torch
        return torch.cat([masked, mask], dim=1)  # 3 + 1 channels

    def train(self, loader, epochs=100):
        raise NotImplementedError("adv + R1 (+ LPIPS) on (masked, original) pairs")

    def fill(self, masked, mask, K=1):
        raise NotImplementedError
