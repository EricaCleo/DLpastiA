"""Method 1: Masked Autoencoder (self-supervised, ViT backbone). Deterministic, fast.

Plan: start from public pretrained MAE weights, fine-tune on AFHQ with ~75% random patch masking
(MSE on hidden patches). At inference, mark every 16x16 patch overlapping the hole as hidden,
predict those patches, then composite.
"""
from .base import Inpainter


class MAEInpainter(Inpainter):
    name = "mae"
    stochastic = False

    def __init__(self, ckpt=None, patch=16, device="cuda"):
        self.ckpt, self.patch, self.device = ckpt, patch, device
        # TODO: load ViT-MAE (e.g. facebook/vit-mae-base via transformers)

    def train(self, loader, epochs=10):
        raise NotImplementedError("fine-tune MAE on AFHQ (mask ratio 0.75, loss on hidden patches)")

    def fill(self, masked, mask, K=1):
        raise NotImplementedError
