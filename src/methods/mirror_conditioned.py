"""Method 4: Mirror-conditioned StyleGAN (supervised). Same model as Method 3 plus a flipped copy.

Input = masked (3) + mask (1) + flipped masked (3) + flipped mask (1) = 8 channels (+ z).
Ablation: 03 (baseline) vs 04a (mirror input) vs 04b (mirror input + symmetry loss).
Test on random, one-sided (one eye/ear) and centre masks.
"""
import torch

from .stylegan_noninversion import CoModGANInpainter


def mirror_input(masked, mask):
    """(B,8,H,W): [masked, mask, flip(masked), flip(mask)]"""
    return torch.cat([masked, mask, torch.flip(masked, [3]), torch.flip(mask, [3])], dim=1)


def symmetry_loss(pred, weight=0.05):
    """L_sym = lambda * || pred - flip(pred) ||_1  (small lambda: faces are not perfectly symmetric)."""
    return weight * (pred - torch.flip(pred, [3])).abs().mean()


class MirrorConditionedInpainter(CoModGANInpainter):
    name = "mirror_conditioned"

    def __init__(self, use_symmetry_loss=False, sym_weight=0.05, **kw):
        super().__init__(in_channels=8, **kw)
        self.use_symmetry_loss, self.sym_weight = use_symmetry_loss, sym_weight

    def build_input(self, masked, mask):
        return mirror_input(masked, mask)
