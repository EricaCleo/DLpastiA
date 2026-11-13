"""Method 4: Mirror-conditioned StyleGAN (supervised). Same model as Method 3 plus a flipped copy of the input.

Input = masked (3) + mask (1) + flipped masked (3) + flipped mask (1) = 8 channels (+ z).
Ablation: 03 (baseline) vs 04a (mirror input) vs 04b (mirror input + symmetry loss, weight --symmetry in training).
Test on random, one-sided (one eye/ear) and centre masks.  See scripts/m4_mirror_conditioned.py.
"""
import torch

from src.gan.inputs import mirror_input  # noqa: F401  (re-exported)
from .stylegan_noninversion import CoModGANInpainter


def symmetry_loss(pred, weight=0.05):
    """L_sym = lambda * || pred - flip(pred) ||_1  (small lambda: faces are not perfectly symmetric)."""
    return weight * (pred - torch.flip(pred, [3])).abs().mean()


class MirrorConditionedInpainter(CoModGANInpainter):
    name = "mirror_conditioned"
    mode = "mirror"
