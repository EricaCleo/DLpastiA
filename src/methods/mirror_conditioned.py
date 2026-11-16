"""Method 4: Mirror-conditioned StyleGAN (supervised). Same model as Method 3 plus a flipped copy of the input.

Input = masked (3) + mask (1) + flipped masked (3) + flipped mask (1) = 8 channels (+ z).
Ablation: 03 (baseline) vs 04a (mirror input) vs 04b (mirror input + symmetry loss, weight --symmetry in training)
vs 04c (mirror input ALIGNED to the face's own symmetry axis, --aligned in training; see src/gan/inputs.py).
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

    @classmethod
    def from_ckpt(cls, path, device="cuda"):
        """Works for 04a/04b (mode 'mirror') and 04c (mode 'mirror_aligned'): the checkpoint says which input it needs."""
        from src.gan import load_generator
        from src.gan.inputs import INPUTS
        G, cfg = load_generator(path, device)
        assert cfg["mode"] in ("mirror", "mirror_aligned"), f"checkpoint is mode '{cfg['mode']}', not a mirror model"
        m = cls(G, device)
        m.mode, m.build_input = cfg["mode"], INPUTS[cfg["mode"]][0]
        return m
