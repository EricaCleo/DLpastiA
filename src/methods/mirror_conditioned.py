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

    # Test-time ablation: does the trained network really USE the mirrored half? (scripts/m4_mirror_conditioned.py --ablate)
    #   'none'    : normal input
    #   'blank'   : the mirror channels say "nothing visible" (image 0, mask 1)
    #   'shuffled': the mirror channels come from ANOTHER face in the batch (a plausible but wrong mirror)
    ablate = "none"

    @torch.no_grad()
    def fill(self, masked, mask, K=1):
        if self.ablate == "none":
            return super().fill(masked, mask, K)
        masked, mask = masked.to(self.device), mask.to(self.device)
        inp = self.build_input(masked, mask).clone()
        if self.ablate == "blank":
            inp[:, 4:7], inp[:, 7:8] = 0.0, 1.0
        elif self.ablate == "shuffled":
            assert inp.shape[0] > 1, "shuffled ablation needs a batch of at least 2 images"
            inp[:, 4:8] = torch.roll(inp[:, 4:8], 1, 0)
        else:
            raise ValueError(self.ablate)
        return [self.G(inp, torch.randn(masked.shape[0], self.G.z_dim, device=self.device)).clamp(-1, 1) for _ in range(K)]
