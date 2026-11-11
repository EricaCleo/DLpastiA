"""Method 6: Mirror-conditioned + diffusion (our new idea). DECISION PENDING.

(a) Trained: diffusion model that takes the flipped masked image as extra input (supervised).
(b) Training-free (recommended): adaptive symmetry guidance added to RePaint's sampling loop.
    Per step, with x0_hat the model's clean-image estimate:
        mirror     = flip(x0_hat)
        use_mirror = hole * flip(visible)            # only where the mirrored pixel is visible
        s          = symmetry_score(x_known, visible)  # 0 = asymmetric ... 1 = symmetric
        x0_hat    += lambda * s * use_mirror * (mirror - x0_hat)
Compare: RePaint vs RePaint + fixed guidance vs RePaint + adaptive guidance (ours).
"""
import torch

from .repaint import RePaint


def symmetry_score(x, visible):
    """Global score in [0,1] per image: how similar visible pixels are to their visible mirror partners."""
    pair = visible * torch.flip(visible, [3])
    diff = ((x - torch.flip(x, [3])).abs() * pair).sum((1, 2, 3)) / (3 * pair.sum((1, 2, 3))).clamp_min(1)
    return (1 - diff / 2).clamp(0, 1)  # |diff| <= 2 for values in [-1,1]


def mirror_guidance(x0_hat, x_known, hole, lam=0.1, adaptive=True):
    visible = 1 - hole
    use = hole * torch.flip(visible, [3])
    s = symmetry_score(x_known, visible)[:, None, None, None] if adaptive else 1.0
    return x0_hat + lam * s * use * (torch.flip(x0_hat, [3]) - x0_hat)


class MirrorGuidedRePaint(RePaint):
    name = "mirror_guided_repaint"

    def __init__(self, lam=0.1, adaptive=True, **kw):
        super().__init__(**kw)
        self.lam, self.adaptive = lam, adaptive

    def fill(self, masked, mask, K=1):
        raise NotImplementedError("RePaint sampler with mirror_guidance() applied to x0_hat each step")
