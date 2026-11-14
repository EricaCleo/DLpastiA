"""Method 6 (ours): Mirror-guided RePaint. DECIDED: option (b), training-free adaptive guidance.

No new training: reuse the diffusion model of Method 5. At each sampling step, with x0_hat the
model's clean-image estimate:
    mirror     = flip(x_known)                       # REAL visible pixels, mirrored (not the model's guess)
    use_mirror = hole * flip(visible)               # only hole pixels whose mirrored pixel is visible
    s          = symmetry_score(x_known, visible)   # 0 = asymmetric ... 1 = symmetric (per image)
    x0_hat    += lambda * s * use_mirror * (mirror - x0_hat)
(Where use_mirror = 1 the mirrored location is visible, so flip(x_known) there is real image content.)
Variants to compare: RePaint (no mirror) | fixed guidance (adaptive=False) | adaptive guidance.
Tunables: lam (strength) and guide_until (fraction of the schedule, from the start, where guidance is on).
"""
import torch
import torch.nn.functional as F

from .repaint import RePaint


def symmetry_score(x, visible):
    """Per-image score in [0,1]: how similar visible pixels are to their (visible) mirror partners."""
    pair = visible * torch.flip(visible, [3])
    diff = ((x - torch.flip(x, [3])).abs() * pair).sum((1, 2, 3)) / (3 * pair.sum((1, 2, 3))).clamp_min(1)
    return (1 - diff / 2).clamp(0, 1)  # |diff| <= 2 for values in [-1,1]


def mirror_guidance(x0_hat, x_known, hole, lam=0.1, adaptive=True, feather=0):
    """feather > 0: blur the guidance mask with a box filter of that many pixels, so the pull fades out smoothly at the
    edges of the region where the mirrored side is visible (hard edges show up as rectangular seams)."""
    visible = 1 - hole
    use = hole * torch.flip(visible, [3])
    if feather and feather > 1:
        k = int(feather) | 1
        use = F.avg_pool2d(use, k, stride=1, padding=k // 2, count_include_pad=False) * hole
    s = symmetry_score(x_known, visible)[:, None, None, None] if adaptive else 1.0
    mirror = torch.flip(x_known, [3])  # real known pixels; only used where `use` = 1 (mirrored spot is visible)
    return x0_hat + lam * s * use * (mirror - x0_hat)


class MirrorGuidedRePaint(RePaint):
    name = "mirror_guided_repaint"

    def __init__(self, model, lam=0.1, adaptive=True, guide_frac=1.0, feather=0, **kw):
        """guide_frac: guidance is applied only in the LAST fraction of the schedule (low-noise levels t < guide_frac * steps).
        1.0 = every step (strong: the per-step pulls accumulate over ~1000 steps); small values (0.2-0.5) = light touch."""
        super().__init__(model, **kw)
        self.lam, self.adaptive, self.guide_frac, self.feather = lam, adaptive, guide_frac, feather

    def guide(self, x0_hat, gt, keep, t=None):
        if t is not None and t >= self.guide_frac * self.diff.num_steps:
            return x0_hat
        return mirror_guidance(x0_hat, gt, 1 - keep, self.lam, self.adaptive, self.feather)
