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


def shift_cols(t, d):
    """Shift the columns of every image right by d[b] pixels (negative = left), zero fill. t: (B,C,H,W), d: (B,) ints."""
    out = torch.zeros_like(t)
    W = t.shape[-1]
    for b in range(t.shape[0]):
        k = int(d[b])
        if k == 0:
            out[b] = t[b]
        elif k > 0:
            out[b, ..., k:] = t[b, ..., : W - k]
        else:
            out[b, ..., : W + k] = t[b, ..., -k:]
    return out


def mirrored(t, shifts=None):
    """Left-right flip, then (optionally) shift so the mirror axis matches the face's own symmetry axis."""
    f = torch.flip(t, [3])
    return f if shifts is None else shift_cols(f, shifts)


def best_shift(x, visible, max_shift=16, min_pairs=0.05):
    """Per image: the column shift (-max_shift..max_shift) of the flipped image that best matches the visible pixels.
    Falls back to 0 when too few visible pixel pairs exist to judge."""
    B = x.shape[0]
    best = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        xb, vb = x[b:b + 1], visible[b:b + 1]
        errs = {}
        for d in range(-max_shift, max_shift + 1):
            dd = torch.tensor([d])
            pair = vb * mirrored(vb, dd)
            n = pair.sum()
            if n < min_pairs * vb.numel():
                continue
            errs[d] = ((xb - mirrored(xb, dd)).abs() * pair).sum().item() / (3 * n.item())
        if errs:
            best[b] = min(errs, key=errs.get)
    return best


def symmetry_score(x, visible, shifts=None):
    """Per-image score in [0,1]: how similar visible pixels are to their (visible) mirror partners (after the axis shift)."""
    pair = visible * mirrored(visible, shifts)
    diff = ((x - mirrored(x, shifts)).abs() * pair).sum((1, 2, 3)) / (3 * pair.sum((1, 2, 3))).clamp_min(1)
    return (1 - diff / 2).clamp(0, 1)  # |diff| <= 2 for values in [-1,1]


def mirror_guidance(x0_hat, x_known, hole, lam=0.1, adaptive=True, feather=0, shifts=None, s_lo=0.0, s_hi=1.0):
    """feather > 0: blur the guidance mask so the pull fades out smoothly (hard edges show up as rectangular seams).
    shifts: per-image column shift aligning the mirror axis with the face's symmetry axis (see best_shift).
    s_lo, s_hi: the symmetry score is rescaled (score - s_lo) / (s_hi - s_lo) into [0,1] before it scales the pull:
    real faces score only about 0.65-0.9, so without rescaling the 'adaptive' weight barely varies."""
    visible = 1 - hole
    use = hole * mirrored(visible, shifts)
    if feather and feather > 1:
        k = int(feather) | 1
        use = F.avg_pool2d(use, k, stride=1, padding=k // 2, count_include_pad=False) * hole
    if adaptive:
        sc = symmetry_score(x_known, visible, shifts)
        sc = ((sc - s_lo) / max(s_hi - s_lo, 1e-6)).clamp(0, 1)
        s = sc[:, None, None, None]
    else:
        s = 1.0
    return x0_hat + lam * s * use * (mirrored(x_known, shifts) - x0_hat)


class MirrorGuidedRePaint(RePaint):
    name = "mirror_guided_repaint"

    def __init__(self, model, lam=0.1, adaptive=True, guide_frac=1.0, feather=0, align=False, max_shift=16,
                 s_lo=0.0, s_hi=1.0, **kw):
        """guide_frac: guidance is applied only in the LAST fraction of the schedule (low-noise levels t < guide_frac * steps).
        1.0 = every step (strong: the per-step pulls accumulate over ~1000 steps); small values (0.2-0.5) = light touch."""
        super().__init__(model, **kw)
        self.lam, self.adaptive, self.guide_frac, self.feather = lam, adaptive, guide_frac, feather
        self.align, self.max_shift, self.s_lo, self.s_hi, self._shifts = align, max_shift, s_lo, s_hi, None

    def guide(self, x0_hat, gt, keep, t=None):
        if t is not None and t >= self.guide_frac * self.diff.num_steps:
            return x0_hat
        return mirror_guidance(x0_hat, gt, 1 - keep, self.lam, self.adaptive, self.feather, self._shifts, self.s_lo, self.s_hi)

    def sample(self, gt, mask):
        # the known pixels never change during sampling, so the axis alignment is computed once per image
        self._shifts = best_shift(gt, 1 - mask, self.max_shift).to(gt.device) if self.align else None
        return super().sample(gt, mask)
