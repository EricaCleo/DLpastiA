"""Symmetry losses for the mirror-conditioned GAN (method 4).

4b, strict:   mean | y - flip(y) |                  every pixel must equal its mirror partner
4c, tolerant: only the ROUGH SHAPES must match, and only where the real face is itself symmetric:
  1. outlines   E(.)  edge strength of the grey image (Sobel)        -> ignores lighting / brightness differences
  2. coarse     D(.)  average over 8x8 blocks (128 px -> 16 x 16)     -> ignores fur markings and fine detail
  3. shift      compare with the mirror shifted by -1, 0, +1 blocks, keep the best match -> tolerates off-centre faces
  4. weight     w = exp(-d_real / tau), d_real = the same comparison on the REAL face (in units of its average outline
                strength, tau = 1): full penalty where the real face is symmetric in rough shape, little where it is not
  L = mean over images of  min_shift  sum(w * |c(y) - shift(flip(c(y)))|) / sum(w)       with c = D(E(.))
y is the composite (real visible pixels + generated hole), so only the hole receives gradients.
"""
import torch
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def coarse_outlines(img, cell=8):
    """(B,3,H,W) in [-1,1] -> (B,1,H/cell,W/cell): how much outline (edge strength) there is in each block."""
    g = img.mean(1, keepdim=True)
    kx = _SOBEL_X.to(img.device, img.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), kx)
    gy = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), ky)
    edges = (gx.square() + gy.square() + 1e-8).sqrt()
    return F.avg_pool2d(edges, cell)


def _mirror_diff(c, d):
    """|c - (flipped c shifted d blocks to the right)| and the mask of blocks where both exist. c: (B,1,h,w)."""
    f = torch.flip(c, [3])
    W = c.shape[-1]
    shifted, valid = torch.zeros_like(f), torch.zeros_like(f)
    if d >= 0:
        shifted[..., d:], valid[..., d:] = f[..., :W - d], 1.0
    else:
        shifted[..., :W + d], valid[..., :W + d] = f[..., -d:], 1.0
    return (c - shifted).abs() * valid, valid


def tolerant_symmetry_loss(pred, real, cell=8, max_shift=1, tau=1.0, return_parts=False):
    """4c's symmetry loss (see the module docstring). pred, real: (B,3,H,W) in [-1,1]."""
    cr = coarse_outlines(real.detach(), cell)
    scale = cr.mean((1, 2, 3), keepdim=True) + 1e-6           # each face on the same scale (independent of contrast)
    cr = cr / scale
    cp = coarse_outlines(pred, cell) / scale
    shifts = list(range(-max_shift, max_shift + 1))

    # step 4: how symmetric is the REAL face, at its own best shift?
    dr_all = [_mirror_diff(cr, d) for d in shifts]
    err_r = torch.stack([(dd.sum((1, 2, 3)) / v.sum((1, 2, 3)).clamp_min(1)) for dd, v in dr_all], 1)   # (B, shifts)
    best = err_r.argmin(1)
    dr = torch.stack([dr_all[int(k)][0][b] for b, k in enumerate(best)])                              # (B,1,h,w)
    # tau = 1: a left/right difference as large as the face's average outline strength lowers the penalty to 37%
    w = torch.exp(-dr / tau)

    # steps 1-3 on the prediction, best of the shifts
    losses = []
    for d in shifts:
        dp, v = _mirror_diff(cp, d)
        losses.append((w * dp).sum((1, 2, 3)) / (w * v).sum((1, 2, 3)).clamp_min(1e-6))
    loss = torch.stack(losses, 1).min(1).values.mean()
    if return_parts:
        return loss, {"weight_mean": w.mean().item(), "real_best_shift": [shifts[int(k)] for k in best]}
    return loss


def strict_symmetry_loss(pred):
    """4b's symmetry loss."""
    return (pred - torch.flip(pred, [3])).abs().mean()
