"""Generator inputs for the supervised inpainters (masked image, mask in [-1,1] / {0,1}; mask = 1 in the hole)."""
import torch


def comod_input(masked, mask):
    """Method 3: masked image (3) + mask (1) = 4 channels."""
    return torch.cat([masked, mask], 1)


def mirror_input(masked, mask):
    """Method 4: [masked, mask, flip(masked), flip(mask)] = 8 channels (the mirrored copy sits at the mirrored location)."""
    return torch.cat([masked, mask, torch.flip(masked, [3]), torch.flip(mask, [3])], 1)


def shift_cols_batch(t, d, fill=0.0):
    """Shift the columns of image b right by d[b] pixels (negative = left); pixels shifted in from outside get `fill`."""
    B, C, H, W = t.shape
    src = torch.arange(W, device=t.device)[None, :] - d.to(t.device)[:, None]          # (B, W): source column of each column
    ok = ((src >= 0) & (src < W))[:, None, None, :]
    idx = src.clamp(0, W - 1)[:, None, None, :].expand(B, C, H, W)
    return torch.where(ok, torch.gather(t, 3, idx), torch.full_like(t, fill))


@torch.no_grad()
def mirror_axis_shift(masked, mask, max_shift=16, min_pairs=0.05):
    """Per image: the column shift of the flipped image that makes it match the visible pixels best, i.e. where the
    face's own symmetry axis is (faces are rarely centred exactly). Uses only visible pixels; 0 if too few pairs."""
    vis = 1 - mask
    fx, fv = torch.flip(masked, [3]), torch.flip(vis, [3])
    B, W = masked.shape[0], masked.shape[3]
    errs = []
    for d in range(-max_shift, max_shift + 1):
        sx, sv = torch.zeros_like(fx), torch.zeros_like(fv)
        if d >= 0:
            sx[..., d:], sv[..., d:] = fx[..., :W - d], fv[..., :W - d]
        else:
            sx[..., :W + d], sv[..., :W + d] = fx[..., -d:], fv[..., -d:]
        pair = vis * sv
        n = pair.sum((1, 2, 3))
        e = ((masked - sx).abs() * pair).sum((1, 2, 3)) / (3 * n).clamp_min(1)
        errs.append(torch.where(n >= min_pairs * vis[0].numel(), e, torch.full_like(e, float("inf"))))
    errs = torch.stack(errs, 1)                                                          # (B, 2*max_shift+1)
    best = errs.argmin(1) - max_shift
    return torch.where(torch.isinf(errs.min(1).values), torch.zeros_like(best), best)


def aligned_mirror_input(masked, mask):
    """Method 4c: like mirror_input, but the flipped copy is first shifted so that it is mirrored about the FACE's
    symmetry axis instead of the image centre. Columns that fall outside the picture are marked as hole."""
    d = mirror_axis_shift(masked, mask)
    fm = shift_cols_batch(torch.flip(masked, [3]), d, 0.0)
    fk = shift_cols_batch(torch.flip(mask, [3]), d, 1.0)
    return torch.cat([masked, mask, fm * (1 - fk), fk], 1)


INPUTS = {"comod": (comod_input, 4), "mirror": (mirror_input, 8), "mirror_aligned": (aligned_mirror_input, 8)}
