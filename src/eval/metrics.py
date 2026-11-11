"""Shared metrics. All images are tensors in [-1,1]; masks are (B,1,H,W) with 1 = hole.

PSNR / SSIM / LPIPS are computed on the hole region only, since the visible part is copied
from the original and is always perfect.
"""
import numpy as np
import torch
from skimage.metrics import structural_similarity as _ssim

from src.data.masks import composite

_LPIPS = None


def _lpips(device):
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex").to(device).eval()
    return _LPIPS


def psnr_hole(out, gt, mask):
    mse = (((out - gt) ** 2 * mask).sum((1, 2, 3)) / (3 * mask.sum((1, 2, 3)))).clamp_min(1e-10)
    return 10 * torch.log10(4.0 / mse)  # value range is 2 -> MAX^2 = 4


def ssim_hole(out, gt, mask):
    """SSIM map averaged over hole pixels."""
    vals = []
    for o, g, m in zip(out, gt, mask):
        a = ((o + 1) / 2).permute(1, 2, 0).cpu().numpy()
        b = ((g + 1) / 2).permute(1, 2, 0).cpu().numpy()
        _, smap = _ssim(a, b, channel_axis=2, data_range=1.0, full=True)
        mm = m[0].cpu().numpy().astype(bool)
        vals.append(smap[mm].mean())
    return torch.tensor(vals)


@torch.no_grad()
def lpips_hole(out, gt, mask, device="cpu"):
    # zero everything outside the hole in both images so only the hole contributes
    return _lpips(device)((out * mask).to(device), (gt * mask).to(device)).flatten().cpu()


@torch.no_grad()
def hole_metrics(pred, gt, mask, device="cpu", use_lpips=True):
    """Composite pred into gt's visible region, then score the hole. Returns per-image tensors."""
    out = composite(pred, gt, mask)
    res = {"psnr": psnr_hole(out, gt, mask), "ssim": ssim_hole(out, gt, mask)}
    if use_lpips:
        res["lpips"] = lpips_hole(out, gt, mask, device)
    return res


@torch.no_grad()
def diversity(fills, device="cpu"):
    """Mean pairwise LPIPS between the K fills of the same images (higher = more diverse)."""
    lp = _lpips(device)
    d = [lp(fills[i].to(device), fills[j].to(device)).mean().item()
         for i in range(len(fills)) for j in range(i + 1, len(fills))]
    return float(np.mean(d))


def fid(generated_dir, real_dir):
    import torch_fidelity
    r = torch_fidelity.calculate_metrics(input1=str(generated_dir), input2=str(real_dir),
                                         fid=True, verbose=False)
    return r["frechet_inception_distance"]
