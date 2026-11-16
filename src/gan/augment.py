"""DiffAugment (Zhao et al., 2020, "Differentiable Augmentation for Data-Efficient GAN Training").

With a small dataset the discriminator memorises the real images (its loss collapses towards 0) and the generator
degrades. DiffAugment applies the SAME kind of random, differentiable augmentation to real and generated images before
the discriminator sees them, which keeps the contest balanced. Policy: colour + translation + cutout.
Images are (B,3,H,W) in [-1,1].
"""
import torch
import torch.nn.functional as F


def _color(x):
    B = x.shape[0]
    x = x + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5)                                  # brightness
    m = x.mean(1, keepdim=True)
    x = (x - m) * (torch.rand(B, 1, 1, 1, device=x.device) * 2) + m                          # saturation
    m = x.mean([1, 2, 3], keepdim=True)
    return (x - m) * (torch.rand(B, 1, 1, 1, device=x.device) + 0.5) + m                     # contrast


def _translation(x, ratio=0.125):
    B, C, H, W = x.shape
    sh, sw = int(H * ratio + 0.5), int(W * ratio + 0.5)
    dy = torch.randint(-sh, sh + 1, (B,), device=x.device)
    dx = torch.randint(-sw, sw + 1, (B,), device=x.device)
    ys = (torch.arange(H, device=x.device)[None, :] + dy[:, None]).clamp(-1, H) + 1          # index into the padded image
    xs = (torch.arange(W, device=x.device)[None, :] + dx[:, None]).clamp(-1, W) + 1
    xp = F.pad(x, (1, 1, 1, 1))                                                             # zero border = out of frame
    return xp[torch.arange(B, device=x.device)[:, None, None], :, ys[:, :, None], xs[:, None, :]].permute(0, 3, 1, 2)


def _cutout(x, ratio=0.5):
    B, C, H, W = x.shape
    ch, cw = int(H * ratio + 0.5), int(W * ratio + 0.5)
    oy = torch.randint(0, H + (1 - ch % 2), (B, 1, 1), device=x.device)
    ox = torch.randint(0, W + (1 - cw % 2), (B, 1, 1), device=x.device)
    yy = torch.arange(H, device=x.device)[None, :, None]
    xx = torch.arange(W, device=x.device)[None, None, :]
    inside = ((yy - oy + ch // 2).clamp(0, H + ch) < ch) & ((yy - oy + ch // 2) >= 0) & \
             ((xx - ox + cw // 2).clamp(0, W + cw) < cw) & ((xx - ox + cw // 2) >= 0)
    return x * (~inside)[:, None].to(x.dtype)


def diff_augment(x):
    return _cutout(_translation(_color(x)))
