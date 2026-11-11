"""Run any Inpainter over the fixed test set and report metrics (same masks for all methods)."""
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data import composite, get_datasets, load_config, to_image
from .metrics import diversity, hole_metrics


@torch.no_grad()
def evaluate(method, mask_type="square", n_images=None, batch_size=8, K=None, device="cpu",
             save_dir=None, use_lpips=True, cfg=None):
    """method: object with .fill(masked, mask, K) -> list of K tensors (B,3,H,W) in [-1,1]."""
    from torch.utils.data import DataLoader, Subset
    cfg = cfg or load_config()
    K = K or (cfg["num_samples_K"] if method.stochastic else 1)
    _, test_ds = get_datasets(cfg, mask_type=mask_type)
    if n_images:
        test_ds = Subset(test_ds, list(range(0, len(test_ds), len(test_ds) // n_images))[:n_images])
    acc, div, t0, n = {}, [], time.time(), 0
    for x, m, xm in DataLoader(test_ds, batch_size=batch_size):
        fills = method.fill(xm.to(device), m.to(device), K)
        fills = [f.cpu() for f in fills]
        for f in fills:
            for k, v in hole_metrics(f, x, m, device, use_lpips).items():
                acc.setdefault(k, []).append(v)
        if len(fills) > 1 and use_lpips:
            div.append(diversity(fills, device))
        if save_dir:
            out = Path(save_dir); out.mkdir(parents=True, exist_ok=True)
            for b in range(x.shape[0]):
                img = composite(fills[0][b], x[b], m[b])
                Image.fromarray(to_image(img)).save(out / f"{n + b:05d}.png")
        n += x.shape[0]
    res = {k: torch.cat(v).mean().item() for k, v in acc.items()}
    if div:
        res["diversity"] = float(np.mean(div))
    res["sec_per_image"] = (time.time() - t0) / n
    res["n_images"] = n
    return res
