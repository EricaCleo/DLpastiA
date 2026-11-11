"""Data preprocessing: AFHQ dataset, resizing, [-1,1] normalisation, masks.

Train masks are random each time an image is loaded.
Test masks are fixed by seed (and image index), so every method sees the same holes.
"""
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .masks import centre_mask, one_sided_mask, random_square_mask

ROOT = Path(__file__).resolve().parents[2]
EXTS = {".jpg", ".jpeg", ".png"}


def load_config(path=None):
    return yaml.safe_load(open(path or ROOT / "configs" / "data.yaml"))


def list_images(raw_dir, split, species=None):
    """Sorted image paths for a split. species: None or 'cat'/'dog'/'wild'."""
    base = Path(raw_dir) / split
    if not base.is_absolute():
        base = ROOT / base
    dirs = [base / species] if species else sorted(p for p in base.iterdir() if p.is_dir())
    return sorted(f for d in dirs for f in d.iterdir() if f.suffix.lower() in EXTS)


def to_tensor(img, size):
    """PIL -> float tensor (3,H,W) in [-1,1]:  x' = x / 127.5 - 1."""
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float()
    return x / 127.5 - 1


def to_image(x):
    """Tensor (3,H,W) in [-1,1] -> uint8 HxWx3 array."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


class AFHQInpainting(Dataset):
    """Returns (x, mask, masked) with x:(3,H,W) in [-1,1], mask:(1,H,W), masked=(1-mask)*x.

    mask_type: 'square' (default), 'one_sided', or 'centre'.
    """

    def __init__(self, files, size=256, train=True, ratio_range=(0.10, 0.40),
                 seed=1234, mask_type="square"):
        self.files, self.size, self.train = files, size, train
        self.ratio_range, self.seed, self.mask_type = ratio_range, seed, mask_type

    def __len__(self):
        return len(self.files)

    def _mask(self, i):
        rng = np.random.default_rng(None if self.train else self.seed + i)
        if self.mask_type == "centre":
            return centre_mask(self.size)
        if self.mask_type == "one_sided":
            return one_sided_mask(self.size, side=("left", "right")[i % 2], rng=rng)
        return random_square_mask(self.size, self.ratio_range, rng)

    def __getitem__(self, i):
        x = to_tensor(Image.open(self.files[i]), self.size)
        m = torch.from_numpy(self._mask(i))[None]
        return x, m, (1 - m) * x


def get_datasets(cfg=None, mask_type="square", species=None):
    cfg = cfg or load_config()
    tr = list_images(cfg["raw_dir"], cfg["train_split"], species)
    te = list_images(cfg["raw_dir"], cfg["test_split"], species)
    kw = dict(size=cfg["image_size"], ratio_range=tuple(cfg["mask_ratio"]), seed=cfg["test_seed"])
    return (AFHQInpainting(tr, train=True, **kw),
            AFHQInpainting(te, train=False, mask_type=mask_type, **kw))


def get_loaders(batch_size=16, num_workers=2, cfg=None, **kw):
    train_ds, test_ds = get_datasets(cfg, **kw)
    return (DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
            DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers))
