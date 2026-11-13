"""Shared helpers for the per-method scripts (m1..m6). Each method script has `train` and/or `eval` commands."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def add_eval_args(p):
    p.add_argument("--mask", default="square", choices=["square", "one_sided", "centre"])
    p.add_argument("--n", type=int, default=100, help="number of test images")
    p.add_argument("--K", type=int, default=None, help="fills per image for stochastic methods")
    p.add_argument("--image-size", type=int, default=128, help="must match the trained model")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--save-dir", default=None, help="save composited fills (first sample) as PNGs")


def run_eval(method, a):
    from src.data import load_config
    from src.eval import evaluate, save_result
    cfg = load_config()
    cfg["image_size"] = a.image_size
    res = evaluate(method, a.mask, a.n, batch_size=a.batch, K=a.K, device=device(), use_lpips=not a.no_lpips,
                   save_dir=a.save_dir, cfg=cfg)
    res["image_size"] = a.image_size
    print(method.name, a.mask, res)
    save_result(method.name, a.mask, res)
    return res


def add_gan_train_args(p, ckpt):
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-hours", type=float, default=22.0, help="wall-clock training budget (resumable)")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--r1", type=float, default=1.0, help="R1 gamma")
    p.add_argument("--l1", type=float, default=1.0)
    p.add_argument("--lpips", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--base", type=int, default=16384, help="channel base (width); smaller = faster/lighter")
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--ckpt", default=ckpt)


def gan_train(mode, a, sym_w=0.0):
    from torch.utils.data import DataLoader
    from src.data import get_datasets, load_config
    from src.gan import train_gan
    cfg = load_config()
    cfg["image_size"] = a.image_size
    train_ds, _ = get_datasets(cfg)
    loader = DataLoader(train_ds, a.batch_size, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    gcfg = dict(mode=mode, size=a.image_size, z_dim=512, w_dim=512, base=a.base, cap=512, map_layers=4)
    dev = device()
    print("device", dev, "| mode", mode, "| train images", len(train_ds), "| size", a.image_size, "| batch", a.batch_size,
          "| budget h", a.max_hours, "| sym", sym_w, flush=True)
    sample_dir = Path(a.ckpt).with_suffix("").as_posix() + "_samples"
    train_gan(gcfg, loader, a.ckpt, max_hours=a.max_hours, device=dev, lr=a.lr, r1_gamma=a.r1, l1_w=a.l1,
              lpips_w=a.lpips, sym_w=sym_w, flip=not a.no_flip, sample_dir=sample_dir)
