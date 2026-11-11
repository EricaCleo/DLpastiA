"""Train the unconditional diffusion model used by RePaint (Methods 5 and 6). Run via sbatch.
    python scripts/train_ddpm.py --image-size 128 --max-steps 100000 --ckpt checkpoints/ddpm_128.pt
Resumes automatically if the checkpoint exists (use after a job time limit).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from src.data import get_datasets, load_config
from src.diffusion import EpsUNet
from src.diffusion.train import train_ddpm

ap = argparse.ArgumentParser()
ap.add_argument("--image-size", type=int, default=128)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--max-steps", type=int, default=100_000)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--save-every", type=int, default=2000)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--amp", action="store_true")
ap.add_argument("--ckpt", default="checkpoints/ddpm.pt")
a = ap.parse_args()

cfg = load_config(); cfg["image_size"] = a.image_size
train_ds, _ = get_datasets(cfg)
loader = DataLoader(train_ds, a.batch_size, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, "| train images", len(train_ds))
train_ddpm(EpsUNet(image_size=a.image_size), loader, a.ckpt, a.max_steps, a.lr, save_every=a.save_every, device=dev, amp=a.amp)
