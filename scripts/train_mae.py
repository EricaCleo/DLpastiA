"""Fine-tune a pretrained MAE on AFHQ (Method 1). Self-supervised: random 75% patch masking, loss on hidden patches.
    python scripts/train_mae.py --image-size 128 --epochs 40 --batch-size 64 --ckpt checkpoints/mae_128.pt
Pretrained weights must be cached first, on the LOGIN node (compute nodes are offline): python scripts/download_pretrained.py
Resumes automatically from the checkpoint (use after a job time limit). --from-scratch trains without pretrained weights.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from src.data import get_datasets, load_config
from src.methods.mae import build_mae
from src.methods.mae_train import train_mae

ap = argparse.ArgumentParser()
ap.add_argument("--image-size", type=int, default=128, help="multiple of 16")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--pretrained", default="facebook/vit-mae-base")
ap.add_argument("--from-scratch", action="store_true")
ap.add_argument("--ckpt", default="checkpoints/mae_128.pt")
a = ap.parse_args()
assert a.image_size % 16 == 0, "image size must be a multiple of the patch size (16)"

cfg = load_config(); cfg["image_size"] = a.image_size
train_ds, _ = get_datasets(cfg)
loader = DataLoader(train_ds, a.batch_size, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = build_mae(a.image_size, a.pretrained, a.from_scratch)
print("device", dev, "| train images", len(train_ds), "| params (M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
      "| image", a.image_size, "| epochs", a.epochs, "| batch", a.batch_size, "| from scratch" if a.from_scratch else "| pretrained " + a.pretrained)
train_mae(model, loader, a.ckpt, epochs=a.epochs, lr=a.lr, device=dev)
