"""Method 1: MAE (self-supervised ViT). Fine-tunes pretrained ViT-MAE on AFHQ; fills holes by predicting hidden patches.
    python scripts/m1_mae.py train --image-size 128 --epochs 40 --ckpt checkpoints/mae_128.pt   (weights cached on the login node first)
    python scripts/m1_mae.py eval  --ckpt checkpoints/mae_128.pt --mask one_sided --n 100
"""
import argparse

from _common import add_eval_args, device, run_eval

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
t = sub.add_parser("train")
t.add_argument("--image-size", type=int, default=128, help="multiple of 16")
t.add_argument("--epochs", type=int, default=40)
t.add_argument("--batch-size", type=int, default=64)
t.add_argument("--lr", type=float, default=1e-4)
t.add_argument("--workers", type=int, default=4)
t.add_argument("--pretrained", default="facebook/vit-mae-base")
t.add_argument("--from-scratch", action="store_true")
t.add_argument("--ckpt", default="checkpoints/mae_128.pt")
e = sub.add_parser("eval"); add_eval_args(e)
e.add_argument("--ckpt", default="checkpoints/mae_128.pt")
e.add_argument("--name", default=None, help="result name (default: mae), e.g. mae_n48, so it does not overwrite other runs")
a = ap.parse_args()

if a.cmd == "train":
    from torch.utils.data import DataLoader
    from src.data import get_datasets, load_config
    from src.methods.mae import build_mae
    from src.methods.mae_train import train_mae
    assert a.image_size % 16 == 0, "image size must be a multiple of the patch size (16)"
    cfg = load_config(); cfg["image_size"] = a.image_size
    train_ds, _ = get_datasets(cfg)
    loader = DataLoader(train_ds, a.batch_size, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    model = build_mae(a.image_size, a.pretrained, a.from_scratch)
    print("device", device(), "| train images", len(train_ds), "| image", a.image_size, "| epochs", a.epochs, "| batch", a.batch_size, flush=True)
    train_mae(model, loader, a.ckpt, epochs=a.epochs, lr=a.lr, device=device())
else:
    from src.methods.mae import MAEInpainter, load_mae
    m = MAEInpainter(load_mae(a.ckpt, device()), device=device())
    if a.name:
        m.name = a.name
    run_eval(m, a)
