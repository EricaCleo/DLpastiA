"""Method 5: RePaint (unsupervised diffusion). `train` trains the unconditional DDPM (also used by Method 6).
    python scripts/m5_repaint.py train --image-size 128 --max-steps 300000 --ckpt checkpoints/ddpm_128.pt
    python scripts/m5_repaint.py eval  --ckpt checkpoints/ddpm_128.pt --mask one_sided --n 48 --K 3 --resamples 5
"""
import argparse

from _common import add_eval_args, device, run_eval

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
t = sub.add_parser("train")
t.add_argument("--image-size", type=int, default=128)
t.add_argument("--batch-size", type=int, default=16)
t.add_argument("--max-steps", type=int, default=300_000)
t.add_argument("--lr", type=float, default=1e-4)
t.add_argument("--save-every", type=int, default=1000)
t.add_argument("--workers", type=int, default=4)
t.add_argument("--amp", action="store_true")
t.add_argument("--channels", default="128,256,256,512")
t.add_argument("--attn-levels", default="3")
t.add_argument("--grad-checkpoint", action="store_true")
t.add_argument("--ckpt", default="checkpoints/ddpm_128.pt")
e = sub.add_parser("eval"); add_eval_args(e)
e.add_argument("--ckpt", default="checkpoints/ddpm_128.pt")
e.add_argument("--steps", type=int, default=250)
e.add_argument("--jump-length", type=int, default=10)
e.add_argument("--resamples", type=int, default=10)
a = ap.parse_args()

if a.cmd == "train":
    from torch.utils.data import DataLoader
    from src.data import get_datasets, load_config
    from src.diffusion import EpsUNet
    from src.diffusion.train import train_ddpm
    cfg = load_config(); cfg["image_size"] = a.image_size
    train_ds, _ = get_datasets(cfg)
    loader = DataLoader(train_ds, a.batch_size, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    model = EpsUNet(image_size=a.image_size, channels=tuple(int(c) for c in a.channels.split(",")),
                    attn_levels=tuple(int(c) for c in a.attn_levels.split(",") if c), grad_checkpoint=a.grad_checkpoint)
    print("device", device(), "| train images", len(train_ds), "| params (M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 1), flush=True)
    train_ddpm(model, loader, a.ckpt, a.max_steps, a.lr, save_every=a.save_every, device=device(), amp=a.amp)
else:
    from src.diffusion import load_eps_model
    from src.methods.repaint import RePaint
    run_eval(RePaint(load_eps_model(a.ckpt, device()), steps=a.steps, jump_length=a.jump_length, resamples=a.resamples, device=device()), a)
