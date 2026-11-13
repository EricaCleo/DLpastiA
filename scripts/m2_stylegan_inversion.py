"""Method 2: StyleGAN inversion (unsupervised).
    python scripts/m2_stylegan_inversion.py train --max-hours 22 --ckpt checkpoints/g_uncond_128.pt   # unconditional generator
    python scripts/m2_stylegan_inversion.py eval  --ckpt checkpoints/g_uncond_128.pt --mask one_sided --n 100 --steps 500
"""
import argparse

from _common import add_eval_args, add_gan_train_args, gan_train, run_eval

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
t = sub.add_parser("train"); add_gan_train_args(t, "checkpoints/g_uncond_128.pt")
e = sub.add_parser("eval"); add_eval_args(e)
e.add_argument("--ckpt", default="checkpoints/g_uncond_128.pt")
e.add_argument("--steps", type=int, default=500, help="optimisation steps per image")
e.add_argument("--lr", type=float, default=0.05)
a = ap.parse_args()

if a.cmd == "train":
    a.l1 = a.lpips = 0.0
    gan_train("uncond", a)
else:
    from _common import device
    from src.methods.stylegan_inversion import StyleGANInversion
    run_eval(StyleGANInversion.from_ckpt(a.ckpt, device(), steps=a.steps, lr=a.lr, use_lpips=not a.no_lpips), a)
