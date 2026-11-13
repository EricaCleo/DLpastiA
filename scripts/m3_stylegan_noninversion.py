"""Method 3: StyleGAN non-inversion, CoModGAN-style (supervised).
    python scripts/m3_stylegan_noninversion.py train --max-hours 22 --ckpt checkpoints/comod_128.pt
    python scripts/m3_stylegan_noninversion.py eval  --ckpt checkpoints/comod_128.pt --mask one_sided --n 100
"""
import argparse

from _common import add_eval_args, add_gan_train_args, gan_train, run_eval

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
t = sub.add_parser("train"); add_gan_train_args(t, "checkpoints/comod_128.pt")
e = sub.add_parser("eval"); add_eval_args(e)
e.add_argument("--ckpt", default="checkpoints/comod_128.pt")
a = ap.parse_args()

if a.cmd == "train":
    gan_train("comod", a)
else:
    from _common import device
    from src.methods.stylegan_noninversion import CoModGANInpainter
    run_eval(CoModGANInpainter.from_ckpt(a.ckpt, device()), a)
