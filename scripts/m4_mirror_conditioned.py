"""Method 4: Mirror-conditioned StyleGAN (supervised). Variants: 04a = no symmetry loss, 04b = --symmetry 0.1.
    python scripts/m4_mirror_conditioned.py train --ckpt checkpoints/mirror_04a_128.pt
    python scripts/m4_mirror_conditioned.py train --symmetry 0.1 --ckpt checkpoints/mirror_04b_128.pt
    python scripts/m4_mirror_conditioned.py eval  --ckpt checkpoints/mirror_04a_128.pt --name mirror_04a --mask one_sided
"""
import argparse

from _common import add_eval_args, add_gan_train_args, gan_train, run_eval

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
t = sub.add_parser("train"); add_gan_train_args(t, "checkpoints/mirror_04a_128.pt")
t.add_argument("--symmetry", type=float, default=0.0, help="symmetry-loss weight (0 = variant 04a, e.g. 0.1 = 04b)")
e = sub.add_parser("eval"); add_eval_args(e)
e.add_argument("--ckpt", default="checkpoints/mirror_04a_128.pt")
e.add_argument("--name", default="mirror_conditioned", help="result name, e.g. mirror_04a / mirror_04b")
a = ap.parse_args()

if a.cmd == "train":
    gan_train("mirror", a, sym_w=a.symmetry)
else:
    from _common import device
    from src.methods.mirror_conditioned import MirrorConditionedInpainter
    m = MirrorConditionedInpainter.from_ckpt(a.ckpt, device())
    m.name = a.name
    run_eval(m, a)
