"""Method 6 (ours): Mirror-guided RePaint. NO training of its own: it reuses the diffusion model of Method 5.
    python scripts/m6_mirror_guided_repaint.py eval --ckpt checkpoints/ddpm_128.pt --mask one_sided --adaptive --lam 0.1
    python scripts/m6_mirror_guided_repaint.py eval --ckpt checkpoints/ddpm_128.pt --mask one_sided --fixed --lam 0.1
"""
import argparse

from _common import add_eval_args, device, run_eval

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
sub.add_parser("train", help="(no training: train the diffusion model with m5_repaint.py train)")
e = sub.add_parser("eval"); add_eval_args(e)
e.add_argument("--ckpt", default="checkpoints/ddpm_128.pt")
e.add_argument("--steps", type=int, default=250)
e.add_argument("--jump-length", type=int, default=10)
e.add_argument("--resamples", type=int, default=10)
e.add_argument("--lam", type=float, default=0.1, help="guidance strength per step")
e.add_argument("--feather", type=int, default=0, help="blur the guidance mask by this many pixels (e.g. 9) to avoid hard seams; 0 = off")
e.add_argument("--guide-frac", type=float, default=1.0, help="apply guidance only in the last fraction of the steps (1.0 = all steps)")
g = e.add_mutually_exclusive_group()
g.add_argument("--adaptive", action="store_true", help="scale guidance by the face's symmetry score (default)")
g.add_argument("--fixed", action="store_true", help="constant guidance strength")
e.add_argument("--name", default=None, help="result name (default mirror_repaint_adaptive / mirror_repaint_fixed)")
a = ap.parse_args()

if a.cmd == "train":
    raise SystemExit("Method 6 needs no training: it reuses the model from `python scripts/m5_repaint.py train`.")
from src.diffusion import load_eps_model
from src.methods.mirror_diffusion import MirrorGuidedRePaint
adaptive = not a.fixed
m = MirrorGuidedRePaint(load_eps_model(a.ckpt, device()), lam=a.lam, adaptive=adaptive, guide_frac=a.guide_frac, feather=a.feather, steps=a.steps,
                        jump_length=a.jump_length, resamples=a.resamples, device=device())
m.name = a.name or ("mirror_repaint_adaptive" if adaptive else "mirror_repaint_fixed")
run_eval(m, a)
