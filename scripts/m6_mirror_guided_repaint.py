"""Method 6 (ours): Mirror + diffusion. NO training of its own: it reuses the diffusion model of Method 5.
Version 1, mirror-guided RePaint (--adaptive / --fixed): the flipped visible pixels pull the sampler toward the mirror.
Version 2, mirror-GAN + diffusion refinement (--gan-init): the mirror-conditioned GAN (4a) fills the hole, then RePaint
refines that fill from an intermediate noise level (--start-frac):
    python scripts/m6_mirror_guided_repaint.py eval --ckpt checkpoints/uvit_128.pt --gan-init checkpoints/mirror_04a_128.pt --start-frac 0.3 --mask one_sided
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
e.add_argument("--align", action="store_true", help="shift the mirror axis per face to match its own symmetry axis")
e.add_argument("--max-shift", type=int, default=16, help="largest axis shift searched, in pixels")
e.add_argument("--s-lo", type=float, default=0.0, help="symmetry score that maps to weight 0 (real faces: about 0.65)")
e.add_argument("--s-hi", type=float, default=1.0, help="symmetry score that maps to weight 1 (real faces: about 0.9)")
e.add_argument("--feather", type=int, default=0, help="blur the guidance mask by this many pixels (e.g. 9) to avoid hard seams; 0 = off")
e.add_argument("--guide-frac", type=float, default=1.0, help="fraction of the steps with guidance (1.0 = all steps)")
e.add_argument("--guide-when", default="late", choices=["late", "early"],
               help="late = the LAST guide-frac of the steps (detail); early = the FIRST guide-frac (rough layout), then normal RePaint")
g = e.add_mutually_exclusive_group()
g.add_argument("--adaptive", action="store_true", help="scale guidance by the face's symmetry score (default)")
g.add_argument("--fixed", action="store_true", help="constant guidance strength")
e.add_argument("--gan-init", default=None, help="version 2: checkpoint of the GAN that makes the first fill (e.g. checkpoints/mirror_04a_128.pt)")
e.add_argument("--start-frac", type=float, default=0.3, help="version 2: fraction of the diffusion steps used to refine the GAN fill")
e.add_argument("--name", default=None, help="result name (default mirror_repaint_adaptive / mirror_repaint_fixed / mirror_gan_repaint)")
a = ap.parse_args()

if a.cmd == "train":
    raise SystemExit("Method 6 needs no training: it reuses the model from `python scripts/m5_repaint.py train`.")
from src.diffusion import load_eps_model
from src.methods.mirror_diffusion import MirrorGANRefinedRePaint, MirrorGuidedRePaint
if a.gan_init:
    from src.gan import load_generator
    from src.methods.mirror_conditioned import MirrorConditionedInpainter
    from src.methods.stylegan_noninversion import CoModGANInpainter
    _, gcfg = load_generator(a.gan_init, "cpu")
    G = (MirrorConditionedInpainter if gcfg["mode"] != "comod" else CoModGANInpainter).from_ckpt(a.gan_init, device())
    m = MirrorGANRefinedRePaint(load_eps_model(a.ckpt, device()), G, start_frac=a.start_frac, steps=a.steps,
                                jump_length=a.jump_length, resamples=a.resamples, device=device())
    m.name = a.name or "mirror_gan_repaint"
    run_eval(m, a)
    raise SystemExit
adaptive = not a.fixed
m = MirrorGuidedRePaint(load_eps_model(a.ckpt, device()), lam=a.lam, adaptive=adaptive, guide_frac=a.guide_frac, guide_when=a.guide_when, feather=a.feather, align=a.align, max_shift=a.max_shift, s_lo=a.s_lo, s_hi=a.s_hi, steps=a.steps,
                        jump_length=a.jump_length, resamples=a.resamples, device=device())
m.name = a.name or ("mirror_repaint_adaptive" if adaptive else "mirror_repaint_fixed")
run_eval(m, a)
