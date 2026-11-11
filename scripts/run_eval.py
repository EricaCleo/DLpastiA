"""Evaluate a method on the fixed test set.
    python scripts/run_eval.py --method mirror_paste --n 100 --mask square
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.eval import evaluate
from src.methods import GreyFill, MirrorPaste

BASELINES = {"grey_fill": GreyFill, "mirror_paste": MirrorPaste}
METHODS = list(BASELINES) + ["repaint", "mirror_repaint_fixed", "mirror_repaint_adaptive"]

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True, choices=METHODS)
ap.add_argument("--mask", default="square", choices=["square", "one_sided", "centre"])
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--no-lpips", action="store_true")
ap.add_argument("--ckpt", default="checkpoints/ddpm.pt", help="diffusion checkpoint (repaint methods)")
ap.add_argument("--steps", type=int, default=250)
ap.add_argument("--jump-length", type=int, default=10)
ap.add_argument("--resamples", type=int, default=10)
ap.add_argument("--lam", type=float, default=0.1)
ap.add_argument("--K", type=int, default=None)
ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
if a.method in BASELINES:
    method = BASELINES[a.method]()
else:
    from src.diffusion import load_eps_model
    from src.methods.mirror_diffusion import MirrorGuidedRePaint
    from src.methods.repaint import RePaint
    model = load_eps_model(a.ckpt, dev)
    kw = dict(steps=a.steps, jump_length=a.jump_length, resamples=a.resamples, device=dev)
    if a.method == "repaint":
        method = RePaint(model, **kw)
    else:
        method = MirrorGuidedRePaint(model, lam=a.lam, adaptive=a.method.endswith("adaptive"), **kw)
        method.name = a.method
res = evaluate(method, a.mask, a.n, batch_size=a.batch, K=a.K, device=dev, use_lpips=not a.no_lpips)
print(a.method, a.mask, res)
from src.eval import save_result
save_result(method.name, a.mask, res)
