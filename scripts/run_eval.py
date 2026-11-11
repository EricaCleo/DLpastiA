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

METHODS = {"grey_fill": GreyFill, "mirror_paste": MirrorPaste}  # add trained methods as they're ready

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True, choices=METHODS)
ap.add_argument("--mask", default="square", choices=["square", "one_sided", "centre"])
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--no-lpips", action="store_true")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(a.method, a.mask, evaluate(METHODS[a.method](), a.mask, a.n, device=dev, use_lpips=not a.no_lpips))
