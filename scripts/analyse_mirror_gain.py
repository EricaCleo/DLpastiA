"""WHEN does the mirror input help? Splits the face-by-face gain of method A over method B (final results) by
   - species (cat / dog / wild)
   - face symmetry: how well the full original face matches its own mirror image (after finding its symmetry axis)
   - axis offset: how far the face's symmetry axis is from the image centre (pixels at 128x128)
   - hole size: fraction of the image that is missing
   - mirror coverage: fraction of hole pixels whose mirrored pixel is visible (1 for one-sided holes; varies for square)
Runs on the CPU in a few minutes (no GPU needed):
    python scripts/analyse_mirror_gain.py mirror_04a stylegan_noninversion --mask one_sided
    python scripts/analyse_mirror_gain.py mirror_04a stylegan_noninversion --mask square --csv outputs/gain_square.csv
Negative 'A-B' LPIPS = A is better. Groups are thirds (low / mid / high) of each property.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--mask", default="one_sided", choices=["one_sided", "centre", "square"])
ap.add_argument("--metric", default="lpips")
ap.add_argument("--dir", default=str(ROOT / "outputs" / "results_final"))
ap.add_argument("--image-size", type=int, default=128)
ap.add_argument("--csv", default=None, help="also write one row per face")
a = ap.parse_args()

from src.data import get_datasets, load_config
from src.gan.inputs import mirror_axis_shift, shift_cols_batch


def load(name):
    r = json.loads((Path(a.dir) / f"{name}__{a.mask}.json").read_text())
    return {i: v for i, v in zip(r["indices"], r["per_image"][a.metric])}


ra, rb = load(a.a), load(a.b)
idx = sorted(set(ra) & set(rb))
cfg = load_config(); cfg["image_size"] = a.image_size
_, ds = get_datasets(cfg, mask_type=a.mask)

rows = []
for start in range(0, len(idx), 64):
    chunk = idx[start:start + 64]
    items = [ds[i] for i in chunk]
    x = torch.stack([t[0] for t in items]); m = torch.stack([t[1] for t in items])
    none = torch.zeros_like(m)
    d = mirror_axis_shift(x, none)                                   # symmetry axis of the FULL face
    fx = shift_cols_batch(torch.flip(x, [3]), d)
    valid = shift_cols_batch(torch.ones_like(m), d)                   # columns that exist after the shift
    sym = 1 - ((x - fx).abs() * valid).sum((1, 2, 3)) / (3 * valid.sum((1, 2, 3))) / 2
    vis = 1 - m
    cover = (m * torch.flip(vis, [3])).sum((1, 2, 3)) / m.sum((1, 2, 3))
    for k, i in enumerate(chunk):
        rows.append(dict(index=i, species=Path(ds.files[i]).parent.name, symmetry=float(sym[k]),
                         axis_offset_px=abs(int(d[k])) / 2, hole_fraction=float(m[k].mean()),
                         mirror_coverage=float(cover[k]), A=ra[i], B=rb[i], diff=ra[i] - rb[i]))
    print(f"\r{len(rows)}/{len(idx)} faces", end="", flush=True)
print()


def stats(ds_):
    n = len(ds_); mu = sum(ds_) / n
    se = math.sqrt(sum((v - mu) ** 2 for v in ds_) / (n - 1) / n) if n > 1 else float("nan")
    p = math.erfc(abs(mu / se) / math.sqrt(2)) if se and se > 0 else float("nan")
    return n, mu, se, p


lower_better = a.metric == "lpips"
print(f"\n{a.a} (A) vs {a.b} (B), {a.mask} holes, {a.metric}: A-B < 0 means A is better" if lower_better else
      f"\n{a.a} (A) vs {a.b} (B), {a.mask} holes, {a.metric}: A-B > 0 means A is better")
n, mu, se, p = stats([r["diff"] for r in rows])
print(f"{'all faces':28s} n={n:5d}  A-B {mu:+.4f} +/- {se:.4f}  p={p:.2g}")


def report(title, groups):
    print(f"\n-- by {title}")
    for label, sel in groups:
        if len(sel) < 2:
            continue
        n, mu, se, p = stats([r["diff"] for r in sel])
        meanA = sum(r["A"] for r in sel) / n; meanB = sum(r["B"] for r in sel) / n
        rel = 100 * mu / meanB if meanB else float("nan")
        print(f"   {label:25s} n={n:5d}  A {meanA:.4f}  B {meanB:.4f}  A-B {mu:+.4f} +/- {se:.4f} ({rel:+.1f}%)  p={p:.2g}")


report("species", [(s, [r for r in rows if r["species"] == s]) for s in sorted({r["species"] for r in rows})])
for key, unit in [("symmetry", ""), ("axis_offset_px", " px"), ("hole_fraction", ""), ("mirror_coverage", "")]:
    vals = sorted(r[key] for r in rows)
    if vals[0] == vals[-1]:
        print(f"\n-- by {key}: the same for every face ({vals[0]:.3g}{unit}), not split")
        continue
    t1, t2 = vals[len(vals) // 3], vals[2 * len(vals) // 3]
    report(key + " (thirds)", [(f"low  (< {t1:.3g}{unit})", [r for r in rows if r[key] < t1]),
                               (f"mid  ({t1:.3g}-{t2:.3g}{unit})", [r for r in rows if t1 <= r[key] < t2]),
                               (f"high (>= {t2:.3g}{unit})", [r for r in rows if r[key] >= t2])])

if a.csv:
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\nwrote", a.csv)
