"""Compare two methods FACE BY FACE on the final results (same faces, same holes), e.g. does the mirror input help?
    python scripts/compare_paired.py mirror_04a stylegan_noninversion
    python scripts/compare_paired.py mirror_04c stylegan_noninversion --mask one_sided --metric lpips
Prints, per mask type: both means, the mean difference A - B with its standard error, a two-sided p-value (paired
test, normal approximation) and on how many faces A is better. For LPIPS lower is better; for PSNR/SSIM higher.
A difference is only worth reporting as real when p is small (below 0.05).
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--mask", choices=["one_sided", "centre", "square"], default=None)
ap.add_argument("--metric", nargs="+", default=["lpips", "psnr", "ssim"])
ap.add_argument("--dir", default=str(ROOT / "outputs" / "results_final"))
a = ap.parse_args()


def load(name, mask):
    p = Path(a.dir) / f"{name}__{mask}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    if "per_image" not in r:
        raise SystemExit(f"{p} has no per-face values: it comes from an old run, evaluate it again")
    return r


for mask in ["one_sided", "centre", "square"]:
    if a.mask and mask != a.mask:
        continue
    ra, rb = load(a.a, mask), load(a.b, mask)
    if ra is None or rb is None:
        print(f"\n{mask}: missing result for {a.a if ra is None else a.b}"); continue
    common = sorted(set(ra["indices"]) & set(rb["indices"]))
    ia = {i: k for k, i in enumerate(ra["indices"])}; ib = {i: k for k, i in enumerate(rb["indices"])}
    print(f"\n=== {mask} holes: {a.a} (A) vs {a.b} (B) on {len(common)} shared faces ===")
    for m in a.metric:
        if m not in ra["per_image"] or m not in rb["per_image"]:
            continue
        va = [ra["per_image"][m][ia[i]] for i in common]; vb = [rb["per_image"][m][ib[i]] for i in common]
        d = [x - y for x, y in zip(va, vb)]
        n = len(d); mean = sum(d) / n
        se = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n) if n > 1 else float("nan")
        z = mean / se if se > 0 else 0.0
        p = math.erfc(abs(z) / math.sqrt(2))
        better = sum((x < 0) if m == "lpips" else (x > 0) for x in d) / n
        good = (mean < 0) if m == "lpips" else (mean > 0)
        verdict = ("A better" if good else "B better") if p < 0.05 else "no clear difference"
        print(f"{m:6s} A {sum(va) / n:.4f}  B {sum(vb) / n:.4f}  A-B {mean:+.4f} +/- {se:.4f}  p={p:.3g}  "
              f"A better on {100 * better:.0f}% of faces  -> {verdict}")
