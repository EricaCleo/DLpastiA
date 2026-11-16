"""Print every saved result as one table per mask type (sorted by LPIPS, best first).
    python scripts/show_results.py                  # all masks
    python scripts/show_results.py --mask one_sided
    python scripts/show_results.py --csv results_table.csv      # also write a CSV
    python scripts/show_results.py --dir outputs/results_final  # the final (large) runs; lpips_se = standard error
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--mask", choices=["one_sided", "centre", "square"], default=None)
ap.add_argument("--dir", default=str(ROOT / "outputs" / "results"))
ap.add_argument("--csv", default=None)
a = ap.parse_args()

rows = [json.loads(p.read_text()) for p in sorted(Path(a.dir).glob("*.json"))]
cols = ["psnr", "ssim", "lpips", "lpips_se", "diversity", "sec_per_image", "n_images", "image_size"]
fmt = lambda r, c: ("%.4f" % r[c] if c == "lpips_se" else "%.3f" % r[c] if c in ("ssim", "lpips", "diversity") else "%.2f" % r[c] if c in ("psnr", "sec_per_image")
                    else str(r[c])) if c in r else "-"
for mask in ["one_sided", "centre", "square"]:
    if a.mask and mask != a.mask:
        continue
    rs = sorted([r for r in rows if r.get("mask") == mask], key=lambda r: r.get("lpips", 9))
    if not rs:
        continue
    w = max(len(r["method"]) for r in rs) + 2
    print(f"\n=== {mask} holes ({len(rs)} results, sorted by LPIPS: lower is better) ===")
    print("method".ljust(w) + "".join(c.rjust(14) for c in cols))
    for r in rs:
        print(r["method"].ljust(w) + "".join(fmt(r, c).rjust(14) for c in cols))
if a.csv:
    with open(a.csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "mask"] + cols)
        for r in rows:
            wr.writerow([r["method"], r.get("mask")] + [r.get(c, "") for c in cols])
    print("\nwrote", a.csv)
