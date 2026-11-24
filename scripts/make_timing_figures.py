"""Figures for the method-6 timing study (stage `timing_study`): WHEN to nudge toward the mirror, and HOW STRONGLY.
Runs on the LOGIN NODE or a laptop (CPU, seconds):
    python scripts/make_timing_figures.py
Reads outputs/results_timing/m6t_*.json and writes to outputs/figures/:
  fig8a_timing_heatmap.png    LPIPS for every schedule x strength; colour = change against plain RePaint (* = significant)
  fig8b_timing_vs_lambda.png  LPIPS, diversity and PSNR against the strength, one line per schedule
  fig8c_timing_tradeoff.png   LPIPS against diversity for every run (the quality / variety trade-off)
  fig8d_timing_best.png       each schedule at its best strength: paired change against RePaint with 95% intervals
  fig8e_timing_examples.png   example faces: RePaint and each schedule at its best strength (needs outputs/samples_timing)
and outputs/timing_summary.csv (next to the figures folder). Trajectory curves (Asyrp figure 3 style) come from scripts/diffusion_trajectory.py.
"""
import argparse
import csv
import glob
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=str(ROOT / "outputs" / "results_timing"))
ap.add_argument("--samples", default=str(ROOT / "outputs" / "samples_timing"))
ap.add_argument("--timeline", default=str(ROOT / "outputs" / "diffusion_timeline.json"))
ap.add_argument("--out", default=str(ROOT / "outputs" / "figures"))
ap.add_argument("--rows", type=int, default=6, help="faces in the example picture")
ap.add_argument("--dpi", type=int, default=200)
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)

try:
    F = json.load(open(a.timeline))["guide_frac"]
except (OSError, KeyError):
    F = None
pct = f"{round(F * 100)}%" if F else "t_edit"
ORDER = ["all", "first50", "last50", "current", "tedit", "asyrp"]
LABEL = {"all": "every step", "first50": "first 50%", "last50": "last 50%", "current": "last 20% (current)",
         "tedit": f"until t_edit, first {pct} (normal update)", "asyrp": f"Asyrp: until t_edit + asymmetric update"}
COL = {"all": "#2a78d6", "first50": "#eb6834", "last50": "#6250d6", "current": "#1baf7a", "tedit": "#eda100",
       "asyrp": "#008300", "repaint": "#e87ba4"}
INK2, MUTED, SURF = "#3d3c39", "#6b6a65", "#ffffff"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5df", "legend.frameon": False, "lines.linewidth": 2.2, "axes.titleweight": "bold",
                     "axes.titlesize": 12, "figure.facecolor": SURF, "axes.facecolor": SURF})

R, LSTR = {}, {}          # LSTR keeps the strength exactly as written in the run name (folder names use it)
for p in glob.glob(str(Path(a.results) / "m6t_*__one_sided.json")):
    r = json.loads(Path(p).read_text())
    if r["method"] == "m6t_repaint":
        R["repaint"] = r; continue
    m = re.match(r"m6t_(\w+?)_l([\d.]+)$", r["method"])
    if m:
        R[(m.group(1), float(m.group(2)))] = r; LSTR[(m.group(1), float(m.group(2)))] = m.group(2)
if "repaint" not in R:
    sys.exit(f"no plain-RePaint run (m6t_repaint) in {a.results}: run the timing_study stage first")
base = R["repaint"]
scheds = [s for s in ORDER if any(k[0] == s for k in R if k != "repaint")]
lams = sorted({k[1] for k in R if k != "repaint"})
print(f"{len(R) - 1} runs, schedules {scheds}, strengths {lams}; RePaint LPIPS {base['lpips']:.4f}")


def paired(r):
    """% change of LPIPS against RePaint (negative = better), its 95% half-width, and the p-value (same faces, same noise)."""
    ia = dict(zip(r["indices"], r["per_image"]["lpips"])); ib = dict(zip(base["indices"], base["per_image"]["lpips"]))
    common = sorted(set(ia) & set(ib))
    d = np.array([ia[i] - ib[i] for i in common]); b = np.mean([ib[i] for i in common])
    mu, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
    return 100 * mu / b, 100 * 1.96 * se / b, (math.erfc(abs(mu / se) / math.sqrt(2)) if se > 0 else 1.0)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def save(fig, name):
    fig.savefig(OUT / name, dpi=a.dpi, bbox_inches="tight", facecolor=SURF); plt.close(fig); print("saved", OUT / name)


# ---------------- summary table ----------------
rows = []
for s in scheds:
    for l in lams:
        r = R.get((s, l))
        if r:
            g, ci, p = paired(r)
            rows.append(dict(schedule=s, lam=l, lpips=r["lpips"], psnr=r["psnr"], ssim=r["ssim"], diversity=r.get("diversity"),
                             change_pct=g, ci_pct=ci, p=p))
with open(OUT.parent / "timing_summary.csv", "w", newline="") as f:          # next to the figures folder
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
best = {s: min((r for r in rows if r["schedule"] == s), key=lambda r: r["lpips"]) for s in scheds}
print(f"\n{'schedule':10s} {'best lambda':>11s} {'LPIPS':>8s} {'vs RePaint':>11s} {'p':>9s} {'diversity':>10s}")
print(f"{'RePaint':10s} {'-':>11s} {base['lpips']:8.4f} {'':>11s} {'':>9s} {base.get('diversity', float('nan')):10.3f}")
for s in scheds:
    b = best[s]
    print(f"{s:10s} {b['lam']:11g} {b['lpips']:8.4f} {b['change_pct']:+10.1f}% {b['p']:9.2g} {b['diversity']:10.3f}")

# ---------------- 8a heatmap ----------------
cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
Z = np.full((len(scheds), len(lams)), np.nan)
for i, s in enumerate(scheds):
    for j, l in enumerate(lams):
        r = R.get((s, l))
        if r:
            Z[i, j] = paired(r)[0]
lim = max(1.0, np.nanmax(np.abs(Z)))
fig, ax = plt.subplots(figsize=(1.6 * len(lams) + 4.5, 0.75 * len(scheds) + 1.8))
im = ax.imshow(Z, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
for i, s in enumerate(scheds):
    for j, l in enumerate(lams):
        r = R.get((s, l))
        if r:
            g, _, p = paired(r)
            ax.text(j, i, f"{r['lpips']:.3f}\n{g:+.1f}%{stars(p)}", ha="center", va="center", fontsize=9.5,
                    color="white" if abs(g) > 0.6 * lim else INK2)
ax.set_xticks(range(len(lams))); ax.set_xticklabels([f"{l:g}" for l in lams])
ax.set_yticks(range(len(scheds))); ax.set_yticklabels([LABEL[s] for s in scheds])
ax.set_xlabel("nudge strength (lambda)"); ax.grid(False)
ax.set_title(f"LPIPS for every schedule and strength (RePaint without nudge: {base['lpips']:.3f})", loc="left")
cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("change against RePaint (%)\nblue = better, red = worse")
fig.text(0.01, -0.04, f"One-sided holes, {base['n_images']} test faces, same random noise in every run. "
         "* p < 0.05, ** p < 0.01, *** p < 0.001 (paired test against RePaint).", color=MUTED, fontsize=9)
fig.tight_layout(); save(fig, "fig8a_timing_heatmap.png")

# ---------------- 8b metrics against strength ----------------
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
for ax, key, yl in [(axs[0], "lpips", "LPIPS (lower is better)"), (axs[1], "diversity", "diversity (higher = more varied)"),
                    (axs[2], "psnr", "PSNR in the hole (higher is better)")]:
    ax.axhline(base[key], color=COL["repaint"], linestyle="--", linewidth=2)
    ax.annotate("RePaint, no nudge", (0, base[key]), xycoords=("axes fraction", "data"), xytext=(6, 4),
                textcoords="offset points", fontsize=9.5, color=INK2)
    for s in scheds:
        pts = [(l, R[(s, l)][key]) for l in lams if (s, l) in R and key in R[(s, l)]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COL[s], marker="o", markersize=6, label=LABEL[s])
    ax.set_xscale("log"); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(lams); ax.set_xlabel("nudge strength (lambda, log scale)"); ax.set_ylabel(yl)
axs[0].set_title("LPIPS against strength", loc="left"); axs[1].set_title("Diversity against strength", loc="left")
axs[2].set_title("PSNR against strength", loc="left")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.14))
fig.tight_layout(w_pad=2.5); save(fig, "fig8b_timing_vs_lambda.png")

# ---------------- 8c trade-off ----------------
fig, ax = plt.subplots(figsize=(9, 5.5))
for s in scheds:
    pts = [(R[(s, l)].get("diversity", np.nan), R[(s, l)]["lpips"], l) for l in lams if (s, l) in R]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=[40 + 160 * (np.log10(p[2]) - np.log10(min(lams))) /
               max(np.log10(max(lams)) - np.log10(min(lams)), 1e-9) for p in pts], color=COL[s], alpha=0.85,
               edgecolor=SURF, linewidth=1.2, label=LABEL[s], zorder=3)
ax.scatter([base.get("diversity", np.nan)], [base["lpips"]], marker="*", s=320, color=COL["repaint"], edgecolor=INK2,
           zorder=4, label="RePaint, no nudge")
ax.set_xlabel("diversity between fills (higher = more varied)"); ax.set_ylabel("LPIPS (lower is better)")
ax.set_title("Quality against variety (larger marker = stronger nudge)", loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9.5)
fig.text(0.01, -0.03, "Best corner: bottom right (low LPIPS, high diversity).", color=MUTED, fontsize=9)
fig.tight_layout(); save(fig, "fig8c_timing_tradeoff.png")

# ---------------- 8d best strength per schedule ----------------
fig, ax = plt.subplots(figsize=(10, 4.8))
for i, s in enumerate(scheds):
    b = best[s]; gain = -b["change_pct"]
    ax.bar(i, gain, width=0.62, color=COL[s], edgecolor=SURF, linewidth=1.5)
    ax.errorbar(i, gain, yerr=b["ci_pct"], color="#0b0b0b", capsize=3, linewidth=1.2)
    ax.text(i, max(gain + b["ci_pct"], 0) + 0.2, f"{gain:+.1f}% {stars(b['p']) or 'n.s.'}\nlambda {b['lam']:g}",
            ha="center", va="bottom", fontsize=9.5, color=INK2)
ax.axhline(0, color=INK2, linewidth=1)
SHORT = {"all": "every\nstep", "first50": "first\n50%", "last50": "last\n50%", "current": "last 20%\n(current)",
         "tedit": f"until t_edit\n({pct}, normal)", "asyrp": "Asyrp\n(t_edit +\nasymmetric)"}
ax.set_xticks(range(len(scheds))); ax.set_xticklabels([SHORT[s] for s in scheds], fontsize=10)
ax.grid(axis="x", visible=False)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0"))
ax.set_ylabel("LPIPS improvement over RePaint")
ax.set_title("Each schedule at its best strength, against RePaint without nudge", loc="left")
lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 1.2)
fig.text(0.01, -0.05, "Above 0: better than RePaint. Paired over the same faces and noise; error bars = 95% intervals. "
         "Note: the best strength is picked on these faces, so treat it as a first look and confirm on new faces.",
         color=MUTED, fontsize=9)
fig.tight_layout(); save(fig, "fig8d_timing_best.png")

# ---------------- 8e examples ----------------
names = ["m6t_repaint"] + [f"m6t_{s}_l{LSTR[(s, best[s]['lam'])]}" for s in scheds]
if all((Path(a.samples) / f"{n}_one_sided").is_dir() for n in names):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_grid.py"), "--mask", "one_sided", "--methods", *names,
                    "--n", str(base["n_images"]), "--rows", str(a.rows), "--samples-dir", a.samples,
                    "--out", str(OUT / "fig8e_timing_examples.png")], check=False)
else:
    print("fig8e: sample folders not found in", a.samples, "- skipped")
