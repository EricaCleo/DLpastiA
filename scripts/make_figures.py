"""Report figures. Runs on the LOGIN NODE (CPU only, no Slurm, a minute or two):
    python scripts/make_figures.py
Writes PNGs to outputs/figures/:
  fig1_training_curves.png   training loss over time for every trained model (read from the program-*.log files)
  fig2_lpips_by_method.png    final LPIPS of every method, per hole type, with standard-error bars
  fig3_quality_vs_speed.png   LPIPS against seconds per image (one-sided holes)
  fig4_lpips_vs_hole_size.png how each method gets worse as the (square) hole grows
  fig5_mirror_gain.png        the ablation: gain of 4a / 4b / 4c over plain StyleGAN, per hole type, 95% intervals
  fig6_mirror_guidance.png    method 6: stronger mirror guidance in diffusion -> worse LPIPS and less diversity
Any figure whose data is missing is skipped with a message; the others are still made.
"""
import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--logs", default=str(ROOT / "program-*.log"), help="glob of training logs")
ap.add_argument("--results", default=str(ROOT / "outputs" / "results_final"))
ap.add_argument("--sweep", default=str(ROOT / "outputs" / "results"), help="48-face runs incl. the mirror-guidance sweep")
ap.add_argument("--out", default=str(ROOT / "outputs" / "figures"))
ap.add_argument("--image-size", type=int, default=128)
ap.add_argument("--dpi", type=int, default=200)
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)

# ---------------- style (validated categorical palette, fixed order; one colour per method everywhere) ----------------
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
C = {"mae": "#2a78d6", "stylegan_inversion": "#eb6834", "stylegan_noninversion": "#1baf7a",
     "mirror_04a": "#eda100", "mirror_04b": "#eda100", "mirror_04c": "#eda100",
     "repaint_uvit": "#e87ba4", "mirror_guided_repaint_uvit": "#008300", "mirror_paste": "#b9b8b2", "grey_fill": "#b9b8b2"}
NAME = {"mae": "1  MAE", "stylegan_inversion": "2  StyleGAN inversion", "stylegan_noninversion": "3  StyleGAN non-inversion",
        "mirror_04a": "4a Mirror-conditioned StyleGAN", "mirror_04b": "4b  + symmetry loss", "mirror_04c": "4c  axis-aligned mirror",
        "repaint_uvit": "5  RePaint (U-ViT)", "mirror_guided_repaint_uvit": "6  Mirror-guided RePaint",
        "mirror_paste": "Mirror paste (baseline)", "grey_fill": "Grey fill (baseline)"}
HATCH = {"mirror_04b": "////", "mirror_04c": "...."}
MASKS = [("one_sided", "One-sided holes"), ("square", "Random square holes"), ("centre", "Centre holes")]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "legend.frameon": False, "lines.linewidth": 2})


def save(fig, name):
    fig.savefig(OUT / name, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def smooth(y, w):
    y = np.asarray(y, float)
    if len(y) < 3 or w <= 1:
        return y
    w = min(w, len(y))
    k = np.ones(w) / w
    pad = np.concatenate([np.full(w // 2, y[0]), y, np.full(w - 1 - w // 2, y[-1])])
    return np.convolve(pad, k, mode="valid")


# ======================================= 1. training curves from the logs =======================================
def parse_logs(pattern):
    """Returns {model: {metric: {x: value}}}. Segments start at the 'device ...' header line each training run prints."""
    runs = {}
    gan_re = re.compile(r"^step (\d+) D ([\d.eE+-]+) (.*?) \| [\d.]+ h")
    dif_re = re.compile(r"^step (\d+) loss ([\d.eE+-]+) \(")
    mae_re = re.compile(r"^epoch (\d+)/(\d+) step (\d+)/(\d+) loss ([\d.eE+-]+)")
    for f in sorted(glob.glob(pattern)):
        model = None
        for line in open(f, errors="replace"):
            line = line.strip()
            if line.startswith("device "):
                mm = re.search(r"\| mode (\w+)", line)
                if mm:
                    mode = mm.group(1)
                    sym = float(re.search(r"\| sym ([\d.]+)", line).group(1)) if "| sym" in line else 0.0
                    model = {"uncond": "g_uncond", "comod": "stylegan_noninversion", "mirror_aligned": "mirror_04c"}.get(
                        mode, "mirror_04b" if sym > 0 else "mirror_04a")
                    if "train masks mix" in line:
                        model += "_ft"
                elif "| arch uvit" in line:
                    model = "uvit"
                elif "| arch unet" in line:
                    model = "unet"
                elif "| epochs" in line or "params (M)" in line and "arch" not in line:
                    model = "mae_or_unet"         # resolved by the kind of step line that follows
                continue
            if model is None:
                continue
            g = gan_re.match(line)
            if g and model not in ("uvit", "unet", "mae_or_unet"):
                step = int(g.group(1)); rec = runs.setdefault(model, {})
                rec.setdefault("D", {})[step] = float(g.group(2))
                for k, v in re.findall(r"(\w+) ([\d.eE+-]+)", g.group(3)):
                    rec.setdefault(k, {})[step] = float(v)
                continue
            d = dif_re.match(line)
            if d:
                m = "unet" if model == "mae_or_unet" else model
                runs.setdefault(m, {}).setdefault("loss", {})[int(d.group(1))] = float(d.group(2))
                continue
            e = mae_re.match(line)
            if e:
                ep, _, i, n, loss = e.groups()
                x = int(ep) - 1 + int(i) / max(int(n), 1)
                runs.setdefault("mae", {}).setdefault("loss", {})[round(x, 4)] = float(loss)
    return runs


def series(runs, model, metric):
    d = runs.get(model, {}).get(metric, {})
    if not d:
        return None, None
    xs = np.array(sorted(d)); return xs, np.array([d[x] for x in xs])


def fig_training():
    runs = parse_logs(a.logs)
    if not runs:
        print("fig1: no training lines found in", a.logs, "- skipped"); return
    print("fig1: found", {k: {m: len(v) for m, v in r.items()} for k, r in runs.items()})
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    axs = axs.ravel()

    def single(ax, model, metric, title, xlabel, ylabel, w=25, logy=False):
        x, y = series(runs, model, metric)
        if x is None:
            ax.text(0.5, 0.5, "no log found", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        else:
            ax.plot(x, y, color=AXIS, linewidth=0.6)
            ax.plot(x, smooth(y, w), color=INK2)
            if logy:
                ax.set_yscale("log")
        ax.set_title(title, loc="left"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    single(axs[0], "mae", "loss", "Method 1: MAE (pixel MSE on hidden patches)", "epoch", "loss", w=15)
    single(axs[1], "uvit", "loss", "Methods 5 and 6: diffusion model (U-ViT)", "training step", "noise MSE (log)", w=50, logy=True)

    ax = axs[2]                                     # method 2: unconditional generator, D and G adversarial losses
    for metric, lab, col, ls in [("D", "Discriminator", "#6250d6", "-"), ("adv", "Generator (adversarial)", "#e34948", "--")]:
        x, y = series(runs, "g_uncond", metric)
        if x is not None:
            ax.plot(x, smooth(y, 50), color=col, linestyle=ls, label=lab)
    ax.set_title("Method 2: unconditional StyleGAN (for inversion)", loc="left"); ax.set_xlabel("training step"); ax.set_ylabel("loss")
    if ax.lines:
        ax.legend()

    sup = ["stylegan_noninversion", "mirror_04a", "mirror_04b", "mirror_04c"]
    styles = {"stylegan_noninversion": "-", "mirror_04a": "-", "mirror_04b": "--", "mirror_04c": ":"}
    for ax, metric, title, lab in [(axs[3], "l1", "Methods 3 and 4: hole L1 loss", "L1 (hole)"),
                                   (axs[4], "lpips", "Methods 3 and 4: LPIPS loss", "LPIPS"),
                                   (axs[5], "D", "Methods 3 and 4: discriminator loss", "D loss")]:
        for m in sup:
            x, y = series(runs, m, metric)
            if x is not None:
                ax.plot(x, smooth(y, 80), color=C[m], linestyle=styles[m], label=NAME[m].strip())
        ax.set_title(title, loc="left"); ax.set_xlabel("training step"); ax.set_ylabel(lab)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.suptitle("Training losses over time (faint: raw values logged every 100 steps; bold: moving average)",
                 x=0.01, ha="left", color=INK2, fontsize=9)
    fig.tight_layout()
    save(fig, "fig1_training_curves.png")


# ======================================= results-based figures =======================================
def load_results(d):
    out = {}
    for p in glob.glob(str(Path(d) / "*.json")):
        r = json.loads(Path(p).read_text())
        out[(r["method"], r["mask"])] = r
    return out


R = load_results(a.results)
# the six methods of the report (4c and the baselines appear only in the ablation figure / the table)
METHODS = ["mae", "stylegan_inversion", "stylegan_noninversion", "mirror_04a", "mirror_04b",
           "repaint_uvit", "mirror_guided_repaint_uvit"]


def fig_bars():
    if not R:
        print("fig2: no results in", a.results, "- skipped"); return
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (mask, title) in zip(axs, MASKS):
        ms = [m for m in METHODS if (m, mask) in R]
        y = np.arange(len(ms))[::-1]
        for yi, m in zip(y, ms):
            r = R[(m, mask)]
            ax.barh(yi, r["lpips"], height=0.7, color=C[m], hatch=HATCH.get(m), edgecolor=SURF, linewidth=1)
            ax.errorbar(r["lpips"], yi, xerr=1.96 * r.get("lpips_se", 0), color=INK, capsize=2, linewidth=1)
            ax.text(r["lpips"] + 0.006, yi, f"{r['lpips']:.3f}", va="center", fontsize=8, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in ms])
        ax.set_title(title, loc="left"); ax.set_xlabel("LPIPS in the hole (lower is better)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(R[(m, mask)]["lpips"] for m in ms) * 1.18)
    fig.suptitle("Final results: 1,500 test faces (300 for the slow methods 2, 5, 6); bars = mean, lines = 95% interval",
                 x=0.01, ha="left", color=INK2, fontsize=9)
    fig.tight_layout()
    save(fig, "fig2_lpips_by_method.png")


def fig_speed():
    ms = [m for m in METHODS if (m, "one_sided") in R]
    if not ms:
        print("fig3: no results - skipped"); return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in ms:
        r = R[(m, "one_sided")]
        ax.scatter(r["sec_per_image"], r["lpips"], s=70, color=C[m], edgecolor=SURF, linewidth=2, zorder=3,
                   marker="s" if m in HATCH else "o")
    # labels, nudged apart vertically where points sit close together
    placed = []
    for m in sorted(ms, key=lambda m: R[(m, "one_sided")]["lpips"]):
        r = R[(m, "one_sided")]; ly = r["lpips"]
        for py in placed:
            if abs(ly - py) < 0.006:
                ly = py + 0.006
        placed.append(ly)
        ax.annotate(NAME[m].strip(), (r["sec_per_image"], r["lpips"]), xytext=(r["sec_per_image"] * 1.25, ly),
                    va="center", fontsize=8, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.6) if abs(ly - r["lpips"]) > 1e-9 else None)
    ax.set_xscale("log")
    ax.set_xlabel("seconds per image (log scale, RTX 5090, 3 fills per image)")
    ax.set_ylabel("LPIPS, one-sided holes (lower is better)")
    ax.set_title("Quality against speed", loc="left")
    xs = [R[(m, "one_sided")]["sec_per_image"] for m in ms]
    ax.set_xlim(min(xs) / 2, max(xs) * 12)
    fig.tight_layout()
    save(fig, "fig3_quality_vs_speed.png")


def fig_hole_size():
    mask = "square"
    ms = [m for m in METHODS if (m, mask) in R and "per_image" in R[(m, mask)] and m != "mirror_04b"]
    if not ms:
        print("fig4: no per-face results - skipped"); return
    from src.data import get_datasets, load_config
    cfg = load_config(); cfg["image_size"] = a.image_size
    _, ds = get_datasets(cfg, mask_type=mask)
    frac = {}
    for m in ms:
        for i in R[(m, mask)]["indices"]:
            if i not in frac:
                frac[i] = float(ds._mask(i).mean())          # same seeded mask as in the evaluation, no image loading
    edges = np.quantile(list(frac.values()), np.linspace(0, 1, 6))
    centres = 100 * (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m in ms:
        r = R[(m, mask)]
        f = np.array([frac[i] for i in r["indices"]]); v = np.array(r["per_image"]["lpips"])
        mu, lo, hi = [], [], []
        for b in range(5):
            sel = v[(f >= edges[b]) & ((f < edges[b + 1]) if b < 4 else (f <= edges[b + 1]))]
            se = sel.std(ddof=1) / math.sqrt(len(sel)) if len(sel) > 1 else 0
            mu.append(sel.mean() if len(sel) else np.nan); lo.append(mu[-1] - 1.96 * se); hi.append(mu[-1] + 1.96 * se)
        ls = {"mirror_04b": "--", "mirror_04c": ":"}.get(m, "-")
        ax.plot(centres, mu, color=C[m], linestyle=ls, marker="o", markersize=5, label=NAME[m].strip())
        ax.fill_between(centres, lo, hi, color=C[m], alpha=0.12, linewidth=0)
    ax.set_xlabel("hole size (% of the image, five equal-count groups)")
    ax.set_ylabel("LPIPS in the hole (lower is better)")
    ax.set_title("LPIPS by hole size (random square holes; bands = 95% intervals)", loc="left")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    save(fig, "fig4_lpips_vs_hole_size.png")


def paired(ma, mb, mask, metric="lpips"):
    ra, rb = R.get((ma, mask)), R.get((mb, mask))
    if not ra or not rb or "per_image" not in ra or "per_image" not in rb:
        return None
    ia = dict(zip(ra["indices"], ra["per_image"][metric])); ib = dict(zip(rb["indices"], rb["per_image"][metric]))
    d = np.array([ia[i] - ib[i] for i in sorted(set(ia) & set(ib))])
    if len(d) < 2:
        return None
    mu, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
    base = np.mean([ib[i] for i in sorted(set(ia) & set(ib))])
    return 100 * mu / base, 100 * 1.96 * se / base, math.erfc(abs(mu / se) / math.sqrt(2)) if se > 0 else float("nan")


def fig_gain():
    variants = [m for m in ["mirror_04a", "mirror_04b", "mirror_04c"] if any((m, k) in R for k, _ in MASKS)]
    if not variants or not any(("stylegan_noninversion", k) in R for k, _ in MASKS):
        print("fig5: results missing - skipped"); return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    off = {m: o for m, o in zip(variants, np.linspace(-0.22, 0.22, len(variants)))}
    marker = {"mirror_04a": "o", "mirror_04b": "s", "mirror_04c": "D"}
    for gi, (mask, title) in enumerate(MASKS):
        for m in variants:
            res = paired(m, "stylegan_noninversion", mask)
            if res is None:
                continue
            g, ci, p = res
            ax.errorbar(gi + off[m], -g, yerr=ci, fmt=marker[m], color=C[m], markersize=7, capsize=3,
                        markeredgecolor=INK2, label=NAME[m].strip() if gi == 0 else None)
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.text(gi + off[m], -g + ci + 0.25, star, ha="center", fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(range(len(MASKS))); ax.set_xticklabels([t for _, t in MASKS])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("LPIPS improvement over plain StyleGAN (%)\nabove 0 = mirror version is better")
    ax.set_title("Ablation: LPIPS change of the mirror variants relative to plain StyleGAN (method 3)", loc="left")
    ax.legend(fontsize=8, loc="upper right")
    fig.text(0.01, -0.02, "Paired over 1,500 test faces; bars = 95% intervals.   *** p < 0.001   ** p < 0.01   * p < 0.05   n.s. not significant", color=MUTED, fontsize=8)
    fig.tight_layout()
    save(fig, "fig5_mirror_gain.png")


def fig_guidance():
    S = load_results(a.sweep)
    pts = []
    for (meth, mask), r in S.items():
        mm = re.match(r"mirror_adaptive_uvit_l([\d.]+)_f([\d.]+)_k(\d+)_a(\d)$", meth)
        if mm and mask == "one_sided":
            pts.append((float(mm.group(1)), float(mm.group(2)), int(mm.group(3)), int(mm.group(4)), r))
    base = S.get(("repaint_uvit", "one_sided"))
    if not pts or not base:
        print("fig6: mirror sweep results not found in", a.sweep, "- skipped"); return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    groups = [("plain, guidance in last 20% of steps", lambda p: p[1] == 0.2 and p[2] == 0 and p[3] == 0, "#2a78d6", "o"),
              ("plain, last 50%", lambda p: p[1] == 0.5 and p[2] == 0 and p[3] == 0, "#eb6834", "s"),
              ("feathered + axis-aligned", lambda p: p[2] > 0 and p[3] == 1, "#1baf7a", "D")]
    for ax, key, lab in [(axs[0], "lpips", "LPIPS, one-sided holes (lower is better)"),
                         (axs[1], "diversity", "diversity between fills (higher = more varied)")]:
        ax.axhline(base[key], color=INK2, linewidth=1.2, linestyle="--")
        ax.text(0.205, base[key], " RePaint, no mirror", va="bottom", ha="right", fontsize=8, color=INK2)
        for name, sel, col, mk in groups:
            g = sorted([p for p in pts if sel(p)], key=lambda p: p[0])
            if g:
                ax.plot([p[0] for p in g], [p[4][key] for p in g], color=col, marker=mk, markersize=6, label=name)
        ax.set_xlabel("mirror guidance strength (lambda)"); ax.set_ylabel(lab)
    axs[0].set_title("Method 6: LPIPS against mirror guidance strength", loc="left")
    axs[1].set_title("Method 6: diversity against mirror guidance strength", loc="left")
    axs[0].legend(fontsize=8)
    fig.text(0.01, -0.03, "One-sided holes, 48 test faces, 3 fills each. Dashed line: RePaint without mirror guidance (method 5).",
             color=MUTED, fontsize=8)
    fig.tight_layout()
    save(fig, "fig6_mirror_guidance.png")


for fn in (fig_training, fig_bars, fig_speed, fig_hole_size, fig_gain, fig_guidance):
    try:
        fn()
    except Exception as e:                      # one broken figure must not stop the others
        print(f"{fn.__name__} failed: {type(e).__name__}: {e}")
print("done:", OUT)
