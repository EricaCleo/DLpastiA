"""Report figures. Runs on the LOGIN NODE or a laptop (CPU only, no Slurm, a minute or two):
    python scripts/make_figures.py
Writes PNGs to outputs/figures/:
  fig1a_training_diffusion.png     training loss over time: diffusion model (methods 2, 2a)
  fig1b_training_supervised.png    training loss over time: StyleGAN (1) and its symmetry versions (1a, 1b, 1c)
  fig2_lpips_by_method.png         final LPIPS of the six methods, per hole type, with 95% intervals
  fig3_quality_vs_speed.png        LPIPS against seconds per image (one-sided holes)
  fig4_lpips_vs_hole_size.png      LPIPS for small to large square holes
  fig5_mirror_gain.png             ablation: LPIPS improvement of 1a / 1b / 1c over plain StyleGAN (1), per hole type
  fig6_mirror_guidance.png         method 2a: LPIPS and diversity at different mirror-nudge strengths
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
from matplotlib.ticker import FuncFormatter

ap = argparse.ArgumentParser()
ap.add_argument("--logs", default=str(ROOT / "program-*.log"), help="glob of training logs")
ap.add_argument("--results", default=str(ROOT / "outputs" / "results_final"))
ap.add_argument("--sweep", default=str(ROOT / "outputs" / "results"), help="48-face runs incl. the mirror-guidance sweep")
ap.add_argument("--out", default=str(ROOT / "outputs" / "figures"))
ap.add_argument("--image-size", type=int, default=128)
ap.add_argument("--dpi", type=int, default=200)
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)

# ---------------- style: one fixed colour per method in every figure (validated categorical palette) ----------------
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#3d3c39", "#6b6a65", "#e6e5df", "#b9b8b2", "#ffffff"
C = {"mae": "#2a78d6", "stylegan_inversion": "#eb6834", "stylegan_noninversion": "#1baf7a",
     "mirror_04a": "#eda100", "mirror_04b": "#b37900", "mirror_04c": "#f5c95c", "mirror_04c_tol": "#f5c95c",
     "repaint_uvit": "#e87ba4", "mirror_guided_repaint_uvit": "#008300"}
NAME = {"mae": "MAE (1)", "stylegan_inversion": "StyleGAN inversion (2)", "stylegan_noninversion": "StyleGAN (1)",
        "mirror_04a": "StyleGAN + mirror input (1a)", "mirror_04b": "StyleGAN + strict symmetry loss (1b)",
        "mirror_04c": "Mirror StyleGAN, aligned axis (old)", "mirror_04c_tol": "StyleGAN + tolerant symmetry loss (1c)",
        "repaint_uvit": "RePaint (2)", "mirror_guided_repaint_uvit": "RePaint + late mirror nudge (2a)"}
MASKS = [("one_sided", "One-sided holes"), ("square", "Random square holes"), ("centre", "Centre holes")]
# the methods of the report (4c = tolerant symmetry loss; the earlier aligned-mirror variant is not shown)
METHODS = ["stylegan_noninversion", "mirror_04a", "mirror_04b", "mirror_04c_tol",
           "repaint_uvit", "mirror_guided_repaint_uvit"]          # MAE (1) and StyleGAN inversion (2) were dropped          # MAE was dropped from the report
plt.rcParams.update({"font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "axes.titlepad": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "legend.frameon": False, "legend.fontsize": 10, "lines.linewidth": 2.2})
KSTEPS = FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if abs(v) >= 1000 else f"{v:.0f}")   # 120000 -> 120k


def save(fig, name):
    fig.savefig(OUT / name, dpi=a.dpi, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("saved", OUT / name)


def smooth(y, w):
    y = np.asarray(y, float)
    if len(y) < 3 or w <= 1:
        return y
    w = min(w, len(y))
    pad = np.concatenate([np.full(w // 2, y[0]), y, np.full(w - 1 - w // 2, y[-1])])
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def trend_ylim(ax, ys, skip=0.03):
    """y-range from the smoothed curves, ignoring the very first steps (where losses start very high)."""
    vals = np.concatenate([y[int(len(y) * skip):] for y in ys if len(y)])
    if len(vals):
        lo, hi = vals.min(), vals.max(); pad = 0.08 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad)


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
                    if "| sym mode tolerant" in line:
                        model = "mirror_04c_tol"
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

    # ---- 1a: the diffusion model (methods 2 and 2a)
    x, y = series(runs, "uvit", "loss")
    if x is None:
        print("fig1a: no diffusion training log found - skipped")
    else:
        fig, ax = plt.subplots(figsize=(7, 4.3))
        ys = smooth(y, 60); ax.plot(x, ys, color=C["repaint_uvit"]); trend_ylim(ax, [ys])
        ax.set_title("Diffusion model, U-ViT (methods 2, 2a)", loc="left"); ax.set_xlabel("training step")
        ax.set_ylabel("noise-prediction loss"); ax.xaxis.set_major_formatter(KSTEPS)
        fig.tight_layout()
        save(fig, "fig1a_training_diffusion.png")

    # ---- 1b: the supervised GANs (same training length, so the curves can be compared directly)
    sup = [m for m in ["stylegan_noninversion", "mirror_04a", "mirror_04b", "mirror_04c_tol"] if m in runs]
    if not sup:
        print("fig1b: no supervised GAN logs - skipped"); return
    ls = {"stylegan_noninversion": "-", "mirror_04a": "-", "mirror_04b": "--", "mirror_04c": ":", "mirror_04c_tol": ":"}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, metric, title, yl in [(axs[0], "l1", "Hole L1 loss", "L1 in the hole"),
                                  (axs[1], "lpips", "LPIPS loss", "LPIPS"),
                                  (axs[2], "D", "Discriminator loss", "discriminator loss")]:
        ys_all = []
        for m in sup:
            x, y = series(runs, m, metric)
            if x is not None:
                ys = smooth(y, 100); ys_all.append(ys)
                ax.plot(x, ys, color=C[m], linestyle=ls[m], label=NAME[m])
        trend_ylim(ax, ys_all)
        ax.set_title(title, loc="left"); ax.set_xlabel("training step"); ax.set_ylabel(yl)
        ax.xaxis.set_major_formatter(KSTEPS)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(w_pad=3)
    save(fig, "fig1b_training_supervised.png")


# ======================================= results-based figures =======================================
def load_results(d):
    out = {}
    for p in glob.glob(str(Path(d) / "*.json")):
        r = json.loads(Path(p).read_text())
        out[(r["method"], r["mask"])] = r
    return out


R = load_results(a.results)


def fig_bars():
    if not R:
        print("fig2: no results in", a.results, "- skipped"); return
    xmax = max(R[(m, k)]["lpips"] + 1.96 * R[(m, k)].get("lpips_se", 0) for m in METHODS for k, _ in MASKS if (m, k) in R)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, sharex=True)
    for ax, (mask, title) in zip(axs, MASKS):
        ms = [m for m in METHODS if (m, mask) in R]
        y = np.arange(len(ms))[::-1]
        for yi, m in zip(y, ms):
            r = R[(m, mask)]; ci = 1.96 * r.get("lpips_se", 0)
            ax.barh(yi, r["lpips"], height=0.68, color=C[m], edgecolor=SURF, linewidth=1.5)
            ax.errorbar(r["lpips"], yi, xerr=ci, color=INK, capsize=3, linewidth=1.2)
            ax.text(r["lpips"] + ci + 0.006, yi, f"{r['lpips']:.3f}", va="center", fontsize=10, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in ms])
        ax.set_title(title, loc="left"); ax.set_xlabel("LPIPS in the hole (lower is better)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, xmax * 1.22)
    fig.tight_layout(w_pad=2)
    if any(("mirror_guided_repaint_uvit", k) in R for k, _ in MASKS):
        fig.text(0.01, -0.04, "2a shown with the nudge in the last 20% of steps (lambda = 0.02). In the timing study, nudging early "
                 "instead ties RePaint (2); no timing beats it.\nError bars = 95% intervals. StyleGANs: 1,500 test faces; "
                 "RePaint: 300 test faces.", color=MUTED, fontsize=9)
    save(fig, "fig2_lpips_by_method.png")


def fig_speed():
    ms = [m for m in METHODS if (m, "one_sided") in R]
    if not ms:
        print("fig3: no results - skipped"); return
    fig, ax = plt.subplots(figsize=(8.5, 5))
    gans = [m for m in ("stylegan_noninversion", "mirror_04a", "mirror_04b", "mirror_04c_tol") if m in ms]
    for m in ms:
        r = R[(m, "one_sided")]
        if m in gans:
            continue
        ax.scatter(r["sec_per_image"], r["lpips"], s=110, color=C[m], edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.annotate(NAME[m], (r["sec_per_image"], r["lpips"]), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK2)
    if gans:      # the four StyleGANs sit on the same spot: one marker group, one label
        for k, m in enumerate(gans):
            r = R[(m, "one_sided")]
            ax.scatter(r["sec_per_image"], r["lpips"], s=110, color=C[m], edgecolor=SURF, linewidth=1.5, zorder=3 + k)
        r = R[(gans[0], "one_sided")]
        lp = " / ".join(f"{R[(m, 'one_sided')]['lpips']:.3f}" for m in gans)
        ax.annotate("StyleGAN (1, 1a, 1b, 1c)\nLPIPS " + lp, (r["sec_per_image"], r["lpips"]), xytext=(12, 14),
                    textcoords="offset points", va="bottom", fontsize=10, color=INK2)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
    ax.set_xlabel("time per image (log scale; GPU, 3 fills per image)")
    ax.set_ylabel("LPIPS, one-sided holes (lower is better)")
    ax.set_title("Quality against speed", loc="left")
    xs = [R[(m, "one_sided")]["sec_per_image"] for m in ms]
    ax.set_xlim(min(xs) / 2, max(xs) * 15)
    ys = [R[(m, "one_sided")]["lpips"] for m in ms]
    ax.set_ylim(min(ys) - 0.015, max(ys) + 0.015)
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
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in ms:
        r = R[(m, mask)]
        f = np.array([frac[i] for i in r["indices"]]); v = np.array(r["per_image"]["lpips"])
        mu, lo, hi = [], [], []
        for b in range(5):
            sel = v[(f >= edges[b]) & ((f < edges[b + 1]) if b < 4 else (f <= edges[b + 1]))]
            se = sel.std(ddof=1) / math.sqrt(len(sel)) if len(sel) > 1 else 0
            mu.append(sel.mean() if len(sel) else np.nan); lo.append(mu[-1] - 1.96 * se); hi.append(mu[-1] + 1.96 * se)
        ax.plot(centres, mu, color=C[m], marker="o", markersize=6, label=NAME[m])
        ax.fill_between(centres, lo, hi, color=C[m], alpha=0.12, linewidth=0)
    ax.set_xlabel("hole size (% of the image; five groups of equal size)")
    ax.set_ylabel("LPIPS in the hole (lower is better)")
    ax.set_title("LPIPS by hole size, random square holes", loc="left")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.text(0.01, -0.02, "Shaded bands: 95% intervals. StyleGAN (1) and StyleGAN + mirror input (1a) lie on top of each other.",
             color=MUTED, fontsize=9)
    fig.tight_layout()
    save(fig, "fig4_lpips_vs_hole_size.png")


def paired(ma, mb, mask, metric="lpips"):
    ra, rb = R.get((ma, mask)), R.get((mb, mask))
    if not ra or not rb or "per_image" not in ra or "per_image" not in rb:
        return None
    ia = dict(zip(ra["indices"], ra["per_image"][metric])); ib = dict(zip(rb["indices"], rb["per_image"][metric]))
    common = sorted(set(ia) & set(ib))
    d = np.array([ia[i] - ib[i] for i in common])
    if len(d) < 2:
        return None
    mu, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
    base = np.mean([ib[i] for i in common])
    return 100 * mu / base, 100 * 1.96 * se / base, math.erfc(abs(mu / se) / math.sqrt(2)) if se > 0 else float("nan")


def fig_gain():
    variants = [m for m in ["mirror_04a", "mirror_04b", "mirror_04c_tol"] if any((m, k) in R for k, _ in MASKS)]
    if not variants or not any(("stylegan_noninversion", k) in R for k, _ in MASKS):
        print("fig5: results missing - skipped"); return
    fig, ax = plt.subplots(figsize=(11, 5.2))
    wbar = 0.32
    tops, bots = [0.0], [0.0]
    for gi, (mask, _) in enumerate(MASKS):
        for k, m in enumerate(variants):
            res = paired(m, "stylegan_noninversion", mask)
            if res is None:
                continue
            g, ci, p = res
            gain = -g                                            # lower LPIPS = improvement, shown as a positive bar
            x = gi + (k - (len(variants) - 1) / 2) * wbar
            ax.bar(x, gain, width=wbar * 0.92, color=C[m], edgecolor=SURF, linewidth=1.5,
                   label=NAME[m] if gi == 0 else None)
            ax.errorbar(x, gain, yerr=ci, color=INK, capsize=3, linewidth=1.2)
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ytxt = max(gain + ci, 0) + 0.15
            ax.text(x, ytxt, f"{gain:+.1f}%\n{star}", ha="center", va="bottom", fontsize=9.5, color=INK2, linespacing=1.1)
            tops.append(ytxt + 0.9); bots.append(min(gain - ci, 0) - 0.3)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_ylim(min(bots), max(tops))
    ax.set_xticks(range(len(MASKS))); ax.set_xticklabels([t for _, t in MASKS], fontsize=11)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0"))
    ax.set_ylabel("LPIPS improvement over StyleGAN (1)")
    ax.set_title("Ablation: effect of symmetry, relative to StyleGAN (1)", loc="left")
    ax.legend(loc="upper right")
    fig.text(0.01, -0.03, "Above 0: the mirror version is better. Paired comparison over 1,500 test faces; error bars = 95% intervals.\n"
             "*** p < 0.001   ** p < 0.01   * p < 0.05   n.s. = not significant", color=MUTED, fontsize=9)
    fig.tight_layout()
    save(fig, "fig5_mirror_gain.png")


def fig_guidance():
    S = load_results(a.sweep)
    pts = []
    for (meth, mask), r in S.items():
        mm = re.match(r"mirror_adaptive_uvit_l([\d.]+)_f([\d.]+)_k(\d+)_a(\d)$", meth)
        if mm and mask == "one_sided":
            pts.append((float(mm.group(1)), float(mm.group(2)), int(mm.group(3)), int(mm.group(4)), r))
        me = re.match(r"mirror_early_uvit_l([\d.]+)_f([\d.]+)$", meth)
        if me and mask == "one_sided":
            pts.append((float(me.group(1)), float(me.group(2)), 0, 9, r))        # align flag 9 = early guidance
    base = S.get(("repaint_uvit", "one_sided"))
    if not pts or not base:
        print("fig6: mirror sweep results not found in", a.sweep, "- skipped"); return
    groups = [("nudge in the last 20% of steps", lambda p: p[1] == 0.2 and p[2] == 0 and p[3] == 0, "#2a78d6", "o", "-"),
              ("nudge in the last 50% of steps", lambda p: p[1] == 0.5 and p[2] == 0 and p[3] == 0, "#6250d6", "s", "--"),
              ("with softened edges + aligned axis", lambda p: p[2] > 0 and p[3] == 1 and p[1] <= 0.5, "#008300", "D", ":")]
    early_cols = ["#eb6834", "#e34948", "#eda100", "#1baf7a", "#e87ba4"]
    for k, f in enumerate(sorted({p[1] for p in pts if p[3] == 9})):       # one line per EARLY guidance range
        groups.append((f"nudge in the first {round(f * 100)}% of steps", lambda p, f=f: p[3] == 9 and p[1] == f,
                       early_cols[k % len(early_cols)], "^", "-"))
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, key, yl, title in [(axs[0], "lpips", "LPIPS, one-sided holes (lower is better)", "2a: LPIPS against nudge strength"),
                               (axs[1], "diversity", "diversity between fills (higher = more varied)", "2a: diversity against nudge strength")]:
        ax.axhline(base[key], color=C["repaint_uvit"], linewidth=2.2, linestyle="-")
        ax.annotate("RePaint, no nudge (2)", (0, base[key]), xycoords=("axes fraction", "data"), xytext=(6, 5),
                    textcoords="offset points", ha="left", va="bottom", fontsize=10, color=INK2)
        for name, sel, col, mk, lsty in groups:
            g = sorted([p for p in pts if sel(p)], key=lambda p: p[0])
            if g:
                ax.plot([p[0] for p in g], [p[4][key] for p in g], color=col, marker=mk, markersize=7, linestyle=lsty, label=name)
        ax.set_xlabel("nudge strength (lambda)"); ax.set_ylabel(yl); ax.set_title(title, loc="left")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 3), bbox_to_anchor=(0.5, -0.16))
    fig.text(0.01, -0.22, "RePaint + mirror nudge (2a), one-sided holes, 48 test faces with 3 fills each.", color=MUTED, fontsize=9)
    fig.tight_layout(w_pad=3)
    save(fig, "fig6_mirror_guidance.png")


for fn in (fig_training, fig_bars, fig_speed, fig_hole_size, fig_gain, fig_guidance):
    try:
        fn()
    except Exception as e:                      # one broken figure must not stop the others
        print(f"{fn.__name__} failed: {type(e).__name__}: {e}")
print("done:", OUT)
