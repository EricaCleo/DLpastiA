"""WHEN should the mirror nudge happen? Measures our diffusion model's timeline, following Asyrp (Kwon et al., ICLR 2023,
section 4 / figure 3), so that the guidance interval is CHOSEN BY A MEASUREMENT instead of guessed.

For real (training) faces x and every noise level t (pure noise t = 999 ... clean t = 0):
    x_t  = x noised to level t                       (what the sampler holds at that moment)
    P_t  = the model's predicted clean face from x_t  ("predicted x0")
    LPIPS(x, P_t)  how far the model's current idea of the face still is from the final face
                   -> small = the content (layout: where the eye/ear is) is already decided
    LPIPS(x, x_t)  how much noise is left ("quality deficiency" in the paper)
Asyrp's rules: edit in [T, t_edit] with LPIPS(x, P_t_edit) = 0.33 (the shortest early interval after which the content is
fixed), and boost quality (stochastic steps) in [t_boost, 0] with LPIPS(x, x_t_boost) = 1.2. Between them: plain denoising.
We also measure the same curves INSIDE the hole only (one-sided holes), because that is the part our guidance changes.

Runs on the LOGIN NODE (CPU, a few minutes) or a GPU:
    python scripts/diffusion_timeline.py
    python scripts/diffusion_timeline.py --n 64 --every 10 --device cuda
    python scripts/diffusion_timeline.py --plot-only       # only redraw the figure from the saved numbers
Writes outputs/figures/fig7_diffusion_timeline.png, outputs/diffusion_timeline.csv and .json, and prints t_edit / t_boost and the
matching '--guide-frac' for `m6_mirror_guided_repaint.py --guide-when early`.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "uvit_128.pt"))
ap.add_argument("--n", type=int, default=32, help="test faces")
ap.add_argument("--every", type=int, default=20, help="measure every this many timesteps (of 1000)")
ap.add_argument("--mask", default="one_sided", choices=["one_sided", "square", "centre"], help="holes for the hole-only curves")
ap.add_argument("--image-size", type=int, default=128)
ap.add_argument("--split", default="train_split", choices=["train_split", "test_split"],
                help="faces to measure on (default: training faces, so the chosen setting is not tuned on the test set)")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--edit-threshold", type=float, default=0.33, help="Asyrp: LPIPS(x, P_t_edit)")
ap.add_argument("--boost-threshold", type=float, default=1.2, help="Asyrp: LPIPS(x, x_t_boost)")
ap.add_argument("--metric", default="lpips", choices=["lpips", "l1"], help="l1 only for quick offline tests")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=str(ROOT / "outputs"))
ap.add_argument("--plot-only", action="store_true", help="only redraw the figure from outputs/diffusion_timeline.csv")
a = ap.parse_args()

if a.plot_only:          # redraw the figure from the saved numbers (no model, no GPU)
    with open(Path(a.out) / "diffusion_timeline.csv") as f:
        rows = [{k: (int(v) if k == "t" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
    X = [None] * json.load(open(Path(a.out) / "diffusion_timeline.json")).get("n_faces", 0)
else:
    from src.data import load_config
    from src.data.preprocessing import AFHQInpainting, list_images
    from src.diffusion import Diffusion, load_eps_model

    dev = a.device
    torch.manual_seed(a.seed)
    cfg = load_config(); cfg["image_size"] = a.image_size
    # TRAINING faces (with fixed, seeded holes): t_edit is a setting we choose, so it must not be chosen on the test faces
    ds = AFHQInpainting(list_images(cfg["raw_dir"], cfg[a.split]), size=a.image_size, train=False,
                        ratio_range=tuple(cfg["mask_ratio"]), seed=cfg["test_seed"], mask_type=a.mask)
    idx = list(range(0, len(ds), max(1, len(ds) // a.n)))[: a.n]
    X = torch.stack([ds[i][0] for i in idx]); M = torch.stack([ds[i][1] for i in idx])
    model = load_eps_model(a.ckpt, dev)
    diff = Diffusion(T=1000, steps=None, device=dev)

    if a.metric == "lpips":
        import lpips
        _lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
        dist = lambda p, q: _lp(p, q).flatten()
    else:
        dist = lambda p, q: (p - q).abs().mean((1, 2, 3))

    ts = list(range(999, -1, -a.every))
    if ts[-1] != 0:
        ts.append(0)
    rows = []
    with torch.no_grad():
        for t in ts:
            acc = {"P": [], "x": [], "P_hole": [], "x_hole": []}
            for s in range(0, len(X), a.batch):
                x, m = X[s:s + a.batch].to(dev), M[s:s + a.batch].to(dev)
                tt = torch.full((x.shape[0],), t, device=dev, dtype=torch.long)
                xt, _ = diff.q_sample(x, tt)
                eps = model(xt, tt)
                ab = diff.alphas_cumprod[t]
                P = ((xt - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
                acc["P"].append(dist(x, P)); acc["x"].append(dist(x, xt))
                acc["P_hole"].append(dist(x * m, P * m)); acc["x_hole"].append(dist(x * m, xt * m))
            r = {"t": t, **{k: torch.cat(v).mean().item() for k, v in acc.items()}}
            rows.append(r)
            print(f"t={t:4d}  LPIPS(x,P_t)={r['P']:.3f}  hole {r['P_hole']:.3f}   LPIPS(x,x_t)={r['x']:.3f}", flush=True)


def first_below(key, thr):
    """Largest t (earliest in sampling) at which the curve has come down to thr."""
    for r in rows:                     # rows go from t = 999 down to 0
        if r[key] <= thr:
            return r["t"]
    return 0


t_edit = first_below("P", a.edit_threshold)
# masking changes the LPIPS scale, so the hole curve uses the SAME RELATIVE point: where it has come down as far, relative
# to its start at pure noise, as the whole-face curve has at the threshold
ratio = a.edit_threshold / rows[0]["P"]
for r in rows:
    r["P_hole_rel"] = r["P_hole"] * rows[0]["P"] / max(rows[0]["P_hole"], 1e-9)     # hole curve on the whole-face scale
t_edit_hole = first_below("P_hole_rel", a.edit_threshold)
t_boost = first_below("x", a.boost_threshold)
out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)
if not a.plot_only:
    with open(out / "diffusion_timeline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
frac = lambda t: (1000 - t) / 1000
if not a.plot_only:
    json.dump({"t_edit": t_edit, "t_edit_hole": t_edit_hole, "t_boost": t_boost, "rho": ratio,
               "guide_frac": round(frac(t_edit_hole), 3), "guide_frac_whole": round(frac(t_edit), 3),
               "edit_threshold": a.edit_threshold, "boost_threshold": a.boost_threshold, "n_faces": len(X), "metric": a.metric},
              open(out / "diffusion_timeline.json", "w"), indent=2)
print(f"\nt_edit  (LPIPS(x, P_t) = {a.edit_threshold}, whole face): t = {t_edit}  -> guide the FIRST {frac(t_edit):.2f} of the steps")
print(f"t_edit  (same relative point, inside the hole):  t = {t_edit_hole}  -> guide the FIRST {frac(t_edit_hole):.2f} of the steps")
print(f"t_boost (LPIPS(x, x_t) = {a.boost_threshold}):                     t = {t_boost}  -> the LAST {1 - frac(t_boost):.2f} of the steps")
print(f"rho = {ratio:.3f}  (relative threshold: phi_t = hole distance / hole distance at pure noise)")
print(f"use:  --guide-when early --guide-frac {frac(t_edit_hole):.2f}   (hole-based t_edit; whole-face rule gives {frac(t_edit):.2f})")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5df", "legend.frameon": False})
fig, ax = plt.subplots(figsize=(10, 5))
T = [r["t"] for r in rows]
ax.plot(T, [r["P"] for r in rows], color="#2a78d6", label="predicted face P_t vs final (whole face)")
ax.plot(T, [r["P_hole_rel"] for r in rows], color="#2a78d6", linestyle="--", label="same, inside the hole (rescaled)")
ax.plot(T, [r["x"] for r in rows], color="#eb6834", label="noisy image x_t vs final (noise left)")
ax.axhline(a.edit_threshold, color="#6b6a65", linewidth=1); ax.axhline(a.boost_threshold, color="#6b6a65", linewidth=1)
t_boost_plot = min(t_boost, t_edit_hole)                  # the phases must not overlap in the picture
ax.axvspan(999, t_edit_hole, color="#2a78d6", alpha=0.08); ax.axvspan(t_boost_plot, 0, color="#eb6834", alpha=0.08)
ax.text(999, a.edit_threshold + 0.02, f" {a.edit_threshold}", color="#6b6a65", fontsize=9)
ax.text(999, a.boost_threshold + 0.02, f" {a.boost_threshold}", color="#6b6a65", fontsize=9)
ax.axvline(t_edit_hole, color="#2a78d6", linewidth=1.5)
ax.axvline(t_edit, color="#2a78d6", linewidth=1, linestyle=":")
top = max(r["x"] for r in rows)
ax.text(t_edit_hole, top * 0.62, f" t_edit, hole\n t = {t_edit_hole}\n (used: first {round(frac(t_edit_hole) * 100)}%)",
        fontsize=9, color="#2a78d6", va="top")
ax.text(t_edit, top * 0.86, f"t_edit, whole face \nt = {t_edit} ", fontsize=9, color="#2a78d6", va="top", ha="right")
for lo, hi, txt, col in [(t_edit_hole, 999, "1  layout decided\n(nudge here)", "#2a78d6"),
                         (t_boost_plot, t_edit_hole, "2  plain\ndenoising", "#3d3c39"), (0, t_boost_plot, "3  fine\ndetail", "#eb6834")]:
    if hi - lo >= 60:                                     # label a phase only when it is wide enough to hold the text
        ax.text((lo + hi) / 2, 0.02, txt, ha="center", va="bottom", fontsize=9, color=col)
ax.set_xlim(1000, 0)
ax.set_xlabel("noise level t  (sampling runs from left = pure noise to right = clean)")
ax.set_ylabel("LPIPS to the final face" if a.metric == "lpips" else "L1 to the final face")
ax.set_title("Diffusion model (U-ViT): when is the content of the face decided?", loc="left", fontweight="bold")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9.5)
fig.tight_layout()
fig.savefig(out / "figures" / "fig7_diffusion_timeline.png", dpi=200, bbox_inches="tight")
print("saved", out / "figures" / "fig7_diffusion_timeline.png", "and", out / "diffusion_timeline.csv")
