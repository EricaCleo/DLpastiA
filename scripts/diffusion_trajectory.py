"""WHAT does the nudging do during sampling? Records, at every noise level of the actual RePaint sampling, the model's
current predicted face (after the nudge) for each guidance schedule, and plots Asyrp-style curves (figure 3 of Kwon et al.):
  (a) LPIPS(true face, predicted face)   inside the hole: how close the current idea of the hole is to the TRUE face
  (b) LPIPS(final result, predicted face) inside the hole: how much of the hole's content is still undecided
  (c) LPIPS(mirror, predicted face)      where the mirror is visible: how strongly the hole is being pulled to the mirror
Schedules: RePaint (no nudge), every step, first 50%, last 50%, last 20% (current), until t_edit with the normal update,
and Asyrp (until t_edit + asymmetric update).
All schedules start from the SAME random noise, so the curves differ only because of the nudging.
    python scripts/diffusion_trajectory.py                       # schedules of the timing study, strength --lam 0.1
    python scripts/diffusion_trajectory.py --lam 0.3 --n 16
Writes outputs/figures/fig9_trajectory_l<lam>.png and outputs/diffusion_trajectory_l<lam>.csv. GPU recommended (~1-2 min
per schedule for 16 faces); it also runs on a CPU, slowly.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "uvit_128.pt"))
ap.add_argument("--lam", type=float, default=0.1, help="guidance strength used for every schedule")
ap.add_argument("--timeline", default=str(ROOT / "outputs" / "diffusion_timeline.json"), help="from diffusion_timeline.py (Asyrp t_edit)")
ap.add_argument("--asyrp-frac", type=float, default=None, help="override the Asyrp guide fraction")
ap.add_argument("--n", type=int, default=16, help="test faces (one fill each)")
ap.add_argument("--mask", default="one_sided", choices=["one_sided", "square", "centre"])
ap.add_argument("--image-size", type=int, default=128)
ap.add_argument("--steps", type=int, default=250)
ap.add_argument("--jump-length", type=int, default=10)
ap.add_argument("--resamples", type=int, default=5)
ap.add_argument("--every", type=int, default=5, help="record every this many sampling levels (of --steps)")
ap.add_argument("--metric", default="lpips", choices=["lpips", "l1"], help="l1 only for quick offline tests")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", default=str(ROOT / "outputs"))
a = ap.parse_args()

from src.data import get_datasets, load_config
from src.diffusion import load_eps_model
from src.methods.mirror_diffusion import MirrorGuidedRePaint, mirrored

dev = a.device
asyrp = a.asyrp_frac
if asyrp is None:
    try:
        asyrp = json.load(open(a.timeline))["guide_frac"]
    except (OSError, KeyError):
        asyrp = 0.35
        print(f"no timeline file at {a.timeline}: using guide fraction {asyrp} for the Asyrp schedule (run diffusion_timeline.py first)")
# name -> (when, fraction, strength, colour, label)
SCHED = {"repaint": ("late", 1.0, 0.0, "#e87ba4", "RePaint, no nudge (2)"),
         "all": ("late", 1.0, a.lam, "#2a78d6", "2a: nudge every step"),
         "first50": ("early", 0.5, a.lam, "#eb6834", "2a: nudge first 50%"),
         "last50": ("late", 0.5, a.lam, "#6250d6", "2a: nudge last 50%"),
         "current": ("late", 0.2, a.lam, "#1baf7a", "2a: nudge last 20%"),
         "tedit": ("early", asyrp, a.lam, "#eda100", f"2a: nudge until t_edit (first {round(asyrp * 100)}%)"),
         "asyrp": ("early", asyrp, a.lam, "#008300", f"2a: Asyrp (until t_edit + asymmetric update)")}
ASYM = {"asyrp"}

cfg = load_config(); cfg["image_size"] = a.image_size
_, ds = get_datasets(cfg, mask_type=a.mask)
idx = list(range(0, len(ds), max(1, len(ds) // a.n)))[: a.n]
X = torch.stack([ds[i][0] for i in idx]).to(dev); M = torch.stack([ds[i][1] for i in idx]).to(dev)
XM = X * (1 - M)
U = M * mirrored(1 - M)                                     # hole pixels whose mirror partner is visible
Y = mirrored(XM)                                            # the mirror target
model = load_eps_model(a.ckpt, dev)

if a.metric == "lpips":
    import lpips
    _lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    dist = lambda p, q, w: _lp(p * w, q * w).flatten()
else:
    dist = lambda p, q, w: ((p - q).abs() * w).sum((1, 2, 3)) / (3 * w.sum((1, 2, 3))).clamp_min(1)

rows, curves = [], {}
for name, (when, frac, lam, col, label) in SCHED.items():
    m = MirrorGuidedRePaint(model, lam=lam, adaptive=True, guide_frac=frac, guide_when=when, asymmetric=name in ASYM, steps=a.steps,
                            jump_length=a.jump_length, resamples=a.resamples, device=dev)
    rec = {}
    base_guide = m.guide

    def guide(x0_hat, gt, keep, t=None, base_guide=base_guide, rec=rec):
        out = base_guide(x0_hat, gt, keep, t)
        if t is not None and t % a.every == 0:
            rec[t] = out.detach()                           # the LAST visit of each level (after RePaint's resampling)
        return out

    m.guide = guide
    torch.manual_seed(a.seed)
    with torch.no_grad():
        final = m.sample(XM, M).clamp(-1, 1)
        final = M * final + (1 - M) * X
        for t in sorted(rec, reverse=True):
            P = rec[t].clamp(-1, 1)
            T0 = int(m.diff.timestep_map[t])               # back to the 0..999 scale of figure 7
            r = {"schedule": name, "t": T0, "true": dist(X, P, M).mean().item(), "final": dist(final, P, M).mean().item(),
                 "mirror": dist(Y, P, U).mean().item()}
            rows.append(r); curves.setdefault(name, []).append(r)
        end = dist(X, final, M).mean().item()
    print(f"{name:8s} {label:45s} final LPIPS to the true face {end:.4f}", flush=True)
    SCHED[name] = (when, frac, lam, col, f"{label}  [{end:.3f}]")

out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)
tag = f"l{a.lam:g}"
with open(out / f"diffusion_trajectory_{tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5df", "legend.frameon": False, "lines.linewidth": 2})
fig, axs = plt.subplots(1, 3, figsize=(17, 5), sharex=True)
panels = [("true", "(a) predicted face vs TRUE face", "distance to the true face, in the hole"),
          ("final", "(b) predicted face vs its own final result", "content still undecided, in the hole"),
          ("mirror", "(c) predicted face vs the MIRROR", "distance to the mirror (where it is visible)")]
t_edit = (1 - asyrp) * 1000
for ax, (key, title, yl) in zip(axs, panels):
    for name, (when, frac, lam, col, label) in SCHED.items():
        c = curves.get(name, [])
        ax.plot([r["t"] for r in c], [r[key] for r in c], color=col, label=label,
                linestyle="--" if name == "repaint" else "-", zorder=5 if name == "repaint" else 2)   # baseline on top
    ax.axvline(t_edit, color="#6b6a65", linestyle=":", linewidth=1.2)
    ax.text(t_edit, ax.get_ylim()[1], " t_edit", va="top", fontsize=9, color="#6b6a65")
    ax.set_xlim(1000, 0); ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("noise level t  (left = pure noise, right = clean)"); ax.set_ylabel(yl)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
fig.text(0.01, -0.17, f"{a.mask.replace('_', '-')} holes, {len(idx)} test faces, strength lambda = {a.lam:g}, same random noise "
         f"for every schedule. Numbers in brackets: final {a.metric.upper()} to the true face (lower is better).",
         color="#6b6a65", fontsize=9)
fig.tight_layout(w_pad=2.5)
fig.savefig(out / "figures" / f"fig9_trajectory_{tag}.png", dpi=200, bbox_inches="tight")
print("saved", out / "figures" / f"fig9_trajectory_{tag}.png")
