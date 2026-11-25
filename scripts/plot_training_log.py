"""Plot the losses of ONE GAN training run from its log (works while the job is still running). Login node, seconds:
    python scripts/plot_training_log.py program-1866000.log
    python scripts/plot_training_log.py program-1866000.log --out outputs/figures/training_4c.png
One panel per loss (hole L1, LPIPS, symmetry, adversarial, discriminator, R1), raw values faint, moving average bold.
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--out", default=None, help="default: outputs/figures/training_<log name>.png")
ap.add_argument("--target", type=int, default=128339, help="last training step (drawn as a dashed line)")
ap.add_argument("--smooth", type=int, default=50)
a = ap.parse_args()

steps, vals, hours, title = [], {}, [], None
for line in open(a.log, errors="replace"):
    line = line.strip()
    if line.startswith("device ") and "| mode" in line:
        title = line
    m = re.match(r"^step (\d+) D ([\d.eE+-]+) (.*?) \| ([\d.]+) h", line)
    if not m:
        continue
    s = int(m.group(1))
    if steps and s <= steps[-1]:                  # a resumed run repeats steps: keep the newest values
        k = np.searchsorted(steps, s); steps[k:] = []; hours[k:] = []
        for v in vals.values():
            v[k:] = []
    steps.append(s); hours.append(float(m.group(4)))
    vals.setdefault("D", []).append(float(m.group(2)))
    for k, v in re.findall(r"(\w+) ([\d.eE+-]+)", m.group(3)):
        vals.setdefault(k, []).append(float(v))
if not steps:
    raise SystemExit(f"no 'step ...' lines in {a.log} yet")

NAMES = [("l1", "Hole L1 loss"), ("lpips", "LPIPS loss"), ("sym", "Symmetry loss"), ("adv", "Generator adversarial loss"),
         ("D", "Discriminator loss"), ("r1", "R1 penalty")]
panels = [(k, t) for k, t in NAMES if k in vals and len(vals[k]) == len(steps)]
cols = 3; rows = (len(panels) + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(15, 4.2 * rows), squeeze=False)
plt.rcParams.update({"font.size": 11})


def smooth(y, w):
    y = np.asarray(y, float)
    if len(y) < 3:
        return y
    w = max(1, min(w, len(y)))
    pad = np.concatenate([np.full(w // 2, y[0]), y, np.full(w - 1 - w // 2, y[-1])])
    return np.convolve(pad, np.ones(w) / w, mode="valid")


for ax, (k, t) in zip(axs.ravel(), panels):
    y = np.array(vals[k])
    ax.plot(steps, y, color="#b9b8b2", linewidth=0.6)
    ax.plot(steps, smooth(y, a.smooth), color="#2a78d6", linewidth=2.2)
    ax.set_title(t, loc="left", fontweight="bold"); ax.set_xlabel("training step")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlim(0, max(a.target, steps[-1]) * 1.02)
    ax.axvline(a.target, color="#6b6a65", linestyle="--", linewidth=1)
    tail = smooth(y, a.smooth)[int(len(y) * 0.02):]
    lo, hi = tail.min(), tail.max(); ax.set_ylim(lo - 0.1 * (hi - lo + 1e-9), hi + 0.1 * (hi - lo + 1e-9))
    ax.grid(color="#e6e5df"); ax.spines[["top", "right"]].set_visible(False)
for ax in axs.ravel()[len(panels):]:
    ax.axis("off")
pct = 100 * steps[-1] / a.target
rate = steps[-1] / max(hours[-1], 1e-6)
left = max(a.target - steps[-1], 0) / max(rate, 1e-6)
fig.suptitle(f"{Path(a.log).name}: step {steps[-1]:,} of {a.target:,} ({pct:.0f}%), {hours[-1]:.1f} h trained, "
             f"about {left:.1f} h to go (dashed line = end)", x=0.01, ha="left", fontsize=12)
if title:
    fig.text(0.01, -0.01, title[:200], fontsize=8, color="#6b6a65")
fig.tight_layout()
out = Path(a.out) if a.out else Path("outputs/figures") / f"training_{Path(a.log).stem}.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"step {steps[-1]} of {a.target} ({pct:.0f}%), {hours[-1]:.2f} h, ~{left:.1f} h to go")
print("saved", out)
