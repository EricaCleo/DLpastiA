"""Example faces for the mirror model: StyleGAN non-inversion (3) vs Mirror StyleGAN (4a), side by side.
Runs on the LOGIN NODE or a laptop (CPU, a minute): it re-creates the fills for a few test faces with both checkpoints.
    python scripts/mirror_examples.py                    # one-sided holes, 6 faces where 4a helps most + 6 random faces
    python scripts/mirror_examples.py --mask square --n-best 4 --n-random 4
Columns: masked input | what 4a sees as its mirror input | method 3 | method 4a | original.
Faces are chosen from the FINAL per-face results (outputs/results_final), so the numbers under each face are the
evaluation's LPIPS (mean over its 3 fills); the picture shows one new fill per model with the same random code.
Two pictures are written: the 'best' faces (where 4a improves most) and RANDOM faces (to show the typical case honestly).
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image, ImageDraw, ImageFont

from src.data import composite, get_datasets, load_config, to_image
from src.gan import load_generator
from src.gan.inputs import INPUTS

ap = argparse.ArgumentParser()
ap.add_argument("--mask", default="one_sided", choices=["one_sided", "square", "centre"])
ap.add_argument("--base", default="checkpoints/comod_128.pt")
ap.add_argument("--mirror", default="checkpoints/mirror_04a_128.pt")
ap.add_argument("--results", default=str(ROOT / "outputs" / "results_final"))
ap.add_argument("--n-best", type=int, default=6)
ap.add_argument("--n-random", type=int, default=6)
ap.add_argument("--image-size", type=int, default=128)
ap.add_argument("--scale", type=int, default=2, help="enlarge each face this many times in the picture")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=str(ROOT / "outputs" / "figures"))
a = ap.parse_args()


def per_face(name):
    r = json.loads((Path(a.results) / f"{name}__{a.mask}.json").read_text())
    return dict(zip(r["indices"], r["per_image"]["lpips"]))


b3, b4 = per_face("stylegan_noninversion"), per_face("mirror_04a")
common = sorted(set(b3) & set(b4))
gain = {i: b3[i] - b4[i] for i in common}                         # > 0: 4a has lower (better) LPIPS
best = sorted(common, key=lambda i: -gain[i])[: a.n_best]
rng = random.Random(a.seed)
rand = sorted(rng.sample([i for i in common if i not in best], min(a.n_random, len(common) - len(best))))
share = sum(g > 0 for g in gain.values()) / len(gain)
print(f"{a.mask}: 4a better on {100 * share:.0f}% of {len(gain)} faces; best gains {[round(gain[i], 3) for i in best]}")

cfg = load_config(); cfg["image_size"] = a.image_size
_, ds = get_datasets(cfg, mask_type=a.mask)
G3, c3 = load_generator(ROOT / a.base if not Path(a.base).is_absolute() else a.base)
G4, c4 = load_generator(ROOT / a.mirror if not Path(a.mirror).is_absolute() else a.mirror)
inp3, inp4 = INPUTS[c3["mode"]][0], INPUTS[c4["mode"]][0]

S = a.image_size * a.scale
PAD, HEAD, CAP = 6, 40, 30
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 15); small = ImageFont.truetype("DejaVuSans.ttf", 13)
except OSError:
    font = small = ImageFont.load_default()
COLS = ["Masked input", "Mirror input of 4a", "StyleGAN (3)", "Mirror StyleGAN (4a)", "Original"]


def big(t):
    return Image.fromarray(to_image(t)).resize((S, S), Image.NEAREST)


def picture(indices, title, fname):
    W = len(COLS) * (S + PAD) + PAD
    H = HEAD + 26 + len(indices) * (S + CAP + PAD)
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    d.text((PAD, 8), title, fill="black", font=font)
    for c, name in enumerate(COLS):
        d.text((PAD + c * (S + PAD) + 4, HEAD), name, fill="#333333", font=small)
    for r, i in enumerate(indices):
        x, m, xm = ds[i]
        x, m, xm = x[None], m[None], xm[None]
        g = torch.Generator().manual_seed(a.seed + i)
        z = torch.randn(1, G3.z_dim, generator=g)
        with torch.no_grad():
            f3 = composite(G3(inp3(xm, m), z).clamp(-1, 1), x, m)[0]
            f4 = composite(G4(inp4(xm, m), z).clamp(-1, 1), x, m)[0]
        mir = inp4(xm, m)[0, 4:7]                                  # the flipped masked face that 4a receives
        y = HEAD + 26 + r * (S + CAP + PAD)
        for c, t in enumerate([xm[0], mir, f3, f4, x[0]]):
            img.paste(big(t), (PAD + c * (S + PAD), y))
        cap = [None, None, f"LPIPS {b3[i]:.3f}", f"LPIPS {b4[i]:.3f}  ({100 * gain[i] / max(b3[i], 1e-9):+.0f}%)", None]
        for c, t in enumerate(cap):
            if t:
                d.text((PAD + c * (S + PAD) + 4, y + S + 6), t, fill="#333333", font=small)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    img.save(out / fname); print("saved", out / fname)


picture(best, f"{a.mask.replace('_', '-')} holes: the {len(best)} test faces where 4a improves most over 3", f"examples_{a.mask}__best.png")
picture(rand, f"{a.mask.replace('_', '-')} holes: {len(rand)} RANDOM test faces (typical case)", f"examples_{a.mask}__random.png")
