"""Side-by-side picture: masked input | method 1 | method 2 | ... | original, one row per test image.
Uses the fills saved by the evaluation runs (outputs/samples/<method>_<mask>/NNNNN.png) and rebuilds the masked input and
the original from the same fixed test masks.
    python scripts/make_grid.py --mask one_sided --methods mae --rows 8
    python scripts/make_grid.py --mask one_sided --methods mae --rows 8 --no-original      # masked input | result only
    python scripts/make_grid.py --mask one_sided --methods mae --rows 8 --separate         # one file per image (masked, result)
    python scripts/make_grid.py --mask one_sided --methods mae repaint_uvit mirror_repaint_adaptive_uvit --rows 8
Must use the same --n / --image-size as the evaluation runs (defaults: 100 images, 128 px).

File names say what is in them, e.g.
    outputs/grid_one_sided__masked+mae__n100__no-original.png
    outputs/grid_square__masked+mae_n48+repaint_uvit+mirror_fixed_uvit+mirror_adaptive_uvit+original__n48.png
and with --separate:  outputs/pairs/<mask>/<mask>__<row>__masked.png , <mask>__<row>__<method>.png
Column headings use the full method names (see NAMES).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import textwrap

from PIL import Image, ImageDraw

from src.data import get_datasets, load_config, to_image

NAMES = {
    "mae": "MAE (self-supervised ViT)", "mae_n48": "MAE (self-supervised ViT)",
    "repaint_uvit": "RePaint (U-ViT)",
    "mirror_fixed_uvit": "Mirror-guided RePaint, fixed (U-ViT)", "mirror_repaint_fixed_uvit": "Mirror-guided RePaint, fixed (U-ViT)",
    "mirror_adaptive_uvit": "Mirror-guided RePaint, adaptive (U-ViT)", "mirror_repaint_adaptive_uvit": "Mirror-guided RePaint, adaptive (U-ViT)",
    "grey_fill": "Grey fill (baseline)", "mirror_paste": "Mirror paste (baseline)",
    "stylegan_inversion": "StyleGAN inversion", "stylegan_noninversion": "StyleGAN non-inversion",
    "mirror_04a": "Mirror-conditioned StyleGAN (04a)", "mirror_04b": "Mirror-conditioned StyleGAN + symmetry loss (04b)",
}

ap = argparse.ArgumentParser()
ap.add_argument("--mask", default="one_sided", choices=["square", "one_sided", "centre"])
ap.add_argument("--methods", nargs="+", required=True, help="sample-folder prefixes, e.g. mae repaint_uvit")
ap.add_argument("--n", type=int, default=100, help="images used by the evaluation run")
ap.add_argument("--rows", type=int, default=8)
ap.add_argument("--image-size", type=int, default=128)
ap.add_argument("--samples-dir", default=str(ROOT / "outputs" / "samples"))
ap.add_argument("--out", default=None)
ap.add_argument("--no-original", action="store_true", help="leave out the original column")
ap.add_argument("--separate", action="store_true", help="also save each masked input and each result as its own PNG (outputs/pairs/<mask>/)")
a = ap.parse_args()

cfg = load_config()
cfg["image_size"] = a.image_size
_, test_ds = get_datasets(cfg, mask_type=a.mask)
idxs = list(range(0, len(test_ds), len(test_ds) // a.n))[: a.n]      # same subset as evaluate()
S = a.image_size
labels = ["Masked input (hole in white)"] + [NAMES.get(m, m) for m in a.methods] + ([] if a.no_original else ["Original"])
wrapped = [textwrap.wrap(t, width=max(8, S // 6 - 1)) for t in labels]
head = 12 * max(len(w) for w in wrapped) + 8
cols = labels
canvas = Image.new("RGB", (S * len(cols), head + S * a.rows), "white")
draw = ImageDraw.Draw(canvas)
for c, lines in enumerate(wrapped):
    for i, line in enumerate(lines):
        draw.text((c * S + 4, 4 + 12 * i), line, fill="black")
missing = set()
for r in range(a.rows):
    x, m, xm = test_ds[idxs[r]]
    shown = (1 - m) * x + m * 1.0                                      # hole drawn white
    tiles = [Image.fromarray(to_image(shown))]
    for meth in a.methods:
        f = Path(a.samples_dir) / f"{meth}_{a.mask}" / f"{r:05d}.png"
        if f.exists():
            tiles.append(Image.open(f).convert("RGB").resize((S, S)))
        else:
            missing.add(str(f.parent)); tiles.append(Image.new("RGB", (S, S), "lightgray"))
    if not a.no_original:
        tiles.append(Image.fromarray(to_image(x)))
    if a.separate:
        d = ROOT / "outputs" / "pairs" / a.mask
        d.mkdir(parents=True, exist_ok=True)
        tiles[0].save(d / f"{a.mask}__{r:02d}__masked.png")
        for meth, t in zip(a.methods, tiles[1:]):
            t.save(d / f"{a.mask}__{r:02d}__{meth}.png")
    for c, t in enumerate(tiles):
        canvas.paste(t, (c * S, head + r * S))
parts = ["masked"] + a.methods + ([] if a.no_original else ["original"])
out = a.out or str(ROOT / "outputs" / f"grid_{a.mask}__{'+'.join(parts)}__n{a.n}{'__no-original' if a.no_original else ''}.png")
Path(out).parent.mkdir(parents=True, exist_ok=True)
canvas.save(out)
print("saved", out, f"({canvas.size[0]}x{canvas.size[1]})")
if missing:
    print("WARNING: no saved fills found in:", *sorted(missing), sep="\n  ")
