"""Check the dataset and (optionally) export resized real test images for FID.

    python scripts/prepare_data.py            # print split sizes
    python scripts/prepare_data.py --export   # write real test images to data/real_test_<size>/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from src.data import list_images, load_config

cfg = load_config()
ap = argparse.ArgumentParser()
ap.add_argument("--export", action="store_true")
args = ap.parse_args()

for split in (cfg["train_split"], cfg["test_split"]):
    for sp in ("cat", "dog", "wild"):
        print(split, sp, len(list_images(cfg["raw_dir"], split, sp)))

if args.export:
    out = Path(__file__).resolve().parents[1] / "data" / f"real_test_{cfg['image_size']}"
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(list_images(cfg["raw_dir"], cfg["test_split"])):
        Image.open(f).convert("RGB").resize((cfg["image_size"],) * 2, Image.BICUBIC).save(out / f"{i:05d}.png")
    print("exported to", out)
