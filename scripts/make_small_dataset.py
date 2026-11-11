"""Resize AFHQ to a smaller square size (default 256) and zip it, so it is quick to upload.

    python scripts/make_small_dataset.py                 # uses configs/data.yaml raw_dir
    python scripts/make_small_dataset.py --size 256 --out ~/Desktop/afhq256

Output layout is identical to the original (afhq/{train,val}/{cat,dog,wild}/*.jpg), so on the
cluster:  unzip into DLpastiA/data/raw  ->  data/raw/afhq/train/...  and set image_size accordingly.
"""
import argparse
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from src.data import list_images, load_config

ROOT = Path(__file__).resolve().parents[1]


def convert(args):
    f, dst, size, q = args
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(f).convert("RGB").resize((size, size), Image.BICUBIC).save(dst, "JPEG", quality=q)


if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / cfg["raw_dir"]), help="folder containing train/ and val/")
    ap.add_argument("--out", default=str(Path.home() / "Desktop" / "afhq_small"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--limit", type=int, default=None, help="only N images per split/species (for testing)")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    jobs = []
    for split in (cfg["train_split"], cfg["test_split"]):
        for sp in ("cat", "dog", "wild"):
            files = list_images(src, split, sp)[: a.limit]
            jobs += [(f, out / "afhq" / split / sp / (f.stem + ".jpg"), a.size, a.quality) for f in files]
    print(f"converting {len(jobs)} images to {a.size}x{a.size} ...")
    with Pool() as p:
        p.map(convert, jobs, chunksize=64)
    zip_path = shutil.make_archive(str(out), "zip", root_dir=out, base_dir="afhq")  # top folder = afhq/
    print("done:", zip_path, f"({Path(zip_path).stat().st_size / 1e6:.0f} MB)")
