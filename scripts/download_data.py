"""Download AFHQ (Kaggle: andrewmvd/animal-faces) and link the download to data/raw.

Layout afterwards:  data/raw/afhq/{train,val}/{cat,dog,wild}/*.jpg   (= configs/data.yaml raw_dir)

Needs Kaggle credentials: KAGGLE_API_TOKEN env var, or ~/.kaggle/access_token.
"""
import os
import shutil
from pathlib import Path

import kagglehub
import yaml

root = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(root / "configs" / "data.yaml"))

src = Path(kagglehub.dataset_download(cfg["kaggle_dataset"]))  # contains afhq/
link = root / "data" / "raw"                                   # data/raw -> download folder
link.parent.mkdir(parents=True, exist_ok=True)

if link.is_symlink():
    link.unlink()
elif link.is_dir() and all(p.is_symlink() for p in link.iterdir()):
    shutil.rmtree(link)  # leftover from an older version of this script (only symlinks inside)
if not link.exists():
    os.symlink(src, link)

images = root / cfg["raw_dir"]
print("Dataset cache:", src)
print("Linked:", link, "->", src)
print("Image folder:", images, "(exists)" if images.is_dir() else "(MISSING: check the layout)")
