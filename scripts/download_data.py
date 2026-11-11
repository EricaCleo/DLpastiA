"""Download AFHQ (Kaggle: andrewmvd/animal-faces) and link it to data/raw.

Needs Kaggle credentials: KAGGLE_USERNAME + KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json.
"""
import os
from pathlib import Path

import kagglehub
import yaml

root = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(root / "configs" / "data.yaml"))

src = Path(kagglehub.dataset_download(cfg["kaggle_dataset"]))
dst = root / cfg["raw_dir"]
dst.parent.mkdir(parents=True, exist_ok=True)
if dst.is_symlink():
    dst.unlink()
if not dst.exists():
    os.symlink(src, dst)

print("Dataset cache:", src)
print("Linked to:    ", dst)
for p in sorted(src.iterdir()):
    print("  ", p.name)
