"""Cache pretrained weights. Run ONCE on the LOGIN node (it has internet; compute nodes do not).
    python scripts/download_pretrained.py                      # facebook/vit-mae-base from huggingface.co
    python scripts/download_pretrained.py --endpoint https://hf-mirror.com   # only if huggingface.co is unreachable (third-party mirror)
Weights are cached in ~/.cache/huggingface and then used offline by the training jobs.
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="facebook/vit-mae-base")
ap.add_argument("--endpoint", default=None, help="optional alternative Hugging Face endpoint")
a = ap.parse_args()
if a.endpoint:
    os.environ["HF_ENDPOINT"] = a.endpoint

from transformers import ViTMAEForPreTraining
m = ViTMAEForPreTraining.from_pretrained(a.name)
print("cached:", a.name, "| params (M):", round(sum(p.numel() for p in m.parameters()) / 1e6, 1),
      "| image_size", m.config.image_size, "| patch", m.config.patch_size, "| norm_pix_loss", m.config.norm_pix_loss)
