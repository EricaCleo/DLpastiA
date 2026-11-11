# Filling the Missing Square: Supervised vs Unsupervised Inpainting on Animal Faces

Deep learning course project. Inpainting a square hole (10-40% of the image) on AFHQ.

## Methods
| # | File | Method | Supervision |
|---|------|--------|-------------|
| 1 | `src/methods/mae.py` | MAE | self-supervised |
| 2 | `src/methods/stylegan_inversion.py` | StyleGAN Inversion | unsupervised |
| 3 | `src/methods/stylegan_noninversion.py` | StyleGAN Non-Inversion | supervised |
| 4 | `src/methods/mirror_conditioned.py` | Mirror-Conditioned (StyleGAN) | supervised |
| 5 | `src/methods/repaint.py` | RePaint | unsupervised |
| 6 | `src/methods/mirror_diffusion.py` | Mirror-Guided RePaint, training-free adaptive guidance (ours) | unsupervised |

## Layout
```
configs/data.yaml            dataset + mask settings (shared by all methods)
scripts/download_data.py     Kaggle download (andrewmvd/animal-faces)
scripts/prepare_data.py      split sizes; --export real test images for FID
scripts/run_eval.py          evaluate a method on the fixed test set
src/data/preprocessing.py    dataset, resize, [-1,1] normalisation, fixed test masks
src/data/masks.py            square / one-sided / centre masks, compositing
src/eval/metrics.py          PSNR, SSIM, LPIPS (hole only), FID, diversity
src/eval/evaluate.py         shared evaluation loop
src/methods/base.py          Inpainter interface: fill(masked, mask, K)
src/methods/baselines.py     grey fill, mirror paste (sanity baselines)
notebooks/inpainting_afhq.ipynb   driver notebook (Colab / Kaggle / local)
```

## Data
Kaggle `andrewmvd/animal-faces` (official AFHQ split: 14,630 train / 1,500 val, 512x512). Not committed.
```bash
pip install -r requirements.txt
export KAGGLE_API_TOKEN=...     # never commit this
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/run_eval.py --method mirror_paste --n 100
```
Test masks are fixed by seed, so every method is scored on identical holes.

## Status
Done: data pipeline, masks, metrics, eval loop, baselines. Method files are stubs to fill in.
