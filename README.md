# Filling the Missing Square: Supervised vs Unsupervised Inpainting on Animal Faces

Deep learning course project. Inpainting a square hole (10-40% of the image) on AFHQ.

## Methods
| # | Method | Supervision |
|---|--------|-------------|
| 1 | MAE | self-supervised |
| 2 | StyleGAN Inversion | unsupervised |
| 3 | StyleGAN Non-Inversion | supervised |
| 4 | Mirror-Conditioned (StyleGAN) | supervised |
| 5 | RePaint | unsupervised |
| 6 | Mirror-Conditioned + Diffusion (our new idea) | TBD |

## Data
Kaggle dataset `andrewmvd/animal-faces` (AFHQ). Not committed to git.

```bash
pip install -r requirements.txt
export KAGGLE_USERNAME=... KAGGLE_KEY=...
python scripts/download_data.py
```

Fixed test masks (seed in `configs/data.yaml`) are shared by all methods.

## Layout
`configs/` settings | `scripts/` data download | `src/masks.py` mask + compositing |
`src/methods/` one module per method | `src/eval/` PSNR, SSIM, LPIPS, FID, diversity
