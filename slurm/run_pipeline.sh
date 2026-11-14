#!/bin/bash
# Runs one pipeline STAGE (several commands in a row, on the same GPU, no re-queueing).
#   bash slurm/run_pipeline.sh <stage>
# Stages:  train_ddpm | eval_diffusion | train_mae | eval_mae | train_g_uncond | train_comod | train_mirror_04a | train_mirror_04b | eval_gan | train_uvit | eval_uvit | sweep_mirror
# Override settings with environment variables, e.g.  N=100 K=3 RESAMPLES=10 bash slurm/run_pipeline.sh eval_diffusion
# Add new stages (train_mae, train_gan, ...) as new `case` branches when those methods exist.
set -e
STAGE="${1:?usage: bash slurm/run_pipeline.sh <train_ddpm|eval_diffusion>}"
CKPT="${CKPT:-checkpoints/ddpm_128.pt}"
SIZE="${SIZE:-128}"
N="${N:-48}"; K="${K:-3}"; RESAMPLES="${RESAMPLES:-5}"; MAX_STEPS="${MAX_STEPS:-300000}"

case "$STAGE" in
  train_ddpm)   # resumes automatically from $CKPT if it exists; ends immediately if already at MAX_STEPS
    python scripts/train_ddpm.py --image-size "$SIZE" --batch-size 16 --amp \
      --max-steps "$MAX_STEPS" --save-every 1000 --ckpt "$CKPT"
    ;;
  eval_diffusion)  # Methods 5 and 6 on three mask types
    for mask in one_sided centre square; do
      for method in repaint mirror_repaint_fixed mirror_repaint_adaptive; do
        echo "=== $method / $mask ==="
        python scripts/run_eval.py --method "$method" --mask "$mask" --n "$N" --K "$K" \
          --image-size "$SIZE" --ckpt "$CKPT" --resamples "$RESAMPLES" \
          --save-dir "outputs/samples/${method}_${mask}"
      done
    done
    ;;
  train_mae)    # needs the pretrained weights cached on the login node first (scripts/download_pretrained.py)
    python scripts/train_mae.py --image-size "$SIZE" --epochs "${EPOCHS:-40}" --batch-size 64 --ckpt "${MAE_CKPT:-checkpoints/mae_128.pt}"
    ;;
  eval_mae)
    for mask in one_sided centre square; do
      echo "=== mae / $mask ==="
      python scripts/run_eval.py --method mae --mask "$mask" --n "${N_MAE:-100}" --image-size "$SIZE" \
        --mae-ckpt "${MAE_CKPT:-checkpoints/mae_128.pt}" --save-dir "outputs/samples/mae_${mask}"
    done
    ;;
  train_g_uncond)   # Method 2: unconditional generator (budget HOURS, resumable)
    python scripts/m2_stylegan_inversion.py train --image-size "$SIZE" --max-hours "${HOURS:-22}" --ckpt checkpoints/g_uncond_128.pt ;;
  train_comod)      # Method 3
    python scripts/m3_stylegan_noninversion.py train --image-size "$SIZE" --max-hours "${HOURS:-22}" --ckpt checkpoints/comod_128.pt ;;
  train_mirror_04a) # Method 4, mirror input only
    python scripts/m4_mirror_conditioned.py train --image-size "$SIZE" --max-hours "${HOURS:-22}" --ckpt checkpoints/mirror_04a_128.pt ;;
  train_mirror_04b) # Method 4, mirror input + symmetry loss
    python scripts/m4_mirror_conditioned.py train --symmetry "${SYM:-0.1}" --image-size "$SIZE" --max-hours "${HOURS:-22}" --ckpt checkpoints/mirror_04b_128.pt ;;
  eval_gan)         # Methods 2, 3, 4 on three mask types
    for mask in one_sided centre square; do
      echo "=== stylegan_inversion / $mask ==="
      python scripts/m2_stylegan_inversion.py eval --ckpt checkpoints/g_uncond_128.pt --mask "$mask" --n "${N_INV:-48}" --K 3 --image-size "$SIZE" --save-dir "outputs/samples/stylegan_inversion_${mask}"
      echo "=== stylegan_noninversion / $mask ==="
      python scripts/m3_stylegan_noninversion.py eval --ckpt checkpoints/comod_128.pt --mask "$mask" --n "${N_GAN:-100}" --K 3 --image-size "$SIZE" --save-dir "outputs/samples/stylegan_noninversion_${mask}"
      for v in 04a 04b; do
        echo "=== mirror_$v / $mask ==="
        python scripts/m4_mirror_conditioned.py eval --ckpt "checkpoints/mirror_${v}_128.pt" --name "mirror_$v" --mask "$mask" --n "${N_GAN:-100}" --K 3 --image-size "$SIZE" --save-dir "outputs/samples/mirror_${v}_${mask}"
      done
    done ;;
  train_uvit)       # Method 5b: RePaint with a U-ViT backbone (diffusion model trained from scratch)
    python scripts/m5_repaint.py train --arch uvit --amp --image-size "$SIZE" --batch-size 16 \
      --max-steps "${MAX_STEPS:-300000}" --save-every 1000 --ckpt "${UVIT_CKPT:-checkpoints/uvit_128.pt}" ;;
  eval_uvit)        # RePaint and mirror-guided RePaint on the U-ViT model (separate result names)
    for mask in one_sided centre square; do
      U="${UVIT_CKPT:-checkpoints/uvit_128.pt}"
      python scripts/m5_repaint.py eval --ckpt "$U" --name repaint_uvit --mask "$mask" --n "$N" --K "$K" --image-size "$SIZE" --resamples "$RESAMPLES" --save-dir "outputs/samples/repaint_uvit_${mask}"
      python scripts/m6_mirror_guided_repaint.py eval --ckpt "$U" --fixed --name mirror_repaint_fixed_uvit --mask "$mask" --n "$N" --K "$K" --image-size "$SIZE" --resamples "$RESAMPLES" --save-dir "outputs/samples/mirror_fixed_uvit_${mask}"
      python scripts/m6_mirror_guided_repaint.py eval --ckpt "$U" --adaptive --name mirror_repaint_adaptive_uvit --mask "$mask" --n "$N" --K "$K" --image-size "$SIZE" --resamples "$RESAMPLES" --save-dir "outputs/samples/mirror_adaptive_uvit_${mask}"
    done ;;
  sweep_mirror)     # mirror-guidance strength / schedule sweep on the U-ViT model, one-sided holes (the case where the mirror should help)
    U="${UVIT_CKPT:-checkpoints/uvit_128.pt}"
    for cfg in "0.02 0.5 0" "0.05 0.5 0" "0.02 0.2 0" "0.05 0.2 0" "0.05 0.5 9" "0.1 0.3 9" "0.2 0.3 9"; do
      set -- $cfg; LAM=$1; FRAC=$2; FEA=$3
      echo "=== mirror adaptive, lam=$LAM guide_frac=$FRAC feather=$FEA ==="
      python scripts/m6_mirror_guided_repaint.py eval --ckpt "$U" --adaptive --lam "$LAM" --guide-frac "$FRAC" --feather "$FEA" \
        --name "mirror_adaptive_uvit_l${LAM}_f${FRAC}_k${FEA}" --mask one_sided --n "$N" --K "$K" --image-size "$SIZE" \
        --resamples "$RESAMPLES" --save-dir "outputs/samples/mirror_adaptive_uvit_l${LAM}_f${FRAC}_k${FEA}_one_sided"
    done ;;
  *) echo "unknown stage: $STAGE"; exit 2 ;;
esac
