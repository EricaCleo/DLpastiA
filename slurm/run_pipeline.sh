#!/bin/bash
# Runs one pipeline STAGE (several commands in a row, on the same GPU, no re-queueing).
#   bash slurm/run_pipeline.sh <stage>
# Stages:  train_ddpm | eval_diffusion | train_mae | eval_mae
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
  *) echo "unknown stage: $STAGE"; exit 2 ;;
esac
