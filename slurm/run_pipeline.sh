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

# ---- locks: let two jobs share the GAN work without ever training the same model at the same time ----
# A GAN training stage takes a lock (checkpoints/.lock_<model>.d, owner = job id). If another job holds it, we wait; when
# that job has finished, our stage resumes the finished checkpoint, sees the budget is used up, and ends at once.
# Stale locks (owner job no longer in the queue) are removed. eval_gan waits until NO training is running.
POLL="${LOCK_POLL:-120}"
lock_alive() {   # 0 = the lock is held by a job that still exists (or we cannot tell); 1 = clearly stale
  local owner out rc; owner="$(cat "$1/owner" 2>/dev/null)"
  [ -z "$owner" ] && return 0                              # owner file not written yet: lock is just being taken
  [ "$owner" = "local" ] && return 0
  command -v squeue >/dev/null 2>&1 || return 0            # cannot ask Slurm: assume alive (never steal a lock by mistake)
  out="$(squeue -j "$owner" -h 2>&1)"; rc=$?
  echo "$out" | grep -qi "invalid job id" && return 1      # Slurm says the job does not exist any more
  [ $rc -eq 0 ] && [ -z "$out" ] && return 1               # job is not in the queue any more
  return 0                                                 # running/pending, or any other error: assume alive
}
acquire_lock() {  # acquire_lock <model name>
  LOCKDIR="checkpoints/.lock_$1.d"; mkdir -p checkpoints
  until mkdir "$LOCKDIR" 2>/dev/null; do
    if ! lock_alive "$LOCKDIR"; then echo "[lock] removing stale lock of $1"; rm -rf "$LOCKDIR"; continue; fi
    echo "[lock] $1 is being trained by job $(cat "$LOCKDIR/owner" 2>/dev/null); waiting"; sleep "$POLL"
  done
  echo "${SLURM_JOB_ID:-local}" > "$LOCKDIR/owner"
  trap 'rm -rf "$LOCKDIR"' EXIT INT TERM
  echo "[lock] took the lock for $1"
}
wait_all_locks() {
  while ls -d checkpoints/.lock_*.d >/dev/null 2>&1; do
    for d in checkpoints/.lock_*.d; do lock_alive "$d" || { echo "[lock] removing stale $d"; rm -rf "$d"; }; done
    if ls -d checkpoints/.lock_*.d >/dev/null 2>&1; then echo "[lock] other trainings still running; waiting"; sleep "$POLL"; fi
  done
}

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
    acquire_lock g_uncond
    python scripts/m2_stylegan_inversion.py train --image-size "$SIZE" --max-hours "${HOURS:-22}" --ckpt checkpoints/g_uncond_128.pt ;;
  train_comod)      # Method 3
    acquire_lock comod
    python scripts/m3_stylegan_noninversion.py train --image-size "$SIZE" --max-hours "${HOURS:-22}" --ckpt checkpoints/comod_128.pt ;;
  train_mirror_04a) # Method 4, mirror input only
    acquire_lock mirror_04a
    python scripts/m4_mirror_conditioned.py train --image-size "$SIZE" --max-hours "${HOURS:-22}" --ckpt checkpoints/mirror_04a_128.pt ;;
  train_mirror_04b) # Method 4, mirror input + symmetry loss
    acquire_lock mirror_04b
    python scripts/m4_mirror_conditioned.py train --symmetry "${SYM:-0.1}" --image-size "$SIZE" --max-hours "${HOURS:-22}" --ckpt checkpoints/mirror_04b_128.pt ;;
  eval_gan)         # Methods 2, 3, 4 on three mask types
    wait_all_locks
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
  sweep_mirror)     # mirror-guidance sweep on the U-ViT model, one-sided holes (where the mirror should help most)
    # each setting: lam frac feather align   (align=1 also uses the per-face axis search and the rescaled symmetry weight)
    U="${UVIT_CKPT:-checkpoints/uvit_128.pt}"
    for cfg in "0.02 0.5 0 0" "0.05 0.5 0 0" "0.02 0.2 0 0" "0.05 0.2 0 0" "0.05 0.5 9 0" "0.1 0.3 9 0" \
               "0.05 0.5 9 1" "0.1 0.3 9 1" "0.2 0.3 9 1" "0.1 1.0 9 1"; do
      set -- $cfg; LAM=$1; FRAC=$2; FEA=$3; ALN=$4
      EXTRA=""; [ "$ALN" = "1" ] && EXTRA="--align --s-lo 0.65 --s-hi 0.9"
      echo "=== mirror adaptive, lam=$LAM guide_frac=$FRAC feather=$FEA align=$ALN ==="
      python scripts/m6_mirror_guided_repaint.py eval --ckpt "$U" --adaptive --lam "$LAM" --guide-frac "$FRAC" --feather "$FEA" $EXTRA \
        --name "mirror_adaptive_uvit_l${LAM}_f${FRAC}_k${FEA}_a${ALN}" --mask one_sided --n "$N" --K "$K" --image-size "$SIZE" \
        --resamples "$RESAMPLES" --save-dir "outputs/samples/mirror_adaptive_uvit_l${LAM}_f${FRAC}_k${FEA}_a${ALN}_one_sided"
    done ;;
  *) echo "unknown stage: $STAGE"; exit 2 ;;
esac
