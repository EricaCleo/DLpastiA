#!/bin/bash
# Runs one pipeline STAGE (several commands in a row, on the same GPU, no re-queueing).
#   bash slurm/run_pipeline.sh <stage>
# Stages:  train_ddpm | eval_diffusion | train_mae | eval_mae | train_g_uncond | train_comod | train_mirror_04a | train_mirror_04b | eval_gan | train_uvit | eval_uvit | sweep_mirror | train_mirror_04c | final_eval | ablate_mirror | extend_gan | finetune_gan | eval_hybrid
# Override settings with environment variables, e.g.  N=100 K=3 RESAMPLES=10 bash slurm/run_pipeline.sh eval_diffusion
# Add new stages (train_mae, train_gan, ...) as new `case` branches when those methods exist.
set -e
STAGE="${1:?usage: bash slurm/run_pipeline.sh <train_ddpm|eval_diffusion>}"
CKPT="${CKPT:-checkpoints/ddpm_128.pt}"
SIZE="${SIZE:-128}"
N="${N:-48}"; K="${K:-3}"; RESAMPLES="${RESAMPLES:-5}"; MAX_STEPS="${MAX_STEPS:-300000}"
# Methods 3, 4a and 4b must get the SAME number of training steps for a fair ablation: method 3 stopped at this step
# (18 h budget). The mirror models train more slowly per hour, so they are brought up to this step afterwards.
TARGET_STEPS="${TARGET_STEPS:-128339}"

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
  train_mirror_04a) # Method 4a (mirror input only): stops at the hour budget OR at TARGET_STEPS, whichever comes first
    acquire_lock mirror_04a
    python scripts/m4_mirror_conditioned.py train --image-size "$SIZE" --max-hours "${HOURS:-22}" --max-steps "$TARGET_STEPS" --ckpt checkpoints/mirror_04a_128.pt ;;
  train_mirror_04b) # Method 4b (mirror input + symmetry loss).
    # (1) first bring 04a up to TARGET_STEPS (it stopped on its hour budget with fewer steps than method 3); resumes the checkpoint
    acquire_lock mirror_04a
    python scripts/m4_mirror_conditioned.py train --image-size "$SIZE" --max-hours 200 --max-steps "$TARGET_STEPS" --ckpt checkpoints/mirror_04a_128.pt
    rm -rf "$LOCKDIR"; trap - EXIT INT TERM
    # (2) then 04b itself, also up to TARGET_STEPS
    acquire_lock mirror_04b
    python scripts/m4_mirror_conditioned.py train --symmetry "${SYM:-0.1}" --image-size "$SIZE" --max-hours 200 --max-steps "$TARGET_STEPS" --ckpt checkpoints/mirror_04b_128.pt ;;
  eval_gan)         # Methods 2, 3, 4 on three mask types
    wait_all_locks
    for mask in one_sided centre square; do
      echo "=== stylegan_inversion / $mask ==="
      python scripts/m2_stylegan_inversion.py eval --ckpt checkpoints/g_uncond_128.pt --mask "$mask" --n "${N_INV:-$N}" --K 3 --image-size "$SIZE" --save-dir "outputs/samples/stylegan_inversion_${mask}"
      echo "=== stylegan_noninversion / $mask ==="
      python scripts/m3_stylegan_noninversion.py eval --ckpt checkpoints/comod_128.pt --mask "$mask" --n "${N_GAN:-$N}" --K 3 --image-size "$SIZE" --save-dir "outputs/samples/stylegan_noninversion_${mask}"
      for v in 04a 04b; do
        echo "=== mirror_$v / $mask ==="
        python scripts/m4_mirror_conditioned.py eval --ckpt "checkpoints/mirror_${v}_128.pt" --name "mirror_$v" --mask "$mask" --n "${N_GAN:-$N}" --K 3 --image-size "$SIZE" --save-dir "outputs/samples/mirror_${v}_${mask}"
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
  train_mirror_04c) # Method 4c: mirror input aligned to the face's own symmetry axis; same number of steps as 3 / 4a / 4b
    acquire_lock mirror_04c
    python scripts/m4_mirror_conditioned.py train --aligned --image-size "$SIZE" --max-hours 200 --max-steps "$TARGET_STEPS" --ckpt checkpoints/mirror_04c_128.pt
    rm -rf "$LOCKDIR"; trap - EXIT INT TERM
    for mask in one_sided centre square; do   # small run (same 48 faces as the other methods, for the pictures) + final run
      echo "=== mirror_04c / $mask (48 faces) ==="
      python scripts/m4_mirror_conditioned.py eval --ckpt checkpoints/mirror_04c_128.pt --name mirror_04c --mask "$mask" --n 48 --K 3 --image-size "$SIZE" --save-dir "outputs/samples/mirror_04c_${mask}"
      echo "=== mirror_04c / $mask (final) ==="
      RESULTS_DIR=outputs/results_final python scripts/m4_mirror_conditioned.py eval --ckpt checkpoints/mirror_04c_128.pt --name mirror_04c --mask "$mask" --n "${N_FAST:-1500}" --K 3 --batch 32 --image-size "$SIZE" --save-dir "outputs/samples_final/mirror_04c_${mask}"
    done ;;
  final_eval)       # FINAL numbers for the report: fast methods on all test faces, slow methods on N_SLOW evenly spread faces.
    # Results go to outputs/results_final (with per-face values for scripts/compare_paired.py). A finished result is
    # skipped, so the job can simply be submitted again if it hits its time limit.
    export RESULTS_DIR=outputs/results_final
    NF="${N_FAST:-1500}"; NS="${N_SLOW:-300}"; U="${UVIT_CKPT:-checkpoints/uvit_128.pt}"; S=outputs/samples_final
    run() {   # run <result name> <mask> <command...>
      local name="$1" mask="$2"; shift 2
      if [ -f "$RESULTS_DIR/${name}__${mask}.json" ]; then echo "=== $name / $mask: already done, skipped ==="; return 0; fi
      echo "=== $name / $mask ==="; "$@"
    }
    for mask in one_sided centre square; do   # fast methods first
      for b in grey_fill mirror_paste; do
        run "$b" "$mask" python scripts/run_eval.py --method "$b" --mask "$mask" --n "$NF" --image-size "$SIZE" --save-dir "$S/${b}_${mask}"
      done
      run mae "$mask" python scripts/m1_mae.py eval --ckpt "${MAE_CKPT:-checkpoints/mae_128.pt}" --mask "$mask" --n "$NF" --batch 32 --image-size "$SIZE" --save-dir "$S/mae_${mask}"
      run stylegan_noninversion "$mask" python scripts/m3_stylegan_noninversion.py eval --ckpt checkpoints/comod_128.pt --mask "$mask" --n "$NF" --K 3 --batch 32 --image-size "$SIZE" --save-dir "$S/stylegan_noninversion_${mask}"
      for v in 04a 04b; do
        run "mirror_$v" "$mask" python scripts/m4_mirror_conditioned.py eval --ckpt "checkpoints/mirror_${v}_128.pt" --name "mirror_$v" --mask "$mask" --n "$NF" --K 3 --batch 32 --image-size "$SIZE" --save-dir "$S/mirror_${v}_${mask}"
      done
    done
    for mask in one_sided centre square; do   # slow methods
      run stylegan_inversion "$mask" python scripts/m2_stylegan_inversion.py eval --ckpt checkpoints/g_uncond_128.pt --mask "$mask" --n "$NS" --K 3 --image-size "$SIZE" --save-dir "$S/stylegan_inversion_${mask}"
      run repaint_uvit "$mask" python scripts/m5_repaint.py eval --ckpt "$U" --name repaint_uvit --mask "$mask" --n "$NS" --K "$K" --image-size "$SIZE" --resamples "$RESAMPLES" --save-dir "$S/repaint_uvit_${mask}"
      run mirror_guided_repaint_uvit "$mask" python scripts/m6_mirror_guided_repaint.py eval --ckpt "$U" --adaptive --lam 0.02 --guide-frac 0.2 --name mirror_guided_repaint_uvit --mask "$mask" --n "$NS" --K "$K" --image-size "$SIZE" --resamples "$RESAMPLES" --save-dir "$S/mirror_guided_repaint_uvit_${mask}"
    done ;;
  ablate_mirror)    # Does 4a really use the mirror? Same faces, mirror channels blanked or taken from another face.
    export RESULTS_DIR=outputs/results_final
    for mask in one_sided square centre; do
      for ab in blank shuffled; do
        echo "=== mirror_04a, mirror $ab / $mask ==="
        python scripts/m4_mirror_conditioned.py eval --ckpt checkpoints/mirror_04a_128.pt --name "mirror_04a_${ab}" --ablate "$ab" --mask "$mask" --n "${N_FAST:-1500}" --K 3 --batch 32 --image-size "$SIZE"
      done
    done ;;
  extend_gan)       # Train a finished GAN LONGER on a COPY of its checkpoint, then evaluate it as <name>_long.
    # MODEL=comod (method 3) or MODEL=mirror_04a. Give both the same EXT_STEPS so the comparison stays fair.
    MODEL="${MODEL:?set MODEL=comod or MODEL=mirror_04a}"; EXT="${EXT_STEPS:-260000}"
    SRC="checkpoints/${MODEL}_128.pt"; DST="checkpoints/${MODEL}_long_128.pt"
    [ -f "$DST" ] || cp "$SRC" "$DST"                      # the original checkpoint is never changed
    acquire_lock "${MODEL}_long"
    if [ "$MODEL" = "comod" ]; then SCRIPT=scripts/m3_stylegan_noninversion.py; NAME=stylegan_noninversion_long
    else SCRIPT=scripts/m4_mirror_conditioned.py; NAME="${MODEL}_long"; fi
    EXTRA="--name $NAME"
    python "$SCRIPT" train --image-size "$SIZE" --max-hours 200 --max-steps "$EXT" --ckpt "$DST"
    rm -rf "$LOCKDIR"; trap - EXIT INT TERM
    for mask in one_sided centre square; do
      echo "=== $NAME / $mask (48 faces, for pictures) ==="
      python "$SCRIPT" eval --ckpt "$DST" $EXTRA --mask "$mask" --n 48 --K 3 --image-size "$SIZE" --save-dir "outputs/samples/${NAME}_${mask}"
      echo "=== $NAME / $mask (final) ==="
      RESULTS_DIR=outputs/results_final python "$SCRIPT" eval --ckpt "$DST" $EXTRA --mask "$mask" --n "${N_FAST:-1500}" --K 3 --batch 32 --image-size "$SIZE"
    done ;;
  finetune_gan)     # Fine-tune a finished GAN on a COPY of its checkpoint with targeted hyperparameters, evaluate as <name>_ft:
    #   lower learning rate (FT_LR, default 2e-4 = 10x lower), harder training holes (squares up to FT_RATIO=0.6 of the
    #   image, plus one-sided and centre holes), for FT_STEPS more steps. MODEL=comod (method 3) or MODEL=mirror_04a.
    MODEL="${MODEL:?set MODEL=comod or MODEL=mirror_04a}"
    SRC="checkpoints/${MODEL}_128.pt"; DST="checkpoints/${MODEL}_ft_128.pt"
    [ -f "$DST" ] || cp "$SRC" "$DST"                      # the original checkpoint is never changed
    acquire_lock "${MODEL}_ft"
    if [ "$MODEL" = "comod" ]; then SCRIPT=scripts/m3_stylegan_noninversion.py; NAME=stylegan_noninversion_ft
    else SCRIPT=scripts/m4_mirror_conditioned.py; NAME="${MODEL}_ft"; fi
    python "$SCRIPT" train --image-size "$SIZE" --max-hours 200 --max-steps $(( TARGET_STEPS + ${FT_STEPS:-60000} )) \
      --lr "${FT_LR:-2e-4}" --train-masks mix --train-max-ratio "${FT_RATIO:-0.6}" --ckpt "$DST"
    rm -rf "$LOCKDIR"; trap - EXIT INT TERM
    for mask in one_sided centre square; do
      echo "=== $NAME / $mask (48 faces, for pictures) ==="
      python "$SCRIPT" eval --ckpt "$DST" --name "$NAME" --mask "$mask" --n 48 --K 3 --image-size "$SIZE" --save-dir "outputs/samples/${NAME}_${mask}"
      echo "=== $NAME / $mask (final) ==="
      RESULTS_DIR=outputs/results_final python "$SCRIPT" eval --ckpt "$DST" --name "$NAME" --mask "$mask" --n "${N_FAST:-1500}" --K 3 --batch 32 --image-size "$SIZE"
    done ;;
  eval_hybrid)      # Method 6 version 2: mirror GAN (4a) fill, refined by RePaint from START_FRACS of the steps.
    # Also a control with the plain StyleGAN (method 3) as first fill, to see what the mirror adds inside the hybrid.
    U="${UVIT_CKPT:-checkpoints/uvit_128.pt}"; GM="${HYB_GAN:-checkpoints/mirror_04a_128.pt}"; NS="${N_SLOW:-300}"
    for mask in one_sided centre square; do
      for f in ${START_FRACS:-0.2 0.4}; do
        for init in mirror comod; do
          [ "$init" = comod ] && [ "$f" != "${CONTROL_FRAC:-0.2}" ] && continue
          G="$GM"; NAME="mirror_gan_repaint_s${f}"
          [ "$init" = comod ] && G=checkpoints/comod_128.pt && NAME="stylegan_repaint_s${f}"
          echo "=== $NAME / $mask (48 faces, for pictures) ==="
          python scripts/m6_mirror_guided_repaint.py eval --ckpt "$U" --gan-init "$G" --start-frac "$f" --name "$NAME" --mask "$mask" --n 48 --K "$K" --image-size "$SIZE" --resamples "$RESAMPLES" --save-dir "outputs/samples/${NAME}_${mask}"
          echo "=== $NAME / $mask (final, $NS faces) ==="
          RESULTS_DIR=outputs/results_final python scripts/m6_mirror_guided_repaint.py eval --ckpt "$U" --gan-init "$G" --start-frac "$f" --name "$NAME" --mask "$mask" --n "$NS" --K "$K" --image-size "$SIZE" --resamples "$RESAMPLES"
        done
      done
    done ;;
  *) echo "unknown stage: $STAGE"; exit 2 ;;
esac
