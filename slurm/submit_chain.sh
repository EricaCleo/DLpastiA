#!/bin/bash
# Submits the stages as chained jobs: each starts automatically after the previous one SUCCEEDS.
# Run on the login node from the project folder:   bash slurm/submit_chain.sh
#   TRAIN_JOB=1805564 bash slurm/submit_chain.sh     # training already running/queued: only chain the evaluation after it
#   N=100 RESAMPLES=10 bash slurm/submit_chain.sh    # settings are passed on to the stages
# If a stage fails, later jobs stay pending with reason DependencyNeverSatisfied: cancel them with scancel.
set -e
PART="${PART:-gpu_5090}"

if [ -n "$TRAIN_JOB" ]; then
  J1="$TRAIN_JOB"; echo "training job (existing): $J1"
else
  J1=$(sbatch --parsable -p "$PART" --time="${TRAIN_TIME:-72:00:00}" slurm/run_job.sbatch "bash slurm/run_pipeline.sh train_ddpm")
  echo "training job: $J1"
fi
J2=$(sbatch --parsable -p "$PART" --time="${EVAL_TIME:-12:00:00}" --dependency=afterok:"$J1" slurm/run_job.sbatch "bash slurm/run_pipeline.sh eval_diffusion")
echo "evaluation job: $J2 (starts after $J1 succeeds)"
echo "check with: squeue -u \$USER"
