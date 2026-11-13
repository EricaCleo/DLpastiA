#!/bin/bash
# Submit from the project folder on the login node:
#   sbatch slurm/run_job.sbatch "python scripts/run_eval.py --method mirror_paste --n 200"
# Then:  squeue -u $USER   |   tail -f program.log (newest job; each job also keeps program-<JOBID>.log)   |   scancel <JOBID>
#
# EDIT THESE (the #SBATCH lines are read by Slurm, so they cannot use variables):
#SBATCH -J dlpastia
#SBATCH -p gpu_5090   # allowed for this account: gpu_5090, gpu_4090 (hp_* are rejected: invalid account/partition)
#SBATCH --time=01:00:00   # time limit; raise for training (e.g. 24:00:00). Without it the scheduler cannot plan.
#SBATCH --gpus=1
# Slurm writes slurm-<JOBID>.out / .err into the folder you submitted from.

set -e
ENV_NAME="${ENV_NAME:-dlpastia}"
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
RUN_CMD="${1:-python scripts/run_eval.py --method mirror_paste --n 100}"

cd "$PROJECT_DIR"; mkdir -p logs
START=$(date +%s)
echo "Job ${SLURM_JOB_ID:-unknown} on $(hostname) | dir $(pwd)"; echo "Cmd: $RUN_CMD"
echo "Code version: $(git rev-parse --short HEAD 2>/dev/null) $(git log -1 --format=%s 2>/dev/null)"   # which code produced this result

source /etc/profile || true
command -v module >/dev/null 2>&1 && module load miniforge3 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1   # ignore ~/.local packages that can shadow the conda env
export HF_HUB_OFFLINE=1     # compute nodes have no internet: use the weights cached on the login node
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

set +e
LOG="program-${SLURM_JOB_ID:-local}.log"   # one log per job (no more stale/overwritten logs)
ln -sf "$LOG" program.log                    # program.log always points at the newest job
eval "$RUN_CMD" > "$LOG" 2>&1
STATUS=$?
set -e
echo "Finished: status $STATUS, $(( $(date +%s) - START )) s. Log: $PROJECT_DIR/$LOG"
exit $STATUS
