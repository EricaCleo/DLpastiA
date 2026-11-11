#!/bin/bash
# Submit from the project folder on the login node:
#   sbatch slurm/run_job.sbatch "python scripts/run_eval.py --method mirror_paste --n 200"
# Then:  parajobs   (status)   |   tail -f program.log   |   scancel <JOBID>
#
# EDIT THESE (the #SBATCH lines are read by Slurm, so they cannot use variables):
#SBATCH -J dlpastia
#SBATCH -p gpu_a800
#SBATCH --gpus=1
# Slurm writes slurm-<JOBID>.out / .err into the folder you submitted from.

set -e
ENV_NAME="${ENV_NAME:-dlpastia}"
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
RUN_CMD="${1:-python scripts/run_eval.py --method mirror_paste --n 100}"

cd "$PROJECT_DIR"; mkdir -p logs
START=$(date +%s)
echo "Job ${SLURM_JOB_ID:-unknown} on $(hostname) | dir $(pwd)"; echo "Cmd: $RUN_CMD"

source /etc/profile || true
command -v module >/dev/null 2>&1 && module load miniforge3 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

set +e
eval "$RUN_CMD" > program.log 2>&1
STATUS=$?
set -e
echo "Finished: status $STATUS, $(( $(date +%s) - START )) s. Log: $PROJECT_DIR/program.log"
exit $STATUS
