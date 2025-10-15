#!/bin/bash
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --job-name=ez_wandb_agent
#SBATCH --time=1-23:00:00
#SBATCH --output=logs/vgdl_run/%x_%j.out
#SBATCH --error=logs/vgdl_run/%x_%j.err

set -eo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <sweep_id> [additional wandb agent args]" >&2
  exit 1
fi

SWEEP_ID=$1
shift || true

module load Miniforge3/24.1.2-0
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-GNU}
eval "$(conda shell.bash hook)"
conda activate ez-vgdl-py38

export PATH="$HOME/.local/bin:$PATH"
export RAY_REDIS_EXECUTABLE="$HOME/.local/bin/redis-server"
export RAY_TMPDIR=/well/costa/users/zqa082/ray_tmp
mkdir -p "$RAY_TMPDIR" logs/vgdl_run

wandb agent "$SWEEP_ID" "$@"
