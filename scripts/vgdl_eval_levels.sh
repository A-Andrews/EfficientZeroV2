#!/bin/bash
# Launches evaluation for all VGDL levels of a checkpoint and logs to Weights & Biases.
# Usage: sbatch scripts/vgdl_eval_levels.sh CHECKPOINT_PATH GAME_NAME [EPISODES_PER_LEVEL]
# Optional env vars:
#   EXP_CONFIG      (default: ez/config/exp/vgdl.yaml)
#   LEVELS          comma-separated list of levels to eval; defaults to curriculum levels in the config
#   WANDB_PROJECT   (default: ez-v2-evals)
#   WANDB_ENTITY    (default: empty)
#   WANDB_TAGS      (default: "vgdl,eval")

#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --job-name=vgdl_eval_levels
#SBATCH --output=logs/vgdl_eval_levels/%x_%j.out
#SBATCH --error=logs/vgdl_eval_levels/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CHECKPOINT_PATH GAME_NAME [EPISODES_PER_LEVEL] [extra overrides...]" >&2
  exit 1
fi

CKPT="$1"
GAME="$2"
EPISODES="${3:-10}"
EXP_CONFIG="${EXP_CONFIG:-ez/config/exp/vgdl.yaml}"
LEVELS="${LEVELS:-}"
WANDB_PROJECT="${WANDB_PROJECT:-ez-v2-evals}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TAGS="${WANDB_TAGS:-vgdl,eval}"
shift 3 || true
OVERRIDES=("${@:1}")

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
BASE_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR/..}"
cd "$BASE_DIR"

# Avoid set -u breaking conda's MKL activation scripts.
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-GNU}

module load Miniforge3/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate ez-vgdl-py38
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/gpfs3/well/costa/users/zqa082/brain-wide_strategies/RC_RL:${PYTHONPATH:-}"
which redis-server && redis-server --version

export RAY_REDIS_EXECUTABLE="$HOME/.local/bin/redis-server"
export RAY_TMPDIR=/well/costa/users/zqa082/ray_tmp
mkdir -p "$RAY_TMPDIR"
mkdir -p "$BASE_DIR/logs/vgdl_eval_levels"

export OMP_NUM_THREADS=1 HYDRA_FULL_ERROR=1

python_args=(
  scripts/vgdl_eval_levels.py
  --ckpt "$CKPT"
  --game "$GAME"
  --episodes "$EPISODES"
  --exp-config "$EXP_CONFIG"
  --wandb-project "$WANDB_PROJECT"
  --wandb-entity "$WANDB_ENTITY"
  --wandb-tags "$WANDB_TAGS"
)

if [[ -n "$LEVELS" ]]; then
  python_args+=(--levels "$LEVELS")
fi

for override in "${OVERRIDES[@]}"; do
  python_args+=(--override "$override")
done

python "${python_args[@]}"

echo "Done!"
