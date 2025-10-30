#!/bin/bash
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --job-name=vgdl_function_test
#SBATCH --output=logs/vgdl_function_test/%x_%j.out
#SBATCH --error=logs/vgdl_function_test/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"
module load Miniforge3/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate ez-vgdl-py38
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/gpfs3/well/costa/users/zqa082/brain-wide_strategies/RC_RL:$PYTHONPATH"
which redis-server && redis-server --version

export RAY_REDIS_EXECUTABLE="$HOME/.local/bin/redis-server"
export RAY_TMPDIR=/well/costa/users/zqa082/ray_tmp
mkdir -p $RAY_TMPDIR
mkdir -p logs/vgdl_run
set -euo pipefail

cleanup() {
  ray stop >/dev/null 2>&1 || true
}
trap cleanup EXIT

export OMP_NUM_THREADS=1 HYDRA_FULL_ERROR=1

python ez/train.py exp_config=ez/config/exp/vgdl.yaml "$@"

echo "Done!"
