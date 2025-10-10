#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=redius_version_test
#SBATCH --output=logs/redis_version_test/%x_%j.out
#SBATCH --error=logs/redis_version_test/%x_%j.err
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
which redis-server && redis-server --version
