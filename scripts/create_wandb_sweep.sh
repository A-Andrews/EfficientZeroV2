#!/bin/bash
set -eo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <sweep_yaml>" >&2
  exit 1
fi

SWEEP_CONFIG=$1
shift || true

module load Miniforge3/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate ez-vgdl-py38

wandb sweep "$SWEEP_CONFIG" "$@"
