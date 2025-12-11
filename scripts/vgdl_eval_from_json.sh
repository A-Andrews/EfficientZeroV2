#!/bin/bash
# Submit vgdl_eval_levels.sh for every game->checkpoint pair in a JSON file.
# Usage: bash scripts/vgdl_eval_from_json.sh [JSON_PATH] [EPISODES_PER_LEVEL] [extra overrides...]
# Defaults: JSON_PATH=attention_suite_model_paths.json, EPISODES_PER_LEVEL=${EPISODES:-10}

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
BASE_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR/..}"
cd "$BASE_DIR"

JSON_PATH="${1:-attention_suite_model_paths.json}"
EPISODES="${2:-${EPISODES:-10}}"
EXTRA_ARGS=("${@:3}")

PY_BIN="${PY_BIN:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "Python is required to parse the model paths JSON" >&2
  exit 1
fi

if [[ ! -f "$JSON_PATH" ]]; then
  echo "JSON file not found: $JSON_PATH" >&2
  exit 1
fi

while IFS=$'\t' read -r game ckpt; do
  echo "Submitting eval for $game -> $ckpt"
  sbatch --job-name="vgdl_${game}_eval" "$SCRIPT_DIR/vgdl_eval_levels.sh" "$ckpt" "$game" "$EPISODES" "${EXTRA_ARGS[@]}"
done < <("$PY_BIN" - "$JSON_PATH" <<'PY'
import json, sys, pathlib
path = pathlib.Path(sys.argv[1])
with path.open() as f:
    data = json.load(f)
for game, ckpt in data.items():
    if game and ckpt:
        print(f"{game}\t{ckpt}")
PY
)
