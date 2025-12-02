#!/bin/bash

# Launches vgdl_function_test for every VGDL game config at a specific level
# with curriculum disabled. The first argument is the desired level index.
# Any extra arguments are forwarded to sbatch.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 LEVEL [additional sbatch args...]" >&2
  exit 1
fi

level="$1"
shift

games=(
  vgfmri4_avoidgeorge
  vgfmri4_bait
  vgfmri4_chase
  vgfmri4_helper
  vgfmri4_lemmings
  vgfmri4_zelda
)

for game in "${games[@]}"; do
  echo "Submitting vgdl_function_test for ${game} at level ${level}..."
  sbatch scripts/vgdl_function_test.sh \
    +env.game="${game}" \
    +env.initial_level="${level}" \
    +env.curriculum=null \
    "$@"
done

echo "Submitted ${#games[@]} VGDL jobs at level ${level} (curriculum disabled)."
