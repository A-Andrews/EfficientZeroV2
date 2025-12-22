#!/bin/bash

# Launches vgdl_function_test for every VGDL game config using sbatch.
# Extra arguments passed to this script are forwarded to sbatch so you can
# override defaults (e.g., --time, hydra overrides, etc.).

set -euo pipefail

games=(
  vgfmri4_avoidgeorge
  vgfmri4_bait
  vgfmri4_chase
  vgfmri4_helper
  vgfmri4_lemmings
  vgfmri4_zelda
)

for game in "${games[@]}"; do
  echo "Submitting vgdl_function_test for ${game}..."
  sbatch scripts/vgdl_function_test.sh +env.game="${game}" "$@"
done

echo "Submitted ${#games[@]} VGDL jobs."
