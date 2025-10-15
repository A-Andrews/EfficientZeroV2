#!/bin/bash
# Submit a focused VGDL-bait hyperparameter sweep as separate Slurm jobs.
# Each entry below toggles sparse-reward-sensitive knobs (exploration noise,
# replay bias, planner depth) and runs through the stock test_run.sh launcher.

set -euo pipefail

# Wall-clock limit per job (Slurm format). Override by exporting JOB_TIME.
JOB_TIME=${JOB_TIME:-1-23:00:00}

# Base launcher invoked by sbatch; customization lives in test_run.sh.
BASE_JOB_SCRIPT=${BASE_JOB_SCRIPT:-scripts/test_run.sh}

# Curriculum defaults keep runs focused on one level until it is solved.
CURRICULUM_OVERRIDES=${CURRICULUM_OVERRIDES:-"+env.curriculum.enabled=True +env.curriculum.start_level=0 +env.curriculum.target_win_rate=0.4 +env.curriculum.min_episodes=50 +env.curriculum.recent_window=40 +env.curriculum.max_episodes=400"}
# Learner batch overrides trim GPU memory pressure for image augmentations.
BATCH_OVERRIDES=${BATCH_OVERRIDES:-"+train.batch_size=96 +train.mini_batch_size=96"}

# Allow a dry run (print commands instead of submitting).
DRY_RUN=false

usage() {
  cat <<'USAGE'
Usage: scripts/sweep_vgdl_bait.sh [--dry-run]

Environment variables:
  JOB_TIME          Slurm time limit (default: 1-23:00:00)
  BASE_JOB_SCRIPT   Path to sbatch launcher (default: scripts/test_run.sh)
  CURRICULUM_OVERRIDES Space-separated Hydra overrides appended to every job (default enables sequential win-gated progression)
  BATCH_OVERRIDES     Extra overrides appended to trim learner memory (default lowers batch_size to 96)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$BASE_JOB_SCRIPT" ]]; then
  echo "Launcher script '$BASE_JOB_SCRIPT' not found." >&2
  exit 1
fi

# Label each sweep leg; sbatch job names inherit these labels. Start at E to
# keep historical sweep identifiers intact.
LABELS=(
  "sweepE"
  "sweepF"
  "sweepG"
  "sweepH"
  "sweepI"
  "sweepJ"
  "sweepK"
  "sweepL"
  "sweepM"
  "sweepN"
)

# Hyperparameter overrides per leg (space-separated Hydra args).
# Keep lines short and focused so each job explores a coherent idea.
OVERRIDES=(
  "+env.max_episode_steps=4200 +mcts.explore_frac=0.26 +mcts.dirichlet_alpha=0.32 +mcts.num_simulations=210 +train.entropy_coeff=1.8e-3 +priority.priority_prob_beta=0.80 +data.buffer_size=120000 +data.total_transitions=120000 +data.top_transitions=90000 +train.eval_interval=4000 +train.random_warmup_steps=5500"
  "+env.max_episode_steps=3600 +mcts.explore_frac=0.22 +mcts.dirichlet_alpha=0.25 +mcts.num_simulations=240 +train.entropy_coeff=1.2e-3 +priority.priority_prob_beta=0.70 +data.buffer_size=90000 +data.total_transitions=90000 +data.top_transitions=70000 +train.eval_interval=3500 +train.training_steps=90000"
  "+env.max_episode_steps=5000 +mcts.explore_frac=0.30 +mcts.dirichlet_alpha=0.40 +mcts.num_simulations=256 +train.entropy_coeff=2.5e-3 +priority.priority_prob_beta=0.85 +data.buffer_size=140000 +data.total_transitions=140000 +data.top_transitions=100000 +train.eval_interval=4500 +train.training_steps=120000"
  "+env.max_episode_steps=4000 +mcts.explore_frac=0.18 +mcts.dirichlet_alpha=0.18 +mcts.num_simulations=180 +train.entropy_coeff=9e-4 +priority.priority_prob_beta=0.60 +data.buffer_size=80000 +data.total_transitions=80000 +data.top_transitions=60000 +train.random_warmup_steps=6500 +train.eval_interval=3200"
  "+env.max_episode_steps=4400 +mcts.explore_frac=0.24 +mcts.dirichlet_alpha=0.28 +mcts.num_simulations=300 +train.entropy_coeff=1.5e-3 +priority.priority_prob_beta=0.75 +data.buffer_size=110000 +data.total_transitions=110000 +data.top_transitions=75000 +train.training_steps=130000 +train.eval_interval=5000"
  "+env.max_episode_steps=3800 +mcts.explore_frac=0.20 +mcts.dirichlet_alpha=0.22 +mcts.num_simulations=200 +train.entropy_coeff=1.0e-3 +priority.priority_prob_beta=0.55 +data.buffer_size=70000 +data.total_transitions=70000 +data.top_transitions=50000 +train.random_warmup_steps=7000 +train.eval_interval=3000"
  "+env.max_episode_steps=4600 +mcts.explore_frac=0.27 +mcts.dirichlet_alpha=0.34 +mcts.num_simulations=280 +train.entropy_coeff=2.2e-3 +priority.priority_prob_beta=0.78 +data.buffer_size=130000 +data.total_transitions=130000 +data.top_transitions=95000 +train.training_steps=115000 +train.random_warmup_steps=4800 +train.eval_interval=4200"
  "+env.max_episode_steps=3400 +mcts.explore_frac=0.16 +mcts.dirichlet_alpha=0.15 +mcts.num_simulations=170 +train.entropy_coeff=8e-4 +priority.priority_prob_beta=0.62 +data.buffer_size=65000 +data.total_transitions=65000 +data.top_transitions=45000 +train.training_steps=95000 +train.eval_interval=2800"
  "+env.max_episode_steps=4100 +mcts.explore_frac=0.23 +mcts.dirichlet_alpha=0.30 +mcts.num_simulations=260 +train.entropy_coeff=1.6e-3 +priority.priority_prob_beta=0.72 +data.buffer_size=100000 +data.total_transitions=100000 +data.top_transitions=85000 +train.offline_training_steps=4000 +train.eval_interval=3600"
  "+env.max_episode_steps=4800 +mcts.explore_frac=0.29 +mcts.dirichlet_alpha=0.38 +mcts.num_simulations=320 +train.entropy_coeff=2.8e-3 +priority.priority_prob_beta=0.82 +data.buffer_size=150000 +data.total_transitions=150000 +data.top_transitions=110000 +train.training_steps=140000 +train.random_warmup_steps=6000 +train.eval_interval=4800"
)

if [[ ${#LABELS[@]} -ne ${#OVERRIDES[@]} ]]; then
  echo "LABELS and OVERRIDES length mismatch." >&2
  exit 1
fi

echo "Submitting ${#LABELS[@]} VGDL-bait sweep jobs..."
echo "  Time limit per job: ${JOB_TIME}"
echo "  Launcher: ${BASE_JOB_SCRIPT}"

for idx in "${!LABELS[@]}"; do
  label=${LABELS[$idx]}
  override=${OVERRIDES[$idx]}
  combined_override="${override}"
  if [[ -n "${CURRICULUM_OVERRIDES}" ]]; then
    combined_override+=" ${CURRICULUM_OVERRIDES}"
  fi
  if [[ -n "${BATCH_OVERRIDES}" ]]; then
    combined_override+=" ${BATCH_OVERRIDES}"
  fi
  read -r -a override_args <<< "${combined_override}"

  job_tag="EZ-V2-${label}"
  sbatch_cmd=(
    sbatch
    --time="${JOB_TIME}"
    --job-name="${label}"
    "${BASE_JOB_SCRIPT}"
    "tag=${job_tag}"
    "${override_args[@]}"
  )

  if $DRY_RUN; then
    printf 'DRY RUN: %q ' "${sbatch_cmd[@]}"
    printf '\n'
  else
    echo "  -> ${label} (${job_tag})"
    "${sbatch_cmd[@]}"
  fi

done
