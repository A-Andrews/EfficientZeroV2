#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=vgdl_curric_smoke
#SBATCH --output=logs/vgdl_smoke/%x_%j.out
#SBATCH --error=logs/vgdl_smoke/%x_%j.err

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

python - <<'PY'
from ez.envs import make_vgdl

cfg = dict(
    game="vgfmri4_bait",
    game_folder="/gpfs3/well/costa/users/zqa082/brain-wide_strategies/RC_RL/all_games_recovered",
    max_episode_steps=200,
    obs_shape=[3, 96, 96],
    n_skip=1,
    n_stack=1,
    gray_scale=False,
    image_based=True,
    clip_reward=True,
    initial_level=0,
    curriculum=dict(levels=[0, 1], min_episodes=1, win_threshold=0.0),
)

env = make_vgdl(cfg["game"], seed=0, save_path=None, **cfg)
for episode in range(3):
    obs = env.reset()
    done = False
    steps = 0
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        steps += 1
    print(f"Episode {episode}: steps={steps}, info={info}")
env.close()
PY
