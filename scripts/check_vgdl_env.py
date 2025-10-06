#!/usr/bin/env python3
"""Quick smoke test for the VGDL wrapper.

Usage:
    python scripts/check_vgdl_env.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ez.envs.vgdl_env import VGDLAtariLikeEnv


def main():
    env = VGDLAtariLikeEnv(
        game_name="VGDL_aliens",
        resize=84,
        games_folder="all_games_original",
        rc_rl_root="/gpfs3/well/costa/users/zqa082/brain-wide_strategies/RC_RL",
    )
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    print(f"reset: obs shape={getattr(obs, 'shape', None)} type={type(obs)}")
    episode_reward = 0.0
    for step in range(10):
        action = env.action_space.sample()
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out
        episode_reward += reward
        print(
            f"step={step} action={action} reward={reward:.3f} done={done} "
            f"win={info.get('win')} obs_shape={getattr(obs, 'shape', None)}"
        )
        if done:
            print(f"Episode finished with total reward {episode_reward:.3f}")
            episode_reward = 0.0
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    env.close()


if __name__ == "__main__":
    main()
