import importlib.util
import os
import sys

# Force-load RC_RL utils so VGDLEnv finds load_game instead of the site-package utils module.
RC_RL_ROOT = os.environ.get(
    "RC_RL_PATH",
    "/gpfs3/well/costa/users/zqa082/brain-wide_strategies/RC_RL",
)
if RC_RL_ROOT not in sys.path:
    sys.path.insert(0, RC_RL_ROOT)

utils_path = os.path.join(RC_RL_ROOT, "utils.py")
spec = importlib.util.spec_from_file_location("utils", utils_path)
rc_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc_utils)
sys.modules["utils"] = (
    rc_utils  # ensures everyone sees the RC_RL version of utils to allow the game to be loaded
)

import random

import gym
import numpy as np
from VGDLEnv import VGDLEnv

from ..base import BaseWrapper


class VGDLWrapper(BaseWrapper):
    """
    Make your own wrapper: VGDL Wrapper
    """

    def __init__(self, env, obs_to_string=False, clip_reward=False):
        super().__init__(env, obs_to_string, clip_reward)


class RawVGDL(gym.Env):
    """Adapter that exposes RC_RL.VGDLEnv through the Gym API (step/reset/action_space/etc.)."""

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, game_name, game_folder, max_episode_steps):
        super().__init__()
        self._env = VGDLEnv(game_name=game_name, game_folder=game_folder)
        self._env.lvl = 2
        self._env.set_level(2)
        self._max_episode_steps = max_episode_steps
        self._elapsed = 0

        frame = self._env.render()
        if frame is None:
            raise ValueError("VGDL environment did not return any frame on reset.")
        first_frame = np.asarray(frame, dtype=np.uint8)
        self.frame_shape = first_frame.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.frame_shape, dtype=np.uint8
        )

        # VGDL already indexes actions internally, so we keep the actions in [0, |A|-1]
        self.action_space = gym.spaces.Discrete(len(self._env.actions))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self._env.reset()
        self._elapsed = 0
        return np.asarray(self._env.render(), dtype=np.uint8)

    def step(self, action):
        """VGDL returns (reward, ended, win); convert to Gym’s (obs, reward, done, info)"""
        reward, ended, win = self._env.step(action)
        self._elapsed += 1
        obs = np.asarray(self._env.render(), dtype=np.uint8)
        truncated = self._elapsed >= self._max_episode_steps
        done = ended or truncated
        info = {"win": bool(win)}
        if truncated and not ended:
            info["TimeLimit.truncated"] = True
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        return np.asarray(self._env.render(), dtype=np.uint8)

    def seed(self, seed=None):
        """VGDLEnv has no RNG hook; seed Python/Numpy locally for reproducibility."""
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        return [seed]

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()
        super().close()
