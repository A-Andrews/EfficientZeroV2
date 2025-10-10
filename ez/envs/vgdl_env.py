import os
import sys
from typing import Optional

import cv2
import numpy as np
import importlib.util
import random

try:
    import gymnasium as gym
    from gymnasium import spaces

    _is_gymnasium = True
except Exception:  # pragma: no cover - gymnasium not available
    import gym  # type: ignore
    from gym import spaces  # type: ignore

    _is_gymnasium = False

from ez.utils.format import arr_to_str

DEFAULT_RC_RL_ROOT = os.environ.get(
    "RC_RL_ROOT",
    "/gpfs3/well/costa/users/zqa082/brain-wide_strategies/RC_RL",
)

RC_RL_ROOT = os.path.expanduser(DEFAULT_RC_RL_ROOT)
if RC_RL_ROOT not in sys.path:
    sys.path.append(RC_RL_ROOT)

UTILS_PATH = os.path.join(RC_RL_ROOT, 'utils.py')
if os.path.isfile(UTILS_PATH):
    spec = importlib.util.spec_from_file_location('rc_rl_utils', UTILS_PATH)
    rc_rl_utils = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(rc_rl_utils)
    sys.modules.setdefault('rc_rl_utils', rc_rl_utils)
    sys.modules['utils'] = rc_rl_utils

try:
    from VGDLEnv import VGDLEnv  # type: ignore
except ImportError as exc:  # pragma: no cover - developer feedback
    raise ImportError(
        "Unable to import RC_RL.VGDLEnv. Set RC_RL_ROOT env var or adjust the path."
    ) from exc


class VGDLAtariLikeEnv(gym.Env):
    """Thin wrapper so RC_RL's VGDL env looks like an Atari-style Gym env."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        game_name: str = "VGDL_aliens",
        *,
        resize: int = 84,
        games_folder: Optional[str] = None,
        rc_rl_root: Optional[str] = None,
        max_episode_steps: Optional[int] = None,
        obs_to_string: bool = False,
        clip_reward: bool = False,
        level_mode: str = "sequential",
        level_seed: Optional[int] = 0,
        fixed_level: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.rc_rl_root = os.path.expanduser(rc_rl_root or RC_RL_ROOT)
        self.games_folder = os.path.join(
            self.rc_rl_root, games_folder or "all_games_original"
        )
        if not os.path.isdir(self.games_folder):
            raise FileNotFoundError(
                f"VGDL games folder '{self.games_folder}' does not exist."
            )

        self.game_name = self._resolve_game_name(game_name)
        self.resize = int(resize)
        if self.resize <= 0:
            raise ValueError("resize must be a positive integer")
        self.max_episode_steps = max_episode_steps
        self.obs_to_string = obs_to_string
        self.clip_reward = clip_reward

        self._env = VGDLEnv(self.game_name, self.games_folder)
        self._elapsed_steps = 0

        self._num_levels = max(len(getattr(self._env, "env_list", []) or []), 1)
        level_mode_normalized = (level_mode or "sequential").lower()
        if level_mode_normalized not in {"sequential", "random", "fixed"}:
            raise ValueError(
                "level_mode must be one of {'sequential', 'random', 'fixed'}"
            )
        self.level_mode = level_mode_normalized
        self.fixed_level = fixed_level if fixed_level is not None else 0
        self.level_seed = 0 if level_seed is None else int(level_seed)
        self._sequential_cursor = None
        self._current_level = None
        # Keep a RNG even for sequential mode so random start offsets are deterministic.
        self._level_rng = random.Random(self.level_seed)

        actions = getattr(self._env, "actions", None)
        if not actions:
            raise RuntimeError("RC_RL VGDLEnv did not expose a valid action list")

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.resize, self.resize, 1),
            dtype=np.uint8,
        )

    @staticmethod
    def _resolve_game_name(game_name: str) -> str:
        if not game_name:
            raise ValueError("game_name cannot be empty")
        normalized = game_name.lower()
        if normalized.startswith("vgdl_"):
            normalized = normalized.split("_", 1)[1]
        return normalized

    def _render_frame(self) -> np.ndarray:
        frame = self._env.render()
        if frame is None:
            raise RuntimeError("RC_RL VGDLEnv.render() returned None")
        return frame

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)[..., np.newaxis]

    def _format_obs(self, obs: np.ndarray):
        if self.obs_to_string:
            return arr_to_str(obs)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None and hasattr(self._env, "seed"):
            try:
                self._env.seed(seed)
            except Exception:
                pass

        self._sync_level_metadata()

        requested_level = None
        if options and "level" in options:
            requested_level = int(options["level"])

        level = self._select_level(requested_level)
        if hasattr(self._env, "lvl"):
            self._env.lvl = level

        self._env.reset()
        self._sync_level_metadata()
        self._elapsed_steps = 0
        self._current_level = getattr(self._env, "lvl", level)

        frame = self._preprocess(self._render_frame())
        obs = self._format_obs(frame)

        return obs

    def step(self, action):
        action_idx = int(action)
        reward, ended, win = self._env.step(action_idx)
        self._current_level = getattr(self._env, "lvl", self._current_level)

        frame = self._preprocess(self._render_frame())
        obs = self._format_obs(frame)

        self._elapsed_steps += 1
        terminated = bool(ended)
        truncated = False
        if self.max_episode_steps is not None and self._elapsed_steps >= self.max_episode_steps:
            truncated = True

        info = {
            "win": bool(win),
            "terminated": terminated,
            "truncated": truncated,
            "level": self._current_level if self._current_level is not None else getattr(self._env, "lvl", None),
            "raw_reward": float(reward),
        }

        if self.clip_reward:
            reward = np.sign(reward)

        done = terminated or truncated
        return obs, float(reward), bool(done), info

    def render(self, mode: str = "rgb_array"):
        frame = self._env.render()
        if mode not in (None, "rgb_array", "human"):
            raise NotImplementedError(f"Unsupported render mode '{mode}'")
        return frame

    def seed(self, seed: Optional[int] = None):
        if seed is not None and hasattr(self._env, "seed"):
            try:
                self._env.seed(seed)
            except Exception:
                pass
        self._rng_seed = seed
        return [seed]

    def close(self):
        close_fn = getattr(self._env, 'close', None)
        if callable(close_fn):
            try:
                return close_fn()
            except Exception:
                pass
        pygame_module = getattr(self._env, 'pygame', None)
        if pygame_module is not None and hasattr(pygame_module, 'quit'):
            try:
                pygame_module.quit()
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sync_level_metadata(self) -> None:
        env_list = getattr(self._env, "env_list", None)
        if env_list is not None:
            self._num_levels = max(len(env_list), 1)
        if self.level_mode == "sequential":
            if self._sequential_cursor is None:
                # deterministically offset the start with the seed
                self._sequential_cursor = self.level_seed % self._num_levels
            else:
                self._sequential_cursor %= self._num_levels
        if self.level_mode == "fixed":
            self.fixed_level = self._normalize_level(self.fixed_level)

    def _normalize_level(self, level: Optional[int]) -> int:
        if level is None:
            return 0
        if self._num_levels <= 0:
            return int(level)
        return int(level) % self._num_levels

    def _select_level(self, override: Optional[int]) -> int:
        if override is not None:
            return self._normalize_level(override)

        if self.level_mode == "fixed":
            return self._normalize_level(self.fixed_level)

        if self.level_mode == "random":
            return self._level_rng.randrange(self._num_levels)

        # Sequential (default)
        if self._sequential_cursor is None:
            self._sequential_cursor = self.level_seed % self._num_levels
        level = self._sequential_cursor
        self._sequential_cursor = (self._sequential_cursor + 1) % self._num_levels
        return level
