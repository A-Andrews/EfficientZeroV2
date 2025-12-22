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

def _load_game_noncontiguous(game_name, games_folder, level_transform=None):
    """
    Drop-in replacement for RC_RL.utils.load_game that supports non-contiguous level indices.
    Keeps the original padding/transform behavior but iterates over actual level ids instead of assuming density.
    """
    import os
    import pygame
    from vgdl.rlenvironmentnonstatic import createRLInputGameFromStrings

    def _load_level(game_string, level_string):
        headless = True
        rle_create_func = lambda: createRLInputGameFromStrings(game_string, level_string)
        rle = rle_create_func()
        rle.visualize = True
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        return rle

    def _pad_level_to_size(level_string, target_width, target_height):
        """Pad a level string with walls to reach target dimensions."""
        lines = level_string.strip().split("\n")
        current_height = len(lines)
        current_width = max(len(line) for line in lines) if lines else 0

        padded_lines = []
        for line in lines:
            padding_needed = target_width - len(line)
            padded_line = line + "w" * padding_needed
            padded_lines.append(padded_line)

        rows_needed = target_height - current_height
        for _ in range(rows_needed):
            padded_lines.append("w" * target_width)

        return "\n".join(padded_lines)

    file_list = {}
    for file in os.listdir(games_folder):
        if "DS" in file:
            continue
        if "expt_ee" in game_name:
            if game_name in file:
                if "lvl" not in file:
                    level = file.split("desc_")[1][0]
                    file_list[f"game_{level}"] = file
                else:
                    level = file.split("_lvl")[1].split(".")[0]
                    file_list[level] = file
        else:
            if game_name == file.split(".txt")[0] or game_name == file.split("_lvl")[0]:
                if "lvl" not in file:
                    file_list["game"] = file
                else:
                    level = file.split("_lvl")[1].split(".")[0]
                    file_list[level] = file

    if "expt_ee" not in game_name:
        try:
            with open(f"{games_folder}/{file_list['game']}", "r") as game:
                game_string = game.read()
        except KeyError as exc:
            raise ValueError(f"Game definition file for {game_name} not found in {games_folder}") from exc

    def _sort_level_tokens(tokens):
        def sort_key(tok):
            tok_str = str(tok)
            if tok_str.isdigit():
                return (0, int(tok_str), len(tok_str), tok_str)
            return (1, tok_str)
        return sorted(tokens, key=sort_key)

    level_ids = _sort_level_tokens(k for k in file_list.keys() if not str(k).startswith("game"))
    env_list = {}
    if not level_ids:
        return env_list

    level_strings = {}
    max_width = 0
    max_height = 0
    for lvl_idx in level_ids:
        with open(f"{games_folder}/{file_list[lvl_idx]}", "r") as level_file:
            level_strings[lvl_idx] = level_file.read()

        lines = level_strings[lvl_idx].strip().split("\n")
        height = len(lines)
        width = max(len(line) for line in lines) if lines else 0
        max_width = max(max_width, width)
        max_height = max(max_height, height)

    transformed_levels = {}
    for lvl_idx in level_ids:
        padded_level = _pad_level_to_size(level_strings[lvl_idx], max_width, max_height)
        lvl_for_transform = int(lvl_idx) if str(lvl_idx).isdigit() else lvl_idx
        if callable(level_transform):
            try:
                candidate = level_transform(padded_level, lvl_for_transform, max_width, max_height)
            except TypeError:
                candidate = level_transform(padded_level, lvl_for_transform)
            if candidate is None:
                transformed_levels[lvl_idx] = padded_level
            elif isinstance(candidate, str):
                transformed_levels[lvl_idx] = candidate
            else:
                raise TypeError("level_transform must return either a string layout or None")
        else:
            transformed_levels[lvl_idx] = padded_level

    final_max_width = 0
    final_max_height = 0
    for lvl_idx in level_ids:
        lines = transformed_levels[lvl_idx].split("\n")
        height = len([line for line in lines if line != ""]) if lines != [""] else 0
        if height == 0:
            continue
        width = max(len(line) for line in lines)
        final_max_width = max(final_max_width, width)
        final_max_height = max(final_max_height, height)

    if final_max_width == 0:
        final_max_width = max_width
    if final_max_height == 0:
        final_max_height = max_height

    for lvl_idx in level_ids:
        if "expt_ee" in game_name:
            with open(f"{games_folder}/{file_list[f'game_{lvl_idx}']}", "r") as game_file:
                game_string = game_file.read()

        final_level = _pad_level_to_size(
            transformed_levels.get(lvl_idx, level_strings[lvl_idx]),
            final_max_width,
            final_max_height,
        )
        env_instance = _load_level(game_string, final_level)
        key_str = str(lvl_idx)
        env_list[key_str] = env_instance
        if key_str.isdigit():
            key_int = int(key_str)
            if key_int not in env_list:
                env_list[key_int] = env_instance

    return env_list


# Swap in the non-contiguous-friendly loader so VGDLEnv picks it up.
rc_utils.load_game = _load_game_noncontiguous

import random

import gym
import numpy as np
from VGDLEnv import VGDLEnv
from ez.envs.vgdl.curriculum import VGDLLevelCurriculum

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

    def __init__(
        self,
        game_name,
        game_folder,
        max_episode_steps,
        initial_level=None,
        curriculum_config=None,
        reward_from_score=False,
        level_aliases=None,
    ):
        super().__init__()
        self._env = VGDLEnv(game_name=game_name, game_folder=game_folder)
        self._env.env_list = self._normalize_env_keys(self._env.env_list)
        self._levels_by_int = self._build_levels_by_int(self._env.env_list.keys())
        self._level_aliases = self._normalize_aliases(level_aliases)
        self._levels = self._sort_levels(set(self._env.env_list.keys()) | set(self._level_aliases.keys()))
        if not self._levels:
            raise ValueError(f"No VGDL levels found for {game_name} in {game_folder}")
        self._curriculum = (
            VGDLLevelCurriculum.from_dict(curriculum_config, self._levels)
            if curriculum_config
            else None
        )
        init_level = self._coerce_level_key(initial_level) if initial_level is not None else None
        if init_level is not None and init_level in self._levels:
            self._current_level = init_level
        elif self._curriculum:
            self._current_level = self._curriculum.current_level
        else:
            self._current_level = self._clip_level(10)

        self._set_level(self._current_level)
        self._max_episode_steps = max_episode_steps
        self._elapsed = 0
        self._reward_from_score = reward_from_score

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
        if self._curriculum:
            stats = self._curriculum.maybe_transition()
            if stats["changed"]:
                self._set_level(stats["level"])
        self._env.reset()
        self._elapsed = 0
        obs = np.asarray(self._env.render(), dtype=np.uint8)
        self._last_info = {"level": self._current_level, "resolved_level": self._resolved_level}
        if self._curriculum:
            self._last_info["curriculum"] = stats
        return obs
    
    def step(self, action):
        prev_score = self._env.current_env._game.score
        reward, ended, win = self._env.step(action)
        if self._reward_from_score:
            score = self._env.current_env._game.score
            reward = score - prev_score
        self._elapsed += 1
        obs = np.asarray(self._env.render(), dtype=np.uint8)
        truncated = self._elapsed >= self._max_episode_steps
        done = ended or truncated
        info = {"win": bool(win), "level": self._current_level, "resolved_level": self._resolved_level}
        if truncated and not ended:
            info["TimeLimit.truncated"] = True
        if self._curriculum and done:
            self._curriculum.record_episode(bool(win))
            info["curriculum"] = self._curriculum.maybe_transition()
            if info["curriculum"]["changed"]:
                self._set_level(info["curriculum"]["level"])
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

    def _coerce_level_key(self, level):
        if isinstance(level, str):
            return level
        try:
            return str(int(level))
        except (TypeError, ValueError):
            return str(level)

    def _normalize_env_keys(self, env_list):
        """Return a new dict with canonical string keys, keeping first occurrence to preserve variants like '0' and '00'."""
        normalized = {}
        for key, env in env_list.items():
            key_str = self._coerce_level_key(key)
            if key_str in normalized:
                continue
            normalized[key_str] = env
        return normalized

    def _sort_levels(self, level_keys):
        def sort_key(k):
            k_str = self._coerce_level_key(k)
            if k_str.isdigit():
                return (0, int(k_str), len(k_str), k_str)
            return (1, 0, k_str)

        return sorted({self._coerce_level_key(k) for k in level_keys}, key=sort_key)

    def _build_levels_by_int(self, level_keys):
        mapping = {}
        for key in level_keys:
            key_str = self._coerce_level_key(key)
            if key_str.isdigit():
                mapping.setdefault(int(key_str), []).append(key_str)
        for _, keys in mapping.items():
            keys.sort(key=lambda s: (len(s), s))
        return mapping

    def _clip_level(self, level):
        key = self._coerce_level_key(level)
        if key in self._levels:
            return key
        if key in self._level_aliases:
            return key
        numeric_levels = sorted(self._levels_by_int.keys())
        if key.isdigit() and numeric_levels:
            requested = int(key)
            closest_num = min(numeric_levels, key=lambda n: abs(n - requested))
            return self._levels_by_int[closest_num][0]
        return self._levels[0]

    def _normalize_aliases(self, alias_cfg):
        if not alias_cfg:
            return {}
        normalized = {}
        for alias, target in alias_cfg.items():
            alias_idx = self._coerce_level_key(alias)
            if isinstance(target, (list, tuple, set)):
                normalized[alias_idx] = [self._coerce_level_key(t) for t in target]
            else:
                normalized[alias_idx] = [self._coerce_level_key(target)]
        return normalized

    def _resolve_level(self, level):
        key = self._coerce_level_key(level)
        if key in self._env.env_list:
            return key
        if key in self._level_aliases:
            for candidate in self._level_aliases[key]:
                cand_key = self._coerce_level_key(candidate)
                if cand_key in self._env.env_list:
                    return cand_key
            raise ValueError(
                f"Alias {key} resolved to {self._level_aliases[key]} but none of those levels exist; "
                f"available levels: {sorted(self._env.env_list.keys())}"
            )
        if key.isdigit() and self._levels_by_int:
            numeric = int(key)
            if numeric in self._levels_by_int:
                return self._levels_by_int[numeric][0]
            closest_num = min(self._levels_by_int.keys(), key=lambda n: abs(n - numeric))
            return self._levels_by_int[closest_num][0]
        if self._levels:
            return self._levels[0]
        raise ValueError("No VGDL levels loaded.")

    def _set_level(self, level):
        self._current_level = self._clip_level(level)
        self._resolved_level = self._resolve_level(self._current_level)
        self._env.lvl = self._resolved_level
        self._env.set_level(self._resolved_level)
