import os
import sys
from typing import Dict, Optional, Sequence, Tuple

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
        enable_level_augmentation: bool = False,
        level_augmentations: Optional[Sequence[str]] = None,
        reward_shaping: Optional[Dict[str, float]] = None,
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

        self.shaping_cfg: Dict[str, float] = dict(reward_shaping or {})
        self._shaping_enabled = bool(self.shaping_cfg)
        self._shaping_initial_counts: Dict[str, int] = {}
        self._shaping_distance_norm: float = 1.0
        self._prev_potential: float = 0.0

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
        # Track the last transform applied so it can be reported to callers.
        self._current_level_transform = "identity"

        default_aug_list = (
            "identity",
            "rotate_90",
            "rotate_180",
            "rotate_270",
            "mirror_horizontal",
            "mirror_vertical",
        )
        if level_augmentations is None:
            level_aug_choices = default_aug_list
        else:
            level_aug_choices = tuple(level_augmentations)
        valid_augs = set(default_aug_list)
        for aug_name in level_aug_choices:
            if aug_name not in valid_augs:
                raise ValueError(
                    f"Unsupported level augmentation '{aug_name}'. Valid options: {sorted(valid_augs)}"
                )
        self._level_aug_choices = level_aug_choices
        self._use_level_aug = bool(enable_level_augmentation and self._level_aug_choices)


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

    @property
    def num_levels(self) -> int:
        return int(self._num_levels)

    @property
    def current_level(self) -> Optional[int]:
        return self._current_level

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

        transform_name = "identity"
        if self._use_level_aug:
            transform_name = self._level_rng.choice(self._level_aug_choices)

        def _maybe_transform_level(level_string, lvl_idx, *args):
            if lvl_idx != level or transform_name == "identity":
                return level_string
            return self._apply_level_transform(level_string, transform_name)

        try:
            self._env.reset(level_transform=_maybe_transform_level)
        except TypeError:
            # Backwards compatibility if the legacy env does not accept the kwarg.
            self._env.reset()
        self._sync_level_metadata()
        self._elapsed_steps = 0
        self._current_level = getattr(self._env, "lvl", level)
        self._current_level_transform = transform_name

        frame = self._preprocess(self._render_frame())
        obs = self._format_obs(frame)

        if self._shaping_enabled:
            self._cache_initial_shaping_state()
            potential, _ = self._compute_potential()
            self._prev_potential = potential

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
            "level_transform": self._current_level_transform,
        }

        base_reward = float(reward)
        if self.clip_reward:
            reward = np.sign(reward)

        shaped_reward = float(reward)
        if self._shaping_enabled:
            potential, components = self._compute_potential()
            delta = potential - self._prev_potential
            scale = float(self.shaping_cfg.get("scale", 1.0))
            if scale != 0.0:
                shaped_reward += scale * delta
            step_bonus = float(self.shaping_cfg.get("step_penalty", 0.0))
            if step_bonus != 0.0:
                shaped_reward += step_bonus
                components["step_penalty"] = float(step_bonus)
            win_bonus = float(self.shaping_cfg.get("win_bonus", 0.0)) if win else 0.0
            if win_bonus != 0.0:
                shaped_reward += win_bonus
                components["win_bonus"] = float(win_bonus)
            info["shaping/potential"] = float(potential)
            info["shaping/delta"] = float(delta)
            if components:
                info["shaping/components"] = {k: float(v) for k, v in components.items()}
            self._prev_potential = potential
        info["raw_reward"] = base_reward

        done = terminated or truncated
        return obs, float(shaped_reward), bool(done), info

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

    @staticmethod
    def _get_first_sprite(sprites) -> Optional[object]:
        if not sprites:
            return None
        try:
            return sprites[0]
        except (TypeError, IndexError):
            return None

    def _get_game(self):
        current_env = getattr(self._env, "current_env", None)
        if current_env is None:
            return None
        return getattr(current_env, "_game", None)

    def _cache_initial_shaping_state(self) -> None:
        if not self._shaping_enabled:
            return
        game = self._get_game()
        if game is None:
            self._shaping_initial_counts = {}
            self._shaping_distance_norm = 1.0
            return
        sprite_groups = getattr(game, "sprite_groups", {})

        def _count(name: str) -> int:
            group = sprite_groups.get(name, [])
            try:
                return len(group)
            except TypeError:
                return 0

        self._shaping_initial_counts = {
            "box": _count("box"),
            "mushroom": _count("mushroom"),
            "key": _count("key"),
        }
        screensize = getattr(game, "screensize", (84, 84))
        try:
            width, height = int(screensize[0]), int(screensize[1])
        except Exception:
            width, height = 84, 84
        norm = max(width + height, 1)
        self._shaping_distance_norm = float(norm)

    def _compute_potential(self) -> Tuple[float, Dict[str, float]]:
        if not self._shaping_enabled:
            return 0.0, {}
        game = self._get_game()
        if game is None:
            return 0.0, {}
        sprite_groups = getattr(game, "sprite_groups", {})
        avatar = self._get_first_sprite(sprite_groups.get("avatar"))
        if avatar is None:
            return 0.0, {}

        components: Dict[str, float] = {}
        potential = 0.0

        def _count(name: str) -> int:
            group = sprite_groups.get(name, [])
            try:
                return len(group)
            except TypeError:
                return 0

        def _manhattan_distance(src, dst) -> float:
            try:
                ax, ay = src
                bx, by = dst
            except Exception:
                return 0.0
            return float(abs(ax - bx) + abs(ay - by))

        avatar_center = getattr(getattr(avatar, "rect", None), "center", None)

        init_boxes = self._shaping_initial_counts.get("box", 0)
        if init_boxes > 0:
            current_boxes = _count("box")
            delta_boxes = init_boxes - current_boxes
            weight = float(self.shaping_cfg.get("w_box", 0.0))
            if weight and delta_boxes:
                value = weight * float(delta_boxes)
                potential += value
                components["box_progress"] = float(value)

        init_mushrooms = self._shaping_initial_counts.get("mushroom", 0)
        if init_mushrooms > 0:
            current_mushrooms = _count("mushroom")
            delta_mushrooms = init_mushrooms - current_mushrooms
            weight = float(self.shaping_cfg.get("w_mushroom", 0.0))
            if weight and delta_mushrooms:
                value = weight * float(delta_mushrooms)
                potential += value
                components["mushroom_progress"] = float(value)

        has_key = False
        avatar_resources = getattr(avatar, "resources", {})
        if isinstance(avatar_resources, dict):
            has_key = avatar_resources.get("key", 0) > 0

        distance_norm = self._shaping_distance_norm

        if avatar_center is not None and not has_key:
            key_sprite = self._get_first_sprite(sprite_groups.get("key"))
            if key_sprite is not None:
                key_center = getattr(getattr(key_sprite, "rect", None), "center", None)
                if key_center is not None:
                    dist = _manhattan_distance(avatar_center, key_center)
                    weight = float(self.shaping_cfg.get("w_key_dist", 0.0))
                    if weight:
                        value = weight * max(distance_norm - dist, 0.0) / distance_norm
                        potential += value
                        components["key_distance"] = float(value)
        elif avatar_center is not None and has_key:
            goal_sprite = self._get_first_sprite(sprite_groups.get("goal"))
            if goal_sprite is not None:
                goal_center = getattr(getattr(goal_sprite, "rect", None), "center", None)
                if goal_center is not None:
                    dist = _manhattan_distance(avatar_center, goal_center)
                    weight = float(self.shaping_cfg.get("w_goal_dist", 0.0))
                    if weight:
                        value = weight * max(distance_norm - dist, 0.0) / distance_norm
                        potential += value
                        components["goal_distance"] = float(value)

        return potential, components

    @staticmethod
    def _apply_level_transform(layout: str, transform_name: str) -> str:
        lines = layout.split('\n')
        if not lines or lines == ['']:
            return layout
        row_lengths = {len(line) for line in lines if line}
        if len(row_lengths) > 1:
            target_width = max(row_lengths)
            lines = [line.ljust(target_width, 'w') for line in lines]
        array = np.array([list(line) for line in lines], dtype='<U1')

        if transform_name == "identity":
            transformed = array.copy()
        elif transform_name == "rotate_90":
            transformed = np.rot90(array, k=1).copy()
        elif transform_name == "rotate_180":
            transformed = np.rot90(array, k=2).copy()
        elif transform_name == "rotate_270":
            transformed = np.rot90(array, k=3).copy()
        elif transform_name == "mirror_horizontal":
            transformed = np.fliplr(array).copy()
        elif transform_name == "mirror_vertical":
            transformed = np.flipud(array).copy()
        else:
            raise ValueError(f"Unknown level transform '{transform_name}'")

        return '\n'.join(''.join(row.tolist()) for row in transformed)
