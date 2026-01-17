from dataclasses import dataclass, field
from typing import Iterable, List, Optional


@dataclass
class CurriculumConfig:
    levels: Iterable[int]
    min_episodes: int = 5
    win_threshold: float = 0.6
    fallback_threshold: Optional[float] = None
    patience: int = 0  # optional hysteresis
    force_advance_steps: Optional[int] = None


class VGDLLevelCurriculum:

    def __init__(self, cfg: CurriculumConfig):
        unique_levels = []
        seen = set()
        for lvl in cfg.levels:
            if lvl in seen:
                continue
            unique_levels.append(lvl)
            seen.add(lvl)
        self.levels: List[int] = unique_levels
        if not self.levels:
            raise ValueError("Curriculum levels list cannot be empty.")
        self.cfg = cfg
        self.current_idx = 0
        self.history: List[bool] = []
        self.steps_since_transition = 0

    @classmethod
    def from_dict(cls, cfg_dict, available_levels):
        cfg = CurriculumConfig(
            levels=cfg_dict.get("levels", available_levels),
            min_episodes=cfg_dict.get("min_episodes", 5),
            win_threshold=cfg_dict.get("win_threshold", 0.6),
            fallback_threshold=cfg_dict.get("fallback_threshold"),
            patience=cfg_dict.get("patience", 0),
            force_advance_steps=cfg_dict.get("force_advance_steps"),
        )
        return cls(cfg)

    @property
    def current_level(self):
        return self.levels[self.current_idx]

    def record_episode(self, win: bool, steps: Optional[int] = None):
        self.history.append(win)
        if len(self.history) > max(self.cfg.min_episodes, self.cfg.patience):
            self.history.pop(0)
        if steps is not None:
            try:
                steps_int = int(steps)
            except (TypeError, ValueError):
                steps_int = 0
            if steps_int > 0:
                self.steps_since_transition += steps_int

    def _reset_level_stats(self):
        self.history.clear()
        self.steps_since_transition = 0

    def _should_force_advance(self):
        max_steps = self.cfg.force_advance_steps
        return (
            max_steps is not None
            and max_steps > 0
            and self.steps_since_transition >= max_steps
            and self.current_idx < len(self.levels) - 1
        )

    def maybe_transition(self):
        stats = {"level": self.current_level, "changed": False}
        if self._should_force_advance():
            self.current_idx += 1
            self._reset_level_stats()
            stats.update({"level": self.current_level, "changed": True, "reason": "forced"})
            return stats
        if len(self.history) < self.cfg.min_episodes:
            return stats

        win_rate = sum(self.history[-self.cfg.min_episodes:]) / self.cfg.min_episodes
        if win_rate >= self.cfg.win_threshold and self.current_idx < len(self.levels) - 1:
            self.current_idx += 1
            self._reset_level_stats()
            stats.update({"level": self.current_level, "changed": True, "reason": "advance"})
        elif (
            self.cfg.fallback_threshold is not None
            and win_rate <= self.cfg.fallback_threshold
            and self.current_idx > 0
        ):
            self.current_idx -= 1
            self._reset_level_stats()
            stats.update({"level": self.current_level, "changed": True, "reason": "fallback"})
        return stats
