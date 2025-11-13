from dataclasses import dataclass, field
from typing import Iterable, List, Optional


@dataclass
class CurriculumConfig:
    levels: Iterable[int]
    min_episodes: int = 5
    win_threshold: float = 0.6
    fallback_threshold: Optional[float] = None
    patience: int = 0  # optional hysteresis


class VGDLLevelCurriculum:

    def __init__(self, cfg: CurriculumConfig):
        self.levels: List[int] = sorted(set(cfg.levels))
        if not self.levels:
            raise ValueError("Curriculum levels list cannot be empty.")
        self.cfg = cfg
        self.current_idx = 0
        self.history: List[bool] = []

    @classmethod
    def from_dict(cls, cfg_dict, available_levels):
        cfg = CurriculumConfig(
            levels=cfg_dict.get("levels", available_levels),
            min_episodes=cfg_dict.get("min_episodes", 5),
            win_threshold=cfg_dict.get("win_threshold", 0.6),
            fallback_threshold=cfg_dict.get("fallback_threshold"),
            patience=cfg_dict.get("patience", 0),
        )
        return cls(cfg)

    @property
    def current_level(self):
        return self.levels[self.current_idx]

    def record_episode(self, win: bool):
        self.history.append(win)
        if len(self.history) > max(self.cfg.min_episodes, self.cfg.patience):
            self.history.pop(0)

    def maybe_transition(self):
        stats = {"level": self.current_level, "changed": False}
        if len(self.history) < self.cfg.min_episodes:
            return stats

        win_rate = sum(self.history[-self.cfg.min_episodes:]) / self.cfg.min_episodes
        if win_rate >= self.cfg.win_threshold and self.current_idx < len(self.levels) - 1:
            self.current_idx += 1
            self.history.clear()
            stats.update({"level": self.current_level, "changed": True, "reason": "advance"})
        elif (
            self.cfg.fallback_threshold is not None
            and win_rate <= self.cfg.fallback_threshold
            and self.current_idx > 0
        ):
            self.current_idx -= 1
            self.history.clear()
            stats.update({"level": self.current_level, "changed": True, "reason": "fallback"})
        return stats
