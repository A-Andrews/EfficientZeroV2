#!/usr/bin/env python3
"""
Evaluate a VGDL checkpoint on every level and log results to Weights & Biases.

Typical usage:
python scripts/vgdl_eval_levels.py \
  --ckpt results/VGDL/vgfmri4_zelda/EZ-V2-seed=0-.../models/model_100000.p \
  --game vgfmri4_zelda \
  --episodes 10
"""

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional

import ray
import torch
import wandb
from omegaconf import OmegaConf, open_dict

from ez import agents
from ez.eval import eval as eval_fn
from ez.envs import make_vgdl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a VGDL checkpoint on all levels.")
    parser.add_argument("--ckpt", required=True, help="Path to the model checkpoint (.p).")
    parser.add_argument("--game", required=True, help="VGDL game name (e.g., vgfmri4_zelda).")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per level.")
    parser.add_argument(
        "--exp-config",
        default="ez/config/exp/vgdl.yaml",
        help="Experiment config path merged on top of ez/config/config.yaml.",
    )
    parser.add_argument(
        "--levels",
        default=None,
        help="Comma-separated list of level indices to evaluate. "
        "Defaults to curriculum levels or all levels found in the VGDL env.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Base save path for eval outputs. "
        "Defaults to results/VGDL/<game>/eval_levels/<ckpt-stem>.",
    )
    parser.add_argument("--wandb-project", default="ez-v2-evals", help="Weights & Biases project.")
    parser.add_argument("--wandb-entity", default=None, help="Weights & Biases entity (optional).")
    parser.add_argument(
        "--wandb-tags",
        default="vgdl,eval",
        help="Comma-separated W&B tags (vgdl,eval will always be included).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional W&B run name. Defaults to <game>-eval-<ckpt-stem>-<timestamp>.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra-style dotlist overrides (can be specified multiple times).",
    )
    return parser.parse_args()


def resolve_levels(cfg, game: str, levels_arg: Optional[str]) -> List[int]:
    """Decide which levels to evaluate."""
    if levels_arg:
        return [int(x) for x in levels_arg.split(",") if x]

    curriculum = getattr(cfg.env, "curriculum", None)
    if curriculum and getattr(curriculum, "levels", None):
        return list(curriculum.levels)

    raw_env = make_vgdl(game, seed=0, save_path=None, **cfg.env)
    try:
        levels = list(getattr(raw_env.unwrapped, "_levels", []))
    finally:
        raw_env.close()

    if not levels:
        raise ValueError(f"Could not infer levels for game {game}. Pass --levels explicitly.")
    return levels


def parse_tags(tag_str: str) -> List[str]:
    tags = [t.strip() for t in tag_str.split(",") if t.strip()]
    tags = list({*tags, "vgdl", "eval"})
    return tags


def init_wandb(args, levels: Iterable[int]):
    ckpt_name = Path(args.ckpt).stem
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.name or f"{args.game}-eval-{ckpt_name}-{timestamp}"
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        tags=parse_tags(args.wandb_tags),
        config={
            "game": args.game,
            "checkpoint": args.ckpt,
            "episodes_per_level": args.episodes,
            "levels": list(levels),
            "exp_config": args.exp_config,
            "overrides": args.override,
        },
    )


def build_config(args) -> OmegaConf:
    base_cfg = OmegaConf.load("ez/config/config.yaml")
    exp_cfg = OmegaConf.load(args.exp_config)
    cfg = OmegaConf.merge(base_cfg, exp_cfg)

    if args.override:
        override_cfg = OmegaConf.from_dotlist(args.override)
        cfg = OmegaConf.merge(cfg, override_cfg)

    default_save = Path("results") / "VGDL" / args.game / "eval_levels" / Path(args.ckpt).stem
    with open_dict(cfg):
        cfg.eval.model_path = args.ckpt
        cfg.eval.save_path = str(args.save_path or default_save)
        cfg.env.game = args.game
        cfg.env.curriculum = None
        cfg.train.eval_n_episode = args.episodes

    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)
    levels = resolve_levels(cfg, args.game, args.levels)

    run = init_wandb(args, levels)
    ray.init(
        num_gpus=torch.cuda.device_count(),
        num_cpus=os.cpu_count(),
        object_store_memory=(
            150 * 1024 * 1024 * 1024 if cfg.env.image_based else 100 * 1024 * 1024 * 1024
        ),
    )

    agent = agents.names[cfg.agent_name](cfg)
    model = agent.build_model()
    state_dict = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state_dict)

    results = {}
    try:
        for level in levels:
            with open_dict(cfg):
                cfg.env.initial_level = int(level)
            level_save = Path(cfg.eval.save_path) / f"level_{level}"
            level_save.mkdir(parents=True, exist_ok=True)
            scores = eval_fn(
                agent,
                model,
                args.episodes,
                level_save,
                cfg,
                max_steps=cfg.env.max_episode_steps,
                use_pb=False,
                verbose=getattr(cfg.eval, "verbose", 0),
            )
            stats = {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "min": float(scores.min()),
                "max": float(scores.max()),
            }
            results[level] = stats
            wandb.log(
                {
                    "level": level,
                    "return_mean": stats["mean"],
                    "return_std": stats["std"],
                    "return_min": stats["min"],
                    "return_max": stats["max"],
                }
            )

        overall = {
            "overall_mean": float(sum(v["mean"] for v in results.values()) / len(results)),
            "overall_min": float(min(v["min"] for v in results.values())),
            "overall_max": float(max(v["max"] for v in results.values())),
        }
        wandb.log(overall)
    finally:
        run.finish(quiet=True)
        ray.shutdown()


if __name__ == "__main__":
    main()
