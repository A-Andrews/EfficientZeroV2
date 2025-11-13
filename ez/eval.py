# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import os
import sys

sys.path.append(os.getcwd())

import copy
import multiprocessing
import time
from pathlib import Path

import cv2
import hydra
import imageio
import numpy as np
import ray
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torch.cuda.amp import autocast as autocast
from tqdm.auto import tqdm

from ez import agents, mcts
from ez.envs import make_envs
from ez.mcts.cy_mcts import Gumbel_MCTS
from ez.utils.distribution import SquashedNormal, TruncatedNormal
from ez.utils.format import (
    DiscreteSupport,
    formalize_obs_lst,
    prepare_obs_lst,
    profile,
    symexp,
)


def _write_episode_video(
    video_dir, frames, reward_trace, traj, config, episode_idx, suffix
):
    """Dump a single-episode video (no-op when recording is disabled)."""
    if video_dir is None or not frames:
        return

    writer = imageio.get_writer(video_dir / f"epi_{episode_idx}_{suffix}.mp4")
    overlay_rewards = list(reward_trace)
    if overlay_rewards:
        overlay_rewards[0] = sum(reward_trace)

    for j, frame in enumerate(frames):
        reward_value = overlay_rewards[j] if j < len(overlay_rewards) else 0.0
        frame_to_write = frame
        if config.env.game == "hopper_hop" and j < len(traj.action_lst):
            pil_frame = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_frame)
            draw.text(
                (5, 5),
                f"mu={traj.action_lst[j][0]:.2f},{traj.action_lst[j][1]:.2f}",
            )
            draw.text(
                (5, 20),
                f"{traj.action_lst[j][2]:.2f},{traj.action_lst[j][3]:.2f}",
            )
            draw.text((5, 35), f"r={reward_value:.2f}")
            frame_to_write = np.array(pil_frame)
        writer.append_data(frame_to_write)
    writer.close()


@hydra.main(config_path="./config", config_name="config", version_base="1.1")
def main(config):
    if config.exp_config is not None:
        exp_config = OmegaConf.load(config.exp_config)
        config = OmegaConf.merge(config, exp_config)

    # update config
    agent = agents.names[config.agent_name](config)

    num_gpus = torch.cuda.device_count()
    num_cpus = multiprocessing.cpu_count()
    ray.init(
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        object_store_memory=(
            150 * 1024 * 1024 * 1024
            if config.env.image_based
            else 100 * 1024 * 1024 * 1024
        ),
    )

    # prepare model
    model = agent.build_model()
    if os.path.exists(config.eval.model_path):
        weights = torch.load(config.eval.model_path)
        model.load_state_dict(weights)
        print("resume model from: ", config.eval.model_path)
    if int(torch.__version__[0]) == 2:
        model = torch.compile(model)

    n_episodes = 1
    save_path = Path(config.eval.save_path)

    eval(
        agent,
        model,
        n_episodes,
        save_path,
        config,
        max_steps=config.env.max_episode_steps,
        use_pb=True,
        verbose=config.eval.verbose,
    )


@torch.no_grad()
def eval(
    agent, model, n_episodes, save_path, config, max_steps=None, use_pb=False, verbose=0
):
    model.cuda()
    model.eval()

    # prepare logs
    if save_path is not None:
        video_path = save_path / "recordings"
        video_path.mkdir(parents=True, exist_ok=True)
    else:
        video_path = None

    total_episodes = n_episodes
    parallel_envs_cfg = getattr(config.eval, "parallel_envs", None)
    parallel_envs = (
        parallel_envs_cfg if parallel_envs_cfg is not None else total_episodes
    )
    parallel_envs = max(1, min(parallel_envs, total_episodes))

    # make env
    if max_steps is not None:
        config.env.max_episode_steps = max_steps
    envs = make_envs(
        config.env.env,
        config.env.game,
        parallel_envs,
        config.env.base_seed,
        save_path=video_path,
        episodic_life=False,
        **config.env,
    )

    # initialization
    stack_obs_windows, game_trajs = agent.init_envs(envs, max_steps)

    # set infinity trajectory size
    [traj.set_inf_len() for traj in game_trajs]

    frames = [[] for _ in range(parallel_envs)]
    reward_traces = [[] for _ in range(parallel_envs)]
    episode_returns = []
    episodes_finished = 0
    video_suffix = max_steps if max_steps is not None else config.env.max_episode_steps
    pb = tqdm(total=total_episodes, leave=True) if use_pb else None

    # begin to evaluate
    while episodes_finished < total_episodes:
        # debug
        if verbose:
            import ipdb

            ipdb.set_trace()

        # stack obs
        current_stacked_obs = formalize_obs_lst(
            stack_obs_windows, image_based=config.env.image_based
        )
        # obtain the statistics at current steps
        with torch.no_grad():
            with autocast():
                states, values, policies = model.initial_inference(current_stacked_obs)

        values = values.detach().cpu().numpy().flatten()

        # tree search for policies
        tree = mcts.names[config.mcts.language](
            # num_actions=config.env.action_space_size if config.env.env == 'Atari' else config.mcts.num_top_actions,
            num_actions=(
                config.env.action_space_size
                if config.env.env in ("Atari", "VGDL")
                else config.mcts.num_sampled_actions
            ),
            discount=config.rl.discount,
            env=config.env.env,
            **config.mcts,  # pass mcts related params
            **config.model,  # pass the value and reward support params
        )
        if config.env.env in ("Atari", "VGDL"):
            if config.mcts.use_gumbel:
                r_values, r_policies, best_actions, _ = tree.search(
                    model,
                    parallel_envs,
                    states,
                    values,
                    policies,
                    use_gumble_noise=False,
                    verbose=verbose,
                )
            else:
                r_values, r_policies, best_actions, _ = tree.search_ori_mcts(
                    model, parallel_envs, states, values, policies, use_noise=False
                )
        else:
            r_values, r_policies, best_actions, _, _, _ = tree.search_continuous(
                model,
                parallel_envs,
                states,
                values,
                policies,
                use_gumble_noise=False,
                verbose=verbose,
                add_noise=False,
            )

        stop_eval = False

        # step action in environments
        for i in range(parallel_envs):
            if episodes_finished >= total_episodes:
                stop_eval = True
                break

            action = best_actions[i]
            obs, reward, done, info = envs[i].step(action)
            if config.env.env == "VGDL":
                video_frame = envs[i].unwrapped.render(mode="rgb_array")
            else:
                video_frame = obs if config.env.image_based else envs[i].render(mode="rgb_array")
            frames[i].append(video_frame)
            reward_value = info.get("raw_reward", reward)
            reward_traces[i].append(reward_value)

            # save data to trajectory buffer
            game_trajs[i].store_search_results(values[i], r_values[i], r_policies[i])
            game_trajs[i].append(action, obs, reward)
            if (
                config.env.env == "Atari"
            ):  # keep this as just Atari as VGDL does not have ALE
                game_trajs[i].snapshot_lst.append(envs[i].ale.cloneState())
            elif hasattr(envs[i], "physics"): # VGDL games dont have physics so this should prevent the video from failing
                game_trajs[i].snapshot_lst.append(envs[i].physics.get_state())
            else:
                game_trajs[i].snapshot_lst.append(None)

            del stack_obs_windows[i][0]
            stack_obs_windows[i].append(obs)

            if done:
                finished_traj = game_trajs[i]
                total_reward = sum(reward_traces[i]) if reward_traces[i] else 0.0
                episode_returns.append(total_reward)
                _write_episode_video(
                    video_path,
                    frames[i],
                    reward_traces[i],
                    finished_traj,
                    config,
                    episodes_finished,
                    video_suffix,
                )
                episodes_finished += 1

                if pb:
                    avg_reward = (
                        float(np.mean(episode_returns)) if episode_returns else 0.0
                    )
                    pb.set_description(
                        f"{config.env.game} eval {episodes_finished}/{total_episodes} avg={avg_reward:.3f}"
                    )
                    pb.update(1)

                if episodes_finished >= total_episodes:
                    stop_eval = True
                    break

                stacked_obs, traj = agent.init_env(envs[i], max_steps)
                stack_obs_windows[i] = stacked_obs
                game_trajs[i] = traj
                game_trajs[i].set_inf_len()
                frames[i] = []
                reward_traces[i] = []

        if stop_eval:
            break

    if pb:
        pb.close()

    [env.close() for env in envs]

    return np.asarray(episode_returns)


if __name__ == "__main__":
    main()
