# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import os
import time
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
import ray
import wandb
import hydra
import torch
import multiprocessing

import sys
sys.path.append(os.getcwd())

import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from pathlib import Path
from omegaconf import OmegaConf, open_dict

from ez import agents
from ez.utils.format import set_seed, init_logger
from ez.worker import start_workers, join_workers
from ez.eval import eval


@hydra.main(config_path='./config', config_name='config', version_base='1.1')
def main(config):
    # Allow experiment configs to add new keys before merging.
    OmegaConf.set_struct(config, False)

    if config.exp_config is not None:
        exp_config = OmegaConf.load(config.exp_config)
        OmegaConf.set_struct(exp_config, False)
        # Merge experiment defaults first so CLI overrides stay in control.
        config = OmegaConf.merge(exp_config, config)
        if getattr(config, 'agent_name', None) is None and getattr(exp_config, 'agent_name', None) is not None:
            with open_dict(config):
                config.agent_name = exp_config.agent_name

    if config.ray.single_process:
        config.train.self_play_update_interval = 1
        config.train.reanalyze_update_interval = 1
        config.actors.data_worker = 1
        config.actors.batch_worker = 1
        config.data.num_envs = 1

    if config.ddp.world_size > 1:
        mp.spawn(start_ddp_trainer, args=(config,), nprocs=config.ddp.world_size)
    else:
        start_ddp_trainer(0, config)


def _get_total_memory_bytes():
    """Best-effort detection of total system memory."""
    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            return page_size * phys_pages
        except (ValueError, OSError, AttributeError):
            pass
    return None


def _resolve_object_store_memory(config):
    if hasattr(config.ray, "object_store_memory") and config.ray.object_store_memory is not None:
        return config.ray.object_store_memory

    target = 150 * 1024 * 1024 * 1024 if config.env.image_based else 100 * 1024 * 1024 * 1024
    total = _get_total_memory_bytes()
    if total is None:
        return target

    # Leave enough headroom for the learner and environment processes.
    safety_cap = max(int(total * 0.4), 512 * 1024 * 1024)
    return min(target, safety_cap)


def start_ddp_trainer(rank, config):
    assert rank >= 0
    print(f'start {rank} train worker...')
    agent = agents.names[config.agent_name](config)         # update config
    manager = None
    num_gpus = torch.cuda.device_count()
    num_cpus = multiprocessing.cpu_count()
    object_store_memory = _resolve_object_store_memory(config)
    print(f"Ray object_store_memory={object_store_memory / (1024 ** 3):.2f} GiB")
    ray.init(num_gpus=num_gpus, num_cpus=num_cpus, object_store_memory=object_store_memory)
    set_seed(config.env.base_seed + rank)                  # set seed
    # set log

    if rank == 0:
        # wandb logger
        if config.ddp.training_size == 1:
            wandb_name = config.env.game + '-' + config.wandb.tag
            print(f'wandb_name={wandb_name}')
            logger = wandb.init(
                name=wandb_name,
                project=config.wandb.project,
                # config=config,
            )
        else:
            logger = None
        # file logger
        log_path = os.path.join(config.save_path, 'logs')
        os.makedirs(log_path, exist_ok=True)
        init_logger(log_path)
    else:
        logger = None

    # train
    final_weights = train(rank, agent, manager, logger, config)

    # final evaluation
    if rank == 0:
        model = agent.build_model()
        model.set_weights(final_weights)
        save_path = Path(config.save_path) / 'recordings' / 'final'

        eval_result = eval(agent, model, config.train.eval_n_episode, save_path, config)
        print('final score: ', np.mean(eval_result.scores))
        if config.env.env == 'VGDL':
            print('final win rate: ', eval_result.win_rate)


def train(rank, agent, manager, logger, config):
    # launch for the main process
    if rank == 0:
        workers, server_lst = start_workers(agent, manager, config)
    else:
        workers, server_lst = None, None

    # train
    storage_server, replay_buffer_server, watchdog_server, batch_storage = server_lst

    if config.ddp.training_size == 1:
        final_weights, final_model = agent.train(rank, replay_buffer_server, storage_server, batch_storage, logger)
    else:
        from ez.agents.base import train_ddp
        time.sleep(1)
        train_workers = [
            train_ddp.remote(
                agent, rank * config.ddp.training_size + rank_i,
                replay_buffer_server, storage_server, batch_storage, logger
            ) for rank_i in range(config.ddp.training_size)
        ]
        time.sleep(1)
        final_weights, final_model = ray.get(train_workers)

    eval_result = eval(agent, final_model, config.train.eval_n_episode, Path(config.save_path) / 'evaluation' / 'final', config,
                          max_steps=2700, use_pb=False, verbose=config.eval.verbose)
    print(f'final_mean_score={eval_result.scores.mean():.3f}')
    if config.env.env == 'VGDL':
        print(f'final_win_rate={eval_result.win_rate:.3f}')

    # join process
    if rank == 0:
        print(f'[main process] master worker finished')
        time.sleep(1)
        join_workers(workers, server_lst)

    # return
    dist.destroy_process_group()
    return final_weights


if __name__ == '__main__':
    main()
