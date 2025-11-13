# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import os
import time
import ray
import torch
from omegaconf import OmegaConf
import torch.distributed as dist
import torch.multiprocessing as mp

from ez.worker.watchdog_worker import start_watchdog_server
from ez.data.global_storage import GlobalStorage
from ez.data.replay_buffer import ReplayBuffer
from ez.worker.data_worker import start_data_worker
from ez.worker.batch_worker import start_batch_worker, start_batch_worker_cpu, start_batch_worker_gpu
from ez.worker.eval_worker import start_eval_worker
from ez.utils.format import RayQueue, PreQueue


def start_workers(agent, manager, config):
    # ==================================================================================================================
    # start server
    # ==================================================================================================================

    # global storage server
    storage_server = GlobalStorage.remote(agent.build_model(), agent.build_model(), agent.build_model())
    print('[main process] Global storage server has been started from main process.')
    initial_level = getattr(config.env, 'initial_level', None)
    curriculum_cfg = getattr(config.env, 'curriculum', None)
    if initial_level is None and curriculum_cfg is not None:
        try:
            initial_level = curriculum_cfg.levels[0]
        except AttributeError:
            initial_level = curriculum_cfg.get('levels', [None])[0]
    if initial_level is not None:
        storage_server.update_curriculum_level.remote(int(initial_level))

    # batch queue
    batch_storage = RayQueue(15, 20)
    print('[main process] Batch storage has been initialized.')

    # replay buffer server
    replay_buffer_server = ReplayBuffer.remote(batch_size=config.train.batch_size,
                                               buffer_size=config.data.buffer_size, 
                                               top_transitions=config.data.top_transitions,
                                               use_priority=config.priority.use_priority,
                                               env=config.env.env,
                                               total_transitions=config.data.total_transitions)
    print('[main process] Replay buffer server has been started from main process.')

    # watchdog server
    watchdog_server = start_watchdog_server(manager)
    print('[main process] Watchdog server has been started from main process.')

    # ==================================================================================================================
    # start worker
    # ==================================================================================================================

    # data workers
    data_workers = [start_data_worker(rank, agent, replay_buffer_server, storage_server, config)
                    for rank in range(0, config.actors.data_worker)]
    print('[main process] Data workers have all been launched.')

    # batch worker
    batch_workers = [start_batch_worker(rank, agent, replay_buffer_server, storage_server, batch_storage, config)
                     for rank in range(0, config.actors.batch_worker)]
    print('[main process] Batch workers have all been launched.')

    # eval worker
    eval_worker = [start_eval_worker(agent, replay_buffer_server, storage_server, config)]

    if int(torch.__version__[0]) == 2:
        print(f'[main process] torch version is {torch.__version__}, enabled torch_compile.')

    # trainer (in current process)
    worker_lst = [data_workers, batch_workers, eval_worker]
    server_lst = [storage_server, replay_buffer_server, watchdog_server, batch_storage]

    return worker_lst, server_lst


def join_workers(worker_lst, server_lst):
    data_workers, batch_workers, eval_worker = worker_lst
    storage_server, replay_buffer_server, watchdog_server, smos_server = server_lst

    # wait for all workers to finish
    for data_worker, ref in data_workers:
        ray.get(ref)
    for batch_worker, ref in batch_workers:
        ray.get(ref)
    ray.get(eval_worker[0][1])
    print(f'[main process] All workers have stopped.')

    # stop servers
    storage_server.terminate()
    replay_buffer_server.terminate()
    watchdog_server.terminate()
    smos_server.stop()
    print(f'[main process] All servers have stopped.')
