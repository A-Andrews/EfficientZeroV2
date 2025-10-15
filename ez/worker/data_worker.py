# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import copy
import os
import time
import ray
import torch
import numpy as np

from torch.nn import L1Loss
from pathlib import Path
from torch.cuda.amp import autocast as autocast

from ez.worker.base import Worker
from ez import mcts
from ez.envs import make_envs, make_env
from ez.utils.format import formalize_obs_lst, DiscreteSupport, allocate_gpu, prepare_obs_lst, symexp
from ez.mcts.cy_mcts import Gumbel_MCTS

# @ray.remote(num_gpus=0.05)
@ray.remote(num_gpus=0.05)
class DataWorker(Worker):
    def __init__(self, rank, agent, replay_buffer, storage, config):
        super().__init__(rank, agent, replay_buffer, storage, config)

        self.model_update_interval = config.train.self_play_update_interval
        self.traj_pool = []
        self.pool_size = 1

        curriculum_cfg = getattr(config.env, 'curriculum', None)
        self.curriculum_cfg = curriculum_cfg
        self.curriculum_enabled = bool(curriculum_cfg and getattr(curriculum_cfg, 'enabled', False) and config.env.env == 'VGDL')
        self.curriculum_configured = False
        try:
            start_level = int(getattr(curriculum_cfg, 'start_level', 0)) if self.curriculum_enabled else 0
        except Exception:
            start_level = 0
        self.curriculum_level = max(0, start_level)
        base_warmup = getattr(self.config.train, 'random_warmup_steps', 0)
        self.initial_random_warmup_steps = max(0, int(base_warmup or 0))
        level_warmup = getattr(self.config.train, 'new_level_random_steps', 0)
        self.new_level_random_steps = max(0, int(level_warmup or 0))

        # time.sleep(10000)

    @torch.no_grad()
    def run(self):
        config = self.config

        # create the model for self-play data collection
        self.model = self.agent.build_model()
        self.model.cuda()
        if int(torch.__version__[0]) == 2:
            self.model = torch.compile(self.model)
        self.model.eval()
        self.resume_model()

        # make env
        num_envs = config.data.num_envs
        save_path = Path(config.save_path)
        if config.data.save_video:
            video_path = save_path / 'self_play_videos'
        else:
            video_path = None
        cur_seed = config.env.base_seed

        envs = make_envs(config.env.env, config.env.game, num_envs, cur_seed + self.rank * num_envs,
                         save_path=video_path, episodic_life=config.env.episodic, **config.env)   # prev episodic_life=True
        self._setup_curriculum(envs)

        warmup_threshold = self.initial_random_warmup_steps
        approx_env_steps = 0
        level_random_counters = [0 for _ in range(num_envs)]

        # initialization
        trained_steps = 0           # current training steps
        collected_transitions = ray.get(self.replay_buffer.get_transition_num.remote())   # total transitions collected
        start_training = False      # is training
        max_transitions = config.data.total_transitions // config.actors.data_worker  # max transitions to collect in this worker
        dones = [False for _ in range(num_envs)]
        traj_len = [0 for _ in range(num_envs)]

        stack_obs_windows = []
        game_trajs = []
        env_levels = []
        for idx, env in enumerate(envs):
            stacked_obs, traj, level = self._init_env_state(env, level=self.curriculum_level if self.curriculum_enabled else None)
            stack_obs_windows.append(stacked_obs)
            game_trajs.append(traj)
            parsed_level = self._parse_level_value(level)
            env_levels.append(parsed_level)
        prev_game_trajs = [None for _ in range(num_envs)]  # previous game trajectories (split a full game trajectory into several sub trajectories)
        episode_win = [False for _ in range(num_envs)]

        # log data
        episode_return = [0. for _ in range(num_envs)]
        prev_train_steps = -1

        # while loop for collecting data
        while not self.is_finished(trained_steps):
            trained_steps = ray.get(self.storage.get_counter.remote())
            if not start_training:
                start_training = ray.get(self.storage.get_start_signal.remote())

            # get the fresh model weights
            self.get_recent_model(trained_steps, 'self_play')

            if collected_transitions > max_transitions:
                time.sleep(10)
                continue

            # self-play is faster than training speed or finished
            if start_training and (collected_transitions / max_transitions) > (trained_steps / self.config.train.training_steps):
                time.sleep(1)
                continue

            if self.config.ray.single_process:
                trained_steps = ray.get(self.storage.get_counter.remote())
                if start_training and trained_steps <= prev_train_steps:
                    time.sleep(0.1)
                    continue
                prev_train_steps = trained_steps
                print(f'selfplay[{self.rank}] rollouts at step {trained_steps}, collected transitions {collected_transitions}')

            # print('self-playing')
            # temperature
            temperature = self.agent.get_temperature(trained_steps=trained_steps) #* np.ones((num_envs, 1))

            # stack obs
            current_stacked_obs = formalize_obs_lst(stack_obs_windows, image_based=config.env.image_based)
            # obtain the statistics at current steps
            with autocast():
                states, values, policies = self.model.initial_inference(current_stacked_obs)

            # process outputs
            values = values.detach().cpu().numpy().flatten()

            if collected_transitions % 200 == 0 and self.config.model.noisy_net and self.rank == 0:
                print('*******************************')
                print(f'w_ep={self.model.value_policy_model.pi_net[0].weight_epsilon.mean()}')
                print(f'w_mu={self.model.value_policy_model.pi_net[0].weight_mu.mean()}')
                print(f'w_si={self.model.value_policy_model.pi_net[0].weight_sigma.mean()}')
                print(f'b_ep={self.model.value_policy_model.pi_net[0].bias_epsilon.mean()}')
                print(f'b_mu={self.model.value_policy_model.pi_net[0].bias_mu.mean()}')
                print(f'b_si={self.model.value_policy_model.pi_net[0].bias_sigma.mean()}')

            # tree search for policies
            tree = mcts.names[config.mcts.language](
                # num_actions=config.env.action_space_size if config.env.env in ('Atari', 'VGDL') else config.mcts.num_top_actions,
                num_actions=config.env.action_space_size if config.env.env in ('Atari', 'VGDL') else config.mcts.num_sampled_actions,
                discount=config.rl.discount,
                env=config.env.env,
                **config.mcts,  # pass mcts related params
                **config.model,  # pass the value and reward support params
            )
            warmup_active = warmup_threshold > 0 and approx_env_steps < warmup_threshold
            if warmup_active:
                random_envs = [True for _ in range(num_envs)]
            else:
                random_envs = [level_random_counters[i] > 0 for i in range(num_envs)]
            any_random = any(random_envs)
            all_random = all(random_envs) if num_envs > 0 else False
            action_space_size = self.config.env.action_space_size if self.config.env.env in ('Atari', 'VGDL') else None

            if all_random and action_space_size is not None:
                best_actions = [envs[i].action_space.sample() for i in range(num_envs)]
                r_values = np.zeros(num_envs, dtype=np.float32)
                r_policies = np.full((num_envs, action_space_size), 1.0 / max(1, action_space_size), dtype=np.float32)
                for idx in range(num_envs):
                    if not warmup_active and level_random_counters[idx] > 0:
                        level_random_counters[idx] -= 1
            else:
                if self.config.env.env in ('Atari', 'VGDL'):
                    if self.config.mcts.use_gumbel:
                        r_values, r_policies, best_actions, _ = tree.search(self.model, num_envs, states, values, policies,
                                                                            # use_gumble_noise=False, # for test search
                                                                            temperature=temperature)
                    else:
                        r_values, r_policies, best_actions, _ = tree.search_ori_mcts(self.model, num_envs, states, values, policies,
                                                                                        use_noise=True, temperature=temperature)
                else:
                    r_values, r_policies, best_actions, sampled_actions, best_indexes, mcts_info = tree.search_continuous(
                            self.model, num_envs, states, values, policies, temperature=temperature,
                            # use_gumble_noise=True,
                            input_noises=None
                        )
                best_actions = list(best_actions)
                if any_random and action_space_size is not None:
                    uniform_policy = np.full((action_space_size,), 1.0 / max(1, action_space_size), dtype=np.float32)
                    for idx, need_random in enumerate(random_envs):
                        if not need_random:
                            continue
                        best_actions[idx] = envs[idx].action_space.sample()
                        r_values[idx] = 0.0
                        r_policies[idx] = uniform_policy
                        if not warmup_active and level_random_counters[idx] > 0:
                            level_random_counters[idx] -= 1
            if self.rank == 0 and self.config.env.env == 'VGDL' and self.config.log.log_interval > 0 \
                    and collected_transitions % max(1, self.config.log.log_interval // max(1, num_envs)) == 0:
                action_hist = np.bincount(np.asarray(best_actions, dtype=np.int64),
                                          minlength=self.config.env.action_space_size)
                total_actors = max(1, num_envs)
                action_freq_logs = {
                    f'self_play/action_frac_{act}': float(action_hist[act] / total_actors)
                    for act in range(len(action_hist))
                }
                self.storage.add_log_scalar.remote(action_freq_logs)
                hist_samples = np.repeat(np.arange(len(action_hist), dtype=np.float32), action_hist.astype(np.int64))
                if hist_samples.size > 0:
                    self.storage.add_log_distribution.remote({
                        'dist/self_play_action_hist': hist_samples
                    })

            # step action in environments
            for i in range(num_envs):
                action = best_actions[i]
                obs, reward, done, info = envs[i].step(action)
                dones[i] = done
                traj_len[i] += 1
                episode_return[i] += info['raw_reward']
                episode_win[i] = episode_win[i] or bool(info.get('win', False))
                prev_level = env_levels[i]
                current_level_val = info.get('level', prev_level)
                parsed_level = self._parse_level_value(current_level_val)
                if parsed_level is not None:
                    if parsed_level != prev_level and self.new_level_random_steps > 0:
                        level_random_counters[i] = self.new_level_random_steps
                    env_levels[i] = parsed_level
                elif self.curriculum_enabled:
                    env_levels[i] = current_level_val

                # save data to trajectory buffer
                game_trajs[i].store_search_results(values[i], r_values[i], r_policies[i])
                game_trajs[i].append(action, obs, reward)
                # game_trajs[i].raw_obs_lst.append(obs)
                if self.config.env.env in ('Atari', 'VGDL'):
                    game_trajs[i].snapshot_lst.append([])
                else:
                    game_trajs[i].snapshot_lst.append([])

                # fresh stack windows
                del stack_obs_windows[i][0]
                stack_obs_windows[i].append(obs)

                # if current trajectory is full; we will save the previous trajectory
                if game_trajs[i].is_full():
                    if prev_game_trajs[i] is not None:
                        self.save_previous_trajectory(i, prev_game_trajs, game_trajs,
                                                      # padding=not dones[i]
                                                      )

                    prev_game_trajs[i] = game_trajs[i]

                    # new trajectory
                    game_trajs[i] = self.agent.new_game(max_steps=self.config.data.trajectory_size)
                    game_trajs[i].init(stack_obs_windows[i])

                    traj_len[i] = 0

                approx_env_steps += 1

                # reset an env if done
                if dones[i]:
                    # save the previous trajectory
                    if prev_game_trajs[i] is not None:
                        self.save_previous_trajectory(i, prev_game_trajs, game_trajs,
                                                      # padding=False
                                                      )

                    if len(game_trajs[i]) > 0:
                        # save current trajectory
                        game_trajs[i].pad_over([], [], [], [], [])
                        game_trajs[i].save_to_memory()
                        self.put_trajs(game_trajs[i])

                    # log
                    self.storage.add_log_scalar.remote({
                        'self_play/episode_len': traj_len[i],
                        'self_play/episode_return': episode_return[i],
                        'self_play/temperature': temperature
                    })

                    if self.curriculum_enabled:
                        self._report_curriculum_episode(env_levels[i], episode_win[i], traj_len[i], episode_return[i], envs[i])

                    # reset the finished env and new a env
                    if self.config.env.env == 'DMC':
                        envs[i] = make_env(config.env.env, config.env.game, num_envs, cur_seed + self.rank * num_envs,
                             save_path=video_path, episodic_life=config.env.episodic, **config.env)
                    prev_level_val = env_levels[i]
                    stacked_obs, traj, level = self._init_env_state(envs[i], level=self.curriculum_level if self.curriculum_enabled else None)
                    stack_obs_windows[i] = stacked_obs
                    game_trajs[i] = traj
                    parsed_level = self._parse_level_value(level)
                    env_levels[i] = parsed_level
                    if self.new_level_random_steps > 0 and parsed_level is not None:
                        if parsed_level != prev_level_val:
                            level_random_counters[i] = self.new_level_random_steps
                        else:
                            level_random_counters[i] = 0
                    else:
                        if parsed_level is None:
                            level_random_counters[i] = 0
                    prev_game_trajs[i] = None

                    traj_len[i] = 0
                    episode_return[i] = 0
                    episode_win[i] = False
                collected_transitions += 1

    def _setup_curriculum(self, envs):
        if not self.curriculum_enabled or self.curriculum_configured:
            return
        num_levels = None
        if envs:
            primary = envs[0]
            num_levels = getattr(primary, 'num_levels', getattr(primary, '_num_levels', None))
        target_win_rate = getattr(self.curriculum_cfg, 'target_win_rate', 0.0) if self.curriculum_cfg else 0.0
        min_episodes = getattr(self.curriculum_cfg, 'min_episodes', 0) if self.curriculum_cfg else 0
        recent_window = getattr(self.curriculum_cfg, 'recent_window', None) if self.curriculum_cfg else None
        max_episodes = getattr(self.curriculum_cfg, 'max_episodes', None) if self.curriculum_cfg else None
        try:
            min_episodes = int(min_episodes)
        except Exception:
            min_episodes = 0
        try:
            target_win_rate = float(target_win_rate)
        except Exception:
            target_win_rate = 0.0
        try:
            recent_window = int(recent_window) if recent_window is not None else None
        except Exception:
            recent_window = None
        if recent_window is None or recent_window <= 0:
            recent_window = min_episodes if min_episodes > 0 else 1
        try:
            max_episodes = int(max_episodes) if max_episodes is not None else None
        except Exception:
            max_episodes = None
        if max_episodes is not None and max_episodes <= 0:
            max_episodes = None
        settings = {
            'enabled': True,
            'target_win_rate': target_win_rate,
            'min_episodes': max(0, min_episodes),
            'recent_window': max(1, recent_window),
            'max_episodes': max_episodes,
            'start_level': self.curriculum_level,
            'num_levels': num_levels,
        }
        result = ray.get(self.storage.configure_curriculum.remote(settings))
        if not result.get('enabled', False):
            self.curriculum_enabled = False
        else:
            self.curriculum_level = int(result.get('current_level', self.curriculum_level))
        self.curriculum_configured = True

    @staticmethod
    def _parse_level_value(level):
        if level is None:
            return None
        try:
            return int(level)
        except (TypeError, ValueError):
            try:
                return int(float(level))
            except (TypeError, ValueError):
                return None

    def _init_env_state(self, env, level=None):
        if self.curriculum_enabled:
            return self._reset_env_for_curriculum(env, level=level)
        stacked_obs, traj = self.agent.init_env(env, max_steps=self.config.data.trajectory_size)
        current_level = getattr(env, 'current_level', None)
        if current_level is None and hasattr(env, 'lvl'):
            current_level = getattr(env, 'lvl')
        return stacked_obs, traj, current_level

    def _reset_env_for_curriculum(self, env, level=None):
        reset_kwargs = {}
        if level is not None:
            try:
                reset_kwargs['options'] = {'level': int(level)}
            except Exception:
                reset_kwargs['options'] = {'level': 0}
        try:
            obs = env.reset(**reset_kwargs)
        except TypeError:
            reset_kwargs.pop('options', None)
            obs = env.reset(**reset_kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]
        stacked_obs = [obs for _ in range(self.config.env.n_stack)]
        traj = self.agent.new_game(max_steps=self.config.data.trajectory_size)
        traj.init(stacked_obs)
        current_level = getattr(env, 'current_level', None)
        if current_level is None and hasattr(env, '_current_level'):
            current_level = getattr(env, '_current_level')
        if current_level is None and hasattr(env, 'lvl'):
            current_level = getattr(env, 'lvl')
        if current_level is None:
            current_level = level
        return stacked_obs, traj, current_level

    def _report_curriculum_episode(self, level, win, steps, reward, env):
        if not self.curriculum_enabled:
            return None
        num_levels = getattr(env, 'num_levels', getattr(env, '_num_levels', None))
        result = ray.get(self.storage.report_curriculum_episode.remote(
            level=0 if level is None else level,
            win=bool(win),
            steps=int(steps),
            total_reward=float(reward),
            num_levels=num_levels,
        ))
        log_scalars = result.get('log_scalars')
        if log_scalars:
            self.storage.add_log_scalar.remote(log_scalars)
        new_level = result.get('current_level')
        if new_level is not None:
            self.curriculum_level = int(new_level)
        if result.get('advanced') and self.rank == 0:
            reason = result.get('reason') or ''
            print(f"[Data worker] Curriculum advanced to level {self.curriculum_level} (reason={reason or 'threshold'}).")
        return result


    def save_previous_trajectory(self, idx, prev_game_trajs, game_trajs, padding=True):
        """put the previous game trajectory into the pool if the current trajectory is full
        Parameters
        ----------
        idx: int
            index of the traj to handle
        prev_game_trajs: list
            list of the previous game trajectories
        game_trajs: list
            list of the current game trajectories
        """
        if padding:
            # pad over last block trajectory
            if self.config.model.value_target == 'bootstrapped':
                gap_step = self.config.env.n_stack + self.config.rl.td_steps
            else:
                extra = max(0, min(int(1 / (1 - self.config.rl.td_lambda)), self.config.model.GAE_max_steps) - self.config.rl.unroll_steps - 1)
                gap_step = self.config.env.n_stack + 1 + extra + 1

            beg_index = self.config.env.n_stack
            end_index = beg_index + self.config.rl.unroll_steps

            pad_obs_lst = game_trajs[idx].obs_lst[beg_index:end_index]

            pad_policy_lst = game_trajs[idx].policy_lst[0:self.config.rl.unroll_steps]
            pad_reward_lst = game_trajs[idx].reward_lst[0:gap_step - 1]
            pad_pred_values_lst = game_trajs[idx].pred_value_lst[0:gap_step]
            pad_search_values_lst = game_trajs[idx].search_value_lst[0:gap_step]

            # pad over and save
            prev_game_trajs[idx].pad_over(pad_obs_lst, pad_reward_lst, pad_pred_values_lst, pad_search_values_lst,
                                          pad_policy_lst)
        prev_game_trajs[idx].save_to_memory()
        self.put_trajs(prev_game_trajs[idx])

        # reset last block
        prev_game_trajs[idx] = None

    def put_trajs(self, traj):
        if self.config.priority.use_priority:
            traj_len = len(traj)
            pred_values = torch.from_numpy(np.array(traj.pred_value_lst)).cuda().float()
            # search_values = torch.from_numpy(np.array(traj.search_value_lst)).cuda().float()
            if self.config.model.value_target == 'bootstrapped':
                target_values = torch.from_numpy(np.asarray(traj.get_bootstrapped_value())).cuda().float()
            elif self.config.model.value_target == 'GAE':
                target_values = torch.from_numpy(np.asarray(traj.get_gae_value())).cuda().float()
            else:
                raise NotImplementedError
            priorities = L1Loss(reduction='none')(pred_values[:traj_len], target_values[:traj_len]).detach().cpu().numpy() + self.config.priority.min_prior
        else:
            priorities = None
        self.traj_pool.append(traj)
        # save the game histories and clear the pool
        if len(self.traj_pool) >= self.pool_size:
            self.replay_buffer.save_pools.remote(self.traj_pool, priorities)
            del self.traj_pool[:]

# ======================================================================================================================
# data worker for self-play
# ======================================================================================================================
def start_data_worker(rank, agent, replay_buffer, storage, config):
    """
    Start a data worker. Call this method remotely.
    """
    data_worker = DataWorker.remote(rank, agent, replay_buffer, storage, config)
    data_worker.run.remote()
    print(f'[Data worker] Start data worker {rank} at process {os.getpid()}.')
