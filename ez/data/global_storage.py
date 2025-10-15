# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import os
import time
import ray
import numpy as np
from collections import deque

@ray.remote
class GlobalStorage:
    def __init__(self, self_play_model, reanalyze_model, latest_model):
        self.models = {
            'self_play': self_play_model,
            'reanalyze': reanalyze_model,
            'latest': latest_model
        }
        self.log_scalar = {}
        self.eval_log_scalar = {}
        self.log_distribution = {}
        self.counter = 0
        self.eval_counter = 0
        self.best_score = - np.inf
        self.start = False
        # self.batch = None
        self.curriculum = {
            'configured': False,
            'enabled': False,
            'current_level': 0,
            'min_episodes': 0,
            'recent_window': 1,
            'target_win_rate': 0.0,
            'max_episodes': None,
            'num_levels': None,
            'stats': {},
        }

    def get_weights(self, model_name):
        assert model_name in self.models.keys()
        return self.models[model_name].get_weights()

    def set_weights(self, weights, model_name):
        assert model_name in self.models.keys()
        # print('[Update] set recent model of {}'.format(model_name))
        return self.models[model_name].set_weights(weights)

    def increase_counter(self):
        self.counter += 1

    def get_counter(self):
        return self.counter

    def set_eval_counter(self, counter):
        self.eval_counter = counter

    def get_eval_counter(self):
        return self.eval_counter

    def set_start_signal(self):
        self.start = True

    def get_start_signal(self):
        return self.start

    def set_best_score(self, score):
        self.best_score = max(self.best_score, score)

    def get_best_score(self):
        return self.best_score

    def configure_curriculum(self, settings):
        curr = self.curriculum
        if settings is None:
            curr.update({
                'configured': True,
                'enabled': False,
                'current_level': 0,
                'min_episodes': 0,
                'recent_window': 1,
                'target_win_rate': 0.0,
                'max_episodes': None,
                'num_levels': None,
                'stats': {},
            })
            return {
                'enabled': False,
                'current_level': 0,
                'num_levels': None,
            }

        if not curr.get('configured', False):
            curr['configured'] = True
            curr['enabled'] = bool(settings.get('enabled', False))
            curr['target_win_rate'] = float(settings.get('target_win_rate', 0.0))
            curr['min_episodes'] = max(0, int(settings.get('min_episodes', 0)))
            recent_window = settings.get('recent_window', curr['min_episodes'] or 1)
            curr['recent_window'] = max(1, int(recent_window))
            max_eps = settings.get('max_episodes')
            if max_eps is None:
                curr['max_episodes'] = None
            else:
                max_eps = int(max_eps)
                curr['max_episodes'] = max_eps if max_eps > 0 else None
            curr['current_level'] = max(0, int(settings.get('start_level', 0)))
            num_levels = settings.get('num_levels')
            if num_levels is not None:
                try:
                    num_levels = int(num_levels)
                except Exception:
                    num_levels = None
            if num_levels is not None and num_levels > 0:
                curr['num_levels'] = num_levels
            else:
                curr['num_levels'] = None
            curr['stats'] = {}
        else:
            if settings.get('num_levels') is not None:
                try:
                    num_levels = int(settings['num_levels'])
                except Exception:
                    num_levels = None
                if num_levels is not None and num_levels > 0:
                    prev = curr.get('num_levels')
                    curr['num_levels'] = max(prev or 0, num_levels)
        return {
            'enabled': curr.get('enabled', False),
            'current_level': curr.get('current_level', 0),
            'num_levels': curr.get('num_levels'),
        }

    def report_curriculum_episode(self, level, win, steps, total_reward, num_levels=None):
        curr = self.curriculum
        if num_levels is not None:
            try:
                num_levels = int(num_levels)
            except Exception:
                num_levels = None
            if num_levels is not None and num_levels > 0:
                prev = curr.get('num_levels')
                curr['num_levels'] = max(prev or 0, num_levels)

        if not curr.get('enabled', False):
            return {
                'current_level': curr.get('current_level', 0),
                'advanced': False,
                'reason': '',
                'log_scalars': {},
            }

        try:
            level = int(level)
        except Exception:
            level = 0
        level = max(0, level)
        stats = curr.setdefault('stats', {})
        if level not in stats:
            stats[level] = {
                'episodes': 0,
                'wins': 0,
                'recent': deque(maxlen=max(1, curr.get('recent_window', 1))),
                'total_steps': 0,
                'total_reward': 0.0,
            }
        entry = stats[level]
        entry['episodes'] += 1
        if win:
            entry['wins'] += 1
        entry['recent'].append(1 if win else 0)
        entry['total_steps'] += int(steps)
        entry['total_reward'] += float(total_reward)

        recent_win_rate = sum(entry['recent']) / max(1, len(entry['recent']))
        overall_win_rate = entry['wins'] / max(1, entry['episodes'])

        advanced = False
        reason = ''
        if level == curr.get('current_level', 0):
            min_window = min(curr.get('recent_window', 1), max(1, curr.get('min_episodes', 0)))
            window_ready = len(entry['recent']) >= min_window
            enough = entry['episodes'] >= curr.get('min_episodes', 0)
            target = curr.get('target_win_rate', 0.0)
            hit_target = recent_win_rate >= target if target > 0 else enough
            if enough and window_ready and hit_target:
                advanced = True
                reason = 'threshold'
            else:
                max_eps = curr.get('max_episodes')
                if max_eps is not None and entry['episodes'] >= max_eps:
                    advanced = True
                    reason = 'max_episodes'
            if advanced:
                next_level = curr.get('current_level', 0) + 1
                limit = curr.get('num_levels')
                if limit is not None and next_level >= limit:
                    advanced = False
                    next_level = curr.get('current_level', 0)
                    reason = ''
                curr['current_level'] = next_level

        log_scalars = {
            'curriculum/current_level': float(curr.get('current_level', 0)),
            f'curriculum/level_{level}_episodes': float(entry['episodes']),
            f'curriculum/level_{level}_recent_win_rate': float(recent_win_rate),
            f'curriculum/level_{level}_overall_win_rate': float(overall_win_rate),
        }
        if advanced:
            log_scalars['curriculum/advanced_to_level'] = float(curr.get('current_level', 0))

        return {
            'current_level': curr.get('current_level', 0),
            'advanced': advanced,
            'reason': reason,
            'log_scalars': log_scalars,
        }

    def get_curriculum_state(self):
        curr = self.curriculum
        return {
            'enabled': curr.get('enabled', False),
            'current_level': curr.get('current_level', 0),
            'num_levels': curr.get('num_levels'),
        }

    def add_log_scalar(self, dic):
        for key, val in dic.items():
            if key not in self.log_scalar.keys():
                self.log_scalar[key] = []

            self.log_scalar[key].append(val)

    def add_eval_log_scalar(self, dic):
        for key, val in dic.items():
            if key not in self.eval_log_scalar.keys():
                self.eval_log_scalar[key] = []
            self.eval_log_scalar[key].append(val)

    def add_log_distribution(self, dic):
        for key, val in dic.items():
            if key not in self.log_distribution.keys():
                self.log_distribution[key] = []

            self.log_distribution[key] += val.tolist()

    def get_log(self):
        # for scalar
        scalar = {}
        for key, val in self.log_scalar.items():
            scalar[key] = np.mean(val)

        eval_scalar = {}
        for key, val in self.eval_log_scalar.items():
            eval_scalar[key] = np.mean(val)

        # for distribution
        distribution = {}
        for key, val in self.log_distribution.items():
            distribution[key] = np.array(val).flatten()

        self.log_scalar = {}
        self.eval_log_scalar = {}
        self.log_distribution = {}
        return eval_scalar, scalar, distribution

# ======================================================================================================================
# global storage server
# ======================================================================================================================
