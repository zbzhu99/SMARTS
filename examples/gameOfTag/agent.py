# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import utils
from examples.gameOfTag import model as got_model
from typing import Sequence, Union


class TagAgent():
    def __init__(self, name, config):
        # Verify name
        if name == "predator" or name == "prey":
            self.name = name
        else:
            raise Exception(f"Expected predator or prey, but got {name}.")    
        self.config = config
        self.states = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.advantages = None
        self.returns = None
        self.last_value = None

    def reset(self):
        self.states = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.advantages = None
        self.returns = None
        self.last_value = None

    def add_trajectory(self, state, action, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def store_last_value(self, last_value):
        self.last_value = last_value

    def compute_gae(self):
        advantages = utils.compute_gae(
            rewards=self.rewards,
            values=self.values,
            last_values=self.last_value,
            dones=self.dones,
            discount_factor=self.config['model_para']['discount_factor'], 
            gae_lambda=self.config['model_para']['gae_lambda'],
            )
        self.advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)
        self.returns = advantages + self.values  


class TagModel():
    def __init__(self, name, env, config):
        self.name = name
        self.config = config
        self.model = None
        self._create_model(env)

    def _create_model(self, env):
        # Environment
        input_shape = env.observation_space.shape
        num_actions = env.action_space.shape[0]
        action_min = env.action_space.low
        action_max = env.action_space.high

        ppo_epsilon = self.config['model_para']['ppo_epsilon']
        value_scale = self.config['model_para']['value_scale']
        entropy_scale = self.config['model_para']['entropy_scale']

        self.model = got_model.PPO(input_shape, 
            num_actions, action_min, action_max,
            epsilon=ppo_epsilon,
            value_scale=value_scale, 
            entropy_scale=entropy_scale,
            model_name=self.name)

    def act(self, obs):
        actions = {}
        values = {}
        for vehicle, state in obs.items():
            if self.name in vehicle:
                actions_t, values_t = self.model.predict(state)
                actions[vehicle]=actions_t
                values[vehicle]=values_t
        return actions, values

    def save(self):
        return self.model.save()


def stack_vars(state: Union[np.ndarray, Sequence[np.ndarray]])->np.ndarray:
    if not isinstance(state, (list, tuple)):
        state = state.reshape((1,) + state.shape)
    else:
        state = list(map(lambda x: x.reshape((1,) + x.shape), state))
        state = np.vstack(state)

    return state
