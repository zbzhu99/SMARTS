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


class TagModel:
    def __init__(self, name, env, config, model_checkpoint):
        self.name = name
        self.config = config
        self.model = None
        self._create_model(env, model_checkpoint)

    def _create_model(self, env, model_checkpoint):
        # Environment
        input_shape = env.observation_space.shape
        num_actions = env.action_space.shape[0]
        action_min = env.action_space.low
        action_max = env.action_space.high

        ppo_epsilon = self.config["model_para"]["ppo_epsilon"]
        value_scale = self.config["model_para"]["value_scale"]
        entropy_scale = self.config["model_para"]["entropy_scale"]
        base_path = self.config["benchmark"]["base_path"]

        self.model = got_model.Model(
            input_shape,
            num_actions,
            action_min,
            action_max,
            epsilon=ppo_epsilon,
            value_scale=value_scale,
            entropy_scale=entropy_scale,
            model_name=self.name,
            model_checkpoint=model_checkpoint,
            base_path=base_path,
        )

    def act(self, obs):
        actions = {}
        values = {}
        for vehicle, state in obs.items():
            if self.name in vehicle:
                actions_t, values_t = self.model.predict(np.expand_dims(state, axis=0))
                actions[vehicle] = np.squeeze(actions_t, axis=0)
                values[vehicle] = np.squeeze(values_t, axis=-1)
        return actions, values

    def save(self):
        return self.model.save()

    def train(self, input_states, taken_actions, returns, advantage, learning_rate):
        return self.model.train(
            input_states, taken_actions, returns, advantage, learning_rate
        )

    def write_to_summary(self, name, value):
        return self.model.write_to_summary(name, value)

    def update_old_policy(self):
        return self.model.update_old_policy()


# def stack_vars(var: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
#     return np.vstack(var)
