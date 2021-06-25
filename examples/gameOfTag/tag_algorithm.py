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

"""Build PPO algorithm."""
import lz4
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Sequence, Union
from xt.algorithm import Algorithm
from xt.model import model_builder
from zeus.common.util.register import Registers


ZFILL_LENGTH = 7

@Registers.algorithm
class Tag(Algorithm):
    """Build PPO algorithm."""

    def __init__(self, model_info, alg_config, **kwargs):
        """
        Create Algorithm instance.

        Will create their model within the `__init__`.
        :param model_info:
        :param alg_config:
        :param kwargs:
        """
        # import_config(globals(), alg_config)
        self.model_information = model_info

        model_info_predator = model_info['actor'].copy()
        model_info_prey = model_info['actor'].copy()

        if 'init_weights' in model_info['actor']:
            assert len(model_info['actor']['init_weights']) == 2, f"Expected two saved model paths, but got {len(model_info['actor']['init_weights'])} paths."
            for path in model_info['actor']['init_weights']:
                if 'predator' in path:
                    model_info_predator['init_weights'] = path
                elif 'prey' in path:
                    model_info_prey['init_weights'] = path 
                else:
                    raise Exception(f"Expected predator and prey saved model path, but got {path}.")   

        super().__init__(
            alg_name=kwargs.get('name') or 'Tag',
            model_info=model_info_predator,
            alg_config=alg_config
        )

        self.actor_prey = model_builder(model_info_prey)

        self._init_train_list()
        self.async_flag = False  # fixme: refactor async_flag

    def _init_train_list(self):
        self.obs_predator = list()
        self.behavior_action_predator = list()
        self.old_logp_predator = list()
        self.adv_predator = list()
        self.old_v_predator = list()
        self.target_v_predator = list()

        self.obs_prey=list()
        self.behavior_action_prey=list()
        self.old_logp_prey=list()
        self.adv_prey=list()
        self.old_v_prey=list()
        self.target_v_prey=list()

    def train(self, **kwargs):
        """Train PPO Agent."""
        obs = np.concatenate(self.obs_predator)
        behavior_action = np.concatenate(self.behavior_action_predator)
        old_logp = np.concatenate(self.old_logp_predator)
        adv = np.concatenate(self.adv_predator)
        old_v = np.concatenate(self.old_v_predator)
        target_v = np.concatenate(self.target_v_predator)

        # adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss_predator = self.actor.train([obs], [behavior_action, old_logp, adv, old_v, target_v])

        obs = np.concatenate(self.obs_prey)
        behavior_action = np.concatenate(self.behavior_action_prey)
        old_logp = np.concatenate(self.old_logp_prey)
        adv = np.concatenate(self.adv_prey)
        old_v = np.concatenate(self.old_v_prey)
        target_v = np.concatenate(self.target_v_prey)

        # adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss_prey = self.actor_prey.train([obs], [behavior_action, old_logp, adv, old_v, target_v])

        self._init_train_list()

        return loss_predator + loss_prey

    def prepare_data(self, train_data, **kwargs):
        # Decompress train_data
        train_data = decompress(train_data, self.state_dim, self.model_information['actor']['model_config']['action_type'])

        if 'predator' in kwargs['ctr_info']['agent_id']:
            self.obs_predator.append(train_data['cur_state'])
            self.behavior_action_predator.append(train_data['action'])
            self.old_logp_predator.append(train_data['logp'])
            self.adv_predator.append(train_data['adv'])
            self.old_v_predator.append(train_data['old_value'])
            self.target_v_predator.append(train_data['target_value'])
        elif 'prey' in kwargs['ctr_info']['agent_id']:
            self.obs_prey.append(train_data['cur_state'])
            self.behavior_action_prey.append(train_data['action'])
            self.old_logp_prey.append(train_data['logp'])
            self.adv_prey.append(train_data['adv'])
            self.old_v_prey.append(train_data['old_value'])
            self.target_v_prey.append(train_data['target_value'])
        else:
            raise ValueError(f"Expected agent type 'predator' or 'prey', but received unknown agent type {kwargs['agent_id']}.")

    def predict(self, state: Union[np.ndarray, Sequence[np.ndarray]], kind: List[str])->np.ndarray:
        """Overwrite the predict function, owing to the special input."""
        if not isinstance(state, (list, tuple)):
            state = state.reshape((1,) + state.shape)
        else:
            state = list(map(lambda x: x.reshape((1,) + x.shape), state))
            state = np.vstack(state)

        if all("predator" in agent_id for agent_id in kind):
            pred = self.actor.predict(state)
        elif all("prey" in agent_id for agent_id in kind):
            pred = self.actor_prey.predict(state)
        else:
            raise ValueError(f"Expected agent type 'predator' or 'prey', but received unknown agent type {type}.")

        return pred

    def restore(self, model_name=None, model_weights=None):
        if model_weights is not None:
            self.set_weights(model_weights)
        else:
            raise ValueError(f"Expected model_weights, but got model_weights = {model_weights}.")

    def get_weights(self):
        """Get the actor model weights as default."""
        return [self.actor.get_weights(), self.actor_prey.get_weights()]

    def set_weights(self, weights):
        """Set the actor model weights as default."""
        self.actor.set_weights(weights[0]) 
        self.actor_prey.set_weights(weights[1])

    def save(self, model_path, model_index):
        model_name_predator = self.actor.save_model(
            os.path.join(model_path, "actor_predator_{}".format(str(model_index).zfill(ZFILL_LENGTH))))
        model_name_prey = self.actor_prey.save_model(
            os.path.join(model_path, "actor_prey_{}".format(str(model_index).zfill(ZFILL_LENGTH))))

        return [model_name_predator, model_name_prey]

