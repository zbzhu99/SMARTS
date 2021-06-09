# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

import copy
import gym
from typing import Dict, Tuple, Union
from smarts.core import sensors


class ContinueEpisode(gym.Wrapper):
    """This wrapper continues the previous unifinished episode.
    If previous episode finished, reset resets the environment.
    If previous episode is unfinished, reset returns the last
    observation from previous episode."""

    def __init__(self, env: gym.Env):
        super(ContinueEpisode, self).__init__(env)
        self.real_done = True
        self.last_obs = None

    def step(
        self, agent_actions: Dict
    ) -> Tuple[
        Dict[str, sensors.Observation],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict[str, Union[float, sensors.Observation]]],
    ]:

        observations, rewards, dones, infos = super(ContinueEpisode, self).step(
            agent_actions
        )

        # Store the true episode completion flag and last observation
        self.real_done = dones["__all__"]
        self.last_obs = copy.deepcopy(observations)
        # Remove done agents
        [
            self.last_obs.pop(key)
            for key, value in dones.items()
            if key != "__all__" and value == True
        ]

        return observations, rewards, dones, infos

    def reset(self) -> Dict[str, sensors.Observation]:
        if self.real_done:
            observations = super(ContinueEpisode, self).reset()
            self.last_obs = copy.deepcopy(observations)
        else:
            observations = self.last_obs

        return observations
