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


from typing import Any, Dict

import gym
import numpy as np


class RGBImage(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        agent_specs = env.agent_specs

        for agent_id in agent_specs.keys():
            assert agent_specs[agent_id].interface, (
                f"To use RGBImage wrapper, enable RGB "
                f"functionality in {agent_id}'s AgentInterface."
            )

        self.observation_space = gym.spaces.Dict(
            {
                agent_id: gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        agent_specs[agent_id].interface.rgb.width,
                        agent_specs[agent_id].interface.rgb.height,
                        3,
                    ),
                    dtype=np.uint8,
                )
                for agent_id in agent_specs.keys()
            }
        )

    def observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        obs = {
            agent_id: agent_obs.top_down_rgb.data.astype(np.uint8)
            for agent_id, agent_obs in obs.items()
        }
        return obs
