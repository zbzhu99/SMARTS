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
from collections import deque
from typing import Dict, List, Tuple, Union
from smarts.core import sensors


class SingleAgent(gym.Wrapper):
    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): Gym environment to be wrapped.
            num_stack (int, optional): Number of frames to be stacked. Defaults to 3.
            num_skip (int, optional): Frequency of frames used in returned stacked frames. Defaults to 1.
        """
        assert num_stack > 1, f"Expected num_stack > 1, but got {num_stack}."
        assert num_skip > 0, f"Expected num_skip > 0, but got {num_skip}."
        super(FrameStack, self).__init__(env)
        self._num_stack = (num_stack - 1) * num_skip + 1
        self._num_skip = num_skip
        self._frames = {
            key: deque(maxlen=self._num_stack) for key in self.env.agent_specs.keys()
        }

    def act(
        self, frame: sensors.Observation
    ) -> Dict[str, List[sensors.Observation]]:
        """Update and return frames stack with given latest single frame."""

        new_frames = dict.fromkeys(frame)

        for agent_id, observation in frame.items():
            self._frames[agent_id].appendleft(observation)
            frames_list = list(self._frames[agent_id])
            new_frames[agent_id] = copy.deepcopy(frames_list[:: self._num_skip])

        return new_frames

    def step(
        self, agent_actions: Dict
    ) -> Tuple[
        Dict[str, List[sensors.Observation]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict[str, Union[float, sensors.Observation]]],
    ]:
        """Steps the environment by one step.

        Args:
            agent_actions (Dict): Actions for each agent.

        Returns:
            Tuple[ Dict[str, List[sensors.Observation]], Dict[str, float], Dict[str, bool], Dict[str, Dict[str, Union[float, sensors.Observation]]] ]: Observation, reward, done, info, for each agent.
        """
        env_observations, rewards, dones, infos = super(FrameStack, self).step(
            agent_actions
        )

        return self._get_observations(env_observations), rewards, dones, infos

    def reset(self) -> Dict[str, List[sensors.Observation]]:
        """Resets the environment.

        Returns:
            Dict[str, List[sensors.Observation]]: Observation upon reset for each agent.
        """
        env_observations = super(FrameStack, self).reset()
        for agent_id, observation in env_observations.items():
            for _ in range(self._num_stack - 1):
                self._frames[agent_id].appendleft(observation)

        return self._get_observations(env_observations)

    def close(self):
        pass
