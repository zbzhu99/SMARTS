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

import gym
import numpy as np
import pytest

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import RGB, AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.env.wrappers.single_agent import SingleAgent


def make_agent_specs(num_agent):
    agent_specs = {
        "AGENT_"
        + str(agent_id): AgentSpec(
            interface=AgentInterface(
                rgb=RGB(),
                action=ActionSpaceType.Lane,
            ),
            agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
            observation_adapter=lambda obs: obs.top_down_rgb.data,
            reward_adapter=lambda obs, reward: reward,
            info_adapter=lambda obs, reward, info: info["score"],
        )
        for agent_id in range(num_agent)
    }

    obs_space = gym.spaces.Dict(
        {
            "AGENT_"
            + str(agent_id): gym.spaces.Box(
                low=0,
                high=255,
                shape=(
                    agent_specs["AGENT_" + str(agent_id)].interface.rgb.width,
                    agent_specs["AGENT_" + str(agent_id)].interface.rgb.height,
                    3,
                ),
                dtype=np.uint8,
            )
            for agent_id in range(num_agent)
        }
    )

    return agent_specs, obs_space


def make_env(agent_specs, obs_space):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=["scenarios/figure_eight"],
        agent_specs=agent_specs,
        headless=True,
        visdom=False,
        fixed_timestep_sec=0.01,
    )
    env.observation_space = obs_space
    return env


@pytest.mark.parametrize("num_agent", [1, 2])
def test_init(num_agent):
    agent_specs, obs_space = make_agent_specs(num_agent)
    env = make_env(agent_specs, obs_space)

    # Test wrapping an env containing more than one agent
    if num_agent > 1:
        with pytest.raises(AssertionError):
            env = SingleAgent(env)
            env.close()
        return

    # Test wrapping an env containing one agent
    env = SingleAgent(env)

    # Test env observation space
    assert isinstance(env.observation_space, gym.spaces.Box)
    agent_id = next(iter(agent_specs.keys()))
    assert env.observation_space.shape == obs_space[agent_id].shape
    assert env.observation_space.dtype == obs_space[agent_id].dtype

    env.close()


def test_reset_and_step():
    agent_specs, obs_space = make_agent_specs(1)
    env = make_env(agent_specs, obs_space)
    env = SingleAgent(env)

    # Test reset
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    agent_id = next(iter(agent_specs.keys()))
    assert obs.shape == obs_space[agent_id].shape

    # Test step
    obs, reward, done, info = env.step("keep_lane")
    assert isinstance(obs, np.ndarray)
    assert obs.shape == obs_space[agent_id].shape
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, float)

    env.close()
