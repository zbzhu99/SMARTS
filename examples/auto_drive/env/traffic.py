from pathlib import Path
from typing import Dict

import gym
import matplotlib.pyplot as plt
import numpy as np

import examples.auto_drive.env.adapter as adapter
from smarts.core import agent as smarts_agent
from smarts.core import agent_interface as smarts_agent_interface
from smarts.core import controllers as smarts_controllers
from smarts.env import hiway_env as smarts_hiway_env
from smarts.env.wrappers import frame_stack as smarts_frame_stack


class Traffic(gym.Env):
    def __init__(self, config: Dict, seed: int):
        self._config = config
        self.agent_ids = config["agent_ids"]

        vehicle_interface = smarts_agent_interface.AgentInterface(
            max_episode_steps=config["max_episode_steps"],
            neighborhood_vehicles=smarts_agent_interface.NeighborhoodVehicles(
                radius=config["neighborhood_radius"]
            ),
            rgb=smarts_agent_interface.RGB(
                width=256, height=256, resolution=config["rgb_wh"] / 256
            ),
            vehicle_color="BrightRed",
            action=getattr(
                smarts_controllers.ActionSpaceType,
                config["action_space_type"],
            ),
            done_criteria=smarts_agent_interface.DoneCriteria(
                collision=True,
                off_road=True,
                off_route=False,
                on_shoulder=False,
                wrong_way=False,
                not_moving=False,
            ),
        )

        # Create agent spec
        agent_specs = {
            agent_id: smarts_agent.AgentSpec(
                interface=vehicle_interface,
                agent_builder=None,
                observation_adapter=adapter.observation_adapter_1,
                reward_adapter=adapter.reward_adapter,
                action_adapter=adapter.action_adapter(config["action_adapter"]),
                info_adapter=adapter.info_adapter,
            )
            for agent_id in self.agent_ids
        }

        base = (Path(__file__).absolute().parents[3]).joinpath("scenarios")
        scenarios = [base.joinpath(scenario) for scenario in config["scenarios"]]
        env = smarts_hiway_env.HiWayEnv(
            scenarios=scenarios,
            agent_specs=agent_specs,
            headless=config["headless"],
            visdom=config["visdom"],
            seed=seed,
        )
        # Wrap env with FrameStack to stack multiple observations
        env = smarts_frame_stack.FrameStack(
            env=env,
            num_stack=config["num_stack"],
            num_skip=config["num_skip"],
        )

        self._env = env

        # Action space
        self.action_space = adapter.action_space(config["action_adapter"])
        # Observation space
        self.single_observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=config["observation1_dim"],
                    dtype=np.uint8,
                ),
                "scalar": gym.spaces.Box(
                    low=np.array([0, -1]),
                    high=np.array([1e3, 1]),
                    dtype=np.float32,
                ),
            }
        )
        # scalar consists of
        # obs.ego_vehicle_state.speed = [0, 1e3]
        # obs.ego_vehicle_state.steering = [-1, 1]

    def reset(self) -> Dict[str, np.ndarray]:

        raw_state = self._env.reset()
        stacked_state = _stack_obs(raw_state)

        self.init_state = stacked_state
        return stacked_state

    def step(self, action):

        raw_state, reward, done, info = self._env.step(action)
        stacked_state = _stack_obs(raw_state)

        # Plot for debugging purposes
        # import matplotlib.pyplot as plt
        # fig=plt.figure(figsize=(10,10))
        # columns = 4 # number of stacked images
        # rgb_gray = 3 # 3 for rgb and 1 for grayscale
        # rows = len(stacked_state.keys())
        # for row, (_, state) in enumerate(stacked_state.items()):
        #     for col in range(0, columns):
        #         img = state["image"][:,:,col*rgb_gray:col*rgb_gray+rgb_gray]
        #         fig.add_subplot(rows, columns, row*columns + col + 1)
        #         plt.imshow(img)
        # plt.show()

        return stacked_state, reward, done, info

    def close(self):
        if self._env is not None:
            return self._env.close()
        return None


def _stack_obs(state: Dict):
    stacked_state = {}
    for agent_id, agent_state in state.items():
        images, scalars = zip(*(map(lambda x: (x["image"], x["scalar"]), agent_state)))
        stacked_image = np.dstack(images)
        stacked_scalar = np.concatenate(scalars, axis=0)
        stacked_state[agent_id] = {"image": stacked_image, "scalar": stacked_scalar}

    return stacked_state
