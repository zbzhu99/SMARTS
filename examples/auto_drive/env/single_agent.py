import gym
import matplotlib.pyplot as plt
import numpy as np

import examples.auto_drive.env.adapter as adapter

from smarts.core import agent as smarts_agent
from smarts.core import agent_interface as smarts_agent_interface
from smarts.core import controllers as smarts_controllers
from smarts.env import hiway_env as smarts_hiway_env
from typing import Dict

NEIGHBOURHOOD_RADIUS = 55


class SingleAgent(gym.Wrapper):
    def __init__(self, config: Dict, seed: int):
        self._neighborhood_radius = config["neighborhood_radius"]
        self._rgb_wh = config["rgb_wh"]
        self.agent_id = config["agent_ids"][0]

        vehicle_interface = smarts_agent_interface.AgentInterface(
            max_episode_steps=config["max_episode_steps"],
            neighborhood_vehicles=smarts_agent_interface.NeighborhoodVehicles(
                radius=self._neighborhood_radius
            ),
            rgb=smarts_agent_interface.RGB(
                width=256, height=256, resolution=self._rgb_wh / 256
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

        agent_specs = {
            self.agent_id: smarts_agent.AgentSpec(
                interface=vehicle_interface,
                agent_builder=None,
                observation_adapter=adapter.observation_adapter,
                reward_adapter=adapter.reward_adapter,
                action_adapter=adapter.action_adapter(config["action_adapter"]),
                info_adapter=adapter.info_adapter,
            )
        }

        env = smarts_hiway_env.HiWayEnv(
            scenarios=config["scenarios"],
            agent_specs=agent_specs,
            headless=config["headless"],
            visdom=config["visdom"],
            seed=seed,
        )

        # Initialize base env
        super(SingleAgent, self).__init__(env)

        # Action space
        self.action_space = adapter.action_space(config['action_adapter'])
        # Observation space
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
        )

    def reset(self) -> np.ndarray:
        raw_states = self.env.reset()

        return raw_states[self.agent_id]

    def step(self, action):
        raw_states, rewards, dones, infos = self.env.step({self.agent_id: action})

        # Plot for debugging purposes
        # import matplotlib.pyplot as plt
        # columns = 1 # number of frames stacked for each agent
        # rgb_gray = 3 # 3 for rgb and 1 for grayscale
        # n_states = len(raw_states.keys())
        # fig, axes = plt.subplots(1, n_states*columns, figsize=(10, 10))
        # fig.tight_layout()
        # ax = axes.ravel()
        # for row, (agent_id, state) in enumerate(raw_states.items()):
        #     for col in range(columns):
        #         img = state[:,:,rgb_gray*col:rgb_gray*col+rgb_gray]
        #         ax[row*columns+col].imshow(img)
        #         ax[row*columns+col].set_title(agent_id)
        # plt.show()

        return (
            raw_states[self.agent_id],
            rewards[self.agent_id],
            dones[self.agent_id],
            infos[self.agent_id],
        )

    def close(self):
        if self.env is not None:
            return self.env.close()
        return None


