import gym
import matplotlib.pyplot as plt
import numpy as np

from smarts.core import agent as smarts_agent
from smarts.core import agent_interface as smarts_agent_interface
from smarts.core import colors as smarts_colors
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
                observation_adapter=observation_adapter,
                reward_adapter=reward_adapter,
                action_adapter=action_adapter(config["action_space_type"]),
                info_adapter=info_adapter,
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
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float
        )
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
        # columns = 2 # number of frames stacked for each agent
        # rgb_gray = 3 # 3 for rgb and 1 for grayscale
        # n_states = len(states.keys())
        # fig, axes = plt.subplots(1, n_states*columns, figsize=(10, 10))
        # fig.tight_layout()
        # ax = axes.ravel()
        # for row, (agent_id, state) in enumerate(states.items()):
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


def info_adapter(obs, reward, info):
    return info


def action_adapter(controller):
    # Action space
    # throttle: [0, 1]
    # brake: [0, 1]
    # steering: [-1, 1]

    if controller == "Continuous":

        def continuous(model_action):
            throttle, brake, steering = model_action
            # Modify action space limits
            throttle = (throttle + 1) / 2
            brake = (brake + 1) / 2
            return np.array([throttle, brake, steering], dtype=np.float)

        return continuous

    if controller == "Lane":

        def lane(model_action):
            if model_action == 0:
                return "keep_lane"
            if model_action == 1:
                return "slow_down"
            if model_action == 2:
                return "change_lane_left"
            if model_action == 3:
                return "change_lane_right"
            raise Exception("Unknown model action.")

        return lane

    if controller == "LaneWithContinuousSpeed":

        def lane_speed(model_action):
            speeds = [0, 3, 6, 9]  # Speed selection in m/s
            lanes = [
                -1,
                0,
                1,
            ]  # Lane change relative to current lane
            target_speed = speeds[model_action[0]]
            lane_change = lanes[model_action[1]]
            return np.array([target_speed, lane_change], dtype=np.float32)

        return lane_speed

    if controller == "Discrete":

        def discrete(model_action):
            # Modify action space limits
            if model_action == 0:
                # Cruise
                throttle = 0.3
                brake = 0
                steering = 0
            elif model_action == 1:
                # Accelerate
                throttle = 0.6
                brake = 0
                steering = 0
            elif model_action == 2:
                # Turn left
                throttle = 0.5
                brake = 0
                steering = -0.8
            elif model_action == 3:
                # Turn right
                throttle = 0.5
                brake = 0
                steering = 0.8
            elif model_action == 4:
                # Brake
                throttle = 0
                brake = 0.8
                steering = 0
            else:
                raise Exception("Unknown model action.")
            return np.array([throttle, brake, steering], dtype=np.float32)

        return discrete

    raise Exception("Unknown controller.")


def observation_adapter(obs) -> np.ndarray:
    # RGB grid map
    rgb = obs.top_down_rgb.data
    # Replace self color to Lime
    coloured_self = rgb.copy()
    coloured_self[123:132, 126:130, 0] = smarts_colors.Colors.Lime.value[0] * 255
    coloured_self[123:132, 126:130, 1] = smarts_colors.Colors.Lime.value[1] * 255
    coloured_self[123:132, 126:130, 2] = smarts_colors.Colors.Lime.value[2] * 255
    frame = coloured_self.astype(np.uint8)

    # Plot graph
    # fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    # ax = axes.ravel()
    # ax[0].imshow(rgb)
    # ax[0].set_title("RGB")
    # ax[1].imshow(frame)
    # ax[1].set_title("Frame")
    # fig.tight_layout()
    # plt.show()
    # sys.exit(2)

    return frame


def reward_adapter(obs, env_reward):
    ego = obs.ego_vehicle_state
    reward = 0

    # Penalty for driving off road
    if obs.events.off_road:
        reward = 0
        print(f"Vehicle {ego.id} went off road.")
        return np.float32(reward)

    # Reward for colliding
    if len(obs.events.collisions) > 0:
        reward = 0
        print(f"Vehicle {ego.id} collided.")
        return np.float32(reward)

    # Distance based reward
    reward += env_reward

    return np.float32(reward)
