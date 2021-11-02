import gym
import matplotlib.pyplot as plt
import numpy as np

from smarts.core import agent as smarts_agent
from smarts.core import agent_interface as smarts_agent_interface
from smarts.core import colors as smarts_colors
from smarts.core import controllers as smarts_controllers
from smarts.core import sensors as smarts_sensors
from smarts.env import hiway_env as smarts_hiway_env
from smarts.env.wrappers import frame_stack as smarts_frame_stack
from typing import Dict


NEIGHBOURHOOD_RADIUS = 55


class Traffic(gym.Env):
    def __init__(self, config: Dict):
        self._config = config
        self._neighborhood_radius = config["env_para"]["neighborhood_radius"]
        self._rgb_wh = config["env_para"]["rgb_wh"]
        self.agent_ids = config["env_para"]["agent_ids"]

        agent_interface = smarts_agent_interface.AgentInterface(
            max_episode_steps=config["env_para"]["max_episode_steps"],
            neighborhood_vehicles=smarts_agent_interface.NeighborhoodVehicles(
                radius=self._neighborhood_radius
            ),
            rgb=smarts_agent_interface.RGB(
                width=256, height=256, resolution=self._rgb_wh / 256
            ),
            vehicle_color="BrightRed",
            action=getattr(
                smarts_controllers.ActionSpaceType,
                config["env_para"]["action_space_type"],
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
                interface=agent_interface,
                agent_builder=None,
                observation_adapter=observation_adapter,
                reward_adapter=reward_adapter,
                action_adapter=action_adapter(config["env_para"]["action_space_type"]),
                info_adapter=info_adapter,
            )
            for agent_id in self.agent_ids
        }

        env = smarts_hiway_env.HiWayEnv(
            scenarios=config["env_para"]["scenarios"],
            agent_specs=agent_specs,
            headless=config["env_para"]["headless"],
            visdom=config["env_para"]["visdom"],
            seed=config["env_para"]["seed"],
        )
        # Wrap env with FrameStack to stack multiple observations
        env = smarts_frame_stack.FrameStack(
            env=env,
            num_stack=config["env_para"]["num_stack"],
            num_skip=config["env_para"]["num_skip"],
        )

        self._env = env

        # Categorical action space
        self.single_action_space = gym.spaces.Discrete(
            config["model_para"]["action_dim"]
        )
        # Observation space
        self.single_observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=config["model_para"]["observation1_dim"],
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


def info_adapter(obs, reward, info):
    return reward


def action_adapter(controller):
    # Action space
    # throttle: [0, 1]
    # brake: [0, 1]
    # steering: [-1, 1]

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

    if controller == "Continuous":

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


def observation_adapter(obs: smarts_sensors.Observation) -> Dict[str, np.ndarray]:
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

    scalar = np.array(
        (
            obs.ego_vehicle_state.speed,
            obs.ego_vehicle_state.steering,
        ),
        dtype=np.float32,
    )

    return {"image": frame, "scalar": scalar}


def get_targets(vehicles, target: str):
    target_vehicles = [vehicle for vehicle in vehicles if target in vehicle.id]
    return target_vehicles


def distance_to_targets(ego, targets):
    distances = (
        [np.linalg.norm(ego.position - target.position) for target in targets],
    )
    return distances


def inverse(x: float) -> float:
    return -x + NEIGHBOURHOOD_RADIUS


def reward_adapter(obs, env_reward):
    ego = obs.ego_vehicle_state
    reward = 0

    # Penalty for driving off road
    if obs.events.off_road:
        reward -= 200
        print(f"Vehicle {ego.id} went off road.")
        return np.float32(reward)

    # Reward for colliding
    if len(obs.events.collisions) == 0:
        reward -= 200
        print(f"Vehicle {ego.id} collided.")
        return np.float32(reward)

    # Distance based reward
    reward += env_reward
    print("Env Reward:: ------> ", env_reward)
    return np.float32(reward)
