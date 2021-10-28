import gym
import matplotlib.pyplot as plt
import numpy as np

from examples.gameOfTag import agent as got_agent
from examples.gameOfTag.types import AgentType
from smarts.core import agent as smarts_agent
from smarts.core import agent_interface as smarts_agent_interface
from smarts.core import colors as smarts_colors
from smarts.core import controllers as smarts_controllers
from smarts.core import sensors as smarts_sensors
from smarts.env import hiway_env as smarts_hiway_env
from smarts.env.wrappers import frame_stack as smarts_frame_stack
from typing import Dict, List

import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

NEIGHBOURHOOD_RADIUS = 55


class TagEnvTF(py_environment.PyEnvironment):
    def __init__(self, config: Dict, seed: int = 42):
        super(TagEnvTF).__init__()
        self._action_dim = config["model_para"]["action_dim"]
        self._discount = config["model_para"]["gamma"]
        self._neighborhood_radius = config["env_para"]["neighborhood_radius"]
        self._rgb_wh = config["env_para"]["rgb_wh"]
        self.agent_ids = config["env_para"]["agent_ids"]
        self.predators = []
        self.preys = []
        for agent_id in self.agent_ids:
            if AgentType.PREDATOR in agent_id:
                self.predators.append(agent_id)
            elif AgentType.PREY in agent_id:
                self.preys.append(agent_id)
            else:
                raise ValueError(
                    f"Expected agent_id to have prefix from {AgentType.__members__}, but got {agent_id}."
                )

        predator_interface = smarts_agent_interface.AgentInterface(
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
                collision=False,
                off_road=True,
                off_route=False,
                on_shoulder=False,
                wrong_way=False,
                not_moving=False,
                agents_alive=smarts_agent_interface.AgentsAliveDoneCriteria(
                    agent_lists_alive=[
                        smarts_agent_interface.AgentsListAlive(
                            agents_list=self.preys, minimum_agents_alive_in_list=1
                        ),
                    ]
                ),
            ),
        )

        prey_interface = smarts_agent_interface.AgentInterface(
            max_episode_steps=config["env_para"]["max_episode_steps"],
            neighborhood_vehicles=smarts_agent_interface.NeighborhoodVehicles(
                radius=self._neighborhood_radius
            ),
            rgb=smarts_agent_interface.RGB(
                width=256, height=256, resolution=self._rgb_wh / 256
            ),
            vehicle_color="BrightBlue",
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
                agents_alive=smarts_agent_interface.AgentsAliveDoneCriteria(
                    agent_lists_alive=[
                        smarts_agent_interface.AgentsListAlive(
                            agents_list=self.predators, minimum_agents_alive_in_list=1
                        ),
                    ]
                ),
            ),
        )

        # Create agent spec
        agent_specs = {
            agent_id: smarts_agent.AgentSpec(
                interface=predator_interface,
                agent_builder=got_agent.TagAgent,
                observation_adapter=observation_adapter,
                reward_adapter=predator_reward_adapter,
                action_adapter=action_adapter(config["env_para"]["action_space_type"]),
                info_adapter=info_adapter,
            )
            if AgentType.PREDATOR in agent_id
            else smarts_agent.AgentSpec(
                interface=prey_interface,
                agent_builder=got_agent.TagAgent,
                observation_adapter=observation_adapter,
                reward_adapter=prey_reward_adapter,
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
            seed=seed,
        )
        # Wrap env with FrameStack to stack multiple observations
        env = smarts_frame_stack.FrameStack(
            env=env,
            num_stack=config["env_para"]["num_stack"],
            num_skip=config["env_para"]["num_skip"],
        )

        self._env = env

        # Single agent's action spec
        self._single_action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=config["model_para"]["action_dim"],
        )
        # Multi agent observation spec
        self._action_spec = {
            agent_id: self._single_action_spec for agent_id in self.agent_ids
        }
        # Single agent's observation spec
        self._single_observation_spec = array_spec.BoundedArraySpec(
            shape=(config["model_para"]["observation1_dim"]),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
        )
        # Multi agent observation spec
        self._observation_spec = {
            agent_id: self._single_observation_spec for agent_id in self.agent_ids
        }

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def single_action_spec(self):
        return self._single_action_spec

    def single_observation_spec(self):
        return self._single_observation_spec

    def _reset(self):
        self._episode_ended = False
        raw_state = self._env.reset()
        stacked_state = {}
        for agent_id, state in raw_state.items():
            stacked_state[agent_id] = _stack_matrix(state)
        return ts.restart(observation=stacked_state)

    def _step(self, action):
        raw_state, reward, done, info = self._env.step(action)
        stacked_state = {}
        for agent_id, state in raw_state.items():
            stacked_state[agent_id] = _stack_matrix(state)

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

        # return stacked_state, reward, done, info

        if done["__all__"] == True:
            self._episode_ended = True
            return ts.termination(
                observation=stacked_state, reward=reward, outer_dims=1
            )

        return ts.transition(
            observation=stacked_state,
            reward=reward,
            discount=self._discount,
            outer_dims=1,
        )

    def close(self):
        if self._env is not None:
            return self._env.close()
        return None


def _stack_matrix(states: List[np.ndarray]) -> np.ndarray:
    # Stack 2D images along the depth dimension
    if states[0].ndim == 2 or states[0].ndim == 3:
        return np.dstack(states)
    else:
        raise Exception(
            f"Expected input numpy array with 2 or 3 dimensions, but received input with {states[0].ndim} dimensions."
        )


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
                throttle = 0.4
                brake = 0
                steering = -0.8
            elif model_action == 3:
                # Turn right
                throttle = 0.4
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


def observation_adapter(obs: smarts_sensors.Observation) -> np.ndarray:
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


def predator_reward_adapter(obs, env_reward):
    reward = 0
    ego = obs.ego_vehicle_state

    # Penalty for driving off road
    if obs.events.off_road:
        reward -= 50
        print(f"Predator {ego.id} went off road.")
        return np.float32(reward)

    # Reward for colliding
    for c in obs.events.collisions:
        if "prey" in c.collidee_id:
            reward += 30
            print(f"Predator {ego.id} collided with prey vehicle {c.collidee_id}.")
        else:
            reward -= 20
            print(f"Predator {ego.id} collided with predator vehicle {c.collidee_id}.")

    # Distance based reward
    targets = get_targets(obs.neighborhood_vehicle_states, "prey")
    if targets:
        distances = distance_to_targets(ego, targets)
        min_distance = np.amin(distances)
        dist_reward = inverse(min_distance)
        reward += (
            np.clip(dist_reward, 0, NEIGHBOURHOOD_RADIUS) / NEIGHBOURHOOD_RADIUS * 10
        )  # Reward [0:10]
    # else:  # No neighborhood preys
    #     reward -= 1

    # Penalty for not moving
    # if obs.events.not_moving:
    if obs.ego_vehicle_state.speed > 5.0:
        reward += 1

    # Reward for staying on road
    # reward += 1

    return np.float32(reward)


def prey_reward_adapter(obs, env_reward):
    reward = 0
    ego = obs.ego_vehicle_state

    # Penalty for driving off road
    if obs.events.off_road:
        reward -= 50
        print(f"Prey {ego.id} went off road.")
        return np.float32(reward)

    # Penalty for colliding
    for c in obs.events.collisions:
        if "predator" in c.collidee_id:
            reward -= 30
            print(f"Prey {ego.id} collided with predator vehicle {c.collidee_id}.")
        else:
            reward -= 30
            print(f"Prey {ego.id} collided with prey vehicle {c.collidee_id}.")

    # Distance based reward
    targets = get_targets(obs.neighborhood_vehicle_states, "predator")
    if targets:
        distances = distance_to_targets(ego, targets)
        min_distance = np.amin(distances)
        dist_reward = inverse(min_distance)
        reward -= (
            np.clip(dist_reward, 0, NEIGHBOURHOOD_RADIUS) / NEIGHBOURHOOD_RADIUS * 10
        )  # Reward [-10:0]
        # pass
    # else:  # No neighborhood predators
    #     reward += 1

    # Penalty for not moving
    if obs.ego_vehicle_state.speed > 5.0:
        reward += 1

    # Reward for staying on road
    # reward += 1

    return np.float32(reward)
