import gym
import matplotlib.pyplot as plt
import numpy as np

from examples.gameOfTag import agent as got_agent
from smarts.core import agent as smarts_agent
from smarts.core import agent_interface as smarts_agent_interface
from smarts.core import colors as smarts_colors
from smarts.core import controllers as smarts_controllers
from smarts.core import sensors as smarts_sensors
from smarts.env import hiway_env as smarts_hiway_env
from smarts.env.wrappers import frame_stack as smarts_frame_stack
from typing import Dict, List


NEIGHBOURHOOD_RADIUS = 55


class TagEnv(gym.Env):
    def __init__(self, config: Dict, seed: int = 42):
        # Update the agents number and env api type.
        self.config = config
        self.controller = config["env_para"]["controller"]  # Smarts controller
        self.neighborhood_radius = config["env_para"]["neighborhood_radius"]
        self.rgb_wh = config["env_para"]["rgb_wh"]
        self.predators = []
        self.preys = []
        for agent_id in config["env_para"]["agent_ids"]:
            if "predator" in agent_id:
                self.predators.append(agent_id)
            elif "prey" in agent_id:
                self.preys.append(agent_id)
            else:
                raise ValueError(
                    f"Expected agent_id to have prefix of 'predator' or 'prey', but got {agent_id}."
                )

        predator_interface = smarts_agent_interface.AgentInterface(
            max_episode_steps=config["env_para"]["max_episode_steps"],
            neighborhood_vehicles=smarts_agent_interface.NeighborhoodVehicles(
                radius=self.neighborhood_radius
            ),
            rgb=smarts_agent_interface.RGB(
                width=256, height=256, resolution=self.rgb_wh / 256
            ),
            vehicle_color="BrightRed",
            action=getattr(smarts_controllers.ActionSpaceType, "Continuous"),
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
                radius=self.neighborhood_radius
            ),
            rgb=smarts_agent_interface.RGB(
                width=256, height=256, resolution=self.rgb_wh / 256
            ),
            vehicle_color="BrightBlue",
            action=getattr(smarts_controllers.ActionSpaceType, "Continuous"),
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
                action_adapter=action_adapter(self.controller),
                info_adapter=info_adapter,
            )
            if "predator" in agent_id
            else smarts_agent.AgentSpec(
                interface=prey_interface,
                agent_builder=got_agent.TagAgent,
                observation_adapter=observation_adapter,
                reward_adapter=prey_reward_adapter,
                action_adapter=action_adapter(self.controller),
                info_adapter=info_adapter,
            )
            for agent_id in config["env_para"]["agent_ids"]
        }

        env = smarts_hiway_env.HiWayEnv(
            scenarios=config["env_para"]["scenarios"],
            agent_specs=agent_specs,
            headless=config["env_para"]["headless"],
            visdom=config["env_para"]["visdom"],
            seed=seed,
        )
        # Wrap env with FrameStack to stack multiple observations
        env = smarts_frame_stack.FrameStack(env=env, num_stack=3, num_skip=1)

        self.env = env

        # Continuous action space
        # self.action_space = gym.spaces.Box(
        #     low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        # )  # throttle, break, steering
        # Categorical action space
        self.action_space = gym.spaces.Discrete(config["model_para"]["action_dim"])
        # Observation space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=config["model_para"]["observation_dim"],
            dtype=np.uint8,
        )
        # self.observation_space = gym.spaces.Dict(
        #     {
        #         "image": gym.spaces.Box(
        #             low=0, high=255, shape=config["model_para"]["observation_dim"], dtype=np.uint8
        #         ),
        #         "scalar": gym.spaces.Box(
        #             low=np.array([-np.pi, 0, -1]),
        #             high=np.array([np.pi, 1e3, 1]),
        #             dtype=np.float32,
        #         ),
        #     }
        # )
        # scalar consists of
        # obs.ego_vehicle_state.heading = [-pi, pi]
        # obs.ego_vehicle_state.speed = [0, 1e3]
        # obs.ego_vehicle_state.steering = [-1, 1]

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment, if done is true, must clear obs array.

        :return: the observation of gym environment
        """

        raw_states = self.env.reset()

        # Stack observation into 3D numpy matrix
        states = {
            agent_id: stack_matrix(raw_state)
            for agent_id, raw_state in raw_states.items()
        }

        self.init_state = states
        return states

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.

        Accepts an action and returns a tuple (state, reward, done, info).

        :param action: action
        :param agent_index: the index of agent
        :return: state, reward, done, info
        """

        raw_states, rewards, dones, infos = self.env.step(action)

        # Stack observation into 3D numpy matrix
        states = {
            agent_id: stack_matrix(raw_state)
            for agent_id, raw_state in raw_states.items()
        }

        # Plot for debugging purposes
        # import matplotlib.pyplot as plt
        # fig=plt.figure(figsize=(10,10))
        # columns = 3
        # rows = len(states.keys())
        # for row, (agent_id, state) in enumerate(states.items()):
        #     for col in range(0, columns):
        #         img = state[:,:,col]
        #         fig.add_subplot(rows, columns, row*columns + col + 1)
        #         plt.imshow(img)
        # plt.show()

        return states, rewards, dones, infos

    def close(self):
        if self.env is not None:
            return self.env.close()
        return None


def stack_matrix(states: List[np.ndarray]) -> np.ndarray:
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

    if controller == "Continuous":
        # For DiagGaussian action space
        def action_adapter_continuous(model_action):
            throttle, brake, steering = model_action
            # Modify action space limits
            throttle = (throttle + 1) / 2
            brake = (brake + 1) / 2
            return np.array([throttle, brake, steering], dtype=np.float32)

        return action_adapter_continuous

    elif controller == "Categorical":
        # For Categorical action space
        def action_adapter_categorical(model_action):
            # Modify action space limits
            if model_action == 0:
                # Cruise
                throttle = 0.3
                brake = 0
                steering = 0
            elif model_action == 1:
                # Accelerate
                throttle = 0.5
                brake = 0
                steering = 0
            elif model_action == 2:
                # Turn left
                throttle = 0.3
                brake = 0
                steering = -0.8
            elif model_action == 3:
                # Turn right
                throttle = 0.3
                brake = 0
                steering = 0.8
            elif model_action == 4:
                # Brake
                throttle = 0
                brake = 0.8
                steering = 0
            else:
                raise Exception("Unknown model action category.")
            return np.array([throttle, brake, steering], dtype=np.float32)

        return action_adapter_categorical

    else:
        raise Exception(f"Unknown controller type.")


def observation_adapter(obs: smarts_sensors.Observation) -> Dict[str, np.ndarray]:
    # RGB grid map
    rgb = obs.top_down_rgb.data
    # # Replace self color to Lime
    coloured_self = rgb.copy()
    coloured_self[123:132, 126:130, 0] = smarts_colors.Colors.Lime.value[0] * 255
    coloured_self[123:132, 126:130, 1] = smarts_colors.Colors.Lime.value[1] * 255
    coloured_self[123:132, 126:130, 2] = smarts_colors.Colors.Lime.value[2] * 255
    frame = coloured_self
    frame = frame.astype(np.uint8)

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

    # scalar = np.array(
    #     (
    #         obs.ego_vehicle_state.heading,
    #         obs.ego_vehicle_state.speed,
    #         obs.ego_vehicle_state.steering,
    #     ),
    #     dtype=np.float32,
    # )

    # return {"image": frame, "scalar": scalar}
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

    # Reward for colliding
    for c in obs.events.collisions:
        if "prey" in c.collidee_id:
            reward += 20
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
            np.clip(dist_reward, 0, NEIGHBOURHOOD_RADIUS) / NEIGHBOURHOOD_RADIUS * 5
        )  # Reward [0:1]
    else:  # No neighborhood preys
        # reward -= 1
        pass

    # Reward for staying on road
    reward += 1

    return np.float32(reward)


def prey_reward_adapter(obs, env_reward):
    reward = 0
    ego = obs.ego_vehicle_state

    # Penalty for driving off road
    if obs.events.off_road:
        reward -= 50
        print(f"Prey {ego.id} went off road.")

    # Penalty for colliding
    for c in obs.events.collisions:
        if "predator" in c.collidee_id:
            reward -= 20
            print(f"Prey {ego.id} collided with predator vehicle {c.collidee_id}.")
        else:
            reward -= 20
            print(f"Prey {ego.id} collided with prey vehicle {c.collidee_id}.")

    # Distance based reward
    targets = get_targets(obs.neighborhood_vehicle_states, "predator")
    if targets:
        distances = distance_to_targets(ego, targets)
        min_distance = np.amin(distances)
        dist_reward = inverse(min_distance)
        reward -= (
            np.clip(dist_reward, 0, NEIGHBOURHOOD_RADIUS) / NEIGHBOURHOOD_RADIUS
        )  # Reward [-1:0]
    else:  # No neighborhood predators
        reward += 1

    # Reward for staying on road
    reward += 1

    return np.float32(reward)
