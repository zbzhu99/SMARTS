import gym
import numpy as np

from examples.auto_drive.env.controller import Controller
from smarts.core import colors as smarts_colors

def info_adapter(obs, reward, info):
    return info

def action_space(controller):
    if controller == Controller.CONTINUOUS:
        return gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float
        )

    if controller == Controller.LANE:
        return gym.spaces.Dsicrete(4)
        
    if controller == Controller.DISCRETE:
        return gym.spaces.Dsicrete(5)

    raise Exception("Unknown controller.")


def action_adapter(controller):
    # Action space
    # throttle: [0, 1]
    # brake: [0, 1]
    # steering: [-1, 1]

    if controller == Controller.CONTINUOUS:

        def continuous(model_action):
            throttle, brake, steering = model_action
            # Modify action space limits
            throttle = (throttle + 1) / 2
            brake = (brake + 1) / 2
            return np.array([throttle, brake, steering], dtype=np.float)

        return continuous

    if controller == Controller.LANE:

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

    if controller == Controller.DISCRETE:

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
    # coloured_self = rgb.copy()
    # coloured_self[123:132, 126:130, 0] = smarts_colors.Colors.Lime.value[0] * 255
    # coloured_self[123:132, 126:130, 1] = smarts_colors.Colors.Lime.value[1] * 255
    # coloured_self[123:132, 126:130, 2] = smarts_colors.Colors.Lime.value[2] * 255
    # frame = coloured_self.astype(np.uint8)
    frame = rgb.astype(np.uint8)

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
