from enum import Enum

import gym
import numpy as np


class Action(gym.ActionWrapper):
    def __init__(self, env, wrapper: str):
        super().__init__(env)

        if wrapper == _Wrapper.CONTINUOUS:
            self._wrapper, self.action_space = continuous()
        elif wrapper == _Wrapper.LANE:
            self._wrapper, self.action_space = lane()
        elif wrapper == _Wrapper.DISCRETE:
            self._wrapper, self.action_space = discrete()
        else:
            raise Exception("Unknown action wrapper.")

    def action(self, act):
        wrapped_act = {
            agent_id: self._wrapper(agent_act) for agent_id, agent_act in act.items()
        }
        return wrapped_act


class _Wrapper(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    LANE = "lane"


def continuous():
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def wrapper(model_action):
        throttle, brake, steering = model_action
        throttle = (throttle + 1) / 2
        brake = (brake + 1) / 2
        return np.array([throttle, brake, steering], dtype=np.float32)

    return wrapper, space


def lane():
    space = gym.spaces.Discrete(4)

    def wrapper(model_action):
        if model_action == 0:
            return "keep_lane"
        if model_action == 1:
            return "slow_down"
        if model_action == 2:
            return "change_lane_left"
        if model_action == 3:
            return "change_lane_right"
        raise Exception("Unknown model action.")

    return wrapper, space


def discrete():
    space = gym.spaces.Discrete(5)

    def wrapper(model_action):
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

    return wrapper, space
