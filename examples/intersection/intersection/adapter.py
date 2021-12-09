import numpy as np


def info_adapter(obs, reward, info):
    return info


def reward_adapter(obs, env_reward):
    ego = obs.ego_vehicle_state
    reward = 0

    # Penalty for driving off road
    if obs.events.off_road:
        reward -= 10
        return np.float32(reward)

    # Penalty for colliding
    if len(obs.events.collisions) > 0:
        reward -= 10
        return np.float32(reward)

    if obs.events.wrong_way:
        reward -= 0.5

    if obs.events.off_route:
        reward -= 0.5
    else:
        # Distance based reward
        reward += env_reward

    if obs.events.reached_goal:
        reward += 10

    return np.float32(reward)
