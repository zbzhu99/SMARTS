import numpy as np


def info_adapter(obs, reward, info):
    return info


def get_targets(vehicles, target: str):
    target_vehicles = [vehicle for vehicle in vehicles if target in vehicle.id]
    return target_vehicles


def distance_to_targets(ego, targets):
    distances = (
        [np.linalg.norm(ego.position - target.position) for target in targets],
    )
    return distances


def inverse(x: float, radius: float) -> float:
    return -x + radius


def reward_adapter(obs, env_reward):
    ego = obs.ego_vehicle_state
    reward = 0

    # Penalty for driving off road
    if obs.events.off_road:
        reward -= 200
        print(f"----- Vehicle {ego.id} went off road.")
        return np.float32(reward)

    # Penalty for colliding
    if len(obs.events.collisions) > 0:
        reward -= 200
        print(f"----- Vehicle {ego.id} collided.")
        return np.float32(reward)

    # Distance based reward
    reward += env_reward

    return np.float32(reward)
