import gym
import numpy as np


class Reward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, env_reward, done, info = self.env.step(action)
        return obs, self.reward(obs, env_reward), done, info

    def reward(self, obs, env_reward):
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

        if obs.events.off_route:
            reward -= 10
            return np.float32(reward)

        if obs.events.wrong_way:
            reward -= 0.5

        # Distance based reward
        reward += env_reward

        if obs.events.reached_goal:
            reward += 20

        return np.float32(reward)
