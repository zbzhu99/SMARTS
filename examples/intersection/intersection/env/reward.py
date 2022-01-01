import gym
import numpy as np


class Reward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, env_reward, done, info = self.env.step(action)

        wrapped_reward = {
            agent_id: self._reward(obs[agent_id], agent_reward)
            for agent_id, agent_reward in env_reward.items()
        }

        return obs, wrapped_reward, done, info

    def _reward(self, obs, env_reward):
        reward = 0

        # Penalty for driving off road
        if obs.events.off_road:
            reward -= 200
            print("----- Off road")
            return np.float32(reward)

        # Penalty for colliding
        if len(obs.events.collisions) > 0:
            reward -= 200
            print("----- Collided")
            return np.float32(reward)

        if obs.events.off_route:
            reward -= 200
            print("----- Off route")
            return np.float32(reward)

        if obs.events.wrong_way:
            print("----- Wrong Way")
            reward -= 0.1

        # Distance based reward
        reward += env_reward

        if obs.events.reached_goal:
            reward += 50

        return np.float32(reward)
