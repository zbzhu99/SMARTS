from examples.gameOfTag.types import AgentType
import numpy as np
import tensorflow as tf
import scipy.signal


class TagAgentKeras:
    def __init__(self, name, config, gamma=0.99, lam=0.95):
        if AgentType.PREDATOR in name or AgentType.PREY in name:
            self.name = name
        else:
            raise Exception(f"Expected predator or prey, but got {name}.")
        self._config = config
        self._gamma = config["model_para"]["gamma"]
        self._lam = lam
        self.reset()

    def reset(self):
        # Buffer initialization
        self.observation_buffer = []
        self.action_buffer = []
        self.advantage_buffer = []
        self.reward_buffer = []
        self.return_buffer = []
        self.value_buffer = []
        self.logprobability_buffer = []
        self.done_buffer = []
        self._last_value = None
        self.gamma, self.lam = self._gamma, self._lam

    def add_transition(self, observation, action, reward, value, logprobability, done):
        # Append one step of agent-environment interaction
        self.observation_buffer.append(observation)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.value_buffer.append(value)
        self.logprobability_buffer.append(logprobability)
        self.done_buffer.append(done)

    def add_last_transition(self, value):
        self._last_value = value

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
