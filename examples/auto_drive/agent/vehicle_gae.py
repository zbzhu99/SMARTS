import numpy as np


class VehicleGAE:
    def __init__(self, name, config):
        self.name = name
        self._config = config
        self._gamma = config["gamma"]
        self._lam = config["lam"]
        self.reset()

    def reset(self):
        # Buffer initialization
        self.observation_buffer = []
        self.action_buffer = []
        self.advantage_buffer = []
        self.return_buffer = []
        self.logprobability_buffer = []
        self._done_buffer = []
        self._reward_buffer = []
        self._value_buffer = []
        self._last_value = None

    def add_transition(self, observation, action, reward, value, logprobability, done):
        # Append one step of agent-environment interaction
        self.observation_buffer.append(observation)
        self.action_buffer.append(action)
        self._reward_buffer.append(reward)
        self._value_buffer.append(value)
        self.logprobability_buffer.append(logprobability)
        self._done_buffer.append(done)

    def add_last_transition(self, value):
        self._last_value = value

    def finish_trajectory(self):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        values = np.append(self._value_buffer, self._last_value)
        dones = np.array(self._done_buffer)
        deltas = (
            self._reward_buffer + (self._gamma * values[1:] * (1 - dones)) - values[:-1]
        )

        # Generalised advantage estimate
        self.advantage_buffer = np.append(deltas, 0).astype(np.float32)
        for t in reversed(range(len(deltas))):
            self.advantage_buffer[t] = deltas[
                t
            ] + self._gamma * self._lam * self.advantage_buffer[t + 1] * (1 - dones[t])
        self.advantage_buffer = self.advantage_buffer[:-1]
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std

        # Return = Discounted sum of rewards
        self.return_buffer = np.append(self._reward_buffer, self._last_value).astype(
            np.float32
        )
        for t in reversed(range(len(self._reward_buffer))):
            self.return_buffer[t] = self._reward_buffer[
                t
            ] + self._gamma * self.return_buffer[t + 1] * (1 - dones[t])
        self.return_buffer = self.return_buffer[:-1]


# def discounted_cumulative_sums(x, discount):
#     # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
#     return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
