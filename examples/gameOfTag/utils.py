import numpy as np


def compute_returns(rewards, bootstrap_value, terminals, gamma):
    print("-----------returns------------------")
    print("rewards: ", rewards.shape)
    print("bootstrap_value: ", bootstrap_value)
    print("terminals: ", terminals.shape)
    print("gamme: ", gamma)

    returns = []
    R = bootstrap_value
    for i in reversed(range(len(rewards))):
        R = rewards[i] + (1.0 - terminals[i]) * gamma * R
        returns.append(R)
    returns = reversed(returns)
    return np.array(list(returns))


def compute_gae(rewards, values, bootstrap_values, terminals, gamma, lam):
    values = np.vstack((values, bootstrap_values))

    # Compute delta
    deltas = []
    for i in reversed(range(len(rewards))):
        V = rewards[i] + (1.0 - terminals[i]) * gamma * values[i + 1]
        delta = V - values[i]
        deltas.append(delta)
    deltas = np.array(list(reversed(deltas)))

    # Compute gae
    A = deltas[-1, :]
    advantages = [A]
    for i in reversed(range(len(deltas) - 1)):
        A = deltas[i] + (1.0 - terminals[i]) * gamma * lam * A
        advantages.append(A)
    advantages = reversed(advantages)

    return np.array(list(advantages))