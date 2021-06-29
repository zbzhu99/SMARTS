import numpy as np
import os
import yaml
from examples.gameOfTag import agent as got_agent
from examples.gameOfTag import env as got_env
from pathlib import Path


def evaluate(model_predator, model_prey, config):
    total_reward = 0

    env = got_env.TagEnv(config, 42)
    all_agents = {name:got_agent.TagAgent(name, config) for name in config["env_para"]["agent_ids"]}
    all_predators_id = env.predators
    all_preys_id = env.preys

    states_t = env.reset()
    while True:
        # Predict action given state: π(a_t | s_t; θ)
        actions_t={}
        values_t={}
        actions_t_predator, values_t_predator = model_predator.act(states_t)
        actions_t_prey, values_t_prey = model_prey.act(states_t)
        actions_t.update(actions_t_predator)
        actions_t.update(actions_t_prey)
        values_t.update(values_t_predator)
        values_t.update(values_t_prey)

        next_states_t, rewards_t, dones_t, _ = env.step(actions_t)
        
        # Store state, action and reward
        for agent_id, _ in rewards_t.items():
            all_agents[agent_id].add_trajectory(
                state=states_t[agent_id],
                action=actions_t[agent_id],
                value=np.squeeze(values_t[agent_id], axis=-1),
                reward=rewards_t[agent_id],
                done=dones_t[agent_id],
            )

        # Break when episode completes
        if dones_t['__all__']:
            break

        # Assign next_states to states
        states_t = next_states_t

    total_reward = [np.sum(all_agents[agent_id].rewards) for agent_id in all_predators_id]
    total_reward_predator = np.sum(total_reward)
    total_reward = [np.sum(all_agents[agent_id].rewards) for agent_id in all_preys_id]
    total_reward_prey = np.sum(total_reward)

    value_errors = [value_error(np.array(all_agents[agent_id].values), all_agents[agent_id].compute_returns()) for agent_id in all_predators_id]
    value_error_predator = np.mean(value_errors)
    value_errors = [value_error(np.array(all_agents[agent_id].values), all_agents[agent_id].compute_returns()) for agent_id in all_preys_id]
    value_error_prey = np.mean(value_errors)

    return total_reward_predator, total_reward_prey, value_error_predator, value_error_prey


def value_error(values, returns):
    return np.mean(np.square(values - returns))



# if __name__ == "__main__":
#     config_yaml=(Path(__file__).absolute().parent).joinpath("smarts.yaml")
#     with open(config_yaml, "r") as file:
#         config = yaml.load(file, Loader=yaml.FullLoader)

#     # Silence the logs of TF
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#     evaluate()