import numpy as np
import os
import tensorflow as tf
import yaml

from examples.gameOfTag import agent as got_agent
from examples.gameOfTag import env as got_env
from pathlib import Path


def evaluate(ppo_predator, ppo_prey, config):
    # Create env
    print("[INFO] Creating environments")
    seed = 42
    env = got_env.TagEnv(config, seed)

    # Create agent
    print("[INFO] Creating agents")
    all_agents = {
        name: got_agent.TagAgent(name, config)
        for name in config["env_para"]["agent_ids"]
    }
    all_predators_id = env.predators
    all_preys_id = env.preys

    print("[INFO] Loop ...")
    states_t = env.reset()
    episode = 0
    steps_t = 0
    episode_reward_predator = 0
    episode_reward_prey = 0
    [agent.reset() for _, agent in all_agents.items()]
    while True:
        if steps_t % 100 == 0:
            print(f"Evaluation. Seed: {seed}, Steps: {steps_t}")

        # Predict and value action given state
        actions_t = {}
        action_samples_t = {}
        values_t = {}
        (
            actions_t_predator,
            action_samples_t_predator,
            values_t_predator,
        ) = ppo_predator.act(states_t)
        actions_t_prey, action_samples_t_prey, values_t_prey = ppo_prey.act(states_t)
        actions_t.update(actions_t_predator)
        actions_t.update(actions_t_prey)
        action_samples_t.update(action_samples_t_predator)
        action_samples_t.update(action_samples_t_prey)
        values_t.update(values_t_predator)
        values_t.update(values_t_prey)

        # Sample action from a distribution
        action_numpy_t = {
            vehicle: action_sample_t.numpy()
            for vehicle, action_sample_t in action_samples_t.items()
        }
        next_states_t, rewards_t, dones_t, _ = env.step(action_numpy_t)
        steps_t += 1

        # Store state, action and reward
        for agent_id, reward in rewards_t.items():
            all_agents[agent_id].add_trajectory(
                reward=reward,
            )
            if "predator" in agent_id.name:
                episode_reward_predator += reward
            else:
                episode_reward_prey += reward
            if dones_t[agent_id] == 1:
                # Remove done agents
                del next_states_t[agent_id]
                # Print done agents
                print(f"Done: {agent_id}. Step: {steps_t}.")

        # Break when episode completes
        if dones_t["__all__"]:
            break

        # Assign next_states to states
        states_t = next_states_t

    # Close env
    env.close()

    return (
        episode_reward_predator,
        episode_reward_prey,
    )


if __name__ == "__main__":
    config_yaml = (Path(__file__).absolute().parent).joinpath("got.yaml")
    with open(config_yaml, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Silence the logs of TF
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.random.set_seed(42)
    np.random.seed(42)

    # Load saved models
    if "checkpoint_predator" not in config["benchmark"]:
        raise Exception("Missing predator model checkpoint")
    if "checkpoint_prey" not in config["benchmark"]:
        raise Exception("Missing prey model checkpoint")

    model_checkpoint_predator_dir = config["benchmark"]["checkpoint_predator"]
    model_checkpoint_predator = tf.train.latest_checkpoint(
        model_checkpoint_predator_dir
    )
    model_predator = got_agent.TagModel(
        "predator", env, config, model_checkpoint=model_checkpoint_predator
    )

    model_checkpoint_prey_dir = config["benchmark"]["checkpoint_prey"]
    model_checkpoint_prey = tf.train.latest_checkpoint(model_checkpoint_prey_dir)
    model_prey = got_agent.TagModel(
        "prey", env, config, model_checkpoint=model_checkpoint_prey
    )

    # Evaluate
    (
        avg_reward_predator,
        avg_reward_prey,
        value_error_predator,
        value_error_prey,
    ) = evaluate(model_predator, model_prey, config)

    print("Finished evaluation------------------------")
    print(f"avg_reward_predator: {avg_reward_predator}")
    print(f"avg_reward_prey: {avg_reward_prey}")
    print(f"value_error_predator: {value_error_predator}")
    print(f"value_error_prey: {value_error_prey}")
