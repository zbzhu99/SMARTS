import os

# Set pythonhashseed
os.environ["PYTHONHASHSEED"] = "0"
# Silence the logs of TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
import numpy as np

np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
import random as python_random

python_random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
import tensorflow as tf

tf.random.set_seed(123)

# --------------------------------------------------------------------------

import signal
import sys
import warnings
import yaml


from examples.gameOfTag import env_keras as got_env
from examples.gameOfTag import agent_keras as got_agent
from examples.gameOfTag import ppo_keras as got_ppo
from examples.gameOfTag.types import AgentType, Mode
from pathlib import Path


def main(config):

    print("[INFO] Train")
    save_interval = config["model_para"].get("save_interval", 20)
    mode = Mode(config["model_para"]["mode"])  # Mode: Evaluation or Testing

    # Traning parameters
    n_steps = config["model_para"]["n_steps"]
    max_traj = config["model_para"]["max_traj"]

    # Create env
    print("[INFO] Creating environments")
    seed = config["env_para"]["seed"]
    ## seed = random.randint(0, 4294967295)  # [0, 2^32 -1)
    env = got_env.TagEnvKeras(config, seed)

    # Create agent
    print("[INFO] Creating agents")
    all_agents = {name: got_agent.TagAgentKeras(name, config) for name in env.agent_ids}
    all_predator_ids = env.predators
    all_prey_ids = env.preys

    # Create model
    print("[INFO] Creating model")
    ppo_predator = got_ppo.PPOKeras(
        AgentType.PREDATOR.value,
        config,
        all_predator_ids,
        config["env_para"]["seed"] + 1,
    )
    ppo_prey = got_ppo.PPOKeras(
        AgentType.PREY.value, config, all_prey_ids, config["env_para"]["seed"] + 2
    )

    def interrupt(*args):
        nonlocal mode
        if mode == Mode.TRAIN:
            ppo_predator.save(-1)
            ppo_prey.save(-1)
        env.close()
        print("Interrupt key detected.")
        sys.exit(0)

    # Catch keyboard interrupt and terminate signal
    signal.signal(signal.SIGINT, interrupt)

    print("[INFO] Batch loop")
    obs_t = env.reset()
    episode = 0
    steps_t = 0
    episode_reward_predator = 0
    episode_reward_prey = 0
    for traj_num in range(max_traj):
        [agent.reset() for _, agent in all_agents.items()]
        active_agents = {}

        print(f"[INFO] New batch data collection {traj_num}/{max_traj}")
        for cur_step in range(n_steps):

            # Update all agents which were active in this batch
            active_agents.update({agent_id: True for agent_id, _ in obs_t.items()})

            # Given state, predict action and value
            logit_t = {}
            action_t = {}
            value_t = {}
            logprobability_t = {}

            logit, action = ppo_predator.actor(obs=obs_t, train=mode)
            value = ppo_predator.critic(obs_t)
            logit_t.update(logit)
            action_t.update(action)
            value_t.update(value)

            logit, action = ppo_prey.actor(obs=obs_t, train=mode)
            value = ppo_prey.critic(obs_t)
            logit_t.update(logit)
            action_t.update(action)
            value_t.update(value)

            for agent_id, logit in logit_t.items():
                logprobability_t[agent_id] = got_ppo.logprobabilities(
                    logit, action_t[agent_id]
                )

            # Sample action from a distribution
            action_numpy_t = {
                vehicle: action[0].numpy() for vehicle, action in action_t.items()
            }
            next_obs_t, reward_t, done_t, _ = env.step(action_numpy_t)
            steps_t += 1

            # Store observation, action, and reward
            for agent_id, _ in obs_t.items():
                all_agents[agent_id].add_transition(
                    observation=obs_t[agent_id],
                    action=action_t[agent_id],
                    reward=reward_t[agent_id],
                    value=value_t[agent_id],
                    logprobability=logprobability_t[agent_id],
                    done=int(done_t[agent_id]),
                )
                if AgentType.PREDATOR in agent_id:
                    episode_reward_predator += reward_t[agent_id]
                else:
                    episode_reward_prey += reward_t[agent_id]
                if done_t[agent_id] == 1:
                    # Remove done agents
                    del next_obs_t[agent_id]
                    # Print done agents
                    print(
                        f"   Done: {agent_id}. Cur_Step: {cur_step}. Step: {steps_t}."
                    )

            # Reset when episode completes
            if done_t["__all__"]:
                # Next episode
                next_obs_t = env.reset()
                episode += 1

                # Log rewards
                print(
                    f"   Episode: {episode}. Cur_Step: {cur_step}. "
                    f"Episode reward predator: {episode_reward_predator}, "
                    f"Episode reward prey: {episode_reward_prey}."
                )
                ppo_predator.write_to_tb(
                    [("episode_reward_predator", episode_reward_predator, episode)]
                )
                ppo_prey.write_to_tb(
                    [("episode_reward_prey", episode_reward_prey, episode)]
                )

                # Reset counters
                episode_reward_predator = 0
                episode_reward_prey = 0
                steps_t = 0

            # Assign next_obs to obs
            obs_t = next_obs_t

        # Skip the remainder if evaluating
        if mode == Mode.EVALUATE:
            continue

        # Compute and store last state value
        for agent_id in active_agents.keys():
            if done_t.get(agent_id, None) == 0:  # Agent not done yet
                if AgentType.PREDATOR in agent_id:
                    next_value_t = ppo_predator.critic({agent_id: next_obs_t[agent_id]})
                elif AgentType.PREY in agent_id:
                    next_value_t = ppo_prey.critic({agent_id: next_obs_t[agent_id]})
                else:
                    raise Exception(f"Unknown {agent_id}.")
                all_agents[agent_id].add_last_transition(
                    value=next_value_t[agent_id][0]
                )
            else:  # Agent is done
                all_agents[agent_id].add_last_transition(value=np.float32(0))

        for agent_id in active_agents.keys():
            # Compute generalised advantages
            all_agents[agent_id].finish_trajectory()

            actions = all_agents[agent_id].actions
            action_inds = tf.stack(
                [tf.range(0, len(actions)), tf.cast(actions, tf.int32)], axis=1
            )

            # Compute old probabilities
            probs_softmax = tf.nn.softmax(all_agents[agent_id].probs)
            all_agents[agent_id].old_probs = tf.gather_nd(probs_softmax, action_inds)

        predator_total_loss = np.zeros((num_train_epochs))
        predator_actor_loss = np.zeros((num_train_epochs))
        predator_critic_loss = np.zeros(((num_train_epochs)))
        predator_entropy_loss = np.zeros((num_train_epochs))

        prey_total_loss = np.zeros((num_train_epochs))
        prey_actor_loss = np.zeros((num_train_epochs))
        prey_critic_loss = np.zeros(((num_train_epochs)))
        prey_entropy_loss = np.zeros((num_train_epochs))

        # Elapsed steps
        step = (traj_num + 1) * n_steps

        print("[INFO] Training")
        # Train predator and prey.
        # Run multiple gradient ascent on the samples.
        for epoch in range(num_train_epochs):
            for agent_id in active_agents.keys():
                agent = all_agents[agent_id]
                if agent_id in all_predator_ids:
                    loss_tuple = got_ppo.train_model(
                        model=ppo_predator.model,
                        optimizer=ppo_predator.optimizer,
                        actions=agent.actions,
                        old_probs=agent.old_probs,
                        states=agent.states,
                        advantages=agent.advantages,
                        discounted_rewards=agent.discounted_rewards,
                        clip_value=clip_value,
                        critic_loss_weight=critic_loss_weight,
                        grad_batch=config["model_para"]["grad_batch"],
                    )
                    predator_total_loss[epoch] += loss_tuple[0]
                    predator_actor_loss[epoch] += loss_tuple[1]
                    predator_critic_loss[epoch] += loss_tuple[2]
                    predator_entropy_loss[epoch] += loss_tuple[3]

                if agent_id in all_prey_ids:
                    loss_tuple = got_ppo.train_model(
                        model=ppo_prey.model,
                        optimizer=ppo_prey.optimizer,
                        actions=agent.actions,
                        old_probs=agent.old_probs,
                        states=agent.states,
                        advantages=agent.advantages,
                        discounted_rewards=agent.discounted_rewards,
                        clip_value=clip_value,
                        critic_loss_weight=critic_loss_weight,
                        grad_batch=config["model_para"]["grad_batch"],
                    )
                    prey_total_loss[epoch] += loss_tuple[0]
                    prey_actor_loss[epoch] += loss_tuple[1]
                    prey_critic_loss[epoch] += loss_tuple[2]
                    prey_entropy_loss[epoch] += loss_tuple[3]


        print("[INFO] Record metrics")
        # Record predator performance
        records = []
        records.append(("predator_tot_loss", np.mean(predator_total_loss), step))
        records.append(("predator_critic_loss", np.mean(predator_critic_loss), step))
        records.append(("predator_actor_loss", np.mean(predator_actor_loss), step))
        records.append(("predator_entropy_loss", np.mean(predator_entropy_loss), step))
        ppo_predator.write_to_tb(records)

        # Record prey perfromance
        records = []
        records.append(("prey_tot_loss", np.mean(prey_total_loss), step))
        records.append(("prey_critic_loss", np.mean(prey_critic_loss), step))
        records.append(("prey_actor_loss", np.mean(prey_actor_loss), step))
        records.append(("prey_entropy_loss", np.mean(prey_entropy_loss), step))
        ppo_prey.write_to_tb(records)

        # Save model
        if traj_num % save_interval == 0:
            print("[INFO] Saving model")
            ppo_predator.save(step)
            ppo_prey.save(step)

    # Close env
    env.close()


if __name__ == "__main__":
    config_yaml = (Path(__file__).absolute().parent).joinpath("got_keras.yaml")
    with open(config_yaml, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Setup GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        warnings.warn(
            f"Not configured to use GPU or GPU not available.",
            ResourceWarning,
        )
        # raise SystemError("GPU device not found")

    # strategy = tf.distribute.MirroredStrategy()
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    main(config=config)
