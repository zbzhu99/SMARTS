import tensorflow as tf

# gpus = tf.config.list_physical_devices("GPU")
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.list_logical_devices("GPU")
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

import numpy as np
import os
import random
import signal
import sys
import yaml

from examples.gameOfTag import env as got_env
from examples.gameOfTag import agent as got_agent
from examples.gameOfTag import ppo as got_ppo
from pathlib import Path


def main(config):

    print("[INFO] Train")
    # Save and eval interval
    save_interval = config["model_para"].get("save_interval", 50)
    eval_interval = config["model_para"].get("eval_interval", 50)

    # Traning parameters
    num_train_epochs = config["model_para"]["num_train_epochs"]
    batch_size = config["model_para"]["batch_size"]
    max_batch = config["model_para"]["max_batch"]
    clip_value = config["model_para"]["clip_value"]
    critic_loss_weight = config["model_para"]["critic_loss_weight"]
    ent_discount_val = config["model_para"]["entropy_loss_weight"]
    ent_discount_rate = config["model_para"]["entropy_loss_discount_rate"]

    # Create env
    print("[INFO] Creating environments")
    seed = config["env_para"]["seed"]
    # seed = random.randint(0, 4294967295)  # [0, 2^32 -1)
    env = got_env.TagEnv(config, seed)

    # Create agent
    print("[INFO] Creating agents")
    all_agents = {
        name: got_agent.TagAgent(name, config)
        for name in config["env_para"]["agent_ids"]
    }
    all_predators_id = env.predators
    all_preys_id = env.preys

    # Create model
    print("[INFO] Creating model")
    ppo_predator = got_ppo.PPO("predator", config)
    ppo_prey = got_ppo.PPO("prey", config)

    def interrupt(*args):
        ppo_predator.save()
        ppo_prey.save()
        print("Interrupt key detected.")
        sys.exit(0)

    # Catch keyboard interrupt and terminate signal
    signal.signal(signal.SIGINT, interrupt)

    print("[INFO] Batch loop")
    states_t = env.reset()
    episode = 0
    steps_t = 0
    episode_reward_predator = 0
    episode_reward_prey = 0
    for batch_num in range(max_batch):
        [agent.reset() for _, agent in all_agents.items()]
        active_agents = {}

        print("[INFO] New batch data collection")
        for _ in range(batch_size):

            # Update all agents which were active in this batch
            active_agents.update({agent_id: True for agent_id, _ in states_t.items()})

            # Predict and value action given state
            actions_t = {}
            action_samples_t = {}
            values_t = {}
            (
                actions_t_predator,
                action_samples_t_predator,
                values_t_predator,
            ) = ppo_predator.act(states_t)
            actions_t_prey, action_samples_t_prey, values_t_prey = ppo_prey.act(
                states_t
            )
            actions_t.update(actions_t_predator)
            actions_t.update(actions_t_prey)
            action_samples_t.update(action_samples_t_predator)
            action_samples_t.update(action_samples_t_prey)
            values_t.update(values_t_predator)
            values_t.update(values_t_prey)

            # Sample action from a distribution
            action_numpy_t = {
                vehicle: action_sample_t.numpy()[0]
                for vehicle, action_sample_t in action_samples_t.items()
            }
            next_states_t, rewards_t, dones_t, _ = env.step(action_numpy_t)
            steps_t += 1

            # Store state, action and reward
            for agent_id, _ in states_t.items():
                all_agents[agent_id].add_trajectory(
                    action=action_samples_t[agent_id],
                    value=values_t[agent_id].numpy()[0],
                    state=states_t[agent_id],
                    done=int(dones_t[agent_id]),
                    prob=actions_t[agent_id],
                    reward=rewards_t[agent_id],
                )
                if "predator" in agent_id:
                    episode_reward_predator += rewards_t[agent_id]
                else:
                    episode_reward_prey += rewards_t[agent_id]
                if dones_t[agent_id] == 1:
                    # Remove done agents
                    del next_states_t[agent_id]
                    # Print done agents
                    print(f"Done: {agent_id}. Step: {steps_t}.")

            # Reset when episode completes
            if dones_t["__all__"]:
                # Next episode
                next_states_t = env.reset()
                episode += 1

                # Log rewards
                print(
                    f"Episode: {episode},"
                    f"episode reward predator: {episode_reward_predator}, "
                    f"episode reward prey: {episode_reward_prey}, "
                )
                with ppo_predator.tb.as_default():
                    tf.summary.scalar(
                        "episode_reward_predator", episode_reward_predator, episode
                    )
                with ppo_prey.tb.as_default():
                    tf.summary.scalar(
                        "episode_reward_prey", episode_reward_prey, episode
                    )

                # Reset counters
                episode_reward_predator = 0
                episode_reward_prey = 0
                steps_t = 0

            # Assign next_states to states
            states_t = next_states_t

        # Compute and store last state value
        for agent_id in active_agents.keys():
            if dones_t.get(agent_id, None) == 0:  # Agent not done yet
                # Calculate last values (bootstrap values)
                if "predator" in agent_id:
                    _, _, next_values_t = ppo_predator.act(
                        {agent_id: next_states_t[agent_id]}
                    )
                elif "prey" in agent_id:
                    _, _, next_values_t = ppo_prey.act(
                        {agent_id: next_states_t[agent_id]}
                    )
                else:
                    raise Exception(f"Unknown {agent_id}.")
                # Store last values
                all_agents[agent_id].add_last_transition(
                    value=next_values_t[agent_id].numpy()[0]
                )
            else:  # Agent done
                # Store last values
                all_agents[agent_id].add_last_transition(value=0)

        # Compute generalised advantages
        for agent_id in active_agents.keys():
            all_agents[agent_id].compute_advantages()
            probs_softmax = tf.nn.softmax(
                tf.squeeze(tf.stack(all_agents[agent_id].probs))
            )
            all_agents[agent_id].probs_softmax = probs_softmax
            actions = tf.squeeze(tf.stack(all_agents[agent_id].actions))
            action_inds = tf.stack(
                [tf.range(0, actions.shape[0]), tf.cast(actions, tf.int32)], axis=1
            )
            all_agents[agent_id].action_inds = action_inds

        predator_total_loss = np.zeros((num_train_epochs))
        predator_actor_loss = np.zeros((num_train_epochs))
        predator_critic_loss = np.zeros(((num_train_epochs)))
        predator_entropy_loss = np.zeros((num_train_epochs))

        prey_total_loss = np.zeros((num_train_epochs))
        prey_actor_loss = np.zeros((num_train_epochs))
        prey_critic_loss = np.zeros(((num_train_epochs)))
        prey_entropy_loss = np.zeros((num_train_epochs))

        # Train predator
        for epoch in range(num_train_epochs):
            for agent_id in active_agents.keys():
                agent = all_agents[agent_id]
                print("OLD PROBS inside agent")
                print(agent.probs_softmax)
                if agent_id in all_predators_id:
                    loss_tuple = got_ppo.train_model(
                        model=ppo_predator.model,
                        optimizer=ppo_predator.optimizer,
                        action_inds=agent.action_inds,
                        old_probs=tf.gather_nd(agent.probs_softmax, agent.action_inds),
                        states=agent.states,
                        advantages=agent.advantages,
                        discounted_rewards=agent.discounted_rewards,
                        ent_discount_val=ent_discount_val,
                        clip_value=clip_value,
                        critic_loss_weight=critic_loss_weight,
                    )
                    predator_total_loss[epoch] += loss_tuple[0]
                    predator_critic_loss[epoch] += loss_tuple[1]
                    predator_actor_loss[epoch] += loss_tuple[2]
                    predator_entropy_loss[epoch] += loss_tuple[3]

                if agent_id in all_preys_id:
                    loss_tuple = got_ppo.train_model(
                        model=ppo_predator.model,
                        optimizer=ppo_predator.optimizer,
                        action_inds=agent.action_inds,
                        old_probs=tf.gather_nd(agent.probs_softmax, agent.action_inds),
                        states=agent.states,
                        advantages=agent.advantages,
                        discounted_rewards=agent.discounted_rewards,
                        ent_discount_val=ent_discount_val,
                        clip_value=clip_value,
                        critic_loss_weight=critic_loss_weight,
                    )
                    prey_total_loss[epoch] += loss_tuple[0]
                    prey_critic_loss[epoch] += loss_tuple[1]
                    prey_actor_loss[epoch] += loss_tuple[2]
                    prey_entropy_loss[epoch] += loss_tuple[3]

        ent_discount_val *= ent_discount_rate

        # Elapsed steps
        step = (batch_num + 1) * batch_size

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

        # # Evaluate model
        # if batch_num % eval_interval == 0:
        #     print("[INFO] Running evaluation...")
        #     (
        #         avg_reward_predator,
        #         avg_reward_prey,
        #     ) = evaluate.evaluate(ppo_predator, ppo_prey, config)
        #     model_predator.write_to_summary("eval_avg_reward", avg_reward_predator)
        #     model_predator.write_to_summary("eval_value_error", value_error_predator)
        #     model_prey.write_to_summary("eval_avg_reward", avg_reward_prey)
        #     model_prey.write_to_summary("eval_value_error", value_error_prey)

        # Save model
        if batch_num % save_interval == 0:
            print("[INFO] Saving model...")
            ppo_predator.save()
            ppo_prey.save()

    # Close env
    env.close()


if __name__ == "__main__":
    config_yaml = (Path(__file__).absolute().parent).joinpath("got.yaml")
    with open(config_yaml, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Silence the logs of TF
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Check for GPU device
    # device_name = tf.test.gpu_device_name()
    # if device_name != "/device:GPU:0":
    #     print("Not configured to use GPU or GPU not available.")
    #     raise SystemError('GPU device not found')

    # strategy = tf.distribute.MirroredStrategy()
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    main(config=config)
