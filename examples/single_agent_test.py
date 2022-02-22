import time

import gym
import numpy as np

from examples.argument_parser import default_argument_parser
from smarts.core.agent import Agent
from smarts.core.utils.episodes import episodes
from smarts.env import build_scenario


class ChaseWaypointsAgent(Agent):
    def act(self, obs):
        cur_lane_index = obs.ego["lane_index"]
        next_lane_index = obs.waypoints["lane_index"][0, 0]

        return (
            obs.waypoints["speed_limit"][0, 0] / 4,
            np.sign(next_lane_index - cur_lane_index),
        )


def main(headless, num_episodes):
    env = gym.make(
        "smarts.env:intersection-v0",
        headless=False,
        sumo_headless=True,
        visdom=False,
    )

    for episode in episodes(n=num_episodes):
        agent = ChaseWaypointsAgent()
        observation = env.reset()
        episode.record_scenario(env.scenario_log)

        import time
        first = 0
        done = False
        print("==============================================")
        while not done:
            if first == 0:
                print("Step count == ", env._smarts.step_count)
                first = 1
            agent_action = agent.act(observation)
            observation, reward, done, info = env.step(agent_action)
            episode.record_step(observation, reward, done, info)
            time.sleep(0.1)

        print(observation.events)
        print(info.keys())
        print(info["env_obs"].events.collisions)
        print("Step count == ", env._smarts.step_count)
        print(
            "Score ==",
            info["score"],
            "Pos ==",
            observation.ego["pos"],
        )
        time.sleep(3)


    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    main(
        headless=True,  # args.headless,
        num_episodes=100,  # args.episodes,
    )
