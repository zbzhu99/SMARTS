# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import json
import os
import re
import sys

# Set environment to better support Ray
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import glob, pickle, yaml
import time
from pydoc import locate
import gym
import numpy as np
import psutil
import ray
import torch
import yaml
import multiprocessing
from multiprocessing import Manager, Process
from smarts.zoo.registry import make
from ultra.ultra.utils.episode import LogInfo, episodes
from ultra.ultra.utils.ray import default_ray_kwargs

num_gpus = 1 if torch.cuda.is_available() else 0

# Number of GPUs should be splited between remote functions.
# @ray.remote(num_gpus=num_gpus / 2)
class Evaluation:
    def __init__(self, scenario_info, max_episode_steps, eval_rate, num_episodes, timestep_sec, headless):
        self.max_episode_steps = max_episode_steps
        self.eval_rate = eval_rate
        self.timestep_sec = timestep_sec
        self.headless = headless
        self.scenario_info = scenario_info
        self.num_episodes = num_episodes
        manager = multiprocessing.Manager()
        self.results = manager.dict()
        self.jobs = {}
        self.job_id = 0
        self.current_job_id = 0
        # multiprocessing.set_start_method('spawn')

    def check(
        self,
        # agent,
        episode,
        agent_id,
        policy_class,
        log_dir,
        experiment_dir,
        save_info,
    ):
        agent_itr = episode.get_itr(agent_id)
        print('eval check')
        if (agent_itr + 1) % self.eval_rate == 0 and episode.last_eval_iteration != agent_itr:
            print(
                f"Agent iteration : {agent_itr}, Eval rate : {self.eval_rate}, last_eval_iter : {episode.last_eval_iteration}"
            )
            checkpoint_dir = episode.checkpoint_dir(agent_itr)
            # agent.save(checkpoint_dir)
            for name, data in save_info.items():
                if name=='params.yaml':
                    with open(f'{checkpoint_dir}/{name}', "w") as file:
                        yaml.dump(data, file)
                else:
                    torch.save(pickle.loads(data), f'{checkpoint_dir}/{name}')
            episode.eval_mode()
            # episode.info[episode.active_tag][agent_id] = ray.get(
            #     [
            #
            #     ]
            # )[0]
            eval_proc = multiprocessing.Process(
                target = self.run,
                args=(
                experiment_dir,
                episode.eval_count,
                agent_id,
                policy_class,
                agent_itr,
                checkpoint_dir,
                log_dir,
                episode)
            )
            self.jobs[self.job_id] = (eval_proc, False)
            eval_proc.start()
            self.job_id+=1

            # print(x)
            # episode.eval_count += 1
            episode.last_eval_iteration = agent_itr
            # episode.record_tensorboard()
            episode.train_mode()
            # print(episode.eval_count)
            #

        # initiating the first process
        print(self.current_job_id, len(self.jobs), self.results)
        if self.jobs:
            if self.current_job_id == 0 and not self.jobs[self.current_job_id]:
                self.jobs[self.current_job_id][0].join()
                self.jobs[self.current_job_id][1] = True
                print(' job 0 started')
            # print(self.results)
        if self.current_job_id in self.results:
            episode.info[episode.active_tag][agent_id] = self.results[self.current_job_id]
            episode.record_tensorboard()
            episode.eval_count+=1
            self.current_job_id+=1
            if self.current_job_id in self.job and not self.jobs[self.current_job_id]:
                self.jobs[self.current_job_id].join()
                self.jobs[self.current_job_id][1] = True


    def run(
        self,
        experiment_dir,
        seed,
        agent_id,
        policy_class,
        itr_count,
        checkpoint_dir,
        log_dir,
        episode
    ):

        torch.set_num_threads(1)
        spec = make(
            locator=policy_class,
            checkpoint_dir=checkpoint_dir,
            experiment_dir=experiment_dir,
            max_episode_steps=self.max_episode_steps,
        )

        env = gym.make(
            "ultra.ultra.env:ultra-v0",
            agent_specs={agent_id: spec},
            scenario_info=self.scenario_info,
            headless=self.headless,
            timestep_sec=self.timestep_sec,
            seed=seed,
            eval_mode=True,
        )

        agent = spec.build_agent()
        summary_log = LogInfo()
        logs = []

        for eval_episode in episodes(self.num_episodes, etag=policy_class, log_dir=log_dir):
            observations = env.reset()
            state = observations[agent_id]
            dones, infos = {"__all__": False}, None

            eval_episode.reset(mode="Evaluation")
            while not dones["__all__"]:
                action = agent.act(state, explore=False)
                observations, rewards, dones, infos = env.step({agent_id: action})

                next_state = observations[agent_id]

                state = next_state

                eval_episode.record_step(agent_id=agent_id, infos=infos, rewards=rewards)

            eval_episode.record_episode()
            logs.append(eval_episode.info[eval_episode.active_tag][agent_id].data)

            for key, value in episode.info[eval_episode.active_tag][agent_id].data.items():
                if not isinstance(value, (list, tuple, np.ndarray)):
                    summary_log.data[key] += value

        for key, val in summary_log.data.items():
            if not isinstance(val, (list, tuple, np.ndarray)):
                summary_log.data[key] /= self.num_episodes

        env.close()
        self.results[job_id] = summary_log
        # episode.eval_count += 1
        # episode.record_tensorboard()
        # return summary_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser("intersection-single-agent-evaluation")
    parser.add_argument(
        "--task", help="Tasks available : [0, 1, 2]", type=str, default="1"
    )
    parser.add_argument(
        "--level",
        help="Levels available : [easy, medium, hard, no-traffic]",
        type=str,
        default="easy",
    )
    parser.add_argument("--models", default="models/", help="Directory to saved models")
    parser.add_argument(
        "--episodes", help="Number of training episodes", type=int, default=200
    )
    parser.add_argument(
        "--max-episode-steps",
        help="Maximum number of steps per episode",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--timestep", help="Environment timestep (sec)", type=float, default=0.1
    )
    parser.add_argument(
        "--headless", help="Run without envision", type=bool, default=False
    )
    parser.add_argument(
        "--experiment-dir",
        help="Path to spec file that includes adapters and policy parameters",
        type=str,
    )
    parser.add_argument(
        "--log-dir",
        help="Log directory location",
        default="logs",
        type=str,
    )
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    m = re.search(
        "ultra(.)*([a-zA-Z0-9_]*.)+([a-zA-Z0-9_])+:[a-zA-Z0-9_]+((-)*[a-zA-Z0-9_]*)*",
        args.models,
    )

    try:
        policy_class = m.group(0)
    except AttributeError as e:
        # default policy class
        policy_class = "ultra.baselines.sac:sac-v0"

    if not os.path.exists(args.models):
        raise "Path to model is invalid"

    if not os.listdir(args.models):
        raise "No models to evaluate"

    sorted_models = sorted(
        glob.glob(f"{args.models}/*"), key=lambda x: int(x.split("/")[-1])
    )
    base_dir = os.path.dirname(__file__)
    pool_path = os.path.join(base_dir, "agent_pool.json")

    ray.init()
    try:
        AGENT_ID = "AGENT_008"
        for episode in episodes(
            len(sorted_models),
            etag=policy_class,
            log_dir=args.log_dir,
        ):
            model = sorted_models[episode.index]
            print("model: ", model)
            episode_count = model.split("/")[-1]
            episode.eval_mode()
            episode.info[episode.active_tag][AGENT_ID] = ray.get(
                [
                    evaluate.remote(
                        experiment_dir=args.experiment_dir,
                        agent_id=AGENT_ID,
                        policy_class=policy_class,
                        seed=episode.eval_count,
                        itr_count=0,
                        checkpoint_dir=model,
                        scenario_info=(args.task, args.level),
                        num_episodes=int(args.episodes),
                        max_episode_steps=int(args.max_episode_steps),
                        timestep_sec=float(args.timestep),
                        headless=args.headless,
                        log_dir=args.log_dir,
                    )
                ]
            )[0]
            episode.record_tensorboard()
            episode.eval_count += 1
    finally:
        time.sleep(1)
        ray.shutdown()
