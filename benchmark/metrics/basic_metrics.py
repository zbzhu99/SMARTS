# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
"""
General metrics
"""

import numpy as np
import csv
import time
import os

from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass, field
from scipy.spatial import distance

from smarts.core.utils.episodes import EpisodeLog


def agent_info_adapter(obs, shaped_reward: float, raw_info: dict):
    info = dict()
    info["speed"] = obs.ego_vehicle_state.speed
    info["collision"] = 1 if len(obs.events.collisions) > 0 else 0

    goal = obs.ego_vehicle_state.mission.goal
    goal_pos = goal.position
    ego_2d_pos = obs.ego_vehicle_state.position[:2]

    info["distance_to_goal"] = distance.euclidean(ego_2d_pos, goal_pos)
    info["events"] = obs.events

    return info


def min_max_mean(data: list):
    return {"min": np.min(data), "max": np.max(data), "mean": np.mean(data)}


@dataclass
class EvaluatedEpisode(EpisodeLog):
    ego_speed: dict = field(default_factory=lambda: defaultdict(lambda: []))
    num_collision: dict = field(default_factory=lambda: defaultdict(lambda: 0))
    distance_to_goal: dict = field(default_factory=lambda: defaultdict(lambda: []))
    distance_to_ego_car: dict = field(default_factory=lambda: defaultdict(lambda: []))
    acceleration: dict = field(default_factory=lambda: defaultdict(lambda: 0.0))
    reach_goal: dict = field(default_factory=lambda: defaultdict(lambda: False))
    agent_step: dict = field(default_factory=lambda: defaultdict(lambda: 0))

    def record_step(self, observations=None, rewards=None, dones=None, infos=None):
        for agent_id, info in infos.items():
            if info.get("_group_info") is not None:
                for i, _info in enumerate(info["_group_info"]):
                    name = f"{agent_id}:AGENT-{i}"
                    self.ego_speed[name] = np.mean(_info["speed"])

                    self.num_collision[name] += len(_info["events"].collisions)

                    if dones[agent_id]:
                        self.reach_goal[name] = _info["events"].reached_goal
                        self.distance_to_goal[name] = _info["distance_to_goal"]
            else:
                self.ego_speed[agent_id].append(info["speed"])
                self.num_collision[agent_id] += len(info["events"].collisions)
                self.distance_to_goal[agent_id].append(info["distance_to_goal"])
                self.agent_step[agent_id] += 1

        self.steps += 1


MinMeanMax = namedtuple("MinMeanMax", "min, mean, max")


def get_statistics(data: list):
    return MinMeanMax(np.min(data), np.mean(data), np.max(data))


class MetricKeys:
    AVE_CR = "Average Collision Rate"
    AVE_COMR = "Completion Rate"
    MAX_L = "Max Live Step"
    MIN_L = "Min Live Step"
    MEAN_L = "Mean Live Step"
    MIN_G = "Min Goal Distance"


class MetricHandler:
    """ MetricHandler serves for the metric """

    def __init__(self, num_episode):
        """Create a MetricHandler instance to record the

        Parameters
        ----------
        num_episode
            int, the num of e
        """
        self._logs = [EvaluatedEpisode() for _ in range(num_episode)]

    def log_step(self, observations, rewards, dones, infos, episode):
        self._logs[episode].record_step(observations, rewards, dones, infos)

    def write_to_csv(self, csv_dir):
        csv_dir = f"{csv_dir}/{int(time.time())}"
        for i, logger in enumerate(self._logs):
            sub_dir = f"{csv_dir}/episode_{i}"
            os.makedirs(sub_dir)
            for agent_id in logger.agent_step.keys():
                # get time step
                f_name = f"{sub_dir}/agent_{agent_id}.csv"
                with open(f_name, "w") as f:
                    writer = csv.writer(f, delimiter=",")
                    headers = [""] + [
                        str(i) for i in range(logger.agent_step[agent_id])
                    ]
                    writer.writerow(headers)
                    writer.writerow(["Speed"] + logger.ego_speed[agent_id])
                    writer.writerow(["GDistance"] + logger.distance_to_goal[agent_id])
                    # writer.writerow(
                    #     ["EDistance"] + logger.distance_to_ego_car[agent_id]
                    # )
                    # writer.writerow(["Acceleration"] + logger.acceleration[agent_id])
                    writer.writerow(
                        ["Num_Collision"] + [logger.num_collision[agent_id]]
                    )

    def read_episode(self, csv_dir):
        agent_record = defaultdict(
            lambda: {
                "Speed": None,
                "GDistance": None,
                "EDistance": None,
                "Num_Collision": None,
                "Acceleration": None,
            }
        )
        for f_name in os.listdir(csv_dir):
            if f_name.endswith(".csv"):
                f_path = os.path.join(csv_dir, f_name)
                agent_id = f_path.split(".")[0]
                print(f"Got file `{f_name}` for agent-{agent_id}")
                with open(f_path,) as f:
                    reader = csv.reader(f, delimiter=",")
                    _ = next(reader)
                    agent_record[agent_id]["Speed"] = next(reader)[1:]
                    agent_record[agent_id]["GDistance"] = next(reader)[1:]
                    # agent_record[agent_id]["EDistance"] = next(reader)[1:]
                    # agent_record[agent_id]["Acceleration"] = next(reader)[1:]
                    agent_record[agent_id]["Num_Collision"] = next(reader)
        return agent_record

    def compute(self, csv_dir):
        # list directory
        sub_dirs = [os.path.join(csv_dir, sub_dir) for sub_dir in os.listdir(csv_dir)]
        agent_metrics = defaultdict(
            lambda: {
                MetricKeys.AVE_CR: 0.0,
                MetricKeys.AVE_COMR: 0.0,
                MetricKeys.MAX_L: 0,
                MetricKeys.MIN_L: 0,
                MetricKeys.MEAN_L: 0.0,
                MetricKeys.MIN_G: 0.0,
            }
        )

        goal_dist_th = 2.0

        for sub_dir in sub_dirs:
            episode_agent_record: dict = self.read_episode(sub_dir)
            for aid, record in episode_agent_record.items():
                am = agent_metrics[aid]
                am[MetricKeys.AVE_CR] += record["Num_Collision"]
                min_goal_dist = record["GDistance"][-1]
                am[MetricKeys.AVE_COMR] += 1.0 if min_goal_dist < goal_dist_th else 0.0
                am[MetricKeys.MAX_L] = max(am[MetricKeys.MAX_L], len(record["Speed"]))
                am[MetricKeys.MIN_L] = min(am[MetricKeys.MIN_L], len(record["Speed"]))
                am[MetricKeys.MEAN_L] += len(record["Speed"])
                am[MetricKeys.MIN_G] = min(am[MetricKeys.MIN_G], min_goal_dist)

        for aid, record in agent_metrics.items():
            record[MetricKeys.MEAN_L] /= len(sub_dirs)
            record[MetricKeys.AVE_COMR] /= len(sub_dirs)
            record[MetricKeys.AVE_CR] /= len(sub_dirs)

        print(agent_metrics)
