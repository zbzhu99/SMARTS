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

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Union

import tableprint as tp


class EpisodeLogs:
    def __init__(self, col_width, total_episodes: Union[str, int] = "?") -> None:
        self._col_width = col_width
        self._table = self.context(col_width)
        self._current_episode = None
        self._total_episodes = total_episodes
        self._current_episode_num = 0

    def reset(self):
        e = self._current_episode
        if e:
            self._write_row()
            self._current_episode_num += 1
        self._current_episode = EpisodeLog(self._current_episode_num)
        return self._current_episode

    def _write_row(self):
        e = self._current_episode
        row = (
            f"{e.index}/{self._total_episodes}",
            f"{e.sim2wall_ratio:.2f}",
            e.steps,
            f"{e.steps_per_second:.2f}",
            e.scenario_map[: self._col_width],
            e.scenario_routes[: self._col_width],
            e.mission_hash[: self._col_width],
        )

        score_summaries = [
            f"{score:.2f} - {agent}" for agent, score in e.scores.items()
        ]

        if len(score_summaries) == 0:
            self._table(row + ("",))
        else:
            self._table(row + (score_summaries[0],))
            if len(score_summaries) > 1:
                for s in score_summaries[1:]:
                    self._table(("", "", "", "", "", "", "", s))

    def __enter__(self):
        self._table.__enter__()
        return self

    def __exit__(self, *exc):
        self._table.__exit__(*exc)

    @staticmethod
    def context(col_width):
        return tp.TableContext(
            [
                "Episode",
                "Sim T / Wall T",
                "Total Steps",
                "Steps / Sec",
                "Scenario Map",
                "Scenario Routes",
                "Mission (Hash)",
                "Scores",
            ],
            width=col_width,
            style="round",
        )


@dataclass
class EpisodeLog:
    """An episode logging tool."""

    index: int = 0
    start_time: float = field(default_factory=lambda: time.time())
    fixed_timestep_sec: float = 0
    scores: dict = field(default_factory=lambda: defaultdict(lambda: 0))
    steps: int = 0
    scenario_map: str = ""
    scenario_routes: str = ""
    mission_hash: str = ""

    @property
    def wall_time(self):
        """Time elapsed since instantiation."""
        return time.time() - self.start_time

    @property
    def sim_time(self):
        """An estimation of the total fixed-timestep simulation performed."""
        return self.fixed_timestep_sec * self.steps

    @property
    def sim2wall_ratio(self):
        """The ration of sim time to wall time. Above 1 is hyper-realtime."""
        return self.sim_time / self.wall_time

    @property
    def steps_per_second(self):
        """The rate of steps performed since instantiation."""
        return self.steps / self.wall_time

    def record_scenario(self, scenario_log):
        """Record a scenario end."""
        self.fixed_timestep_sec = scenario_log["fixed_timestep_sec"]
        self.scenario_map = scenario_log["scenario_map"]
        self.scenario_routes = scenario_log["scenario_routes"]
        self.mission_hash = scenario_log["mission_hash"]

    def record_step(self, observations=None, rewards=None, dones=None, infos=None):
        """Record a step end."""
        self.steps += 1

        if not isinstance(observations, dict):
            observations, rewards, dones, infos = self._convert_to_dict(
                observations, rewards, dones, infos
            )

        if dones.get("__all__", False) and infos is not None:
            for agent, score in infos.items():
                self.scores[agent] = score["score"]

    def _convert_to_dict(self, observations, rewards, dones, infos):
        observations, rewards, infos = [
            {"SingleAgent": obj} for obj in [observations, rewards, infos]
        ]
        dones = {"SingleAgent": dones, "__all__": dones}
        return observations, rewards, dones, infos


def episodes(n):
    """An iteration method that provides numbered episodes.
    Acts similar to python's `range(n)` but yielding episode loggers.
    """
    col_width = 18
    with EpisodeLogs(col_width, n) as episode_logs:
        for _ in range(n):
            yield episode_logs.reset()
        episode_logs.reset()


@dataclass
class Episodes:
    current_step: int = 0

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        pass


class Episode:
    """An episode recording object"""

    def __init__(self, episodes: Episodes) -> None:
        self._episodes = episodes

    def register_step(self, observation, reward, done, info):
        self._episodes.current_step += 1
        return not done


def episode_range(max_steps):
    with Episodes(0) as episodes:
        while episodes.current_step < max_steps:
            yield Episode(episodes=episodes)
