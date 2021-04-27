import logging
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

from sys import path

path.append(str(Path(__file__).parent.parent))
from copy_scenario import copy_to_dir

scenario_map_file = "scenarios/merge_2to1_lane"

logger = logging.getLogger(str(Path(__file__)))

s_dir = str(Path(__file__).parent)

try:
    copy_to_dir(scenario_map_file, s_dir)
except Exception as e:
    logger.error(f"Scenario {scenario_map_file} failed to copy")
    raise e

ego_missions = [
    t.Mission(
        route=t.Route(
            begin=("2lane_stretch_1", 0, 1), end=("1lane_stretch_1", 0, "max"),
        ),
    )
]


scenario = t.Scenario(ego_missions=ego_missions,)

gen_scenario(scenario, output_dir=s_dir)
