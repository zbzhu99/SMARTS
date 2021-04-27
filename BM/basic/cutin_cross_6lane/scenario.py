import logging
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

from sys import path

path.append(str(Path(__file__).parent.parent))
from copy_scenario import copy_to_dir

scenario_map_file = "scenarios/6lane_cross"

logger = logging.getLogger(str(Path(__file__)))
s_dir = str(Path(__file__).parent)

try:
    copy_to_dir(scenario_map_file, s_dir)
except Exception as e:
    logger.error(f"Scenario {scenario_map_file} failed to copy")
    raise e

ego_missions = [
    t.Mission(
        t.Route(begin=("east_ew", 1, 20), end=("west_ew", 2, "max")),
        # via=(t.Via(t.JunctionEdgeIDResolver("east_ew", 1, "west_ew", 2), 2, 3, 30),)
    )
]

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(begin=("east_ew", 2, 30), end=("west_ew", 2, "max"),),
            rate=1,
            actors={t.TrafficActor("car"): 1},
        )
    ]
)

scenario = t.Scenario(traffic={"all": traffic}, ego_missions=ego_missions,)

gen_scenario(scenario, output_dir=s_dir)
