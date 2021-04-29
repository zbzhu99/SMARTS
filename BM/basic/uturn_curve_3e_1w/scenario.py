import logging
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t
from smarts.sstudio.types import Mission, Route, UTurn

from sys import path

path.append(str(Path(__file__).parent.parent))
from copy_scenario import copy_to_dir

scenario_map_file = "scenarios/curve_ahead_3e_1w"

logger = logging.getLogger(str(Path(__file__)))

s_dir = str(Path(__file__).parent)

try:
    copy_to_dir(scenario_map_file, s_dir)
except Exception as e:
    logger.error(f"Scenario {scenario_map_file} failed to copy")
    raise e

ego_missions = [
    Mission(
        Route(begin=("-curve_ahead", 0, 30), end=("-curve_ahead", 0, "300")),
        task=UTurn(initial_speed=20),
    ),
]

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("curve_ahead", 1, 30),
                end=("curve_ahead", 1, "max"),
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
        )
    ]
)

scenario = t.Scenario(
    traffic={"all": traffic},
    ego_missions=ego_missions,
)

gen_scenario(scenario, output_dir=s_dir)
