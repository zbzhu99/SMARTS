import logging
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

from sys import path

path.append(str(Path(__file__).parent.parent))
from copy_scenario import copy_to_dir

scenario_map_file = "scenarios/straightaway_3e_1w"

logger = logging.getLogger(str(Path(__file__)))

s_dir = str(Path(__file__).parent)

try:
    copy_to_dir(scenario_map_file, s_dir)
except Exception as e:
    logger.error(f"Scenario {scenario_map_file} failed to copy")
    raise e

ego_missions = [
    t.EndlessMission(
        begin=("-straightaway", 0, 200),
    )
]

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("straightaway", 1, 30),
                end=("straightaway", 1, "max"),
            ),
            rate=1,
            actors={t.TrafficActor("target"): 1},
        )
    ]
)

scenario = t.Scenario(
    traffic={"all": traffic},
    ego_missions=ego_missions,
)

gen_scenario(scenario, output_dir=s_dir)
