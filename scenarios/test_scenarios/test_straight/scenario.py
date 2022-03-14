from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("west", lane_idx, "random"),
                end=("east", lane_idx, "max"),
            ),
            rate=3,
            actors={t.TrafficActor("car"): 1},
        )
        for lane_idx in range(3)
    for _ in range(3)
    ]
)

ego_missions = [t.Mission(t.Route(begin=("west", 1, 5), end=("east", 1, "max")), start_time=0.2)]

scenario = t.Scenario(
    ego_missions=ego_missions,
    traffic={"all": traffic},
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))
