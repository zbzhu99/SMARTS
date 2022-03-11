from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

west_east_flow = [t.Flow(
            route=t.Route(
                begin=("edge-west-WE", lane, "random"), end=("edge-east-WE", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(3)]

east_west_flow = [t.Flow(
            route=t.Route(
                begin=("edge-east-EW", lane, "random"), end=("edge-west-EW", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(3)]

traffic = t.Traffic(
    flows=west_east_flow + east_west_flow
)

agent_prefabs = "scenarios.intersections.4lane_t.agent_prefabs"

ego_missions = [
    t.EndlessMission(
        begin=("edge-south-SN", 1, 20),
    )
]


gen_scenario(
    scenario=t.Scenario(
        traffic={"basic": traffic},
        ego_missions=ego_missions,
    ),
    output_dir=Path(__file__).parent,
)
