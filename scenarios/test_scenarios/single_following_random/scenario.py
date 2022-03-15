from pathlib import Path

import random
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t


begin_edges = ["-E4", "E15", "E16", "E7", "-E8", "E9", "E10", "-E11",'-E14','E12']
end_edges = ["E4", "-E15", "-E16", "-E7", "E8", "-E9", "-E10", "E11",'E14','-E12']

flow = [
    t.Flow(
        route=t.Route(
            begin=(random.choice(begin_edges), random.randint(0, 1), "random"),
            end=(random.choice(end_edges), random.randint(0, 1), "max"),
        ),
        rate=1,
        actors={t.TrafficActor("car"): 1},
    )
    for _ in range(15)
]

flow_lead = [
    t.Flow(
        route=t.Route(
            begin=("E0", 1, 30),
            end=("E3", 1, "max"),
            via=("E1", "E6", "E13", "-E5", "E1", "E2"),
        ), 
        rate=1,
        actors={t.TrafficActor(name='car',vehicle_type='coach'): 1},
    )
]

flow_follow = [
    t.Flow(
        route=t.Route(
            begin=("E0", 1, 0),
            end=("E3", 1, "max"),
            via=("E1", "E6", "E13", "-E5", "E1", "E2"),
        ), 
        rate=1,
        actors={t.TrafficActor(name='car',vehicle_type='coach'): 1},
    )
]

traffic = t.Traffic(flows=flow+flow_lead+flow_follow)

gen_scenario(
    scenario=t.Scenario(
        traffic={"basic": traffic},
    ),
    output_dir=Path(__file__).parent,
)
