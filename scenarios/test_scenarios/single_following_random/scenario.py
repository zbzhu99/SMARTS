from pathlib import Path

import random
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t


begin_edges = ["-E4", "-E5", "-E6", "E7", "-E8", "E9", "E10", "-E11"]
end_edges = ["E4", "E5", "E6", "-E7", "E8", "-E9", "-E10", "E11"]

flow = [t.Flow(
            route = t.Route(
                begin = (random.choice(begin_edges), random.randint(0,1), "random"),
                end = (random.choice(end_edges), random.randint(0,1), "max"),
            ),
            rate = 1,
            actors={t.TrafficActor("car"): 1},
        ) 
        for _ in range(20)]

traffic = t.Traffic(
    # flows=flow1+flow2+flow3+flow4+flow5+flow6+flow7+flow8
    flows=flow
)

gen_scenario(
    scenario=t.Scenario(
        traffic={"basic": traffic},
    ),
    output_dir=Path(__file__).parent,
)
