from pathlib import Path

import random
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

flow1 = [t.Flow(
            route=t.Route(
                begin=("-E5", lane, "random"), end=("-E10", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(2)]

flow2 = [t.Flow(
            route=t.Route(
                begin=("-E6", lane, "random"), end=("E11", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(1)]

flow3 = [t.Flow(
            route=t.Route(
                begin=("-E11", lane, "random"), end=("E6", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(1)]

flow4 = [t.Flow(
            route=t.Route(
                begin=("E9", lane, "random"), end=("E4", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(1)]

flow5 = [t.Flow(
            route=t.Route(
                begin=("-E4", lane, "random"), end=("-E9", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(1)]

flow6 = [t.Flow(
            route=t.Route(
                begin=("E10", lane, "random"), end=("E5", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(1)]

flow7 = [t.Flow(
            route=t.Route(
                begin=("-E8", lane, "random"), end=("-E7", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(1)]

flow8 = [t.Flow(
            route=t.Route(
                begin=("E7", lane, "random"), end=("E8", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(1)]


begin_edges = ["-E4", "-E5", "-E6", "E7", "-E8", "E9", "E10", "-E11"]
end_edges = ["E4", "E5", "E6", "-E7", "E8", "-E9", "-E10", "E11"]

flow = [t.Flow(
            route = t.Route(
                begin = (begin_edges[random.randint(0, len(begin_edges) - 1)], random.randint(0,1), "random"),
                end = (end_edges[random.randint(0, len(end_edges) - 1)], random.randint(0,1), "max"),
            ),
            rate = 1,
            actors={t.TrafficActor("car"): 1},
        ) 
        for _ in range(30)]

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
