from pathlib import Path
from random import randrange

from smarts.core import seed
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

seed(4)
# start_offset = 0
# route1 = t.Route(begin=('-E5',0,start_offset), end=('-E10',0,'max'))

rand_routes = [t.RandomRoute() for _ in range(15)]

# SECS_PER_HOUR = 60.0 * 60.0 
# secs_btw_adding = 40
flow_prob = 1/14
actor = t.TrafficActor(name="car",vehicle_type='passenger')

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=route,
            actors={actor:flow_prob},
            rate=100,
            begin=0
        )
        for route in rand_routes
    ]
)

gen_scenario(
    t.Scenario(
        traffic={'basic':traffic},
    ),
    output_dir=Path(__file__).parent,
)