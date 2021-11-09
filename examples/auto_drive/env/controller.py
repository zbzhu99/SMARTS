from enum import Enum

class Controller(str, Enum):
    CONTINUOUS='continuous'
    LANE='lane'
    DISCRETE='discrete'