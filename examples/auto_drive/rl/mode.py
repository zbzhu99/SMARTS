from enum import Enum


class Mode(str, Enum):
    EVALUATE = "evaluate"
    TRAIN = "train"
