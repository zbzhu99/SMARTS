from dataclasses import dataclass
from typing import Sequence
from enum import IntEnum


class Result(IntEnum):
    FAIL = 0
    PASS = 1
    TBD = 2
    INVALID = 3


@dataclass
class CheckerFrameResult:
    frame_log: str = None
    checker_success: Result = Result.TBD


class Checker:
    def __init__(self, bm_id) -> None:
        self._bm_id = bm_id

    def evaluate(self, sim, observations, rewards, dones, infos) -> CheckerFrameResult:
        raise NotImplementedError()

    def timeout(self):
        raise NotImplementedError()
