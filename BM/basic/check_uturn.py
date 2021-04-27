import math
from typing import Dict

from smarts.core.utils.math import vec_to_radians
from checker_utils import get_lane_vector_at_offset, get_offset_into_lane, is_further_along_route
from smarts.core.smarts import SMARTS
from smarts.core.sensors import EgoVehicleObservation, Observation, VehicleObservation
from checker import Checker, CheckerFrameResult, Result


class UTurnChecker(Checker):
    @staticmethod
    def _get_index(l: list, index, default=None):
        try:
            return l[index]
        except IndexError:
            return default

    def __init__(self, bm_id) -> None:
        super().__init__(bm_id)

        self._uturn_started = False
        self._cut_in_front = False

    def evaluate(
        self, sim: SMARTS, observations: Dict[str, Observation], rewards, dones, infos
    ) -> CheckerFrameResult:
        done = dones.get(self._bm_id, None)
        if done:
            return CheckerFrameResult("Done before performed", Result.FAIL)
        elif done is None:
            return Checker("Vehicle is no longer active", Result.INVALID)

        observation = observations.get(self._bm_id, None)
        if observation is None:
            return CheckerFrameResult(f"Vehicle `{self._bm_id}` not found", Result.TBD)

        near: VehicleObservation = self._get_index(
            observation.neighborhood_vehicle_states, 0
        )
        ego: EgoVehicleObservation = observation.ego_vehicle_state

        if near:
            rn = sim.road_network
            eoff = get_offset_into_lane(rn, ego.lane_id, ego.position[:2])

            if not self._cut_in_front:
                further = is_further_along_route(rn, ego.lane_id, ego.position, near.lane_id, near.position)
                if further:
                    self._cut_in_front = True
            else:
                e_d_of_lane = get_lane_vector_at_offset(rn, ego.lane_id, eoff)

                e_lane_angle = vec_to_radians(e_d_of_lane)
                e_h_relative_to_lane = ego.heading.relative_to(e_lane_angle)
                if abs(e_h_relative_to_lane) < math.pi / 50:
                    return CheckerFrameResult("Successful uturn")

        return CheckerFrameResult("Episode continues", Result.TBD)

    def timeout(self):
        return CheckerFrameResult("Episode ended without u-turn", Result.FAIL)
