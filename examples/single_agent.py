import logging
import math
import numpy as np
import gym

from examples import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.sensors import Observation
from smarts.core.utils.math import evaluate_bezier as bezier
from smarts.core.utils.episodes import episodes
from smarts.core.coordinates import Heading, Pose
from smarts.core.waypoints import Waypoint, Waypoints
from smarts.core.utils.math import (
    lerp,
    low_pass_filter,
    min_angles_difference_signed,
    radians_to_vec,
    vec_to_radians,
    signed_dist_to_line,
)

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


class UTurnAgent(Agent):
    def __init__(self):
        self.lane_index = 1
        self._task_is_triggered = False

    def act(self, obs: Observation):
        from smarts.core.smarts import SMARTS
        from smarts.core.vehicle import Vehicle

        sim: SMARTS = self.sim
        aggressiveness = 10
        print(sim._vehicle_index.agent_vehicle_ids(), "OOOOO")
        # input("kkkk")
        # print(obs.ego_vehicle_state.linear_velocity[1])
        vehicles = sim._vehicle_index.vehicles_by_actor_id(AGENT_ID)

        vehicle: Vehicle = vehicles[0]

        miss = sim._vehicle_index.sensor_state_for_vehicle_id(
            vehicle.id
        ).mission_planner

        neighborhood_vehicles = sim.neighborhood_vehicles_around_vehicle(
            vehicle=vehicle, radius=850
        )
        pose = vehicle.pose

        position = pose.position[:2]
        lane = sim.scenario.road_network.nearest_lane(position)

        # if not neighborhood_vehicles or self.sim.elapsed_sim_time < 1:
        #     return (0,0,0)

        # target_vehicle = neighborhood_vehicles[0]
        # target_position = target_vehicle.pose.position[:2]

        # # if (self._prev_kyber_x_position is None) and (
        # #     self._prev_kyber_y_position is None
        # # ):
        # #     self._prev_kyber_x_position = target_position[0]
        # #     self._prev_kyber_y_position = target_position[1]

        # # velocity_vector = np.array(
        # #     [
        # #         (-self._prev_kyber_x_position + target_position[0]) / self.sim.timestep_sec,
        # #         (-self._prev_kyber_y_position + target_position[1]) / self.sim.timestep_sec,
        # #     ]
        # # )
        # target_velocity = target_vehicle.speed

        # # self._prev_kyber_x_position = target_position[0]
        # # self._prev_kyber_y_position = target_position[1]

        # target_lane = self.sim.scenario.road_network.nearest_lane(target_position)

        # offset = self.sim.scenario.road_network.offset_into_lane(lane, position)
        # target_offset = self.sim.scenario.road_network.offset_into_lane(
        #     target_lane, target_position
        # )

        # # cut-in offset should consider the aggressiveness and the speed
        # # of the other vehicle.

        # cut_in_offset = np.clip(15 - 0.5*aggressiveness, 10, 15)

        # if (
        #     abs(offset - (cut_in_offset + target_offset)) > 1
        #     and lane.getID() != target_lane.getID()
        #     and self._task_is_triggered is False
        # ):
        #     nei_wps = miss._waypoints.waypoint_paths_on_lane_at(
        #         position, lane.getID(), 60
        #     )
        #     speed_limit = np.clip(
        #         np.clip(
        #             (target_velocity * 1.1)
        #             - 6 * (offset - (cut_in_offset + target_offset)),
        #             0.5 * target_velocity,
        #             1.5 * target_velocity,
        #         ),
        #         0.5,
        #         30,
        #     )
        # else:
        #     self._task_is_triggered = True
        #     nei_wps = miss._waypoints.waypoint_paths_on_lane_at(
        #         position, target_lane.getID(), 60
        #     )
        #     self.lane_index=0

        #     cut_in_speed = neighborhood_vehicles[1].speed * 1.5

        #     speed_limit = cut_in_speed

        #     # 1.5 m/s is the threshold for speed offset. If the vehicle speed
        #     # is less than target_velocity plus this offset then it will not
        #     # perform the cut-in task and instead the speed of the vehicle is
        #     # increased.
        #     # if vehicle.speed < target_velocity + 1.5:
        #     #     nei_wps = miss._waypoints.waypoint_paths_on_lane_at(
        #     #         position, lane.getID(), 60
        #     #     )
        #     #     speed_limit = np.clip(target_velocity * 2.1, 0.5, 30)
        #     #     self._task_is_triggered = False
        #     #     self.lane_index=1

        # p0 = position
        # p_temp = nei_wps[0][len(nei_wps[0]) // 3].pos
        # p1 = p_temp
        # p2 = nei_wps[0][2 * len(nei_wps[0]) // 3].pos

        # p3 = nei_wps[0][-1].pos
        # p_x, p_y = bezier([p0, p1, p2, p3], 20)
        # trajectory = []
        # prev = position[:2]
        # for i in range(len(p_x)):
        #     pos = np.array([p_x[i], p_y[i]])
        #     heading = Heading(vec_to_radians(pos - prev))
        #     prev = pos
        #     lane = self.sim.scenario.road_network.nearest_lane(pos)
        #     if lane is None:
        #         continue
        #     lane_id = lane.getID()
        #     lane_index = lane_id.split("_")[-1]
        #     width = lane.getWidth()

        #     wp = Waypoint(
        #         pos=pos,
        #         heading=heading,
        #         lane_width=width,
        #         speed_limit=speed_limit,
        #         lane_id=lane_id,
        #         lane_index=lane_index,
        #     )
        #     trajectory.append(wp)
        # # return [trajectory]
        # print(self.lane_index)

        # if (
        #     len(obs.via_data.near_via_points) < 1
        #     or obs.ego_vehicle_state.edge_id != obs.via_data.near_via_points[0].edge_id
        # ):
        #     return (obs.waypoint_paths[0][0].speed_limit, 0)

        # nearest = obs.via_data.near_via_points[0]
        # if nearest.lane_index == obs.ego_vehicle_state.lane_index:
        #     return (nearest.required_speed, 0)

        # return (
        #     nearest.required_speed,
        #     1 if nearest.lane_index > obs.ego_vehicle_state.lane_index else -1,
        # )
        # print(obs)
        # lane_index = 0
        # num_trajectory_points = min([10, len(obs.waypoint_paths[lane_index])])
        # # Desired speed is in m/s
        # desired_speed = 50 / 3.6
        # trajectory = [
        #     [
        #         obs.waypoint_paths[lane_index][i].pos[0]
        #         for i in range(num_trajectory_points)
        #     ],
        #     [
        #         obs.waypoint_paths[lane_index][i].pos[1]
        #         for i in range(num_trajectory_points)
        #     ],
        #     [
        #         obs.waypoint_paths[lane_index][i].heading
        #         for i in range(num_trajectory_points)
        #     ],
        #     [desired_speed for i in range(num_trajectory_points)],
        # ]
        # print(np.linalg.norm(obs.neighborhood_vehicle_states[0].position-obs.ego_vehicle_state.position),"DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
        # distant=np.linalg.norm(obs.neighborhood_vehicle_states[0].position-obs.ego_vehicle_state.position)
        # relative=obs.neighborhood_vehicle_states[0].position-obs.ego_vehicle_state.position
        # relative1=obs.neighborhood_vehicle_states[1].position-obs.ego_vehicle_state.position
        # # print(np.dot(radians_to_vec(obs.ego_vehicle_state.heading),relative[0:2]),"VVVVVVVVVVVVVVV")
        # error=np.dot(radians_to_vec(obs.ego_vehicle_state.heading),relative[0:2])
        # error_lat=1*np.cross(radians_to_vec(obs.ego_vehicle_state.heading),relative1[0:2])
        # print("lat_error: ", error_lat)
        # # self.lane_index=0
        # # print(self.sim,"||||||||||||||||||||||||||||||||")

        # # if abs(distant-25)<2:
        # #     self.lane_index=0
        # aaa=np.dot(obs.neighborhood_vehicle_states[1].position[0:2]-obs.ego_vehicle_state.position[0:2],radians_to_vec(obs.ego_vehicle_state.heading))
        sw = np.linalg.norm(
            obs.neighborhood_vehicle_states[0].position[0:2]
            - obs.ego_vehicle_state.position[0:2]
        )
        # aaa=1000/aaa
        # error_lat=np.sign(error_lat)*abs(aaa)
        # if sw>5:
        #     aaa=0
        #     error_lat=0
        error_lat = []
        aaa = []
        for ii in range(len(neighborhood_vehicles)):
            sw = np.linalg.norm(
                obs.neighborhood_vehicle_states[ii].position[0:2]
                - obs.ego_vehicle_state.position[0:2]
            )
            relative1 = (
                obs.neighborhood_vehicle_states[ii].position
                - obs.ego_vehicle_state.position
            )
            # error_lat=1*np.cross(radians_to_vec(obs.ego_vehicle_state.heading),relative1[0:2])
            if sw > 25:
                aaa.append(0)
                error_lat.append(0)

            else:
                aaa.append(
                    (1 / (np.linalg.norm(relative1)) ** 2)
                    * np.dot(
                        obs.neighborhood_vehicle_states[ii].position[0:2]
                        - obs.ego_vehicle_state.position[0:2],
                        radians_to_vec(obs.ego_vehicle_state.heading),
                    )
                )
                error_lat.append(
                    (1 / (np.linalg.norm(relative1)) ** 2)
                    * np.cross(
                        radians_to_vec(obs.ego_vehicle_state.heading), relative1[0:2]
                    )
                )

        # aaa1=np.dot(obs.neighborhood_vehicle_states[0].position[0:2]-obs.ego_vehicle_state.position[0:2],radians_to_vec(obs.ego_vehicle_state.heading))

        # sw1=np.linalg.norm(obs.neighborhood_vehicle_states[0].position[0:2]-obs.ego_vehicle_state.position[0:2])
        # aaa1=1000/aaa1
        # if sw1>5:
        #     aaa1=0

        # print(aaa,"::::::::::::",aaa1)
        start_lane = miss._road_network.nearest_lane(
            miss._mission.start.position,
            include_junctions=False,
            include_special=False,
        )
        start_edge = miss._road_network.road_edge_data_for_lane_id(start_lane.getID())
        oncoming_edge = start_edge.oncoming_edges[0]
        oncoming_lanes = oncoming_edge.getLanes()
        target_lane_index = miss._mission.task.target_lane_index
        target_lane_index = min(target_lane_index, len(oncoming_lanes) - 1)
        target_lane = oncoming_lanes[target_lane_index + 0]

        offset = miss._road_network.offset_into_lane(start_lane, pose.position[:2])
        oncoming_offset = max(0, target_lane.getLength() - offset)
        target_p = neighborhood_vehicles[0].pose.position[0:2]
        target_l = miss._road_network.nearest_lane(target_p)
        target_offset = miss._road_network.offset_into_lane(target_l, target_p)
        fq = target_lane.getLength() - offset - target_offset

        paths = miss.paths_of_lane_at(target_lane, oncoming_offset, lookahead=30)

        self.lane_index = 0
        fff = obs.waypoint_paths[self.lane_index]
        if sw < (aggressiveness / 10) * 30 + (1 - aggressiveness / 10) * 100:
            self._task_is_triggered = True
            fff = paths[0]

        target = paths[0][-1]
        print(fq, "[][][][][][][][][]")

        look_ahead_wp_num = 3
        look_ahead_dist = 3
        vehicle_look_ahead_pt = [
            obs.ego_vehicle_state.position[0]
            - look_ahead_dist * math.sin(obs.ego_vehicle_state.heading),
            obs.ego_vehicle_state.position[1]
            + look_ahead_dist * math.cos(obs.ego_vehicle_state.heading),
        ]
        # print(obs.waypoint_paths[lane_index][look_ahead_wp_num],"<<<<<<<<<<<<<<<,")
        reference_heading = fff[look_ahead_wp_num].heading
        heading_error = min_angles_difference_signed(
            (obs.ego_vehicle_state.heading % (2 * math.pi)), reference_heading
        )
        controller_lat_error = fff[look_ahead_wp_num].signed_lateral_error(
            vehicle_look_ahead_pt
        )
        trig = 1
        steer = (
            0.34 * controller_lat_error
            + 1.2 * heading_error
            - trig * 100 * sum(error_lat)
        )
        # print(trajectory[look_ahead_wp_num].speed_limit,"<<<<<<<<<<<<<<")
        throttle = (
            -0.23 * (obs.ego_vehicle_state.speed - 12)
            - 1.1 * abs(obs.ego_vehicle_state.linear_velocity[1])
            - trig * 100 * sum(aaa)
        )

        # if sw<5:
        #     steer=1
        #     print("FFFFFFFFFFFFFFFFFF")
        if throttle >= 0:
            brake = 0
        else:
            brake = abs(throttle)
            throttle = 0
        # if abs(obs.ego_vehicle_state.linear_velocity[1])>.4:
        #     brake=0.5
        #     throttle=0
        # print(obs.ego_vehicle_state.speed,"KKKKKKKKKKKKK")
        print(
            sum(aaa),
            sum(error_lat),
            relative1,
            "TTTTTHHHHRRRRROOOTTTTKLLLLE",
            (throttle, brake, steer),
        )
        return (throttle, brake, steer)


def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.StandardWithAbsoluteSteering, max_episode_steps=max_episode_steps
        ),
        agent_builder=UTurnAgent,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=False,
        sumo_auto_start=False,
        seed=seed,
        # zoo_addrs=[("10.193.241.236", 7432)], # Sample server address (ip, port), to distribute social agents in remote server.
        # envision_record_data_replay_path="./data_replay",
    )
    global vvv
    UTurnAgent.sim = env._smarts
    print(env._smarts, "::::::::::::::::::::::::::")

    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
