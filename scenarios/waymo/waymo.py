"""
Visualization and prototyping script for the Waymo motion dataset.
"""

import argparse
from dataclasses import dataclass
import math
import os
import queue
import time
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from waymo_open_dataset.protos import scenario_pb2
from smarts.core.utils.geometry import buffered_shape
from smarts.sstudio.genhistories import Waymo


def convert_polyline(polyline) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for p in polyline:
        xs.append(p.x)
        ys.append(p.y)
    return xs, ys


def get_map_features_for_scenario(scenario) -> Dict:
    map_features = {"lane": [], "road_line": [], "road_edge": [], "stop_sign": [], "crosswalk": [], "speed_bump": []}
    for i in range(len(scenario.map_features)):
        map_feature = scenario.map_features[i]
        key = map_feature.WhichOneof("feature_data")
        if key is not None:
            map_features[key].append(getattr(map_feature, key))

    return map_features


def get_map_features(path, scenario_id):
    scenario = None
    dataset = Waymo.read_dataset(path)
    for record in dataset:
        parsed_scenario = scenario_pb2.Scenario()
        parsed_scenario.ParseFromString(bytearray(record))
        if parsed_scenario.scenario_id == scenario_id:
            scenario = parsed_scenario
            break

    if scenario is None:
        errmsg = f"Dataset file does not contain scenario with id: {scenario_id}"
        raise ValueError(errmsg)

    features = {}
    lanes = []
    for i in range(len(scenario.map_features)):
        map_feature = scenario.map_features[i]
        key = map_feature.WhichOneof("feature_data")
        if key is not None:
            features[map_feature.id] = getattr(map_feature, key)
            if key == "lane":
                lanes.append((getattr(map_feature, key), map_feature.id))

    return scenario.scenario_id, features, lanes


def plot_map(map_features):
    lanes = map_features["lane"][:1]
    lane_points = [convert_polyline(lane.polyline) for lane in lanes]
    # lanes = list(filter(lambda lane: max(lane[1]) > 8150, lanes))
    for xs, ys in lane_points:
        plt.plot(xs, ys, linestyle=":", color="gray")
    for road_line in map_features["road_line"]:
        xs, ys = convert_polyline(road_line.polyline)
        if road_line.type in [1, 4, 5]:
            plt.plot(xs, ys, "y--")
        else:
            plt.plot(xs, ys, "y-")
    for road_edge in map_features["road_edge"]:
        xs, ys = convert_polyline(road_edge.polyline)
        plt.plot(xs, ys, "k-")
    # for crosswalk in map_features["crosswalk"]:
    #     xs, ys = convert_polyline(crosswalk.polygon)
    #     plt.plot(xs, ys, 'k--')
    # for speed_bump in map_features["speed_bump"]:
    #     xs, ys = convert_polyline(speed_bump.polygon)
    #     plt.plot(xs, ys, 'k:')
    for stop_sign in map_features["stop_sign"]:
        plt.scatter(
            stop_sign.position.x, stop_sign.position.y, marker="o", c="#ff0000", alpha=1
        )


def plot_lane(lane):
    xs, ys = convert_polyline(lane.polyline)
    plt.plot(xs, ys, linestyle="-", c="gray")
    # plt.scatter(xs, ys, s=12, c="gray")
    # plt.scatter(xs[0], ys[0], s=12, c="red")


def plot_road_line(road_line):
    xs, ys = convert_polyline(road_line.polyline)
    plt.plot(xs, ys, "y-")
    plt.scatter(xs, ys, s=12, c="y")
    # plt.scatter(xs[0], ys[0], s=12, c="red")


def plot_road_edge(road_edge):
    xs, ys = convert_polyline(road_edge.polyline)
    plt.plot(xs, ys, "k-")
    plt.scatter(xs, ys, s=12, c="black")
    # plt.scatter(xs[0], ys[0], s=12, c="red")


def line_intersect(a, b, c, d) -> Union[float, None]:
    r = b - a
    s = d - c
    d = r[0] * s[1] - r[1] * s[0]

    if d == 0:
        return None

    u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / d
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / d

    if 0 <= u <= 1 and 0 <= t <= 1:
        return a + t * r

    return None


def rotate(v, angle):
    return np.array([
        math.cos(angle) * v[0] - math.sin(angle) * v[1],
        math.sin(angle) * v[0] + math.cos(angle) * v[1]
    ])


def norm(x, y):
    return math.sqrt(x**2 + y**2)


def nearest_point(point, line):
    dist = np.square(line - point)
    dist = np.sqrt(dist[:, 0] + dist[:, 1])
    return np.argmin(dist)


@dataclass
class Lane:
    lane_id: int
    lane_pts: List
    n_pts: int
    left_pts: List
    right_pts: List
    normals: List
    left_neighbors: List['Lane']
    right_neighbors: List['Lane']
    entry_lanes: List['Lane']
    exit_lanes: List['Lane']
    max_ray_dist: float = 10.0

    def __init__(self, lane_id, map_features, lane_obj):
        self.lane_id = lane_id
        self.lane_pts = np.array([[p.x, p.y] for p in lane_obj.polyline])
        self.n_pts = len(self.lane_pts)
        self.left_pts = [None] * self.n_pts
        self.right_pts = [None] * self.n_pts
        self.normals = [None] * self.n_pts
        self.left_neighbors = []
        self.right_neighbors = []
        self.entry_lanes = []
        self.exit_lanes = []
        self.map_features = map_features
        self.lane_obj = lane_obj

    def is_straight(self):
        v = rotate(self.normals[0], math.pi / 2)
        for n in self.normals:
            if abs(np.dot(v, n)) > 0.01:
                return False
        return True

    def calculate_normals(self):
        for i in range(self.n_pts):
            p = self.lane_pts[i]
            if i < self.n_pts - 1:
                dp = self.lane_pts[i + 1] - p
            else:
                dp = p - self.lane_pts[i - 1]

            dp /= np.linalg.norm(dp)
            angle = math.pi / 2
            normal = np.array([
                math.cos(angle) * dp[0] - math.sin(angle) * dp[1],
                math.sin(angle) * dp[0] + math.cos(angle) * dp[1]
            ])
            self.normals[i] = normal

    def compute_width(self):
        width = np.zeros((len(self.lane_pts), 2))

        width[:, 0] = self.extract_width(self.lane_obj.left_boundaries)
        width[:, 1] = self.extract_width(self.lane_obj.right_boundaries)

        width[width[:, 0] == 0, 0] = width[width[:, 0] == 0, 1]
        width[width[:, 1] == 0, 1] = width[width[:, 1] == 0, 0]
        return width

    def extract_width(self, boundaries):
        l_width = np.zeros(len(self.lane_pts))
        for boundary in boundaries:
            lane_boundary = self.map_features[boundary.boundary_feature_id]
            boundary_polyline = np.array([[p.x, p.y] for p in lane_boundary.polyline])

            start_pos = self.lane_pts[boundary.lane_start_index]
            start_index = nearest_point(start_pos, boundary_polyline)
            seg_len = boundary.lane_end_index - boundary.lane_start_index
            end_index = min(start_index + seg_len, len(boundary_polyline) - 1)
            leng = min(end_index - start_index, seg_len) + 1
            self_range = range(boundary.lane_start_index, boundary.lane_start_index + leng)
            bound_range = range(start_index, start_index + leng)
            centerLane = self.lane_pts[self_range]
            bound = boundary_polyline[bound_range]
            dist = np.square(centerLane - bound)
            dist = np.sqrt(dist[:, 0] + dist[:, 1])
            l_width[self_range] = dist
        return l_width

    def get_lane_width(self):

        right_lanes = self.lane_obj.right_neighbors
        left_lanes = self.lane_obj.left_neighbors

        if len(right_lanes) + len(left_lanes) == 0:
            boundary_width = self.compute_width()
            return max(sum(boundary_width[0]), sum(boundary_width[1]), 6)

        dist_to_left_lane = 0
        dist_to_right_lane = 0

        if len(right_lanes) > 0:
            right_lane = self.map_features[right_lanes[0].feature_id]
            self_start = right_lanes[0].self_start_index
            neighbor_start = right_lanes[0].neighbor_start_index
            right_l_polyline = np.array([[p.x, p.y] for p in right_lane.polyline])
            n_point = right_l_polyline[neighbor_start]
            self_point = self.lane_pts[self_start]
            dist_to_right_lane = norm(n_point[0] - self_point[0], n_point[1] - self_point[1])

        if len(left_lanes) > 0:
            left_lane = self.map_features[left_lanes[-1].feature_id]
            self_start = left_lanes[-1].self_start_index
            neighbor_start = left_lanes[-1].neighbor_start_index
            left_l_polyline = np.array([[p.x, p.y] for p in left_lane.polyline])
            n_point = left_l_polyline[neighbor_start]
            self_point = self.lane_pts[self_start]
            dist_to_left_lane = norm(n_point[0] - self_point[0], n_point[1] - self_point[1])

        return max(dist_to_left_lane, dist_to_right_lane, 4)

    def get_lane_shape(self):
        lane_width = self.get_lane_width()
        lane_shape = buffered_shape(self.lane_pts, lane_width)
        return lane_shape

    def compute_boundaries(self, lane_obj, features):
        for i in range(self.n_pts):
            p = self.lane_pts[i]
            n = self.normals[i]
            if lane_obj.left_boundaries or lane_obj.right_boundaries:
                for name, lst in [("Left", list(lane_obj.left_boundaries)), ("Right", list(lane_obj.right_boundaries))]:
                    sign = -1.0 if name == "Right" else 1.0
                    ray_end = p + sign * self.max_ray_dist * n
                    # plt.plot([p[0], ray_end[0]], [p[1], ray_end[1]], linestyle=":", color="gray")

                    # Check ray-line intersection for each boundary segment
                    for boundary in lst:
                        feature = features[boundary.boundary_feature_id]
                        boundary_xs, boundary_ys = convert_polyline(feature.polyline)
                        boundary_pts = [np.array([x, y]) for x, y in zip(boundary_xs, boundary_ys)]
                        for j in range(len(boundary_pts) - 1):
                            b0 = boundary_pts[j]
                            b1 = boundary_pts[j + 1]
                            intersect_pt = line_intersect(b0, b1, p, ray_end)
                            if intersect_pt is not None:
                                if name == "Left":
                                    self.left_pts[i] = intersect_pt
                                else:
                                    self.right_pts[i] = intersect_pt
                                # plt.scatter(intersect_pt[0], intersect_pt[1], s=12, c="red")
                                break

    def fill_interp_left(self):
        start_width = np.linalg.norm(self.left_pts[0] - self.lane_pts[0])
        end_width = np.linalg.norm(self.left_pts[-1] - self.lane_pts[-1])
        dw = (end_width - start_width) / float(self.n_pts)
        for i in range(self.n_pts):
            new_width = start_width + dw * i
            self.left_pts[i] = self.lane_pts[i] + new_width * self.normals[i]

    def fill_interp_right(self):
        start_width = np.linalg.norm(self.right_pts[0] - self.lane_pts[0])
        end_width = np.linalg.norm(self.right_pts[-1] - self.lane_pts[-1])
        dw = (end_width - start_width) / float(self.n_pts)
        for i in range(self.n_pts):
            new_width = start_width + dw * i
            self.right_pts[i] = self.lane_pts[i] + new_width * self.normals[i]

    def assign_anchor_points(self):
        # Left
        if self.left_pts[0] is None:
            for l in self.entry_lanes:
                print(f"Checking entry lane: {l.lane_id}")
                if l.left_pts[-1] is not None:
                    print("Found anchor point for entry lane")
                    self.left_pts[0] = l.left_pts[-1]
                    break

        if self.left_pts[0] is None:
            for i in range(1, len(self.left_pts)):
                if self.left_pts[i] is not None:
                    self.left_pts[0] = self.left_pts[i]
                    break

        if self.left_pts[-1] is None:
            for l in self.exit_lanes:
                print(f"Checking exit lane: {l.lane_id}")
                if l.left_pts[0] is not None:
                    print("Found anchor point for exit lane")
                    self.left_pts[-1] = l.left_pts[0]
                    break

        if self.left_pts[-1] is None:
            for i in range(len(self.left_pts) - 1, -1, -1):
                if self.left_pts[i] is not None:
                    self.left_pts[-1] = self.left_pts[i]
                    break

        self.fill_interp_left()
        # Right
        if self.right_pts[0] is None:
            for l in self.entry_lanes:
                print(f"Checking entry lane: {l.lane_id}")
                if l.right_pts[-1] is not None:
                    print("Found anchor point for entry lane")
                    self.right_pts[0] = l.right_pts[-1]
                    break

        if self.right_pts[0] is None:
            for i in range(1, len(self.right_pts)):
                if self.right_pts[i] is not None:
                    self.right_pts[0] = self.right_pts[i]
                    break

        if self.right_pts[-1] is None:
            for l in self.exit_lanes:
                print(f"Checking exit lane: {l.lane_id}")
                if l.right_pts[0] is not None:
                    print("Found anchor point for exit lane")
                    self.right_pts[-1] = l.right_pts[0]
                    break

        if self.right_pts[-1] is None:
            for i in range(len(self.right_pts) - 1, -1, -1):
                if self.right_pts[i] is not None:
                    self.right_pts[-1] = self.right_pts[i]
                    break
        self.fill_interp_right()

    def fill_forward(self):
        for i in range(1, self.n_pts):
            if self.left_pts[i] is None:
                boundary_pt = self.left_pts[i - 1]
                if boundary_pt is not None:
                    lane_pt = self.lane_pts[i - 1]
                    width = np.linalg.norm(boundary_pt - lane_pt)
                    self.left_pts[i] = self.lane_pts[i] + width * self.normals[i]
            if self.right_pts[i] is None:
                boundary_pt = self.right_pts[i - 1]
                if boundary_pt is not None:
                    lane_pt = self.lane_pts[i - 1]
                    width = np.linalg.norm(boundary_pt - lane_pt)
                    self.right_pts[i] = self.lane_pts[i] - width * self.normals[i]

    def fill_backward(self):
        for i in range(self.n_pts - 2, -1, -1):
            if self.left_pts[i] is None:
                boundary_pt = self.left_pts[i + 1]
                if boundary_pt is not None:
                    lane_pt = self.lane_pts[i + 1]
                    width = np.linalg.norm(boundary_pt - lane_pt)
                    self.left_pts[i] = self.lane_pts[i] + width * self.normals[i]
            if self.right_pts[i] is None:
                boundary_pt = self.right_pts[i + 1]
                if boundary_pt is not None:
                    lane_pt = self.lane_pts[i + 1]
                    width = np.linalg.norm(boundary_pt - lane_pt)
                    self.right_pts[i] = self.lane_pts[i] - width * self.normals[i]

    def correct_degenerate_cases(self):
        # Check for high variance in widths
        left_widths = [None] * self.n_pts
        right_widths = [None] * self.n_pts
        variance_cutoff = 0.3
        for i in range(self.n_pts):
            if self.left_pts[i] is not None:
                width = np.linalg.norm(self.left_pts[i] - self.lane_pts[i])
                left_widths[i] = width
            if self.right_pts[i] is not None:
                width = np.linalg.norm(self.right_pts[i] - self.lane_pts[i])
                right_widths[i] = width

        if all([w is not None for w in left_widths]):
            left_variance = np.var(left_widths)
            # print(left_variance)
            if left_variance > variance_cutoff:
                start_width = left_widths[0]
                end_width = left_widths[-1]
                dw = (end_width - start_width) / float(self.n_pts)
                for i in range(self.n_pts):
                    new_width = start_width + dw * i
                    self.left_pts[i] = self.lane_pts[i] + new_width * self.normals[i]

        if all([w is not None for w in right_widths]):
            right_variance = np.var(right_widths)
            # print(right_variance)
            if right_variance > variance_cutoff:
                start_width = right_widths[0]
                end_width = right_widths[-1]
                dw = (end_width - start_width) / float(self.n_pts - 1)
                for i in range(self.n_pts):
                    new_width = start_width + dw * i
                    self.right_pts[i] = self.lane_pts[i] - new_width * self.normals[i]


def create_polygons(features, all_lanes):
    start = time.time()

    # Create Lane objects
    lanes: Dict[int, Lane] = dict()
    for lane_obj, lane_id in all_lanes:
        lane = Lane(lane_id, features, lane_obj)
        lanes[lane_id] = lane

    # Create lane connections
    for lane_obj, lane_id in all_lanes:
        lane = lanes[lane_id]
        if lane_obj.left_neighbors:
            for left_lane in lane_obj.left_neighbors:
                lane.left_neighbors.append(lanes[left_lane.feature_id])
        if lane_obj.right_neighbors:
            for right_lane in lane_obj.right_neighbors:
                lane.right_neighbors.append(lanes[right_lane.feature_id])
        if lane_obj.entry_lanes:
            for entry_lane in lane_obj.entry_lanes:
                lane.entry_lanes.append(lanes[entry_lane])
        if lane_obj.exit_lanes:
            for exit_lane in lane_obj.exit_lanes:
                lane.exit_lanes.append(lanes[exit_lane])

    ids = [
        81,
        86,
        88,
        92,
        93,
        89,
        80,
        87,
        96,
        110,
        109,
    ]

    # for lane_id in ids:
    #     lane_obj = features[lane_id]
    #     lane = lanes[lane_id]
    #     print(f"--- Lane {lane_id}")
    #     # print([l.lane_id for l in lane.entry_lanes])
    #     # print([l.lane_id for l in lane.exit_lanes])
    #
    #     lane.calculate_normals()
    #     lane.compute_boundaries(lane_obj, features)
    #     if lane.is_straight():
    #         lane.fill_backward()
    #         lane.fill_forward()
    #         lane.correct_degenerate_cases()

    # Assign anchor points
    # for lane_id in ids:
    #     lane = lanes[lane_id]
    #     lane.assign_anchor_points()
    #     lane.correct_degenerate_cases()

    # Create polygons and plot
    for lane_id in ids:
        lane = lanes[lane_id]
        lane_poly = lane.get_lane_shape()
        poly_xs, poly_ys = lane_poly.exterior.coords.xy
        # for p in lane.left_pts + lane.right_pts[::-1] + [lane.left_pts[0]]:
        #     if p is not None:
        #         poly_xs.append(p[0])
        #         poly_ys.append(p[1])
        plt.plot(poly_xs, poly_ys, "b-")
        # plt.scatter(poly_xs, poly_ys, s=12, c="blue")

    # Plot boundaries
    # for i in ids:
    #     lane_obj = features[i]
    #     plot_lane(lane_obj)
    #     # for lane, lane_id in all_lanes:
    #     if lane_obj.left_boundaries or lane_obj.right_boundaries:
    #         for name, lst in [("Left", list(lane_obj.left_boundaries)), ("Right", list(lane_obj.right_boundaries))]:
    #             for b in lst:
    #                 if b.boundary_type == 0:
    #                     plot_road_edge(features[b.boundary_feature_id])
    #                 else:
    #                     plot_road_line(features[b.boundary_feature_id])

    end = time.time()
    elapsed = round((end - start) * 1000.0, 3)
    print(f"create_polygons took: {elapsed} ms")


def plot(path, scenario_id):
    # Get data
    # trajectories, ego_id = read_trajectory_data(path, scenario_id)
    scenario_id, features, lanes = get_map_features(path, scenario_id)

    # Plot map and trajectories
    fig, ax = plt.subplots()
    ax.set_title(f"Scenario {scenario_id}")
    ax.axis("equal")
    # plot_map(map_features)
    create_polygons(features, lanes)

    # for k, v in trajectories.items():
    #     plt.scatter(v[0], v[1], marker='.')

    mng = plt.get_current_fig_manager()
    mng.resize(1000, 1000)
    # mng.resize(*mng.window.maxsize())
    plt.show()


def dump_plots(outdir, path):
    dataset = Waymo.read_dataset(path)
    scenario = None
    for record in dataset:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(record))

        scenario_id = scenario.scenario_id
        map_features = get_map_features_for_scenario(scenario)

        fig, ax = plt.subplots()
        ax.set_title(f"Scenario {scenario_id}")
        plot_map(map_features)
        mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        w = 1000
        h = 1000
        mng.resize(w, h)
        # plt.show()

        filename = f"scenario-{scenario_id}.png"
        outpath = os.path.join(outdir, filename)
        fig = plt.gcf()
        # w, h = mng.window.maxsize()
        dpi = 100
        fig.set_size_inches(w / dpi, h / dpi)
        print(f"Saving {outpath}")
        fig.savefig(outpath, dpi=100)
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="waymo.py",
        description="Extract map data from Waymo dataset and convert to SUMO.",
    )
    parser.add_argument("file", help="TFRecord file")
    parser.add_argument(
        "--outdir", help="output directory for screenshots", type=str
    )
    parser.add_argument(
        "--gen",
        help="generate sumo map",
        type=str,
        nargs=1,
        metavar="SCENARIO_ID",
    )
    parser.add_argument(
        "--plot",
        help="plot scenario map",
        type=str,
        nargs=1,
        metavar="SCENARIO_ID",
    )
    parser.add_argument(
        "--animate",
        help="plot scenario map and animate trajectories",
        type=str,
        nargs=1,
        metavar="SCENARIO_ID",
    )
    args = parser.parse_args()

    if args.outdir:
        dump_plots(args.outdir, args.file)
    else:
        plot(args.file, args.plot[0])
