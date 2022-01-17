"""
Visualization and prototyping script for the Waymo motion dataset.
"""

import argparse
import math
import os
import queue
import time
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from waymo_open_dataset.protos import scenario_pb2

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


def create_polygons(features, lanes):
    start = time.time()
    seen = set()
    q = queue.Queue()
    q.put(lanes[3])
    while not q.empty():
        lane, lane_id = q.get()
        if lane_id in seen:
            continue
        seen.add(lane_id)
        max_ray_dist = 10
        xs, ys = convert_polyline(lane.polyline)
        lane_pts = [np.array([x, y]) for x, y in zip(xs, ys)]
        n_pts = len(lane_pts)
        left_pts = [None] * n_pts
        right_pts = [None] * n_pts
        normals = [None] * n_pts

        plot_lane(lane)
        print(lane_id)

        if lane.entry_lanes:
            for entry_lane in lane.entry_lanes:
                q.put((features[entry_lane], entry_lane))
        if lane.exit_lanes:
            for exit_lane in lane.exit_lanes:
                q.put((features[exit_lane], exit_lane))
        if lane.left_neighbors:
            for l_n in lane.left_neighbors:
                q.put(features[l_n.feature_id], l_n.feature_id)
        if lane.right_neighbors:
            for r_n in lane.right_neighbors:
                q.put(features[r_n.feature_id], r_n.feature_id)

        if lane.left_boundaries or lane.right_boundaries:
            for name, lst in [("Left", list(lane.left_boundaries)), ("Right", list(lane.right_boundaries))]:
                for i in range(n_pts):
                    p = lane_pts[i]
                    if i < n_pts - 1:
                        dp = lane_pts[i + 1] - p
                    else:
                        dp = p - lane_pts[i - 1]

                    dp /= np.linalg.norm(dp)
                    angle = math.pi / 2
                    normal = np.array([
                        math.cos(angle) * dp[0] - math.sin(angle) * dp[1],
                        math.sin(angle) * dp[0] + math.cos(angle) * dp[1]
                    ])
                    normals[i] = normal
                    sign = -1.0 if name == "Right" else 1.0
                    ray_end = p + sign * max_ray_dist * normal
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
                                    left_pts[i] = intersect_pt
                                else:
                                    right_pts[i] = intersect_pt
                                # plt.scatter(intersect_pt[0], intersect_pt[1], s=12, c="red")
                                break

                # # Plot
                # for b in lst:
                #     if b.boundary_type == 0:
                #         plot_road_edge(features[b.boundary_feature_id])
                #     else:
                #         plot_road_line(features[b.boundary_feature_id])

        left_boundary_empty = all([p is None for p in left_pts])
        right_boundary_empty = all([p is None for p in right_pts])

        if left_boundary_empty and right_boundary_empty:
            pass
        else:
            # Fill in missing vals - backwards pass
            for i in range(n_pts - 2, -1, -1):
                if left_pts[i] is None:
                    boundary_pt = left_pts[i + 1]
                    if boundary_pt is not None:
                        lane_pt = lane_pts[i + 1]
                        width = np.linalg.norm(boundary_pt - lane_pt)
                        left_pts[i] = lane_pts[i] + width * normals[i]
                if right_pts[i] is None:
                    boundary_pt = right_pts[i + 1]
                    if boundary_pt is not None:
                        lane_pt = lane_pts[i + 1]
                        width = np.linalg.norm(boundary_pt - lane_pt)
                        right_pts[i] = lane_pts[i] - width * normals[i]

            # Fill in missing vals - forward pass
            for i in range(1, n_pts):
                if left_pts[i] is None:
                    boundary_pt = left_pts[i - 1]
                    if boundary_pt is not None:
                        lane_pt = lane_pts[i - 1]
                        width = np.linalg.norm(boundary_pt - lane_pt)
                        left_pts[i] = lane_pts[i] + width * normals[i]
                if right_pts[i] is None:
                    boundary_pt = right_pts[i - 1]
                    if boundary_pt is not None:
                        lane_pt = lane_pts[i - 1]
                        width = np.linalg.norm(boundary_pt - lane_pt)
                        right_pts[i] = lane_pts[i] - width * normals[i]

            if left_boundary_empty:
                for i in range(n_pts):
                    boundary_pt = right_pts[i]
                    if boundary_pt is not None:
                        left_pts[i] = lane_pts[i] - (boundary_pt - lane_pts[i])

            if right_boundary_empty:
                for i in range(n_pts):
                    boundary_pt = left_pts[i]
                    if boundary_pt is not None:
                        right_pts[i] = lane_pts[i] - (boundary_pt - lane_pts[i])

        if not all([p is not None for p in left_pts]):
            print(f"Empty left boundary for {lane_id}")
        if not all([p is not None for p in right_pts]):
            print(f"Empty right boundary for {lane_id}")

        # Create polygon
        poly_xs, poly_ys = [], []
        for p in left_pts + right_pts[::-1] + [left_pts[0]]:
            if p is not None:
                poly_xs.append(p[0])
                poly_ys.append(p[1])
        plt.plot(poly_xs, poly_ys, "k-")
        # plt.scatter(poly_xs, poly_ys, s=12, c="black")
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
