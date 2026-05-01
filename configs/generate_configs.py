"""
Generate CityFlow roadnet and flow files for a 3x3 traffic grid.

ID convention:
  - Intersection: node_{row}_{col}
  - Road: road_{src}_{dst}
  - Lane: road_{src}_{dst}_{lane_idx}
"""

import itertools
import json
from pathlib import Path


GRID_N = 3
ROAD_LEN = 300
MAX_SPEED = 11.11
LANE_WIDTH = 3.5
LANES = 3
OUT_DIR = Path(__file__).parent


def node(row: int, col: int) -> str:
    return f"node_{row}_{col}"


def virtual_node(side: str, index: int) -> str:
    return f"virtual_{side}_{index}"


def road(src: str, dst: str) -> str:
    return f"road_{src}_{dst}"


def point(row: int, col: int) -> dict[str, int]:
    return {"x": col * ROAD_LEN, "y": row * ROAD_LEN}


def build_roadnet(n: int = GRID_N) -> dict:
    intersections = []
    roads = []
    positions: dict[tuple[int, int], tuple[str, bool]] = {}

    for row, col in itertools.product(range(n), range(n)):
        positions[(row, col)] = (node(row, col), False)
    for row in range(n):
        positions[(row, -1)] = (virtual_node("west", row), True)
        positions[(row, n)] = (virtual_node("east", row), True)
    for col in range(n):
        positions[(-1, col)] = (virtual_node("north", col), True)
        positions[(n, col)] = (virtual_node("south", col), True)

    node_roads: dict[str, list[str]] = {node_id: [] for node_id, _ in positions.values()}

    def node_at(row: int, col: int) -> str:
        return positions[(row, col)][0]

    def add_road(src: str, dst: str, src_pos: tuple[int, int], dst_pos: tuple[int, int]) -> None:
        rid = road(src, dst)
        roads.append(
            {
                "id": rid,
                "startIntersection": src,
                "endIntersection": dst,
                "points": [point(*src_pos), point(*dst_pos)],
                "lanes": [
                    {"width": LANE_WIDTH, "maxSpeed": MAX_SPEED}
                    for _ in range(LANES)
                ],
            }
        )
        node_roads[src].append(rid)
        node_roads[dst].append(rid)

    for row, col in itertools.product(range(n), range(n)):
        for dst_pos in [(row, col + 1), (row + 1, col)]:
            if dst_pos not in positions:
                continue
            src_pos = (row, col)
            src = node_at(*src_pos)
            dst = node_at(*dst_pos)
            add_road(src, dst, src_pos, dst_pos)
            add_road(dst, src, dst_pos, src_pos)
        if col == 0:
            src_pos = (row, -1)
            dst_pos = (row, col)
            add_road(node_at(*src_pos), node_at(*dst_pos), src_pos, dst_pos)
            add_road(node_at(*dst_pos), node_at(*src_pos), dst_pos, src_pos)
        if row == 0:
            src_pos = (-1, col)
            dst_pos = (row, col)
            add_road(node_at(*src_pos), node_at(*dst_pos), src_pos, dst_pos)
            add_road(node_at(*dst_pos), node_at(*src_pos), dst_pos, src_pos)

    road_by_pair = {
        (item["startIntersection"], item["endIntersection"]): item["id"]
        for item in roads
    }

    for (row, col), (nid, is_virtual) in positions.items():
        if is_virtual:
            intersections.append(
                {
                    "id": nid,
                    "point": point(row, col),
                    "width": 0,
                    "roads": node_roads[nid],
                    "roadLinks": [],
                    "trafficLight": {
                        "roadLinkIndices": [],
                        "lightphases": [
                            {"time": 30, "availableRoadLinks": []},
                            {"time": 30, "availableRoadLinks": []},
                        ],
                    },
                    "virtual": True,
                }
            )
            continue

        road_links = []
        phase_0 = []
        phase_1 = []
        straight_pairs = [
            ((row, col - 1), (row, col + 1), phase_0),
            ((row, col + 1), (row, col - 1), phase_0),
            ((row - 1, col), (row + 1, col), phase_1),
            ((row + 1, col), (row - 1, col), phase_1),
        ]
        for src_pos, dst_pos, phase_bucket in straight_pairs:
            start_road = road_by_pair[(node_at(*src_pos), nid)]
            end_road = road_by_pair[(nid, node_at(*dst_pos))]
            phase_bucket.append(len(road_links))
            road_links.append(
                {
                    "type": "go_straight",
                    "startRoad": start_road,
                    "endRoad": end_road,
                    "direction": 0,
                    "laneLinks": [
                        {
                            "startLaneIndex": lane_idx,
                            "endLaneIndex": lane_idx,
                            "points": [],
                        }
                        for lane_idx in range(LANES)
                    ],
                }
            )
        intersections.append(
            {
                "id": nid,
                "point": point(row, col),
                "width": 10,
                "roads": node_roads[nid],
                "roadLinks": road_links,
                "trafficLight": {
                    "roadLinkIndices": list(range(len(road_links))),
                    "lightphases": [
                        {"time": 30, "availableRoadLinks": phase_0},
                        {"time": 30, "availableRoadLinks": phase_1},
                    ],
                },
                "virtual": False,
            }
        )

    return {"intersections": intersections, "roads": roads}


def build_flow(vph: int) -> list[dict]:
    base_vehicle = {
        "length": 5.0,
        "width": 2.0,
        "maxPosAcc": 2.0,
        "maxNegAcc": 4.5,
        "usualPosAcc": 2.0,
        "usualNegAcc": 4.5,
        "minGap": 2.5,
        "maxSpeed": MAX_SPEED,
        "headwayTime": 2.0,
    }
    n = GRID_N
    routes = [
        [road(virtual_node("west", row), node(row, 0))]
        + [road(node(row, col), node(row, col + 1)) for col in range(n - 1)]
        + [road(node(row, n - 1), virtual_node("east", row))]
        for row in range(n)
    ] + [
        [road(virtual_node("east", row), node(row, n - 1))]
        + [road(node(row, col + 1), node(row, col)) for col in range(n - 2, -1, -1)]
        + [road(node(row, 0), virtual_node("west", row))]
        for row in range(n)
    ] + [
        [road(virtual_node("north", col), node(0, col))]
        + [road(node(row, col), node(row + 1, col)) for row in range(n - 1)]
        + [road(node(n - 1, col), virtual_node("south", col))]
        for col in range(n)
    ] + [
        [road(virtual_node("south", col), node(n - 1, col))]
        + [road(node(row + 1, col), node(row, col)) for row in range(n - 2, -1, -1)]
        + [road(node(0, col), virtual_node("north", col))]
        for col in range(n)
    ]

    weighted_routes = [
        (routes[0], 0.16),
        (routes[1], 0.08),
        (routes[2], 0.14),
        (routes[3], 0.06),
        (routes[4], 0.12),
        (routes[5], 0.07),
        (routes[6], 0.10),
        (routes[7], 0.05),
        (routes[8], 0.09),
        (routes[9], 0.04),
        (routes[10], 0.06),
        (routes[11], 0.03),
    ]

    return [
        {
            "vehicle": base_vehicle,
            "route": route,
            "interval": 3600 / max(1, round(vph * weight)),
            "startTime": 0,
            "endTime": 3600,
        }
        for route, weight in weighted_routes
    ]


if __name__ == "__main__":
    roadnet = build_roadnet()
    (OUT_DIR / "roadnet.json").write_text(json.dumps(roadnet, indent=2), encoding="utf-8")
    print("roadnet.json generated")

    for label, vph in [("low", 300), ("medium", 600), ("high", 900)]:
        flow = build_flow(vph)
        (OUT_DIR / f"flow_{label}.json").write_text(
            json.dumps(flow, indent=2), encoding="utf-8"
        )
        print(f"flow_{label}.json generated (~{vph} veh/h total)")
