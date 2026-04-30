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


def road(src: str, dst: str) -> str:
    return f"road_{src}_{dst}"


def point(row: int, col: int) -> dict[str, int]:
    return {"x": col * ROAD_LEN, "y": row * ROAD_LEN}


def parse_node(node_id: str) -> tuple[int, int]:
    _, row, col = node_id.split("_")
    return int(row), int(col)


def build_roadnet(n: int = GRID_N) -> dict:
    intersections = []
    roads = []
    node_roads: dict[str, list[str]] = {
        node(row, col): [] for row, col in itertools.product(range(n), range(n))
    }

    def add_road(src: str, dst: str) -> None:
        rid = road(src, dst)
        src_row, src_col = parse_node(src)
        dst_row, dst_col = parse_node(dst)
        roads.append(
            {
                "id": rid,
                "startIntersection": src,
                "endIntersection": dst,
                "points": [point(src_row, src_col), point(dst_row, dst_col)],
                "lanes": [
                    {"width": LANE_WIDTH, "maxSpeed": MAX_SPEED}
                    for _ in range(LANES)
                ],
            }
        )
        node_roads[src].append(rid)
        node_roads[dst].append(rid)

    for row, col in itertools.product(range(n), range(n)):
        if col + 1 < n:
            add_road(node(row, col), node(row, col + 1))
            add_road(node(row, col + 1), node(row, col))
        if row + 1 < n:
            add_road(node(row, col), node(row + 1, col))
            add_road(node(row + 1, col), node(row, col))

    road_by_pair = {
        (item["startIntersection"], item["endIntersection"]): item["id"]
        for item in roads
    }

    for row, col in itertools.product(range(n), range(n)):
        nid = node(row, col)
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
            src_row, src_col = src_pos
            dst_row, dst_col = dst_pos
            if not (
                0 <= src_row < n
                and 0 <= src_col < n
                and 0 <= dst_row < n
                and 0 <= dst_col < n
            ):
                continue
            start_road = road_by_pair[(node(src_row, src_col), nid)]
            end_road = road_by_pair[(nid, node(dst_row, dst_col))]
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
    interval = 3600 / vph
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
        [road(node(row, col), node(row, col + 1)) for col in range(n - 1)]
        for row in range(n)
    ] + [
        [road(node(row, col + 1), node(row, col)) for col in range(n - 2, -1, -1)]
        for row in range(n)
    ] + [
        [road(node(row, col), node(row + 1, col)) for row in range(n - 1)]
        for col in range(n)
    ] + [
        [road(node(row + 1, col), node(row, col)) for row in range(n - 2, -1, -1)]
        for col in range(n)
    ]

    return [
        {
            "vehicle": base_vehicle,
            "route": route,
            "interval": interval,
            "startTime": 0,
            "endTime": 3600,
        }
        for route in routes[:4]
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
        print(f"flow_{label}.json generated ({vph} veh/h, interval={3600 / vph:.1f}s)")
