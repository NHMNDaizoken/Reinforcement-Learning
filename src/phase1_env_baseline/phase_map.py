"""Utilities for reading traffic-light phase movements from CityFlow roadnets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PhaseMap = dict[str, list[list[tuple[str, str]]]]


def build_phase_map(roadnet_path: str | Path) -> PhaseMap:
    """Build intersection -> phase -> (incoming lane, outgoing lane) movements."""
    with Path(roadnet_path).open("r", encoding="utf-8") as file:
        roadnet: dict[str, Any] = json.load(file)

    phase_map: PhaseMap = {}
    for intersection in roadnet.get("intersections", []):
        if intersection.get("virtual", False):
            continue

        road_links = intersection.get("roadLinks", [])
        light_phases = intersection.get("trafficLight", {}).get("lightphases", [])
        phases: list[list[tuple[str, str]]] = []

        for light_phase in light_phases:
            movements: list[tuple[str, str]] = []
            for link_index in light_phase.get("availableRoadLinks", []):
                road_link = road_links[link_index]
                start_road = road_link["startRoad"]
                end_road = road_link["endRoad"]
                for lane_link in road_link.get("laneLinks", []):
                    movements.append(
                        (
                            f"{start_road}_{lane_link['startLaneIndex']}",
                            f"{end_road}_{lane_link['endLaneIndex']}",
                        )
                    )
            phases.append(movements)

        phase_map[intersection["id"]] = phases

    return phase_map
