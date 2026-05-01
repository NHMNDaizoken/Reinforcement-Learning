"""Export CityFlow replay data for the React Native dashboard."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase1_env_baseline.max_pressure import MaxPressureBaseline, _write_cityflow_config, cityflow
from src.phase1_env_baseline.phase_map import build_phase_map


def _agent_position(agent_id: str) -> tuple[int, int]:
    _, row, col = agent_id.split("_")
    return int(row), int(col)


def _queue_for_agent(counts: dict[str, int], phases: list[list[tuple[str, str]]]) -> int:
    lanes = {in_lane for movements in phases for in_lane, _ in movements}
    return int(sum(counts.get(lane, 0) for lane in lanes))


def _load_roads(roadnet_path: str) -> dict[str, dict]:
    with Path(roadnet_path).open("r", encoding="utf-8") as file:
        roadnet = json.load(file)
    return {road["id"]: road for road in roadnet["roads"]}


def _vehicle_xy(
    lane_id: str,
    distance: float,
    road_map: dict[str, dict],
    canvas_size: int = 820,
) -> tuple[float, float] | None:
    road_id = lane_id.rsplit("_", 1)[0]
    road = road_map.get(road_id)
    if road is None:
        return None

    start, end = road["points"][0], road["points"][-1]
    dx = float(end["x"] - start["x"])
    dy = float(end["y"] - start["y"])
    length = max(math.hypot(dx, dy), 1.0)
    ratio = max(0.0, min(1.0, distance / length))
    world_x = float(start["x"]) + dx * ratio
    world_y = float(start["y"]) + dy * ratio

    lane_index = int(lane_id.rsplit("_", 1)[1])
    offset = (lane_index - 1) * 5.0
    normal_x = -dy / length
    normal_y = dx / length
    world_x += normal_x * offset
    world_y += normal_y * offset

    margin = 170
    scale = (canvas_size - 2 * margin) / 600
    screen_x = margin + world_x * scale
    screen_y = margin + world_y * scale
    return screen_x, screen_y


def _snapshot_vehicles(engine, road_map: dict[str, dict], max_vehicles: int = 240) -> list[dict]:
    lane_vehicles = engine.get_lane_vehicles()
    distances = engine.get_vehicle_distance()
    speeds = engine.get_vehicle_speed()
    vehicles = []
    for lane_id, vehicle_ids in sorted(lane_vehicles.items()):
        for vehicle_id in vehicle_ids:
            xy = _vehicle_xy(lane_id, float(distances.get(vehicle_id, 0.0)), road_map)
            if xy is None:
                continue
            vehicles.append(
                {
                    "id": vehicle_id,
                    "x": round(xy[0], 1),
                    "y": round(xy[1], 1),
                    "waiting": float(speeds.get(vehicle_id, 0.0)) < 0.1,
                }
            )
            if len(vehicles) >= max_vehicles:
                return vehicles
    return vehicles


def export_replay_data(
    roadnet_path: str,
    flow_path: str,
    steps: int,
    action_interval: int,
    output_path: str,
    algorithm: str = "baseline",
) -> None:
    if cityflow is None:
        raise RuntimeError("cityflow is required to export replay data")
    if algorithm != "baseline":
        raise NotImplementedError(
            "Model replay export requires trained model evaluation data first."
        )

    phase_map = build_phase_map(roadnet_path)
    road_map = _load_roads(roadnet_path)
    engine = cityflow.Engine(_write_cityflow_config(roadnet_path, flow_path), thread_num=1)
    controller = MaxPressureBaseline(engine, phase_map)
    current_actions = {agent_id: 0 for agent_id in phase_map}
    frames = []

    for elapsed in range(steps):
        if elapsed % action_interval == 0:
            current_actions = controller.select_actions()
            for agent_id, action in current_actions.items():
                engine.set_tl_phase(agent_id, action)

        counts = engine.get_lane_vehicle_count()
        agents = []
        total_queue = 0
        for agent_id in sorted(phase_map):
            row, col = _agent_position(agent_id)
            queue = _queue_for_agent(counts, phase_map[agent_id])
            pressures = MaxPressureBaseline.phase_pressures(counts, phase_map[agent_id])
            action = int(current_actions[agent_id])
            total_queue += queue
            agents.append(
                {
                    "id": agent_id,
                    "row": row,
                    "col": col,
                    "action": action,
                    "queue": queue,
                    "phase_pressures": pressures,
                    "selected_pressure": pressures[action],
                }
            )

        throughput_getter = getattr(engine, "get_finished_vehicle_count", None)
        throughput = throughput_getter() if throughput_getter else engine.get_vehicle_count()
        frames.append(
            {
                "time": float(engine.get_current_time()),
                "atl": float(engine.get_average_travel_time()),
                "throughput": int(throughput),
                "total_queue": total_queue,
                "agents": agents,
                "vehicles": _snapshot_vehicles(engine, road_map),
            }
        )
        engine.next_step()

    replay = {
        "roadnet": roadnet_path,
        "flow": flow_path,
        "traffic_level": Path(flow_path).stem.replace("flow_", ""),
        "algorithm": algorithm,
        "algorithm_label": "Baseline - MaxPressure Actuated Control",
        "available_algorithms": ["baseline"],
        "steps": steps,
        "action_interval": action_interval,
        "frames": frames,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(replay, separators=(",", ":")), encoding="utf-8")
    print(f"Replay data written to {output}")
    print(f"Frames: {len(frames)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export replay data for the dashboard.")
    parser.add_argument("--roadnet", default="configs/roadnet.json")
    parser.add_argument("--flow", default="configs/flow_high.json")
    parser.add_argument("--algorithm", choices=["baseline", "model"], default="baseline")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--action-interval", type=int, default=10)
    parser.add_argument("--output", default="web/data/high.json")
    args = parser.parse_args()
    export_replay_data(
        args.roadnet,
        args.flow,
        args.steps,
        args.action_interval,
        args.output,
        args.algorithm,
    )


if __name__ == "__main__":
    main()
