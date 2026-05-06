"""Export CityFlow replay data for the React Native dashboard."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase1_env_baseline.max_pressure import MaxPressureBaseline, _write_cityflow_config, cityflow
from src.phase1_env_baseline.phase_map import build_phase_map
from src.phase2_dqn.dqn_agent import SharedDQNAgent
from src.phase1_env_baseline.traffic_env import TrafficEnv


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
) -> tuple[float, float, float] | None:
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

    angle = math.degrees(math.atan2(dy, dx))

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
    return screen_x, screen_y, angle


def _vehicle_type(vehicle_id: str) -> str:
    try:
        num = int("".join(filter(str.isdigit, vehicle_id)) or "1")
    except ValueError:
        num = 1
    if num % 5 == 0:
        return "bus"
    if num % 2 == 0:
        return "motorbike"
    return "car"


def _snapshot_vehicles(engine, road_map: dict[str, dict], max_vehicles: int = 240) -> list[dict]:
    lane_vehicles = engine.get_lane_vehicles()
    distances = engine.get_vehicle_distance()
    speeds = engine.get_vehicle_speed()
    vehicles = []
    for lane_id, vehicle_ids in sorted(lane_vehicles.items()):
        for vehicle_id in vehicle_ids:
            result = _vehicle_xy(lane_id, float(distances.get(vehicle_id, 0.0)), road_map)
            if result is None:
                continue
            screen_x, screen_y, angle = result
            vehicles.append({
                "id": vehicle_id,
                "x": round(screen_x, 1),
                "y": round(screen_y, 1),
                "angle": round(angle, 1),
                "type": _vehicle_type(vehicle_id),
                "waiting": float(speeds.get(vehicle_id, 0.0)) < 0.1,
            })
            if len(vehicles) >= max_vehicles:
                return vehicles
    return vehicles


def _load_dqn_model(agent: SharedDQNAgent, model_path: str) -> bool:
    """Load trained DQN model weights. Returns True if successful."""
    path = Path(model_path)
    if not path.exists():
        print(f"Model file not found: {model_path}")
        return False
    try:
        state_dict = torch.load(str(path), map_location="cpu")
        agent.q_network.load_state_dict(state_dict)
        agent.q_network.eval()
        print(f"DQN model loaded from {model_path}")
        return True
    except Exception as exc:
        print(f"Failed to load model: {exc}")
        return False


def export_replay_data(
    roadnet_path: str,
    flow_path: str,
    steps: int,
    action_interval: int,
    output_path: str,
    algorithm: str = "baseline",
    model_path: str = "models/best_curriculum.pth",
) -> None:
    if cityflow is None:
        raise RuntimeError("cityflow is required to export replay data")

    phase_map = build_phase_map(roadnet_path)
    road_map = _load_roads(roadnet_path)
    frames = []
    tracker = _VehicleTracker(TrafficEnv._count_generated_vehicles(flow_path))

    # ── BASELINE (MaxPressure) ──────────────────────────────────────────────
    if algorithm == "baseline":
        engine = cityflow.Engine(_write_cityflow_config(roadnet_path, flow_path), thread_num=1)
        controller = MaxPressureBaseline(engine, phase_map)
        current_actions = {agent_id: 0 for agent_id in phase_map}

        for elapsed in range(steps):
            if elapsed % action_interval == 0:
                current_actions = controller.select_actions()
                for agent_id, action in current_actions.items():
                    engine.set_tl_phase(agent_id, action)

            counts = engine.get_lane_vehicle_count()
            agents, total_queue = _build_agents(counts, phase_map, current_actions)
            metrics = tracker.update(engine)

            frames.append({
                "time": float(engine.get_current_time()),
                "atl": float(engine.get_average_travel_time()),
                "throughput": int(metrics["completed"]),
                "completed": int(metrics["completed"]),
                "active": int(metrics["active"]),
                "generated": int(metrics["generated"]),
                "completion_rate": metrics["completion_rate"],
                "total_queue": total_queue,
                "agents": agents,
                "vehicles": _snapshot_vehicles(engine, road_map),
            })
            engine.next_step()

        algorithm_label = "Baseline — MaxPressure Actuated Control"

    # ── DQN MODEL ───────────────────────────────────────────────────────────
    elif algorithm == "model":
        env = TrafficEnv(
            roadnet_path=roadnet_path,
            flow_path=flow_path,
            phase_map=phase_map,
            sim_steps_per_action=action_interval,
        )
        agent = SharedDQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
        loaded = _load_dqn_model(agent, model_path)
        if not loaded:
            raise RuntimeError(f"Cannot load DQN model from {model_path}")

        states = env.reset()
        # FIX: dùng env.engine thay vì tạo engine riêng
        engine = env.engine
        current_actions = {agent_id: 0 for agent_id in env.inter_ids}

        for elapsed in range(steps):
            if elapsed % action_interval == 0:
                action_values = agent.select_actions(states, training=False)
                current_actions = {
                    agent_id: int(action_values[idx])
                    for idx, agent_id in enumerate(env.inter_ids)
                }
                next_states, _, done, info = env.step(current_actions)
                # FIX: cập nhật engine reference sau mỗi step
                engine = env.engine
                states = next_states

            counts = engine.get_lane_vehicle_count()
            agents, total_queue = _build_agents(counts, phase_map, current_actions)
            metrics = tracker.update(engine)

            frames.append({
                "time": float(engine.get_current_time()),
                "atl": float(engine.get_average_travel_time()),
                "throughput": int(metrics["completed"]),
                "completed": int(metrics["completed"]),
                "active": int(metrics["active"]),
                "generated": int(metrics["generated"]),
                "completion_rate": metrics["completion_rate"],
                "total_queue": total_queue,
                "agents": agents,
                "vehicles": _snapshot_vehicles(engine, road_map),
            })
            # FIX: KHÔNG gọi engine.next_step() vì TrafficEnv đã tự step

        algorithm_label = "Trained DQN — Shared Q-Network (9 agents)"

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    replay = {
        "roadnet": roadnet_path,
        "flow": flow_path,
        "traffic_level": Path(flow_path).stem.replace("flow_", ""),
        "algorithm": algorithm,
        "algorithm_label": algorithm_label,
        "available_algorithms": ["baseline", "model"],
        "steps": steps,
        "action_interval": action_interval,
        "frames": frames,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(replay, separators=(",", ":")), encoding="utf-8")
    print(f"Replay data written to {output}")
    print(f"Frames: {len(frames)}")


def _build_agents(
    counts: dict[str, int],
    phase_map: dict,
    current_actions: dict[str, int],
) -> tuple[list[dict], int]:
    agents = []
    total_queue = 0
    for agent_id in sorted(phase_map):
        row, col = _agent_position(agent_id)
        queue = _queue_for_agent(counts, phase_map[agent_id])
        pressures = MaxPressureBaseline.phase_pressures(counts, phase_map[agent_id])
        action = int(current_actions.get(agent_id, 0))
        total_queue += queue
        agents.append({
            "id": agent_id,
            "row": row,
            "col": col,
            "action": action,
            "queue": queue,
            "phase_pressures": pressures,
            "selected_pressure": pressures[action] if action < len(pressures) else 0,
        })
    return agents, total_queue


class _VehicleTracker:
    def __init__(self, generated: int):
        self.generated = generated
        self.seen: set[str] = set()
        self.completed: set[str] = set()

    def update(self, engine) -> dict[str, float]:
        active_ids = _active_vehicle_ids(engine)
        if active_ids is not None:
            self.completed.update(self.seen - active_ids)
            self.seen.update(active_ids)
            active = len(active_ids)
            completed = len(self.completed)
        else:
            active = _active_vehicle_count(engine)
            completed = _finished_vehicle_count(engine)

        generated = max(self.generated, len(self.seen), completed)
        return {
            "completed": float(completed),
            "active": float(active),
            "generated": float(generated),
            "completion_rate": float(completed / generated) if generated else 0.0,
        }


def _active_vehicle_ids(engine) -> set[str] | None:
    getter = getattr(engine, "get_lane_vehicles", None)
    if not callable(getter):
        return None
    try:
        lane_vehicles = getter()
    except Exception:
        return None
    if not isinstance(lane_vehicles, dict):
        return None
    return {
        str(vehicle_id)
        for vehicle_ids in lane_vehicles.values()
        for vehicle_id in vehicle_ids
    }


def _active_vehicle_count(engine) -> int:
    getter = getattr(engine, "get_vehicle_count", None)
    if not callable(getter):
        return 0
    try:
        return int(getter())
    except Exception:
        return 0


def _finished_vehicle_count(engine) -> int:
    getter = getattr(engine, "get_finished_vehicle_count", None)
    if not callable(getter):
        return 0
    try:
        return int(getter())
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Export replay data for the dashboard.")
    parser.add_argument("--roadnet", default="configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json")
    parser.add_argument("--flow", default="configs/flow_high_flat.json")
    parser.add_argument("--algorithm", choices=["baseline", "model"], default="baseline")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--action-interval", type=int, default=10)
    parser.add_argument("--output", default="web/data/high.json")
    parser.add_argument("--model-path", default="models/best_curriculum.pth")
    args = parser.parse_args()
    export_replay_data(
        args.roadnet,
        args.flow,
        args.steps,
        args.action_interval,
        args.output,
        args.algorithm,
        args.model_path,
    )


if __name__ == "__main__":
    main()
