"""
MaxPressure rule-based controller.

This is the greedy local controller DQN must beat after training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.phase1_env_baseline.phase_map import PhaseMap, build_phase_map
from src.phase1_env_baseline.traffic_env import TrafficEnv

try:
    import cityflow
except ModuleNotFoundError:
    cityflow = None


class MaxPressureBaseline:
    def __init__(
        self,
        engine: "cityflow.Engine | Any",
        phase_map: PhaseMap,
    ):
        self.engine = engine
        self.phase_map = phase_map

    @staticmethod
    def phase_pressures(
        counts: dict[str, int],
        phases: list[list[tuple[str, str]]],
    ) -> list[int]:
        """Return MaxPressure score for each phase: sum(q_in) - sum(q_out)."""
        return [
            int(
                sum(
                    counts.get(in_lane, 0) - counts.get(out_lane, 0)
                    for in_lane, out_lane in movements
                )
            )
            for movements in phases
        ]

    def select_actions(self) -> dict[str, int]:
        counts = self.engine.get_lane_vehicle_count()
        actions = {}
        for inter_id, phases in self.phase_map.items():
            pressures = self.phase_pressures(counts, phases)
            actions[inter_id] = max(range(len(pressures)), key=pressures.__getitem__)
        return actions


def _write_cityflow_config(roadnet_path: str, flow_path: str) -> str:
    config = {
        "interval": 1.0,
        "seed": 0,
        "dir": os.getcwd() + os.sep,
        "roadnetFile": roadnet_path,
        "flowFile": flow_path,
        "rlTrafficLight": True,
        "saveReplay": False,
        "roadnetLogFile": "",
        "replayLogFile": "",
        "laneChange": False,
        "threadNum": 1,
    }
    path = os.path.join(tempfile.gettempdir(), "btck_cityflow_baseline.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(config, file)
    return path


def run_baseline(
    roadnet_path: str,
    flow_path: str,
    steps: int = TrafficEnv.DECISION_DURATION,
    action_interval: int = 10,
    clearance_steps: int | None = None,
) -> dict[str, float]:
    """Run MaxPressure in CityFlow and return summary metrics."""
    if cityflow is None:
        raise RuntimeError("cityflow is required to run the baseline simulator")

    phase_map = build_phase_map(roadnet_path)
    env = TrafficEnv(
        roadnet_path=roadnet_path,
        flow_path=flow_path,
        phase_map=phase_map,
        sim_steps_per_action=action_interval,
        decision_duration=steps,
    )
    states = env.reset()
    controller = MaxPressureBaseline(env.engine, phase_map)
    final_info = {
        "atl": 0.0,
        "throughput": 0.0,
        "completed": 0.0,
        "active": 0.0,
        "generated": 0.0,
        "completion_rate": 0.0,
    }

    decision_action_steps = max(1, (steps + max(1, action_interval) - 1) // max(1, action_interval))
    for _ in range(decision_action_steps):
        states, _, done, final_info = env.step(controller.select_actions())
        if done:
            break

    clearance_limit = clearance_steps if clearance_steps is not None else env.clearance_action_steps
    for _ in range(max(0, clearance_limit)):
        if float(final_info.get("active", 0.0)) <= 0 and float(final_info.get("generated", 0.0)) > 0:
            break
        states, _, _, final_info = env.step(controller.select_actions())

    return {
        "atl": float(final_info.get("atl", 0.0)),
        "throughput": float(final_info.get("completed", final_info.get("throughput", 0.0))),
        "completed": float(final_info.get("completed", 0.0)),
        "active": float(final_info.get("active", 0.0)),
        "generated": float(final_info.get("generated", 0.0)),
        "completion_rate": float(final_info.get("completion_rate", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MaxPressure baseline.")
    parser.add_argument("--roadnet", default="configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json")
    parser.add_argument("--flow", default="configs/flow_medium_flat.json")
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--action-interval", type=int, default=10)
    parser.add_argument("--clearance-steps", type=int, default=None)
    args = parser.parse_args()

    metrics = run_baseline(
        args.roadnet,
        args.flow,
        args.steps,
        args.action_interval,
        args.clearance_steps,
    )
    print(f"Baseline ATL: {metrics['atl']:.3f}")
    print(f"Baseline completed vehicles: {metrics['completed']:.0f}")
    print(f"Baseline active vehicles: {metrics['active']:.0f}")
    print(f"Baseline generated vehicles: {metrics['generated']:.0f}")
    print(f"Baseline completion rate: {metrics['completion_rate']:.3f}")


if __name__ == "__main__":
    main()
