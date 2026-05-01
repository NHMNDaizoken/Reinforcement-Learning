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
    steps: int = 3600,
    action_interval: int = 10,
) -> dict[str, float]:
    """Run MaxPressure in CityFlow and return summary metrics."""
    if cityflow is None:
        raise RuntimeError("cityflow is required to run the baseline simulator")

    phase_map = build_phase_map(roadnet_path)
    engine = cityflow.Engine(_write_cityflow_config(roadnet_path, flow_path), thread_num=1)
    controller = MaxPressureBaseline(engine, phase_map)

    for elapsed in range(steps):
        if elapsed % action_interval == 0:
            for inter_id, phase in controller.select_actions().items():
                engine.set_tl_phase(inter_id, phase)
        engine.next_step()

    throughput_getter = getattr(engine, "get_finished_vehicle_count", None)
    throughput = throughput_getter() if throughput_getter else engine.get_vehicle_count()
    return {
        "atl": float(engine.get_average_travel_time()),
        "throughput": float(throughput),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MaxPressure baseline.")
    parser.add_argument("--roadnet", default="configs/roadnet.json")
    parser.add_argument("--flow", default="configs/flow_medium.json")
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--action-interval", type=int, default=10)
    args = parser.parse_args()

    metrics = run_baseline(args.roadnet, args.flow, args.steps, args.action_interval)
    print(f"Baseline ATL: {metrics['atl']:.3f}")
    print(f"Baseline throughput: {metrics['throughput']:.0f}")


if __name__ == "__main__":
    main()
