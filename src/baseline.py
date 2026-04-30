"""
MaxPressure rule-based controller.

This is the greedy local controller DQN must beat after training.
"""

from __future__ import annotations

from typing import Any

try:
    import cityflow
except ModuleNotFoundError:
    cityflow = None


class MaxPressureBaseline:
    def __init__(
        self,
        engine: "cityflow.Engine | Any",
        phase_map: dict[str, list[list[tuple[str, str]]]],
    ):
        self.engine = engine
        self.phase_map = phase_map

    def select_actions(self) -> dict[str, int]:
        counts = self.engine.get_lane_vehicle_count()
        return {
            inter_id: max(
                range(len(phases)),
                key=lambda phase_idx: sum(
                    counts.get(in_lane, 0) - counts.get(out_lane, 0)
                    for in_lane, out_lane in phases[phase_idx]
                ),
            )
            for inter_id, phases in self.phase_map.items()
        }
