"""
TrafficEnv wraps CityFlow for independent traffic-light agents.

State: [incoming queues, outgoing queues, one-hot current phase]
Reward: negative MaxPressure.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import numpy as np

try:
    import cityflow
except ModuleNotFoundError:
    cityflow = None


class TrafficEnv:
    def __init__(
        self,
        roadnet_path: str,
        flow_path: str,
        phase_map: dict[str, list[list[tuple[str, str]]]],
        sim_steps_per_action: int = 10,
    ):
        self.phase_map = phase_map
        self.inter_ids = list(phase_map.keys())
        self.n_agents = len(self.inter_ids)
        self.sim_steps_per_action = sim_steps_per_action
        self._roadnet = roadnet_path
        self._flow = flow_path
        self.engine: "cityflow.Engine | Any | None" = None
        self._current_phases: dict[str, int] = {}
        self._validate_phase_map()

    def reset(self) -> list[np.ndarray]:
        if cityflow is None:
            raise RuntimeError("cityflow is required to run the simulator")
        if self.engine is not None:
            del self.engine

        cfg = {
            "interval": 1.0,
            "seed": 0,
            "dir": os.getcwd() + os.sep,
            "roadnetFile": self._roadnet,
            "flowFile": self._flow,
            "rlTrafficLight": True,
            "saveReplay": False,
            "roadnetLogFile": "",
            "replayLogFile": "",
            "laneChange": False,
            "threadNum": 1,
        }
        path = os.path.join(tempfile.gettempdir(), "cf_cfg.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(cfg, file)

        self.engine = cityflow.Engine(path, thread_num=1)
        self._current_phases = {inter_id: 0 for inter_id in self.inter_ids}
        for inter_id in self.inter_ids:
            self.engine.set_tl_phase(inter_id, 0)
        for _ in range(self.sim_steps_per_action):
            self.engine.next_step()
        return self._get_states()

    def step(self, actions: dict[str, int]):
        if self.engine is None:
            raise RuntimeError("reset() must be called before step()")

        for inter_id, phase in actions.items():
            self._validate_action(inter_id, phase)
            self.engine.set_tl_phase(inter_id, phase)
            self._current_phases[inter_id] = phase

        for _ in range(self.sim_steps_per_action):
            self.engine.next_step()

        states = self._get_states()
        rewards = self._get_rewards()
        done = self.engine.get_current_time() >= 3600
        throughput_getter = getattr(self.engine, "get_finished_vehicle_count", None)
        throughput = throughput_getter() if throughput_getter else self.engine.get_vehicle_count()
        info = {
            "atl": float(self.engine.get_average_travel_time()),
            "throughput": float(throughput),
        }
        return states, rewards, done, info

    def _get_states(self) -> list[np.ndarray]:
        counts = self.engine.get_lane_vehicle_count()
        states = []
        for inter_id in self.inter_ids:
            phases = self.phase_map[inter_id]
            q_in = []
            q_out = []
            for movements in phases:
                for in_lane, out_lane in movements:
                    q_in.append(float(counts.get(in_lane, 0)))
                    q_out.append(float(counts.get(out_lane, 0)))
            one_hot = np.zeros(len(phases), dtype=np.float32)
            one_hot[self._current_phases[inter_id]] = 1.0
            states.append(np.concatenate([q_in, q_out, one_hot]).astype(np.float32))
        return states

    def _get_rewards(self) -> list[float]:
        counts = self.engine.get_lane_vehicle_count()
        rewards = []
        for inter_id in self.inter_ids:
            pressure = sum(
                counts.get(in_lane, 0) - counts.get(out_lane, 0)
                for movements in self.phase_map[inter_id]
                for in_lane, out_lane in movements
            )
            rewards.append(-float(pressure))
        return rewards

    def _phase_state_dim(self, phases: list[list[tuple[str, str]]]) -> int:
        n_movements = sum(len(movements) for movements in phases)
        return 2 * n_movements + len(phases)

    def _validate_phase_map(self) -> None:
        if not self.phase_map:
            raise ValueError("phase_map must contain at least one intersection")

        expected_state_dim: int | None = None
        expected_action_dim: int | None = None
        for inter_id, phases in self.phase_map.items():
            if not phases:
                raise ValueError(f"intersection {inter_id!r} must have at least one phase")

            state_dim = self._phase_state_dim(phases)
            action_dim = len(phases)
            if expected_state_dim is None:
                expected_state_dim = state_dim
                expected_action_dim = action_dim
                continue

            if state_dim != expected_state_dim or action_dim != expected_action_dim:
                raise ValueError(
                    "all intersections must share the same state and action dimensions"
                )

    def _validate_action(self, inter_id: str, phase: int) -> None:
        if inter_id not in self.phase_map:
            raise ValueError(f"unknown intersection id: {inter_id!r}")
        if phase < 0 or phase >= len(self.phase_map[inter_id]):
            raise ValueError(
                f"invalid phase {phase} for intersection {inter_id!r}; "
                f"expected 0..{len(self.phase_map[inter_id]) - 1}"
            )

    @property
    def state_dim(self) -> int:
        phases = self.phase_map[self.inter_ids[0]]
        return self._phase_state_dim(phases)

    @property
    def action_dim(self) -> int:
        return len(self.phase_map[self.inter_ids[0]])
