"""
TrafficEnv wraps CityFlow for independent traffic-light agents.

State: normalized incoming queues, downstream queues, waiting queues,
phase timing, min-green remaining, and one-hot current phase.
Reward: negative absolute pressure for the phase actually applied.
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
    DECISION_DURATION = 3600
    CLEARANCE_DURATION = 900

    def __init__(
        self,
        roadnet_path: str,
        flow_path: str,
        phase_map: dict[str, list[list[tuple[str, str]]]],
        sim_steps_per_action: int = 10,
        decision_duration: int = DECISION_DURATION,
        clearance_duration: int = CLEARANCE_DURATION,
        min_green: int = 10,
        yellow_time: int = 0,
        max_queue: float = 50.0,
        max_waiting: float = 50.0,
    ):
        self.phase_map = phase_map
        self.inter_ids = list(phase_map.keys())
        self.n_agents = len(self.inter_ids)
        self.sim_steps_per_action = sim_steps_per_action
        self.decision_duration = int(decision_duration)
        self.clearance_duration = int(clearance_duration)
        self.min_green = int(min_green)
        self.yellow_time = int(yellow_time)
        self.max_queue = float(max_queue)
        self.max_waiting = float(max_waiting)
        self._roadnet = roadnet_path
        self._flow = flow_path
        self.engine: "cityflow.Engine | Any | None" = None
        self._current_phases: dict[str, int] = {}
        self._phase_elapsed: dict[str, int] = {}
        self._terminal_emitted = False
        self._seen_vehicle_ids: set[str] = set()
        self._completed_vehicle_ids: set[str] = set()
        self._generated_vehicle_count = self._count_generated_vehicles(flow_path)
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
        self._phase_elapsed = {inter_id: 0 for inter_id in self.inter_ids}
        self._terminal_emitted = False
        self._seen_vehicle_ids = set()
        self._completed_vehicle_ids = set()
        for inter_id in self.inter_ids:
            self.engine.set_tl_phase(inter_id, 0)
        self._update_vehicle_tracking()
        return self._get_states()

    def step(self, actions: dict[str, int]):
        if self.engine is None:
            raise RuntimeError("reset() must be called before step()")

        start_time = float(self.engine.get_current_time())
        applied_actions: dict[str, int] = {}
        for inter_id, phase in actions.items():
            self._validate_action(inter_id, phase)
            applied_phase = self._resolve_action(inter_id, phase)
            if applied_phase != self._current_phases[inter_id]:
                self.engine.set_tl_phase(inter_id, applied_phase)
                self._current_phases[inter_id] = applied_phase
                self._phase_elapsed[inter_id] = 0
            applied_actions[inter_id] = applied_phase

        for _ in range(self.sim_steps_per_action):
            self._advance_one_second()

        states = self._get_states()
        rewards = self._get_rewards()
        end_time = float(self.engine.get_current_time())
        done = (
            start_time < self.decision_duration <= end_time
            and not self._terminal_emitted
        )
        if done:
            self._terminal_emitted = True

        metrics = self._get_vehicle_metrics()
        info: dict[str, float | bool | dict[str, int]] = {
            "atl": float(self.engine.get_average_travel_time()),
            "throughput": float(metrics["completed"]),
            "completed": float(metrics["completed"]),
            "active": float(metrics["active"]),
            "generated": float(metrics["generated"]),
            "completion_rate": float(metrics["completion_rate"]),
            "sim_time": end_time,
            "is_decision_phase": start_time < self.decision_duration,
            "is_clearance_phase": start_time >= self.decision_duration,
            "applied_actions": applied_actions,
        }
        return states, rewards, done, info

    def _get_states(self) -> list[np.ndarray]:
        counts = self.engine.get_lane_vehicle_count()
        waiting_counts = self._get_lane_waiting_counts()
        states = []
        for inter_id in self.inter_ids:
            phases = self.phase_map[inter_id]
            q_in = []
            q_out = []
            waiting = []
            for movements in phases:
                for in_lane, out_lane in movements:
                    q_in.append(self._normalize(counts.get(in_lane, 0), self.max_queue))
                    q_out.append(self._normalize(counts.get(out_lane, 0), self.max_queue))
                    waiting.append(
                        self._normalize(waiting_counts.get(in_lane, 0), self.max_waiting)
                    )
            phase_elapsed = min(
                float(self._phase_elapsed.get(inter_id, 0)) / max(1.0, float(self.min_green)),
                1.0,
            )
            min_green_remaining = max(
                0.0,
                float(self.min_green - self._phase_elapsed.get(inter_id, 0))
                / max(1.0, float(self.min_green)),
            )
            one_hot = np.zeros(len(phases), dtype=np.float32)
            one_hot[self._current_phases[inter_id]] = 1.0
            states.append(
                np.concatenate(
                    [q_in, q_out, waiting, [phase_elapsed, min_green_remaining], one_hot]
                ).astype(np.float32)
            )
        return states

    def _get_rewards(self) -> list[float]:
        counts = self.engine.get_lane_vehicle_count()
        rewards = []
        for inter_id in self.inter_ids:
            phase = self._current_phases[inter_id]
            pressure = self.phase_pressure(counts, self.phase_map[inter_id], phase)
            rewards.append(-float(abs(pressure)))
        return rewards

    def _phase_state_dim(self, phases: list[list[tuple[str, str]]]) -> int:
        n_movements = sum(len(movements) for movements in phases)
        return 3 * n_movements + 2 + len(phases)

    @staticmethod
    def phase_pressure(
        counts: dict[str, int],
        phases: list[list[tuple[str, str]]],
        phase: int,
    ) -> int:
        """Return signed pressure for one selected phase."""
        if phase < 0 or phase >= len(phases):
            return 0
        return int(
            sum(
                counts.get(in_lane, 0) - counts.get(out_lane, 0)
                for in_lane, out_lane in phases[phase]
            )
        )

    def action_masks(self) -> list[np.ndarray]:
        """Return per-agent action masks respecting min-green constraints."""
        masks = []
        for inter_id in self.inter_ids:
            mask = np.ones(len(self.phase_map[inter_id]), dtype=np.float32)
            if self._phase_elapsed.get(inter_id, 0) < self.min_green:
                mask[:] = 0.0
                mask[self._current_phases[inter_id]] = 1.0
            masks.append(mask)
        return masks

    @property
    def total_duration(self) -> int:
        return self.decision_duration + self.clearance_duration

    @property
    def clearance_action_steps(self) -> int:
        return int(np.ceil(self.clearance_duration / max(1, self.sim_steps_per_action)))

    def _resolve_action(self, inter_id: str, requested_phase: int) -> int:
        current = self._current_phases[inter_id]
        if requested_phase != current and self._phase_elapsed.get(inter_id, 0) < self.min_green:
            return current
        return requested_phase

    def _advance_one_second(self) -> None:
        self.engine.next_step()
        for inter_id in self.inter_ids:
            self._phase_elapsed[inter_id] = self._phase_elapsed.get(inter_id, 0) + 1
        self._update_vehicle_tracking()

    def _get_lane_waiting_counts(self) -> dict[str, int]:
        getter = getattr(self.engine, "get_lane_waiting_vehicle_count", None)
        if not callable(getter):
            return {}
        try:
            counts = getter()
        except Exception:
            return {}
        return counts if isinstance(counts, dict) else {}

    def _get_active_vehicle_ids(self) -> set[str] | None:
        getter = getattr(self.engine, "get_lane_vehicles", None)
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

    def _update_vehicle_tracking(self) -> None:
        active_ids = self._get_active_vehicle_ids()
        if active_ids is None:
            return
        self._completed_vehicle_ids.update(self._seen_vehicle_ids - active_ids)
        self._seen_vehicle_ids.update(active_ids)

    def _get_vehicle_metrics(self) -> dict[str, float]:
        active_ids = self._get_active_vehicle_ids()
        if active_ids is not None:
            self._completed_vehicle_ids.update(self._seen_vehicle_ids - active_ids)
            self._seen_vehicle_ids.update(active_ids)
            active = len(active_ids)
            completed = len(self._completed_vehicle_ids)
        else:
            active = self._safe_vehicle_count()
            completed = self._finished_vehicle_count()

        generated = max(self._generated_vehicle_count, len(self._seen_vehicle_ids), completed)
        completion_rate = completed / generated if generated > 0 else 0.0
        return {
            "completed": float(completed),
            "active": float(active),
            "generated": float(generated),
            "completion_rate": float(completion_rate),
        }

    def _safe_vehicle_count(self) -> int:
        getter = getattr(self.engine, "get_vehicle_count", None)
        if not callable(getter):
            return 0
        try:
            return int(getter())
        except Exception:
            return 0

    def _finished_vehicle_count(self) -> int:
        getter = getattr(self.engine, "get_finished_vehicle_count", None)
        if not callable(getter):
            return len(self._completed_vehicle_ids)
        try:
            return int(getter())
        except Exception:
            return len(self._completed_vehicle_ids)

    @staticmethod
    def _normalize(value: float, scale: float) -> float:
        return min(max(float(value) / max(scale, 1.0), 0.0), 1.0)

    @staticmethod
    def _count_generated_vehicles(flow_path: str) -> int:
        try:
            with open(flow_path, "r", encoding="utf-8") as file:
                flows = json.load(file)
        except (OSError, json.JSONDecodeError):
            return 0

        total = 0
        for flow in flows:
            start = float(flow.get("startTime", 0))
            end = float(flow.get("endTime", start))
            interval = max(float(flow.get("interval", 1.0)), 1e-6)
            if end <= start:
                total += 1
            else:
                total += int((end - start) // interval) + 1
        return total

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
