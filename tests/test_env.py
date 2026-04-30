from unittest.mock import MagicMock

import pytest

from src.env import TrafficEnv


PHASE_MAP = {
    "i0": [[("in_A", "out_A")], [("in_B", "out_B")]],
    "i1": [[("in_C", "out_C")], [("in_D", "out_D")]],
}


def _mock_env(phase_current: dict, counts: dict) -> TrafficEnv:
    env = TrafficEnv.__new__(TrafficEnv)
    env.phase_map = PHASE_MAP
    env.inter_ids = ["i0", "i1"]
    env.n_agents = 2
    env._current_phases = phase_current
    env.engine = MagicMock()
    env.engine.get_lane_vehicle_count.return_value = counts
    return env


def test_state_dim_uses_queue_counts_and_phase_one_hot():
    env = _mock_env({"i0": 0, "i1": 0}, {})
    assert env.state_dim == 6


def test_reward_is_negative_max_pressure():
    counts = {
        "in_A": 10,
        "out_A": 2,
        "in_B": 3,
        "out_B": 7,
        "in_C": 0,
        "out_C": 0,
        "in_D": 0,
        "out_D": 0,
    }
    env = _mock_env({"i0": 0, "i1": 0}, counts)
    rewards = env._get_rewards()
    assert rewards[0] == pytest.approx(-4.0)


def test_state_contains_current_phase_one_hot():
    counts = {
        key: 0
        for key in ["in_A", "out_A", "in_B", "out_B", "in_C", "out_C", "in_D", "out_D"]
    }
    env = _mock_env({"i0": 0, "i1": 1}, counts)
    states = env._get_states()
    assert states[0][-2] == 1.0 and states[0][-1] == 0.0
    assert states[1][-2] == 0.0 and states[1][-1] == 1.0


def test_state_shape_matches_state_dim():
    counts = {
        key: 0
        for key in ["in_A", "out_A", "in_B", "out_B", "in_C", "out_C", "in_D", "out_D"]
    }
    env = _mock_env({"i0": 0, "i1": 0}, counts)
    states = env._get_states()
    assert all(state.shape == (env.state_dim,) for state in states)
