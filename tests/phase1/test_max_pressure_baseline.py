from unittest.mock import MagicMock

from src.phase1_env_baseline.max_pressure import MaxPressureBaseline
from src.phase1_env_baseline.phase_map import build_phase_map


PHASE_MAP = {
    "i0": [
        [("in_A", "out_A")],
        [("in_B", "out_B")],
    ]
}


def _engine(counts):
    engine = MagicMock()
    engine.get_lane_vehicle_count.return_value = counts
    return engine


def test_selects_highest_pressure_phase():
    engine = _engine({"in_A": 10, "out_A": 2, "in_B": 1, "out_B": 8})
    baseline = MaxPressureBaseline(engine, PHASE_MAP)
    assert baseline.select_actions()["i0"] == 0


def test_avoids_phase_when_downstream_is_congested():
    engine = _engine({"in_A": 5, "out_A": 8, "in_B": 5, "out_B": 0})
    baseline = MaxPressureBaseline(engine, PHASE_MAP)
    assert baseline.select_actions()["i0"] == 1


def test_select_actions_returns_action_for_each_intersection():
    phase_map = {
        "i0": [[("in_A", "out_A")], [("in_B", "out_B")]],
        "i1": [[("in_C", "out_C")], [("in_D", "out_D")]],
    }
    engine = _engine(
        {
            "in_A": 1,
            "out_A": 0,
            "in_B": 0,
            "out_B": 0,
            "in_C": 0,
            "out_C": 0,
            "in_D": 2,
            "out_D": 0,
        }
    )
    baseline = MaxPressureBaseline(engine, phase_map)
    assert baseline.select_actions() == {"i0": 0, "i1": 1}


def test_build_phase_map_reads_generated_roadnet():
    phase_map = build_phase_map("configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json")
    assert len(phase_map) == 9
    assert all(len(phases) >= 2 for phases in phase_map.values())
