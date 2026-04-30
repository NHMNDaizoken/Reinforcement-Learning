from unittest.mock import MagicMock

from src.baseline import MaxPressureBaseline


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
