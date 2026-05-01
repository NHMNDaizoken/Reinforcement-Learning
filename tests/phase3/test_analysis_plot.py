import sqlite3

import pandas as pd

from analysis.plot import load_phase3_data, write_plots


def test_load_phase3_data_uses_reward_as_atl_fallback(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        [
            {"episode": 0, "reward": -2.0},
            {"episode": 0, "reward": -3.0},
            {"episode": 1, "reward": -1.5},
        ]
    ).to_csv(data_dir / "buffer_dqn.csv", index=False)

    data = load_phase3_data(data_dir)

    assert data["source"].tolist() == ["dqn", "dqn"]
    assert data["reward"].tolist() == [-5.0, -1.5]
    assert data["atl"].tolist() == [2.5, 1.5]
    assert data["atl_proxy"].tolist() == [True, True]


def test_load_phase3_data_reads_missing_baseline_from_sqlite(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame([{"episode": 0, "reward": -1.0, "atl": 4.0}]).to_csv(
        data_dir / "buffer_dqn.csv", index=False
    )

    with sqlite3.connect(data_dir / "offline_dataset.db") as connection:
        connection.execute(
            "CREATE TABLE transitions (episode INTEGER, reward REAL, atl REAL, policy TEXT)"
        )
        connection.executemany(
            "INSERT INTO transitions VALUES (?, ?, ?, ?)",
            [(0, -2.0, 7.0, "baseline"), (1, -3.0, 8.0, "baseline")],
        )

    data = load_phase3_data(data_dir)

    assert set(data["source"]) == {"dqn", "baseline"}
    baseline = data[data["source"] == "baseline"].sort_values("episode")
    assert baseline["atl"].tolist() == [7.0, 8.0]


def test_write_plots_creates_pngs_from_small_csvs(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "analysis"
    data_dir.mkdir()
    pd.DataFrame(
        [
            {"episode": 0, "reward": -3.0, "atl": 9.0, "loss": 0.4},
            {"episode": 1, "reward": -2.0, "atl": 6.0, "loss": 0.2},
        ]
    ).to_csv(data_dir / "buffer_dqn.csv", index=False)
    pd.DataFrame(
        [
            {"episode": 0, "reward": -4.0, "atl": 10.0},
            {"episode": 1, "reward": -4.5, "atl": 11.0},
        ]
    ).to_csv(data_dir / "buffer_baseline.csv", index=False)

    outputs = write_plots(data_dir, output_dir)

    assert [path.name for path in outputs] == ["learning_curve.png", "atl_comparison.png"]
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
