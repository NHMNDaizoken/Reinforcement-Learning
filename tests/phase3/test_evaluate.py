import csv
import sqlite3

import numpy as np
import torch

from src.phase3_eval import evaluate as evaluate_module


class FakeEnv:
    def __init__(self, *args, **kwargs):
        self.inter_ids = ["i0", "i1"]
        self.n_agents = 2
        self.state_dim = 3
        self.action_dim = 2
        self.engine = object()
        self.calls = 0

    def reset(self):
        self.calls = 0
        return [
            np.array([1.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
        ]

    def step(self, actions):
        self.calls += 1
        next_states = [
            np.array([0.5, 0.0, 1.0], dtype=np.float32),
            np.array([0.0, 0.5, 1.0], dtype=np.float32),
        ]
        rewards = [-float(actions["i0"] + 1), -float(actions["i1"] + 1)]
        done = self.calls >= 1
        return next_states, rewards, done, {"atl": 12.0, "throughput": 5.0}


class FakeAgent:
    def __init__(self, state_dim, action_dim):
        self.device = "cpu"
        self.epsilon = 1.0
        self.q_network = self
        self.target_network = self

    def load_state_dict(self, state_dict):
        return None

    def eval(self):
        return None

    def select_actions(self, states, training=False):
        return [0, 1]


class FakeBaseline:
    def __init__(self, engine, phase_map):
        self.phase_map = phase_map

    def select_actions(self):
        return {"i0": 1, "i1": 0}


def test_evaluate_writes_policy_csvs_and_offline_db(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_module, "build_phase_map", lambda path: {"i0": [], "i1": []})
    monkeypatch.setattr(evaluate_module, "TrafficEnv", FakeEnv)
    monkeypatch.setattr(evaluate_module, "SharedDQNAgent", FakeAgent)
    monkeypatch.setattr(evaluate_module, "MaxPressureBaseline", FakeBaseline)
    monkeypatch.setattr(evaluate_module.torch, "load", lambda path, map_location=None: {})

    dqn_csv = tmp_path / "buffer_dqn.csv"
    with dqn_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=evaluate_module.REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "episode": 0,
                "step": 0,
                "agent_id": "i0",
                "state_vec": "0 1",
                "action": 0,
                "reward": -1.0,
                "next_state_vec": "1 0",
                "done": 1,
                "atl": 10.0,
                "throughput": 4.0,
            }
        )

    baseline_csv = tmp_path / "buffer_baseline.csv"
    offline_db = tmp_path / "offline_dataset.db"
    model_path = tmp_path / "best.pth"
    model_path.write_bytes(b"fake")

    result = evaluate_module.evaluate(
        roadnet_path="roadnet.json",
        flow_path="flow.json",
        model_path=str(model_path),
        episodes=1,
        steps_per_episode=2,
        sim_steps_per_action=10,
        baseline_csv=str(baseline_csv),
        dqn_csv=str(dqn_csv),
        offline_db=str(offline_db),
    )

    assert result["dqn"]["atl"] == 12.0
    assert result["baseline"]["throughput"] == 5.0
    assert result["db_counts"] == {"dqn": 2, "baseline": 2}

    dqn_rows = list(csv.DictReader(dqn_csv.open(encoding="utf-8")))
    assert len(dqn_rows) == 2
    assert all(row["policy"] == "dqn" for row in dqn_rows)
    assert dqn_rows[0]["state_vec"] != "0 1"

    baseline_rows = list(csv.DictReader(baseline_csv.open(encoding="utf-8")))
    assert baseline_rows[0]["policy"] == "baseline"
    assert evaluate_module.REQUIRED_COLUMNS == [
        column for column in evaluate_module.REQUIRED_COLUMNS if column in baseline_rows[0]
    ]

    with sqlite3.connect(offline_db) as connection:
        columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(transitions)").fetchall()
        ]
        policies = [
            row[0]
            for row in connection.execute(
                "SELECT policy FROM transitions ORDER BY policy"
            ).fetchall()
        ]

    assert all(column in columns for column in evaluate_module.REQUIRED_COLUMNS)
    assert "policy" in columns
    assert policies == ["baseline", "baseline", "dqn", "dqn"]


def test_normalize_checkpoint_state_dict_accepts_old_sequential_keys():
    old_state = {
        "model.0.weight": torch.zeros((128, 26)),
        "model.0.bias": torch.zeros(128),
        "model.2.weight": torch.zeros((64, 128)),
        "model.2.bias": torch.zeros(64),
        "model.4.weight": torch.zeros((2, 64)),
        "model.4.bias": torch.zeros(2),
    }

    normalized = evaluate_module._normalize_checkpoint_state_dict(old_state)

    assert normalized["input_layer.weight"].shape == (26, 128)
    assert normalized["hidden_layer.weight"].shape == (128, 64)
    assert normalized["output_layer.weight"].shape == (64, 2)
