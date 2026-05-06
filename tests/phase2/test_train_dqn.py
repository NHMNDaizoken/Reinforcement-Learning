from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from src.phase2_dqn import train_dqn as train_module


class FakeEnv:
    def __init__(self, *args, **kwargs):
        self.inter_ids = ["i0", "i1"]
        self.n_agents = 2
        self.state_dim = 3
        self.action_dim = 2
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
        return next_states, [-1.0, -2.0], self.calls >= 1, {"atl": 3.0, "throughput": 4.0}


class FakeAgent:
    def __init__(self, state_dim, action_dim):
        self.epsilon = 0.5
        self.q_network = MagicMock()

    def select_actions(self, states):
        return [0, 1]

    def remember(self, *args):
        return None

    def update(self):
        return 0.25

    def decay_epsilon(self):
        pass


def test_train_writes_transition_log_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "build_phase_map", lambda path: {"i0": [], "i1": []})
    monkeypatch.setattr(train_module, "TrafficEnv", FakeEnv)
    monkeypatch.setattr(train_module, "SharedDQNAgent", FakeAgent)
    monkeypatch.setattr(train_module.torch, "save", lambda state, path: Path(path).write_bytes(b"model"))

    csv_path = tmp_path / "buffer_dqn.csv"
    model_path = tmp_path / "best.pth"

    result = train_module.train(
        roadnet_path="roadnet.json",
        flow_path="flow.json",
        episodes=1,
        steps_per_episode=2,
        sim_steps_per_action=10,
        output_csv=str(csv_path),
        model_path=str(model_path),
    )

    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("episode,step,agent_id,state_vec")
    assert len(rows) == 3
    assert model_path.read_bytes() == b"model"
    assert result["best_reward"] == -1.5
