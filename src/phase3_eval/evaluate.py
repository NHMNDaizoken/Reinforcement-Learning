"""Evaluate DQN against MaxPressure and build the offline dataset."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2_dqn.dqn_agent import SharedDQNAgent
from src.phase1_env_baseline.max_pressure import MaxPressureBaseline
from src.phase1_env_baseline.phase_map import build_phase_map
from src.phase1_env_baseline.traffic_env import TrafficEnv


REQUIRED_COLUMNS = [
    "episode",
    "step",
    "agent_id",
    "state_vec",
    "action",
    "reward",
    "next_state_vec",
    "done",
    "atl",
    "throughput",
]
DATASET_COLUMNS = [*REQUIRED_COLUMNS, "policy"]


def _serialize_state(state: np.ndarray) -> str:
    return " ".join(f"{float(value):.6g}" for value in state.tolist())


def _transition_rows(
    episode: int,
    step: int,
    agent_ids: list[str],
    states: list[np.ndarray],
    actions: dict[str, int],
    rewards: list[float],
    next_states: list[np.ndarray],
    done: bool,
    info: dict[str, float],
    policy: str,
) -> list[dict[str, object]]:
    rows = []
    for idx, agent_id in enumerate(agent_ids):
        rows.append(
            {
                "episode": episode,
                "step": step,
                "agent_id": agent_id,
                "state_vec": _serialize_state(states[idx]),
                "action": int(actions[agent_id]),
                "reward": float(rewards[idx]),
                "next_state_vec": _serialize_state(next_states[idx]),
                "done": int(done),
                "atl": float(info.get("atl", 0.0)),
                "throughput": float(info.get("throughput", 0.0)),
                "policy": policy,
            }
        )
    return rows


def _mean_metric(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _load_dqn_model(agent: SharedDQNAgent, model_path: str) -> bool:
    path = Path(model_path)
    if not path.exists():
        print(f"Warning: DQN model not found at {path}; using untrained weights.")
        return False

    try:
        state_dict = torch.load(path, map_location=agent.device)
        state_dict = _normalize_checkpoint_state_dict(state_dict)
        agent.q_network.load_state_dict(state_dict)
        agent.target_network.load_state_dict(agent.q_network.state_dict())
        agent.q_network.eval()
        agent.target_network.eval()
        agent.epsilon = 0.0
    except Exception as exc:  # pragma: no cover - exercised by integration runs.
        print(f"Warning: could not load DQN model from {path}: {exc}")
        print("Using untrained weights so evaluation can still run.")
        return False
    return True


def _normalize_checkpoint_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Accept current DenseLayer checkpoints and older nn.Sequential checkpoints."""
    if "input_layer.weight" in state_dict:
        return state_dict

    sequential_keys = {
        "model.0.weight",
        "model.0.bias",
        "model.2.weight",
        "model.2.bias",
        "model.4.weight",
        "model.4.bias",
    }
    if not sequential_keys.issubset(state_dict):
        return state_dict

    return {
        "input_layer.weight": state_dict["model.0.weight"].T.contiguous(),
        "input_layer.bias": state_dict["model.0.bias"],
        "hidden_layer.weight": state_dict["model.2.weight"].T.contiguous(),
        "hidden_layer.bias": state_dict["model.2.bias"],
        "output_layer.weight": state_dict["model.4.weight"].T.contiguous(),
        "output_layer.bias": state_dict["model.4.bias"],
    }


def _run_dqn(
    roadnet_path: str,
    flow_path: str,
    model_path: str,
    episodes: int,
    steps_per_episode: int,
    sim_steps_per_action: int,
) -> dict[str, object]:
    phase_map = build_phase_map(roadnet_path)
    env = TrafficEnv(
        roadnet_path=roadnet_path,
        flow_path=flow_path,
        phase_map=phase_map,
        sim_steps_per_action=sim_steps_per_action,
    )
    agent = SharedDQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    model_loaded = _load_dqn_model(agent, model_path)

    atl_values: list[float] = []
    throughput_values: list[float] = []
    rows: list[dict[str, object]] = []

    for episode in range(episodes):
        states = env.reset()
        final_info = {"atl": 0.0, "throughput": 0.0}

        for step in range(steps_per_episode):
            action_values = agent.select_actions(states, training=False)
            actions = {
                agent_id: int(action_values[idx])
                for idx, agent_id in enumerate(env.inter_ids)
            }
            next_states, rewards, done, info = env.step(actions)
            final_info = info
            rows.extend(
                _transition_rows(
                    episode,
                    step,
                    env.inter_ids,
                    states,
                    actions,
                    rewards,
                    next_states,
                    done,
                    info,
                    "dqn",
                )
            )
            states = next_states
            if done:
                break

        atl_values.append(float(final_info.get("atl", 0.0)))
        throughput_values.append(float(final_info.get("throughput", 0.0)))

    return {
        "atl": _mean_metric(atl_values),
        "throughput": _mean_metric(throughput_values),
        "rows": rows,
        "model_loaded": model_loaded,
    }


def _run_baseline(
    roadnet_path: str,
    flow_path: str,
    episodes: int,
    steps_per_episode: int,
    sim_steps_per_action: int,
) -> dict[str, object]:
    phase_map = build_phase_map(roadnet_path)
    env = TrafficEnv(
        roadnet_path=roadnet_path,
        flow_path=flow_path,
        phase_map=phase_map,
        sim_steps_per_action=sim_steps_per_action,
    )

    atl_values: list[float] = []
    throughput_values: list[float] = []
    rows: list[dict[str, object]] = []

    for episode in range(episodes):
        states = env.reset()
        controller = MaxPressureBaseline(env.engine, phase_map)
        final_info = {"atl": 0.0, "throughput": 0.0}

        for step in range(steps_per_episode):
            actions = controller.select_actions()
            next_states, rewards, done, info = env.step(actions)
            final_info = info
            rows.extend(
                _transition_rows(
                    episode,
                    step,
                    env.inter_ids,
                    states,
                    actions,
                    rewards,
                    next_states,
                    done,
                    info,
                    "baseline",
                )
            )
            states = next_states
            if done:
                break

        atl_values.append(float(final_info.get("atl", 0.0)))
        throughput_values.append(float(final_info.get("throughput", 0.0)))

    return {
        "atl": _mean_metric(atl_values),
        "throughput": _mean_metric(throughput_values),
        "rows": rows,
    }


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: str, policy: str) -> list[dict[str, object]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []

    rows: list[dict[str, object]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        for raw_row in reader:
            row = {column: raw_row[column] for column in REQUIRED_COLUMNS}
            row["policy"] = raw_row.get("policy") or policy
            rows.append(row)
    return rows


def _write_offline_db(db_path: str, rows: list[dict[str, object]]) -> None:
    output = Path(db_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output) as conn:
        conn.execute("DROP TABLE IF EXISTS transitions")
        conn.execute(
            """
            CREATE TABLE transitions (
                episode INTEGER,
                step INTEGER,
                agent_id TEXT,
                state_vec TEXT,
                action INTEGER,
                reward REAL,
                next_state_vec TEXT,
                done INTEGER,
                atl REAL,
                throughput REAL,
                policy TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO transitions (
                episode, step, agent_id, state_vec, action, reward,
                next_state_vec, done, atl, throughput, policy
            )
            VALUES (
                :episode, :step, :agent_id, :state_vec, :action, :reward,
                :next_state_vec, :done, :atl, :throughput, :policy
            )
            """,
            rows,
        )
        conn.commit()


def _merge_offline_dataset(
    dqn_csv: str,
    baseline_csv: str,
    db_path: str,
) -> dict[str, int]:
    rows = [
        *_read_csv_rows(dqn_csv, "dqn"),
        *_read_csv_rows(baseline_csv, "baseline"),
    ]
    _write_offline_db(db_path, rows)
    counts: dict[str, int] = {}
    for row in rows:
        policy = str(row["policy"])
        counts[policy] = counts.get(policy, 0) + 1
    return counts


def evaluate(
    roadnet_path: str,
    flow_path: str,
    model_path: str,
    episodes: int,
    steps_per_episode: int,
    sim_steps_per_action: int,
    baseline_csv: str,
    dqn_csv: str,
    offline_db: str,
) -> dict[str, object]:
    dqn = _run_dqn(
        roadnet_path,
        flow_path,
        model_path,
        episodes,
        steps_per_episode,
        sim_steps_per_action,
    )
    baseline = _run_baseline(
        roadnet_path,
        flow_path,
        episodes,
        steps_per_episode,
        sim_steps_per_action,
    )

    _write_csv(dqn_csv, dqn["rows"])
    _write_csv(baseline_csv, baseline["rows"])
    db_counts = _merge_offline_dataset(dqn_csv, baseline_csv, offline_db)

    return {
        "dqn": dqn,
        "baseline": baseline,
        "db_counts": db_counts,
    }


def _print_summary(results: dict[str, object], flow_path: str) -> None:
    dqn = results["dqn"]
    baseline = results["baseline"]
    db_counts = results["db_counts"]

    print("")
    print(f"DQN vs Baseline summary for {flow_path}")
    print("policy      atl        throughput")
    print(
        "dqn         {atl:10.3f} {throughput:10.0f}".format(
            atl=float(dqn["atl"]),
            throughput=float(dqn["throughput"]),
        )
    )
    print(
        "baseline    {atl:10.3f} {throughput:10.0f}".format(
            atl=float(baseline["atl"]),
            throughput=float(baseline["throughput"]),
        )
    )
    print(f"DQN model loaded: {bool(dqn['model_loaded'])}")
    print(f"Offline DB rows by policy: {db_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate shared DQN against MaxPressure and build offline data."
    )
    parser.add_argument("--roadnet", default="configs/roadnet.json")
    parser.add_argument("--flow", default="configs/flow_medium.json")
    parser.add_argument("--model", default="models/best.pth")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--sim-steps-per-action", type=int, default=10)
    parser.add_argument("--baseline-csv", default="data/buffer_baseline.csv")
    parser.add_argument("--dqn-csv", default="data/buffer_dqn.csv")
    parser.add_argument("--offline-db", default="data/offline_dataset.db")
    args = parser.parse_args()

    results = evaluate(
        roadnet_path=args.roadnet,
        flow_path=args.flow,
        model_path=args.model,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        sim_steps_per_action=args.sim_steps_per_action,
        baseline_csv=args.baseline_csv,
        dqn_csv=args.dqn_csv,
        offline_db=args.offline_db,
    )
    _print_summary(results, args.flow)


if __name__ == "__main__":
    main()
