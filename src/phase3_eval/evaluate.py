"""Evaluate multiple DQNs against MaxPressure and build the offline dataset."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2_dqn.dqn_agent import SharedDQNAgent
from src.phase1_env_baseline.max_pressure import MaxPressureBaseline
from src.phase1_env_baseline.phase_map import build_phase_map
from src.phase1_env_baseline.traffic_env import TrafficEnv


BASE_COLUMNS = [
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
EXTRA_COLUMNS = [
    "is_decision_phase",
    "is_clearance_phase",
    "sim_time",
    "completed",
    "active",
    "generated",
    "completion_rate",
]
REQUIRED_COLUMNS = [*BASE_COLUMNS, *EXTRA_COLUMNS]
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
                "is_decision_phase": int(bool(info.get("is_decision_phase", True))),
                "is_clearance_phase": int(bool(info.get("is_clearance_phase", False))),
                "sim_time": float(info.get("sim_time", 0.0)),
                "completed": float(info.get("completed", info.get("throughput", 0.0))),
                "active": float(info.get("active", 0.0)),
                "generated": float(info.get("generated", 0.0)),
                "completion_rate": float(info.get("completion_rate", 0.0)),
                "policy": policy,
            }
        )
    return rows


def _mean_metric(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _load_dqn_model(agent: SharedDQNAgent, model_path: str) -> None:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"DQN model not found: {path}")

    try:
        state_dict = torch.load(path, map_location=agent.device)
        state_dict = _normalize_checkpoint_state_dict(state_dict)
        agent.q_network.load_state_dict(state_dict)
        target_state = (
            agent.q_network.state_dict()
            if hasattr(agent.q_network, "state_dict")
            else state_dict
        )
        agent.target_network.load_state_dict(target_state)
        agent.q_network.eval()
        agent.target_network.eval()
        agent.epsilon = 0.0
    except Exception as exc:  # pragma: no cover - exercised by integration runs.
        raise RuntimeError(f"could not load DQN model from {path}: {exc}") from exc


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
        "input_layer.weight": state_dict["model.0.weight"],
        "input_layer.bias": state_dict["model.0.bias"],
        "hidden_layer.weight": state_dict["model.2.weight"],
        "hidden_layer.bias": state_dict["model.2.bias"],
        "output_layer.weight": state_dict["model.4.weight"],
        "output_layer.bias": state_dict["model.4.bias"],
    }


def _run_dqn(
    roadnet_path: str,
    flow_path: str,
    model_path: str,
    episodes: int,
    steps_per_episode: int,
    sim_steps_per_action: int,
    policy_name: str,
) -> dict[str, object]:
    phase_map = build_phase_map(roadnet_path)
    decision_duration = steps_per_episode * sim_steps_per_action
    env = TrafficEnv(
        roadnet_path=roadnet_path,
        flow_path=flow_path,
        phase_map=phase_map,
        sim_steps_per_action=sim_steps_per_action,
        decision_duration=decision_duration,
    )
    agent = SharedDQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    _load_dqn_model(agent, model_path)
    agent.epsilon = 0.0
    agent.q_network.eval()
    agent.target_network.eval()

    atl_values: list[float] = []
    completed_values: list[float] = []
    active_values: list[float] = []
    generated_values: list[float] = []
    completion_rate_values: list[float] = []
    rows: list[dict[str, object]] = []

    print(f"\n🚀 Đang đánh giá DQN Model: {Path(model_path).name} ({episodes} episodes)...")
    for episode in range(episodes):
        start_time = time.time()
        states = env.reset()
        final_info: dict[str, float] = {
            "atl": 0.0,
            "throughput": 0.0,
            "completed": 0.0,
            "active": 0.0,
            "generated": 0.0,
            "completion_rate": 0.0,
        }

        for step in range(steps_per_episode):
            action_values = agent.select_actions(states, training=False)
            actions = {
                agent_id: int(action_values[idx])
                for idx, agent_id in enumerate(env.inter_ids)
            }
            next_states, rewards, done, info = env.step(actions)
            final_info = info
            if bool(info.get("is_decision_phase", True)):
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
                        policy_name,
                    )
                )
            states = next_states
            if done:
                break

        for _ in range(_clearance_action_steps(env)):
            if _clearance_complete(final_info):
                break
            action_values = agent.select_actions(states, training=False)
            actions = {
                agent_id: int(action_values[idx])
                for idx, agent_id in enumerate(env.inter_ids)
            }
            states, _, _, final_info = env.step(actions)

        atl = float(final_info.get("atl", 0.0))
        completed = float(final_info.get("completed", final_info.get("throughput", 0.0)))
        atl_values.append(atl)
        completed_values.append(completed)
        active_values.append(float(final_info.get("active", 0.0)))
        generated_values.append(float(final_info.get("generated", 0.0)))
        completion_rate_values.append(float(final_info.get("completion_rate", 0.0)))
        
        elapsed_time = time.time() - start_time
        print(
            f"  [{policy_name}] Episode {episode + 1}/{episodes} | "
            f"ATL: {atl:6.2f}s | Completed: {completed:4.0f} | "
            f"Active: {float(final_info.get('active', 0.0)):4.0f} | "
            f"Time: {elapsed_time:.2f}s"
        )

    return {
        "atl": _mean_metric(atl_values),
        "throughput": _mean_metric(completed_values),
        "completed": _mean_metric(completed_values),
        "active": _mean_metric(active_values),
        "generated": _mean_metric(generated_values),
        "completion_rate": _mean_metric(completion_rate_values),
        "rows": rows,
        "model_loaded": True,
    }


def _run_baseline(
    roadnet_path: str,
    flow_path: str,
    episodes: int,
    steps_per_episode: int,
    sim_steps_per_action: int,
) -> dict[str, object]:
    phase_map = build_phase_map(roadnet_path)
    decision_duration = steps_per_episode * sim_steps_per_action
    env = TrafficEnv(
        roadnet_path=roadnet_path,
        flow_path=flow_path,
        phase_map=phase_map,
        sim_steps_per_action=sim_steps_per_action,
        decision_duration=decision_duration,
    )

    atl_values: list[float] = []
    completed_values: list[float] = []
    active_values: list[float] = []
    generated_values: list[float] = []
    completion_rate_values: list[float] = []
    rows: list[dict[str, object]] = []

    print(f"\n⏱️ Đang đánh giá MaxPressure Baseline ({episodes} episodes)...")
    for episode in range(episodes):
        start_time = time.time()
        states = env.reset()
        controller = MaxPressureBaseline(env.engine, phase_map)
        final_info: dict[str, float] = {
            "atl": 0.0,
            "throughput": 0.0,
            "completed": 0.0,
            "active": 0.0,
            "generated": 0.0,
            "completion_rate": 0.0,
        }

        for step in range(steps_per_episode):
            actions = controller.select_actions()
            next_states, rewards, done, info = env.step(actions)
            final_info = info
            if bool(info.get("is_decision_phase", True)):
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

        for _ in range(_clearance_action_steps(env)):
            if _clearance_complete(final_info):
                break
            states, _, _, final_info = env.step(controller.select_actions())

        atl = float(final_info.get("atl", 0.0))
        completed = float(final_info.get("completed", final_info.get("throughput", 0.0)))
        atl_values.append(atl)
        completed_values.append(completed)
        active_values.append(float(final_info.get("active", 0.0)))
        generated_values.append(float(final_info.get("generated", 0.0)))
        completion_rate_values.append(float(final_info.get("completion_rate", 0.0)))
        
        elapsed_time = time.time() - start_time
        print(
            f"  [Baseline] Episode {episode + 1}/{episodes} | "
            f"ATL: {atl:6.2f}s | Completed: {completed:4.0f} | "
            f"Active: {float(final_info.get('active', 0.0)):4.0f} | "
            f"Time: {elapsed_time:.2f}s"
        )

    return {
        "atl": _mean_metric(atl_values),
        "throughput": _mean_metric(completed_values),
        "completed": _mean_metric(completed_values),
        "active": _mean_metric(active_values),
        "generated": _mean_metric(generated_values),
        "completion_rate": _mean_metric(completion_rate_values),
        "rows": rows,
    }


def _clearance_action_steps(env: TrafficEnv) -> int:
    return int(getattr(env, "clearance_action_steps", 0))


def _clearance_complete(info: dict[str, object]) -> bool:
    return float(info.get("generated", 0.0)) > 0 and float(info.get("active", 0.0)) <= 0


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
        missing = [column for column in BASE_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        for raw_row in reader:
            row = {column: raw_row[column] for column in BASE_COLUMNS}
            for column in EXTRA_COLUMNS:
                row[column] = raw_row.get(column, 0)
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
                is_decision_phase INTEGER,
                is_clearance_phase INTEGER,
                sim_time REAL,
                completed REAL,
                active REAL,
                generated REAL,
                completion_rate REAL,
                policy TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO transitions (
                episode, step, agent_id, state_vec, action, reward,
                next_state_vec, done, atl, throughput, is_decision_phase,
                is_clearance_phase, sim_time, completed, active, generated,
                completion_rate, policy
            )
            VALUES (
                :episode, :step, :agent_id, :state_vec, :action, :reward,
                :next_state_vec, :done, :atl, :throughput, :is_decision_phase,
                :is_clearance_phase, :sim_time, :completed, :active, :generated,
                :completion_rate, :policy
            )
            """,
            rows,
        )
        conn.commit()


def _merge_offline_dataset_multi(
    dqn_csvs: list[str],
    baseline_csv: str,
    db_path: str,
) -> dict[str, int]:
    rows = [*_read_csv_rows(baseline_csv, "baseline")]
    for dqn_csv in dqn_csvs:
        policy_name = Path(dqn_csv).stem.replace("buffer_", "dqn_")
        rows.extend(_read_csv_rows(dqn_csv, policy_name))
        
    _write_offline_db(db_path, rows)
    counts: dict[str, int] = {}
    for row in rows:
        policy = str(row["policy"])
        counts[policy] = counts.get(policy, 0) + 1
    return counts


def evaluate_multiple(
    roadnet_path: str,
    flow_path: str,
    model_paths: list[str],
    episodes: int,
    steps_per_episode: int,
    sim_steps_per_action: int,
    baseline_csv: str,
    base_dqn_csv: str,
    offline_db: str,
) -> dict[str, object]:
    
    baseline = _run_baseline(
        roadnet_path,
        flow_path,
        episodes,
        steps_per_episode,
        sim_steps_per_action,
    )
    _write_csv(baseline_csv, baseline["rows"])

    models_results = {}
    dqn_csvs = []

    for model_path in model_paths:
        model_name = Path(model_path).stem
        policy_name = f"dqn_{model_name}"
        
        dqn = _run_dqn(
            roadnet_path,
            flow_path,
            model_path,
            episodes,
            steps_per_episode,
            sim_steps_per_action,
            policy_name,
        )
        models_results[model_name] = dqn
        
        csv_path = str(Path(base_dqn_csv).parent / f"buffer_{model_name}.csv")
        _write_csv(csv_path, dqn["rows"])
        dqn_csvs.append(csv_path)

    db_counts = _merge_offline_dataset_multi(dqn_csvs, baseline_csv, offline_db)

    return {
        "models": models_results,
        "baseline": baseline,
        "db_counts": db_counts,
    }


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
    """Backward-compatible single-model evaluation wrapper."""
    baseline = _run_baseline(
        roadnet_path,
        flow_path,
        episodes,
        steps_per_episode,
        sim_steps_per_action,
    )
    _write_csv(baseline_csv, baseline["rows"])

    dqn = _run_dqn(
        roadnet_path,
        flow_path,
        model_path,
        episodes,
        steps_per_episode,
        sim_steps_per_action,
        "dqn",
    )
    _write_csv(dqn_csv, dqn["rows"])

    db_counts = _merge_offline_dataset_multi([dqn_csv], baseline_csv, offline_db)
    return {"dqn": dqn, "baseline": baseline, "db_counts": db_counts}


def _print_summary_multi(results: dict[str, object], flow_path: str) -> None:
    models_res = results["models"]
    baseline = results["baseline"]
    db_counts = results["db_counts"]

    print(f"\n{'='*55}")
    print(f"📊 SUMMARY REPORT FOR FLOW: {Path(flow_path).name}")
    print(f"{'='*55}")
    print(
        f"{'Policy Name':<25} | {'ATL (s)':<10} | {'Completed':<10} | "
        f"{'Active':<8} | {'Generated':<10} | {'Rate':<6}"
    )
    print(f"{'-'*86}")
    
    print(
        f"{'baseline':<25} | {float(baseline['atl']):<10.3f} | "
        f"{float(baseline['completed']):<10.0f} | {float(baseline['active']):<8.0f} | "
        f"{float(baseline['generated']):<10.0f} | {float(baseline['completion_rate']):<6.3f}"
    )
    
    for model_name, res in models_res.items():
        policy_name = f"dqn_{model_name}"
        print(
            f"{policy_name:<25} | {float(res['atl']):<10.3f} | "
            f"{float(res['completed']):<10.0f} | {float(res['active']):<8.0f} | "
            f"{float(res['generated']):<10.0f} | {float(res['completion_rate']):<6.3f}"
        )
        
    print(f"{'-'*86}")
    print(f"Offline DB rows count: {db_counts}")
    print(f"{'='*55}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DQNs against MaxPressure on multiple traffic flows."
    )

    parser.add_argument(
        "--roadnet",
        default="configs/roadnet.json",
        help="Roadnet used for both training and evaluation.",
    )

    parser.add_argument(
        "--flows",
        nargs="+",
        default=[
            "configs/flow_low_flat.json",
            "configs/flow_medium_flat.json",
            "configs/flow_high_flat.json",
            "configs/flow_low_peak.json",
            "configs/flow_medium_peak.json",
            "configs/flow_high_peak.json",
        ],
        help="List of flow files to evaluate, PressLight-style.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["models/best_curriculum.pth"],
        help="List of trained DQN model files.",
    )

    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--sim-steps-per-action", type=int, default=10)

    parser.add_argument("--output-dir", default="data/eval")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suite_rows = []

    for flow_path in args.flows:
        flow_name = Path(flow_path).stem

        print(f"\n\n==============================")
        print(f"Evaluating flow: {flow_name}")
        print(f"==============================")

        results = evaluate_multiple(
            roadnet_path=args.roadnet,
            flow_path=flow_path,
            model_paths=args.models,
            episodes=args.episodes,
            steps_per_episode=args.steps,
            sim_steps_per_action=args.sim_steps_per_action,
            baseline_csv=str(output_dir / f"buffer_baseline_{flow_name}.csv"),
            base_dqn_csv=str(output_dir / f"buffer_dqn_{flow_name}.csv"),
            offline_db=str(output_dir / f"offline_dataset_{flow_name}.db"),
        )

        _print_summary_multi(results, flow_path)

        baseline = results["baseline"]
        suite_rows.append(
            {
                "flow": flow_name,
                "policy": "baseline",
                "atl": float(baseline["atl"]),
                "completed": float(baseline["completed"]),
                "active": float(baseline["active"]),
                "generated": float(baseline["generated"]),
                "completion_rate": float(baseline["completion_rate"]),
            }
        )

        for model_name, res in results["models"].items():
            suite_rows.append(
                {
                    "flow": flow_name,
                    "policy": f"dqn_{model_name}",
                    "atl": float(res["atl"]),
                    "completed": float(res["completed"]),
                    "active": float(res["active"]),
                    "generated": float(res["generated"]),
                    "completion_rate": float(res["completion_rate"]),
                }
            )

    summary_path = output_dir / "summary_all_flows.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "flow",
                "policy",
                "atl",
                "completed",
                "active",
                "generated",
                "completion_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(suite_rows)

    print(f"\nSaved benchmark summary to: {summary_path}")


if __name__ == "__main__":
    main()
