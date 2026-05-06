"""Train a shared DQN controller for the 3x3 traffic grid."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2_dqn.dqn_agent import SharedDQNAgent
from src.phase1_env_baseline.phase_map import build_phase_map
from src.phase1_env_baseline.traffic_env import TrafficEnv


def _serialize_state(state: np.ndarray) -> str:
    return " ".join(f"{float(value):.6g}" for value in state.tolist())


def _episode_rows(
    episode: int,
    step: int,
    agent_ids: list[str],
    states: list[np.ndarray],
    actions: dict[str, int],
    rewards: list[float],
    next_states: list[np.ndarray],
    done: bool,
    info: dict[str, float],
) -> list[dict[str, object]]:
    rows = []
    for idx, agent_id in enumerate(agent_ids):
        rows.append(
            {
                "episode": episode,
                "step": step,
                "agent_id": agent_id,
                "state_vec": _serialize_state(states[idx]),
                "action": actions[agent_id],
                "reward": float(rewards[idx]),
                "next_state_vec": _serialize_state(next_states[idx]),
                "done": int(done),
                "atl": float(info.get("atl", 0.0)),
                "throughput": float(info.get("throughput", 0.0)),
            }
        )
    return rows


def _select_flow(
    episode: int,
    flows: list[str],
    mode: str,
    curriculum_interval: int,
) -> str:
    if mode == "single":
        return flows[0]

    if mode == "random":
        return random.choice(flows)

    if mode == "curriculum":
        max_unlocked_idx = min(episode // curriculum_interval, len(flows) - 1)

        if max_unlocked_idx == 0:
            flow_idx = 0
        else:
            # 75% train flow khó nhất hiện tại, 25% ôn lại flow cũ
            if random.random() < 0.75:
                flow_idx = max_unlocked_idx
            else:
                flow_idx = random.randint(0, max_unlocked_idx - 1)

        return flows[flow_idx]

    return flows[0]


def _make_env(
    roadnet_path: str,
    flow_path: str,
    phase_map: dict,
    sim_steps_per_action: int,
) -> TrafficEnv:
    return TrafficEnv(
        roadnet_path=roadnet_path,
        flow_path=flow_path,
        phase_map=phase_map,
        sim_steps_per_action=sim_steps_per_action,
    )


def train(
    roadnet_path: str,
    flows: list[str],
    mode: str,
    curriculum_interval: int,
    episodes: int,
    steps_per_episode: int,
    sim_steps_per_action: int,
    output_csv: str,
    model_path: str,
) -> dict[str, float]:
    if not flows:
        raise ValueError("flows must contain at least one flow config")

    if curriculum_interval <= 0:
        raise ValueError("curriculum_interval must be > 0")

    if episodes <= 0:
        raise ValueError("episodes must be > 0")

    if steps_per_episode <= 0:
        raise ValueError("steps_per_episode must be > 0")

    phase_map = build_phase_map(roadnet_path)

    current_flow_path = flows[0]
    print(f"\n[INFO] Initializing environment with flow: {current_flow_path}")

    env = _make_env(
        roadnet_path=roadnet_path,
        flow_path=current_flow_path,
        phase_map=phase_map,
        sim_steps_per_action=sim_steps_per_action,
    )

    agent = SharedDQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    model_output = Path(model_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)

    training_log_path = csv_path.parent / "training_log.csv"

    buffer_fieldnames = [
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

    training_log_fields = [
        "episode",
        "flow_scenario",
        "avg_reward",
        "atl",
        "throughput",
        "loss",
        "epsilon",
    ]

    best_reward = float("-inf")
    last_loss = 0.0
    global_step = 0
    target_update_freq = 1000

    with (
        csv_path.open("w", newline="", encoding="utf-8") as buffer_file,
        training_log_path.open("w", newline="", encoding="utf-8") as log_file,
    ):
        buffer_writer = csv.DictWriter(buffer_file, fieldnames=buffer_fieldnames)
        buffer_writer.writeheader()

        log_writer = csv.DictWriter(log_file, fieldnames=training_log_fields)
        log_writer.writeheader()

        for episode in range(episodes):
            new_flow_path = _select_flow(
                episode=episode,
                flows=flows,
                mode=mode,
                curriculum_interval=curriculum_interval,
            )

            # Đổi flow bằng cách tạo lại env, nhưng giữ nguyên agent và replay buffer.
            if new_flow_path != current_flow_path:
                current_flow_path = new_flow_path
                print(
                    f"\n[INFO] Episode {episode}: switching flow -> "
                    f"{Path(current_flow_path).name}\n"
                )

                env = _make_env(
                    roadnet_path=roadnet_path,
                    flow_path=current_flow_path,
                    phase_map=phase_map,
                    sim_steps_per_action=sim_steps_per_action,
                )

            states = env.reset()
            episode_reward = 0.0
            final_info = {"atl": 0.0, "throughput": 0.0}
            actual_steps = 0

            for step in range(steps_per_episode):
                actual_steps = step + 1
                global_step += 1

                action_values = agent.select_actions(states)

                actions = {
                    agent_id: int(action_values[idx])
                    for idx, agent_id in enumerate(env.inter_ids)
                }

                next_states, rewards, done, info = env.step(actions)
                final_info = info

                for idx, agent_id in enumerate(env.inter_ids):
                    agent.remember(
                        states[idx],
                        actions[agent_id],
                        rewards[idx],
                        next_states[idx],
                        done,
                    )

                loss = agent.update()
                if loss is not None:
                    last_loss = float(loss)

                if (
                    global_step % target_update_freq == 0
                    and hasattr(agent, "update_target_network")
                ):
                    agent.update_target_network()

                buffer_writer.writerows(
                    _episode_rows(
                        episode=episode,
                        step=step,
                        agent_ids=env.inter_ids,
                        states=states,
                        actions=actions,
                        rewards=rewards,
                        next_states=next_states,
                        done=done,
                        info=info,
                    )
                )

                episode_reward += float(sum(rewards))
                states = next_states

                if done:
                    break

            agent.decay_epsilon()

            # Average reward đúng theo số step thật sự đã chạy.
            avg_reward = episode_reward / max(1, actual_steps * env.n_agents)

            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(agent.q_network.state_dict(), model_output)

            log_writer.writerow(
                {
                    "episode": episode,
                    "flow_scenario": Path(current_flow_path).name,
                    "avg_reward": round(avg_reward, 4),
                    "atl": round(float(final_info.get("atl", 0.0)), 4),
                    "throughput": round(float(final_info.get("throughput", 0.0)), 1),
                    "loss": round(last_loss, 6),
                    "epsilon": round(agent.epsilon, 4),
                }
            )
            log_file.flush()

            print(
                "episode={episode} flow={flow_name} avg_reward={avg_reward:.3f} "
                "eps={epsilon:.3f} loss={loss:.4f} atl={atl:.1f} "
                "throughput={throughput:.0f}".format(
                    episode=episode,
                    flow_name=Path(current_flow_path).name,
                    avg_reward=avg_reward,
                    epsilon=agent.epsilon,
                    loss=last_loss,
                    atl=float(final_info.get("atl", 0.0)),
                    throughput=float(final_info.get("throughput", 0.0)),
                )
            )

    return {"best_reward": best_reward, "last_loss": last_loss}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train shared DQN traffic controller."
    )

    parser.add_argument("--roadnet", default="configs/roadnet.json")

    parser.add_argument(
        "--flows",
        nargs="+",
        default=["configs/flow_medium.json"],
        help="List of flow config files for curriculum learning.",
    )

    parser.add_argument(
        "--mode",
        choices=["single", "curriculum", "random"],
        default="single",
        help="Training mode.",
    )

    parser.add_argument(
        "--curriculum-interval",
        type=int,
        default=200,
        help="Number of episodes per curriculum phase.",
    )

    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--sim-steps-per-action", type=int, default=10)
    parser.add_argument("--output-csv", default="data/buffer_dqn.csv")
    parser.add_argument("--model-path", default="models/best_curriculum.pth")

    args = parser.parse_args()

    train(
        roadnet_path=args.roadnet,
        flows=args.flows,
        mode=args.mode,
        curriculum_interval=args.curriculum_interval,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        sim_steps_per_action=args.sim_steps_per_action,
        output_csv=args.output_csv,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()