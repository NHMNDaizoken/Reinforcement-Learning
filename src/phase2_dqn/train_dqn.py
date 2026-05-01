"""Train a shared DQN controller for the 3x3 traffic grid."""

from __future__ import annotations

import argparse
import csv
import os
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


def train(
    roadnet_path: str,
    flow_path: str,
    episodes: int,
    steps_per_episode: int,
    sim_steps_per_action: int,
    output_csv: str,
    model_path: str,
) -> dict[str, float]:
    phase_map = build_phase_map(roadnet_path)
    env = TrafficEnv(
        roadnet_path=roadnet_path,
        flow_path=flow_path,
        phase_map=phase_map,
        sim_steps_per_action=sim_steps_per_action,
    )
    agent = SharedDQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    model_output = Path(model_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)

    best_reward = float("-inf")
    last_loss = 0.0
    fieldnames = [
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

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for episode in range(episodes):
            states = env.reset()
            episode_reward = 0.0
            final_info = {"atl": 0.0, "throughput": 0.0}

            for step in range(steps_per_episode):
                action_values = agent.select_actions(states)
                actions = {
                    agent_id: int(action_values[idx])
                    for idx, agent_id in enumerate(env.inter_ids)
                }
                next_states, rewards, done, info = env.step(actions)
                final_info = info

                for idx in range(env.n_agents):
                    agent.remember(
                        states[idx],
                        actions[env.inter_ids[idx]],
                        rewards[idx],
                        next_states[idx],
                        done,
                    )

                loss = agent.update()
                if loss is not None:
                    last_loss = float(loss)
                agent.decay_epsilon()

                writer.writerows(
                    _episode_rows(
                        episode,
                        step,
                        env.inter_ids,
                        states,
                        actions,
                        rewards,
                        next_states,
                        done,
                        info,
                    )
                )
                episode_reward += float(sum(rewards))
                states = next_states

                if done:
                    break

            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(agent.q_network.state_dict(), model_output)

            print(
                "episode={episode} reward={reward:.3f} epsilon={epsilon:.4f} "
                "loss={loss:.6f} atl={atl:.3f} throughput={throughput:.0f}".format(
                    episode=episode,
                    reward=episode_reward,
                    epsilon=agent.epsilon,
                    loss=last_loss,
                    atl=float(final_info.get("atl", 0.0)),
                    throughput=float(final_info.get("throughput", 0.0)),
                )
            )

    return {"best_reward": best_reward, "last_loss": last_loss}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train shared DQN traffic controller.")
    parser.add_argument("--roadnet", default="configs/roadnet.json")
    parser.add_argument("--flow", default="configs/flow_medium.json")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--sim-steps-per-action", type=int, default=10)
    parser.add_argument("--output-csv", default="data/buffer_dqn.csv")
    parser.add_argument("--model-path", default="models/best.pth")
    args = parser.parse_args()

    train(
        roadnet_path=args.roadnet,
        flow_path=args.flow,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        sim_steps_per_action=args.sim_steps_per_action,
        output_csv=args.output_csv,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
