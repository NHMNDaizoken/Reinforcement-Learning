"""Vẽ learning curve từ training_log.csv (1 dòng/episode)."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_PATH = Path("data/training_log.csv")
OUTPUT_DIR = Path("analysis")


def plot_training_log(log_path: Path = LOG_PATH, output_dir: Path = OUTPUT_DIR) -> None:
    if not log_path.exists():
        print(f"Không tìm thấy {log_path}. Hãy train lại với file train_dqn.py mới.")
        return

    episodes, rewards, atls, losses, epsilons = [], [], [], [], []

    with log_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            atls.append(float(row["atl"]))
            losses.append(float(row["loss"]))
            epsilons.append(float(row["epsilon"]))

    if not episodes:
        print("File training_log.csv trống.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Learning Curve: Reward + Loss ───────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=120)
    ax1.plot(episodes, rewards, color="#1f77b4", linewidth=1.5, label="Episode Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(episodes, losses, color="#d62728", linewidth=1.2, linestyle="--", label="Loss")
    ax2.set_ylabel("Loss", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title("DQN Training Learning Curve")
    fig.tight_layout()
    out = output_dir / "learning_curve_training.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Đã lưu: {out}")

    # ── 2. ATL theo Episode ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    ax.plot(episodes, atls, color="#2ca02c", linewidth=1.5, label="ATL (giây)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Travel Time (s)")
    ax.set_title("ATL theo Episode trong Training")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = output_dir / "atl_training.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Đã lưu: {out}")

    # ── 3. Epsilon Decay ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3), dpi=120)
    ax.plot(episodes, epsilons, color="#ff7f0e", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Epsilon Decay (Exploration → Exploitation)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / "epsilon_decay.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Đã lưu: {out}")

    print(f"\nTóm tắt training ({len(episodes)} episodes):")
    print(f"  Reward đầu:  {rewards[0]:.1f}  →  Reward cuối: {rewards[-1]:.1f}")
    print(f"  ATL đầu:     {atls[0]:.2f}s  →  ATL cuối:    {atls[-1]:.2f}s")
    print(f"  Loss đầu:    {losses[0]:.4f}  →  Loss cuối:   {losses[-1]:.4f}")


if __name__ == "__main__":
    plot_training_log()