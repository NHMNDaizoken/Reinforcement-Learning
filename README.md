# BTCK - Smart Traffic Signal Digital Twin

Multi-agent reinforcement learning for traffic signal control on a 3x3 CityFlow grid.
The project compares a shared DQN traffic-light controller against a rule-based
MaxPressure baseline, then exports replay data for an Expo dashboard.

## What Is Implemented

- **Simulator:** CityFlow roadnet + traffic flows generated from `configs/generate_configs.py`
- **Network:** 9 signalized intersections, each with 2 phases and 3 lanes per road
- **RL controller:** independent agents using one shared DQN parameter set
- **Baseline:** greedy local MaxPressure controller
- **State:** incoming queues, outgoing queues, and one-hot current phase
- **Reward:** negative MaxPressure, `-(sum(q_in) - sum(q_out))`
- **Outputs:** transition CSVs, SQLite offline dataset, analysis PNGs, and dashboard replay JSON

## Project Structure

```text
configs/                         CityFlow roadnet + low/medium/high flow generation
src/
  phase1_env_baseline/
    phase_map.py                  Builds intersection -> phase movement maps
    traffic_env.py                CityFlow environment wrapper
    max_pressure.py               Rule-based MaxPressure baseline
  phase2_dqn/
    dqn_agent.py                  DenseLayer, QNetwork, ReplayBuffer, SharedDQNAgent
    train_dqn.py                  Shared-DQN training loop and training_log.csv writer
  phase3_eval/
    evaluate.py                   DQN vs baseline evaluation, CSV + SQLite export
  phase4_export/
    export_replay.py              CityFlow replay export for web/data/*.json
analysis/
  plot.py                         Builds learning_curve.png and atl_comparison.png
  plot_training.py                Builds training-specific plots from training_log.csv
web/                              Expo / React Native dashboard using web/data/*.json
data/                             Generated CSV and SQLite files
models/                           Generated DQN checkpoints
tests/                            Unit and integration-style tests by phase
```

`api/` currently has no source files. The dashboard uses pre-exported JSON from
`web/data/`; it does not call a FastAPI backend in the current codebase.

## Setup

CityFlow is normally installed from source in WSL Ubuntu. See [plan.md](plan.md)
for the full CityFlow build notes.

```bash
cd /mnt/d/Ki2_nam3/Reinforcement_learning/BTCK
source ~/CityFlow/.venv/bin/activate
pip install -r requirements.txt
```

For the dashboard:

```bash
cd web
npm install
```

## Common Commands

```bash
# Generate configs/roadnet.json and configs/flow_{low,medium,high}.json
python configs/generate_configs.py

# Run tests
python -m pytest

# Run MaxPressure baseline
python src/phase1_env_baseline/max_pressure.py --roadnet configs/roadnet.json --flow configs/flow_low.json --steps 3600

# Train shared DQN on one flow file
python src/phase2_dqn/train_dqn.py --roadnet configs/roadnet.json --flow configs/flow_medium.json --episodes 500

# Evaluate DQN against baseline and rebuild data/offline_dataset.db
python src/phase3_eval/evaluate.py --flow configs/flow_medium.json --episodes 5

# Generate analysis plots
python analysis/plot.py
python analysis/plot_training.py

# Export dashboard replay JSON
python src/phase4_export/export_replay.py --flow configs/flow_high.json --algorithm baseline --output web/data/high.json
python src/phase4_export/export_replay.py --flow configs/flow_high.json --algorithm model --output web/data/high_model.json

# Run dashboard
cd web
npm run web
```

## Current DQN Defaults

| Parameter | Value |
|---|---:|
| Learning rate | `5e-4` |
| Discount factor | `0.95` |
| Epsilon start / min / decay | `1.0 / 0.05 / 0.9958` |
| Replay buffer | `50,000` |
| Batch size | `256` |
| Target update frequency | `1,000` updates |
| Huber delta | `1.0` |
| Gradient max norm | `10.0` |
| Network | `state_dim -> 128 -> 64 -> action_dim` |

Training currently accepts a single `--flow` file per run. To compare traffic
levels, run training or evaluation separately for `flow_low.json`,
`flow_medium.json`, and `flow_high.json`.
