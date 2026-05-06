# BTCK - Smart Traffic Signal Digital Twin

Multi-agent reinforcement learning for traffic signal control on a 3x3
CityFlow grid. The project compares a shared DQN traffic-light controller with
a rule-based MaxPressure baseline, exports offline transition data, and
generates replay JSON for an Expo dashboard.

## Current Status

- **Simulator:** CityFlow with a 3x3 roadnet from the open
  `syn_3x3_gaussian_500_1h` dataset.
- **Synthetic flows:** generated Gaussian training seeds in
  `configs/train_flows/` plus flat/peak evaluation benchmarks in
  `configs/eval_flows/`.
- **Open dataset:** `configs/syn_3x3_gaussian_500_1h/` is present and used as
  the main Gaussian scenario.
- **RL controller:** 9 independent intersections sharing one DQN parameter set.
- **Baseline:** greedy local MaxPressure controller.
- **Training modes:** `single`, `random`, and `curriculum` over a flow
  directory.
- **Evaluation:** evaluates MaxPressure plus one or more DQN checkpoints and
  writes CSV/SQLite offline data.
- **Dashboard:** Expo / React Native app reads pre-exported JSON from
  `web/data`; no backend API is currently used.

## Change Info - 2026-05-06

- Split simulator runs into a 3600s decision phase and a 900s clearance phase.
  Training and offline replay rows are written only during the decision phase;
  clearance is used for final ATL/completion metrics.
- Reworked throughput reporting so it means completed vehicles, with additional
  `completed`, `active`, `generated`, and `completion_rate` metrics.
- Expanded `TrafficEnv` state with normalized incoming queues, downstream
  queues, waiting queues, phase elapsed time, min-green remaining, and one-hot
  phase.
- Changed RL reward to negative absolute pressure for the phase actually
  applied, and added min-green action masking/phase-hold support.
- Updated DQN training/evaluation logs and SQLite offline data with decision
  phase flags, simulation time, completion metrics, and flow scenario names.
- Replaced the old single config generator with separate scripts for training
  seeds and evaluation benchmarks.
- Updated baseline, training, evaluation, replay export, plotting, batch
  scripts, and tests for the new environment and metric semantics.

## Project Structure

```text
configs/
  generate_train_flows.py          Generates Gaussian training seeds
  generate_eval_benchmarks.py      Generates flat/peak evaluation flows
  train_flows/                     Gaussian training flow seeds
  eval_flows/                      Low/medium/high flat and peak benchmarks
  syn_3x3_gaussian_500_1h/         Open 3x3 Gaussian roadnet and flow
src/
  phase1_env_baseline/
    phase_map.py                   Builds intersection -> phase movement maps
    traffic_env.py                 CityFlow environment wrapper
    max_pressure.py                Rule-based MaxPressure baseline
  phase2_dqn/
    dqn_agent.py                   DenseLayer, QNetwork, ReplayBuffer, SharedDQNAgent
    train_dqn.py                   Single/random/curriculum shared-DQN training
  phase3_eval/
    evaluate.py                    Multi-model DQN vs baseline evaluation
  phase4_export/
    export_replay.py               CityFlow replay export for web/data/*.json
analysis/
  plot.py                          Evaluation plots
  plot_training.py                 Training plots from data/training_log.csv
data/                              Generated CSV and SQLite files
models/                            Generated DQN checkpoints
tests/                             Unit and integration-style tests by phase
web/                               Expo / React Native dashboard
memory/notes.md                    Persistent project notes and known fixes
```

`api/` exists but currently has no source files.

## Setup

CityFlow is normally installed from source in WSL Ubuntu. See [plan.md](plan.md)
for build notes.

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
# Regenerate training and evaluation flow files
python configs/generate_train_flows.py
python configs/generate_eval_benchmarks.py

# Run tests
python -m pytest

# Run MaxPressure baseline on the Gaussian roadnet
python src/phase1_env_baseline/max_pressure.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flow configs/eval_flows/high_flat_eval.json \
  --steps 3600

# Train a single DQN model on the Gaussian scenario
python src/phase2_dqn/train_dqn.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flows-dir configs/syn_3x3_gaussian_500_1h \
  --mode single \
  --episodes 500 \
  --model-path models/best_single.pth

# Train a curriculum model across generated training flows
python src/phase2_dqn/train_dqn.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flows-dir configs/train_flows \
  --mode curriculum \
  --curriculum-interval 100 \
  --episodes 9000 \
  --steps 360 \
  --sim-steps-per-action 10 \
  --model-path models/best_curriculum.pth

# Evaluate baseline plus multiple DQN checkpoints
python src/phase3_eval/evaluate.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flows \
    configs/eval_flows/low_flat_eval.json \
    configs/eval_flows/medium_flat_eval.json \
    configs/eval_flows/high_flat_eval.json \
    configs/eval_flows/low_peak_eval.json \
    configs/eval_flows/medium_peak_eval.json \
    configs/eval_flows/high_peak_eval.json \
  --models \
    models/best_single.pth \
    models/best_curriculum.pth \
  --episodes 50 \
  --steps 360 \
  --sim-steps-per-action 10

# Run the batch scripts
./run_all_train.sh
./run_all_evals.sh

# Generate analysis plots
python analysis/plot.py
python analysis/plot_training.py

# Export dashboard replay JSON
python src/phase4_export/export_replay.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flow configs/eval_flows/high_flat_eval.json \
  --algorithm baseline \
  --output web/data/high.json

# Run dashboard
cd web
npm run web
```

## Training Outputs

`train_dqn.py` writes:

- `data/buffer_dqn.csv`: transition buffer from the latest training run.
- `data/training_log.csv`: per-episode flow, reward, ATL, throughput, loss, and
  epsilon.
- `models/*.pth`: best checkpoint by average reward.

`evaluate.py` writes:

- `data/buffer_baseline.csv`
- `data/buffer_<model_name>.csv` for each model passed to `--models`
- `data/offline_dataset.db` with a `policy` column for baseline and DQN rows

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

## Traffic Scenarios

| Scenario | Vehicles | File |
|---|---:|---|
| Low flat | 900 | `configs/eval_flows/low_flat_eval.json` |
| Medium flat | 3000 | `configs/eval_flows/medium_flat_eval.json` |
| High flat | 6000 | `configs/eval_flows/high_flat_eval.json` |
| Low peak | 900 | `configs/eval_flows/low_peak_eval.json` |
| Medium peak | 3000 | `configs/eval_flows/medium_peak_eval.json` |
| High peak | 6000 | `configs/eval_flows/high_peak_eval.json` |
| Gaussian | dataset-defined | `configs/syn_3x3_gaussian_500_1h/syn_3x3_gaussian_500_1h.json` |
