# Traffic Signal Control with Reinforcement Learning

Multi-agent DQN controller for smart traffic signal control, benchmarked against a rule-based MaxPressure baseline using the [CityFlow](https://cityflow-project.github.io/) microscopic traffic simulator. Includes a React Native dashboard that replays pre-exported simulation data frame by frame.

---

## Architecture Overview

The project is organized into four sequential phases. Each phase builds on the previous one:

```text
configs/ (roadnet + flow JSON)
         │
         ▼
[Phase 1] phase_map.py  ──►  traffic_env.py       ──►  max_pressure.py
          Parse roadnet       CityFlow wrapper           Rule-based baseline
                │                    │
                └────────────────────┘
                         │
                         ▼
[Phase 2]          train_dqn.py
                   DQN training loop
                   (saves models/*.pth + data/training_log.csv)
                         │
                         ▼
[Phase 3]          evaluate.py
                   Compare DQN vs Baseline
                   (saves data/buffer_*.csv + data/offline_dataset.db)
                         │
                         ▼
[Phase 4]          export_replay.py
                   Per-second simulation snapshots
                   (saves web/data/*.json)
                         │
                         ▼
                   web/App.tsx
                   Expo dashboard — plays back the JSON frames
```

---

## Directory Structure

```text
Reinforcement-Learning/
│
├── configs/                           Traffic scenario configuration files
│   ├── generate_configs.py            Generates flat/peak flow files from the Gaussian routes
│   ├── flow_low_flat.json             ~300 vehicles, uniform distribution
│   ├── flow_low_peak.json             ~300 vehicles, morning/evening peaks
│   ├── flow_medium_flat.json          ~600 vehicles, uniform
│   ├── flow_medium_peak.json          ~600 vehicles, peaks
│   ├── flow_high_flat.json            ~900 vehicles, uniform
│   ├── flow_high_peak.json            ~900 vehicles, peaks
│   ├── syn_3x3_gaussian_500_1h/       Open-source 3×3 Gaussian dataset
│   │   ├── roadnet_3X3.json           Road network: 9 real intersections + 12 virtual boundary nodes
│   │   └── syn_3x3_gaussian_500_1h.json  Vehicle flows sampled from a Gaussian distribution
│   └── hangzhou_4x4/                  Real-world Hangzhou 4×4 dataset
│       ├── roadnet_4X4.json           16 real intersections
│       └── flow.json                  Hangzhou vehicle demand
│
├── src/
│   ├── phase1_env_baseline/
│   │   ├── phase_map.py               Parses roadnet JSON → intersection→phase→lane-movement map
│   │   ├── traffic_env.py             CityFlow environment wrapper (state, reward, action masking)
│   │   └── max_pressure.py            Greedy MaxPressure rule-based controller + standalone runner
│   │
│   ├── phase2_dqn/
│   │   ├── dqn_agent.py               DenseLayer, QNetwork, ReplayBuffer, SharedDQNAgent
│   │   └── train_dqn.py               Training loop — single / random / curriculum modes
│   │
│   ├── phase3_eval/
│   │   └── evaluate.py                Runs baseline + N DQN checkpoints, writes CSV and SQLite
│   │
│   └── phase4_export/
│       └── export_replay.py           Exports per-second simulation frames to JSON for the dashboard
│
├── analysis/
│   ├── plot.py                        Evaluation comparison plots (ATL, throughput by policy)
│   └── plot_training.py               Training curve plots from data/training_log.csv
│
├── web/                               Expo / React Native dashboard
│   ├── App.tsx                        Main component — reads JSON and animates frames
│   ├── app.json                       Expo project config
│   ├── package.json
│   └── data/                          Pre-exported replay JSON files
│       ├── low.json                   Baseline replay, low-flat flow
│       ├── medium.json                Baseline replay, medium-flat flow
│       ├── high.json                  Baseline replay, high-flat flow
│       ├── high_model.json            DQN replay, high-flat flow
│       ├── hangzhou.json              Baseline replay, Hangzhou 4×4
│       └── hangzhou_model.json        DQN replay, Hangzhou 4×4
│
├── data/                              Generated outputs (CSV, SQLite) — gitignored
│   ├── buffer_dqn.csv                 DQN transition buffer from the last training run
│   ├── buffer_baseline.csv            Baseline transitions from evaluation
│   ├── training_log.csv               Per-episode: reward, ATL, throughput, loss, epsilon
│   └── offline_dataset.db             SQLite combining baseline + all DQN policies
│
├── models/                            Saved model checkpoints — gitignored
│   ├── best_single.pth                Best checkpoint from single-flow training
│   └── best_curriculum.pth            Best checkpoint from curriculum training
│
├── tests/                             Pytest test suite organized by phase
├── analysis/                          Training and evaluation plots
├── run_all_train.sh                   Batch script: trains all modes sequentially
├── run_all_evals.sh                   Batch script: evaluates all flow scenarios
└── requirements.txt                   Python dependencies
```

---

## How the Key Files Work

### `src/phase1_env_baseline/phase_map.py`

Reads a CityFlow `roadnet.json` and builds a **PhaseMap**:

```text
PhaseMap = { intersection_id → [ phase_0_movements, phase_1_movements, … ] }
phase_k_movements = [ (incoming_lane_id, outgoing_lane_id), … ]
```

This map is the single source of truth shared by the environment, both controllers, and the export script. Every intersection that is not marked `virtual` is included.

---

### `src/phase1_env_baseline/traffic_env.py` — `TrafficEnv`

Wraps the CityFlow engine into a Gym-style environment for **N independent agents** (one per non-virtual intersection).

**Simulation timeline per episode:**

```text
t = 0 ──────────── t = 3600 ──────────── t = 4500
  ◄── decision phase ──►◄── clearance phase ──►
  (RL actions, rewards, buffer writes)   (vehicles drain, no new training data)
```

**State vector** (one per intersection, all identical in dimension since shared DQN):

| Segment | Size | Description |
| --- | --- | --- |
| `q_in` | N_movements | Normalized incoming lane queue (÷ 50) |
| `q_out` | N_movements | Normalized outgoing lane queue (÷ 50) |
| `waiting` | N_movements | Normalized waiting count on incoming lanes |
| `phase_elapsed` | 1 | How long current phase has been held (÷ min_green) |
| `min_green_remaining` | 1 | Remaining mandatory hold fraction |
| `one_hot_phase` | N_phases | One-hot encoding of the current phase |

**Reward:** negative total queue length across all unique incoming lanes of the intersection. The agent is penalized for every vehicle sitting in an incoming lane, regardless of which phase is active.

**Action masking:** if the current phase has been held for fewer than `min_green=10` steps, the environment silently forces the agent to keep the current phase even if it requested a switch.

---

### `src/phase1_env_baseline/max_pressure.py` — `MaxPressureBaseline`

A greedy rule-based controller used as the performance baseline. At every decision step it reads the lane vehicle counts and selects the phase whose **pressure** (sum of incoming queue minus outgoing queue across all movements in that phase) is greatest. No learning, no memory.

Run standalone:

```bash
python src/phase1_env_baseline/max_pressure.py --flow configs/flow_high_flat.json
```

---

### `src/phase2_dqn/dqn_agent.py` — `SharedDQNAgent`

One DQN parameter set is **shared across all 9 agents**. Each intersection contributes its own state→action→reward tuples to the same replay buffer.

```text
QNetwork:  state_dim → [Linear 128, ReLU] → [Linear 64, ReLU] → action_dim
```

Key design choices:

- **Huber loss** (δ=1.0) for stability with outlier rewards.
- **Target network** synced every 1 000 gradient updates (not every episode).
- **Epsilon decay** happens once per completed episode (not per step).
- The replay buffer is **retained across flow switches** during curriculum training so experience from easier flows is not discarded.

---

### `src/phase2_dqn/train_dqn.py` — Training Loop

Three training modes controlled by `--mode`:

| Mode | Flow selection |
| --- | --- |
| `single` | Always uses `flows[0]` |
| `random` | Uniformly random from the flow list each episode |
| `curriculum` | Unlocks flows gradually every `--curriculum-interval` episodes; 75% train the hardest unlocked flow, 25% revisit an easier one |

Each episode:

1. Select a flow file (mode-dependent).
2. `env.reset()` — restarts CityFlow with that flow.
3. Run `steps_per_episode` decision steps (default 360 × 10 sim-steps = 3 600 s).
4. After `done=True`, run clearance steps until the simulation drains.
5. Update `training_log.csv` and save a checkpoint if average reward improved.

---

### `src/phase3_eval/evaluate.py`

Evaluates **MaxPressure + any number of DQN checkpoints** on a single flow scenario. For each policy it runs `--episodes` independent episodes and averages the metrics. Writes:

- `data/buffer_baseline.csv` — MaxPressure transitions
- `data/buffer_<model_name>.csv` — DQN transitions per checkpoint
- `data/offline_dataset.db` — SQLite table `transitions` merging all policies with a `policy` column

---

### `src/phase4_export/export_replay.py`

Runs one simulation and captures a **snapshot every second** containing:

- Per-intersection: current phase, queue length, phase pressures
- Per-vehicle: screen (x, y) coordinate, angle, waiting flag, vehicle type (car / motorbike / bus)
- Global: ATL, completed count, total queue

Vehicle world coordinates are projected to screen coordinates matching the dashboard canvas (820 × 820 px, 170 px margin). The output is a single JSON file consumed directly by the dashboard with no backend.

---

### `web/App.tsx` — Dashboard

A React Native (Expo) app that:

1. Loads a pre-exported JSON file from `web/data/`.
2. Plays frames at 30 fps using `requestAnimationFrame`.
3. Renders the intersection grid, vehicle dots (colored by type/waiting state), phase indicators, and live metric panels (ATL, throughput, total queue).
4. Provides buttons to switch between `baseline` and `model` algorithms, and between traffic scenarios.

No network requests are made at runtime — all data is bundled at build time via `require()`.

---

## Setup

### Python environment (CityFlow requires Linux / WSL)

CityFlow must be compiled from source on Linux (WSL Ubuntu recommended):

```bash
# In WSL:
git clone https://github.com/cityflow-project/CityFlow.git ~/CityFlow
cd ~/CityFlow && pip install .

# Then in this repo:
source ~/CityFlow/.venv/bin/activate
pip install -r requirements.txt
```

### Dashboard

```bash
cd web
npm install
npm run web      # Opens Expo in browser
```

---

## Common Commands

```bash
# ── Generate synthetic flow files ─────────────────────────────────────────
python configs/generate_configs.py

# ── Run tests ─────────────────────────────────────────────────────────────
python -m pytest

# ── Run MaxPressure baseline (standalone) ─────────────────────────────────
python src/phase1_env_baseline/max_pressure.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flow configs/flow_high_flat.json

# ── Train: single flow ────────────────────────────────────────────────────
python src/phase2_dqn/train_dqn.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flows configs/syn_3x3_gaussian_500_1h/syn_3x3_gaussian_500_1h.json \
  --mode single --episodes 500 \
  --model-path models/best_single.pth

# ── Train: soft curriculum across all flows ───────────────────────────────
python src/phase2_dqn/train_dqn.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flows configs/flow_low_flat.json configs/flow_low_peak.json \
          configs/flow_medium_flat.json configs/flow_medium_peak.json \
          configs/flow_high_flat.json \
          configs/syn_3x3_gaussian_500_1h/syn_3x3_gaussian_500_1h.json \
  --mode curriculum --curriculum-interval 200 --episodes 1200 \
  --model-path models/best_curriculum.pth

# ── Evaluate baseline + checkpoints ──────────────────────────────────────
python src/phase3_eval/evaluate.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flow configs/flow_high_peak.json \
  --models models/best_single.pth models/best_curriculum.pth \
  --episodes 50

# ── Evaluate baseline only ────────────────────────────────────────────────
python src/phase3_eval/evaluate.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flow configs/flow_high_flat.json \
  --baseline-only

# ── Batch scripts ─────────────────────────────────────────────────────────
./run_all_train.sh
./run_all_evals.sh

# ── Generate analysis plots ───────────────────────────────────────────────
python analysis/plot_training.py    # Training curves from data/training_log.csv
python analysis/plot.py             # Evaluation comparison (ATL, throughput)

# ── Export dashboard replay JSON ──────────────────────────────────────────
# Baseline run
python src/phase4_export/export_replay.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flow configs/flow_high_flat.json \
  --algorithm baseline --steps 600 \
  --output web/data/high.json

# DQN model run
python src/phase4_export/export_replay.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flow configs/flow_high_flat.json \
  --algorithm model --steps 600 \
  --model-path models/best_curriculum.pth \
  --output web/data/high_model.json
```

---

## Output Files

### Training (`train_dqn.py`)

| File | Content |
| --- | --- |
| `data/buffer_dqn.csv` | Raw transition tuples from the latest training run (state, action, reward, next_state, done, metrics) |
| `data/training_log.csv` | Per-episode summary: flow scenario, avg reward, ATL, throughput, completion rate, loss, epsilon |
| `models/best_*.pth` | Best QNetwork weights by average reward |

### Evaluation (`evaluate.py`)

| File | Content |
| --- | --- |
| `data/buffer_baseline.csv` | MaxPressure transitions |
| `data/buffer_<model>.csv` | DQN transitions per checkpoint |
| `data/offline_dataset.db` | SQLite `transitions` table — all policies combined with `policy` column |

---

## DQN Hyperparameters

| Parameter | Value |
| --- | ---: |
| Learning rate | `5e-4` |
| Discount factor γ | `0.95` |
| Epsilon start / min / decay | `1.0 / 0.05 / 0.9934` |
| Replay buffer capacity | `50 000` |
| Batch size | `256` |
| Target network update | every `1 000` gradient updates |
| Huber loss δ | `1.0` |
| Gradient clip norm | `10.0` |
| Network architecture | `state_dim → 128 → 64 → action_dim` |

---

## Traffic Scenarios

| Scenario | ~Vehicles | Pattern | File |
| --- | ---: | --- | --- |
| Low flat | 300 | Uniform | `configs/flow_low_flat.json` |
| Low peak | 300 | Morning/evening peak | `configs/flow_low_peak.json` |
| Medium flat | 600 | Uniform | `configs/flow_medium_flat.json` |
| Medium peak | 600 | Morning/evening peak | `configs/flow_medium_peak.json` |
| High flat | 900 | Uniform | `configs/flow_high_flat.json` |
| High peak | 900 | Morning/evening peak | `configs/flow_high_peak.json` |
| Gaussian 3×3 | ~500 | Gaussian demand | `configs/syn_3x3_gaussian_500_1h/` |
| Hangzhou 4×4 | real-world | Real demand | `configs/hangzhou_4x4/` |

---

## Metrics

| Metric | Description |
| --- | --- |
| **ATL** | Average Travel Time — mean time from vehicle spawn to exit (seconds). Lower is better. |
| **Completed** | Number of vehicles that exited the network during the decision phase (3 600 s). |
| **Active** | Vehicles still in the network at end of clearance phase (should approach 0). |
| **Completion Rate** | `completed / generated`. Closer to 1.0 means fewer vehicles stranded. |
| **Avg Reward** | Mean per-agent per-step reward during training. More negative = more congestion. |
