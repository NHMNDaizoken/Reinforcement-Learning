# Smart Traffic Signal Digital Twin

Multi-Agent Reinforcement Learning for traffic signal control.  
9 intersections in a 3×3 CityFlow grid. Algorithm: **PressLight** (Independent DQN + Parameter Sharing + MaxPressure reward).

## How it works

Each intersection is an independent DQN agent. All 9 agents share one neural network (parameter sharing). The reward is negative MaxPressure — minimizing queue pressure is mathematically equivalent to maximizing throughput. A rule-based MaxPressure controller serves as the baseline.

```
State:   [incoming queues, outgoing queues, one-hot(current phase)]
Reward:  r = -(Σq_in − Σq_out)
Loss:    Huber loss, target network updated every 200 steps
```

## Project structure

```
configs/        roadnet.json + flow_low/medium/high.json
src/
  env/          TrafficEnv (CityFlow wrapper) + phase_map builder
  agent/        SharedDQNAgent, QNetwork, ReplayBuffer
  baseline/     MaxPressureBaseline (rule-based, no ML)
  training/     train_dqn.py — 500-episode training loop
  export/       export_replay.py — CityFlow → web/data/*.json
  evaluate.py   head-to-head DQN vs baseline, builds SQLite dataset
analysis/       plot.py → learning_curve.png + atl_comparison.png
api/            FastAPI — reads offline_dataset.db
web/            Expo dashboard — 3×3 grid replay + ATL charts
data/           generated CSVs + SQLite (gitignored)
models/         best.pth (gitignored)
```

## Quick start

CityFlow requires WSL Ubuntu — see [plan.md](plan.md) for full setup.

```bash
# activate CityFlow venv in WSL
source ~/CityFlow/.venv/bin/activate
cd /mnt/d/Ki2_nam3/Reinforcement_learning/BTCK

# 1. generate simulator configs
python configs/generate_configs.py

# 2. run tests
python -m pytest

# 3. train (500 episodes)
python src/training/train_dqn.py --roadnet configs/roadnet.json --flow configs/flow_medium.json --episodes 500

# 4. evaluate DQN vs baseline
python src/evaluate.py --flow configs/flow_medium.json --episodes 50
python analysis/plot.py

# 5. start API + dashboard
uvicorn api.main:app --port 8000
cd web && npm install && npm run dev   # http://localhost:3000
```

## Results

After 500 episodes on medium flow (600 veh/h), DQN should beat MaxPressure baseline on:
- **ATL** (Average Travel Time) — lower is better
- **Throughput** — higher is better

The gap is most visible on high flow (900 veh/h).

## Dependencies

**Python:** `torch>=2.0`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `fastapi`, `uvicorn[standard]`, `pydantic`, `pytest`  
**CityFlow:** install from source (see [plan.md](plan.md))  
**Web:** Node.js, Expo SDK 55, Recharts
