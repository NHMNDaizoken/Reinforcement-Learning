# **BTCK – Implementation Plan**

**Algorithm:** Shared DQN (PressLight-inspired) + MaxPressure reward
**Simulator:** CityFlow
**Scale:** Synthetic (3×3) → Open Dataset (3×3 Gaussian) → Real-world
**Status:** Roadmap 1 completed → Roadmap 2 environment/metrics hardening in progress

---

# **CHANGE INFO — 2026-05-06**

Implemented changes:

* Split simulation into decision and clearance phases:
  * decision phase: action selection, rewards, training, replay/offline rows
  * clearance phase: final ATL/completion metrics only
* Replaced ambiguous throughput fallback with completed-vehicle tracking.
* Added completion metrics everywhere:
  * `completed`
  * `active`
  * `generated`
  * `completion_rate`
* Expanded state representation:
  * normalized incoming queues
  * normalized downstream queues
  * normalized waiting queues
  * phase elapsed time
  * min-green remaining
  * current phase one-hot
* Changed environment reward to negative absolute pressure of the applied phase.
* Added min-green phase-hold/action-mask support in `TrafficEnv`.
* Added explicit DQN target-network sync method.
* Updated training logs, evaluation CSVs, SQLite offline data, replay export JSON,
  plotting, shell commands, and tests to match the new metric schema.
* Updated config generation to produce both legacy low/medium/high flows and
  demand curriculum flows, with all generated vehicles spawned before 2700s.

Impact:

* Replay buffers should no longer contain clearance-phase transitions.
* ATL and completion metrics are computed after vehicles have time to exit.
* Reports should refer to completed vehicles instead of generic throughput.
* Old result tables should be treated as pre-change results and not compared
  directly with new post-clearance metrics.

---

# **ROADMAP 1 — Core System (Completed)**

## **1. Environment Setup**

CityFlow installed from source (WSL Ubuntu):

```bash
sudo apt install -y python3-full python3-venv build-essential cmake git

git clone https://github.com/cityflow-project/CityFlow.git
cd CityFlow

python3 -m venv .venv
source .venv/bin/activate

pip install "cmake<4"

cd extern/pybind11
git fetch https://github.com/pybind/pybind11.git refs/tags/v2.11.1:refs/tags/v2.11.1
git checkout v2.11.1

cd ../..
pip install . --no-build-isolation
```

---

## **2. Phase 1 — Environment & Baseline**

* Generated 3×3 grid (9 intersections)
* Created synthetic traffic flows from the Gaussian dataset routes:

  * Low flat / peak (300 vehicles)
  * Medium flat / peak (600 vehicles)
  * High flat / peak (900 vehicles)

Baseline: **MaxPressure**

---

## **3. Phase 2 — Shared DQN**

### Design

* State: `[q_in, q_out, one_hot(phase)]`
* Reward: `r = -(sum(q_in) - sum(q_out))`
* One shared Q-network for all intersections
* Supports `single`, `random`, and `curriculum` modes through `--flows`

### Key Fixes

| Issue                  | Fix                       |
| ---------------------- | ------------------------- |
| Epsilon decay too fast | Move decay to per-episode |
| Small buffer           | 5k → 50k                  |
| Noisy gradients        | Batch 64 → 256            |
| Unstable target        | Update freq 200 → 1000    |

---

## **4. Phase 3 — Evaluation**

### Results

| Model        | ATL        | Throughput |
| ------------ | ---------- | ---------- |
| DQN (high)   | **114.9s** | 23         |
| MaxPressure  | 119.7s     | 26         |
| DQN (medium) | 117.3s     | 14         |

### Observations

* DQN outperforms baseline on **high traffic (ATL)** in the Roadmap 1 run
* Fails on medium -> **domain mismatch**
* Training improved significantly after epsilon fix

---

## **5. Phase 4 — Dashboard**

Features:

* Dark mode
* Vehicle visualization
* Replay simulation
* DQN vs Baseline comparison

---

## **6. Limitations**

* Synthetic data only
* Single-lane roads
* No peak traffic
* No real dataset comparison
* Weak generalization

---

# **ROADMAP 2 — Expansion & Final Product**

---

## **0. Pre-step — Dashboard UX/UI**

**Goal:** turn dashboard into a presentation-ready product

**Tasks:**

* Identify users (reviewers, evaluators)
* Improve layout (clean, intuitive)
* Show:

  * ATL
  * Throughput
  * Charts
* Add interaction:

  * Dataset switch
  * Replay controls

---

## **I. Goals of Roadmap 2**

* Generalize model across traffic conditions
* Use real-world datasets
* Implement Curriculum Learning
* Build strong visual demo

---

## **II. Phase 1 — Data Upgrade & Curriculum Learning**

### **1. Open Dataset (Critical)**

Dataset:

* `syn_3x3_gaussian_500`

Current local path:

```text
configs/syn_3x3_gaussian_500_1h/
```

Main files:

```text
configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json
configs/syn_3x3_gaussian_500_1h/syn_3x3_gaussian_500_1h.json
```

**Target:**
👉 DQN ATL < **631.75s**

---

### **2. Peak Traffic Scenario**

Created:

```
flow_low_peak.json
flow_medium_peak.json
flow_high_peak.json
```

Also generated flat variants:

```
flow_low_flat.json
flow_medium_flat.json
flow_high_flat.json
```

Peak flows place 70% of vehicles between 900s and 1800s.

---

### **3. Multi-Scenario Training**

| Scenario    | Status |
| ----------- | ------ |
| Low Flat    | ready |
| Medium Flat | ready |
| High Flat   | ready |
| Gaussian    | ready |
| Peak        | ready |

---

### **4. Curriculum Learning (Core)**

```
0-200     -> low_flat
200-400   -> low_peak
400-600   -> medium_flat
600-800   -> medium_peak
800-1000  -> high_flat
1000-1200 -> gaussian
```

**Implemented:**

* `train_dqn.py` accepts `--flows`
* `--mode curriculum` unlocks one additional flow every `--curriculum-interval`
* Current curriculum samples the hardest unlocked flow 75% of the time and old
  unlocked flows 25% of the time

---

### **5. Target Results**

| Method         | Gaussian | High | Peak |
| -------------- | -------- | ---- | ---- |
| Baseline       | 631s     | 119s | ?    |
| DQN            | ?        | 114s | ?    |
| DQN Curriculum | ?        | ?    | ?    |

---

## **III. Phase 2 — Scaling**

### **1. Jinan 3×4**

* 12 intersections
* Real-world dataset

### **2. Hangzhou 4×4**

* Baseline: 240.97s
* Target: beat baseline

### **3. (Optional) Da Nang Map**

* OSM → CityFlow
* Realistic traffic mix

---

## **IV. Phase 3 — Dashboard Upgrade**

**Tasks:**

* Multi-dataset support
* Real-time ATL chart
* Side-by-side comparison
* Multi-lane rendering
* Playback controls

---

## **V. Phase 4 — Report & Demo**

### Results

* Full comparison tables
* Training plots

### Slides (7)

1. Problem
2. Method
3. Dataset
4. Demo (synthetic)
5. Demo (real)
6. Results
7. Conclusion

### Demo

* Switch dataset live
* Compare models
* Show real-time charts

---

## **VI. Timeline**

| Week     | Focus                |
| -------- | -------------------- |
| Week 1   | Dataset + Curriculum |
| Week 2   | Scaling              |
| Week 2–3 | Dashboard            |
| Week 3   | Report               |

---

## **VII. Current Commands**

```bash
# Regenerate flat/peak flows from the local Gaussian routes
python configs/generate_configs.py

# Train single Gaussian model
python src/phase2_dqn/train_dqn.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flows configs/syn_3x3_gaussian_500_1h/syn_3x3_gaussian_500_1h.json \
  --mode single \
  --episodes 500 \
  --model-path models/best_single.pth

# Train curriculum model
python src/phase2_dqn/train_dqn.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flows configs/flow_low_flat.json configs/flow_low_peak.json configs/flow_medium_flat.json configs/flow_medium_peak.json configs/flow_high_flat.json configs/syn_3x3_gaussian_500_1h/syn_3x3_gaussian_500_1h.json \
  --mode curriculum \
  --curriculum-interval 200 \
  --episodes 1200 \
  --model-path models/best_curriculum.pth

# Evaluate baseline + both models
python src/phase3_eval/evaluate.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flow configs/flow_high_peak.json \
  --models models/best_single.pth models/best_curriculum.pth \
  --episodes 50
```

---

# **Summary**

* Roadmap 1 = **working system (baseline + DQN + dashboard)**
* Roadmap 2 = **make it research-grade + demo-ready**
* Final goal = **beat baseline on real datasets + present visually**
