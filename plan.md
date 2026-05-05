# **BTCK – Implementation Plan**

**Algorithm:** Shared DQN (PressLight-inspired) + MaxPressure reward
**Simulator:** CityFlow
**Scale:** Synthetic (3×3) → Open Datasets → Real-world
**Status:** Roadmap 1 completed → transitioning to Roadmap 2

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
* Created synthetic traffic flows:

  * Low (~300 veh/h)
  * Medium (~600 veh/h)
  * High (~900 veh/h)

Baseline: **MaxPressure**

---

## **3. Phase 2 — Shared DQN**

### Design

* State: `[q_in, q_out, one_hot(phase)]`
* Reward: `r = -(sum(q_in) - sum(q_out))`
* One shared Q-network for all intersections

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

* DQN outperforms baseline on **high traffic (ATL)**
* Fails on medium → **domain mismatch**
* Training improved significantly after epsilon fix

---

## **5. Phase 4 — Dashboard**

Features:

* Dark mode
* Vehicle visualization (🚗🚌🛵)
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

```bash
git clone https://github.com/traffic-signal-control/sample-code.git
cp -r sample-code/data/syn_3x3_gaussian_500_1h configs/
```

**Target:**
👉 DQN ATL < **631.75s**

---

### **2. Peak Traffic Scenario**

Create:

```
flow_low_peak.json
flow_medium_peak.json
flow_high_peak.json
```

* Simulate rush-hour traffic spikes

---

### **3. Multi-Scenario Training**

| Scenario    | Status |
| ----------- | ------ |
| Low Flat    | ⏳      |
| Medium Flat | ⏳      |
| High Flat   | ✅      |
| Gaussian    | ⏳      |
| Peak        | ❌      |

---

### **4. Curriculum Learning (Core)**

```
0–200   → low_flat
200–400 → medium_flat
400–600 → high_flat
600–800 → gaussian
800–1000 → high_peak
```

**Tasks:**

* Modify `train_dqn.py`
* Support multiple flows
* Add curriculum mode

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

## **VII. Immediate Actions**

```bash
git checkout -b feat/roadmap2

# Download dataset
git clone https://github.com/traffic-signal-control/sample-code.git /tmp/tssc
cp -r /tmp/tssc/data/syn_3x3_gaussian_500_1h configs/

# Train quick test
python src/phase2_dqn/train_dqn.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet.json \
  --flow configs/syn_3x3_gaussian_500_1h/flow.json \
  --episodes 200
```

---

# **Summary**

* Roadmap 1 = **working system (baseline + DQN + dashboard)**
* Roadmap 2 = **make it research-grade + demo-ready**
* Final goal = **beat baseline on real datasets + present visually**


