# BTCK — Implementation Plan

**Algorithm:** PressLight (Independent DQN + Parameter Sharing + MaxPressure reward)  
**Simulator:** CityFlow, 3×3 grid, 9 intersections  
**Rule:** Làm xong một giai đoạn → dừng checkpoint → mới sang tiếp.

---

## Setup môi trường (WSL Ubuntu)

CityFlow phải cài từ source trên WSL, không có trên PyPI:

```bash
sudo apt install -y python3-full python3-venv build-essential cmake git
cd ~
git clone https://github.com/cityflow-project/CityFlow.git
cd CityFlow
python3 -m venv .venv && source .venv/bin/activate
pip install "cmake<4"
cd extern/pybind11
git fetch https://github.com/pybind/pybind11.git refs/tags/v2.11.1:refs/tags/v2.11.1
git checkout v2.11.1
cd ../..
pip install . --no-build-isolation
```

> Lý do `cmake<4` và `pybind11 v2.11.1`: CMake 4 reject policy cũ của CityFlow; pybind11 bundled v2.3.0 không tương thích Python 3.11.

Cài project deps trong cùng venv:

```bash
cd /mnt/d/Ki2_nam3/Reinforcement_learning/BTCK
source ~/CityFlow/.venv/bin/activate
pip install torch numpy pandas matplotlib seaborn fastapi "uvicorn[standard]" pydantic pytest
```

---

## Giai đoạn 1 — Môi trường & Baseline

**Output:** `configs/roadnet.json`, `configs/flow_*.json`, `src/baseline/max_pressure.py`

```bash
python configs/generate_configs.py
python -m pytest tests/test_max_pressure_baseline.py
python src/baseline/max_pressure.py --roadnet configs/roadnet.json --flow configs/flow_low.json --steps 3600
```

Checkpoint:
- `roadnet.json` có 9 intersections, mỗi cái ≥ 2 phases
- 3 flow files (low 300, medium 600, high 900 xe/h)
- MaxPressure in ra ATL + throughput

---

## Giai đoạn 2 — DQN & Training

**Output:** `src/env/traffic_env.py`, `src/agent/dqn_agent.py`, `src/training/train_dqn.py`, `models/best.pth`, `data/buffer_dqn.csv`

```bash
python -m pytest tests/test_traffic_env.py tests/test_dqn_agent.py
python src/training/train_dqn.py --flow configs/flow_low.json --episodes 5    # smoke test
python src/training/train_dqn.py --episodes 500                                # train mixed flows
```

Thiết kế bắt buộc:
- State: `[q_in, q_out, one_hot(phase)]`
- Reward: `r = -(Σq_in - Σq_out)`
- DQN MLP: 128 → 64 → action_dim, Huber loss, target network update mỗi 200 steps
- 9 agents dùng chung 1 bộ trọng số (parameter sharing)
- **Mixed-flow training**: mỗi episode sample ngẫu nhiên 1 trong 3 flow files (low/medium/high) — theo cách PressLight gốc, giúp model generalize qua mọi mức tải

Hyperparameters: lr=1e-3, γ=0.95, ε: 1.0→0.01 (decay 0.995), buffer=5000, batch=64

---

## Giai đoạn 3 — Đánh giá & Dataset

**Output:** `data/buffer_baseline.csv`, `data/offline_dataset.db`, `analysis/*.png`

```bash
python src/evaluate.py --flow configs/flow_medium.json --episodes 50
python src/evaluate.py --flow configs/flow_high.json --episodes 50
python analysis/plot.py
```

SQLite schema: `(episode, step, agent_id, state_vec, action, reward, next_state_vec, done, atl, throughput, policy)`

Checkpoint: bảng so sánh DQN vs Baseline, learning curve + ATL comparison PNG.

---

## Giai đoạn 4 — API & Dashboard

**Output:** `api/main.py`, `api/db.py`, `web/`

```bash
uvicorn api.main:app --port 8000
# kiểm tra: curl http://localhost:8000/health
cd web && npm install && npm run dev
# mở http://localhost:3000
```

API endpoints: `GET /health`, `GET /episodes`, `GET /replay?episode=N`, `GET /metrics`

Dashboard: dropdown episode, Play/Pause, 3×3 grid SVG, đèn đổi màu theo phase, ATL chart.

> API chỉ đọc SQLite — không chạy simulator trong request.

---

## Thứ tự chạy đầy đủ

```bash
source ~/CityFlow/.venv/bin/activate
cd /mnt/d/Ki2_nam3/Reinforcement_learning/BTCK

python configs/generate_configs.py
python -m pytest

python src/baseline/max_pressure.py --roadnet configs/roadnet.json --flow configs/flow_low.json --steps 3600

python src/training/train_dqn.py --roadnet configs/roadnet.json --episodes 500   # mixed low/medium/high

python src/evaluate.py --flow configs/flow_medium.json --episodes 50
python src/evaluate.py --flow configs/flow_high.json --episodes 50
python analysis/plot.py

uvicorn api.main:app --port 8000
# terminal khác:
cd web && npm run dev
```

---

## Troubleshooting nhanh

| Lỗi | Fix |
|---|---|
| `externally-managed-environment` | Dùng venv |
| `python: command not found` | `source ~/CityFlow/.venv/bin/activate` |
| `No module named 'cityflow'` | Activate đúng venv, nếu vẫn lỗi thì reinstall từ `~/CityFlow` |
| `No module named 'cmake'` | `pip install "cmake<4"` rồi `pip install . --no-build-isolation` |
| CMake 4 compat error | `pip install --force-reinstall "cmake<4"` |
| pybind11 `PyFrameObject` error | Checkout pybind11 v2.11.1 vào `extern/pybind11` |
| `cannot open roadnet file` | `dir` trong CityFlow config phải kết thúc bằng `/` |
| Web no data | Chạy evaluate trước để có `offline_dataset.db` |
| `models/best.pth` key mismatch | `evaluate.py` có key normalization tự động |

---

## Tiêu chí thành công

| Hạng mục | Điều kiện pass |
|---|---|
| Tests | `python -m pytest` pass |
| Config | roadnet 3×3 + 3 flow files |
| Baseline | ATL + throughput in được |
| DQN | `models/best.pth` tồn tại |
| Dataset | `buffer_dqn.csv`, `buffer_baseline.csv`, `offline_dataset.db` |
| Evaluation | Bảng DQN vs Baseline |
| Analysis | `learning_curve.png`, `atl_comparison.png` |
| API | 4 endpoints chạy được |
| Web | Replay + đèn + chart hoạt động |
