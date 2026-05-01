# src/ Phase Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tái cấu trúc `src/` từ subpackage theo loại module sang 4 thư mục theo phase để dễ track tiến độ với `plan.md`.

**Architecture:** Move các file `.py` vào thư mục phase tương ứng, cập nhật toàn bộ import trong source + test files. Không thay đổi logic hay nội dung file.

**Tech Stack:** Python, pytest

---

## File Map

| From | To |
|---|---|
| `src/env/traffic_env.py` | `src/phase1_env_baseline/traffic_env.py` |
| `src/env/phase_map.py` | `src/phase1_env_baseline/phase_map.py` |
| `src/baseline/max_pressure.py` | `src/phase1_env_baseline/max_pressure.py` |
| `src/agent/dqn_agent.py` | `src/phase2_dqn/dqn_agent.py` |
| `src/training/train_dqn.py` | `src/phase2_dqn/train_dqn.py` |
| `src/evaluate.py` | `src/phase3_eval/evaluate.py` |
| `src/export/export_replay.py` | `src/phase4_export/export_replay.py` |

---

### Task 1: Tạo thư mục phase và move file phase1

**Files:**
- Create: `src/phase1_env_baseline/__init__.py`
- Move: `src/env/traffic_env.py` → `src/phase1_env_baseline/traffic_env.py`
- Move: `src/env/phase_map.py` → `src/phase1_env_baseline/phase_map.py`
- Move: `src/baseline/max_pressure.py` → `src/phase1_env_baseline/max_pressure.py`

- [ ] **Step 1: Tạo thư mục và __init__.py**

```bash
mkdir -p src/phase1_env_baseline
touch src/phase1_env_baseline/__init__.py
```

- [ ] **Step 2: Copy file vào phase1 (giữ original để import không bị broken ngay)**

```bash
cp src/env/traffic_env.py src/phase1_env_baseline/traffic_env.py
cp src/env/phase_map.py src/phase1_env_baseline/phase_map.py
cp src/baseline/max_pressure.py src/phase1_env_baseline/max_pressure.py
```

- [ ] **Step 3: Xác nhận file đã tồn tại**

```bash
ls src/phase1_env_baseline/
```

Expected output: `__init__.py  max_pressure.py  phase_map.py  traffic_env.py`

---

### Task 2: Move file phase2 và phase3

**Files:**
- Create: `src/phase2_dqn/__init__.py`
- Create: `src/phase3_eval/__init__.py`
- Move: `src/agent/dqn_agent.py` → `src/phase2_dqn/dqn_agent.py`
- Move: `src/training/train_dqn.py` → `src/phase2_dqn/train_dqn.py`
- Move: `src/evaluate.py` → `src/phase3_eval/evaluate.py`

- [ ] **Step 1: Tạo thư mục**

```bash
mkdir -p src/phase2_dqn src/phase3_eval
touch src/phase2_dqn/__init__.py src/phase3_eval/__init__.py
```

- [ ] **Step 2: Copy file**

```bash
cp src/agent/dqn_agent.py src/phase2_dqn/dqn_agent.py
cp src/training/train_dqn.py src/phase2_dqn/train_dqn.py
cp src/evaluate.py src/phase3_eval/evaluate.py
```

- [ ] **Step 3: Xác nhận**

```bash
ls src/phase2_dqn/ && ls src/phase3_eval/
```

Expected: `__init__.py  dqn_agent.py  train_dqn.py` và `__init__.py  evaluate.py`

---

### Task 3: Move file phase4

**Files:**
- Create: `src/phase4_export/__init__.py`
- Move: `src/export/export_replay.py` → `src/phase4_export/export_replay.py`

- [ ] **Step 1: Tạo thư mục và copy**

```bash
mkdir -p src/phase4_export
touch src/phase4_export/__init__.py
cp src/export/export_replay.py src/phase4_export/export_replay.py
```

- [ ] **Step 2: Xác nhận**

```bash
ls src/phase4_export/
```

Expected: `__init__.py  export_replay.py`

---

### Task 4: Cập nhật import trong phase2_dqn/train_dqn.py

**Files:**
- Modify: `src/phase2_dqn/train_dqn.py`

- [ ] **Step 1: Cập nhật sys.path depth và import**

Trong `src/phase2_dqn/train_dqn.py`, tìm và thay đổi:

```python
# Trước (dòng 18):
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.agent.dqn_agent import SharedDQNAgent
from src.env.phase_map import build_phase_map
from src.env.traffic_env import TrafficEnv
```

Thành:

```python
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2_dqn.dqn_agent import SharedDQNAgent
from src.phase1_env_baseline.phase_map import build_phase_map
from src.phase1_env_baseline.traffic_env import TrafficEnv
```

- [ ] **Step 2: Verify import syntax đúng**

```bash
python -c "import ast; ast.parse(open('src/phase2_dqn/train_dqn.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 5: Cập nhật import trong phase3_eval/evaluate.py

**Files:**
- Modify: `src/phase3_eval/evaluate.py`

- [ ] **Step 1: Cập nhật sys.path depth và import**

Trong `src/phase3_eval/evaluate.py`, thay đổi:

```python
# Trước (dòng 14-16):
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.dqn_agent import SharedDQNAgent
from src.baseline.max_pressure import MaxPressureBaseline
from src.env.phase_map import build_phase_map
from src.env.traffic_env import TrafficEnv
```

Thành:

```python
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2_dqn.dqn_agent import SharedDQNAgent
from src.phase1_env_baseline.max_pressure import MaxPressureBaseline
from src.phase1_env_baseline.phase_map import build_phase_map
from src.phase1_env_baseline.traffic_env import TrafficEnv
```

- [ ] **Step 2: Verify**

```bash
python -c "import ast; ast.parse(open('src/phase3_eval/evaluate.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 6: Cập nhật import trong phase4_export/export_replay.py

**Files:**
- Modify: `src/phase4_export/export_replay.py`

- [ ] **Step 1: Cập nhật sys.path depth và import**

Trong `src/phase4_export/export_replay.py`, thay đổi:

```python
# Trước (dòng 13-15):
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.baseline.max_pressure import MaxPressureBaseline, _write_cityflow_config, cityflow
from src.env.phase_map import build_phase_map
```

Thành:

```python
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase1_env_baseline.max_pressure import MaxPressureBaseline, _write_cityflow_config, cityflow
from src.phase1_env_baseline.phase_map import build_phase_map
```

- [ ] **Step 2: Verify**

```bash
python -c "import ast; ast.parse(open('src/phase4_export/export_replay.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 7: Cập nhật import trong test files (root tests/)

**Files:**
- Modify: `tests/test_max_pressure_baseline.py`
- Modify: `tests/test_dqn_agent.py`
- Modify: `tests/test_train_dqn.py`
- Modify: `tests/test_traffic_env.py`
- Modify: `tests/test_evaluate.py`

- [ ] **Step 1: test_max_pressure_baseline.py**

Thay:
```python
from src.baseline.max_pressure import MaxPressureBaseline
from src.env.phase_map import build_phase_map
```
Thành:
```python
from src.phase1_env_baseline.max_pressure import MaxPressureBaseline
from src.phase1_env_baseline.phase_map import build_phase_map
```

- [ ] **Step 2: test_dqn_agent.py**

Thay:
```python
from src.agent.dqn_agent import DenseLayer, QNetwork, ReplayBuffer, SharedDQNAgent, huber_loss
```
Thành:
```python
from src.phase2_dqn.dqn_agent import DenseLayer, QNetwork, ReplayBuffer, SharedDQNAgent, huber_loss
```

- [ ] **Step 3: test_train_dqn.py**

Thay:
```python
from src.training import train_dqn as train_module
```
Thành:
```python
from src.phase2_dqn import train_dqn as train_module
```

- [ ] **Step 4: test_traffic_env.py**

Thay:
```python
from src.env.traffic_env import TrafficEnv
```
Thành:
```python
from src.phase1_env_baseline.traffic_env import TrafficEnv
```

- [ ] **Step 5: test_evaluate.py**

Thay:
```python
from src import evaluate as evaluate_module
```
Thành:
```python
from src.phase3_eval import evaluate as evaluate_module
```

---

### Task 8: Cập nhật import trong tests/phase*/ subdirectories

**Files:**
- Modify: `tests/phase1/test_max_pressure_baseline.py`
- Modify: `tests/phase2/test_traffic_env.py`
- Modify: `tests/phase2/test_dqn_agent.py`
- Modify: `tests/phase2/test_train_dqn.py`
- Modify: `tests/phase3/test_evaluate.py`

- [ ] **Step 1: tests/phase1/test_max_pressure_baseline.py**

Thay:
```python
from src.baseline.max_pressure import MaxPressureBaseline
from src.env.phase_map import build_phase_map
```
Thành:
```python
from src.phase1_env_baseline.max_pressure import MaxPressureBaseline
from src.phase1_env_baseline.phase_map import build_phase_map
```

- [ ] **Step 2: tests/phase2/test_traffic_env.py**

Thay:
```python
from src.env.traffic_env import TrafficEnv
```
Thành:
```python
from src.phase1_env_baseline.traffic_env import TrafficEnv
```

- [ ] **Step 3: tests/phase2/test_dqn_agent.py**

Thay:
```python
from src.agent.dqn_agent import DenseLayer, QNetwork, ReplayBuffer, SharedDQNAgent, huber_loss
```
Thành:
```python
from src.phase2_dqn.dqn_agent import DenseLayer, QNetwork, ReplayBuffer, SharedDQNAgent, huber_loss
```

- [ ] **Step 4: tests/phase2/test_train_dqn.py**

Thay:
```python
from src.training import train_dqn as train_module
```
Thành:
```python
from src.phase2_dqn import train_dqn as train_module
```

- [ ] **Step 5: tests/phase3/test_evaluate.py** — xem nội dung trước khi sửa

Đọc file `tests/phase3/test_evaluate.py` và cập nhật import tương tự `tests/test_evaluate.py` (thay `from src import evaluate` thành `from src.phase3_eval import evaluate`).

---

### Task 9: Xóa thư mục cũ và __pycache__ rác

**Files:**
- Delete: `src/env/`, `src/agent/`, `src/baseline/`, `src/training/`, `src/export/`
- Delete: `src/__pycache__/` (chứa pyc từ file không còn tồn tại)

- [ ] **Step 1: Xóa __pycache__ rác ở root src/**

```bash
rm -rf src/__pycache__
```

- [ ] **Step 2: Xóa các subpackage cũ**

```bash
rm -rf src/env src/agent src/baseline src/training src/export
```

- [ ] **Step 3: Xác nhận cấu trúc mới**

```bash
find src -name "*.py" | sort
```

Expected output (7 file + 5 `__init__.py`):
```
src/__init__.py
src/phase1_env_baseline/__init__.py
src/phase1_env_baseline/max_pressure.py
src/phase1_env_baseline/phase_map.py
src/phase1_env_baseline/traffic_env.py
src/phase2_dqn/__init__.py
src/phase2_dqn/dqn_agent.py
src/phase2_dqn/train_dqn.py
src/phase3_eval/__init__.py
src/phase3_eval/evaluate.py
src/phase4_export/__init__.py
src/phase4_export/export_replay.py
```

---

### Task 10: Chạy test suite để xác nhận không regression

- [ ] **Step 1: Chạy pytest**

```bash
cd /mnt/d/Ki2_nam3/Reinforcement_learning/BTCK
source ~/CityFlow/.venv/bin/activate
python -m pytest tests/ -v --tb=short 2>&1 | head -80
```

Expected: Các test không liên quan đến CityFlow (unit tests, mock-based tests) pass. Tests cần CityFlow thực có thể skip/xfail nếu CityFlow không available.

- [ ] **Step 2: Nếu có test fail do import, fix từng file theo error message**

Ví dụ nếu thấy `ModuleNotFoundError: No module named 'src.phase3_eval'` → kiểm tra lại Task 9 đã xóa nhầm file chưa.

- [ ] **Step 3: Commit**

```bash
git add src/ tests/
git commit -m "refactor: reorganize src/ into phase-based directories

Move modules from type-based subpackages (env/, agent/, baseline/,
training/, export/) into 4 phase directories matching plan.md phases.
Update all imports in src/ and tests/ accordingly."
```

---

### Task 11: Cập nhật CLAUDE.md project structure section

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Cập nhật Project Structure section**

Tìm đoạn trong CLAUDE.md:
```
│   ├── env/                    # simulator environment + phase-map utilities
│   │   ├── traffic_env.py      # TrafficEnv wrapping CityFlow
│   │   └── phase_map.py        # build_phase_map — reads roadnet.json
│   ├── agent/                  # DQN components
│   │   └── dqn_agent.py        # SharedDQNAgent + ReplayBuffer + QNetwork + huber_loss
│   ├── baseline/               # rule-based controller (no ML)
│   │   └── max_pressure.py     # MaxPressureBaseline + run_baseline
│   ├── training/               # training loop
│   │   └── train_dqn.py        # 500-episode training loop
│   └── export/                 # dashboard data exporter
│       └── export_replay.py    # CityFlow → web/data/*.json
```

Thay thành:
```
│   ├── phase1_env_baseline/    # Phase 1 — simulator env + rule-based baseline
│   │   ├── traffic_env.py      # TrafficEnv wrapping CityFlow
│   │   ├── phase_map.py        # build_phase_map — reads roadnet.json
│   │   └── max_pressure.py     # MaxPressureBaseline + run_baseline
│   ├── phase2_dqn/             # Phase 2 — DQN agent + training loop
│   │   ├── dqn_agent.py        # SharedDQNAgent + ReplayBuffer + QNetwork + huber_loss
│   │   └── train_dqn.py        # 500-episode training loop
│   ├── phase3_eval/            # Phase 3 — evaluation + offline dataset builder
│   │   └── evaluate.py         # DQN vs Baseline, writes CSV + SQLite
│   └── phase4_export/          # Phase 4 — dashboard data exporter
│       └── export_replay.py    # CityFlow → web/data/*.json
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md project structure to reflect phase layout"
```
