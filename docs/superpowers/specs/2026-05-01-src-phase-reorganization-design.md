# Design: Tái cấu trúc src/ theo phase — 2026-05-01

## Mục tiêu

Tổ chức lại `src/` từ cấu trúc theo loại module sang cấu trúc theo phase để dễ track tiến độ theo `plan.md`.

## Cấu trúc mới

```
src/
├── __init__.py
├── phase1_env_baseline/
│   ├── __init__.py
│   ├── traffic_env.py        ← từ src/env/traffic_env.py
│   ├── phase_map.py          ← từ src/env/phase_map.py
│   └── max_pressure.py       ← từ src/baseline/max_pressure.py
├── phase2_dqn/
│   ├── __init__.py
│   ├── dqn_agent.py          ← từ src/agent/dqn_agent.py
│   └── train_dqn.py          ← từ src/training/train_dqn.py
├── phase3_eval/
│   ├── __init__.py
│   └── evaluate.py           ← từ src/evaluate.py (root)
└── phase4_export/
    ├── __init__.py
    └── export_replay.py      ← từ src/export/export_replay.py
```

## Import changes

| File | Old import | New import |
|---|---|---|
| `phase2_dqn/train_dqn.py` | `src.agent.dqn_agent` | `src.phase2_dqn.dqn_agent` |
| `phase2_dqn/train_dqn.py` | `src.env.phase_map`, `src.env.traffic_env` | `src.phase1_env_baseline.*` |
| `phase3_eval/evaluate.py` | `src.agent.dqn_agent` | `src.phase2_dqn.dqn_agent` |
| `phase3_eval/evaluate.py` | `src.baseline.max_pressure`, `src.env.*` | `src.phase1_env_baseline.*` |
| `phase4_export/export_replay.py` | `src.baseline.max_pressure`, `src.env.phase_map` | `src.phase1_env_baseline.*` |
| `tests/test_*.py` | `src.agent.*`, `src.env.*`, `src.baseline.*`, `src.training.*` | tương ứng theo phase |
| `tests/phase*/` | same as above | same mapping |

## sys.path depth fixes

- `phase2_dqn/train_dqn.py`: `parents[3]` → `parents[2]`
- `phase4_export/export_replay.py`: `parents[3]` → `parents[2]`
- `phase3_eval/evaluate.py`: `parents[1]` → `parents[2]`

## Cleanup

- Xóa `src/__pycache__/` (pyc cũ không còn source)
- Xóa các thư mục cũ sau khi move: `src/env/`, `src/agent/`, `src/baseline/`, `src/training/`, `src/export/`

## Không thay đổi

- Logic và nội dung trong tất cả file `.py` giữ nguyên hoàn toàn
- Tên file giữ nguyên
- `plan.md`, `CLAUDE.md`, `analysis/`, `api/`, `web/`, `configs/`, `tests/` không bị ảnh hưởng (ngoài import fixes trong tests)
