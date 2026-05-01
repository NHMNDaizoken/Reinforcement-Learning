# Project Notes

Đọc file này trước khi bắt đầu run mới hoặc sửa code đáng kể.

---

## Quyết định thiết kế (không thay đổi)

- CityFlow, không chuyển SUMO
- MaxPressure baseline thuần rule-based, không có ML
- 9 agents dùng chung 1 DQN (parameter sharing bắt buộc)
- DQN explicit: `DenseLayer` thủ công, forward pass tường minh, Huber loss viết tay
- API + dashboard chỉ đọc SQLite/CSV log, không chạy simulator trong request

## Lịch sử thay đổi quan trọng

| Date | Thay đổi | Files |
|---|---|---|
| 2026-04-30 | CityFlow config: `dir` phải có trailing `/`; roadnet cần virtual boundary intersections | `configs/generate_configs.py`, `configs/roadnet.json` |
| 2026-04-30 | Expo web: dùng `web:headless` script (`CI=1 BROWSER=none EXPO_UNSTABLE_HEADLESS=1`) | `web/package.json` |
| 2026-05-01 | DQN rebuild: explicit `DenseLayer`, không dùng `nn.Sequential` | `src/agent/dqn_agent.py` |
| 2026-05-01 | Src reorganize: `env/`, `agent/`, `baseline/`, `training/`, `export/` sub-packages | toàn bộ `src/` |
| 2026-05-01 | evaluate.py có key normalization để load checkpoint cũ (`model.0.*` → `DenseLayer`) | `src/evaluate.py` |
| 2026-05-01 | DQN fix: DenseLayer weight shape đổi sang (out, in) — PyTorch convention; epsilon decay tách ra khỏi `update()`, gọi 1 lần/env step trong training loop; thêm gradient clipping norm=10; `select_actions` batch thành 1 forward pass | `src/agent/dqn_agent.py`, `src/training/train_dqn.py`, `tests/test_dqn_agent.py` |

---

## Lỗi thường gặp & fix

| Lỗi / Triệu chứng | Root Cause | Fix |
|---|---|---|
| `git: detected dubious ownership` | Repo owner khác sandbox user | `git -c safe.directory=<path> ...` |
| `uv: Access is denied (cache)` | uv cache mặc định ở AppData bị denied | Set `UV_CACHE_DIR=.uv-cache` trong workspace |
| `uv: Failed to query Python314\python.exe` | uv inspect Python cấp admin | Set `UV_PYTHON=.venv-win\Scripts\python.exe` |
| `pytest PermissionError WinError 5 (AppData\Temp)` | pytest temp root bị denied | `--basetemp=.agent_tmp\pytest-base -o cache_dir=.agent_tmp\pytest-cache` |
| `ModuleNotFoundError: No module named 'numpy'` khi pytest | uv isolated env thiếu runtime deps | `uv run --with pytest --with numpy --with torch python -m pytest` |
| `cannot open roadnet file` (CityFlow) | CityFlow concat `dir + file` không có `/` | `dir` trong config JSON phải kết thúc bằng `/` |
| CityFlow build: `No module named 'cmake'` | pip build isolation giấu venv cmake | `pip install . --no-build-isolation` |
| CityFlow build: CMake 4 compat error | CMake 4 bỏ policy cũ | `pip install --force-reinstall "cmake<4"` |
| CityFlow build: pybind11 `PyFrameObject` | pybind11 v2.3.0 không hỗ trợ Python 3.11 | Checkout pybind11 v2.11.1 vào `extern/pybind11` |
| `models/best.pth` key mismatch (`model.0.*`) | Checkpoint từ `nn.Sequential` cũ | `evaluate.py` tự normalize key — không cần làm gì thêm |
| `Wsl/Service/CreateInstance/E_ACCESSDENIED` | Sandbox không có quyền start WSL | Chạy evaluate trong môi trường WSL cho phép |
| `spawn EPERM` khi `Start-Process npm` | Windows npm shim không chạy được qua Start-Process | Dùng `npm.cmd` thay vì `npm` |
| `--non-interactive is not supported` (Expo) | Expo SDK 55 đổi flag | Dùng `CI=1` thay thế |

---

## Cross-run notes

- Trước khi debug: search file này theo tên lỗi / module / exception trước
- Trước khi implement: đọc `plan.md`
- Sau khi fix lỗi mới: append vào bảng "Lỗi thường gặp" ở trên
