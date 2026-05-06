# Deep Reinforcement Learning Traffic Signal Control Audit

Role: Senior AI Engineer + Reviewer  
Scope: flow/dataset, TrafficEnv, reward, DQN, curriculum, training loop, evaluation, throughput, metrics, and upgrade roadmap.

## Executive Summary

The biggest issue in the current project is not that DQN is too weak. The main problem is the **MDP/environment definition**:

- terminal/done handling is not aligned with a traffic simulation where vehicles can spawn near the end of the episode
- throughput is measured at the wrong time and has an invalid fallback
- reward is not tightly coupled to the selected action/phase
- state is not normalized and misses important traffic features
- phase switching has no yellow/all-red transition and no min-green constraint
- evaluation may write clearance steps into the offline transition dataset

If these issues are not fixed first, the replay buffer will be noisy and the learning signal will be misleading. Fix the environment, reward, and metrics before upgrading the DQN architecture.

## Change Info - 2026-05-06

The current working tree implements the main audit fixes below:

- `TrafficEnv` now separates decision duration from clearance duration. The
  default decision phase is 3600s and the default clearance phase is 900s.
- `done` is emitted once at the decision boundary, and evaluation/training rows
  are intended to be written only during the decision phase.
- Throughput is now treated as completed vehicles. The environment, baseline,
  evaluator, training logs, and replay export also expose `completed`, `active`,
  `generated`, and `completion_rate`.
- State features now include normalized queue counts, downstream queues, waiting
  queues, phase elapsed time, min-green remaining, and current phase one-hot.
- Reward now uses negative absolute pressure for the phase actually applied.
- Min-green support was added through action resolution and action masks.
- Evaluation now runs clearance steps after the decision loop, avoids writing
  clearance transitions, and stores the new metric columns in CSV/SQLite.
- Flow generation now caps generated synthetic vehicles before 2700s and creates
  both legacy low/medium/high flows and demand curriculum flows.

Remaining follow-up:

- Add explicit yellow/all-red transition handling if the roadnet and dashboard
  need realistic signal switching.
- Re-run training/evaluation after these semantics changes before reporting
  new benchmark numbers.

## 1. Flow / Simulator Boundary

### Problems

- Vehicles in the original Gaussian flow spawn almost until the end of the episode: the original file has `startTime` up to around `3596s`, while `TrafficEnv.step()` sets `done = current_time >= 3600`.
- Vehicles that spawn near the episode boundary do not have enough time to leave the network, which makes throughput artificially low and biases ATL.
- Evaluation keeps running for clearance, but still receives `done` from the env. If transitions are recorded after terminal time, the offline dataset becomes poisoned.
- `done=1` can appear multiple times if the simulator keeps running after terminal time.

### Code Locations

- `src/phase1_env_baseline/traffic_env.py`: `step()`, `done` computation.
- `src/phase3_eval/evaluate.py`: `_run_dqn()`, `_run_baseline()`, `MAX_EVAL_STEPS` loops.
- `configs/generate_configs.py`: generated flows spawn until `2700s`, while the original Gaussian flow spawns until nearly `3600s`.

### Recommended Fix

Separate the decision phase from the clearance phase:

```python
DECISION_DURATION = 3600
CLEARANCE_DURATION = 900
TOTAL_SIM_DURATION = DECISION_DURATION + CLEARANCE_DURATION
```

- During the decision phase: the agent selects actions, rewards are computed, replay is written, and training is performed.
- During the clearance phase: only run the simulator to compute final metrics. Do not write replay and do not train.
- Do not record repeated `done=1` transitions after the terminal transition.
- In replay data, `done` should only be true for the final decision-phase transition.

## 2. Throughput

### Problems

- Throughput being zero for many early steps is normal because vehicles have not reached their destinations yet.
- Low throughput for the full episode can be caused by late spawning, long routes, missing clearance time, or incorrect completed-vehicle counting.
- The current code falls back to active vehicle count if `get_finished_vehicle_count()` is unavailable. Active vehicle count is not throughput.

### Code Locations

- `src/phase1_env_baseline/traffic_env.py`: `throughput_getter = getattr(...); fallback get_vehicle_count()`.
- `src/phase1_env_baseline/max_pressure.py`: `run_baseline()`.
- `src/phase4_export/export_replay.py`: `_get_throughput()`.

### Recommended Fix

- Throughput should only mean finished/completed vehicles.
- If CityFlow does not expose a finished-count API, manually track vehicle ids:
  - keep a set of vehicles that have appeared
  - read active vehicle ids at each step
  - vehicles that previously appeared and are no longer active should be counted as completed, excluding vehicles that have not spawned yet
- Use throughput only as an end-of-episode evaluation metric, not as the primary reward.
- Report separate metrics:
  - completed vehicles
  - active vehicles
  - generated vehicles
  - completion rate = completed / generated

## 3. Reward Function

### Problems

- `TrafficEnv._get_rewards()` currently computes `sum(abs(in_lane - out_lane))` over all movements of an intersection.
- This reward is not directly tied to the selected phase/action, which weakens credit assignment.
- The MaxPressure baseline uses signed phase pressure: `sum(q_in - q_out)`. The env reward and baseline are not consistent.
- If reward is computed across movements from all phases, the same lane can be double-counted across phases, distorting the reward scale.
- `test_reward_is_negative_max_pressure` expects signed cancellation, but the current code was changed to absolute pressure.

### Code Locations

- `src/phase1_env_baseline/traffic_env.py`: `_get_rewards()`.
- `src/phase1_env_baseline/max_pressure.py`: `phase_pressures()`.
- `tests/phase2/test_traffic_env.py`: `test_reward_is_negative_max_pressure()`.

### Recommended Fix

Choose one reward definition and make it consistent across env, baseline, tests, and reports.

Option A - selected-phase reward:

```text
reward_i = - pressure_of_selected_phase
```

Where phase pressure is computed from the movements served by the selected phase:

```text
pressure(phase) = sum(q_in_movement - q_out_movement)
```

Option B - PressLight-style dense reward:

```text
reward_i = - sum |queue_in_movement - queue_out_movement|
```

But avoid lane/movement double-counting and keep it consistent with action selection and the baseline.

Recommended direction:

- Baseline action: `argmax` signed phase pressure.
- RL reward: negative absolute intersection pressure or negative selected-phase pressure, but log each reward component separately.
- Add tests for:
  - downstream congestion
  - selected action pressure
  - no double-counting
  - reward scale under low and high demand

## 4. State Representation

### Problems

- The current state contains incoming queues, outgoing queues, and one-hot current phase, but it is not normalized.
- Raw queue counts can be large, while the phase one-hot vector is only `[0, 1]`. This creates input scale imbalance.
- Important features are missing, making the MDP less Markovian:
  - phase elapsed time
  - min-green remaining/action mask
  - waiting time
  - lane occupancy
  - speed
  - downstream queue/spillback

### Code Location

- `src/phase1_env_baseline/traffic_env.py`: `_get_states()`.

### Recommended Fix

Normalize queues:

```python
queue_norm = min(queue / max_queue, 1.0)
```

Add features:

- current phase one-hot
- normalized phase elapsed time
- normalized incoming lane queue/vehicle count
- normalized outgoing/downstream queue
- incoming lane waiting time
- lane speed or occupancy
- optional neighbor pressure for coordination

## 5. Yellow Phase / Min Green

### Problems

- `step()` sets the new phase immediately with `engine.set_tl_phase()`.
- There is no yellow/all-red transition.
- There is no min-green constraint.
- The agent can spam phase switching, creating unrealistic behavior and potentially fake reward/ATL improvements.

### Code Locations

- `src/phase1_env_baseline/traffic_env.py`: `step()`.
- `src/phase1_env_baseline/phase_map.py`: phase map currently only reads phases from the roadnet.

### Recommended Fix

- Add `yellow_time = 3s`.
- Add `min_green = 10s`.
- Track `current_phase` and `phase_elapsed`.
- If min-green is not satisfied:
  - apply an action mask that prevents switching, or
  - force the previous phase to remain active.
- If the phase changes:
  - apply yellow/all-red for `yellow_time`
  - either exclude yellow seconds from RL reward or add a small switching penalty

## 6. DQN Implementation / Training

### What Is Correct

- Replay buffer exists.
- Target network exists.
- Huber loss is used.
- Gradient clipping is used.
- Terminal handling is included in the Bellman target.

### Problems / Risks

- The implementation is still vanilla DQN, so it is prone to overestimation bias.
- The epsilon decay comment says decay should happen every environment step, but training calls it once per episode.
- `train_dqn.py` has a global-step target update block, but it only runs if the agent has `update_target_network`; the current class does not define this method. In practice, target sync happens inside `agent.update()` based on `update_count`.
- Keeping the replay buffer across curriculum stages can be useful, but without stratified sampling stale replay can hurt learning under strong demand shifts.
- The best checkpoint is selected by `avg_reward`; if reward is wrong or scenario-dependent in scale, this criterion can be misleading.

### Code Locations

- `src/phase2_dqn/dqn_agent.py`: `SharedDQNAgent.update()`, `decay_epsilon()`.
- `src/phase2_dqn/train_dqn.py`: training loop, `agent.decay_epsilon()`, checkpoint save.

### Recommended Fix

- Decide whether epsilon decays once per episode or once per env step, then make the comment, code, and logs consistent.
- If target update should be based on global step, add:

```python
def update_target_network(self):
    self.target_network.load_state_dict(self.q_network.state_dict())
```

and use:

```python
if global_step % TARGET_UPDATE_FREQ == 0:
    agent.update_target_network()
```

- Compute average reward using the actual number of executed steps:

```python
avg_reward = episode_reward / max(1, actual_steps * env.n_agents)
```

- After the environment is stable, upgrade in this order:
  - Double DQN
  - Dueling DQN
  - Prioritized Replay
  - n-step return

## 7. Curriculum Learning

### Problems / Risks

- Generated flows only contain 300/600/900 vehicles, while the original Gaussian flow contains around 8412 vehicles. The demand jump is too large.
- Moving from light generated flows to the heavy Gaussian flow can cause distribution shift.
- Keeping replay from old scenarios is useful, but without scenario tags or stratified sampling, old transitions can add noise.

### Code Locations

- `src/phase2_dqn/train_dqn.py`: `_select_flow()`.
- `configs/generate_configs.py`: generated demand levels.

### Recommended Fix

Increase demand more smoothly:

```text
300 -> 900 -> 1800 -> 3600 -> 6000 -> 8412
```

Use soft curriculum:

- 75% current flow
- 25% older flows for rehearsal

Add scenario metadata to replay:

- `flow_scenario`
- `demand_level`
- `traffic_pattern`

If PER or stratified replay is added, sample old and new scenarios in a controlled ratio.

## 8. Evaluation

### Problems

- If a model cannot be loaded, evaluation still runs with untrained weights. The summary can become misleading.
- `_run_dqn()` and `_run_baseline()` hard-code `MAX_EVAL_STEPS = 720`, making CLI `--steps` ineffective.
- If clearance steps are written as transitions, the offline dataset is poisoned.
- Evaluation should be fair between DQN and baseline: same flow, seed, duration, clearance, and metric definitions.

### Code Location

- `src/phase3_eval/evaluate.py`: `_load_dqn_model()`, `_run_dqn()`, `_run_baseline()`.

### Recommended Fix

- Fail hard if the model cannot be loaded:

```python
raise FileNotFoundError(model_path)
```

- DQN evaluation must enforce:

```python
agent.epsilon = 0.0
agent.q_network.eval()
```

- Use `steps_per_episode` instead of hard-coded constants.
- Separate decision transitions from clearance metrics.
- Run evaluation with multiple seeds and report mean +/- std.
- Summary should loudly warn if a model failed to load or if throughput counting is invalid.

## 9. Dataset Logging / Offline Dataset

### Problems / Risks

- `state/action/reward/next_state/done` fields exist, but `done` may be wrong if transitions are recorded after terminal time.
- `throughput` and `atl` are logged per step, but they are simulator-level aggregate metrics, not transition-level rewards.
- The offline DB can combine baseline and DQN transitions, but clearance transitions must not be used for offline RL.

### Recommended Fix

Add columns:

- `is_decision_phase`
- `is_clearance_phase`
- `sim_time`
- `flow_scenario`
- `seed`
- `episode_terminal_reason`

Only use transitions with `is_decision_phase=True` for offline RL.

Log end-of-episode metrics in a separate `episode_metrics.csv`.

## 10. Analysis / Tests / API Drift

### Problems

- `analysis/plot_training.py` says it reads `training_log.csv`, but the code reads `row["reward"]`; the current training log writes `avg_reward`.
- `tests/phase3/test_evaluate.py` calls `evaluate_module.evaluate(...)`, but the current file exposes `evaluate_multiple(...)`. The test is out of sync with the API.
- Several CLI defaults point to files that do not exist:
  - `configs/roadnet.json`
  - `configs/flow_medium.json`
  - `models/best.pth`

### Recommended Fix

- Update the plot script to read `avg_reward`.
- Update tests for `evaluate_multiple()` or add a backward-compatible `evaluate()` wrapper.
- Default CLI paths should point to:
  - `configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json`
  - an existing flow under `configs/`
  - a model path that matches the training script output

## Top 10 Most Serious Issues

1. Incorrect `done`/episode boundary with vehicles spawning near the end.
2. No separate clearance phase for metrics.
3. Throughput fallback uses active vehicle count.
4. Reward is not directly tied to the selected action/phase.
5. Env reward and MaxPressure baseline are inconsistent.
6. State is not normalized.
7. No yellow/all-red and no min-green constraint.
8. Evaluation hard-codes steps and may record transitions after terminal time.
9. Evaluation can run untrained weights when model loading fails.
10. Curriculum demand jump is too large, from 900 vehicles to 8412 vehicles.

## Top 10 Highest-Impact Improvements

1. Separate decision duration and clearance duration.
2. Fix completed-vehicle counting.
3. Redesign reward using selected-phase pressure or PressLight-style pressure.
4. Normalize state and add phase elapsed, downstream, and waiting-time features.
5. Add yellow time and min-green/action mask.
6. Align env reward, baseline, tests, and reports.
7. Fail hard when checkpoint loading fails.
8. Evaluate across multiple seeds with mean/std.
9. Use smoother curriculum scaling and scenario-tagged replay.
10. Upgrade Double DQN -> Dueling -> PER -> n-step -> PressLight/MPLight.

## Why Throughput Is Low

Throughput can be low even when the model is not necessarily bad because:

- vehicles spawn near the end of the episode
- routes are long, so vehicles do not reach destinations within 3600s
- there is no clearance phase
- the traffic grid experiences spillback/deadlock
- completed count incorrectly falls back to active count
- training and evaluation use different durations

Report completion rate in addition to raw throughput.

## Why Reward Can Be Misleading

- Aggregate pressure is not tied to the selected action, so DQN gets weak credit assignment.
- Absolute pressure can punish global congestion without identifying which action was wrong.
- Reward scale changes with demand, so best checkpoint by average reward can prefer easier flows.
- Reward does not directly include delay, waiting time, or spillback, so it may not correlate well with ATL.

## Why DQN Can Be Unstable

- Raw queue-count inputs are not normalized.
- Vanilla DQN has overestimation bias.
- Reward magnitude becomes large and negative under high demand.
- Replay mixes scenarios with distribution shift.
- There is no Double DQN, PER, or n-step return for delayed traffic effects.
- Multi-agent shared DQN learns from local transitions while the environment is non-stationary because the other 8 agents also change actions.

## Recommended Architecture Upgrades

Recommended order:

1. Fix environment/reward/metrics first.
2. Add Double DQN to reduce overestimation.
3. Add Dueling DQN to separate state value and action advantage.
4. Add Prioritized Replay to learn from rare jam/spillback transitions.
5. Add n-step return to capture delayed green-wave effects.
6. Move toward PressLight-style reward/state.
7. Use FRAP if phase symmetry encoding is needed.
8. Use MPLight/coordination methods for serious multi-intersection scaling.
9. Try PPO only after the MDP is stable and DQN/PressLight baselines are strong.
10. Rainbow DQN is a reasonable package if staying with value-based RL.

## Recommended Fix Order

### Phase 1 - Mandatory Fixes

1. Fix `done`.
2. Add a real clearance phase.
3. Fix throughput counting.
4. Do not write replay during clearance.
5. Fix reward based on selected phase/action.

### Phase 2 - Fix the MDP

6. Normalize state.
7. Add yellow phase.
8. Add min-green.
9. Add downstream information.
10. Align reward with baseline/tests.

### Phase 3 - Fix Training/Evaluation

11. Fix epsilon decay comment/logic.
12. Fix target update API/logic.
13. Fix avg_reward and checkpoint criteria.
14. Fail if model loading fails.
15. Evaluate with multiple seeds.

### Phase 4 - Upgrade the Agent

16. Double DQN.
17. Dueling DQN.
18. Prioritized Replay.
19. n-step return.
20. PressLight/MPLight-style multi-agent coordination.

## Current Score

- Overall project score: 5.5/10.
- RL theory correctness: 5/10.
- Production readiness: 3/10.
- Research quality: 4/10.
- Paper-readiness: low, because environment/evaluation need to be fixed and serious ablations are required.

## Roadmap

### Beginner

- Fix done/clearance/throughput.
- Fix reward and related tests.
- Fix CLI defaults and plot scripts.
- Log end-of-episode metrics separately.

### Intermediate

- Normalize state.
- Add yellow/min-green.
- Evaluate across multiple seeds.
- Add Double DQN + Dueling DQN.
- Add scenario-tagged replay.

### Research-Grade

- Use PressLight reward/state formulation.
- Add PER + n-step/Rainbow DQN ablations.
- Add FRAP/MPLight-style coordination.
- Benchmark against MaxPressure, FixedTime, DQN, and PressLight using identical flows/seeds.
- Report mean/std, confidence intervals, significance tests, and ablations for reward/state/curriculum.
