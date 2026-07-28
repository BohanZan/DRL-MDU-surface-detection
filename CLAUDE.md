# CLAUDE.md — DRL for MDU Path Planning (IAA 2026)

## Model Configuration
Two models available via DeepSeek API (`api.deepseek.com/anthropic`):
- **`deepseek-v4-pro`** (primary) — slower but better reasoning
- **`deepseek-v4-flash`** (fallback) — faster, for quick iterations

To switch: edit `model` and `env.ANTHROPIC_MODEL` in `.claude/settings.json`
(both must match). Or run: `/config set model deepseek-v4-flash`
then manually update ANTHROPIC_MODEL in settings.json.

## Project Overview
DRL-based cooperative path planning for multiple MDUs (movable detection units)
traversing on a space net enveloping an asteroid (Bennu). Each MDU has a conical
80° FOV sensor. Goal: maximize surface coverage with minimal scan time.

**Current architecture (2026-06-11):**
- Physical-candidate Actor (no stay action — MDUs always move)
- Candidate coverage value (uncovered face fraction per neighbor)
- GRU temporal memory (69→64 GRUCell, sequential BPTT training)
- MAPPO CTDE with centralized critic (14+64=78-dim input)
- Reward: coverage(20× per-face) + exploration(small) + time(90% threshold)
- **Shared configuration** via `src/config.py` — single source of truth for all params
- **Run management** via `src/run_manager.py` — timestamped output directories

Related prior work: AIforSpace2025 — PSO-optimized asteroid capture (C++/MATLAB).

## Project Structure
```
DRL_surface_detection/
├── src/
│   ├── config.py               # ← SINGLE SOURCE OF TRUTH for all parameters
│   ├── run_manager.py          # ← Timestamped output directories
│   ├── agents/mappo.py         # MAPPO + GRU Actor + sequential PPO
│   ├── env/
│   │   ├── mdu_coverage_env.py # Gymnasium env (v5: no stay, candidate value, GRU-compat)
│   │   ├── net_graph.py        # FNS topology + solution loader
│   │   └── asteroid.py         # Asteroid mesh + cone FOV
│   ├── train.py                # Training script (Config-based, RunManager output)
│   ├── generate_trajectory.py  # Decoupled data generation (Config-based)
│   ├── visualize_trajectory.py # Animation render from trajectory NPZ
│   └── render_animation.py     # Standalone cone-FOV renderer
├── checkpoints/                # Legacy — new runs go to results/<timestamp>/checkpoints/
├── results/                    # Stale — new runs create timestamped subdirs here
├── docs/
│   ├── literature-review-and-redesign.md  # Research backing
│   └── superpowers/plans/2026-06-10-gru-temporal-memory.md
└── .claude/settings.json       # Project-level model config
```

## Configuration (src/config.py)

**ALL** project defaults live in `src/config.py`. No other file hardcodes parameter
values. The config is organized as nested dataclasses:

```
Config
├── DataPaths     — fns, solution, polyhedron file paths
├── EnvConfig     — cone (80°), range (300m), thresholds, rewards, start nodes
├── AgentConfig   — lr (3e-4/1e-3), gamma (0.99), epochs (10), hidden (64), etc.
└── TrainConfig   — episodes, seed, device, log_every
```

**To change a default**: edit `src/config.py`. All scripts pick it up automatically.

**To override via CLI**: any argparse script accepts `--cone-angle`, `--mdus`,
`--lr-actor`, `--epochs`, etc. — they layer on top of config defaults.

**To see all current defaults**: `python -c "from src.config import Config; print(Config().dump())"`

## Commands

**Python**: `C:/Users/Lenovo/.conda/envs/comm_python_env/python.exe`
**oneAPI fix**: `set KMP_DUPLICATE_LIB_OK=TRUE` before running

- Train (4 MDUs, default params):
  `python src/train.py --mdus 4 --episodes 1000 --save-plot`
  → outputs to `results/2026-06-11_HHMMSS_4mdu/`

- Quick test (5 episodes):
  `python src/train.py --episodes 5 --max-steps 50 --log-every 1 --tag test`

- Single MDU baseline:
  `python src/train.py --mdus 1 --tag 1mdu`

- Overnight training:
  `python run_overnight.py --mdus 4 --episodes 500`

- Generate trajectory:
  `python src/generate_trajectory.py --mode mappo --checkpoint path/to/model.pt`
  → outputs to `results/<timestamp>/trajectories/`

- Render animation:
  `python src/visualize_trajectory.py --name myrun`
  (auto-finds trajectory in results/ subdirs)

- Standalone cone-FOV renderer:
  `python src/render_animation.py`

**Run output structure** (each run):
```
results/2026-06-11_143052_4mdu/
├── parameters.txt          ← full config dump + CLI overrides
├── checkpoints/
│   ├── mappo_best.pt
│   └── mappo_final.pt
├── trajectories/
│   ├── trajectory_*.npz
│   └── trajectory_*.txt
├── animations/
│   └── animation_*_az000_covXX.gif
└── plots/
    ├── training_curves.png
    └── training_data.npz
```

## Python Environment
- **Conda env:** `comm_python_env` (PyTorch 2.9.1 + CUDA 12.8)
- **Path:** `C:/Users/Lenovo/.conda/envs/comm_python_env/python.exe`
- **oneAPI fix:** `set KMP_DUPLICATE_LIB_OK=TRUE` before running

## Key Architecture Decisions

### Why no stay action?
MDUs must always move — forced exploration prevents policy collapse.
Edge case: if all neighbors occupied, step() falls back to random valid neighbor.

### Why physical-candidate Actor?
Index-based actions have no physical meaning across different nodes.
Instead: each action maps to a candidate position (global + relative + coverage value).
Actor scores positions, not indices. Same candidate encoder shared across all slots.

### Why GRU memory?
Without temporal memory, MDUs oscillate between 2-3 nodes.
GRU hidden state (~25K params) carries trajectory info across steps.
Sequential PPO update (BPTT) preserves temporal dependency in gradients.

### Reward structure (3 components)
1. **Coverage** (main): `20.0 × newly_covered_faces` — per-face, not per-area
2. **Exploration** (small): `1.0 × 0.01` per MDU at low-visit node
3. **Time bonus** (at episode end): `10.0 × (1 - cross_step/max_steps)` if 90% reached

## Current Limitations & Next Steps
- GNN encoder not yet implemented (planned)
- Greedy eval still underperforms stochastic policy
- Need threshold sweep (75%→80%→...→max) to find exploration ceiling
- Single-MDU coverage ceiling confirmed at ~40-50% (physical constraint of one cone)

See `docs/literature-review-and-redesign.md` for the full research-backed roadmap.
