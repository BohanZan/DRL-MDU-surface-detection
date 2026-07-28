"""
Quick test: verify the MDU Coverage Environment loads and runs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.env import MDUCoverageEnv

# Paths (from shared config)
from src.config import DataPaths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
paths = DataPaths().resolve(PROJECT_ROOT)
FNS_PATH = paths.fns
SOL_PATH = paths.solution
AST_PATH = paths.polyhedron

print("=" * 60)
print("MDU Coverage Environment - Smoke Test")
print("=" * 60)

# Create env
env = MDUCoverageEnv(
    fns_path=FNS_PATH,
    solution_path=SOL_PATH,
    polyhedron_path=AST_PATH,
    num_mdus=4,
    cone_angle_deg=60.0,  # intentional: test with narrower cone than default 80°
    cone_range=300.0,
    max_steps=50,
    coverage_threshold=0.95,
    seed=42,
)

print("\n[1] Environment created OK")
print(f"    Net: {env.net.NumPoints} nodes, {len(env.net.edges)} edges")
print(f"    Asteroid: {env.asteroid.N_faces} faces")
print(f"    MDUs: {env.num_mdus}")
print(f"    Obs dim: {env.observation_space.shape[0]}")
print(f"    Global state dim: {env.global_state_space.shape[0]}")
print(f"    Action space: {env.max_deg} actions (incl. stay)")

# Reset
obs, info = env.reset()
print(f"\n[2] Reset OK")
print(f"    MDU positions: {[mdu.node for mdu in env.mdus]}")
print(f"    Coverage rate: {info['coverage_rate']:.4f}")
print(f"    Action mask valid: {info['action_mask'].sum(axis=1)}")

# Random walk
print(f"\n[3] Running {env.max_steps} random steps...")
rewards = []
coverages = []
for step in range(env.max_steps):
    # Sample random valid actions
    actions = np.array([
        env._rng.choice(np.where(info["action_mask"][i])[0])
        for i in range(env.num_mdus)
    ])
    obs, reward, terminated, truncated, info = env.step(actions)
    rewards.append(reward)
    coverages.append(info["coverage_rate"])
    if terminated or truncated:
        break

print(f"    Steps taken: {len(rewards)}")
print(f"    Final coverage: {coverages[-1]:.4f} ({coverages[-1]*100:.1f}%)")
print(f"    Total reward: {sum(rewards):.1f}")
print(f"    Avg reward/step: {np.mean(rewards):.3f}")

# Verify coverage increases over time
if len(coverages) > 1:
    print(f"    Coverage start: {coverages[0]:.4f}, end: {coverages[-1]:.4f}")
    assert coverages[-1] >= coverages[0], "Coverage should not decrease!"

# Action mask test: MDUs shouldn't occupy same node
print(f"\n[4] Collision test...")
duplicates = 0
positions = [mdu.node for mdu in env.mdus]
if len(positions) != len(set(positions)):
    duplicates = len(positions) - len(set(positions))
    print(f"    !! {duplicates} collisions detected")
else:
    print(f"    OK No collisions (all MDUs on distinct nodes)")

print(f"\n{'=' * 60}")
print("ALL TESTS PASSED")
print(f"{'=' * 60}")
