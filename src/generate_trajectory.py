"""
Generate trajectory data for MDUs in a given mode.

Usage:
    python src/generate_trajectory.py --mode mappo --run latest
    python src/generate_trajectory.py --mode mappo --tag 4mdu
    python src/generate_trajectory.py --mode random --mdus 4
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.env import MDUCoverageEnv
from src.config import Config
from src.run_manager import find_run, RunManager


def main():
    _d = Config()

    p = argparse.ArgumentParser(
        description="Generate MDU trajectory. Env defaults from src/config.py.")
    p.add_argument("--mode", choices=["random", "mappo"], default="random")
    p.add_argument("--name", type=str, default=None,
                   help="Trajectory filename prefix (default: same as --mode)")
    p.add_argument("--mdus", type=int, default=_d.env.num_mdus)
    p.add_argument("--steps", type=int, default=_d.env.max_steps)
    p.add_argument("--seed", type=int, default=_d.train.seed)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to MAPPO checkpoint. Auto-detects from --tag if omitted.")

    # Output discovery — only ONE of these is needed:
    p.add_argument("--tag", type=str, default=None,
                   help="Find run by tag (e.g. '4mdu') → latest results/*_<tag>/")
    p.add_argument("--run", type=str, default=None,
                   help="Exact run directory name or path")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override: save trajectory directly to this directory")

    args = p.parse_args()

    name = args.name or args.mode
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = Config.from_args(args, root=root)

    # ── Resolve output directory ──────────────────────────────
    if args.output_dir:
        out_dir = args.output_dir
    else:
        results_root = os.path.join(root, "results")
        run_dir = find_run(results_root, tag=args.tag, run_name=args.run)
        if run_dir:
            out_dir = os.path.join(run_dir, "trajectories")
        else:
            # Fallback: auto-create under latest or results/
            latest = find_run(results_root)
            out_dir = os.path.join(latest, "trajectories") if latest else os.path.join(results_root, "trajectories")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}")

    # ── Build env ─────────────────────────────────────────────
    env = MDUCoverageEnv(**cfg.env_kwargs(root=root))

    # ── Load checkpoint ───────────────────────────────────────
    agent = None
    ckpt = None
    if args.mode == "mappo":
        ckpt = args.checkpoint
        if not ckpt and (args.tag or args.run):
            run_dir = find_run(os.path.join(root, "results"), tag=args.tag, run_name=args.run)
            if run_dir:
                ckpt = RunManager.checkpoint_path(run_dir, "mappo_best.pt")
                if not os.path.exists(ckpt):
                    ckpt = None
        if not ckpt:
            # Last resort: search all runs
            results_root = os.path.join(root, "results")
            if os.path.isdir(results_root):
                for d in sorted(os.listdir(results_root), reverse=True):
                    p = os.path.join(results_root, d, "checkpoints", "mappo_best.pt")
                    if os.path.exists(p):
                        ckpt = p
                        break
        if ckpt and os.path.exists(ckpt):
            from src.agents.mappo import MAPPO
            obs, info = env.reset()
            agent = MAPPO(obs.shape[1], info["global_state"].shape[0], env.max_deg,
                          cfg.to_mappo_config(), device="cpu")
            agent.load(ckpt)
            print(f"Loaded: {ckpt}")
        else:
            print("No checkpoint found, falling back to random")
            args.mode = "random"

    # ── Rollout ───────────────────────────────────────────────
    obs, info = env.reset()
    N = env.asteroid.N_faces
    T = args.steps + 1
    n_mdus = env.num_mdus

    mdu_nodes = np.zeros((T, n_mdus), dtype=int)
    coverage_masks = np.zeros((T, N), dtype=bool)
    coverage_rates = np.zeros(T)
    step_rewards = np.zeros(T)

    mdu_nodes[0] = [mdu.node for mdu in env.mdus]
    coverage_masks[0] = env.coverage_mask.copy()
    coverage_rates[0] = env.coverage_mask.sum() / N
    step_rewards[0] = 0.0

    print(f"Running {args.steps} steps ({args.mode})...")
    if agent is not None:
        agent.reset_hidden(num_mdus=n_mdus)
    prev_actions = np.zeros(n_mdus, dtype=int)
    for step in range(1, args.steps + 1):
        if args.mode == "random":
            mask = info["action_mask"]
            actions = np.array([
                np.random.choice(np.where(mask[i])[0]) if mask[i].any()
                else np.random.randint(env.max_deg)
                for i in range(n_mdus)
            ])
        else:
            cr = float(info.get("coverage_rate", obs[:, 6].mean()))
            actions, _, _ = agent.act(obs, info["global_state"],
                                      action_mask=info["action_mask"], greedy=False,
                                      prev_actions=prev_actions,
                                      coverage_rate=[cr] * n_mdus)
            prev_actions = actions

        obs, reward, terminated, truncated, info = env.step(actions)
        mdu_nodes[step] = [mdu.node for mdu in env.mdus]
        coverage_masks[step] = env.coverage_mask.copy()
        coverage_rates[step] = env.coverage_mask.sum() / N
        step_rewards[step] = reward

        if step % 50 == 0:
            print(f"  Step {step:3d}: coverage={coverage_rates[step]:.2%}, "
                  f"MDUs at {mdu_nodes[step]}")
        if terminated or truncated:
            T = step + 1
            mdu_nodes = mdu_nodes[:T]
            coverage_masks = coverage_masks[:T]
            coverage_rates = coverage_rates[:T]
            step_rewards = step_rewards[:T]
            break

    npz_path = os.path.join(out_dir, f"trajectory_{name}.npz")
    np.savez(npz_path,
             mdu_nodes=mdu_nodes, coverage_masks=coverage_masks,
             coverage_rates=coverage_rates, rewards=step_rewards,
             net_positions=env.net.positions, net_edges=env.net.edges,
             ast_verts=env.asteroid.verts, ast_faces=env.asteroid.faces,
             N_faces=N)
    print(f"Saved: {npz_path}")

    txt_path = os.path.join(out_dir, f"trajectory_{name}.txt")
    with open(txt_path, "w") as f:
        f.write(f"# MDU Trajectory ({args.mode}): {n_mdus} MDUs, {T-1} steps\n")
        f.write(f"# Step  MDU_nodes         Coverage  Reward\n")
        for t in range(T):
            f.write(f"{t:4d}  {' '.join(str(n) for n in mdu_nodes[t])}  "
                    f"{coverage_rates[t]:.4f}  {step_rewards[t]:+.2f}\n")
    print(f"Saved: {txt_path}")

    print(f"\nFinal coverage ({args.mode}): {coverage_rates[-1]:.2%}")


if __name__ == "__main__":
    main()
