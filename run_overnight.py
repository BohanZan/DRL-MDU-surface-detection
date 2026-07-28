"""
Overnight training with file logging.

All configuration comes from src/config.py. Override via CLI arguments.

Usage:
    python run_overnight.py
    python run_overnight.py --mdus 4 --episodes 1000 --epochs 5
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from src.env import MDUCoverageEnv
from src.agents.mappo import MAPPO
from src.config import Config
from src.run_manager import RunManager


def main():
    _d = Config()

    p = argparse.ArgumentParser(
        description="Overnight training. Defaults from src/config.py.")
    p.add_argument("--mdus", type=int, default=_d.env.num_mdus)
    p.add_argument("--episodes", type=int, default=500,
                   help="Number of episodes (default: 500 for overnight)")
    p.add_argument("--epochs", type=int, default=5,
                   help="PPO epochs per update (default: 5 for speed)")
    p.add_argument("--cone-angle", type=float, default=_d.env.cone_angle_deg)
    p.add_argument("--cone-range", type=float, default=_d.env.cone_range)
    p.add_argument("--max-steps", type=int, default=_d.env.max_steps)
    p.add_argument("--completion-threshold", type=float,
                   default=_d.env.completion_threshold)
    p.add_argument("--hidden", type=int, default=_d.agent.hidden_dim)
    p.add_argument("--lr-actor", type=float, default=_d.agent.lr_actor)
    p.add_argument("--lr-critic", type=float, default=_d.agent.lr_critic)
    p.add_argument("--seed", type=int, default=_d.train.seed)
    p.add_argument("--output-root", type=str, default="results")
    p.add_argument("--tag", type=str, default="overnight")
    args = p.parse_args()

    ROOT = os.path.dirname(os.path.abspath(__file__))
    cfg = Config.from_args(args, root=ROOT)

    # Overnight-specific overrides (fewer epochs for speed)
    cfg.agent.num_epochs = args.epochs

    rm = RunManager(output_root=args.output_root, tag=args.tag)
    LOG = os.path.join(rm.run_dir, "log.txt")

    def log(msg):
        with open(LOG, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n"); f.flush()
        print(msg, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    env = MDUCoverageEnv(**cfg.env_kwargs(root=ROOT))
    obs, info = env.reset()
    log(f"Obs: {obs.shape[1]}, Act: {env.max_deg}, "
        f"Init cov: {info['coverage_rate']:.2%}")

    mappo_cfg = cfg.to_mappo_config()
    mappo_cfg.num_epochs = args.epochs  # overnight: fewer epochs
    agent = MAPPO(obs.shape[1], info["global_state"].shape[0], env.max_deg,
                  mappo_cfg, device=device)

    best_cov = 0.0
    t0 = time.time()
    n_mdus = env.num_mdus  # NOT hardcoded!

    ep_cov, ep_ent, ep_hid = [], [], []

    for ep in range(1, args.episodes + 1):
        agent.anneal_entropy((ep - 1) / args.episodes)
        obs, info = env.reset()
        agent.reset_hidden(n_mdus)
        done = False
        total_r = 0.0
        pa = np.zeros(n_mdus, dtype=int)

        while not done:
            cr = float(info.get("coverage_rate", obs[:, 6].mean()))
            acts, vals, lps = agent.act(
                obs, info["global_state"],
                action_mask=info["action_mask"],
                prev_actions=pa,
                coverage_rate=[cr] * n_mdus)
            pre_obs = obs.copy()
            pre_state = info["global_state"].copy()
            pre_mask = info["action_mask"].copy()
            obs, r, term, trunc, info = env.step(acts)
            agent.store(pre_obs, pre_state, acts, r, term or trunc,
                        vals, lps, action_mask=pre_mask)
            pa = acts
            total_r += r
            done = term or trunc

        stats = agent.update()
        cov = info["coverage_rate"]
        ep_cov.append(cov)
        ep_ent.append(stats["entropy"])
        ep_hid.append(stats["hidden_norm"])
        if cov >= best_cov:
            best_cov = cov
            agent.save(rm.cp_path("mappo_best.pt"))
        if ep % 25 == 0 or ep == 1:
            done_step = info.get("completion_step", -1)
            ds = f" Done>{done_step}" if done_step >= 0 else ""
            log(f"Ep {ep:4d} | Cov {cov:.2%}{ds} | "
                f"Ent {stats['entropy']:.3f} | H| {stats['hidden_norm']:.2f}")

    agent.save(rm.cp_path("mappo_final.pt"))
    elap = time.time() - t0
    log(f"DONE! {elap:.0f}s ({elap/60:.1f}min)")
    log(f"Best cov: {best_cov:.2%}")

    # Parameter dump
    rm.dump_params(cfg, extra={
        "device": device,
        "best_coverage": f"{best_cov:.4f}",
        "elapsed_sec": f"{elap:.0f}",
        "overnight_epochs": str(args.epochs),
        "overnight_episodes": str(args.episodes),
    })

    # Save curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.style_config import apply_style
        apply_style()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax = axes[0, 0]
        ax.plot(ep_cov, "g-", alpha=0.5, linewidth=0.5)
        ax.set_xlabel("Episode"); ax.set_ylabel("Coverage")
        ax.grid(alpha=0.3)
        ax.set_title(f"Coverage (best={best_cov:.2%})")
        ax = axes[0, 1]
        ax.plot(ep_ent, "r-", alpha=0.5, linewidth=0.5)
        ax.set_xlabel("Episode"); ax.set_ylabel("Entropy")
        ax.grid(alpha=0.3)
        ax.set_title("Policy Entropy")
        ax = axes[1, 0]
        ax.plot(ep_hid, "purple", alpha=0.5, linewidth=0.5)
        ax.set_xlabel("Episode"); ax.set_ylabel("H| Norm")
        ax.grid(alpha=0.3)
        ax.set_title("GRU Hidden Norm")
        ax = axes[1, 1]
        ax.axis("off")
        ax.text(0.05, 0.95,
                f"Overnight Training\n{args.episodes} eps, {elap:.0f}s\n"
                f"Best cov: {best_cov:.2%}\n"
                f"{cfg.env.cone_angle_deg}deg cone\n"
                f"Complete: {cfg.env.completion_threshold:.0%}\n"
                f"{env.num_mdus} MDUs",
                transform=ax.transAxes, fontsize=11, fontfamily="monospace",
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        plt.tight_layout()
        plt.savefig(rm.plt_path("training_curves.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()

        # Save data
        np.savez(rm.plt_path("training_data.npz"),
                 coverage=np.array(ep_cov),
                 entropy=np.array(ep_ent),
                 hidden_norm=np.array(ep_hid),
                 best_cov=best_cov)
        log(f"Saved plots and data to {rm.plot_dir}")
    except Exception as ex:
        log(f"Plot error: {ex}")

    log(f"All outputs: {rm.run_dir}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
