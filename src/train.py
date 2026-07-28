"""Train MAPPO on MDU Coverage env.

All defaults come from src.config.Config. CLI arguments override them.
Output goes to timestamped directories managed by RunManager.
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.env import MDUCoverageEnv
from src.agents.mappo import MAPPO
from src.config import Config
from src.run_manager import RunManager


def main():
    # Build default config to seed argparse
    _d = Config()

    p = argparse.ArgumentParser(
        description="Train MAPPO on MDU coverage env. Defaults from src/config.py.")
    # Env
    p.add_argument("--start-nodes", type=int, nargs="+", default=None)
    p.add_argument("--mdus", type=int, default=None)
    p.add_argument("--cone-angle", type=float, default=_d.env.cone_angle_deg)
    p.add_argument("--cone-range", type=float, default=_d.env.cone_range)
    p.add_argument("--max-steps", type=int, default=_d.env.max_steps)
    p.add_argument("--coverage-threshold", type=float, default=_d.env.coverage_threshold)
    p.add_argument("--completion-threshold", type=float, default=_d.env.completion_threshold)
    p.add_argument("--r-newly", type=float, default=_d.env.r_newly)
    p.add_argument("--r-completion", type=float, default=_d.env.r_completion)
    p.add_argument("--r-speed", type=float, default=_d.env.r_speed)
    # Agent
    p.add_argument("--lr-actor", type=float, default=_d.agent.lr_actor)
    p.add_argument("--lr-critic", type=float, default=_d.agent.lr_critic)
    p.add_argument("--gamma", type=float, default=_d.agent.gamma)
    p.add_argument("--epochs", type=int, default=_d.agent.num_epochs)
    p.add_argument("--hidden", type=int, default=_d.agent.hidden_dim)
    p.add_argument("--ent-coef-end", type=float, default=_d.agent.ent_coef_end)
    p.add_argument("--bptt-len", type=int, default=_d.agent.bptt_len)
    # Training
    p.add_argument("--episodes", type=int, default=_d.train.episodes)
    p.add_argument("--seed", type=int, default=_d.train.seed)
    p.add_argument("--device", type=str, default=_d.train.device)
    p.add_argument("--log-every", type=int, default=_d.train.log_every)
    # Output
    p.add_argument("--output-root", type=str, default="results",
                   help="Top-level output directory (default: results/)")
    p.add_argument("--tag", type=str, default="",
                   help="Label appended to run timestamp (e.g. '4mdu', 'test')")
    p.add_argument("--save-plot", action="store_true",
                   help="Record every episode and save training curves")
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Build config (defaults + CLI overrides)
    cfg = Config.from_args(args, root=root)

    # Set up run directory
    tag = args.tag or f"{cfg.env.num_mdus}mdu"
    rm = RunManager(output_root=args.output_root, tag=tag)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build env from config
    env_kwargs = cfg.env_kwargs(root=root)
    if args.start_nodes is not None:
        env_kwargs["mdu_start_nodes"] = args.start_nodes
        env_kwargs.pop("num_mdus", None)  # let env infer from explicit nodes
    if args.mdus is not None and args.start_nodes is None:
        env_kwargs["num_mdus"] = args.mdus
    env = MDUCoverageEnv(**env_kwargs)
    if args.seed is not None:
        env._rng = np.random.default_rng(args.seed)

    obs, info = env.reset()
    print(f"\nMDUs: {env.num_mdus} at {env.mdu_start_nodes}")
    print(f"Obs dim: {obs.shape[1]}, State dim: {info['global_state'].shape[0]}, Act dim: {env.max_deg}")
    print(f"Cone: {cfg.env.cone_angle_deg}deg / {cfg.env.cone_range}m")
    print(f"Episodes: {cfg.train.episodes}, Max steps: {cfg.env.max_steps}")
    print(f"Target cov: {cfg.env.coverage_threshold:.0%} | "
          f"Completion threshold: {cfg.env.completion_threshold:.0%}")
    print(f"Run dir: {rm.run_dir}\n")

    # Build agent config
    mappo_cfg = cfg.to_mappo_config()
    agent = MAPPO(obs.shape[1], info["global_state"].shape[0], env.max_deg,
                  mappo_cfg, device=device)

    ep_rew, ep_cov, ep_len = [], [], []
    ep_loss_a, ep_loss_c, ep_hidden = [], [], []
    best_cov = 0.0
    t0 = time.time()
    log_every = 1 if args.save_plot else cfg.train.log_every

    for ep in range(1, cfg.train.episodes + 1):
        agent.anneal_entropy((ep - 1) / cfg.train.episodes)
        obs, info = env.reset()
        agent.reset_hidden(num_mdus=env.num_mdus)
        done = False
        total_r = 0.0
        steps = 0
        prev_actions = np.zeros(env.num_mdus, dtype=int)

        while not done:
            cr = float(info.get("coverage_rate", obs[:, 6].mean()))
            acts, vals, lps = agent.act(
                obs, info["global_state"],
                action_mask=info["action_mask"],
                prev_actions=prev_actions,
                coverage_rate=[cr] * env.num_mdus)
            # Save pre-step state BEFORE stepping the environment
            pre_obs = obs.copy()
            pre_state = info["global_state"].copy()
            pre_mask = info["action_mask"].copy()
            obs, r, term, trunc, info = env.step(acts)
            # Store pre-step obs with post-step reward (correct RL semantics)
            agent.store(pre_obs, pre_state, acts, r, term or trunc,
                        vals, lps, action_mask=pre_mask)
            prev_actions = acts
            total_r += r
            steps += 1
            done = term or trunc

        stats = agent.update()
        ep_rew.append(total_r)
        ep_cov.append(info["coverage_rate"])
        ep_len.append(steps)
        ep_loss_a.append(stats["loss_a"])
        ep_loss_c.append(stats["loss_c"])
        ep_hidden.append(stats["hidden_norm"])
        best_cov = max(best_cov, info["coverage_rate"])
        if info["coverage_rate"] >= best_cov:
            agent.save(rm.cp_path("mappo_best.pt"))

        if ep % log_every == 0 or ep == 1 or ep == cfg.train.episodes:
            avg_r = np.mean(ep_rew[-log_every:])
            avg_c = np.mean(ep_cov[-log_every:])
            avg_l = np.mean(ep_len[-log_every:])
            cross = info.get("completion_step", -1)
            cross_s = f" Done>{cross}" if cross >= 0 else ""
            print(f"Ep {ep:5d} | R {avg_r:+.3f} | Cov {avg_c:.2%}{cross_s} | "
                  f"Len {avg_l:.0f} | Ent {stats['entropy']:.3f} | "
                  f"H| {stats['hidden_norm']:.2f} | LR {agent._ent_coef:.4f}")

    # Greedy eval
    print("\nGreedy eval...")
    obs, info = env.reset()
    agent.reset_hidden(num_mdus=env.num_mdus)
    am = info["action_mask"]
    prev_actions = np.zeros(env.num_mdus, dtype=int)
    for _ in range(cfg.env.max_steps):
        cr = float(info.get("coverage_rate", obs[:, 6].mean()))
        acts, _, _ = agent.act(obs, info["global_state"],
                               action_mask=am, greedy=True,
                               prev_actions=prev_actions,
                               coverage_rate=[cr] * env.num_mdus)
        obs, _, term, trunc, info = env.step(acts)
        am = info["action_mask"]
        prev_actions = acts
        if term or trunc:
            break
    greedy_cov = info["coverage_rate"]
    agent.save(rm.cp_path("mappo_final.pt"))

    elap = time.time() - t0
    print(f"\n{'='*50}")
    print(f"DONE! {elap:.0f}s ({elap/60:.1f}min)")
    print(f"  Best train:  {best_cov:.2%}")
    print(f"  Greedy eval: {greedy_cov:.2%}")
    print(f"{'='*50}")

    # Dump parameters
    rm.dump_params(cfg, extra={
        "device": device,
        "best_coverage": f"{best_cov:.4f}",
        "greedy_coverage": f"{greedy_cov:.4f}",
        "elapsed_sec": f"{elap:.0f}",
        "actual_mdus": str(env.num_mdus),
    })
    print(f"\nParameters: {os.path.join(rm.run_dir, 'parameters.txt')}")

    # ── Save training curves (only when --save-plot) ─────────
    if args.save_plot and len(ep_cov) > 1:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # --- Plot style: Times New Roman, bold labels ---
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = ["Times New Roman"]
            plt.rcParams["mathtext.fontset"] = "stix"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams["axes.titleweight"] = "bold"

            n = min(len(ep_cov), len(ep_hidden), len(ep_loss_a), len(ep_loss_c))
            fig, axes = plt.subplots(2, 3, figsize=(16, 8))

            # Coverage
            ax = axes[0, 0]
            ax.plot(ep_cov[:n], "g-", alpha=0.5, linewidth=0.5)
            ax.plot(ep_cov[:n], "g.", markersize=1)
            if n >= 10:
                s = np.convolve(ep_cov[:n], np.ones(10)/10, mode="valid")
                ax.plot(np.arange(9, n), s, "b-", linewidth=2, label="10-ep avg")
            ax.axhline(0.75, color="gray", ls="--", alpha=0.5, label="75% target")
            ax.axhline(best_cov, color="orange", ls=":", alpha=0.5,
                       label=f"Best {best_cov:.1%}")
            ax.set_xlabel("Episode"); ax.set_ylabel("Coverage")
            ax.legend(); ax.grid(alpha=0.3)
            ax.set_title(f"Coverage (best={best_cov:.2%}, greedy={greedy_cov:.2%})")

            # Episode length
            ax = axes[0, 1]
            ax.plot(ep_len[:n], "r-", alpha=0.5, linewidth=0.5)
            ax.plot(ep_len[:n], "r.", markersize=1)
            ax.set_xlabel("Episode"); ax.set_ylabel("Steps"); ax.grid(alpha=0.3)
            ax.set_title("Episode Length")

            # Total reward
            ax = axes[0, 2]
            ax.plot(ep_rew[:n], "m-", alpha=0.5, linewidth=0.5)
            ax.plot(ep_rew[:n], "m.", markersize=1)
            ax.set_xlabel("Episode"); ax.set_ylabel("Reward"); ax.grid(alpha=0.3)
            ax.set_title("Episode Reward")

            # Actor loss
            ax = axes[1, 0]
            ax.plot(ep_loss_a[:n], "c-", alpha=0.7, linewidth=0.5)
            ax.set_xlabel("Episode"); ax.set_ylabel("Loss A"); ax.grid(alpha=0.3)
            ax.set_title("Actor Loss")

            # Critic loss
            ax = axes[1, 1]
            ax.plot(ep_loss_c[:n], "orange", alpha=0.7, linewidth=0.5)
            ax.set_xlabel("Episode"); ax.set_ylabel("Loss C"); ax.grid(alpha=0.3)
            ax.set_title("Critic Loss")

            # Hidden norm
            ax = axes[1, 2]
            ax.plot(ep_hidden[:n], color="purple", alpha=0.7, linewidth=0.5)
            ax.plot(ep_hidden[:n], color="purple", marker=".", linestyle="none",
                    markersize=4)
            ax.set_xlabel("Episode"); ax.set_ylabel("H| Norm"); ax.grid(alpha=0.3)
            ax.set_title("GRU Hidden Norm")

            plt.tight_layout()
            plot_path = rm.plt_path("training_curves.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Training curves: {plot_path}")

            # Raw data
            npz_path = rm.plt_path("training_data.npz")
            np.savez(npz_path,
                     episode=np.arange(1, n+1),
                     coverage=np.array(ep_cov[:n]),
                     reward=np.array(ep_rew[:n]),
                     length=np.array(ep_len[:n]),
                     loss_a=np.array(ep_loss_a[:n]),
                     loss_c=np.array(ep_loss_c[:n]),
                     hidden_norm=np.array(ep_hidden[:n]),
                     best_cov=best_cov, greedy_cov=greedy_cov)
            print(f"  Training data:   {npz_path}")
        except ImportError:
            print("  (matplotlib not available, skipping plot)")

    print(f"\nAll outputs saved to: {rm.run_dir}")


if __name__ == "__main__":
    main()
