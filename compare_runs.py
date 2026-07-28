"""
Compare two training runs side by side.
Usage:
    python compare_runs.py --tag baseline --tag2 gnn
    python compare_runs.py --run <dir1> --run2 <dir2>
"""
import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from run_manager import find_run
from style_config import TOL_MUTED, apply_style
apply_style()

def load_data(run_dir):
    path = os.path.join(run_dir, "plots", "training_data.npz")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return None
    return np.load(path)

def main():
    p = argparse.ArgumentParser(description="Compare two training runs.")
    p.add_argument("--tag", type=str, default=None, help="Tag for run 1")
    p.add_argument("--tag2", type=str, default=None, help="Tag for run 2")
    p.add_argument("--run", type=str, default=None, help="Exact dir for run 1")
    p.add_argument("--run2", type=str, default=None, help="Exact dir for run 2")
    p.add_argument("--label1", type=str, default="Baseline")
    p.add_argument("--label2", type=str, default="GNN")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory for comparison.png (default: run1 dir)")
    args = p.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(root, "results")

    run1 = find_run(results_root, tag=args.tag, run_name=args.run)
    run2 = find_run(results_root, tag=args.tag2, run_name=args.run2)

    if not run1 or not run2:
        runs = []
        if os.path.isdir(results_root):
            runs = sorted([d for d in os.listdir(results_root)
                          if os.path.isdir(os.path.join(results_root, d))], reverse=True)
        if not run1 and len(runs) >= 1:
            run1 = os.path.join(results_root, runs[0])
        if not run2 and len(runs) >= 2:
            run2 = os.path.join(results_root, runs[1])
        if not run1 or not run2:
            print("ERROR: Need two runs. Use --tag/--tag2 or --run/--run2.")
            print(f"Available runs: {runs}")
            sys.exit(1)

    base = load_data(run1)
    gnn = load_data(run2)

    if base is None or gnn is None:
        print("Cannot load data. Aborting.")
        sys.exit(1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Baseline vs GNN — 4 MDU, 1000 episodes", fontsize=14, fontweight="bold")

    def plot_compare(ax, x1, y1, x2, y2, label1, label2, ylabel, title, smooth=20):
        ax.plot(y1, alpha=0.2, linewidth=0.5, color=TOL_MUTED[0])
        ax.plot(y2, alpha=0.2, linewidth=0.5, color=TOL_MUTED[2])
        if len(y1) >= smooth:
            s1 = np.convolve(y1, np.ones(smooth)/smooth, mode="valid")
            s2 = np.convolve(y2, np.ones(smooth)/smooth, mode="valid")
            xs = np.arange(smooth-1, len(y1))
            ax.plot(xs, s1, "b-", linewidth=2, label=label1)
            ax.plot(xs, s2, "r-", linewidth=2, label=label2)
        ax.set_xlabel("Episode"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(alpha=0.3); ax.set_title(title)

    # Coverage
    plot_compare(axes[0, 0], None, base["coverage"], None, gnn["coverage"],
                 f"Baseline (best={base['best_cov']:.2%}, greedy={base['greedy_cov']:.2%})",
                 f"GNN (best={gnn['best_cov']:.2%}, greedy={gnn['greedy_cov']:.2%})",
                 "Coverage", "Coverage Rate")

    # Reward
    plot_compare(axes[0, 1], None, base["reward"], None, gnn["reward"],
                 "Baseline", "GNN", "Reward", "Episode Reward")

    # Episode Length
    plot_compare(axes[0, 2], None, base["length"], None, gnn["length"],
                 "Baseline", "GNN", "Steps", "Episode Length")

    # Actor Loss
    plot_compare(axes[1, 0], None, base["loss_a"], None, gnn["loss_a"],
                 "Baseline", "GNN", "Loss A", "Actor Loss")

    # Critic Loss
    plot_compare(axes[1, 1], None, base["loss_c"], None, gnn["loss_c"],
                 "Baseline", "GNN", "Loss C", "Critic Loss")

    # Hidden Norm
    plot_compare(axes[1, 2], None, base["hidden_norm"], None, gnn["hidden_norm"],
                 "Baseline", "GNN", "|H|", "GRU Hidden Norm")

    plt.tight_layout()
    out_dir = args.output_dir or os.path.join(run1, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison saved to: {out_path}")

    # Text summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Baseline':>15} {'GNN':>15}")
    print(f"{'-'*55}")
    for name, b_val, g_val in [
        ("Best Coverage", base["best_cov"], gnn["best_cov"]),
        ("Greedy Coverage", base["greedy_cov"], gnn["greedy_cov"]),
        ("Final Coverage (avg last 10)",
         base["coverage"][-10:].mean() if len(base["coverage"]) >= 10 else base["coverage"].mean(),
         gnn["coverage"][-10:].mean() if len(gnn["coverage"]) >= 10 else gnn["coverage"].mean()),
        ("Final Entropy",
         "N/A",
         "N/A"),
    ]:
        print(f"{name:<25} {b_val:>15.4f} {g_val:>15.4f}")

    delta_cov = gnn["best_cov"] - base["best_cov"]
    print(f"\nGNN improvement on best: {delta_cov:+.2%}")
    print(f"GNN improvement on greedy: {gnn['greedy_cov'] - base['greedy_cov']:+.2%}")


if __name__ == "__main__":
    main()
