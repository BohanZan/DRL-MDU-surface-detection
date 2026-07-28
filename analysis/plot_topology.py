#!/usr/bin/env python
"""
Unfolded 2D topology map with MDU exploration routes.

Reads a trajectory NPZ and plots the space net unfolded into its underlying
square grid (derived from the FNS topology matrix), overlaid with colored MDU
paths using directed arrows (alpha 0.7 to 0.9 indicating traversal order).

Usage:
    python analysis/plot_topology.py --tag FINAL_v15
    python analysis/plot_topology.py --npz path/to/trajectory_mappo.npz
"""

import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.run_manager import find_run


def compute_grid_layout(net_edges, NumPoints):
    """Use networkx spring layout to unfold the graph into 2D.

    The spring layout preserves graph connectivity structure without
    requiring knowledge of the FNS grid encoding.
    """
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(NumPoints))
    G.add_edges_from([(int(u), int(v)) for u, v in net_edges])
    pos = nx.spring_layout(G, seed=42, iterations=200, k=2.0)
    grid_pos = {i: (pos[i][1], pos[i][0]) for i in range(NumPoints)}
    return grid_pos


def main():
    p = argparse.ArgumentParser(description="Unfolded topology map with MDU routes")
    p.add_argument("--tag", type=str, default=None, help="Run tag (e.g. FINAL_v15)")
    p.add_argument("--run", type=str, default=None, help="Exact run directory")
    p.add_argument("--npz", type=str, default=None, help="Path to trajectory NPZ")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: analysis/outputs/)")
    p.add_argument("--dpi", type=int, default=150, help="Output DPI")
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Resolve NPZ path
    if args.npz:
        npz_path = args.npz
    else:
        run_dir = find_run(os.path.join(root, "results"), tag=args.tag, run_name=args.run)
        if not run_dir:
            print("Run not found. Use --tag, --run, or --npz.")
            sys.exit(1)
        npz_path = os.path.join(run_dir, "trajectories", "trajectory_mappo.npz")
        if not os.path.exists(npz_path):
            print(f"Trajectory not found: {npz_path}")
            sys.exit(1)

    data = np.load(npz_path)
    mdu_nodes = data["mdu_nodes"]          # (T, N_mdus)
    net_positions = data["net_positions"]  # (369, 3)
    net_edges = data["net_edges"]          # (432, 2)

    T, N_mdus = mdu_nodes.shape
    print(f"Loaded: {T} timesteps, {N_mdus} MDUs")

    # Build grid layout via spring embedding
    grid_pos = compute_grid_layout(net_edges, len(net_positions))
    print(f"Layout: {len(grid_pos)} nodes")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(16, 12))

    # Draw edges
    for u, v in net_edges:
        if int(u) in grid_pos and int(v) in grid_pos:
            r1, c1 = grid_pos[int(u)]
            r2, c2 = grid_pos[int(v)]
            ax.plot([c1, c2], [r1, r2], color=(0.6, 0.6, 0.6, 0.25), linewidth=0.3, zorder=1)

    # Draw nodes
    mapped_nodes = sorted(grid_pos.keys())
    node_r = [grid_pos[i][0] for i in mapped_nodes]
    node_c = [grid_pos[i][1] for i in mapped_nodes]
    ax.scatter(node_c, node_r, s=3, color=(0.2, 0.2, 0.2, 0.4), zorder=2)

    # MDU colors
    colors = plt.cm.tab10(np.linspace(0, 1, max(N_mdus, 10)))

    # Draw MDU trajectories
    for mdu_idx in range(N_mdus):
        path = mdu_nodes[:, mdu_idx]
        color = colors[mdu_idx]
        for t in range(T - 1):
            u = int(path[t])
            v = int(path[t + 1])
            if u == v or u not in grid_pos or v not in grid_pos:
                continue
            r1, c1 = grid_pos[u]
            r2, c2 = grid_pos[v]
            alpha = 0.7 + 0.2 * (t / max(T - 2, 1))
            dr = r2 - r1
            dc = c2 - c1
            dist = np.sqrt(dr**2 + dc**2)
            ax.arrow(c1, r1, dc * 0.80, dr * 0.80,
                     head_width=0.02, head_length=0.03, fc=color, ec=color,
                     alpha=alpha, linewidth=1.2, length_includes_head=True, zorder=5)

        # Start marker
        start_node = int(path[0])
        if start_node in grid_pos:
            sr, sc = grid_pos[start_node]
            ax.scatter(sc, sr, s=100, marker="o", facecolor=color, edgecolor="black",
                       linewidth=1.5, zorder=10, label=f"MDU {mdu_idx + 1}")

    ax.legend(loc="upper right", fontsize=9, markerscale=0.8)
    ax.set_title(f"MDU Exploration Routes on Unfolded Space Net\n"
                 f"({T} timesteps, {N_mdus} MDUs)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("X", fontweight="bold")
    ax.set_ylabel("Y", fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")  # spring layout coords are arbitrary
    plt.tight_layout()

    # Output
    out_dir = args.output_dir or os.path.join(root, "analysis", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.tag or os.path.basename(os.path.dirname(os.path.dirname(npz_path)))
    out_path = os.path.join(out_dir, f"topology_map_{tag}.png")
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
