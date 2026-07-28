#!/usr/bin/env python
"""
Unfolded 2D topology map with MDU exploration routes.

Reads a trajectory NPZ and plots the space net unfolded into its underlying
square grid, overlaid with colored MDU paths (directed arrows, alpha from
0.7 to 0.9 indicating traversal order).

Usage:
    python src/visualization/plot_topology.py --tag FINAL_v15
    python src/visualization/plot_topology.py --npz path/to/trajectory_mappo.npz
"""

import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.run_manager import find_run


def compute_grid_layout(net):
    """Unfold the space net into a 2D grid by BFS from a corner node.

    Returns:
        grid_pos: dict node_id -> (row, col) in grid coordinates
        rows, cols: grid dimensions
    """
    N = net.NumPoints
    # Find a corner node (degree 2) to start
    deg2_nodes = [i for i in range(N) if len(net.get_neighbors(i)) == 2]
    if not deg2_nodes:
        deg2_nodes = [0]
    start = deg2_nodes[0]

    # BFS to assign grid coordinates
    grid_pos = {}
    visited = set()
    queue = deque([start])
    grid_pos[start] = (0, 0)
    visited.add(start)

    while queue:
        u = queue.popleft()
        ru, cu = grid_pos[u]
        nbs = net.get_neighbors(u)
        # Assign unvisited neighbors based on existing neighbor coordinates
        assigned = []
        for v in nbs:
            if v in grid_pos:
                assigned.append(grid_pos[v])
        for v in nbs:
            if v not in visited:
                # Place neighbor in an empty adjacent grid cell
                # Try cardinal directions
                candidates = [(ru - 1, cu), (ru + 1, cu), (ru, cu - 1), (ru, cu + 1)]
                occupied = {grid_pos[w] for w in visited if w in grid_pos}
                occupied.update(grid_pos.get(vv, None) for vv in nbs if vv in grid_pos)
                placed = False
                for cand in candidates:
                    if cand not in occupied:
                        grid_pos[v] = cand
                        visited.add(v)
                        queue.append(v)
                        placed = True
                        break
                if not placed:
                    # Fallback: place at a novel position
                    grid_pos[v] = (ru + 1, cu)
                    visited.add(v)
                    queue.append(v)

    # Normalize to start at (0,0)
    min_r = min(r for r, c in grid_pos.values())
    min_c = min(c for r, c in grid_pos.values())
    grid_pos = {k: (r - min_r, c - min_c) for k, (r, c) in grid_pos.items()}
    rows = max(r for r, c in grid_pos.values()) + 1
    cols = max(c for r, c in grid_pos.values()) + 1
    return grid_pos, rows, cols


def main():
    p = argparse.ArgumentParser(description="Unfolded topology map with MDU routes")
    p.add_argument("--tag", type=str, default=None, help="Run tag (e.g. FINAL_v15)")
    p.add_argument("--run", type=str, default=None, help="Exact run directory")
    p.add_argument("--npz", type=str, default=None, help="Path to trajectory NPZ")
    p.add_argument("--output", type=str, default=None, help="Output path (default: <run>/plots/topology.png)")
    p.add_argument("--dpi", type=int, default=150, help="Output DPI")
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    # Build a minimal adjacency for grid layout
    class SimpleNet:
        def __init__(self, edges, N):
            self.NumPoints = N
            self.adj = [[] for _ in range(N)]
            for u, v in edges:
                self.adj[int(u)].append(int(v))
                self.adj[int(v)].append(int(u))
            for i in range(N):
                self.adj[i] = np.unique(self.adj[i])

        def get_neighbors(self, i):
            return self.adj[i]

    net = SimpleNet(net_edges, len(net_positions))
    grid_pos, rows, cols = compute_grid_layout(net)
    print(f"Grid: {rows} x {cols}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(max(cols / 4, 16), max(rows / 4, 12)))

    # Draw edges (light gray)
    for u, v in net_edges:
        if int(u) in grid_pos and int(v) in grid_pos:
            r1, c1 = grid_pos[int(u)]
            r2, c2 = grid_pos[int(v)]
            ax.plot([c1, c2], [r1, r2], color=(0.7, 0.7, 0.7, 0.3), linewidth=0.4, zorder=1)

    # Draw nodes (tiny dots)
    node_r = [grid_pos[i][0] for i in range(len(net_positions)) if i in grid_pos]
    node_c = [grid_pos[i][1] for i in range(len(net_positions)) if i in grid_pos]
    ax.scatter(node_c, node_r, s=1.5, color=(0.3, 0.3, 0.3, 0.5), zorder=2)

    # MDU colors
    colors = plt.cm.tab10(np.linspace(0, 1, N_mdus))

    # Draw MDU trajectories
    for mdu_idx in range(N_mdus):
        path = mdu_nodes[:, mdu_idx]  # (T,)
        color = colors[mdu_idx]
        for t in range(T - 1):
            u = int(path[t])
            v = int(path[t + 1])
            if u == v or u not in grid_pos or v not in grid_pos:
                continue
            r1, c1 = grid_pos[u]
            r2, c2 = grid_pos[v]
            # Alpha from 0.7 (early) to 0.9 (late)
            alpha = 0.7 + 0.2 * (t / max(T - 2, 1))
            # Arrow
            dr = r2 - r1
            dc = c2 - c1
            ax.arrow(c1, r1, dc * 0.85, dr * 0.85,
                     head_width=0.25, head_length=0.35, fc=color, ec=color,
                     alpha=alpha, linewidth=0.8, length_includes_head=True, zorder=5)

        # Start marker (circle)
        start_node = int(path[0])
        if start_node in grid_pos:
            sr, sc = grid_pos[start_node]
            ax.scatter(sc, sr, s=80, marker="o", facecolor=color, edgecolor="black",
                       linewidth=1.2, zorder=10, label=f"MDU {mdu_idx + 1}")

    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"MDU Exploration Routes on Unfolded Space Net\n"
                 f"({T} timesteps, {N_mdus} MDUs)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Grid column", fontweight="bold")
    ax.set_ylabel("Grid row", fontweight="bold")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # row 0 at top
    plt.tight_layout()

    # Output
    if args.output:
        out_path = args.output
    else:
        run_tag = args.tag or os.path.basename(os.path.dirname(os.path.dirname(npz_path)))
        out_dir = os.path.join(root, "results", run_tag.replace("trajectories", "").strip("_"),
                               "plots") if args.tag else os.path.dirname(npz_path).replace("trajectories", "plots")
        # Fallback: same dir as npz
        out_dir = os.path.dirname(os.path.dirname(npz_path)) if args.tag else os.path.dirname(os.path.dirname(npz_path))
        out_dir = os.path.join(out_dir, "plots") if "plots" not in out_dir else out_dir

    if not os.path.isdir(out_dir):
        out_dir = os.path.dirname(os.path.dirname(npz_path))
        out_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "topology_map.png")
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
