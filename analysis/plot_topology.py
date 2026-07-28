#!/usr/bin/env python
"""
Unfolded 2D topology map with MDU exploration routes.

Two modes:
  --layout bfs     BFS cardinal-direction unfolding (best visual result)
  --layout spring  Spring-force unfolding (connectivity-preserving)

Usage:
    python analysis/plot_topology.py --tag FINAL_v15
    python analysis/plot_topology.py --tag FINAL_v15 --layout bfs
"""

import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.run_manager import find_run

# ── Font: bold Times New Roman labels/titles, regular tick numbers ──
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["font.weight"] = "normal"

COLORS_MDU = plt.cm.tab10(np.linspace(0, 1, 10))


def build_layout_bfs(net_edges, N):
    """BFS from a degree-2 corner node using physical adjacency.

    Places nodes on a 2D grid by trying cardinal directions
    for each unvisited neighbor. This was the original layout
    that produced the cleanest visual result.
    """
    adj = [[] for _ in range(N)]
    for u, v in net_edges:
        uu, vv = int(u), int(v)
        adj[uu].append(vv); adj[vv].append(uu)

    deg2 = [i for i in range(N) if len(adj[i]) == 2]
    start = deg2[0] if deg2 else 0

    grid_pos = {}
    visited = set()
    queue = deque([start])
    grid_pos[start] = (0, 0)
    visited.add(start)

    while queue:
        u = queue.popleft()
        ru, cu = grid_pos[u]
        for v in adj[u]:
            if v not in visited:
                occupied = {grid_pos[w] for w in visited if w in grid_pos}
                placed = False
                for cand in [(ru - 1, cu), (ru + 1, cu), (ru, cu - 1), (ru, cu + 1)]:
                    if cand not in occupied:
                        grid_pos[v] = cand; visited.add(v); queue.append(v)
                        placed = True; break
                if not placed:
                    r = ru + 1
                    while (r, cu) in occupied: r += 1
                    grid_pos[v] = (r, cu); visited.add(v); queue.append(v)

    # Normalize to (0,0) origin
    min_r = min(r for r, c in grid_pos.values())
    min_c = min(c for r, c in grid_pos.values())
    grid_pos = {k: (r - min_r, c - min_c) for k, (r, c) in grid_pos.items()}
    rows = max(r for r, c in grid_pos.values()) + 1
    cols = max(c for r, c in grid_pos.values()) + 1
    return grid_pos, rows, cols


def build_layout_spring(net_edges, N):
    """Spring-force graph unfolding."""
    import networkx as nx
    G = nx.Graph(); G.add_nodes_from(range(N))
    G.add_edges_from([(int(u), int(v)) for u, v in net_edges])
    pos = nx.spring_layout(G, seed=42, iterations=200, k=2.0)
    base = {i: (pos[i][1], pos[i][0]) for i in range(N)}
    return base, None, None


def draw_topology(ax, grid_pos, net_edges, mdu_nodes, N_mdus, T,
                  rows=None, cols=None, show_axes=False):
    """Draw unfolded topology with MDU routes."""
    # Edges
    for u, v in net_edges:
        uu, vv = int(u), int(v)
        if uu in grid_pos and vv in grid_pos:
            r1, c1 = grid_pos[uu]; r2, c2 = grid_pos[vv]
            ax.plot([c1, c2], [r1, r2], color=(0.6, 0.6, 0.6, 0.25),
                    linewidth=0.3, zorder=1)

    # Nodes
    nodes = sorted(grid_pos.keys())
    rs = [grid_pos[i][0] for i in nodes]; cs = [grid_pos[i][1] for i in nodes]
    ax.scatter(cs, rs, s=3, color=(0.2, 0.2, 0.2, 0.4), zorder=2)

    # MDU trajectories (arrows, alpha 0.7→0.9)
    for mdu_idx in range(N_mdus):
        path = mdu_nodes[:, mdu_idx]; color = COLORS_MDU[mdu_idx]
        for t in range(T - 1):
            u, v = int(path[t]), int(path[t + 1])
            if u == v or u not in grid_pos or v not in grid_pos:
                continue
            r1, c1 = grid_pos[u]; r2, c2 = grid_pos[v]
            alpha = 0.7 + 0.2 * (t / max(T - 2, 1))
            dc = c2 - c1; dr = r2 - r1
            ax.arrow(c1, r1, dc * 0.82, dr * 0.82,
                     head_width=0.02, head_length=0.03, fc=color, ec=color,
                     alpha=alpha, linewidth=1.2, length_includes_head=True, zorder=5)
        # Start marker
        sn = int(path[0])
        if sn in grid_pos:
            sr, sc = grid_pos[sn]
            ax.scatter(sc, sr, s=100, marker="o", facecolor=color,
                       edgecolor="black", linewidth=1.5, zorder=10,
                       label=f"MDU {mdu_idx + 1}")

    ax.legend(loc="upper right", fontsize=9, markerscale=0.8)
    ax.set_title(f"MDU Exploration Routes on Unfolded Space Net\n"
                 f"({T} timesteps, {N_mdus} MDUs)", fontweight="bold", fontsize=13)

    if show_axes and rows and cols:
        ax.set_xlabel("Grid column", fontweight="bold")
        ax.set_ylabel("Grid row", fontweight="bold")
        ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
        ax.tick_params(labelsize=8)
    else:
        ax.set_xlabel("X", fontweight="bold")
        ax.set_ylabel("Y", fontweight="bold")

    ax.set_aspect("equal")
    ax.invert_yaxis()


def main():
    p = argparse.ArgumentParser(description="Unfolded topology map with MDU routes")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--run", type=str, default=None)
    p.add_argument("--npz", type=str, default=None)
    p.add_argument("--layout", choices=["bfs", "spring", "both"], default="both")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.npz:
        npz_path = args.npz
    else:
        run_dir = find_run(os.path.join(root, "results"), tag=args.tag, run_name=args.run)
        if not run_dir: print("Run not found."); sys.exit(1)
        npz_path = os.path.join(run_dir, "trajectories", "trajectory_mappo.npz")

    data = np.load(npz_path)
    mdu_nodes = data["mdu_nodes"]; net_edges = data["net_edges"]
    T, N_mdus = mdu_nodes.shape; N_nodes = len(data["net_positions"])
    print(f"Loaded: {T} timesteps, {N_mdus} MDUs")

    out_dir = args.output_dir or os.path.join(root, "analysis", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.tag or os.path.splitext(os.path.basename(npz_path))[0]

    if args.layout in ("bfs", "both"):
        grid_pos, rows, cols = build_layout_bfs(net_edges, N_nodes)
        print(f"BFS layout: {rows}x{cols}, {len(grid_pos)} nodes")
        fig, ax = plt.subplots(figsize=(max(cols / 4, 16), max(rows / 4, 12)))
        draw_topology(ax, grid_pos, net_edges, mdu_nodes, N_mdus, T,
                      rows, cols, show_axes=True)
        out = os.path.join(out_dir, f"topology_bfs_{tag}.png")
        plt.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close()
        print(f"Saved: {out}")

    if args.layout in ("spring", "both"):
        pos, _, _ = build_layout_spring(net_edges, N_nodes)
        print(f"Spring layout: {len(pos)} nodes")
        fig, ax = plt.subplots(figsize=(16, 12))
        draw_topology(ax, pos, net_edges, mdu_nodes, N_mdus, T)
        out = os.path.join(out_dir, f"topology_spring_{tag}.png")
        plt.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close()
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
