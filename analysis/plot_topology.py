#!/usr/bin/env python
"""
Unfolded 2D topology maps with MDU exploration routes.

Two modes:
  --layout spring    Spring-force unfolding (connectivity-preserving)
  --layout grid      Rectangular index-based grid from FNS topology matrix

Usage:
    python analysis/plot_topology.py --tag FINAL_v15
    python analysis/plot_topology.py --tag FINAL_v15 --layout grid
"""

import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.run_manager import find_run

# ── Font setup ─────────────────────────────────────────────
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
# Tick labels: regular weight (not bold)
plt.rcParams["font.weight"] = "normal"

COLORS_MDU = plt.cm.tab10(np.linspace(0, 1, 10))


def build_grid_layout(fns_path):
    """Use FNS topology matrix to assign (row, col) to each node.

    Topo columns: [type, left, right, down, up, ...]
    """
    with open(fns_path) as f:
        h = f.readline().split()
        NP = int(h[0]); NE = int(h[1]); cT = int(h[3])
        for _ in range(NE): f.readline()
        topo = np.array([list(map(int, f.readline().split())) for _ in range(NP)])

    grid = {}; q = deque([0]); grid[0] = (0, 0)
    dirs = [(1, 0, -1), (2, 0, 1), (3, 1, 0), (4, -1, 0)]  # (topo_col, dr, dc)
    while q:
        u = q.popleft(); ru, cu = grid[u]
        for tc, dr, dc in dirs:
            if tc < topo.shape[1]:
                v = topo[u, tc]
                if v >= 0 and v < NP and v not in grid and v != u:
                    grid[v] = (ru + dr, cu + dc); q.append(v)
    min_r = min(r for r, c in grid.values()); min_c = min(c for r, c in grid.values())
    grid = {k: (r - min_r, c - min_c) for k, (r, c) in grid.items()}
    rows = max(r for r, c in grid.values()) + 1
    cols = max(c for r, c in grid.values()) + 1

    # Handle nodes sharing a cell: offset within cell
    from collections import Counter
    cell_counts = Counter(grid.values())
    offsets = {}
    cell_idx = {}
    for node, cell in grid.items():
        idx = cell_idx.get(cell, 0)
        cell_idx[cell] = idx + 1
        n_in_cell = cell_counts[cell]
        if n_in_cell > 1:
            # Spread nodes evenly within cell
            ox = (idx % 3 - 1) * 0.15
            oy = ((idx // 3) % 3 - 1) * 0.15
        else:
            ox, oy = 0.0, 0.0
        offsets[node] = (ox, oy)

    return grid, offsets, rows, cols


def build_spring_layout(net_edges, N):
    """Spring-force graph unfolding."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from([(int(u), int(v)) for u, v in net_edges])
    pos = nx.spring_layout(G, seed=42, iterations=200, k=2.0)
    base = {i: (pos[i][1], pos[i][0]) for i in range(N)}
    offsets = {i: (0.0, 0.0) for i in range(N)}
    return base, offsets, None, None


def draw_topology(ax, grid_pos, offsets, net_edges, mdu_nodes, N_mdus, T,
                  rows=None, cols=None, grid_mode=False):
    """Draw the unfolded topology with MDU routes."""
    # Edges
    for u, v in net_edges:
        uu, vv = int(u), int(v)
        if uu in grid_pos and vv in grid_pos:
            r1, c1 = grid_pos[uu]; o1r, o1c = offsets[uu]
            r2, c2 = grid_pos[vv]; o2r, o2c = offsets[vv]
            ax.plot([c1 + o1c, c2 + o2c], [r1 + o1r, r2 + o2r],
                    color=(0.6, 0.6, 0.6, 0.25), linewidth=0.3, zorder=1)

    # Nodes
    rs = [grid_pos[i][0] + offsets[i][0] for i in sorted(grid_pos)]
    cs = [grid_pos[i][1] + offsets[i][1] for i in sorted(grid_pos)]
    ax.scatter(cs, rs, s=3, color=(0.2, 0.2, 0.2, 0.4), zorder=2)

    # MDU trajectories
    for mdu_idx in range(N_mdus):
        path = mdu_nodes[:, mdu_idx]; color = COLORS_MDU[mdu_idx]
        for t in range(T - 1):
            u, v = int(path[t]), int(path[t + 1])
            if u == v or u not in grid_pos or v not in grid_pos:
                continue
            r1, c1 = grid_pos[u]; o1r, o1c = offsets[u]
            r2, c2 = grid_pos[v]; o2r, o2c = offsets[v]
            alpha = 0.7 + 0.2 * (t / max(T - 2, 1))
            dc = (c2 + o2c) - (c1 + o1c); dr = (r2 + o2r) - (r1 + o1r)
            ax.arrow(c1 + o1c, r1 + o1r, dc * 0.82, dr * 0.82,
                     head_width=0.02, head_length=0.03, fc=color, ec=color,
                     alpha=alpha, linewidth=1.2, length_includes_head=True, zorder=5)
        # Start marker
        sn = int(path[0])
        if sn in grid_pos:
            sr, sc = grid_pos[sn]; osr, osc = offsets[sn]
            ax.scatter(sc + osc, sr + osr, s=100, marker="o", facecolor=color,
                       edgecolor="black", linewidth=1.5, zorder=10,
                       label=f"MDU {mdu_idx + 1}")

    ax.legend(loc="upper right", fontsize=9, markerscale=0.8)
    cov_start = 0.0  # t=0 coverage rate (not stored easily here, use 0)
    ax.set_title(f"MDU Exploration Routes on Unfolded Space Net\n"
                 f"({T} timesteps, {N_mdus} MDUs)", fontweight="bold", fontsize=13)

    if grid_mode and rows and cols:
        ax.set_xlabel("Grid column index", fontweight="bold")
        ax.set_ylabel("Grid row index", fontweight="bold")
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.tick_params(labelsize=9)
    else:
        ax.set_xlabel("X", fontweight="bold")
        ax.set_ylabel("Y", fontweight="bold")
        ax.axis("off")
    ax.set_aspect("equal")
    if grid_mode:
        ax.invert_yaxis()


def main():
    p = argparse.ArgumentParser(description="Unfolded topology map with MDU routes")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--run", type=str, default=None)
    p.add_argument("--npz", type=str, default=None)
    p.add_argument("--layout", choices=["spring", "grid", "both"], default="both")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.npz:
        npz_path = args.npz
    else:
        run_dir = find_run(os.path.join(root, "results"), tag=args.tag, run_name=args.run)
        if not run_dir:
            print("Run not found."); sys.exit(1)
        npz_path = os.path.join(run_dir, "trajectories", "trajectory_mappo.npz")

    data = np.load(npz_path)
    mdu_nodes = data["mdu_nodes"]; net_edges = data["net_edges"]
    net_positions = data["net_positions"]
    T, N_mdus = mdu_nodes.shape
    N_nodes = len(net_positions)
    print(f"Loaded: {T} timesteps, {N_mdus} MDUs, {N_nodes} nodes")

    out_dir = args.output_dir or os.path.join(root, "analysis", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.tag or os.path.splitext(os.path.basename(npz_path))[0]

    # ── Spring layout ──
    if args.layout in ("spring", "both"):
        grid_pos, offsets, _, _ = build_spring_layout(net_edges, N_nodes)
        fig, ax = plt.subplots(figsize=(16, 12))
        draw_topology(ax, grid_pos, offsets, net_edges, mdu_nodes, N_mdus, T)
        out = os.path.join(out_dir, f"topology_spring_{tag}.png")
        plt.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close()
        print(f"Saved: {out}")

    # ── Grid (index-based) layout ──
    if args.layout in ("grid", "both"):
        fns_path = os.path.join(root, "FNS_square_fold-50m.txt")
        grid_pos, offsets, rows, cols = build_grid_layout(fns_path)
        fig, ax = plt.subplots(figsize=(max(cols * 1.2, 14), max(rows * 1.2, 10)))
        draw_topology(ax, grid_pos, offsets, net_edges, mdu_nodes, N_mdus, T,
                      rows=rows, cols=cols, grid_mode=True)
        out = os.path.join(out_dir, f"topology_grid_{tag}.png")
        plt.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close()
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
