#!/usr/bin/env python
"""
Unfolded 2D topology map with MDU exploration routes.

Projects the 3D space net positions to 2D via PCA, which naturally
recovers the square grid aspect ratio of the FNS net.

Usage:
    python analysis/plot_topology.py --tag FINAL_v15
"""

import sys, os, argparse, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.run_manager import find_run

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["font.weight"] = "normal"

COLORS_MDU = plt.cm.tab10(np.linspace(0, 1, 10))


def compute_2d_layout(positions):
    """PCA projection of 3D net positions to 2D.

    The FNS net is a square grid folded around the asteroid.
    PCA finds the two principal directions of the net surface,
    recovering the square aspect ratio.
    """
    pts = np.array(positions, dtype=np.float64)
    pts -= pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
    proj = pts @ Vt[:2].T                                     # (N, 2)
    # Equalize variance → square aspect ratio
    stds = proj.std(axis=0) + 1e-8
    proj = proj / stds
    # Map to grid dict: node -> (row, col)
    grid = {i: (proj[i, 1], proj[i, 0]) for i in range(len(positions))}
    rs, cs = proj[:, 1], proj[:, 0]
    rows = int(np.ceil(rs.max() - rs.min()) * 3) + 1
    cols = int(np.ceil(cs.max() - cs.min()) * 3) + 1
    return grid, rows, cols


def draw(ax, grid, net_edges, mdu_nodes, N_mdus, T):
    """Draw unfolded topology with MDU routes."""
    for u, v in net_edges:
        uu, vv = int(u), int(v)
        if uu in grid and vv in grid:
            r1, c1 = grid[uu]; r2, c2 = grid[vv]
            ax.plot([c1, c2], [r1, r2], color=(0.6, 0.6, 0.6, 0.25),
                    linewidth=0.4, zorder=1)

    nodes = sorted(grid)
    rs = [grid[i][0] for i in nodes]; cs = [grid[i][1] for i in nodes]
    ax.scatter(cs, rs, s=3, color=(0.2, 0.2, 0.2, 0.4), zorder=2)

    for mdu_idx in range(N_mdus):
        path = mdu_nodes[:, mdu_idx]; color = COLORS_MDU[mdu_idx]
        for t in range(T - 1):
            u, v = int(path[t]), int(path[t + 1])
            if u == v or u not in grid or v not in grid: continue
            r1, c1 = grid[u]; r2, c2 = grid[v]
            alpha = 0.7 + 0.2 * (t / max(T - 2, 1))
            dc, dr = c2 - c1, r2 - r1
            ax.arrow(c1, r1, dc * 0.85, dr * 0.85,
                     head_width=0.015, head_length=0.025, fc=color, ec=color,
                     alpha=alpha, linewidth=1.6, length_includes_head=True, zorder=5)
        sn = int(path[0])
        if sn in grid:
            sr, sc = grid[sn]
            ax.scatter(sc, sr, s=100, marker="o", facecolor=color,
                       edgecolor="black", linewidth=2.0, zorder=10,
                       label=f"MDU {mdu_idx + 1}")

    leg = ax.legend(loc="upper right", fontsize=24, markerscale=0.8,
                    framealpha=1.0, facecolor="white", edgecolor="black")
    leg.set_zorder(100)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=18)
    ax.set_xlabel("")
    ax.set_ylabel("")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--run", type=str, default=None)
    p.add_argument("--npz", type=str, default=None)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.npz: npz_path = args.npz
    else:
        run_dir = find_run(os.path.join(root, "results"), tag=args.tag, run_name=args.run)
        if not run_dir: print("Run not found."); sys.exit(1)
        npz_path = os.path.join(run_dir, "trajectories", "trajectory_mappo.npz")

    data = np.load(npz_path)
    mdu_nodes = data["mdu_nodes"]; net_edges = data["net_edges"]
    positions = data["net_positions"]
    T, N_mdus = mdu_nodes.shape

    grid, rows, cols = compute_2d_layout(positions)
    print(f"Layout: {rows}×{cols}, {len(grid)} nodes")

    out_dir = args.output_dir or os.path.join(root, "analysis", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.tag or os.path.splitext(os.path.basename(npz_path))[0]

    fig, ax = plt.subplots(figsize=(14, 12))
    draw(ax, grid, net_edges, mdu_nodes, N_mdus, T)
    out = os.path.join(out_dir, f"topology_map_{tag}.png")
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
