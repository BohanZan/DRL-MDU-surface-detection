#!/usr/bin/env python
"""
3D snapshot of the MDU coverage state at a specified timestep and viewing angle.

Proper occlusion (hemisphere culling) and perspective projection.

Usage:
    python src/visualization/snapshot_3d.py --tag FINAL_v15 --step 80 --azim 45 --elev 20
    python src/visualization/snapshot_3d.py --npz path/to/traj.npz --step 100 --azim 0,90,180
"""

import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import to_rgba_array

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.run_manager import find_run
from src.style_config import TOL_MUTED, apply_style
apply_style()

COLOR_UNCOV = (0.5, 0.5, 0.5, 0.95)
COLOR_COV = (0.2, 0.8, 0.2, 0.95)
COLOR_MDU_OCCLUDED = (0.6, 0.6, 0.6, 0.3)
COLOR_NET_NODE = (0.3, 0.3, 0.8, 0.3)
COLOR_NET_EDGE = (0.4, 0.4, 0.7, 0.12)
COLOR_AST_EDGE = (0.3, 0.3, 0.3, 0.12)


def render_snapshot(ax, data, step, azim, elev):
    """Render one frame at the given step and viewing angle."""
    mdu_nodes = data["mdu_nodes"]
    cov_masks = data["coverage_masks"]
    cov_rates = data["coverage_rates"]
    net_positions = data["net_positions"]
    net_edges = data["net_edges"]
    ast_verts = data["ast_verts"]
    ast_faces = data["ast_faces"]

    n_mdus = mdu_nodes.shape[1]
    ast_tris = ast_verts[ast_faces]
    ast_center = np.mean(ast_verts, axis=0)
    all_pts = np.vstack([ast_verts, net_positions])
    half = np.ptp(all_pts, axis=0).max() / 2
    mid = np.mean(all_pts, axis=0)

    # Camera look-direction for occlusion test
    az_rad = np.deg2rad(azim)
    el_rad = np.deg2rad(elev)
    look_dir = np.array([
        np.cos(el_rad) * np.cos(az_rad),
        np.cos(el_rad) * np.sin(az_rad),
        np.sin(el_rad),
    ])

    ax.clear()
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_xlabel("X (m)", fontweight="bold")
    ax.set_ylabel("Y (m)", fontweight="bold")
    ax.set_zlabel("Z (m)", fontweight="bold")
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"Step {step} | Coverage: {cov_rates[step]:.1%} | "
                 f"Azim: {azim}deg, Elev: {elev}deg", fontsize=12, fontweight="bold")

    # 1. Asteroid surface (colored by coverage)
    n_faces = ast_faces.shape[0]
    fc = np.tile(COLOR_UNCOV, (n_faces, 1))
    fc[cov_masks[step]] = COLOR_COV
    ax.add_collection3d(Poly3DCollection(
        ast_tris, facecolors=to_rgba_array(fc), edgecolor="none", zorder=1))

    # 2. Asteroid wireframe (subsampled)
    ast_edges = ast_tris[:, [0, 1, 2, 0], :]
    for edge_verts in ast_edges[::25]:
        ax.plot(edge_verts[:, 0], edge_verts[:, 1], edge_verts[:, 2],
                color=COLOR_AST_EDGE, linewidth=0.3)

    # 3. Net edges
    net_lines = [net_positions[e] for e in net_edges]
    for pts in net_lines:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=COLOR_NET_EDGE, linewidth=0.6)

    # 4. Net nodes
    ax.scatter(net_positions[:, 0], net_positions[:, 1], net_positions[:, 2],
               s=1.5, c=[COLOR_NET_NODE])

    # 5. MDUs with occlusion
    for i in range(n_mdus):
        pos = net_positions[int(mdu_nodes[step, i])]
        mdu_vec = pos - ast_center
        occluded = np.dot(mdu_vec, look_dir) <= 0
        mdu_hex = TOL_MUTED[i % len(TOL_MUTED)].lstrip("#")
        mdu_rgb = tuple(int(mdu_hex[j:j+2], 16) / 255.0 for j in (0, 2, 4))
        color = COLOR_MDU_OCCLUDED if occluded else (*mdu_rgb, 1.0)
        size = 60 if occluded else 150
        edge_c = "gray" if occluded else "black"
        lw = 0.5 if occluded else 1.0
        ax.scatter(*pos, s=size, c=[color], marker="o", zorder=10,
                   edgecolors=edge_c, linewidths=lw)

    # 6. Info text
    ax.text2D(0.02, 0.98, f"Cov: {cov_rates[step]:.1%} | Step: {step} | "
              f"Az: {azim}deg, El: {elev}deg",
              transform=ax.transAxes, fontsize=10,
              verticalalignment="top",
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))


def main():
    p = argparse.ArgumentParser(description="3D snapshot at specified timestep")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--run", type=str, default=None)
    p.add_argument("--npz", type=str, default=None)
    p.add_argument("--step", type=int, required=True, help="Timestep to render")
    p.add_argument("--azim", type=str, default="0,45,90,135,180,225,270,315",
                   help="Comma-separated azimuth angles")
    p.add_argument("--elev", type=float, default=20, help="Elevation angle")
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.npz:
        npz_path = args.npz
    else:
        run_dir = find_run(os.path.join(root, "results"), tag=args.tag, run_name=args.run)
        if not run_dir:
            print("Run not found.")
            sys.exit(1)
        npz_path = os.path.join(run_dir, "trajectories", "trajectory_mappo.npz")

    data = np.load(npz_path)
    T = data["mdu_nodes"].shape[0]
    if args.step >= T:
        print(f"Step {args.step} >= trajectory length {T}. Using step {T - 1}.")
        step = T - 1
    else:
        step = args.step

    cov_rate = data["coverage_rates"][step]
    print(f"Step {step}/{T - 1}, Coverage: {cov_rate:.2%}")

    # Parse azimuths
    azim_list = [int(a.strip()) for a in args.azim.split(",")]

    # Output directory (default: analysis/outputs/)
    if args.output:
        out_dir = args.output
    else:
        out_dir = os.path.join(root, "analysis", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    for azim in azim_list:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        render_snapshot(ax, data, step, azim, args.elev)
        out_path = os.path.join(out_dir,
                                f"snapshot_step{step:04d}_az{azim:03d}_el{int(args.elev):02d}.png")
        plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
