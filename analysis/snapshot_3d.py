#!/usr/bin/env python
"""
3D snapshot of the MDU coverage state at a specified timestep and viewing angle.

Proper occlusion (hemisphere culling), perspective projection, FOV cone.

Usage:
    python analysis/snapshot_3d.py --tag FINAL_v15 --step 80 --azim "0,90"
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
COLOR_NET_EDGE = (0.3, 0.3, 0.6, 0.6)      # thicker, more visible
COLOR_NET_NODE = (0.2, 0.2, 0.5, 0.5)
COLOR_AST_EDGE = (0.3, 0.3, 0.3, 0.15)
CONE_ANGLE = np.deg2rad(80.0)  # full cone angle
CONE_RANGE = 300.0


def draw_fov_cone(ax, mdu_pos, ast_center, n_lines=16):
    """Draw the FOV cone wireframe from MDU pointing toward asteroid center.

    Cone axis: MDU → asteroid center (always inward).
    Radius at base scaled by cos(half_angle) for proper 80deg cone.
    """
    to_center = ast_center - mdu_pos
    dist_to_center = np.linalg.norm(to_center)
    cone_axis = to_center / (dist_to_center + 1e-10)

    # Build perpendicular basis
    ref = np.array([0.0, 0.0, 1.0]) if abs(cone_axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    perp1 = np.cross(cone_axis, ref)
    perp1 /= np.linalg.norm(perp1) + 1e-10
    perp2 = np.cross(cone_axis, perp1)

    # Reduce cone size 3x
    half_angle = CONE_ANGLE / 2
    R = CONE_RANGE / 3  # shorter cone
    r = R * np.tan(half_angle)
    center = mdu_pos + R * cone_axis

    # Base circle (closed)
    theta = np.linspace(0, 2 * np.pi, n_lines + 1)
    circle = np.array([center + r * (np.cos(t) * perp1 + np.sin(t) * perp2)
                       for t in theta])

    # Lines from apex to circle
    for pt in circle[:-1]:
        ax.plot([mdu_pos[0], pt[0]], [mdu_pos[1], pt[1]], [mdu_pos[2], pt[2]],
                color=(0.0, 0.0, 0.0, 1.0), linewidth=0.5, zorder=3)
    # Closed base circle
    ax.plot(circle[:, 0], circle[:, 1], circle[:, 2],
            color=(0.0, 0.0, 0.0, 1.0), linewidth=0.5, zorder=3)


def render_snapshot(ax, data, step, azim, elev, show_cone=True):
    mdu_nodes = data["mdu_nodes"]; cov_masks = data["coverage_masks"]
    cov_rates = data["coverage_rates"]; net_positions = data["net_positions"]
    net_edges = data["net_edges"]; ast_verts = data["ast_verts"]
    ast_faces = data["ast_faces"]

    n_mdus = mdu_nodes.shape[1]
    ast_tris = ast_verts[ast_faces]
    ast_center = np.mean(ast_verts, axis=0)

    all_pts = np.vstack([ast_verts, net_positions])
    half = np.ptp(all_pts, axis=0).max() / 2
    mid = np.mean(all_pts, axis=0)

    az_rad, el_rad = np.deg2rad(azim), np.deg2rad(elev)
    look_dir = np.array([np.cos(el_rad) * np.cos(az_rad),
                         np.cos(el_rad) * np.sin(az_rad), np.sin(el_rad)])

    ax.clear()
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_xlabel("X (m)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Y (m)", fontsize=14, fontweight="bold")
    ax.set_zlabel("Z (m)", fontsize=14, fontweight="bold")
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"Step {step}  |  Coverage: {cov_rates[step]:.1%}",
                 fontsize=18, fontweight="bold", pad=10)
    ax.tick_params(labelsize=11)

    # 1. Asteroid surface
    n_faces = ast_faces.shape[0]
    fc = np.tile(COLOR_UNCOV, (n_faces, 1))
    fc[cov_masks[step]] = COLOR_COV
    ax.add_collection3d(Poly3DCollection(
        ast_tris, facecolors=to_rgba_array(fc), edgecolor="none", zorder=1))

    # 2. Asteroid wireframe (subsampled)
    ast_edges = ast_tris[:, [0, 1, 2, 0], :]
    for ev in ast_edges[::25]:
        ax.plot(ev[:, 0], ev[:, 1], ev[:, 2],
                color=COLOR_AST_EDGE, linewidth=0.3)

    # 3. Net edges — thicker, more visible
    net_lines = [net_positions[e] for e in net_edges]
    for pts in net_lines:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=COLOR_NET_EDGE, linewidth=1.2, zorder=2)

    # 4. Net nodes — larger
    ax.scatter(net_positions[:, 0], net_positions[:, 1], net_positions[:, 2],
               s=5, c=[COLOR_NET_NODE], zorder=2)

    # 5. FOV cones
    if show_cone:
        for i in range(n_mdus):
            mdu_pos = net_positions[int(mdu_nodes[step, i])]
            draw_fov_cone(ax, mdu_pos, ast_center)

    # 6. MDUs with occlusion
    for i in range(n_mdus):
        pos = net_positions[int(mdu_nodes[step, i])]
        occluded = np.dot(pos - ast_center, look_dir) <= 0
        mdu_hex = TOL_MUTED[i % len(TOL_MUTED)].lstrip("#")
        mdu_rgb = tuple(int(mdu_hex[j:j+2], 16) / 255.0 for j in (0, 2, 4))
        color = COLOR_MDU_OCCLUDED if occluded else (*mdu_rgb, 1.0)
        size = 60 if occluded else 180
        ec = "gray" if occluded else "black"
        ax.scatter(*pos, s=size, c=[color], marker="o", zorder=10,
                   edgecolors=ec, linewidths=0.8 if occluded else 1.2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default=None); p.add_argument("--run", type=str, default=None)
    p.add_argument("--npz", type=str, default=None); p.add_argument("--step", type=int, required=True)
    p.add_argument("--azim", type=str, default="0,90,180,270")
    p.add_argument("--elev", type=float, default=20); p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--no-cone", action="store_true")
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.npz: npz_path = args.npz
    else:
        run_dir = find_run(os.path.join(root, "results"), tag=args.tag, run_name=args.run)
        if not run_dir: print("Run not found."); sys.exit(1)
        npz_path = os.path.join(run_dir, "trajectories", "trajectory_mappo.npz")

    data = np.load(npz_path)
    T = data["mdu_nodes"].shape[0]
    step = min(args.step, T - 1)
    print(f"Step {step}/{T - 1}, Coverage: {data['coverage_rates'][step]:.2%}")

    azim_list = [int(a.strip()) for a in args.azim.split(",")]
    out_dir = args.output or os.path.join(root, "analysis", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    for azim in azim_list:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        render_snapshot(ax, data, step, azim, args.elev, show_cone=not args.no_cone)
        out_path = os.path.join(out_dir,
                                f"snapshot_step{step:04d}_az{azim:03d}_el{int(args.elev):02d}.png")
        plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
