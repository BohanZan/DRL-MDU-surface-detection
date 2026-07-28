"""
Render animation from pre-generated trajectory data.

Reads trajectory NPZ and outputs animation GIF(s) to a run directory.

Usage:
    python src/visualize_trajectory.py --name my_trajectory
    python src/visualize_trajectory.py --data-dir results/2026-06-11_143052_4mdu/trajectories
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import to_rgba_array

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Visualization constants (renderer-specific, not from config)
FPS = 10
COLOR_UNCOV = (0.5, 0.5, 0.5, 0.95)   # opaque asteroid surface
COLOR_COV = (0.2, 0.8, 0.2, 0.95)
COLOR_MDU = (1.0, 0.2, 0.0)
COLOR_MDU_OCCLUDED = (0.6, 0.6, 0.6, 0.3)  # ghosted when behind asteroid
COLOR_NODE = (0.3, 0.3, 0.8, 0.25)
COLOR_EDGE = (0.4, 0.4, 0.7, 0.15)


def main():
    p = argparse.ArgumentParser(
        description="Render animation from trajectory NPZ.")
    p.add_argument("--name", type=str, default="mappo",
                   help="Trajectory name (matches generate_trajectory.py --name)")
    # Run discovery (same API as generate_trajectory.py)
    p.add_argument("--tag", type=str, default=None,
                   help="Find run by tag (e.g. '4mdu')")
    p.add_argument("--run", type=str, default=None,
                   help="Exact run directory name or path")
    # Overrides
    p.add_argument("--data-dir", type=str, default=None,
                   help="Override: trajectory directory")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override: animation output directory")
    # Rendering
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--views", type=str, default="0,180",
                   help="Comma-separated azimuth angles, e.g. '0,180'")
    p.add_argument("--elevation", type=float, default=20)
    p.add_argument("--dpi", type=int, default=80)
    p.add_argument("--stride", type=int, default=1,
                   help="Render every Nth frame (default 1; 2 = half frames)")
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from src.run_manager import find_run

    # Find trajectory data
    if args.data_dir:
        data_path = os.path.join(args.data_dir, f"trajectory_{args.name}.npz")
        data_parent = args.data_dir
    else:
        results_root = os.path.join(root, "results")
        run_dir = find_run(results_root, tag=args.tag, run_name=args.run)
        if run_dir:
            data_path = os.path.join(run_dir, "trajectories", f"trajectory_{args.name}.npz")
            data_parent = os.path.join(run_dir, "trajectories")
        else:
            # Last resort: search all runs
            for d in sorted(os.listdir(results_root), reverse=True):
                p = os.path.join(results_root, d, "trajectories", f"trajectory_{args.name}.npz")
                if os.path.exists(p):
                    data_path = p
                    data_parent = os.path.join(results_root, d, "trajectories")
                    break
            else:
                print(f"ERROR: trajectory_{args.name}.npz not found in any results/ subdir.")
                print("Use --run, --tag, or --data-dir to specify.")
                return

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found.")
        return

    print(f"Loading {data_path}...")
    d = np.load(data_path)
    mdu_nodes = d["mdu_nodes"]
    cov_masks = d["coverage_masks"]
    cov_rates = d["coverage_rates"]
    net_pos = d["net_positions"]
    net_edges = d["net_edges"]
    ast_verts = d["ast_verts"]
    ast_faces = d["ast_faces"]

    N_faces = ast_faces.shape[0]
    T = len(mdu_nodes)
    ast_tris = ast_verts[ast_faces]
    net_lines = [net_pos[e] for e in net_edges]
    n_mdus = mdu_nodes.shape[1]

    print(f"  {T} timesteps, {N_faces} faces, {n_mdus} MDUs, "
          f"final cov={cov_rates[-1]:.2%}")

    all_pts = np.vstack([ast_verts, net_pos])
    half = np.ptp(all_pts, axis=0).max() / 2
    mid = np.mean(all_pts, axis=0)

    # Output directory: default to animations/ in the same run
    if args.output_dir:
        out_dir = args.output_dir
    else:
        run_dir = find_run(os.path.join(root, "results"), tag=args.tag, run_name=args.run)
        if run_dir:
            out_dir = os.path.join(run_dir, "animations")
        elif os.path.basename(data_parent) == "trajectories":
            out_dir = os.path.join(os.path.dirname(data_parent), "animations")
        else:
            out_dir = data_parent
    os.makedirs(out_dir, exist_ok=True)

    # Parse view angles
    view_angles = [int(a) for a in args.views.split(",")]
    final_cov = cov_rates[-1]
    n_views = len(view_angles)
    ast_edges = ast_tris[:, [0, 1, 2, 0], :]
    print(f"  Rendering {n_views} view(s): {view_angles}, final cov={final_cov:.2%}")

    for vi, azim in enumerate(view_angles):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

        tag = f"az{azim:03d}_cov{final_cov*100:.0f}"

        # Precompute camera look-direction for occlusion test.
        # In matplotlib: azim = rotation around Z (0 = +x direction),
        # elev = angle above XY plane. The camera looks AT the origin.
        az_rad = np.deg2rad(azim)
        el_rad = np.deg2rad(args.elevation)
        # Unit vector from camera toward the center (look direction)
        look_dir = np.array([
            np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            np.sin(el_rad),
        ])
        # Asteroid center
        ast_center = np.mean(ast_verts, axis=0)

        def draw(t, ax=ax, azim=azim):
            ax.clear()
            ax.set_xlim(mid[0] - half, mid[0] + half)
            ax.set_ylim(mid[1] - half, mid[1] + half)
            ax.set_zlim(mid[2] - half, mid[2] + half)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=args.elevation, azim=azim)
            ax.set_title(f"{args.name.upper()} | Step {t}/{T-1} | "
                         f"Cov: {cov_rates[t]:.1%}")

            # 1. Asteroid surface
            fc = np.array([COLOR_UNCOV] * N_faces)
            fc[cov_masks[t]] = COLOR_COV
            ax.add_collection3d(Poly3DCollection(
                ast_tris, facecolors=to_rgba_array(fc), edgecolor="none"))

            # 2. Net edges
            for pts in net_lines:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        color=COLOR_EDGE, linewidth=0.8)

            # 3. Asteroid wireframe (subsampled)
            for edge_verts in ast_edges[::20]:
                ax.plot(edge_verts[:, 0], edge_verts[:, 1], edge_verts[:, 2],
                        color=(0.3, 0.3, 0.3, 0.15), linewidth=0.3)

            # 4. Net nodes
            ax.scatter(net_pos[:, 0], net_pos[:, 1], net_pos[:, 2],
                       s=2, c=[COLOR_NODE])

            # 5. MDUs — with hemisphere occlusion check
            for i in range(n_mdus):
                pos = net_pos[mdu_nodes[t, i]]
                # Vector from asteroid center to MDU
                mdu_vec = pos - ast_center
                # MDU is on far side if it's in opposite hemisphere from camera
                occluded = np.dot(mdu_vec, look_dir) <= 0
                color = COLOR_MDU_OCCLUDED if occluded else COLOR_MDU
                size = 60 if occluded else 150
                edge_c = "gray" if occluded else "black"
                lw = 0.5 if occluded else 1.0
                ax.scatter(*pos, s=size, c=[color], marker="o", zorder=10,
                          edgecolors=edge_c, linewidths=lw)

            # 6. Info text
            ax.text2D(0.02, 0.98, f"Cov: {cov_rates[t]:.1%} | Az: {azim}°",
                      transform=ax.transAxes, fontsize=11,
                      verticalalignment="top",
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        print(f"  View {vi+1}/{n_views}: azimuth={azim} "
              f"(rendering {T} frames)...")
        frames_sub = range(0, T, args.stride)
        anim = animation.FuncAnimation(fig, draw, frames=frames_sub,
                                        repeat=False)
        out_path = os.path.join(out_dir, f"animation_{args.name}_{tag}.gif")
        print(f"  Saving to {out_path}...")
        anim.save(out_path, writer="pillow", fps=args.fps, dpi=args.dpi)
        plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
