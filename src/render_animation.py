"""
Standalone renderer: simulates MDU detection with cone FOV in real time.

Keeps its own rendering constants (colors, FPS, cone geometry) as a
standalone utility. Imports DataPaths and EnvConfig for shared geometry
and file path parameters.

Usage:
    python src/render_animation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import to_rgba_array

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.env import MDUCoverageEnv
from src.env.asteroid import compute_face_centroids
from src.config import DataPaths, EnvConfig as _DFLT

# Rendering constants (standalone — not from config)
N_MDUS = _DFLT.num_mdus
CONE_ANGLE = _DFLT.cone_angle_deg
CONE_RANGE = _DFLT.cone_range
MAX_STEPS = _DFLT.max_steps
FPS = 10

COLOR_UNCOVERED = (0.7, 0.7, 0.7, 0.5)
COLOR_COVERED = (0.2, 0.8, 0.2, 0.8)
COLOR_MDU = (1.0, 0.2, 0.0)
COLOR_NET_NODE = (0.3, 0.3, 0.8, 0.3)
COLOR_NET_EDGE = (0.4, 0.4, 0.7, 0.2)
COLOR_CONE = (1.0, 0.8, 0.0, 0.12)


def make_cone_polygons(apex, axis, half_angle, radius, n_segments=16):
    """Generate triangles for a 3D cone."""
    if abs(axis[2]) < 0.9:
        up = np.array([0, 0, 1.0])
    else:
        up = np.array([1.0, 0, 0])
    right = np.cross(axis, up)
    right = right / (np.linalg.norm(right) + 1e-10)
    up = np.cross(right, axis)

    base_center = apex + radius * axis
    base_radius = radius * np.tan(half_angle)
    angles = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    base_points = base_center + base_radius * (
        np.outer(np.cos(angles), right) + np.outer(np.sin(angles), up)
    )
    vertices = np.vstack([apex.reshape(1, 3), base_points])
    triangles = np.zeros((n_segments, 3), dtype=int)
    for i in range(n_segments):
        triangles[i] = [0, i + 1, (i + 1) % n_segments + 1]
    return vertices, triangles


def main():
    print("Setting up environment...")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = DataPaths().resolve(root)

    env = MDUCoverageEnv(
        fns_path=paths.fns,
        solution_path=paths.solution,
        polyhedron_path=paths.polyhedron,
        mdu_start_nodes=list(_DFLT.mdu_start_nodes)[:N_MDUS],
        cone_angle_deg=CONE_ANGLE,
        cone_range=CONE_RANGE,
        max_steps=MAX_STEPS,
        seed=42,
    )

    # Load trained model if available
    best_path = os.path.join(root, "checkpoints", "mappo_best.pt")
    use_model = os.path.exists(best_path)
    if use_model:
        from src.agents.mappo import MAPPO
        obs, info = env.reset()
        agent = MAPPO(obs.shape[1], info["global_state"].shape[0], env.max_deg)
        agent.load(best_path)
        print("Using trained model")
    else:
        print("No trained model, using random actions")

    # Precompute for rendering
    ast_triangles = env.asteroid.verts[env.asteroid.faces]
    net_lines = [env.net.positions[e] for e in env.net.edges]

    # Figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    all_pts = np.vstack([env.asteroid.verts, env.net.positions])
    half_range = np.ptp(all_pts, axis=0).max() / 2
    mid = np.mean(all_pts, axis=0)
    lims = (mid[0] - half_range, mid[0] + half_range,
            mid[1] - half_range, mid[1] + half_range,
            mid[2] - half_range, mid[2] + half_range)

    # Mutable state for the closure
    st = {"obs": None, "info": None, "done": False}

    def draw_frame(step):
        ax.clear()
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[2], lims[3])
        ax.set_zlim(lims[4], lims[5])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_box_aspect([1, 1, 1])

        cov_rate = (st["info"].get("coverage_rate", 0)
                    if st["info"] is not None else 0)
        ax.set_title(f"Step {step}/{MAX_STEPS}  Coverage: {cov_rate:.1%}")

        if step == 0:
            return

        # Step environment
        if not st["done"]:
            if use_model:
                acts, _, _ = agent.act(st["obs"], st["info"]["global_state"],
                                       greedy=True)
            else:
                mask = st["info"]["action_mask"]
                acts = np.array([np.random.choice(np.where(mask[i])[0])
                                 if mask[i].any() else env.max_deg - 1
                                 for i in range(N_MDUS)])
            nxt, _, term, trunc, st["info"] = env.step(acts)
            st["obs"] = nxt
            st["done"] = term or trunc

        # 1. Asteroid surface (colored by coverage)
        fc = np.array([COLOR_UNCOVERED] * env.asteroid.N_faces)
        if env.coverage_mask.any():
            fc[env.coverage_mask] = COLOR_COVERED
        mesh = Poly3DCollection(ast_triangles, facecolors=to_rgba_array(fc),
                                edgecolor="none", alpha=0.6)
        ax.add_collection3d(mesh)

        # 2. Net edges
        for pts in net_lines:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=COLOR_NET_EDGE, linewidth=0.3)

        # 3. Net nodes
        ax.scatter(env.net.positions[:, 0], env.net.positions[:, 1],
                   env.net.positions[:, 2], s=2, c=[COLOR_NET_NODE], alpha=0.3)

        # 4. MDUs
        for i, mdu in enumerate(env.mdus):
            pos = env.net.get_position(mdu.node)
            ax.scatter(*pos, s=80, c=[COLOR_MDU], marker="o", zorder=10)

        # 5. Cone FOV
        for mdu in env.mdus:
            pos = env.net.get_position(mdu.node)
            to_face = env.asteroid.centroids - pos
            dists = np.linalg.norm(to_face, axis=1)
            if dists.min() < CONE_RANGE:
                ni = np.argmin(dists)
                axis = to_face[ni] / dists[ni]
                verts, tris = make_cone_polygons(
                    pos, axis, np.deg2rad(CONE_ANGLE / 2),
                    min(dists[ni] * 0.9, CONE_RANGE)
                )
                cone = Poly3DCollection(verts[tris], alpha=COLOR_CONE[3],
                                        facecolor=COLOR_CONE[:3],
                                        edgecolor="none")
                ax.add_collection3d(cone)

        # 6. Info text
        cov = (st["info"].get("coverage_rate", 0)
               if st["info"] is not None else 0)
        ax.text2D(0.02, 0.98,
                  f"Covered: {cov:.1%}  "
                  f"Faces: {env.coverage_mask.sum()}/{env.asteroid.N_faces}",
                  transform=ax.transAxes, fontsize=11,
                  verticalalignment="top",
                  bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Init
    st["obs"], st["info"] = env.reset()
    st["done"] = False

    print(f"Rendering {MAX_STEPS + 1} frames...")
    anim = animation.FuncAnimation(fig, draw_frame,
                                   frames=range(MAX_STEPS + 1),
                                   repeat=False)

    out_dir = os.path.join(root, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mdu_coverage_animation.gif")
    print(f"Saving to {out_path}...")
    anim.save(out_path, writer="pillow", fps=FPS, dpi=120)
    print(f"Done! Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
