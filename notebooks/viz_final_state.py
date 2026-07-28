"""
Visualization: Final state of the FNS (Flexible Net System) wrapped around asteroid Bennu.

Reads:
  - FNS_square_fold-50m.txt   → net topology (nodes + edges)
  - Solution.dat              → trajectory (last row = final positions)
  - polyhedron_bennu.txt      → asteroid mesh (vertices + faces)

Output: 3D matplotlib figure showing net + asteroid
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# ──────────────────────────────────────────────
# Paths (from shared config)
# ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DataPaths, EnvConfig
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
paths = DataPaths().resolve(PROJECT_ROOT)
FNS_PATH = paths.fns
SOL_PATH = paths.solution
AST_PATH = paths.polyhedron


def load_fns_topology(path):
    """Load FNS file: returns NumPoints, edge list, and initial node positions."""
    with open(path, "r") as f:
        # Line 1: header
        header = f.readline().split()
        NumPoints = int(header[0])
        NumEdges  = int(header[1])
        # NumRopeNodes = int(header[2])
        # col_Topo = int(header[3])
        # Radius = float(header[4])

        # Read edges: NumEdges lines, each "node_i node_j"
        edges = []
        for _ in range(NumEdges):
            line = f.readline().split()
            edges.append([int(line[0]), int(line[1])])
        edges = np.array(edges, dtype=int)

        # Skip Topo matrix: NumPoints rows × col_Topo cols
        col_Topo = int(header[3])
        for _ in range(NumPoints):
            f.readline()

        # Read initial Pt: NumPoints rows, each "x y z"
        Pt = np.zeros((NumPoints, 3))
        for i in range(NumPoints):
            line = f.readline().split()
            Pt[i] = [float(line[0]), float(line[1]), float(line[2])]

        # Skip EL and BL (not needed for visualization)
        # (We don't read the rest — but file reading is done)

    return NumPoints, NumEdges, edges, Pt


def load_solution_final_state(path, NumPoints, center=True):
    """Load the LAST row of the solution file (final net state).

    Args:
        path: path to solution file
        NumPoints: number of net nodes
        center: if True, subtract the net's mean position to approximately
                align with the asteroid body frame (origin at center).
                The solution file positions are in the simulation frame,
                while the asteroid mesh is origin-centered.
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        row = data
    else:
        row = data[-1, :]  # last row = final state

    # First 3*NumPoints values = positions
    positions = row[:3*NumPoints].reshape(NumPoints, 3)

    if center:
        # Align net frame with asteroid body frame
        net_center = np.mean(positions, axis=0)
        positions = positions - net_center
        print(f"  Centered (subtracted net center [{net_center[0]:.1f}, {net_center[1]:.1f}, {net_center[2]:.1f}])")

    return positions


def load_asteroid_polyhedron(path, center=True):
    """Load polyhedron file: vertices (N×3) and faces (M×3).

    Args:
        path: path to polyhedron file
        center: if True, shift vertices so mean is at origin
    """
    with open(path, "r") as f:
        header = f.readline().split()
        NumVerts = int(header[0])
        NumFaces = int(header[1])

        # Vertices: NumVerts lines, each "x y z"
        verts = np.zeros((NumVerts, 3))
        for i in range(NumVerts):
            line = f.readline().split()
            verts[i] = [float(line[0]), float(line[1]), float(line[2])]

        if center:
            vert_center = np.mean(verts, axis=0)
            verts = verts - vert_center
            print(f"  Centered asteroid (subtracted [{vert_center[0]:.1f}, {vert_center[1]:.1f}, {vert_center[2]:.1f}])")

        # Faces: NumFaces lines, each "v1 v2 v3"
        faces = np.zeros((NumFaces, 3), dtype=int)
        for i in range(NumFaces):
            line = f.readline().split()
            faces[i] = [int(line[0]), int(line[1]), int(line[2])]

    return verts, faces


def plot_final_state(ast_verts, ast_faces, net_positions, net_edges, highlight_actuators=True):
    """3D plot: asteroid surface (grey) + net (blue wireframe).

    Args:
        highlight_actuators: if True, mark the 4 actuator nodes
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # ── Asteroid surface ──
    triangles = ast_verts[ast_faces]
    mesh = Poly3DCollection(triangles, alpha=0.6, facecolor="lightgrey", edgecolor="grey", linewidth=0.1)
    ax.add_collection3d(mesh)

    # ── Net nodes ──
    ax.scatter(
        net_positions[:, 0], net_positions[:, 1], net_positions[:, 2],
        s=6, c="blue", alpha=0.6, label="Net nodes"
    )

    # ── Net edges ──
    for edge in net_edges:
        pts = net_positions[edge]
        ax.plot(
            pts[:, 0], pts[:, 1], pts[:, 2],
            color="royalblue", linewidth=0.3, alpha=0.4
        )

    # ── Actuator nodes (4 corners) ──
    if highlight_actuators:
        act_indices = list(EnvConfig.mdu_start_nodes)  # from shared config
        act_pos = net_positions[act_indices]
        ax.scatter(
            act_pos[:, 0], act_pos[:, 1], act_pos[:, 2],
            s=120, c="red", marker="^", edgecolors="darkred",
            linewidths=1.5, zorder=5, label="Actuators"
        )

    # ── Axes ──
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Final State: FNS Wrapped Around Bennu", fontsize=14)

    # Auto-scale
    all_pts = np.vstack([ast_verts, net_positions])
    max_range = np.max(np.ptp(all_pts, axis=0)) / 2
    mid = np.mean(all_pts, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_box_aspect([1, 1, 1])

    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig, ax


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading FNS topology...")
    NumPoints, NumEdges, edges, Pt_init = load_fns_topology(FNS_PATH)
    print(f"  Nodes: {NumPoints}, Edges: {NumEdges}")

    print("Loading solution (final state)...")
    net_positions = load_solution_final_state(SOL_PATH, NumPoints)
    print(f"  Positions range: X [{net_positions[:,0].min():.1f}, {net_positions[:,0].max():.1f}]")
    print(f"                   Y [{net_positions[:,1].min():.1f}, {net_positions[:,1].max():.1f}]")
    print(f"                   Z [{net_positions[:,2].min():.1f}, {net_positions[:,2].max():.1f}]")

    print("Loading asteroid polyhedron...")
    ast_verts, ast_faces = load_asteroid_polyhedron(AST_PATH)
    print(f"  Vertices: {len(ast_verts)}, Faces: {len(ast_faces)}")

    print("Plotting...")
    fig, ax = plot_final_state(ast_verts, ast_faces, net_positions, edges)
    plt.show()

    # Save as image
    out_path = os.path.join(PROJECT_ROOT, "viz_final_state.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")
