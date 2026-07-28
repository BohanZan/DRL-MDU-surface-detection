#!/usr/bin/env python
"""
3D snapshot with proper occlusion via PyVista/VTK z-buffer.

Usage:
    python analysis/snapshot_pyvista.py --tag FINAL_v15 --step 80 --azim "0,90"
"""

import sys, os, argparse
import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.run_manager import find_run
from src.style_config import TOL_MUTED

CONE_RANGE = 300.0
CONE_HALF_ANGLE = np.deg2rad(40.0)


def make_asteroid_mesh(verts, faces, cov_mask):
    """Build asteroid PolyData with per-face coverage coloring."""
    # PyVista faces format: [n_verts, v0, v1, v2, ...]
    pv_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    mesh = pv.PolyData(verts, pv_faces)

    # Per-face colors: uncovered=gray, covered=green
    n_faces = faces.shape[0]
    colors = np.full((n_faces, 3), [0.5, 0.5, 0.5])         # gray default
    colors[cov_mask] = [0.2, 0.8, 0.2]                       # green covered
    mesh.cell_data["colors"] = colors
    return mesh


def make_net(positions, edges):
    """Build space net as lines + nodes."""
    # Lines
    line_segments = []
    for u, v in edges:
        line_segments.append([2, int(u), int(v)])  # 2 = line
    lines = pv.PolyData(positions, np.array(line_segments, dtype=np.int64))
    # Nodes
    nodes = pv.PolyData(positions)
    return lines, nodes


def make_fov_cone(mdu_pos, ast_center, n_segments=24):
    """Build an 80-degree FOV cone wireframe pointing toward asteroid center."""
    to_center = ast_center - mdu_pos
    cone_axis = to_center / (np.linalg.norm(to_center) + 1e-10)

    # Perpendicular basis
    ref = np.array([0., 0., 1.]) if abs(cone_axis[2]) < 0.9 else np.array([1., 0., 0.])
    perp1 = np.cross(cone_axis, ref)
    perp1 /= np.linalg.norm(perp1) + 1e-10
    perp2 = np.cross(cone_axis, perp1)

    R = CONE_RANGE / 3
    r = R * np.tan(CONE_HALF_ANGLE)
    apex = mdu_pos
    base_center = mdu_pos + R * cone_axis

    # Wireframe: lines from apex to base circle + closed base
    theta = np.linspace(0, 2 * np.pi, n_segments + 1)
    all_points = [apex]
    lines_array = []
    for i in range(n_segments):
        pt = base_center + r * (np.cos(theta[i]) * perp1 + np.sin(theta[i]) * perp2)
        all_points.append(pt)
        lines_array.append([2, 0, i + 1])  # apex -> circle point
    # Base circle
    base_start = len(all_points)
    for i in range(n_segments):
        pt = base_center + r * (np.cos(theta[i]) * perp1 + np.sin(theta[i]) * perp2)
        all_points.append(pt)
        if i < n_segments - 1:
            lines_array.append([2, base_start + i, base_start + i + 1])
    # Close the circle
    lines_array.append([2, base_start + n_segments - 1, base_start])

    cone = pv.PolyData(np.array(all_points), np.array(lines_array, dtype=np.int64))
    return cone


def make_mdu_sphere(center, radius=25.0):
    """Small sphere at MDU position."""
    return pv.Sphere(radius=radius, center=center)


def render(data, step, azim, elev, output_path, show_cone=True):
    mdu_nodes = data["mdu_nodes"]; cov_masks = data["coverage_masks"]
    cov_rates = data["coverage_rates"]; net_positions = data["net_positions"]
    net_edges = data["net_edges"]; ast_verts = data["ast_verts"]
    ast_faces = data["ast_faces"]
    ast_center = np.mean(ast_verts, axis=0)
    n_mdus = mdu_nodes.shape[1]

    # ── Build scene ──
    plotter = pv.Plotter(window_size=(1600, 1200), off_screen=True)
    plotter.set_background("white")

    # Asteroid
    ast = make_asteroid_mesh(ast_verts, ast_faces, cov_masks[step])
    plotter.add_mesh(ast, scalars="colors", rgb=True, show_scalar_bar=False,
                     ambient=0.5, diffuse=0.5)

    # Space net
    net_lines, net_nodes = make_net(net_positions, net_edges)
    plotter.add_mesh(net_lines, color=[0.3, 0.3, 0.6], line_width=1.5,
                     opacity=0.7, render_lines_as_tubes=False)
    plotter.add_mesh(net_nodes, color=[0.2, 0.2, 0.5], point_size=6,
                     opacity=0.6, render_points_as_spheres=True)

    # FOV cones
    if show_cone:
        for i in range(n_mdus):
            mdu_pos = net_positions[int(mdu_nodes[step, i])]
            cone = make_fov_cone(mdu_pos, ast_center)
            plotter.add_mesh(cone, color=[0.0, 0.0, 0.0], line_width=1.0,
                             opacity=0.4)

    # MDU markers
    for i in range(n_mdus):
        mdu_pos = net_positions[int(mdu_nodes[step, i])]
        mdu_hex = TOL_MUTED[i % len(TOL_MUTED)].lstrip("#")
        mdu_rgb = tuple(int(mdu_hex[j:j+2], 16) / 255.0 for j in (0, 2, 4))
        sphere = make_mdu_sphere(mdu_pos, radius=22.0)
        plotter.add_mesh(sphere, color=list(mdu_rgb), ambient=0.6, diffuse=0.4)

    # ── Camera ──
    # Position camera on a sphere around the center at fixed distance
    ast_center_arr = np.array(ast_center)
    radius = 900.0  # camera distance from asteroid center
    az_rad, el_rad = np.deg2rad(azim), np.deg2rad(elev)
    cam_pos = ast_center_arr + radius * np.array([
        np.cos(el_rad) * np.cos(az_rad),
        np.cos(el_rad) * np.sin(az_rad),
        np.sin(el_rad)
    ])
    up = np.array([0.0, 0.0, 1.0])
    plotter.camera.position = cam_pos
    plotter.camera.focal_point = ast_center_arr
    plotter.camera.up = up

    # ── Title ──
    plotter.add_title(f"Step {step}  |  Coverage: {cov_rates[step]:.1%}",
                      font_size=18, color="black", font="times")

    # ── Render ──
    plotter.show(auto_close=False)
    plotter.screenshot(output_path)
    plotter.close()
    print(f"Saved: {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default=None); p.add_argument("--run", type=str, default=None)
    p.add_argument("--npz", type=str, default=None); p.add_argument("--step", type=int, required=True)
    p.add_argument("--azim", type=str, default="0,90,180,270")
    p.add_argument("--elev", type=float, default=20); p.add_argument("--dpi", type=int, default=150)
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
        out_path = os.path.join(out_dir,
                                f"snapshot_pv_step{step:04d}_az{azim:03d}_el{int(args.elev):02d}.png")
        render(data, step, azim, args.elev, out_path, show_cone=not args.no_cone)


if __name__ == "__main__":
    main()
