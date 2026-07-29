#!/usr/bin/env python
"""
3D snapshot with proper occlusion via PyVista/VTK z-buffer.

Our data uses Z-up; PyVista uses Y-up. All 3D coordinates are converted
by swapping Y<->Z: (x, y, z) → (x, z, y).

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


def to_pv(xyz):
    """Convert Z-up coords to PyVista Y-up: (x, y, z) → (x, z, y)."""
    out = np.array(xyz, dtype=np.float64, copy=True)
    if out.ndim == 1:
        out[1], out[2] = out[2], out[1]
    else:
        out[:, [1, 2]] = out[:, [2, 1]]
    return out


def make_asteroid_mesh(verts, faces, cov_mask):
    pv_verts = to_pv(verts)
    pv_faces = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64),
                          np.array(faces, dtype=np.int64)]).ravel()
    mesh = pv.PolyData(pv_verts, pv_faces)
    n_faces = faces.shape[0]
    colors = np.full((n_faces, 3), [0.50, 0.50, 0.50])
    colors[cov_mask] = [0.20, 0.80, 0.20]
    mesh.cell_data["colors"] = colors
    return mesh


def make_net(positions, edges):
    pv_pos = to_pv(positions)
    N = len(pv_pos)
    lines = []
    for u, v in edges:
        lines.extend([2, int(u), int(v)])
    lines_pd = pv.PolyData()
    lines_pd.points = pv_pos
    lines_pd.lines = np.array(lines, dtype=np.int64)
    nodes_pd = pv.PolyData(pv_pos)
    return lines_pd, nodes_pd


def make_fov_cone(mdu_pos, ast_center, n_segments=18):
    """Build FOV cone in Z-up, convert to PyVista Y-up."""
    to_center = ast_center - mdu_pos
    dist = np.linalg.norm(to_center)
    cone_axis = to_center / (dist + 1e-10)

    ref = np.array([0., 0., 1.]) if abs(cone_axis[2]) < 0.9 else np.array([1., 0., 0.])
    perp1 = np.cross(cone_axis, ref)
    perp1 /= np.linalg.norm(perp1) + 1e-10
    perp2 = np.cross(cone_axis, perp1)

    R = CONE_RANGE / 3; r = R * np.tan(CONE_HALF_ANGLE)
    apex = np.array(mdu_pos)
    base_c = apex + R * cone_axis

    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    circle = np.array([base_c + r * (np.cos(t) * perp1 + np.sin(t) * perp2)
                       for t in theta])

    pts_zup = np.vstack([apex.reshape(1, 3), circle])
    pts_pv = to_pv(pts_zup)

    lines = []
    for i in range(n_segments):
        lines.extend([2, 0, i + 1])
        lines.extend([2, i + 1, 1 + (i + 1) % n_segments])

    cone = pv.PolyData(); cone.points = pts_pv
    cone.lines = np.array(lines, dtype=np.int64)
    return cone


def render(data, step, azim, elev, output_path, show_cone=True):
    mdu_nodes = data["mdu_nodes"]; cov_masks = data["coverage_masks"]
    cov_rates = data["coverage_rates"]; net_positions = data["net_positions"]
    net_edges = data["net_edges"]; ast_verts = data["ast_verts"]
    ast_faces = data["ast_faces"]
    ast_center = np.mean(ast_verts, axis=0)
    n_mdus = mdu_nodes.shape[1]

    plotter = pv.Plotter(window_size=(1600, 1200), off_screen=True)
    plotter.set_background("white")

    # Asteroid
    ast = make_asteroid_mesh(ast_verts, ast_faces, cov_masks[step])
    plotter.add_mesh(ast, scalars="colors", rgb=True, show_scalar_bar=False,
                     ambient=0.4, diffuse=0.6, opacity=0.55)

    # Net
    net_l, net_n = make_net(net_positions, net_edges)
    plotter.add_mesh(net_l, color=[0.25, 0.25, 0.55], line_width=2.5,
                     opacity=0.9, render_lines_as_tubes=True)
    plotter.add_mesh(net_n, color=[0.15, 0.15, 0.45], point_size=10,
                     render_points_as_spheres=True, opacity=0.8)

    # FOV cones
    if show_cone:
        for i in range(n_mdus):
            mdu_pos = net_positions[int(mdu_nodes[step, i])]
            cone = make_fov_cone(mdu_pos, ast_center)
            plotter.add_mesh(cone, color=[0.0, 0.0, 0.0], line_width=2.5,
                             opacity=1.0, render_lines_as_tubes=True)

    # MDU markers
    for i in range(n_mdus):
        mdu_pos_zu = net_positions[int(mdu_nodes[step, i])]
        mdu_hex = TOL_MUTED[i % len(TOL_MUTED)].lstrip("#")
        mdu_rgb = tuple(int(mdu_hex[j:j+2], 16) / 255.0 for j in (0, 2, 4))
        sphere = pv.Sphere(radius=15.0, center=to_pv(mdu_pos_zu))
        plotter.add_mesh(sphere, color=list(mdu_rgb))

    # Camera: PyVista Y-up. (azim, elev) from our Z-up convention.
    # In Z-up: camera at distance R with spherical (azim, elev).
    # Convert to Y-up for PyVista.
    az_rad, el_rad = np.deg2rad(azim), np.deg2rad(elev)
    R = 1200.0
    cam_zu = R * np.array([
        np.cos(el_rad) * np.cos(az_rad),
        np.cos(el_rad) * np.sin(az_rad),
        np.sin(el_rad)
    ])
    cam_pv = to_pv(cam_zu)
    focal_pv = to_pv(ast_center)
    plotter.camera.position = cam_pv
    plotter.camera.focal_point = focal_pv
    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.camera.clipping_range = (10.0, 10000.0)

    # Title
    plotter.add_title(f"Step {step}  |  Coverage: {cov_rates[step]:.1%}",
                      font_size=18, color="black")

    plotter.show(auto_close=False)
    plotter.screenshot(output_path, return_img=False)
    plotter.close()
    print(f"Saved: {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default=None); p.add_argument("--run", type=str, default=None)
    p.add_argument("--npz", type=str, default=None); p.add_argument("--step", type=int, required=True)
    p.add_argument("--azim", type=str, default="0,90,180,270")
    p.add_argument("--elev", type=float, default=20)
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
