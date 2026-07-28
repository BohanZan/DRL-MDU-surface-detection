"""
Asteroid model: polyhedron mesh for coverage computation.
"""

import numpy as np
from typing import Tuple


def load_polyhedron(path: str, center: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load asteroid polyhedron: vertices and faces.

    Args:
        path: path to polyhedron_*.txt
        center: if True, shift vertices so mean is at origin

    Returns:
        verts: (N_verts, 3) vertex positions
        faces: (N_faces, 3) face vertex indices (0-indexed)
    """
    with open(path, "r") as f:
        header = f.readline().split()
        NumVerts = int(header[0])
        NumFaces = int(header[1])

        verts = np.zeros((NumVerts, 3))
        for i in range(NumVerts):
            line = f.readline().split()
            verts[i] = [float(line[0]), float(line[1]), float(line[2])]

        if center:
            vert_center = np.mean(verts, axis=0)
            verts = verts - vert_center

        faces = np.zeros((NumFaces, 3), dtype=int)
        for i in range(NumFaces):
            line = f.readline().split()
            faces[i] = [int(line[0]), int(line[1]), int(line[2])]

    return verts, faces


def compute_face_centroids(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute centroid of each triangular face."""
    return verts[faces].mean(axis=1)


def compute_face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute outward-pointing normal of each face."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return normals / norms


def compute_face_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area of each triangular face."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


class Asteroid:
    """Asteroid model: polyhedron mesh for coverage checking."""

    def __init__(self, polyhedron_path: str, center: bool = True):
        # Load mesh
        self.verts, self.faces = load_polyhedron(polyhedron_path, center=center)

        # Precompute face properties
        self.centroids = compute_face_centroids(self.verts, self.faces)
        self.normals = compute_face_normals(self.verts, self.faces)
        self.areas = compute_face_areas(self.verts, self.faces)
        self.N_faces = len(self.faces)

        # Overall stats
        self.total_area = float(self.areas.sum())
        self.radius_estimate = float(np.linalg.norm(self.verts, axis=1).max())

    def summary(self) -> str:
        return (
            f"Asteroid: {self.N_faces} faces, {len(self.verts)} vertices\n"
            f"  Total area: {self.total_area:.1f} m2\n"
            f"  Radius: ~{self.radius_estimate:.1f} m"
        )

    def compute_visible_faces(
        self,
        mdu_position: np.ndarray,
        cone_angle_deg: float = 80.0,
        range_max: float = 300.0,
    ) -> np.ndarray:
        """Compute which asteroid faces are visible from an MDU position.

        Cone-FOV check:
        1. Face centroid within cone half-angle
        2. Face centroid within range

        Args:
            mdu_position: (3,) MDU position in 3D space
            cone_angle_deg: full cone angle in degrees
            range_max: maximum detection range in meters

        Returns:
            visible: (N_faces,) boolean array
        """
        # Direction from MDU toward each face centroid
        to_face = self.centroids - mdu_position
        distances = np.linalg.norm(to_face, axis=1)

        # Range check
        in_range = distances < range_max
        if not np.any(in_range):
            return np.zeros(self.N_faces, dtype=bool)

        # Cone axis: from MDU toward the NEAREST surface point
        nearest_idx = np.argmin(distances)
        cone_axis = to_face[nearest_idx] / (distances[nearest_idx] + 1e-10)

        # Angle check
        dirs = to_face / (distances[:, np.newaxis] + 1e-10)
        cos_angles = dirs @ cone_axis
        half_angle_rad = np.deg2rad(cone_angle_deg / 2)
        in_cone = cos_angles > np.cos(half_angle_rad)

        # Occlusion: face must point toward MDU (backface culling)
        # A face whose normal points away from the MDU is on the
        # opposite side of the asteroid and cannot be seen.
        to_mdu = mdu_position - self.centroids
        to_mdu_norm = to_mdu / (np.linalg.norm(to_mdu, axis=1, keepdims=True) + 1e-10)
        facing_mdu = np.sum(self.normals * to_mdu_norm, axis=1) > 0.1

        # Visible = in range AND in cone AND facing MDU
        visible = in_range & in_cone & facing_mdu
        return visible
