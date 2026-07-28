"""
Net Graph: Load FNS topology + solution state, build traversal graph for MDUs.
"""

import numpy as np
from typing import Tuple, List


def load_fns_topology(path: str) -> Tuple[int, np.ndarray, np.ndarray]:
    """Load FNS file: edge connectivity + initial node positions.

    Args:
        path: path to FNS_*.txt file

    Returns:
        NumPoints: number of net nodes
        edges: (NumEdges, 2) array of edge connectivity (0-indexed)
        Pt_init: (NumPoints, 3) initial node positions
    """
    with open(path, "r") as f:
        header = f.readline().split()
        NumPoints = int(header[0])
        NumEdges = int(header[1])
        col_Topo = int(header[3])

        # Read edges
        edges = np.zeros((NumEdges, 2), dtype=int)
        for i in range(NumEdges):
            line = f.readline().split()
            edges[i] = [int(line[0]), int(line[1])]

        # Skip Topo matrix (NumPoints rows, col_Topo cols)
        for _ in range(NumPoints):
            f.readline()

        # Read initial Pt
        Pt_init = np.zeros((NumPoints, 3))
        for i in range(NumPoints):
            line = f.readline().split()
            Pt_init[i] = [float(line[0]), float(line[1]), float(line[2])]

    return NumPoints, edges, Pt_init


def load_solution_final_state(path: str, NumPoints: int) -> np.ndarray:
    """Load the LAST row of the solution file (final net positions).

    Args:
        path: path to Solution.dat or *_solution.txt
        NumPoints: number of net nodes

    Returns:
        positions: (NumPoints, 3) final node positions
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        row = data
    else:
        row = data[-1, :]  # last row = final state
    return row[:3 * NumPoints].reshape(NumPoints, 3)


def build_adjacency(edges: np.ndarray, NumPoints: int) -> List[np.ndarray]:
    """Build adjacency list from edge list.

    Args:
        edges: (NumEdges, 2) edge connectivity
        NumPoints: number of nodes

    Returns:
        adj: list of length NumPoints, adj[i] = array of neighbor indices
    """
    adj = [[] for _ in range(NumPoints)]
    for e in edges:
        u, v = int(e[0]), int(e[1])
        adj[u].append(v)
        adj[v].append(u)
    # Sort each neighbor list for deterministic ordering
    return [np.array(sorted(nb), dtype=int) for nb in adj]


def compute_edge_lengths(positions: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Compute Euclidean length of each edge."""
    p1 = positions[edges[:, 0]]
    p2 = positions[edges[:, 1]]
    return np.linalg.norm(p1 - p2, axis=1)


class NetGraph:
    """The fixed net graph - topology + final positions after asteroid capture."""

    def __init__(self, fns_path: str, solution_path: str, center: bool = True):
        # Load topology
        self.NumPoints, self.edges, self.Pt_init = load_fns_topology(fns_path)

        # Load final positions (the fixed net state for MDU traversal)
        self.positions = load_solution_final_state(solution_path, self.NumPoints)
        if center:
            self.center_offset = np.mean(self.positions, axis=0)
            self.positions = self.positions - self.center_offset
        else:
            self.center_offset = np.zeros(3)

        # Build adjacency list
        self.adj = build_adjacency(self.edges, self.NumPoints)

        # Precompute edge lengths
        self.edge_lengths = compute_edge_lengths(self.positions, self.edges)

        # Build edge-to-node mapping for fast lookup
        self._build_edge_map()

        # Validate
        assert len(self.adj) == self.NumPoints, \
            f"Adjacency size {len(self.adj)} != NumPoints {self.NumPoints}"

    def _build_edge_map(self):
        """Build a map from (u,v) to edge_index for fast lookup."""
        self.edge_map = {}
        for i, (u, v) in enumerate(self.edges):
            self.edge_map[(int(u), int(v))] = i
            self.edge_map[(int(v), int(u))] = i

    def get_neighbors(self, node: int) -> np.ndarray:
        """Get neighbor indices of a node."""
        return self.adj[node]

    def get_degree(self, node: int) -> int:
        """Degree (number of neighbors) of a node."""
        return len(self.adj[node])

    def get_edge_length(self, u: int, v: int) -> float:
        """Length of the edge between nodes u and v."""
        return self.edge_lengths[self.edge_map[(u, v)]]

    def max_degree(self) -> int:
        """Maximum degree across all nodes."""
        return max(len(nb) for nb in self.adj)

    def get_position(self, node: int) -> np.ndarray:
        """3D position of a node."""
        return self.positions[node]

    def summary(self) -> str:
        lines = [
            f"NetGraph: {self.NumPoints} nodes, {len(self.edges)} edges",
            f"  Max degree: {self.max_degree()}",
        ]
        return "\n".join(lines)
