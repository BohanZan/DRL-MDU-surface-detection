"""
MDU Coverage Environment — v4 (memory-efficient).
Precomputes all visible faces ONCE. No redundant numpy allocations.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, List
from dataclasses import dataclass
from .net_graph import NetGraph
from .asteroid import Asteroid


@dataclass
class MDUState:
    node: int
    path: List[int]


class MDUCoverageEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, fns_path, solution_path, polyhedron_path,
                 mdu_start_nodes=None, num_mdus=None,
                 cone_angle_deg=None, cone_range=None,
                 max_steps=None, coverage_threshold=None,
                 completion_threshold=None,
                 r_newly=None, r_completion=None, r_speed=None,
                 seed=None, render_mode=None):
        super().__init__()
        # Pull defaults from shared config (None = use default)
        from src.config import EnvConfig as _DFLT
        self._rng = np.random.default_rng(seed)
        self.net = NetGraph(fns_path, solution_path, center=True)
        self.asteroid = Asteroid(polyhedron_path, center=True)

        default_nodes = list(_DFLT.mdu_start_nodes)
        if mdu_start_nodes is not None:
            self.mdu_start_nodes = list(mdu_start_nodes)
        elif num_mdus is not None:
            self.mdu_start_nodes = default_nodes[:num_mdus]
        else:
            self.mdu_start_nodes = default_nodes
        self.num_mdus = len(self.mdu_start_nodes)
        assert self.num_mdus >= 1

        self.cone_angle_deg = cone_angle_deg if cone_angle_deg is not None else _DFLT.cone_angle_deg
        self.cone_range = cone_range if cone_range is not None else _DFLT.cone_range
        self.max_steps = max_steps if max_steps is not None else _DFLT.max_steps
        self.coverage_threshold = coverage_threshold if coverage_threshold is not None else _DFLT.coverage_threshold
        self.completion_threshold = completion_threshold if completion_threshold is not None else _DFLT.completion_threshold
        self.render_mode = render_mode

        # Reward coefficients
        self.r_newly = r_newly if r_newly is not None else _DFLT.r_newly
        self.r_completion = r_completion if r_completion is not None else _DFLT.r_completion
        self.r_speed = r_speed if r_speed is not None else _DFLT.r_speed

        # ═══ PRECOMPUTE EVERYTHING ═══
        self._precompute_graph_distances()
        self._precompute_visible_faces()
        self._precompute_khops_and_coverage()

        # NO stay action
        self.max_deg = self.net.max_degree()
        self.action_space = spaces.Discrete(self.max_deg)

        # obs: [self_pos(3), local_cov(3), global_cov(1),
        #       other_mdus_relative(3*(N-1)), graph_dist_to_others(N-1),
        #       visit(1), step(1),
        #       candidate_positions(max_deg*8)]
        #       candidate = (gx,gy,gz, rx,ry,rz, uncovered_value, inv_visit)
        self.obs_dim = (3 + 3 + 1 + 3 * (self.num_mdus - 1)
                        + (self.num_mdus - 1)  # graph distances
                        + 1 + 1
                        + self.max_deg * 8)
        self.observation_space = spaces.Box(-1, 1, (self.obs_dim,), dtype=np.float32)
        self.global_dim = 3 * self.num_mdus + 1 + 1
        self.global_state_space = spaces.Box(-1, 1, (self.global_dim,), dtype=np.float32)

        self.mdus = []
        self.coverage_mask = np.zeros(self.asteroid.N_faces, dtype=bool)
        self.step_count = 0
        self._visit_counts = np.zeros(self.net.NumPoints, dtype=int)
        self._completion_step = -1  # first step where completion_threshold was hit

    # ── Graph distance precomputation ────────────────────────

    def _precompute_graph_distances(self):
        """Precompute all-pairs shortest path distances (graph hops)."""
        N = self.net.NumPoints
        # BFS from each node
        self._graph_dist = np.full((N, N), N, dtype=np.int32)  # N = "infinity"
        for i in range(N):
            self._graph_dist[i, i] = 0
            dist = 0
            frontier = [i]
            visited = {i}
            while frontier:
                dist += 1
                next_frontier = []
                for u in frontier:
                    for v in self.net.adj[u]:
                        if v not in visited:
                            visited.add(v)
                            self._graph_dist[i, v] = dist
                            next_frontier.append(v)
                frontier = next_frontier
        self._graph_diam = int(self._graph_dist.max())

    # ── Precomputations ─────────────────────────────────────────

    def _precompute_visible_faces(self):
        """Compute visible faces for ALL 369 nodes ONCE."""
        N = self.net.NumPoints
        self._node_visible = np.zeros((N, self.asteroid.N_faces), dtype=bool)
        for i in range(N):
            pos = self.net.get_position(i)
            self._node_visible[i] = self.asteroid.compute_visible_faces(
                pos, self.cone_angle_deg, self.cone_range
            )

    def _precompute_khops_and_coverage(self):
        """3-hop neighbors + precomputed coverage features for local coverage."""
        adj = [set(self.net.adj[i]) for i in range(self.net.NumPoints)]
        self._khop = {}
        self._node_local_visible_count = np.zeros(self.net.NumPoints, dtype=int)
        self._node_local_visible = {}  # node -> set of faces visible from its 1/2/3 hop

        for i in range(self.net.NumPoints):
            one = adj[i]
            two = set()
            for n in one:
                two.update(adj[n])
            two -= {i}
            two -= one
            three = set()
            for n in two:
                three.update(adj[n])
            three -= {i}
            three -= one
            three -= two
            self._khop[i] = (one, two, three)

            # Pre-merge visible face indices for each hop level
            self._node_local_visible[i] = {}
            for name, nh in [("1", one), ("2", two), ("3", three)]:
                if not nh:
                    self._node_local_visible[i][name] = np.array([], dtype=int)
                else:
                    # Union of visible faces across neighbors in this hop
                    vis_set = set()
                    for n in nh:
                        vis_set.update(np.where(self._node_visible[n])[0].tolist())
                    self._node_local_visible[i][name] = np.array(list(vis_set), dtype=int)

    # ── Observation ─────────────────────────────────────────────

    def _local_coverage(self, node):
        """Fast local coverage using precomputed face indices."""
        result = []
        for name in ["1", "2", "3"]:
            face_ids = self._node_local_visible[node][name]
            if len(face_ids) == 0:
                result.append(0.0)
            else:
                covered_count = self.coverage_mask[face_ids].sum()
                result.append(covered_count / len(face_ids))
        return np.array(result, dtype=np.float32)

    def _get_obs(self):
        R = self.asteroid.radius_estimate
        obs = np.zeros((self.num_mdus, self.obs_dim), dtype=np.float32)
        for i, mdu in enumerate(self.mdus):
            pos = self.net.get_position(mdu.node)
            o = obs[i]
            o[0:3] = pos / R
            o[3:6] = self._local_coverage(mdu.node)
            o[6] = self.coverage_mask.mean()
            idx = 7
            for j, other in enumerate(self.mdus):
                if j == i:
                    continue
                o[idx:idx+3] = (self.net.get_position(other.node) - pos) / R
                idx += 3
            # Graph distances to other MDUs (normalized by graph diameter)
            gd_start = idx
            for j, other in enumerate(self.mdus):
                if j == i:
                    continue
                o[idx] = self._graph_dist[mdu.node, other.node] / max(self._graph_diam, 1)
                idx += 1
            o[idx] = min(self._visit_counts[mdu.node] / 10.0, 1.0)
            o[idx+1] = self.step_count / self.max_steps

            # ── Physical candidate positions ──────────────────────────
            # Layout: [neighbor_k_g(3), neighbor_k_r(3), uncovered_value(1),
            #           inv_visit(1), ...]
            # Invalid slots (padded beyond actual degree) remain zero.
            cand_start = idx + 2
            nbs = self.net.get_neighbors(mdu.node)
            for k, nb in enumerate(nbs):
                nb_pos = self.net.get_position(nb)
                off = cand_start + k * 8
                o[off:off+3] = nb_pos / R                    # global position
                o[off+3:off+6] = (nb_pos - pos) / R          # relative offset
                # Coverage value: fraction of this node's visible faces
                # that are NOT yet covered
                visible_faces = self._node_visible[nb]
                newly = visible_faces & ~self.coverage_mask
                o[off+6] = newly.sum() / max(visible_faces.sum(), 1)
                # Inverse visit count: prefer unexplored nodes
                # (global counter shared across all MDUs)
                o[off+7] = 1.0 / max(self._visit_counts[nb], 1)
        return obs

    def _get_global_state(self):
        R = self.asteroid.radius_estimate
        s = np.zeros(self.global_dim, dtype=np.float32)
        idx = 0
        for mdu in self.mdus:
            s[idx:idx+3] = self.net.get_position(mdu.node) / R
            idx += 3
        s[idx] = self.coverage_mask.mean()
        s[idx+1] = self.step_count / self.max_steps
        return s

    def _get_action_mask(self):
        mask = np.zeros((self.num_mdus, self.max_deg), dtype=bool)
        occ = set(mdu.node for mdu in self.mdus)
        for i, mdu in enumerate(self.mdus):
            for j, nb in enumerate(self.net.get_neighbors(mdu.node)):
                if nb not in occ:
                    mask[i, j] = True
        # Edge case: if all neighbors occupied, force-random a fallback
        for i in range(self.num_mdus):
            if not mask[i].any():
                mask[i, :] = True  # let the agent pick something (will be handled in step)
        return mask

    # ── Reward ────────────────────────────────────────────────

    def _compute_reward(self, newly_faces, r_explore=0.0):
        """Per-step: coverage discovery + exploration bonus."""
        return float(self.r_newly * newly_faces + r_explore)

    def _get_coverage_bonus(self, cr: float, step_count: int) -> float:
        """Gompertz S-curve bonus at episode end.

        Double-exponential: 0 for cr<=70%, convex rise 70-75%, concave 75-100%.
        Output range [0, 1], scaled by bonus_scale.
        """
        b = 5.0
        k = 45.0
        if cr <= 0.70:
            f = 0.0
        else:
            f = (np.exp(-b * np.exp(-k * (cr - 0.70))) - np.exp(-b)) / (1.0 - np.exp(-b))
        # Speed factor: up to 1.5x for fast coverage, 1.0x at max_steps
        speed_factor = 1.0 + 0.5 * (1.0 - step_count / self.max_steps)
        bonus_scale = 500.0  # max bonus at 100% coverage
        return float(bonus_scale * f * speed_factor)

    # ── Reset / Step ────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.coverage_mask = np.zeros(self.asteroid.N_faces, dtype=bool)
        self.step_count = 0
        self._visit_counts = np.zeros(self.net.NumPoints, dtype=int)
        self._completion_step = -1
        self.mdus = [MDUState(node=n, path=[n]) for n in self.mdu_start_nodes]
        for mdu in self.mdus:
            self._visit_counts[mdu.node] += 1
        for i in range(self.num_mdus):
            self.coverage_mask[self._node_visible[self.mdus[i].node]] = True
        obs = self._get_obs()
        info = {"global_state": self._get_global_state(),
                "action_mask": self._get_action_mask(),
                "coverage_rate": float(self.coverage_mask.mean())}
        return obs, info

    def step(self, actions):
        assert len(actions) == self.num_mdus
        am = self._get_action_mask()
        for i, a in enumerate(actions):
            if a < 0 or a >= self.max_deg or not am[i, a]:
                valid = np.where(am[i])[0]
                a = self._rng.choice(valid) if len(valid) > 0 else 0
            nb = self.net.get_neighbors(self.mdus[i].node)
            if a < len(nb):
                nn = nb[a]
                self.mdus[i].node = nn
                self.mdus[i].path.append(nn)
                self._visit_counts[nn] += 1

        self.step_count += 1

        # Coverage update + exploration bonus
        newly = 0.0
        newly_faces = 0
        r_explore = 0.0
        for mdu in self.mdus:
            vis = self._node_visible[mdu.node]
            new_faces = vis & ~self.coverage_mask
            newly += float(self.asteroid.areas[new_faces].sum())
            newly_faces += int(new_faces.sum())
            self.coverage_mask[vis] = True
            # Exploration: reward visiting nodes with low global visit count
            # 1.0 at first visit, decays as 1/visit_count
            vc = self._visit_counts[mdu.node]
            if vc == 1:
                r_explore += 1.0  # first visit by anyone
            elif vc <= 3:
                r_explore += 0.5 / vc  # diminishing returns

        cr = float(self.coverage_mask.mean())

        trun = self.step_count >= self.max_steps
        # Early termination at Gompertz inflection point (~75%)
        term = cr >= 0.75
        r = self._compute_reward(newly_faces, r_explore)
        if trun:
            r += self._get_coverage_bonus(cr, self.step_count)
        obs = self._get_obs()
        info = {"global_state": self._get_global_state(),
                "action_mask": self._get_action_mask(),
                "coverage_rate": cr,
                "completion_step": self._completion_step}
        return obs, float(r), term, trun, info

    def render(self):
        if self.render_mode == "human":
            print(f"Step {self.step_count} | Cov: {self.coverage_mask.mean():.1%}")

    def close(self):
        pass
