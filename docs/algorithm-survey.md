# Algorithm & Reward Design Survey — MDU Path Planning

> Target: multi-agent DRL for cooperative coverage path planning on a graph
> (space net enveloping an asteroid). Simple, effective, proven algorithms.
> **Framework: PyTorch** for all models, training loops, and environment logic.

---

## 0. Problem Geometry: 2D Web on 3D Asteroid

This is the central geometric challenge of the project.

### The Setup

```
                     ┌─────────────────────────┐
                     │      MDU (on net)        │
                     │    ●━━━━━━━━━━━━━━━━     │
                     │    ╲  cone FOV           │
                     │     ╲   ╱                │
                     │      ╲ ╱                 │
  Net (2D mesh ──────┼───────●────────────────  │
  wrapped around     │    asteroid surface      │
  asteroid)          │    (3D target)           │
                     └─────────────────────────┘
```

### Key Geometric Facts

| Property | Description |
|----------|-------------|
| **Net** | 2D triangular mesh embedded in 3D space. Nodes have `(x,y,z)` positions. Edges define connectivity between adjacent nodes. |
| **Asteroid** | 3D convex/irregular body centered at origin (~0,0,0). Represented as polyhedron mesh (1348 verts for Bennu) AND spherical harmonics smooth surface. |
| **Net-to-asteroid** | Net wraps around asteroid at some distance. The final solution file has net positions ~1300m offset from asteroid center (simulation frame). Need coordinate alignment. |
| **MDU movement** | MDU moves along graph **edges** between net **nodes**. Movement is on the 2D manifold, not through free space. |
| **Detection** | MDU's cone FOV points **inward** toward asteroid surface, detecting the 3D surface below. |

### How to Simulate MDU Movement on the Net

#### Step 1: Construct the Traversal Graph

```
From solution file final state:
  nodes[i] = (x_i, y_i, z_i)            # 3D positions of all net nodes
  edges[k] = (node_a, node_b)            # connectivity (unchanged from FNS file)
  
For each node, precompute:
  neighbors[i] = [j | edge(i,j) exists]  # adjacency list
  edge_length[i][j] = ||nodes[i] - nodes[j]||₂  # Euclidean distance in 3D
```

#### Step 2: Define MDU Position at Any Time

An MDU is always at one of two states:

```
State A — ON A NODE (discrete position):
  mdu_pos = node_index    # exact graph node
  
State B — TRANSITING ALONG AN EDGE (optional refinement):
  mdu_pos = lerp(nodes[a], nodes[b], t)   # t ∈ [0, 1] along edge
```

**Recommended: Use State A (node-only).** Rationale:
- Keeps action space discrete and simple
- The net is dense enough (369 nodes) that node-to-node movement is sufficient
- Each step = move to adjacent node (one edge traversal)
- Avoids continuous control complexity
- For a 369-node net with average degree ~4, exploration is non-trivial

#### Step 3: Movement Physics (Simplified)

Since we don't need net dynamics (net is fixed):

```
Per step:
  1. MDU selects action a ∈ {0..deg(current_node)-1}
  2. MDU moves to neighbor node[action]
  3. Time cost = edge_length / mdu_speed  (constant per edge)
  4. At new node: activate detection cone → compute coverage
```

**Movement constraint:** MDUs cannot occupy the same node simultaneously.
(Simple collision avoidance — mask out occupied nodes from action space.)

#### Step 4: Cone FOV → 3D Surface Coverage

At each node position `P`, the MDU's detection cone is defined:

```
Cone apex:    P (MDU position on net)
Cone axis:    points from P toward nearest point on asteroid surface (inward)
Cone angle:   α (half-angle, e.g., 30° = 60° full FOV)
Range:        R_max (max detection distance, e.g., 100m)
```

Coverage computation (per MDU step):

```python
def compute_coverage(mdu_pos, asteroid_mesh, cone_angle, range_max):
    """
    Returns set of asteroid mesh faces visible to this MDU.
    """
    # 1. Cone axis = direction from mdu_pos toward asteroid center
    axis = -mdu_pos / ||mdu_pos||  # points inward
    
    # 2. For each face centroid C of asteroid mesh:
    for face in asteroid_mesh.faces:
        centroid = face.centroid
        direction = centroid - mdu_pos
        
        # 3. Check if face is within cone:
        if dot(direction/||direction||, axis) > cos(cone_angle):  # within cone
        if ||direction|| < range_max:                              # within range
        if not occluded_by_net_or_other_faces():                   # line-of-sight
            faces.add(face)
    
    return faces
```

**Performance note:** For 2692 faces × 4 MDUs, this is ~10K checks per step.
Optimize with spatial hashing or GPU parallelization if needed.

### Coordinate Frame Alignment

The solution file positions are in the **simulation frame** (net at y≈+1300).
The asteroid SH surface is in the **body frame** (origin at center of mass).

```python
# Alignment: center the net around the asteroid
net_center = mean(solution_positions, axis=0)  # ~[0, 1300, 0]
net_positions_body = solution_positions - net_center  # now centered

# Now both net and asteroid share the same coordinate frame
# Verify: asteroid surface R(θ,φ) ~ 250m at origin
# Verify: net nodes at ~[±200-400, 1300→0, ±200-400] after centering
```

---

## 1. Algorithm Recommendations

### 🥇 Primary Choice: MAPPO + GNN Encoder

**Why:** Stable, CTDE (centralized training/decentralized execution), strong
cooperative benchmarks, simple to implement.

```
┌─────────────────────────────────────────────┐
│  MAPPO Architecture                          │
│                                              │
│  Training:                                   │
│    Actor_i(obs_i) → action_i   (per MDU)     │
│    Critic(obs_1..N, state) → V(s) (shared)   │
│                                              │
│  Execution:                                  │
│    Actor_i(obs_i) → action_i   (no critic)   │
│                                              │
│  Loss: L = L_clip + c1·L_value - c2·H(π)    │
└─────────────────────────────────────────────┘
```

**Key references:**
- Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (NeurIPS 2022) — MAPPO paper
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017) — base PPO

**Why for your case:**
- Discrete actions (move to neighbor node) — PPO handles this natively
- MDUs are homogeneous — share policy parameters (parameter sharing)
- Centralized critic sees all MDU positions + global coverage map
- Has been proven on StarCraft, MPE, and coverage tasks

### 🥈 Backup: QMIX

**Why:** Value decomposition is elegant for cooperative tasks. Individual Q-values
are easy to interpret. Monotonic mixing ensures consistency.

```
Q_tot(s, a) = f_mix(Q_1(o_1, a_1), ..., Q_N(o_N, a_N))
where ∂Q_tot/∂Q_i ≥ 0 (monotonic constraint)
```

**Reference:** Rashid et al., "QMIX: Monotonic Value Function Factorisation for
Deep Multi-Agent Reinforcement Learning" (ICML 2018)

**Caveat:** The monotonic constraint can theoretically limit complex coordination,
but for coverage path planning, it's usually sufficient.

### 🥉 GNN Enhancement (add to either MAPPO or QMIX)

Replace the MLP policy network with a Graph Neural Network that operates on the
net topology:

```
obs_i → GNN(local neighborhood) → message passing → action
```

**Why it fits your problem perfectly:**
- The net IS a graph — GNN is the natural representation
- GNNs are permutation-invariant (MDUs are interchangeable)
- Message passing = implicit communication between nearby MDUs
- Handles variable graph sizes if you change net configuration

**Reference:** Jiang et al., "Graph Neural Network for Multi-Agent Reinforcement
Learning" (ICLR 2021)

---

## 2. Action Space Design

### Discrete: Move to Neighbor Node (Recommended)

```
A_i = {0, 1, 2, ..., deg(v_i)}  → index of neighbor node to move to
where deg(v_i) = number of adjacent nodes from current position
```

**Handling varying degree:** Pad to max degree with mask (invalid action masking).
PPO/QMIX both support action masking naturally.

**Simpler alternative:** 4-directional on net (if the square net has regular topology):
```
A_i = {up, down, left, right, stay}
```

### Why discrete over continuous:
- MDUs move on discrete graph nodes — discrete is the natural fit
- No need for low-level motion control (motors, speeds)
- Easier to train, fewer samples needed
- Action masking prevents invalid moves (off-net, collisions)

---

## 3. State / Observation Space Design

### Per-MDU Observation (local, partial)

```
obs_i = [
    node_embedding_i,        # current node index or coordinates (3)
    neighbor_mask,           # which neighbors exist (max_deg binary)
    coverage_mask_local,     # local surface coverage status (M values)
    relative_pos_other_MDUs, # positions of other MDUs relative to self (3*(N-1))
    time_remaining,          # normalized time step remaining (1)
    cone_fov_params,         # cone angle, range (2, if variable)
]
```

### Global State (for centralized critic only)

```
state = [
    all_MDU_positions,       # (3*N)
    global_coverage_mask,    # total surface coverage (binary or float)
    coverage_rate,           # % covered so far (1)
    time_step,               # (1)
]
```

### Graph Embedding (if using GNN)

Each node feature:
```
node_feature[v] = [
    is_MDU_present,       # binary (1)
    MDU_id_if_present,    # one-hot or -1 (1 or N)
    node_coordinates,     # 3D position (3)
    is_visited,           # has an MDU been here? (1)
    local_coverage,       # coverage ratio of nearby surface (1)
]
```

Edge features:
```
edge_feature[u,v] = [
    edge_length,          # Euclidean distance (1)
]
```

---

## 4. Reward Function Design

### Core Reward Components

```python
def compute_reward(mdus, surface_mesh, coverage_mask, step):
    """
    mdus: list of MDU positions (nodes on net)
    surface_mesh: asteroid surface triangulation
    coverage_mask: which faces have been covered (boolean array)
    """
    R = 0.0
    
    # 1. NEWLY COVERED SURFACE (primary driver)
    new_coverage = compute_newly_covered_faces(mdus, surface_mesh, coverage_mask)
    R += w1 * new_coverage          # w1 ~ 1.0 (main reward)
    
    # 2. COVERAGE COMPLETION BONUS
    if coverage_mask.all():
        R += w2 * (1.0 / max(step, 1))  # w2 ~ 100 (reward faster completion)
        # Alternative: R += w2 * (max_steps - step) / max_steps
    
    # 3. STEP PENALTY (encourage efficiency)
    R -= w3                           # w3 ~ 0.01 (small, per step)
    
    # 4. OVERLAP PENALTY (discourage redundancy)
    overlap = compute_overlap_faces(mdus, surface_mesh)
    R -= w4 * overlap                 # w4 ~ 0.1
    
    # 5. PROXIMITY BONUS (spread MDUs out)
    min_dist = min_pairwise_distance(mdus)
    R += w5 * sigmoid(min_dist - threshold)  # w5 ~ 0.05
    
    # 6. BOUNDARY PENALTY (stay on net)
    if any_mdu_off_net(mdus):
        R -= w6                       # w6 ~ 10 (big penalty)
    
    return R
```

### Recommended Weights (starting point)

| Component | Symbol | Value | Sensitivity |
|-----------|--------|-------|-------------|
| New coverage | w1 | 1.0 | Primary signal |
| Completion bonus | w2 | 100 | High — triggers at end |
| Step penalty | w3 | 0.01 | Low — prevents infinite loops |
| Overlap penalty | w4 | 0.1 | Medium — discourages clustering |
| Spread bonus | w5 | 0.05 | Low — encourages exploration |
| Off-net penalty | w6 | 10 | High — hard constraint |

### Potential-Based Reward Shaping

To help exploration in sparse coverage environments, add a potential-based shaping
term (preserves optimal policy):

```python
Φ(s) = α * coverage_rate(s)  # potential = fraction covered
R_shaped = R + γ * Φ(s') - Φ(s)  # add to each step reward
```

### Curriculum Learning

Start simple, then increase difficulty:

| Stage | MDUs | Net size | Asteroid | Cone FOV | Max steps |
|-------|------|----------|----------|----------|-----------|
| 1 | 1 | Small (particle_earth) | Sphere | 120° wide | 50 |
| 2 | 2 | Small | Sphere | 90° | 100 |
| 3 | 4 | Medium (square) | Bennu | 60° | 200 |
| 4 | 4 | Full | Bennu realistic | 30° tight | 500 |

---

## 5. Communication Architecture

### Default: CTDE (no explicit communication)

```
Training:   Centralized critic sees ALL positions + coverage
Execution:  Each MDU runs its own policy from LOCAL observation
```

**Why it works:**
- During training, the centralized critic learns to coordinate
- During execution, each MDU implicitly coordinates through the shared policy
- The policy learns: "if another MDU is nearby, move away" from past training experience

### Optional: Graph Message Passing

If MDUs can communicate along net edges:

```
h_i^(t+1) = σ(W·h_i^(t) + Σ_{j∈N(i)} Φ(h_i^(t), h_j^(t)))
```

Simple = concatenate neighbor embeddings + MLP. No attention needed.

---

## 6. Implementation Roadmap

```
Phase 1 — Single-Agent Baseline (MAPPO, 1 MDU)
  ├── Simple net (FNS_particle_earth: 3 nodes, 3 edges)
  ├── Random asteroid (sphere approximation)
  ├── Goal: get training loop working
  └── Expected: 30 min to converge

Phase 2 — Multi-Agent (MAPPO, 2-4 MDUs)
  ├── Square net (FNS_square: 369 nodes)
  ├── Bennu asteroid mesh
  ├── Add action masking + coverage computation
  ├── Goal: coordination emerges
  └── Expected: 2-4 hours training

Phase 3 — GNN Enhancement
  ├── Replace MLP with GNN encoder
  ├── Add message passing between MDUs
  ├── Goal: better generalization to unseen net states
  └── Expected: marginal improvement over baseline

Phase 4 — Ablation + Evaluation
  ├── Compare: MAPPO vs QMIX vs GNN-MAPPO
  ├── Test on Bennu, Didymos, Gelovka
  ├── Metric: coverage % vs time steps
  └── Publish results for IAA 2026
```

---

## 7. Key Papers (No Web Link Needed)

| Paper | Year | Key Idea |
|-------|------|----------|
| PPO (Schulman et al.) | 2017 | Clipped surrogate objective, stable PG |
| MAPPO (Yu et al.) | 2022 | PPO + CTDE for cooperative MARL |
| QMIX (Rashid et al.) | 2018 | Monotonic value decomposition |
| GNN-MARL (Jiang et al.) | 2021 | Graph nets for agent coordination |
| Parameter Sharing (Gupta et al.) | 2017 | All agents share policy weights |
| Action Masking (Huang et al.) | 2022 | Mask invalid actions in discrete spaces |
| Potential-Based Shaping (Ng et al.) | 1999 | Safe reward shaping preserves optimality |
| Coverage Path Planning (Galceran et al.) | 2013 | Survey of classical CPP approaches |
