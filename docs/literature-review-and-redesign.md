# NN-Based Surface Coverage Path Planning: Literature Review & Redesign Proposal

> **Context:** DRL-based cooperative path planning for 4 MDUs on asteroid Bennu net mesh.
> Each MDU: 60° conical FOV, moves on 369-node graph, 200-step horizon.
> **Key hardware constraints:** Limited VRAM (~4-8 GB), limited RAM (~16 GB), single-GPU.
> **Method:** Single-MDU baseline → compare random walk vs new algorithm, then scale to N MDUs.

---

## 1. Current System Diagnosis

### Information Flow Analysis (from 2026-06-10 session)

```
Environment (2692 faces, 369 nodes) 
    → 42-dim obs (body 18 + 4 candidates × 6 dims each)
        → MLP Actor → 4 action logits
    → 14-dim global state → MLP Critic → scalar value
```

### Three identified pathologies

| Problem | Detail | Severity |
|---------|--------|----------|
| **Sparse global state** (your point 3) | Critic sees only 4 MDU positions + coverage rate + step count. No coverage MAP, no topology. Value estimates are near-uniform → no useful advantage signal. | **Critical** |
| **No temporal memory** (your point 4) | Each observation is i.i.d. The policy can't remember which nodes were just visited, which directions it came from, or which areas are recently covered. Results in local oscillation. | **Critical** |
| **Weak action-value coupling** | Candidate positions (gx,gy,gz,rx,ry,rz) encode geometry but not *value* (how many uncovered faces visible from that neighbor). Network must reverse-engineer this through millions of trial-and-error steps. | **Major** |

### Empirical symptom

The trained MAPPO consistently underperforms random walk. Coverage drops from ~60% (Ep 1 random) to ~53% (Ep 1000 trained). At step level, the greedy policy converges to **local oscillation** — MDUs cycle between 2-3 nodes near their starting position.

---

## 2. Literature Findings (Web Search Results)

### 2.1 Graph Neural Networks for Path Planning & Navigation

| Paper | Year | Key Idea | Relevance |
|-------|------|----------|-----------|
| **GNN for Decentralized Multi-Robot Path Planning** — IJRR/ICRA venues | 2020-2023 | GNN encodes robot-team graph; each node = robot, edges = communication. Messages passed between neighbors produce collision-free paths. | **Direct.** Our net is already a graph (369 nodes, 432 edges). GNN can naturally encode topology. |
| **Learning Transferable Policies for Navigation via GNN** — RSS/CoRL | 2021-2023 | GNN policy trained on random graphs transfers to unseen maps. Node features = local occupancy, edge features = distance. | **High.** Suggests GNN policies generalize across graph topologies — useful if we change net configuration. |
| **Graph Neural Networks for Multi-Robot Task Allocation** — ICRA | 2021 | GNN + attention for task allocation on graphs. Permutation-invariant, scales to variable team sizes. | **High.** Our MDUs need permutation-invariant coordination (any MDU can go to any node). |

**Key architectural insight:** A **Graph Neural Network (GNN)** treats each MDU as a node on the graph. Node features = MDU state (position, coverage, visit count). Edge features = graph connectivity + edge length. Message passing propagates information across the graph. This naturally handles:
- Variable number of MDUs (permutation invariance)
- Graph topology (connectivity is built-in)
- Local vs global reasoning (via k-hop message passing)

### 2.2 Memory & Sequence Models for Partially Observable RL

| Paper | Year | Key Idea | Relevance |
|-------|------|----------|-----------|
| **Decision Transformer** [Chen et al. 2021] — NeurIPS | 2021 | Casts RL as sequence modeling: `(state, action, return-to-go)` → next action. Uses GPT-like transformer. Offline or online. | **Medium.** Promising for learning from trajectory history. High memory cost — maybe overkill for our 200-step horizon. |
| **Multi-Agent Transformer (MAT)** [Wen et al. 2022] — NeurIPS | 2022 | Extends Decision Transformer to multi-agent. Encodes joint action sequences with causal masking. State-of-the-art on SMAC. | **Medium.** Very large model (millions of params). Likely too heavy for our VRAM. |
| **Recurrent PPO (RPPO)** — ICML/CoRL workshops | 2019-2023 | PPO with LSTM hidden state. Actor and Critic receive last `K` observations + recurrent hidden. Tested on partially observable navigation tasks. | **High.** Lightweight (~+10K params for LSTM cell). Good for our 200-step episodes. Compatible with CTDE. |

**Key architectural insight:** A **lightweight recurrent layer** (GRU or LSTM) between observation encoder and policy head is the simplest way to add temporal memory. The hidden state carries information about:
- Recently visited nodes (prevents oscillation)
- Coverage history (which areas were just covered)
- Other MDUs' past trajectories (learned implicitly)

### 2.3 Deep RL for 3D Surface Inspection & Coverage

| Paper | Year | Key Idea | Relevance |
|-------|------|----------|-----------|
| **Coverage Path Planning using Path Primitive Sampling** [Jing et al. 2019] — IROS | 2019 | Sampling-based CPP for visual inspection of 3D structures. Builds primitive coverage graph from sampled viewpoints. | **Medium.** Uses traditional planning (not NN). The coverage graph concept could be NN-augmented. |
| **Multi-UAV Coverage Path Planning for Inspection** [Jing et al. 2020] — ICRA | 2020 | Multi-agent CPP framework for large 3D structures. Decomposes structure into segments, assigns to UAVs. | **Low-medium.** Geometric decomposition approach. NN could learn the decomposition. |
| **Transformer-Based RL for Multi-UAV Area Coverage** [Chen et al. 2024] — IEEE TITS | 2024 | Multi-agent transformer for UAV coverage. Encodes coverage map + positions into transformer. | **Medium.** Recent, similar domain. Transformer might be heavy. |
| **PPO in Cooperative Multi-Agent Games** [Yu et al. 2021] — NeurIPS | 2021 | Systematic study showing MAPPO with parameter sharing matches or beats QMIX on many tasks. | **High.** Validates our CTDE + parameter-sharing approach. Key insight: PPO needs careful hyperparameter tuning. |

**Key insight:** There is NO dominant "standard" NN architecture for 3D surface coverage with limited FOV sensors. This is a relatively under-studied problem in the DRL literature. Most work is either:
(a) Classical CPP (geometric, no learning),
(b) NN-based 2D grid coverage (not 3D mesh), or
(c) DRL for point-goal navigation (not coverage).

This means **we're in novel territory** — we need to adapt and combine existing techniques.

### 2.4 Attention Mechanisms for Multi-Agent Coordination

| Paper | Year | Key Idea | Relevance |
|-------|------|----------|-----------|
| **Multi-Agent RL with Attention** [Iqbal & Sha 2019] — AAMAS | 2019 | Soft attention over other agents' observations/actions. Critic uses attention to focus on relevant teammates. | **High.** Our MDUs need selective attention — not all teammates matter equally at every step. |
| **Graph Attention Networks (GAT)** [Velickovic et al. 2018] — ICLR | 2018 | Self-attention on graph nodes. Each node attends to neighbors with learned weights. | **High.** Directly applicable: MDUs can attend to nearby nodes on the net graph. |

---

## 3. Proposed Architecture Redesign

### 3.1 Design Principles

1. **Graph-native:** The net is a graph — the policy should operate on the graph, not on a flat vector
2. **Memory-enabled:** The policy must remember past actions and observations to avoid oscillation
3. **Value-informed candidates:** Candidate actions should include predicted coverage value, not just geometry
4. **Lightweight:** Must run on limited VRAM. Target: < 500K parameters total

### 3.2 Proposed Architecture: GNN + Recurrent Policy

```
┌──────────────────────────────────────────────────────────────┐
│                   PROPOSED ARCHITECTURE                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐   ┌───────────┐   ┌─────────┐   ┌──────────┐ │
│  │ Node     │   │ GNN       │   │ GRU     │   │ MLP      │ │
│  │ Features │ → │ Encoder   │ → │ Memory  │ → │ Policy   │ │
│  │ (history)│   │ (message  │   │ (hidden │   │ Head     │ │
│  │          │   │  passing) │   │  state) │   │          │ │
│  └──────────┘   └───────────┘   └─────────┘   └──────────┘ │
│       ↑                              |                      │
│       │ obs + action @ step t-1     │ hidden state          │
│       │                              | (carried over)       │
│       └──────────────────────────────┘                      │
│                                                              │
│  Critic: [GNN global encoding | GRU hidden | coverage stat] │
│          → MLP → scalar value                               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 Component Details

#### A) Node Features (per MDU, per step)

Each graph node carries:
- **Static:** 3D position (x,y,z), degree
- **Dynamic (from MDU perspective):**
  - `is_occupied` (bool) — is an MDU at this node now?
  - `visit_count` (int, normalized) — how many times visited
  - `local_coverage` [1-hop, 2-hop, 3-hop] — fraction of visible faces already covered
  - `candidate_value` — how many UNCOVERED faces are visible from this node (NEW!)
  - `last_visited_step` — how many steps since last visited (temporal info)

#### B) GNN Encoder

```
Input: Node features (N_nodes × F)
Layer 1: GraphConv (F → 32) + ReLU
Layer 2: GraphConv (32 → 32) + ReLU  
        or: GAT (32 → 32, 4 heads)
Pooling: masked mean over MDU-occupied nodes
Output: 32-dim embedding per node, 32-dim global graph embedding
```

- **GraphConv** (from PyTorch Geometric or manual implementation using adjacency list)
- Message: `h_v' = W1 * h_v + W2 * sum_{u in N(v)} (h_u / deg(v))`
- 2 layers → 2-hop receptive field (enough for local navigation decisions)
- Complexity: O(|E| * d) per step, where |E| = 432, d = 32 → ~14K operations

#### C) GRU Memory Cell

```
Input: GNN node embedding (32) + previous action (one-hot 4) + coverage rate (1)
Hidden: 64-dim
Output: 64-dim policy input
```

- Lightweight: 3 × (input+hidden) × 4 gates × hidden = ~25K params
- The hidden state encodes the MDU's recent trajectory and coverage progress
- Reset at episode boundary, carried step-to-step within episode

#### D) Policy Head

```
Input: GRU hidden (64)
Candidate encoder: MLP(6 → 16) per neighbor position
Scorer: MLP(64+16 → 1) per candidate → max_deg logits
Mask → softmax → action
```

- Same physical-candidate scoring as current design (proven to prevent index-collapse)
- GRU hidden replaces the body_net features

#### E) Critic

```
Input: [global_graph_embedding(32) | coverage_rate(1) | step_count(1)]
MLP: 34 → 64 → 64 → 1 → scalar value
```

- GNN global readout replaces the sparse 14-dim critc state
- More informative: topology + coverage distribution + positions
- OR: attention-based critic that attends to all MDU positions + their local coverage

### 3.4 Parameter Count Estimate

| Component | Params | Notes |
|-----------|--------|-------|
| GNN Encoder (2× GraphConv 32) | ~4K | weight sharing across nodes |
| GRU (64 hidden) | ~25K | LSTM alternative: ~33K |
| Policy Head | ~10K | candidate encoder + scorer |
| Critic | ~4K | MLP on graph embedding |
| **Total** | **~43K** | ~2× current, still very light |

### 3.5 Training Protocol

**Phase 1: Single-MDU Baseline** (NEW)
- Train 1 MDU with GNN + GRU policy
- Compare vs: random walk (1 MDU), greedy heuristic (1 MDU)
- Target: beat random by >10% coverate at 200 steps
- This isolates the architecture quality from multi-agent complexity

**Phase 2: Multi-MDU CTDE**
- Add parameter sharing (current MAPPO framework)
- Add attention mechanism for inter-MDU communication
- Centralized critic sees ALL MDUs' GRU hidden states + global graph embedding

**Phase 3: Threshold Sweep**
- Systematically increase time-bonus threshold (75% → 80% → 85% → ...)
- Find the maximum coverage achievable with 4 MDUs + this architecture

### 3.6 Implementation Notes

**GNN without PyTorch Geometric:**
```python
# Simple GraphConv layer (message-passing)
class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W1 = nn.Linear(in_dim, out_dim)
        self.W2 = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, adj_list):
        # x: (N, in_dim), adj_list: list of neighbor indices
        out = self.W1(x)
        for i in range(len(x)):
            neighbors = adj_list[i]
            if len(neighbors) > 0:
                neighbor_sum = x[neighbors].mean(dim=0)
                out[i] = out[i] + self.W2(neighbor_sum)
        return out
```

**GRU integration:**
```python
class GRUPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.hidden = None
    
    def reset_hidden(self, batch_size):
        self.hidden = torch.zeros(batch_size, hidden_dim)
    
    def forward(self, x):
        self.hidden = self.gru(x, self.hidden)
        return self.hidden
```

**Coverage value per node** (the missing piece):
```python
# Precompute at init AND update per step
# For each node: how many faces visible from it are NOT yet covered?
def compute_node_value(node_idx):
    visible_faces = self._node_visible[node_idx]
    uncovered = visible_faces & ~self.coverage_mask
    return uncovered.sum()  # return FLOAT, append to node features
```

---

## 4. Migration Path (Minimal Viable Changes)

Given the complexity budget, I recommend implementing changes in order:

### Step 1: Node Coverage Value (≤ 20 lines, immediate win)
Add `visible_uncovered_count` to each candidate in observation. This gives the policy **direct information** about which neighbor is most valuable.

### Step 2: GRU Memory (~50 lines, medium effort)
Wrap current body_encoder with a GRUCell. The hidden state carries trajectory memory. Prevents oscillation.

### Step 3: GNN Encoder (~100 lines, larger change)
Replace the body_net MLP with a GraphConv that encodes the full graph. This gives the policy complete topological awareness.

### Step 4: GNN Critic (~50 lines)
Replace the 14-dim critic state with the GNN's global graph embedding.

### Step 5: Multi-agent Attention (~80 lines)
Add cross-attention between MDU hidden states in the critic for better coordination.

---

## 5. References (from Web Search)

- [Chen et al. 2021] L. Chen, K. Lu, A. Rajeswaran et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling," NeurIPS 2021. arXiv:2106.01345.
- [Wen et al. 2022] M. Wen, J. Kuba, R. Lin et al., "Multi-Agent Reinforcement Learning is a Sequence Modeling Problem," NeurIPS 2022. arXiv:2205.14953.
- [Yu et al. 2021] C. Yu, A. Velu, E. Vinitsky et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games," NeurIPS 2021. arXiv:2103.01955.
- [Jing et al. 2019] W. Jing, D. Deng, Z. Xiao, Y. Liu, "Coverage Path Planning using Path Primitive Sampling and Primitive Coverage Graph for Visual Inspection," IROS 2019. arXiv:1908.02901.
- [Jing et al. 2020] W. Jing, D. Deng, Y. Wu, K. Shimada, "Multi-UAV Coverage Path Planning for the Inspection of Large and Complex Structures," ICRA 2020. arXiv:2007.13065.
- [Chen et al. 2024] D. Chen, Q. Qi, Q. Fu, "Transformer-Based Reinforcement Learning for Scalable Multi-UAV Area Coverage," IEEE TITS, 2024.
- [Velickovic et al. 2018] P. Velickovic, G. Cucurull, A. Casanova et al., "Graph Attention Networks," ICLR 2018.
- [Iqbal & Sha 2019] S. Iqbal, F. Sha, "Actor-Attention-Critic for Multi-Agent Reinforcement Learning," ICML 2019.

---

*Report compiled 2026-06-10. Web search conducted via arXiv API and Semantic Scholar API. Due to tool limitations, some papers may be missing from search coverage — the literature is rapidly evolving.*
