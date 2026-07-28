# SNN for Spatial Trajectory Planning — Literature Collection

> Collected 2026-06-10 via arXiv API. For the DRL surface detection project.

---

## 1. Core Papers Found

### 1.1 Most Relevant: SNN Path Planning

**[2404.15524] Espino, Bain, Krichmar — "A Rapid Adapting and Continual Learning Spiking Neural Network Path Planning Algorithm for Mobile Robots"** (2024)
- **Venue:** arXiv:2404.15524 (cs.RO)
- **Summary:** Uses a Spiking Neural Network Wavefront Planner + E-prop learning for concurrent mapping and path planning in large, complex environments. The SNN learns traversal costs and plans paths based on spiking activity.
- **Relevance:** HIGH — Directly addresses SNN for robot path planning. Uses E-prop (online learning rule for SNNs). Similar domain to our coverage problem.
- **Link:** https://arxiv.org/abs/2404.15524

### 1.2 SNN Trajectory Control

**[2501.17172] Casanueva-Morato, Wu, Indiveri et al. — "Towards spiking analog hardware implementation of a trajectory interpolation mechanism for smooth closed-loop control"** (2025)
- **Venue:** arXiv:2501.17172 (cs.NE, cs.RO)
- **Summary:** Closed-loop neuromorphic control for a robotic arm. Uses a shifted Winner-Take-All (WTA) circuit for trajectory interpolation. Implements smooth trajectory generation with spiking neurons.
- **Relevance:** MEDIUM — Trajectory interpolation, not full planning. But shows SNN can do closed-loop control.
- **Link:** https://arxiv.org/abs/2501.17172

### 1.3 Dold's Theoretical SNN Work

**[2603.13478] Dold — "What is better: having neurons spike only once, or multiple times?"** (2026)
- **Venue:** arXiv:2603.13478
- **Key result:** Proves single-spike and multi-spike encoding are computationally equivalent. Multi-spike can be more expressive per neuron but not per layer.
- **Relevance:** Theoretical foundation — informs SNN architecture design.

**[2504.14015] Dold — "Causal pieces: a concept to analyse and improve spiking neural networks"** (2025)
- **Venue:** arXiv:2504.14015
- **Summary:** Introduces a framework for analyzing causal structure in SNNs. Can identify which neurons/spikes contribute to decisions.
- **Relevance:** Could be used for debugging/analyzing SNN policies.

### 1.4 Other Relevant Papers Found

| arXiv ID | Year | Title | Relevance |
|----------|------|-------|-----------|
| 1606.00825 | 2016 | Training HMM with Bayesian SNN | LOW — classification, not control |
| 2506.14138 | 2025 | NeuroCoreX: FPGA SNN Emulator | LOW — hardware, not algorithms |
| 1810.03199 | 2018 | Pre-Synaptic Pool Modification (PSPM) | LOW — supervised SNN learning |
| 1705.09132 | 2017 | First-spike visual categorization with R-STDP | LOW — vision, not planning |
| 2004.14942 | 2020 | Memristors: from IMC to SNN to Neuromorphic | LOW — hardware survey |

### 1.5 SPAICE Conference (ESA/IAA AI in Space)

**SPAICE 2024** — Proceedings: https://doi.org/10.5281/zenodo.13889941
- 116 MB PDF, covers AI for space (ML, RL, SNN, planning)
- Keywords: AI, ESA, IAA, Space Exploration, ML
- Dold (dodo47.github.io) is deeply involved in SPAICE community

**SPAICE 2025** — Chaired by Dold + IAA
**SPAICE 2026** — ESTEC, call for papers open

### 1.6 SNN for Space Applications

**[2501.02916] Courtois, Miramond, Pegatoquet — "Spiking monocular event based 6D pose estimation for space application"** (2025)
- **Venue:** arXiv:2501.02916 (cs.CV, cs.LG)
- **Summary:** Investigates fully event-based spacecraft pose estimation using SNN + event cameras. Targets on-orbit servicing and debris removal. Shows SNN is viable for space perception tasks.
- **Relevance:** HIGH — Directly applies SNN to space domain. Companion to our SNN control work.
- **Link:** https://arxiv.org/abs/2501.02916

### 1.6 Dold's Other Projects

- **SpOC 3.0 (Space Optimization Competition):** https://github.com/esa/SpOC3 — "Programmable Cubes" challenge
- **Reprogrammable lattice structures for space:** arXiv:2411.15266

---

## 2. Additional High-Relevance Papers (from 2nd search pass)

### 2.1 SNN + DRL for Robot Control

**[2010.09635] Tang, Kumar, Yoo, Michmizos — "Deep Reinforcement Learning with Population-Coded Spiking Neural Network for Continuous Control"** (2020)
- **Venue:** arXiv:2010.09635 (cs.RO, cs.NE)
- **Method:** Population coding maps continuous values to spike trains. SNN policy trained with PPO + surrogate gradient. First demonstration of SNN+DRL for continuous robot control.
- **Key result:** SNN policy achieves comparable performance to ANN policy at ~100x lower estimated energy cost.
- **Relevance:** HIGH — validates SNN+PPO pipeline viability.
- **Link:** https://arxiv.org/abs/2010.09635

**[2310.02361] Wang, Dong, Zhang, Zhou — "Event-Enhanced Multi-Modal Spiking Neural Network for Dynamic Obstacle Avoidance"** (2023)
- **Venue:** arXiv:2310.02361 (cs.RO)
- **Method:** Event camera + traditional frame fusion with SNN + DRL for real-time obstacle avoidance.
- **Key result:** SNN achieves comparable obstacle avoidance to ANN with much lower computation.
- **Relevance:** HIGH — SNN + DRL navigation in dynamic environments.
- **Link:** https://arxiv.org/abs/2310.02361

### 2.2 Training Methods

**[2406.19645] Li, Zhao, Zhao, Zeng — "Directly Training Temporal Spiking Neural Network with Sparse Surrogate Gradient"** (2024)
- **Method:** Sparse surrogate gradient that only activates where needed during backprop.
- **Key result:** Reduces training FLOPs by 40-60% without accuracy loss.
- **Link:** https://arxiv.org/abs/2406.19645

**[2201.10879] Suetake et al. — "S3NN: Time Step Reduction of Spiking Surrogate Gradients"** (2022)
- **Method:** Optimizes encoding time steps via surrogate gradient.
- **Key result:** Reduces T_enc from 5 to 1-2 with minimal accuracy loss.
- **Relevance:** HIGH — directly impacts inference speed on neuromorphic hardware.
- **Link:** https://arxiv.org/abs/2201.10879

### 2.3 Neuron Model Variants

**[2402.04663] Huang et al. — "CLIF: Complementary Leaky Integrate-and-Fire Neuron"** (2024)
- **Method:** Dual-pathway LIF (excitatory + inhibitory).
- **Key result:** Better temporal feature extraction than standard LIF.
- **Link:** https://arxiv.org/abs/2402.04663

**[2210.13768] Yao et al. — "GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron"** (2022)
- **Method:** Adds gating mechanism to LIF (analogous to GRU vs RNN).
- **Key result:** More expressive than LIF with minimal overhead.
- **Link:** https://arxiv.org/abs/2210.13768

---

## 3. Existing SNN Libraries for PyTorch

### snnTorch (Recommended)
- **PyPI:** `snntorch`
- **Homepage:** https://snntorch.readthedocs.io/
- **Features:**
  - Surrogate gradient descent (compatible with standard PyTorch optimizer)
  - Pre-built neuron models: LIF, R-LIF, synaptic, etc.
  - Works with torch.nn, torch.optim — minimal changes to existing code
  - Supports GPU acceleration
  - Includes spike encoding/decoding utilities
- **Why for our project:** Could replace our current GRU cell with an SNN-based recurrent layer, trained with the same PPO pipeline (surrogate gradients enable backprop).
- **snnTorch GitHub:** https://github.com/jeshraghian/snntorch

### PySNN
- **Status:** Less actively maintained than snnTorch
- **Features:** Similar surrogate gradient approach
- **Recommendation:** Use snnTorch instead

---

## 3. How to Implement SNN for Temporal Trajectory Planning in PyTorch

Using snnTorch, the key idea is to **replace the GRU memory cell with a spiking recurrent layer (e.g., R-LIF)**. The SNN's time-step dynamics naturally encode temporal information.

### Proposed Architecture

```
Current (ANN + GRU):                    Proposed (SNN):
                                        ┌──────────────────────┐
obs → body_encoder → GRU → scorer       │ obs → enc → LIF → scorer │
    (64 hidden)  (64 dim)               │     spikes   mem    │
                                        └──────────────────────┘
```

### Key Changes from Current Code

1. **Replace GRUCell with snnTorch Leaky-Integrate-and-Fire (LIF) neuron**
2. **The LIF neuron's membrane potential replaces the GRU hidden state**
3. **Spike output provides event-based communication**
4. **Surrogate gradient (e.g., `slayer` or `straight-through estimator`) enables backprop**

### Implementation Sketch

```python
import snntorch as snn
import torch.nn as nn

class SNNActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        # Body encoder
        self.body_net = nn.Sequential(
            nn.Linear(body_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        
        # Spiking recurrent layer (replaces GRU)
        # LIF: Leaky Integrate-and-Fire neuron
        # R-LIF: Recurrent LIF (has feedback connections)
        self.snn_layer = snn.RLeaky(
            input_size=hidden + act_dim + 1,  # same input as current GRU
            hidden_size=hidden,
            spike_grad=snn.surrogate.fast_sigmoid(slope=25),
            beta=0.5,  # membrane time constant
            learn_beta=True,
        )
        
        # Candidate encoder + scorer (same as current)
        self.cand_net = nn.Sequential(...)
        self.score_net = nn.Sequential(...)
    
    def forward(self, obs, mask=None, hidden=None, ...):
        # body encoding
        body_feats = self.body_net(body)  # (B, H)
        gru_input = torch.cat([body_feats, prev_actions, coverage_rate], dim=-1)
        
        # SNN forward: spiking_output, membrane_potential
        # hid = (mem, spk) for LIF neurons
        spk, mem = self.snn_layer(gru_input, hidden)
        
        # Use membrane potential (mem) as "hidden state" for candidate scoring
        ...
```

### Training Adaptations

| Component | Current (GRU) | Proposed (SNN) |
|-----------|--------------|----------------|
| **Memory cell** | GRUCell(69→64) | RLIF(69→64) or LIF |
| **Backprop** | Truncated BPTT (16 steps) | Truncated BPTT + surrogate gradient |
| **PPO update** | Sequential with retain_graph | Same, but snnTorch handles gradients |
| **Hidden state** | Continuous (64-dim) | Membrane potential + spikes |
| **Parameters** | ~38K | ~39K (similar) |
| **Speed** | ~17s/ep | Likely slower (SNN needs more time steps) |

### Open Challenges

1. **Speed:** SNNs need multiple time steps to propagate information. Our 200-step episodes × SNN internal steps could be slow.
2. **Training stability:** Surrogate gradients are approximate — PPO might be less stable.
3. **Advantage over GRU:** SNN's main advantage (energy efficiency) requires neuromorphic hardware. On GPU, GRU is faster.
4. **Maturity:** snnTorch + PPO hasn't been well-tested in the literature.

### Recommended Path

1. **Test snnTorch:** Run the snnTorch MNIST example to verify installation and surrogate gradient behavior
2. **Replace GRU → LIF:** Minimal change, test with 10 episodes
3. **Compare:** GRU vs SNN on coverage, speed, convergence
4. **Deploy:** Only worth it if we target neuromorphic hardware (Loihi, etc.)

---

## 4. Conclusions

| Question | Answer |
|----------|--------|
| Is there existing SNN path planning work? | **Yes** — Espino et al. 2024 (2404.15524) is the closest. Casanueva-Morato 2025 for control. |
| Is there a ready-to-use project? | **Not directly.** snnTorch is a general SNN library, but no ready-made SNN+PPO coverage planner exists. |
| Can we implement it? | **Yes** — Replace GRU with snnTorch LIF/R-LIF neuron. Same PPO pipeline, surrogate gradients. |
| Should we? | **Probably not yet** — GRU achieves 88.04% with 17s/ep. SNN would be slower on GPU with no clear advantage unless targeting neuromorphic hardware. |

### Papers to Read

1. **Espino et al. 2024** — [arXiv:2404.15524](https://arxiv.org/abs/2404.15524) — SNN path planning ← **Start here**
2. **Dold 2026** — [arXiv:2603.13478](https://arxiv.org/abs/2603.13478) — Spike equivalence theory
3. **Casanueva-Morato 2025** — [arXiv:2501.17172](https://arxiv.org/abs/2501.17172) — SNN trajectory control
4. **snnTorch docs** — [https://snntorch.readthedocs.io/](https://snntorch.readthedocs.io/)
