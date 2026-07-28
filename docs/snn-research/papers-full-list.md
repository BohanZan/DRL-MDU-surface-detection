# SNN Research Papers — Full List with Details

> Collected via arXiv and Semantic Scholar, 2026-06-10. Organized by relevance.

---

## Tier 1: Directly Relevant (SNN + Planning/Control)

| # | arXiv ID | Year | First Author | Title | Venue | Why Relevant |
|---|----------|------|-------------|-------|-------|-------------|
| 1 | 2404.15524 | 2024 | Espino | SNN Wavefront Planner for Mobile Robots | cs.RO | **Most similar** — SNN path planning with E-prop |
| 2 | 2010.09635 | 2020 | Tang | Deep RL with Population-Coded SNN for Continuous Control | cs.RO | SNN + DRL for robot control |
| 3 | 2310.02361 | 2023 | Wang | Event-Enhanced Multi-Modal SNN for Dynamic Obstacle Avoidance | cs.RO | SNN + DRL for obstacle avoidance |
| 4 | 2501.17172 | 2025 | Casanueva-Morato | SNN trajectory interpolation for closed-loop control | cs.RO | SNN trajectory generation |
| 5 | 2404.15524 | 2024 | Espino | Rapid Adapting SNN Path Planning | cs.RO | Continual learning for path planning |

### Details

**#1 [2404.15524] Espino, Bain, Krichmar — "A Rapid Adapting and Continual Learning SNN Path Planning Algorithm"**
- Method: SNN Wavefront Planner + E-prop learning
- Key: Concurrent mapping and path planning with spiking activity
- Learning: E-prop (online, eligibility-trace-based) for SNN
- Relevance: HIGH — directly addresses SNN for graph-based path planning
- Link: https://arxiv.org/abs/2404.15524

**#2 [2010.09635] Tang, Kumar, Yoo, Michmizos — "Deep RL with Population-Coded SNN for Continuous Control"**
- Method: Population coding of continuous values into spike trains
- Key: First to show SNN + DRL for continuous robot control tasks
- Learning: Surrogate gradient + PPO compatible
- Architecture: Population-coded SNN policy network
- Relevance: HIGH — shows SNN + PPO pipeline works
- Link: https://arxiv.org/abs/2010.09635

**#3 [2310.02361] Wang, Dong, Zhang, Zhou — "Event-Enhanced Multi-Modal SNN for Dynamic Obstacle Avoidance"**
- Method: Event camera + SNN + DRL for obstacle avoidance
- Key: Multi-modal fusion (event + frame) with spiking neurons
- Learning: Surrogate gradient with temporal backprop
- Relevance: HIGH — event-based SNN + DRL navigation
- Link: https://arxiv.org/abs/2310.02361

---

## Tier 2: SNN Training & Theory

| # | arXiv ID | Year | Title | Why Important |
|---|----------|------|-------|---------------|
| 6 | 2406.19645 | 2024 | Directly Training Temporal SNN with Sparse Surrogate Gradient | Training method |
| 7 | 2201.10879 | 2022 | S3NN: Time Step Reduction of Surrogate Gradients | Encoding efficiency |
| 8 | 2603.13478 | 2026 | Dold — One spike vs multiple spikes | Theoretical foundation |
| 9 | 2504.14015 | 2025 | Dold — Causal pieces for SNN | Debugging/analysis |
| 10 | 2006.09985 | 2020 | SNN gesture recognition on Loihi | Neuromorphic deployment |

### Details

**#6 [2406.19645] Li, Zhao, Zhao, Zeng — "Directly Training Temporal SNN with Sparse Surrogate Gradient"**
- Method: Sparse surrogate gradient that only activates where needed
- Key: Reduces training computation without accuracy loss
- Relevance: HIGH — directly applicable to our PPO training
- Link: https://arxiv.org/abs/2406.19645

**#7 [2201.10879] Suetake, Ikegawa, Saiin, Sawada — "S3NN: Time Step Reduction"**
- Method: Reduces encoding time steps using surrogate gradient optimization
- Key: Can reduce from T_enc=5 to T_enc=1-2 with minimal accuracy loss
- Relevance: HIGH — directly impacts our platform speed
- Link: https://arxiv.org/abs/2201.10879

---

## Tier 3: Neuron Models & Architecture Variants

| # | arXiv ID | Year | Title | Improvement |
|---|----------|------|-------|-------------|
| 11 | 2402.04663 | 2024 | CLIF: Complementary LIF Neuron | Better temporal dynamics |
| 12 | 2210.13768 | 2022 | GLIF: Gated LIF Neuron | More expressive |
| 13 | 2506.14138 | 2025 | NeuroCoreX: FPGA SNN Emulator | Hardware deployment |
| 14 | 2501.02916 | 2025 | Courtois — SNN 6D Pose for Space | Space application |

### Details

**#11 [2402.04663] Huang, Lin, Ren, Fu — "CLIF: Complementary LIF Neuron"**
- Method: Dual-pathway LIF (excitatory + inhibitory)
- Key: Better temporal feature extraction than standard LIF
- Link: https://arxiv.org/abs/2402.04663

**#12 [2210.13768] Yao, Li, Mo, Cheng — "GLIF: Gated LIF Neuron"**
- Method: Gating mechanism for LIF (like GRU vs RNN)
- Key: More expressive than LIF with minimal computational overhead
- Link: https://arxiv.org/abs/2210.13768

---

## Tier 4: Space & Neuromorphic Hardware

| # | arXiv ID | Year | Title | Focus |
|---|----------|------|-------|-------|
| 15 | 2501.02916 | 2025 | SNN 6D pose for space (Courtois) | Space SNN |
| 16 | 2506.14138 | 2025 | NeuroCoreX (Gautam) | FPGA SNN |
| 17 | 2004.14942 | 2020 | Memristors to SNN to Neuromorphic | Survey |
| 18 | 2304.04640 | 2023 | NeuroBench (Yik) | Benchmarking |

---

## Tier 5: Multi-Agent & Graph

| # | arXiv ID | Year | Title | Note |
|---|----------|------|-------|------|
| 19 | 2509.05397 | 2025 | RoboBallet: GNN + RL for multi-robot | GNN not SNN, but relevant |
| 20 | 2203.08975 | 2022 | Survey of MADRL with Communication | Background |
| 21 | 2006.15482 | 2020 | Robot Inner Attention for multi-robot | Attention for coordination |

---

## Paper Count: 21 papers across 5 tiers
