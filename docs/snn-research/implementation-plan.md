# SNN-Based Temporal Trajectory Planning — Implementation Plan

> **Assumption:** SNN will work and outperform ANN+GRU on the target neuromorphic platform.
> **Target platform:** Space-grade neuromorphic hardware (Loihi-class, FPGA-based, or equivalent).
> **Ground simulation:** PyTorch + snnTorch with surrogate gradients.

---

## 1. Overall Strategy

### Why SNN for This Problem

| Factor | ANN+GRU (current) | SNN (proposed) |
|--------|-------------------|----------------|
| **Temporal encoding** | Hidden state vector | Spike timing + membrane potential |
| **Energy efficiency** | ~10-30 W (GPU) | ~10-100 mW (neuromorphic) |
| **Event-driven** | No — always computes | Yes — compute only on input |
| **On-chip learning** | Limited | Possible (STDP, E-prop) |
| **Radiation tolerance** | Standard | Potentially better (spike-based) |
| **Speed on GPU** | 17s/ep | Slower (needs time-step simulation) |
| **Speed on neuromorphic** | N/A | Real-time (native parallelism) |

### Implementation Philosophy

**Train on GPU (simulate), Deploy on Neuromorphic (infer).**

1. Train the SNN policy on GPU using snnTorch surrogate gradients + PPO
2. Convert/simplify for neuromorphic deployment (quantize weights, remove surrogate)
3. On neuromorphic: pure inference + on-chip adaptation (STDP/E-prop)

---

## 2. Architecture Design

### 2.1 Overall Network

```
Observation(46,)
    │
    ▼
┌─────────────────────────────────────────┐
│  Spike Encoder                          │
│  Converts continuous obs → spike trains │
│  Rate coding or phase coding            │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Body Network (SNN)                     │
│  LIF neurons, fully connected           │
│  Input: encoded obs(46) → hidden(64)    │
│  Temporal: membrane potential carries   │
│  info across time steps                 │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Candidate Encoder + Scorer (ANNU)      │
│  Same as current — scores neighbor      │
│  positions using membrane potential     │
│  as "body features"                     │
└─────────────────────────────────────────┘
    │
    ▼
  Action (4 discrete)
```

### 2.2 Spike Encoding

The continuous observation must be encoded into spike trains:

**Method: Population Rate Coding**
- Each observation dimension → group of neurons
- Value magnitude → firing rate (Poisson process)
- Time window: T_enc steps per environment step

```python
def rate_encode(obs, T_enc=10):
    """Encode continuous obs into spike trains over T_enc time steps."""
    spikes = []
    for t in range(T_enc):
        # Poisson: probability of firing = clipped value
        probs = torch.clamp(obs, 0, 1)
        spike = torch.bernoulli(probs)  # (B, obs_dim)
        spikes.append(spike)
    return torch.stack(spikes, dim=0)  # (T_enc, B, obs_dim)
```

**Alternative: Phase Coding**
- Spike phase relative to a background oscillation encodes value
- More energy-efficient (one spike per dimension per step)

### 2.3 SNN Body Network

Replace the current GRU with a **Leaky Integrate-and-Fire (LIF) neuron layer**:

```python
import snntorch as snn
import snntorch.surrogate as surrogate

class SNNBody(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden)
        # LIF neuron with surrogate gradient
        self.lif = snn.Leaky(
            beta=0.5,           # membrane time constant
            learn_beta=True,     # learn decay rate
            spike_grad=surrogate.fast_sigmoid(slope=25),
        )
        self.hidden = hidden

    def forward(self, x, mem=None):
        # x: (B, input_dim) at one time step
        # mem: (B, hidden) previous membrane potential
        if mem is None:
            mem = torch.zeros(x.size(0), self.hidden, device=x.device)
        cur = self.fc(x)        # (B, hidden) — input current
        spk, mem = self.lif(cur, mem)  # spike + membrane
        return spk, mem
```

### 2.4 Complete SNNActor

```python
class SNNActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64, T_enc=5):
        super().__init__()
        self.max_deg = act_dim
        self.hidden_dim = hidden
        self.T_enc = T_enc  # encoding time steps per env step
        cand_feat_dim = 7
        candidate_dim = self.max_deg * cand_feat_dim
        body_dim = obs_dim - candidate_dim

        # Spike encoder
        self.enc_fc = nn.Linear(body_dim, hidden)

        # SNN body (LIF neurons)
        self.lif = snn.Leaky(
            beta=0.5, learn_beta=True,
            spike_grad=surrogate.fast_sigmoid(slope=25),
        )

        # Candidate encoder (same as current — ANN, no spikes needed)
        self.cand_net = nn.Sequential(
            nn.Linear(cand_feat_dim, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, hidden // 2), nn.Tanh(),
        )

        # Scorer: membrane potential + candidate features → logit
        self.score_net = nn.Sequential(
            nn.Linear(hidden + hidden // 2, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, mask=None, hidden=None,
                prev_actions=None, coverage_rate=None):
        B = obs.shape[0]
        cand_feat_dim = 7
        body = obs[:, :-self.max_deg * cand_feat_dim]
        candidates = obs[:, -self.max_deg * cand_feat_dim:].reshape(
            B, self.max_deg, cand_feat_dim)

        # Spike encoding over T_enc steps
        if hidden is None:
            mem = torch.zeros(B, self.hidden_dim, device=obs.device)
        else:
            mem = hidden  # membrane potential from previous env step

        if prev_actions is None:
            prev_actions = torch.zeros(B, self.max_deg, device=obs.device)
        if coverage_rate is None:
            coverage_rate = obs[:, 6:7]

        # Encode input features
        enc_input = torch.cat([body, prev_actions, coverage_rate], dim=-1)
        cur = self.enc_fc(enc_input)  # (B, hidden) — input current

        # Run LIF for T_enc steps (temporal integration)
        for _ in range(self.T_enc):
            spk, mem = self.lif(cur, mem)  # spk is spike (0/1), mem is potential

        # Use membrane potential as "body features"
        body_feats = mem

        # Candidate scoring (same as current ANN design)
        flat = candidates.reshape(-1, cand_feat_dim)
        cand_feats = self.cand_net(flat).reshape(B, self.max_deg, -1)
        body_exp = body_feats.unsqueeze(1).expand(-1, self.max_deg, -1)
        scores = self.score_net(
            torch.cat([body_exp, cand_feats], dim=-1)
        ).squeeze(-1)

        if mask is not None:
            scores[~mask] = float('-inf')

        return torch.softmax(scores, dim=-1), mem  # hidden = membrane
```

---

## 3. Training Protocol

### 3.1 Surrogate Gradient PPO

The same PPO pipeline, but gradients flow through the LIF's surrogate gradient:

```
1. Rollout: SNNActor acts (T_enc internal steps per env step)
2. Store: obs, action, reward, done, log_prob (same as now)
3. Update:
   a. Compute GAE (same as now)
   b. Forward pass through SNN (with surrogate gradients)
   c. PPO clipped loss (same as now)
   d. Backward: gradients flow through LIF membrane dynamics
```

### 3.2 Key Hyperparameters

| Parameter | Current (GRU) | SNN | Reason |
|-----------|--------------|-----|--------|
| Hidden dim | 64 | 64 | Same capacity |
| Time encoding | 1 step/env step | T_enc=5 | SNN needs time to integrate |
| Learning rate | 3e-4 | 1e-4 | Surrogate gradients are noisier |
| Entropy coef | 0.05→0.001 | 0.05→0.001 | Same exploration schedule |
| Truncated BPTT | 16 steps | 16 × T_enc | Longer due to internal steps |

### 3.3 ANN-to-SNN Initialization

**Critical trick:** Pre-train the weights as an ANN, then convert to SNN.
1. Train current GRU architecture → good weights
2. Copy weights to SNN (fc layers are identical)
3. Fine-tune with surrogate gradients (50-100 episodes)
4. Deploy as pure SNN (no surrogate)

This avoids training SNN from scratch, which is notoriously difficult.

---

## 4. Deployment on Neuromorphic Hardware

### 4.1 Target Platforms

| Platform | Power | Maturity | Space-qualified |
|----------|-------|----------|-----------------|
| Intel Loihi 2 | ~1W | Production | Not yet |
| BrainScaleS | ~10W | Research | No |
| SpiNNaker2 | ~1W | Production | No |
| **FPGA (custom)** | ~0.5W | **Prototype** | **Yes (Rad-hard)** |
| **MEMS/analog** | ~10mW | Research | Feasible |

**Recommendation:** Design for FPGA-based deployment (rad-hard FPGAs are space-qualified).

### 4.2 Conversion Steps

1. **Remove surrogate gradients** → standard LIF inference
2. **Quantize weights** → int8 or binary for FPGA efficiency
3. **Optimize T_enc** → minimum encoding steps for task performance
4. **Hardware-in-loop test** → FPGA emulation of SNN policy

---

## 5. Comparison: GRU vs SNN Roadmap

### Phase 1: GRU Baseline (DONE — 88.04%)
```
ANN + GRU (PyTorch GPU) → Best: 88.04%
```

### Phase 2: SNN Simulation (NEXT)
```
SNN + LIF (snnTorch GPU) → Target: >85% coverage
Same PPO pipeline, just replace GRU with LIF
```

### Phase 3: SNN Optimization
```
Optimize T_enc, weight quantization, fine-tune
Target: >85% with minimal encoding steps
```

### Phase 4: Neuromorphic Deployment
```
Export to FPGA / neuromorphic platform
Target: Real-time inference at <100mW
```

---

## 6. Concrete Steps to Implement

### Step 1: Install snnTorch
```bash
pip install snntorch
```

### Step 2: Create `src/agents/snn_actor.py`
- Copy `mappo.py` → `snn_actor.py`
- Replace GRUCell with `snn.Leaky`
- Add spike encoding
- Keep candidate encoder + scorer (ANN)

### Step 3: Quick Test (5 episodes)
- Compare: SNN vs GRU coverage
- Monitor: firing rates, membrane potentials

### Step 4: Full Training (200 episodes)
- Compare loss curves, convergence speed

### Step 5: Optional — Hybrid (ANN body + SNN memory)
- Body encoder: ANN (efficient)
- Temporal memory: SNN LIF
- Decision: ANN candidate scorer

---

## 7. Key References

1. **Espino et al. 2024** — SNN Wavefront Planner for mobile robots
   - arXiv:2404.15524 — Most similar to our problem
2. **Courtois et al. 2025** — SNN 6D pose estimation for space
   - arXiv:2501.02916 — Shows SNN viability in space domain
3. **Dold 2026** — Single vs multi-spike equivalence
   - arXiv:2603.13478 — Theoretical foundation
4. **Casanueva-Morato 2025** — SNN trajectory interpolation
   - arXiv:2501.17172 — SNN closed-loop control
5. **snnTorch docs** — https://snntorch.readthedocs.io/
6. **SPAICE 2024** — https://doi.org/10.5281/zenodo.13889941
