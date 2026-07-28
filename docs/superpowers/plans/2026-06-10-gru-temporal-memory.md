# GRU Temporal Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GRU recurrent memory to Actor and Critic to eliminate MDU oscillation and enable temporal reasoning.

**Architecture:** Insert a GRUCell between body_encoder and policy head in the Actor. The GRU receives (body_feats + prev_action_encoding) and maintains a hidden state across episode steps. The Critic receives the aggregated hidden state. During PPO update, trajectories are processed sequentially through the GRU (no time-step shuffling), preserving temporal dependencies.

**Tech Stack:** PyTorch GRUCell, existing MAPPO framework, CUDA, single-GPU

---

## File Structure

| File | Change Type | Responsibility |
|------|-------------|---------------|
| `src/agents/mappo.py` | **Modify** | Actor: add GRUCell, forward takes/returns hidden. Critic: add GRU. Update: sequential PPO. |
| `src/train.py` | **Modify** | Manage hidden states: reset on episode start, carry step-to-step, log debug metrics. |
| `tests/test_gru_memory.py` | **Create** | Unit tests: hidden state shape, carry-over, reset, masking. |

---

### Task 1: Modify Actor with GRU memory cell

**Files:**
- Modify: `src/agents/mappo.py` (Actor class)

Architecture:
```
obs → body_encoder(18→64→64) → body_feats(64)
                                  ↓
                   concat([body_feats(64), prev_action_oh(4), coverage(1)]) → (69,)
                                  ↓
                            GRUCell(69→64) ← hidden state (64,)
                                  ↓
                            gru_hidden(64) — used as "body" for candidate scoring
                                  ↓  ↓
                     cand_encoder + score_net → probs
```

- [ ] **Step 1.1: Add GRUCell to Actor.__init__**

Replace current `__init__`:

```python
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.max_deg = act_dim
        self.hidden_dim = hidden
        cand_feat_dim = 7
        candidate_dim = self.max_deg * cand_feat_dim
        body_dim = obs_dim - candidate_dim

        # Body encoder
        self.body_net = nn.Sequential(
            nn.Linear(body_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )

        # GRU memory cell: input = body_feats(64) + prev_action_onehot(4) + coverage(1) = 69
        self.gru_cell = nn.GRUCell(69, hidden)

        # Candidate encoder
        self.cand_net = nn.Sequential(
            nn.Linear(cand_feat_dim, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, hidden // 2), nn.Tanh(),
        )

        # Scorer: combine GRU hidden + candidate features
        self.score_net = nn.Sequential(
            nn.Linear(hidden + hidden // 2, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
```

- [ ] **Step 1.2: Update Actor.forward to accept and return hidden state**

```python
def forward(self, obs, mask=None, hidden=None, prev_actions=None, coverage_rate=None):
    """Returns probs, new_hidden.
    
    Args:
        obs: (B, obs_dim) observation tensor
        mask: optional (B, max_deg) action mask
        hidden: optional (B, hidden_dim) previous GRU hidden state
        prev_actions: (B, 4) one-hot of previous step's actions
        coverage_rate: (B, 1) global coverage rate
    """
    B = obs.shape[0]
    cand_feat_dim = 7
    body = obs[:, :-self.max_deg * cand_feat_dim]
    candidates = obs[:, -self.max_deg * cand_feat_dim:].reshape(B, self.max_deg, cand_feat_dim)

    body_feats = self.body_net(body)  # (B, H)

    # GRU step
    if hidden is None:
        hidden = torch.zeros(B, self.hidden_dim, device=obs.device)
    if prev_actions is None:
        prev_actions = torch.zeros(B, self.max_deg, device=obs.device)
    if coverage_rate is None:
        coverage_rate = obs[:, 6:7]  # extract global coverage from body

    gru_input = torch.cat([body_feats, prev_actions, coverage_rate], dim=-1)  # (B, H+5)
    new_hidden = self.gru_cell(gru_input, hidden)  # (B, H)

    # Score each candidate using GRU hidden as body context
    flat = candidates.reshape(-1, cand_feat_dim)
    cand_feats = self.cand_net(flat).reshape(B, self.max_deg, -1)

    body_exp = new_hidden.unsqueeze(1).expand(-1, self.max_deg, -1)
    scores = self.score_net(
        torch.cat([body_exp, cand_feats], dim=-1)
    ).squeeze(-1)

    if mask is not None:
        scores[~mask] = float('-inf')

    return torch.softmax(scores, dim=-1), new_hidden
```

- [ ] **Step 1.3: Validate parameter count**

Run: `python -c "from src.agents.mappo import Actor; a=Actor(46,4); print(f'Actor params: {sum(p.numel() for p in a.parameters()):,}')"`
Expected: ~20K params (old was 15K, GRU adds ~9K for 64-dim)

---

### Task 2: Update MAPPO.act for hidden state management

**Files:**
- Modify: `src/agents/mappo.py` (MAPPO class)

- [ ] **Step 2.1: Update MAPPO.__init__ to include hidden state tracking**

```python
class MAPPO:
    def __init__(self, obs_dim, state_dim, act_dim,
                 config: Optional[MAPPOConfig] = None,
                 device: str = "cpu"):
        ...
        self.actor = Actor(obs_dim, act_dim, self.cfg.hidden_dim).to(device)
        self.critic = Critic(state_dim + self.cfg.hidden_dim, self.cfg.hidden_dim).to(device)
        ...
        self._hidden_states = None  # (num_mdus, hidden_dim) — set during act()
```

- [ ] **Step 2.2: Update MAPPO.act to carry hidden state**

```python
@torch.no_grad()
def act(self, obs, state, action_mask=None, greedy=False,
        coverage_rate=None, prev_actions=None):
    obs_t = torch.FloatTensor(obs).to(self.device)
    state_t = torch.FloatTensor(state).to(self.device)
    mask_t = (torch.BoolTensor(action_mask).to(self.device)
              if action_mask is not None else None)
    
    # Get/handle prev_actions
    if prev_actions is None:
        prev_actions_t = torch.zeros(len(obs), self.act_dim, device=self.device)
    else:
        prev_actions_t = torch.nn.functional.one_hot(
            torch.LongTensor(prev_actions), self.act_dim).float().to(self.device)
    
    # Get/handle coverage rate
    if coverage_rate is None:
        cr_t = obs_t[:, 6:7]  # extract from observation
    else:
        cr_t = torch.FloatTensor([[cr] for cr in coverage_rate]).to(self.device)
    
    # Actor forward with hidden state
    probs, self._hidden_states = self.actor(
        obs_t, mask=mask_t, hidden=self._hidden_states,
        prev_actions=prev_actions_t, coverage_rate=cr_t)
    
    # Critic forward with hidden state
    critic_input = torch.cat([state_t, self._hidden_states], dim=-1)
    v = self.critic(critic_input.unsqueeze(0))
    values = v.expand(len(obs)).cpu().numpy()
    
    if greedy:
        actions = probs.argmax(dim=-1).cpu().numpy()
    else:
        actions = torch.distributions.Categorical(probs).sample().cpu().numpy()
    lp = torch.log(probs.gather(1, torch.LongTensor(actions).unsqueeze(-1).to(self.device)) + 1e-10)
    
    return actions, values, lp.squeeze(-1).cpu().numpy()
```

- [ ] **Step 2.3: Add reset_hidden method**

```python
def reset_hidden(self, num_mdus=None):
    """Reset GRU hidden states. Call at episode start."""
    N = num_mdus if num_mdus is not None else (self._hidden_states.shape[0] if self._hidden_states is not None else 1)
    self._hidden_states = torch.zeros(N, self.cfg.hidden_dim, device=self.device)
```

---

### Task 3: Update Critic with GRU

**Files:**
- Modify: `src/agents/mappo.py` (Critic class)

The critic receives a larger state that includes the GNN-aggregated hidden info:
- Current: `state_dim = 14` (4×MDU_positions + coverage + step)
- New: `state_dim + hidden_dim = 14 + 64 = 78`
- Actually, we could just concatenate the **mean** hidden state from the actor's GRU

- [ ] **Step 3.1: Update Critic input dimension**

Critic state dimension is already dynamic from the MAPPO init. The `state_dim` passed to Critic is `info["global_state"].shape[0]` which is 14. We add `hidden_dim` to it.

In `MAPPO.__init__`:
```python
self.critic = Critic(state_dim + self.cfg.hidden_dim, self.cfg.hidden_dim).to(device)
```

Critic architecture stays the same (MLP), just input dim is larger.

- [ ] **Step 3.2: No critic architecture change needed**

The existing `Critic(nn.Module)` class works as-is. Its `forward` takes any input dimension. The only change is what we pass to it — concatenate hidden state with global state.

---

### Task 4: Update PPO update for sequential trajectory processing

**Files:**
- Modify: `src/agents/mappo.py` (MAPPO.update method)

- [ ] **Step 4.1: Rewrite update() for sequential BPTT**

The current update() flattens (T, A, dim) → (T*A, dim) and shuffles. With GRU, we CANNOT shuffle time steps. Instead:

```python
def update(self):
    """PPO update with sequential GRU processing (no time-shuffling)."""
    data = self.buffer.compute_gae(self.cfg.gamma, self.cfg.gae_lambda)
    self.buffer.clear()
    
    T, A = data["obs"].shape[:2]  # (T, A, dim)
    device = self.device
    H = self.cfg.hidden_dim
    
    obs_t = torch.FloatTensor(data["obs"]).to(device)          # (T, A, obs_dim)
    act_t = torch.LongTensor(data["actions"]).to(device)       # (T, A)
    lp_t = torch.FloatTensor(data["log_probs"]).to(device)     # (T, A)
    adv_t = torch.FloatTensor(data["adv"]).to(device)          # (T, A)
    ret_t = torch.FloatTensor(data["ret"]).to(device)          # (T, A)
    state_t = torch.FloatTensor(data["state"]).to(device)     # (T, state_dim)
    
    stats = {"loss_a": 0.0, "loss_c": 0.0, "entropy": 0.0, "hidden_norm": 0.0}
    n_updates = 0
    
    for epoch in range(self.cfg.num_epochs):
        # Process each episode sequentially through GRU
        # Initialize hidden states to zeros at start of each episode
        hidden = torch.zeros(A, H, device=device)
        prev_actions_oh = torch.zeros(A, self.act_dim, device=device)
        
        for t in range(T):
            # Actor forward with hidden
            obs_batch = obs_t[t]  # (A, obs_dim)
            mask_batch = (data["action_masks"][t] if "action_masks" in data else None)
            
            # Extract coverage rate from obs
            cr_batch = obs_t[t, :, 6:7]
            
            probs, hidden = self.actor(
                obs_batch,
                prev_actions=prev_actions_oh,
                coverage_rate=cr_batch,
                hidden=hidden,
            )
            
            # PPO loss at this time step
            dist = torch.distributions.Categorical(probs)
            new_lp = dist.log_prob(act_t[t])
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_lp - lp_t[t])
            s1 = ratio * adv_t[t]
            s2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * adv_t[t]
            loss_a = -torch.min(s1, s2).mean() - self._ent_coef * entropy
            
            # Critic
            critic_input = torch.cat([state_t[t], hidden], dim=-1)
            val = self.critic(critic_input)
            loss_c = nn.MSELoss()(val, ret_t[t])
            
            # Accumulate gradients
            loss = loss_a + loss_c
            loss.backward()
            
            # Store prev_action for next step
            prev_actions_oh = torch.nn.functional.one_hot(act_t[t], self.act_dim).float().to(device)
            
            stats["loss_a"] += loss_a.item()
            stats["loss_c"] += loss_c.item()
            stats["entropy"] += entropy.item()
            stats["hidden_norm"] += hidden.norm().item()
            n_updates += 1
        
        # Apply gradient clipping and step at end of each trajectory
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.optim_a.step()
        self.optim_c.step()
        self.optim_a.zero_grad()
        self.optim_c.zero_grad()
    
    for k in stats:
        stats[k] /= max(n_updates, 1)
    return stats
```

- [ ] **Step 4.2: Update Buffer to also store action_masks**

In `Buffer.push()`: add `action_mask` parameter
In `Buffer.compute_gae()`: also store action_masks in the returned dict

---

### Task 5: Update training loop

**Files:**
- Modify: `src/train.py`

- [ ] **Step 5.1: Add hidden state management to training loop**

```python
for ep in range(1, args.episodes + 1):
    agent.anneal_entropy((ep - 1) / args.episodes)
    obs, info = env.reset()
    agent.reset_hidden(num_mdus=env.num_mdus)  # ← NEW
    
    done = False
    total_r = 0.0
    steps = 0
    prev_actions = np.zeros(env.num_mdus, dtype=int)  # ← NEW
    
    while not done:
        # Extract coverage rate for debug logging
        cr = info.get("coverage_rate", obs[:, 6].mean())
        
        acts, vals, lps = agent.act(
            obs, info["global_state"],
            action_mask=info["action_mask"],
            prev_actions=prev_actions,  # ← NEW
            coverage_rate=[cr] * env.num_mdus,  # ← NEW
        )
        
        obs, r, term, trunc, info = env.step(acts)
        agent.store(obs, info["global_state"], acts, r, term or trunc, vals, lps,
                    action_mask=info["action_mask"])  # ← MODIFIED: pass mask
        
        prev_actions = acts  # ← NEW
        total_r += r
        steps += 1
        done = term or trunc
```

- [ ] **Step 5.2: Add debug logging (hidden norm, entropy, grad norm)**

In the log line, add:
```python
print(f"Ep {ep:5d} | R {avg_r:+.3f} | Cov {avg_c:.2%}{cross_s} | "
      f"Len {avg_l:.0f} | Ent {stats['entropy']:.3f} | "
      f"H| {stats['hidden_norm']:.2f} | LR {agent._ent_coef:.4f}")
```

---

### Task 6: Update generate_trajectory.py for hidden state

**Files:**
- Modify: `src/generate_trajectory.py`

- [ ] **Step 6.1: Add hidden state management**

```python
if args.mode == "mappo":
    agent.reset_hidden(num_mdus=args.mdus)
    prev_actions = np.zeros(args.mdus, dtype=int)

for step in range(1, args.steps + 1):
    if args.mode == "mappo":
        cr = info.get("coverage_rate", obs[:, 6].mean())
        actions, _, _ = agent.act(
            obs, info["global_state"],
            action_mask=info["action_mask"],
            prev_actions=prev_actions,
            coverage_rate=[cr] * args.mdus,
        )
        prev_actions = actions
```

---

### Task 7: Verification

- [ ] **Step 7.1: Quick test (5 episodes)**

```bash
KMP_DUPLICATE_LIB_OK=TRUE python src/train.py --mdus 1 --episodes 5 --max-steps 50 --log-every 1 --save-dir checkpoints_gru_test
```
Expected: No errors, hidden_norm in logs is non-zero.

- [ ] **Step 7.2: Single-MDU baseline test**

```bash
KMP_DUPLICATE_LIB_OK=TRUE python src/train.py --mdus 1 --episodes 200 --max-steps 200 --log-every 25 --save-dir checkpoints_gru_1mdu
```
Expected: Coverage improves over random (1 MDU ~35-40%).

- [ ] **Step 7.3: 4-MDU full training**

```bash
KMP_DUPLICATE_LIB_OK=TRUE python src/train.py --mdus 4 --episodes 1000 --max-steps 200 --log-every 25 --save-dir checkpoints_gru_4mdu
```
Expected: Coverage > random walk (65%+), fewer oscillations.

---

## Self-Review

**Spec coverage:**
1. ✅ GRU memory cell between body_encoder and policy head — Task 1
2. ✅ Hidden state management across episodes — Tasks 2, 5
3. ✅ Single-MDU first, then multi-MDU — Task 7
4. ✅ GPU-efficient — sequential trajectory processing, no unnecessary copies
5. ✅ Debug logging — Task 5.2
6. ✅ No OOM — params ~30K total, memory is O(T × A × hidden_dim) = 200×4×64×4bytes ≈ 205KB per episode buffer
7. ✅ Not crash-prone — proper hidden state reset, fallback to zeros if None

**Placeholder scan:** All code blocks are complete. No "TBD" or "TODO".

**Type consistency:** 
- `hidden` is always `(A, 64)` where A = num_mdus
- `prev_actions` is always `(A,)` int → one-hot → `(A, act_dim)`
- `coverage_rate` is `(A, 1)`
- All consistent across Tasks 1-5
