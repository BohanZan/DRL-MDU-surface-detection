"""
MAPPO with entropy annealing + truncated BPTT for GRU temporal memory.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Optional


class Actor(nn.Module):
    """GRU-augmented policy: scores physical candidate positions with temporal memory.

    Architecture:
        obs → body_net → body_feats(64)
            ↓
        concat([body_feats, prev_action_onehot, coverage])
            ↓
        GRUCell(69→64) ── hidden_state (carried step-to-step)
            ↓
        gru_hidden → ⊕ candidate_encoder → score per candidate → softmax
    """

    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.max_deg = act_dim
        self.hidden_dim = hidden
        cand_feat_dim = 8
        body_dim = obs_dim - self.max_deg * cand_feat_dim

        # Body: encodes context (self, other MDUs, coverage stats, visit, step)
        self.body_net = nn.Sequential(
            nn.Linear(body_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )

        # GRU memory: input = body_feats(64) + prev_action_oh(act_dim) + coverage(1)
        self.gru_cell = nn.GRUCell(hidden + act_dim + 1, hidden)

        # Candidate encoder (shared weights, applied per candidate slot)
        self.cand_net = nn.Sequential(
            nn.Linear(cand_feat_dim, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, hidden // 2), nn.Tanh(),
        )

        # Scorer: GRU hidden + candidate features → logit
        self.score_net = nn.Sequential(
            nn.Linear(hidden + hidden // 2, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, mask=None, hidden=None, prev_actions=None, coverage_rate=None):
        """Returns (probs, new_hidden).

        Args:
            obs: (B, obs_dim) — last body_dims are the 7-dim candidate features
            mask: optional (B, max_deg) action validity mask
            hidden: optional (B, H) GRU hidden state from previous step
            prev_actions: optional (B, act_dim) one-hot of previous step's actions
            coverage_rate: optional (B, 1) global coverage rate at this step
        """
        B = obs.shape[0]
        cand_feat_dim = 8
        body = obs[:, :-self.max_deg * cand_feat_dim]
        candidates = obs[:, -self.max_deg * cand_feat_dim:].reshape(B, self.max_deg, cand_feat_dim)

        body_feats = self.body_net(body)                                    # (B, H)

        # GRU step
        if hidden is None:
            hidden = torch.zeros(B, self.hidden_dim, device=obs.device)
        if prev_actions is None:
            prev_actions = torch.zeros(B, self.max_deg, device=obs.device)
        if coverage_rate is None:
            coverage_rate = obs[:, 6:7]  # extract from body region

        gru_input = torch.cat([body_feats, prev_actions, coverage_rate], dim=-1)  # (B, H+5)
        new_hidden = self.gru_cell(gru_input, hidden)                              # (B, H)

        # Encode and score each candidate using GRU hidden as context
        flat = candidates.reshape(-1, cand_feat_dim)
        cand_feats = self.cand_net(flat).reshape(B, self.max_deg, -1)       # (B, max_deg, H/2)

        body_exp = new_hidden.unsqueeze(1).expand(-1, self.max_deg, -1)
        scores = self.score_net(
            torch.cat([body_exp, cand_feats], dim=-1)
        ).squeeze(-1)                                                        # (B, max_deg)

        if mask is not None:
            scores[~mask] = float('-inf')

        return torch.softmax(scores, dim=-1), new_hidden


class Critic(nn.Module):
    def __init__(self, state_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


class Buffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.state = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.action_masks = []

    def push(self, obs, state, actions, reward, done, values, log_probs,
             action_mask=None):
        self.obs.append(obs)
        self.state.append(state)
        self.actions.append(actions)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(values)
        self.log_probs.append(log_probs)
        self.action_masks.append(action_mask)

    def compute_gae(self, gamma, gae_lambda):
        T = len(self.obs)
        A_agents = len(self.values[0])  # num_mdus

        obs_a = np.array(self.obs)
        state_a = np.array(self.state)
        act_a = np.array(self.actions)
        rew_a = np.array(self.rewards)
        done_a = np.array(self.dones)
        val_a = np.array(self.values)
        lp_a = np.array(self.log_probs)

        # Per-agent GAE: each MDU gets its own advantage based on its own value
        adv = np.zeros_like(val_a)          # (T, A)
        last = np.zeros(A_agents)           # (A,) — per-agent trace
        for t in reversed(range(T)):
            nv = val_a[t + 1] if t < T - 1 else np.zeros(A_agents)
            delta = rew_a[t] + gamma * nv * (1 - done_a[t]) - val_a[t]
            last = delta + gamma * gae_lambda * (1 - done_a[t]) * last
            adv[t] = last

        ret = adv + val_a                   # (T, A)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        if self.action_masks[0] is not None:
            am_a = np.array(self.action_masks)
        else:
            am_a = None

        return {"obs": obs_a, "state": state_a, "actions": act_a,
                "log_probs": lp_a, "adv": adv, "ret": ret,
                "action_masks": am_a}


@dataclass
class MAPPOConfig:
    """PPO hyperparameters. Defaults pulled from src.config.AgentConfig."""
    from src.config import AgentConfig as _AC
    lr_actor: float = _AC.lr_actor
    lr_critic: float = _AC.lr_critic
    gamma: float = _AC.gamma
    gae_lambda: float = _AC.gae_lambda
    clip_epsilon: float = _AC.clip_epsilon
    ent_coef_start: float = _AC.ent_coef_start
    ent_coef_end: float = _AC.ent_coef_end
    max_grad_norm: float = _AC.max_grad_norm
    num_epochs: int = _AC.num_epochs
    hidden_dim: int = _AC.hidden_dim
    batch_size: int = _AC.batch_size
    bptt_len: int = _AC.bptt_len


class MAPPO:
    def __init__(self, obs_dim, state_dim, act_dim,
                 config=None, device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = config or MAPPOConfig()
        self.device = torch.device(device)
        self.act_dim = act_dim
        H = self.cfg.hidden_dim

        self.actor = Actor(obs_dim, act_dim, H).to(device)
        # Critic sees global state + aggregated (mean) GRU hidden
        self.critic = Critic(state_dim + H, H).to(device)
        self.optim_a = optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.optim_c = optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic)

        self.buffer = Buffer()
        self._ent_coef = self.cfg.ent_coef_start
        self._hidden_states = None  # (num_mdus, H) set during act()

    def anneal_entropy(self, progress: float):
        # Cosine schedule: stays high for most of training, drops near the end
        self._ent_coef = self.cfg.ent_coef_end + 0.5 * (
            self.cfg.ent_coef_start - self.cfg.ent_coef_end
        ) * (1 + np.cos(np.pi * progress))

    def reset_hidden(self, num_mdus=None):
        """Reset GRU hidden states at episode start."""
        N = num_mdus if num_mdus is not None else (
            self._hidden_states.shape[0] if self._hidden_states is not None else 1)
        self._hidden_states = torch.zeros(N, self.cfg.hidden_dim, device=self.device)

    @torch.no_grad()
    def act(self, obs, state, action_mask=None, greedy=False,
            prev_actions=None, coverage_rate=None):
        B = len(obs)
        obs_t = torch.FloatTensor(obs).to(self.device)
        mask_t = (torch.BoolTensor(action_mask).to(self.device)
                  if action_mask is not None else None)

        # Handle state: broadcast from 1D or (1, state_dim) → (B, state_dim)
        state_np = np.asarray(state)
        if state_np.ndim == 1:
            state_t = torch.FloatTensor(state_np).to(self.device).unsqueeze(0).expand(B, -1)
        else:
            state_t = torch.FloatTensor(state_np).to(self.device)

        # Handle prev_actions (int array → one-hot)
        if prev_actions is None:
            prev_actions_t = torch.zeros(B, self.act_dim, device=self.device)
        else:
            prev_actions_t = torch.nn.functional.one_hot(
                torch.LongTensor(prev_actions).to(self.device), self.act_dim).float()

        # Handle coverage_rate
        if coverage_rate is not None:
            cr_t = torch.FloatTensor(coverage_rate).to(self.device).unsqueeze(-1)
        else:
            cr_t = obs_t[:, 6:7]

        # Actor forward with GRU hidden state
        probs, self._hidden_states = self.actor(
            obs_t, mask=mask_t, hidden=self._hidden_states,
            prev_actions=prev_actions_t, coverage_rate=cr_t)

        # Critic sees global state + hidden state per MDU
        critic_input = torch.cat([state_t, self._hidden_states], dim=-1)
        v = self.critic(critic_input.unsqueeze(0))
        values = v.view(-1).cpu().numpy()

        if greedy:
            actions = probs.argmax(dim=-1).cpu().numpy()
        else:
            actions = torch.distributions.Categorical(probs).sample().cpu().numpy()
        lp = torch.log(probs.gather(1, torch.LongTensor(actions).unsqueeze(-1).to(self.device)) + 1e-10)
        return actions, values, lp.squeeze(-1).cpu().numpy()

    def store(self, obs, state, actions, reward, done, values, log_probs,
              action_mask=None):
        self.buffer.push(obs, state, actions, reward, done, values, log_probs, action_mask)

    def update(self):
        """PPO update with truncated BPTT for GRU efficiency.

        Trajectory is split into chunks of `bptt_len` steps. Within each chunk,
        time steps are processed sequentially. At chunk boundaries: clip, step,
        and free the computation graph. Hidden state carries across chunks.
        """
        data = self.buffer.compute_gae(self.cfg.gamma, self.cfg.gae_lambda)
        self.buffer.clear()

        T, A = data["obs"].shape[:2]
        device = self.device
        H = self.cfg.hidden_dim
        act_dim = self.act_dim
        bptt_len = self.cfg.bptt_len
        has_masks = data.get("action_masks") is not None

        obs_t = torch.FloatTensor(data["obs"]).to(device)                 # (T, A, obs_dim)
        act_t = torch.LongTensor(data["actions"]).to(device)              # (T, A)
        lp_t = torch.FloatTensor(data["log_probs"]).to(device)            # (T, A)
        adv_t = torch.FloatTensor(data["adv"]).to(device)                 # (T, A)
        ret_t = torch.FloatTensor(data["ret"]).to(device)                 # (T, A)
        state_t = torch.FloatTensor(data["state"]).to(device)             # (T, state_dim)

        if has_masks:
            am_t = torch.BoolTensor(data["action_masks"]).to(device)

        # Normalize returns to zero mean, unit variance
        ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

        stats = {"loss_a": 0.0, "loss_c": 0.0, "entropy": 0.0,
                 "hidden_norm": 0.0, "grad_norm": 0.0}
        n_steps = 0

        for epoch in range(self.cfg.num_epochs):
            hidden = torch.zeros(A, H, device=device)
            prev_actions_oh = torch.zeros(A, act_dim, device=device)
            self.optim_a.zero_grad()
            self.optim_c.zero_grad()

            for chunk_start in range(0, T, bptt_len):
                chunk_end = min(chunk_start + bptt_len, T)

                for t in range(chunk_start, chunk_end):
                    cr_t = obs_t[t, :, 6:7]
                    mask = am_t[t] if has_masks else None

                    probs, hidden = self.actor(
                        obs_t[t], mask=mask, hidden=hidden,
                        prev_actions=prev_actions_oh, coverage_rate=cr_t)

                    dist = torch.distributions.Categorical(probs)
                    new_lp = dist.log_prob(act_t[t])
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_lp - lp_t[t])
                    s1 = ratio * adv_t[t]
                    s2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon,
                                     1 + self.cfg.clip_epsilon) * adv_t[t]
                    loss_a = -torch.min(s1, s2).mean() - self._ent_coef * entropy

                    state_step = state_t[t].unsqueeze(0).expand(A, -1)
                    critic_input = torch.cat([state_step, hidden], dim=-1)
                    val = self.critic(critic_input)
                    loss_c = nn.MSELoss()(val, ret_t[t])

                    is_last_in_chunk = (t == chunk_end - 1)
                    (loss_a + loss_c).backward(retain_graph=not is_last_in_chunk)

                    stats["loss_a"] += loss_a.item()
                    stats["loss_c"] += loss_c.item()
                    stats["entropy"] += entropy.item()
                    stats["hidden_norm"] += hidden.norm().item()
                    n_steps += 1

                    prev_actions_oh = torch.nn.functional.one_hot(
                        act_t[t], act_dim).float().to(device)

                # End of chunk: clip + step
                params = list(self.actor.parameters()) + list(self.critic.parameters())
                g_norm = nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
                stats["grad_norm"] += float(g_norm) if isinstance(g_norm, torch.Tensor) else g_norm
                self.optim_a.step()
                self.optim_c.step()
                # Detach hidden from graph for next chunk
                hidden = hidden.detach()
                prev_actions_oh = prev_actions_oh.detach()
                self.optim_a.zero_grad()
                self.optim_c.zero_grad()

        for k in stats:
            stats[k] /= max(n_steps, 1)
        return stats

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                     "critic": self.critic.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
