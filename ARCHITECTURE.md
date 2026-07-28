# Architecture — DRL MDU Surface Detection

```
DRL_surface_detection/
│
├── src/
│   ├── config.py                  ← 单一配置源，所有默认参数
│   ├── run_manager.py             ← 统一输出路径管理
│   │
│   ├── env/                       ← 环境层（物理模拟 + 图拓扑）
│   │   ├── asteroid.py            # Bennu 小行星 polyhedron mesh
│   │   ├── net_graph.py           # 空间网 FNS 拓扑 (369 节点 / 432 边)
│   │   └── mdu_coverage_env.py    # Gymnasium Env：reset / step / obs / reward
│   │
│   ├── agents/                    ← 智能体层（MAPPO + GRU + GNN）
│   │   ├── mappo.py               # Actor(Critic(Buffer(MAPPO — 完整 RL pipeline
│   │   └── gnn.py                 # GraphSAGE encoder（可选，未启用）
│   │
│   ├── train.py                   ← 主训练脚本
│   ├── generate_trajectory.py     ← 轨迹生成（推理）
│   ├── visualize_trajectory.py    ← 轨迹渲染 → GIF 动画
│   ├── render_animation.py        ← 独立渲染器（遗留）
│   │
│   └── __init__.py
│
├── tests/
│   └── test_env_smoke.py          ← 环境冒烟测试
│
├── compare_runs.py                ← 对比两个训练 run
├── run_overnight.py               ← 过夜训练（简化版 train.py）
│
├── notebooks/
│   └── viz_final_state.py
│
├── FNS_square_fold-50m.txt        ← 空间网拓扑数据
├── Solution.dat                   ← 网捕获后最终状态
├── polyhedron_bennu.txt           ← 小行星 mesh (1348 顶点 / 2692 面)
│
├── docs/                          ← 文献 + 设计文档
├── results/                       ← 训练输出（按时间戳管理）
└── checkpoints/                   ← 遗留，不再使用
```

---

## 工作流

### 完整实验生命周期

```
                        ╔══════════════════════╗
                        ║    1. 设置环境        ║
                        ║  conda + KMP fix      ║
                        ╚══════════╤═══════════╝
                                   │
                        ╔══════════▼═══════════╗
                        ║  2. 训练 (train.py)  ║
                        ║  生成 checkpoint +   ║
                        ║  曲线 + 参数记录      ║
                        ╚══════════╤═══════════╝
                                   │
                        ╔══════════▼═══════════╗
                        ║  3. 轨迹生成         ║
                        ║  generate_           ║
                        ║  trajectory.py       ║
                        ╚══════════╤═══════════╝
                                   │
                        ╔══════════▼═══════════╗
                        ║  4. 动画渲染         ║
                        ║  visualize_          ║
                        ║  trajectory.py       ║
                        ╚══════════╤═══════════╝
                                   │
                        ╔══════════▼═══════════╗
                        ║  5. 对比分析         ║
                        ║  compare_runs.py     ║
                        ╚══════════════════════╝
```

每个步骤通过 `--tag <tag>` 自动找到同一 run 目录，无需手动指定路径。

---

### 步骤 1: 训练 (train.py) — 完整数据流

```
入口: python src/train.py --mdus 4 --episodes 500 --save-plot --tag 4mdu

┌──────────────────────────────────────────────────────────────────────┐
│ 1. 配置层                                                           │
│                                                                      │
│   Config()                          ← 读取 src/config.py 所有默认值  │
│   Config.from_args(args, root=...)  ← CLI 参数覆盖默认值             │
│   RunManager("results", "4mdu")     ← 创建时间戳输出目录             │
│       → results/2026-07-26_HHMMSS_4mdu/                             │
│                                                                      │
│ 2. 物理层                                                            │
│                                                                      │
│   NetGraph(fns_path, solution_path) ← 369 节点, 432 边, 3D 坐标     │
│   │   ├─ load_fns_topology()        ← FNS_square_fold-50m.txt       │
│   │   └─ load_solution_final_state()← Solution.dat                  │
│   │                                                                  │
│   Asteroid(polyhedron_path)         ← 1348 顶点, 2692 三角面         │
│       ├─ centroids, normals, areas  ← 全部预计算                     │
│       └─ radius_estimate            ← 用于归一化                     │
│                                                                      │
│   MDUCoverageEnv(**cfg.env_kwargs())                                 │
│       ├─ _precompute_graph_distances() ← BFS 全对最短路径            │
│       ├─ _precompute_gnn_graph()       ← 边索引 + 边特征 (静态)      │
│       ├─ _precompute_visible_faces()   ← 369 节点 × 2692 面 可见矩阵 │
│       └─ _precompute_khops_and_coverage() ← 1/2/3-hop 邻居面片集合   │
│                                                                      │
│ 3. 智能体层                                                           │
│                                                                      │
│   MAPPO(obs_dim, state_dim, act_dim, cfg.to_mappo_config())          │
│       ├─ Actor(obs_dim=49, act_dim=4, hidden=64)                     │
│       │   ├─ body_net:  Linear(21→64→64)   ← 编码 body 特征         │
│       │   ├─ gru_cell:   GRUCell(69→64)     ← 时序记忆               │
│       │   ├─ cand_net:   Linear(7→32→32)    ← 编码候选位置 (共享权重) │
│       │   └─ score_net:  Linear(96→64→1)    ← 候选打分               │
│       │                                                              │
│       ├─ Critic(state_dim+hidden, hidden=64)                         │
│       │   └─ net: Linear(78→64→64→1)                                 │
│       │                                                              │
│       └─ Buffer()  ← 暂存完整 episode 轨迹                           │
│                                                                      │
│ 4. 训练循环 (for ep 1..500)                                          │
│                                                                      │
│   ┌─ anneal_entropy(ep/500)  →  cosine 0.05 → 0.01                 │
│   │                                                                  │
│   ├─ env.reset()                                                    │
│   │   ├─ coverage_mask ← zeros(2692)                                │
│   │   ├─ MDU 位置      ← 起始节点 (288, 296, 360, 368)              │
│   │   ├─ 初始覆盖      ← MDU 所在节点的可见面片                      │
│   │   └─ return obs(4×49), info                                     │
│   │                                                                  │
│   ├─ agent.reset_hidden(4)  →  GRU h[0..3] = 0                     │
│   │                                                                  │
│   ├─ while not done:  ← rollout (最多 200 步)                        │
│   │   │                                                              │
│   │   ├─ env.get_graph_features()                                    │
│   │   │   → node_feats(369×7)  edge_idx(2×864)  edge_feat(864×4)    │
│   │   │                                                              │
│   │   ├─ agent.act(obs, state, mask, graph)                          │
│   │   │   ├─ Actor.forward(obs, hidden, prev_actions, cov_rate)     │
│   │   │   │   ├─ body_net(obs.body)        → body_feats(64)         │
│   │   │   │   ├─ GRUCell(body || prev_act || cov, hidden) → h'(64)  │
│   │   │   │   ├─ cand_net(each candidate)  → cand_feats(32) × 4     │
│   │   │   │   ├─ score_net(h || cand_feat) → logit                  │
│   │   │   │   └─ softmax → probs, mask→-inf → sample → action       │
│   │   │   │                                                          │
│   │   │   ├─ Critic(state || h) → V(s)                              │
│   │   │   └─ return actions(4,), values(4,), log_probs(4,)          │
│   │   │                                                              │
│   │   ├─ env.step(actions)                                           │
│   │   │   ├─ 检查 action valid, fallback to random if not            │
│   │   │   ├─ 更新 MDU 位置 (沿边移动到选中邻居)                       │
│   │   │   ├─ 更新 coverage_mask: mask[node_visible] = True           │
│   │   │   ├─ 检查 completion_threshold (75%) → completion_step       │
│   │   │   ├─ 计算: r = 20.0 × newly_covered_faces                   │
│   │   │   ├─ 如果 term/trunc: r += 10.0 + 5.0×(1 - step/max)       │
│   │   │   └─ return obs'(49,), reward, done, info                   │
│   │   │                                                              │
│   │   └─ agent.store(obs, state, actions, r, done, vals, lps, graph) │
│   │       → Buffer 累积一条 transition                               │
│   │                                                                  │
│   └─ agent.update()  ← PPO + BPTT                                  │
│       │                                                              │
│       ├─ Buffer.compute_gae(γ=0.99, λ=0.95)                         │
│       │   ├─ 反向遍历: δ = r + γ·V(t+1)·(1-done) - V(t)            │
│       │   ├─ GAE: A = δ + γλ·(1-done)·A(t+1)                       │
│       │   └─ return = advantage + value, z-score normalize          │
│       │                                                              │
│       ├─ for epoch in 1..num_epochs(4):                             │
│       │   │                                                          │
│       │   ├─ for chunk in 0..bptt_len..T:                           │
│       │   │   │                                                      │
│       │   │   ├─ for t in chunk_start..chunk_end:  ← BPTT            │
│       │   │   │   ├─ Actor.forward(obs[t], hidden) → probs, h'      │
│       │   │   │   ├─ ratio = exp(log_prob_new - log_prob_old)       │
│       │   │   │   ├─ L_clip = -min(ratio·A, clip(ratio)·A)          │
│       │   │   │   ├─ L_ent = -ent_coef × entropy                    │
│       │   │   │   ├─ Critic.forward(state[t], h) → V_pred           │
│       │   │   │   ├─ L_critic = MSE(V_pred, return[t])              │
│       │   │   │   ├─ backward(L_clip + L_ent + L_critic)             │
│       │   │   │   └─ retain_graph=True (除非 last_in_chunk)          │
│       │   │   │                                                      │
│       │   │   ├─ clip_grad_norm_(params, 0.5)                       │
│       │   │   ├─ optim_a.step(), optim_c.step()                      │
│       │   │   ├─ optim_a.zero_grad(), optim_c.zero_grad()            │
│       │   │   └─ hidden.detach()  ← 跨 chunk 切断计算图              │
│       │   │                                                          │
│       │   └─ (下一 epoch)                                            │
│       │                                                              │
│       └─ return {loss_a, loss_c, entropy, hidden_norm, grad_norm}   │
│                                                                      │
│   └─ if cov ≥ best_cov: agent.save(mappo_best.pt)                   │
│                                                                      │
│ 5. 后处理                                                             │
│                                                                      │
│   ├─ Greedy eval (no noise)                                         │
│   ├─ agent.save(mappo_final.pt)                                     │
│   ├─ rm.dump_params()         → parameters.txt                      │
│   ├─ matplotlib 6-panel fig    → training_curves.png                 │
│   └─ np.savez                  → training_data.npz                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 步骤 2: 轨迹生成 (generate_trajectory.py)

```
入口: python src/generate_trajectory.py --mode mappo --tag 4mdu

┌──────────────────────────────────────────────────────────────┐
│ 1. 路径解析                                                   │
│                                                              │
│   find_run("results", tag="4mdu")                            │
│       → results/2026-07-26_HHMMSS_4mdu/                     │
│                                                              │
│   checkpoint = <run>/checkpoints/mappo_best.pt               │
│   output_dir = <run>/trajectories/                           │
│                                                              │
│ 2. 环境重建                                                   │
│                                                              │
│   Config.from_args(args, root)  ← 从 config.py 读取参数       │
│   MDUCoverageEnv(**cfg.env_kwargs())  ← 重建与训练一致的 env   │
│                                                              │
│ 3. 加载模型                                                   │
│                                                              │
│   MAPPO(obs_dim, state_dim, act_dim, cfg, device="cpu")      │
│   agent.load(checkpoint)  ← strict=False, 兼容 GNN/无GNN     │
│                                                              │
│ 4. 推理 rollout (no_grad)                                     │
│                                                              │
│   env.reset()  →  agent.reset_hidden()                       │
│                                                              │
│   for step in 1..max_steps:                                  │
│       agent.act(obs, state, mask, greedy=False)  ← 随机采样   │
│       env.step(actions)                                      │
│       记录: mdu_nodes, coverage_masks, rates, rewards         │
│                                                              │
│ 5. 保存                                                       │
│                                                              │
│   npz: trajectory_mappo.npz                                  │
│       mdu_nodes(T×4)  coverage_masks(T×2692)                 │
│       coverage_rates(T)  rewards(T)                          │
│       net_positions(369×3)  net_edges(432×2)                 │
│       ast_verts(1348×3)  ast_faces(2692×3)                   │
│                                                              │
│   txt: trajectory_mappo.txt  (人类可读)                        │
└──────────────────────────────────────────────────────────────┘
```

---

### 步骤 3: 动画渲染 (visualize_trajectory.py)

```
入口: python src/visualize_trajectory.py --tag 4mdu

┌──────────────────────────────────────────────────────────────┐
│ 1. 路径解析                                                   │
│                                                              │
│   find_run("results", tag="4mdu")                            │
│   data_path  = <run>/trajectories/trajectory_mappo.npz       │
│   output_dir = <run>/animations/                             │
│                                                              │
│ 2. 加载数据                                                   │
│                                                              │
│   np.load(data_path)                                         │
│   → mdu_nodes, coverage_masks, rates                         │
│     net_positions, ast_verts, ast_faces                      │
│                                                              │
│ 3. 逐帧渲染 (matplotlib 3D)                                   │
│                                                              │
│   for each frame t (stride=2):                               │
│       ├─ 小行星表面: 绿色=已覆盖, 灰色=未覆盖                   │
│       ├─ 空间网: 蓝色节点 + 半透明边                           │
│       ├─ MDU: 橙色(正面) / 灰色小点(背面)                      │
│       └─ 信息栏: Step / Coverage / View angle                │
│                                                              │
│   views: azim=0° (前) + azim=180° (后)                       │
│   → animation_mappo_az000_covXX.gif                          │
│   → animation_mappo_az180_covXX.gif                          │
└──────────────────────────────────────────────────────────────┘
```

---

### 步骤 4: 对比分析 (compare_runs.py)

```
入口: python compare_runs.py --tag baseline --tag2 gnn

┌──────────────────────────────────────────────────────────────┐
│ 1. 路径解析                                                   │
│                                                              │
│   find_run("results", tag="baseline") → run1                 │
│   find_run("results", tag="gnn")      → run2                 │
│                                                              │
│ 2. 加载 training_data.npz × 2                                │
│                                                              │
│ 3. 6-panel 对比图                                             │
│   ┌──────────┬──────────┬──────────┐                         │
│   │ Coverage │  Reward  │  Length  │                         │
│   ├──────────┼──────────┼──────────┤                         │
│   │ Actor L  │ Critic L │ GRU |H|  │                         │
│   └──────────┴──────────┴──────────┘                         │
│                                                              │
│ 4. 输出: <run1>/plots/comparison.png                          │
└──────────────────────────────────────────────────────────────┘
```

---

### 单步详解: env.step() 内部逻辑

```
env.step(actions=[a₀, a₁, a₂, a₃])  ← 4 个 MDU 各自选一个邻居

┌──────────────────────────────────────────────────────────────┐
│ 1. Action 验证                                                │
│                                                              │
│   for each MDU i:                                            │
│       if actions[i] 无效 (不在邻接表中 或 目标被占用):          │
│           actions[i] ← random choice(valid neighbors)         │
│                                                              │
│ 2. 移动 MDU                                                   │
│                                                              │
│   for each MDU i:                                            │
│       MDU[i].node ← neighbors[actions[i]]                    │
│       MDU[i].path.append(new_node)                           │
│       visit_counts[new_node] += 1                             │
│   step_count += 1                                             │
│                                                              │
│ 3. 覆盖更新                                                   │
│                                                              │
│   for each MDU i:                                            │
│       precomputed_visible = node_visible[MDU[i].node]        │
│       new_faces = visible & ~coverage_mask  ← 新发现的面片    │
│       coverage_mask[visible] = True          ← sticky        │
│       total_newly += new_faces.sum()                          │
│                                                              │
│ 4. 终止检查                                                   │
│                                                              │
│   coverage_rate = coverage_mask.mean()                        │
│   if coverage_rate ≥ 75% and completion_step < 0:            │
│       completion_step = step_count  ← 记录首次达标步数         │
│                                                              │
│   terminated = (completion_step ≥ 0)  ← 达标即终止            │
│   truncated  = (step_count ≥ max_steps) ← 超时              │
│                                                              │
│ 5. 奖励                                                       │
│                                                              │
│   r_step = 20.0 × new_faces  ← 按新覆盖的面片数               │
│   if terminated or truncated:                                │
│       r_bonus = 10.0 + 5.0 × (1 - completion_step/max_steps) │
│       r_step += r_bonus                                       │
│                                                              │
│ 6. 构造 Observation                                           │
│                                                              │
│   for each MDU i:                                            │
│       obs[i] = [                                             │
│           self_pos(3),       ← MDU 当前位置 / R              │
│           local_cov(3),      ← 1/2/3-hop 邻域覆盖率           │
│           global_cov(1),     ← 全局覆盖率                     │
│           others_relative(9),← 其他 MDU 相对位置 / R          │
│           graph_dist(3),     ← 到其他 MDU 的图距离 / diam     │
│           visit_norm(1),     ← min(visit/10, 1)              │
│           step_norm(1),      ← step / max_steps              │
│           candidates(28),    ← 4 候选 × 7 特征               │
│       ]                                                       │
│                                                              │
│ 7. 返回                                                       │
│                                                              │
│   return obs(4×49), reward, terminated, truncated, info       │
│   info = {global_state, action_mask, coverage_rate,           │
│            completion_step, graph}                             │
└──────────────────────────────────────────────────────────────┘
```

---

### 单步详解: agent.update() — PPO + BPTT

```
agent.update()

┌──────────────────────────────────────────────────────────────┐
│ 输入: Buffer 中存储的完整 episode 轨迹                          │
│       T 步 × 4 MDU × (obs, state, action, reward, log_prob)   │
│                                                              │
│ 1. GAE 计算                                                   │
│                                                              │
│   for t in reversed(0..T-1):                                 │
│       next_V = V[t+1] (or 0 if last)                         │
│       δ = r[t] + γ × next_V × (1-done[t]) - V[t]             │
│       A[t] = δ + γλ × (1-done[t]) × A[t+1]                   │
│                                                              │
│   return[t] = A[t] + V[t]                                    │
│   A = z_score_normalize(A)                                   │
│   return = z_score_normalize(return)                         │
│                                                              │
│ 2. PPO Epochs (× num_epochs)                                 │
│                                                              │
│   for epoch 1..4:                                            │
│                                                              │
│     hidden = zeros(4, 64)         ← GRU 初始状态               │
│     prev_actions = zeros(4, 4)    ← 上一步动作 one-hot         │
│                                                              │
│     for chunk_start in 0..bptt_len..T:                       │
│         for t in chunk_start..chunk_end:                     │
│                                                              │
│           # Forward                                           │
│           probs, hidden = Actor(obs[t], hidden, prev_act,    │
│                                 cov[t], mask[t])              │
│           dist = Categorical(probs)                           │
│           new_log_prob = dist.log_prob(actions[t])            │
│           entropy = dist.entropy().mean()                     │
│                                                              │
│           # PPO clipped loss                                  │
│           ratio = exp(new_log_prob - old_log_prob[t])         │
│           L_clip = -min(ratio×A[t], clip(ratio)×A[t]).mean()  │
│           L_ent  = -ent_coef × entropy                       │
│           L_actor = L_clip + L_ent                            │
│                                                              │
│           # Critic loss                                       │
│           V_pred = Critic(state[t], hidden)                   │
│           L_critic = MSE(V_pred, return[t])                   │
│                                                              │
│           # Backward (保留计算图直到 chunk 结束)               │
│           (L_actor + L_critic).backward(                     │
│               retain_graph=not_last_in_chunk                  │
│           )                                                   │
│                                                              │
│           prev_actions = one_hot(actions[t])                  │
│                                                              │
│         # Chunk 结束                                          │
│         clip_grad_norm(actor_params + critic_params, 0.5)    │
│         optim_a.step()   ← 更新 Actor + GRU                  │
│         optim_c.step()   ← 更新 Critic                       │
│         optim_a.zero_grad()                                   │
│         optim_c.zero_grad()                                   │
│         hidden = hidden.detach()  ← 切断跨 chunk 梯度          │
│                                                              │
│ 3. 返回统计                                                    │
│                                                              │
│   return {loss_a, loss_c, entropy, hidden_norm, grad_norm}   │
└──────────────────────────────────────────────────────────────┘
```

---

### 观察空间构建细节

```
Observation (49 dims per MDU, 4 MDU total):
┌─────────────────────────────────────────────────────────────┐
│  索引    │ 维度  │ 内容                     │ 归一化          │
│──────────┼───────┼──────────────────────────┼────────────────│
│  0:3     │   3   │ 当前 MDU 3D 位置         │ ÷ asteroid_R   │
│  3:6     │   3   │ 1/2/3-hop 邻域覆盖率     │ [0,1]          │
│  6:7     │   1   │ 全局覆盖率               │ [0,1]          │
│  7:16    │   9   │ 其他 3 个 MDU 相对位置   │ ÷ asteroid_R   │
│  16:19   │   3   │ 图距离到其他 MDU (跳数)  │ ÷ graph_diam   │
│  19:20   │   1   │ 本节点访问次数           │ min(x/10, 1)   │
│  20:21   │   1   │ 时间步归一化             │ step/max_steps │
│──────────┼───────┼──────────────────────────┼────────────────│
│  21:49   │  28   │ 4 候选 × 7 特征:         │                │
│          │       │  · global_pos(3)         │ ÷ asteroid_R   │
│          │       │  · relative_pos(3)        │ ÷ asteroid_R   │
│          │       │  · uncovered_value(1)     │ [0,1]          │
└─────────────────────────────────────────────────────────────┘

Global State (14 dims, Critic 输入):
  0:12: 4 个 MDU 的位置 (3×4) ÷ R
  12:13: 全局覆盖率
  13:14: 归一化步数
```

---

### 配置加载流程

```
Config()                              ← 所有字段取 dataclass 默认值
    │
    ├─ Config.from_args(args, root)   ← CLI 覆盖
    │   │
    │   for arg in [cone_angle, mdus, lr_actor, ...]:
    │       if getattr(args, arg) is not None:
    │           sub_config.field = args.arg
    │
    ├─ .resolve_paths(root)           ← 相对路径 → 绝对路径
    │   DataPaths(
    │       fns         = root/FNS_square_fold-50m.txt
    │       solution    = root/Solution.dat
    │       polyhedron  = root/polyhedron_bennu.txt
    │   )
    │
    ├─ .env_kwargs(root) → dict       ← 传给 MDUCoverageEnv()
    │   {
    │       fns_path, solution_path, polyhedron_path,  ← 绝对路径
    │       mdu_start_nodes[:num_mdus],  ← 切片到实际 MDU 数
    │       cone_angle_deg, cone_range, max_steps,
    │       coverage_threshold, completion_threshold,
    │       r_newly, r_completion, r_speed, seed
    │   }
    │
    └─ .to_mappo_config() → MAPPOConfig ← 传给 MAPPO()
        {
            lr_actor, lr_critic, gamma, gae_lambda,
            clip_epsilon, ent_coef_start, ent_coef_end,
            max_grad_norm, num_epochs, hidden_dim,
            batch_size, bptt_len, use_gnn, gnn_hidden
        }
```

---

### 路径解析工作流

```
所有输出路径的唯一入口: src/run_manager.py

训练时 (train.py, run_overnight.py):
┌──────────────────────────────────────────────────────────────┐
│  rm = RunManager("results", tag="4mdu")                      │
│      ↓                                                       │
│  rm.run_dir           = "results/2026-07-26_HHMMSS_4mdu/"    │
│  rm.cp_path("x.pt")   = ".../checkpoints/x.pt"               │
│  rm.traj_path("x.npz")= ".../trajectories/x.npz"             │
│  rm.anim_path("x.gif")= ".../animations/x.gif"               │
│  rm.plt_path("x.png") = ".../plots/x.png"                    │
│  rm.params_path()     = ".../parameters.txt"                 │
└──────────────────────────────────────────────────────────────┘

推理时 (generate_trajectory.py, visualize_trajectory.py):
┌──────────────────────────────────────────────────────────────┐
│  run_dir = find_run("results", tag="4mdu")                   │
│      ↓                                                       │
│  results/ 内搜索: 文件名以 _4mdu 结尾的最新目录               │
│      ↓                                                       │
│  RunManager.checkpoint_path(run_dir, "mappo_best.pt")        │
│  RunManager.trajectory_path(run_dir, "trajectory_mappo.npz") │
│  RunManager.animation_path(run_dir, "animation_*.gif")       │
│  RunManager.plot_path(run_dir, "comparison.png")             │
└──────────────────────────────────────────────────────────────┘

优先级: 精确 run_name > tag 匹配 > 最新目录 > 报错
```

---

## 核心模块

### 1. `src/config.py` — 单一配置源

```
Config
├── DataPaths      — FNS / Solution / polyhedron 文件路径
├── EnvConfig      — cone (80°) / range (300m) / max_steps / thresholds / rewards
├── AgentConfig    — lr / gamma / PPO params / GRU hidden / GNN flag
└── TrainConfig    — episodes / seed / device / log_every
```

所有脚本通过 `Config()` 或 `Config.from_args(args)` 读取，CLI 参数叠加覆盖默认值。

### 2. `src/run_manager.py` — 统一路径管理

全项目唯一的输出路径入口。

| API | 用途 |
|-----|------|
| `RunManager(root, tag)` | **训练时**创建新 run 目录（带时间戳） |
| `find_run(root, tag)` | **推理时**按 tag 查找已有 run |
| `find_run(root)` | 查找最新 run |
| `RunManager.checkpoint_path(run_dir, file)` | 拼接 checkpoint 路径 |
| `RunManager.trajectory_path(run_dir, file)` | 拼接 trajectory 路径 |
| `RunManager.animation_path(run_dir, file)` | 拼接 animation 路径 |
| `RunManager.plot_path(run_dir, file)` | 拼接 plot 路径 |

输出结构：
```
results/<timestamp>_<tag>/
├── parameters.txt
├── checkpoints/   → mappo_best.pt, mappo_final.pt
├── trajectories/  → trajectory_*.npz, trajectory_*.txt
├── animations/    → animation_*.gif
└── plots/         → training_curves.png, training_data.npz
```

### 3. `src/env/` — 物理环境

**`asteroid.py`**
- `Asteroid(polyhedron_path)`: 加载 polyhedron mesh
- `compute_visible_faces(mdu_pos, cone_angle, range)`: 3 道门检测
  1. Range < 300m
  2. Cone angle < 40° (half-cone)
  3. Backface cull: `dot(normal_outward, to_mdu) > 0.1`
- 无真光线追踪遮挡

**`net_graph.py`**
- `NetGraph(fns_path, solution_path)`: 空间网拓扑
- 369 节点 / 432 边，邻接表，3D 坐标
- 邻接矩阵预计算

**`mdu_coverage_env.py`**
- `MDUCoverageEnv` — Gymnasium Env
- Observation (49 dims for 4 MDU):
  ```
  [self_pos(3), local_cov(3), global_cov(1),
   other_mdus_relative(9), graph_dist_to_others(3),
   visit(1), step(1),
   candidates(max_deg × 7)]
  ```
  - candidate = (global_pos(3), relative_pos(3), uncovered_value(1))
- Action: Discrete(max_deg=4) — 选邻居（无 stay）
- Reward: `20 × newly_covered_faces` + completion bonus
- Coverage: sticky（一旦覆盖永久标记）
- 预计算: 所有 369 节点的可见面片 + k-hop 邻居

### 4. `src/agents/mappo.py` — RL 算法

#### 类关系

```
MAPPO (顶层调度)
├── Actor (策略网络)
│   ├── body_net      Linear(21→64→64)
│   ├── fusion_net    Linear(128→64→64)  ← GNN 模式专用
│   ├── gru_cell      GRUCell(69→64)
│   ├── cand_net      Linear(7→32→32)    ← 权重共享
│   └── score_net     Linear(96→64→1)
│
├── Critic (价值网络)
│   └── net           Linear(78→64→64→1)
│
├── GNNEncoder (图编码器, 可选)
│   ├── node_pre      Linear(7→64→64)
│   ├── edge_pre      Linear(4→64)
│   ├── conv1         GraphSAGEConv(64,64→64)
│   └── conv2         GraphSAGEConv(64,64→64)
│
├── Buffer (轨迹存储)
└── Optimizers: Adam(Actor) + Adam(Critic)
```

---

#### Actor.forward() — 逐层数据流

```
输入: obs(B,49), hidden(B,64), prev_actions(B,4), cov_rate(B,1), [node_emb(B,64)]

┌─────────────────────────────────────────────────────────────────────┐
│ Layer 1: 拆分 Observation                                          │
│                                                                     │
│   obs (B, 49)                                                       │
│   ├── body (B, 21)       ← 前 21 维: pos, coverage, others, ...    │
│   └── candidates (B, 4×7) ← 后 28 维: 每个候选的 7 维特征            │
│                                                                     │
│   candidates → reshape → (B, max_deg=4, cand_feat_dim=7)            │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 2: Body Encoder                                               │
│                                                                     │
│   body (B, 21)                                                      │
│     → Linear(21, 64) → Tanh                                         │
│     → Linear(64, 64) → Tanh                                         │
│     → body_feats (B, 64)                                            │
│                                                                     │
│   [GNN 模式]                                                        │
│   if node_emb is not None:                                          │
│       body_feats (B, 64) + node_emb (B, 64) → cat → (B, 128)       │
│         → Linear(128, 64) → Tanh                                    │
│         → Linear(64, 64) → Tanh                                     │
│         → body_feats (B, 64)                                        │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 3: GRU 时序记忆                                               │
│                                                                     │
│   gru_input = cat([body_feats(B,64), prev_actions(B,4),            │
│                     cov_rate(B,1)], dim=-1)    ← (B, 69)            │
│                                                                     │
│   GRUCell(69→64):                                                   │
│     r = σ(W_ir·x + b_ir + W_hr·h + b_hr)     ← reset gate         │
│     z = σ(W_iz·x + b_iz + W_hz·h + b_hz)     ← update gate         │
│     n = tanh(W_in·x + b_in + r*(W_hn·h + b_hn)) ← new gate         │
│     h' = (1-z)*n + z*h                        ← output              │
│                                                                     │
│   hidden=0(B,64) ──→ GRUCell ──→ new_hidden(B,64)                  │
│       ↑                          │                                  │
│       └──────────────────────────┘ (下一步传入)                      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 4: Candidate Encoder (共享权重, 逐候选调用)                     │
│                                                                     │
│   candidates (B, 4, 7) → flatten → (B*4, 7)                        │
│     → Linear(7, 32) → Tanh                                          │
│     → Linear(32, 32) → Tanh                                         │
│     → cand_feats (B*4, 32) → reshape → (B, 4, 32)                  │
│                                                                     │
│   候选特征 (7 dims):                                                 │
│     [global_x, global_y, global_z,  ← 邻居绝对位置 / R               │
│      relative_x, relative_y, relative_z, ← 邻居相对位置 / R          │
│      uncovered_value]                ← 该节点可见但未覆盖的面占比      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 5: Score Network (逐候选打分)                                  │
│                                                                     │
│   for each candidate k in 1..4:                                     │
│       h (B, 64) → expand → (B, 4, 64)                              │
│       cat([h(B,4,64), cand_feats(B,4,32)], dim=-1) → (B, 4, 96)    │
│         → Linear(96, 64) → Tanh                                     │
│         → Linear(64, 1)                                             │
│         → scores (B, 4)  ← 每个候选一个 logit                        │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 6: Mask + Softmax                                             │
│                                                                     │
│   if mask is not None:                                              │
│       scores[~mask] = -inf   ← 无效邻居的 logit 置为 -∞             │
│                                                                     │
│   probs = softmax(scores, dim=-1)  ← (B, 4)                        │
│                                                                     │
│   Action 选择:                                                       │
│     随机: Categorical(probs).sample()                               │
│     Greedy: probs.argmax()                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
输出: probs(B,4), new_hidden(B,64)
```

---

#### Critic.forward() — 价值估计

```
输入: state_and_hidden (B, 78)

  [state (B, 14): 4个MDU位置(12) + 全局覆盖(1) + 步数(1)]
  || [GRU hiddens (B, 64)]
  ↓
  Linear(78, 64) → Tanh
  → Linear(64, 64) → Tanh
  → Linear(64, 1) → squeeze → V(B,)

[GNN 模式]
  Critic 输入 = 14 + 64 + 64 = 142 维
  Linear(142, 64) → ... → V(B,)
```

---

#### Buffer — 轨迹存储 + GAE 计算

```
Buffer 内部存储 (list of arrays):
┌──────────────────────────────────────────────────────────────┐
│ 属性                │ 形状       │ 说明                       │
│─────────────────────┼────────────┼───────────────────────────│
│ obs                 │ (T, A, 49) │ 每步每MDU的观测             │
│ state               │ (T, 14)    │ 全局状态                   │
│ actions             │ (T, A)     │ 动作索引                   │
│ rewards             │ (T,)       │ 即时奖励                   │
│ dones               │ (T,)       │ 终止标志                   │
│ values              │ (T, A)     │ Critic 估值                 │
│ log_probs           │ (T, A)     │ 旧策略 log prob             │
│ action_masks        │ (T, A, 4)  │ 有效动作掩码               │
│ graph_node_feats    │ (T, 369, 7)│ GNN 节点特征 (可选)         │
│ graph_edge_index    │ (2, 864)   │ 有向边索引 (静态, 存一次)   │
│ graph_edge_feats    │ (864, 4)   │ 边特征 (静态, 存一次)      │
└──────────────────────────────────────────────────────────────┘

compute_gae(γ=0.99, λ=0.95):
┌──────────────────────────────────────────────────────────────┐
│  输入: 完整轨迹 (T 步 × A 个 MDU)                              │
│                                                              │
│  for t = T-1, T-2, ..., 0:                                   │
│      next_V = V[t+1].mean() if t < T-1 else 0                │
│      δ[t] = r[t] + γ × next_V × (1-done[t]) - V[t].mean()    │
│      A[t] = δ[t] + γλ × (1-done[t]) × A[t+1]                 │
│                                                              │
│  return[t] = A[t] + V[t]                                     │
│  A = z_score_normalize(A)                                    │
│  return = z_score_normalize(return)                           │
└──────────────────────────────────────────────────────────────┘
```

---

#### MAPPO.update() — PPO + BPTT 完整流程

```
输入: Buffer 中的完整轨迹 (T 步)

┌─────────────────────────────────────────────────────────────────────┐
│ Phase 0: GAE 计算                                                   │
│                                                                     │
│   data = Buffer.compute_gae()                                       │
│   → {obs(T,A,49), state(T,14), actions(T,A),                       │
│      log_probs(T,A), adv(T,A), ret(T,A), masks, graph}              │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 1: GPU 数据传输                                                │
│                                                                     │
│   obs_t, act_t, lp_t, adv_t, ret_t, state_t → torch.Tensor.to(GPU) |
│   return = z_score_normalize(return)                                 │
│                                                                     │
│   [GNN 模式] 预编码图嵌入 (detached)                                  │
│   for t in 0..T-1:                                                  │
│       gnn_embs[t] = GNN(node_feats[t], edges, edge_feats)  no_grad  │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 2: PPO Epochs 循环 (× num_epochs)                             │
│                                                                     │
│   for epoch in 1..4:                                                │
│                                                                     │
│       hidden = zeros(A, 64)          ← GRU 初始                     │
│       prev_actions = zeros(A, 4)     ← 动作 onehot                   │
│                                                                     │
│       for chunk_start in 0, bptt_len, 2*bptt_len, ...:              │
│           chunk_end = min(chunk_start + bptt_len, T)                │
│                                                                     │
│           for t in chunk_start .. chunk_end-1:                      │
│                                                                     │
│               # ── Forward ──                                       │
│               probs, hidden = Actor(obs[t], hidden,                 │
│                                     prev_actions, cov_rate,         │
│                                     [gnn_embs[t]])                  │
│               new_lp = log_prob(probs, actions[t])                  │
│               entropy = Categorical(probs).entropy().mean()          │
│                                                                     │
│               # ── PPO Clipped Loss ──                              │
│               ratio = exp(new_lp - old_lp[t])                       │
│               L1 = ratio × adv[t]                                   │
│               L2 = clamp(ratio, 1-ε, 1+ε) × adv[t]                 │
│               L_actor = -min(L1, L2).mean() - ent_coef × entropy   │
│                                                                     │
│               # ── Critic Loss ──                                   │
│               V_pred = Critic(state[t], hidden)                     │
│               L_critic = MSE(V_pred, return[t])                     │
│                                                                     │
│               # ── Backward ──                                      │
│               is_last = (t == chunk_end - 1)                        │
│               (L_actor + L_critic).backward(                        │
│                   retain_graph = NOT is_last                        │
│               )                                                     │
│                                                                     │
│               prev_actions = one_hot(actions[t])                    │
│                                                                     │
│           # ── Chunk 结束 ──                                        │
│           clip_grad_norm(actor_params + critic_params, 0.5)         │
│           optim_a.step()    ← 更新 Actor + GRU                      │
│           optim_c.step()    ← 更新 Critic                           │
│           optim_a.zero_grad()                                        │
│           optim_c.zero_grad()                                        │
│           hidden = hidden.detach()       ← 切断跨 chunk 梯度         │
│           prev_actions = prev_actions.detach()                       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 3: 返回统计                                                    │
│                                                                     │
│   return {                                                           │
│       loss_a:     平均 Actor loss,                                   │
│       loss_c:     平均 Critic loss,                                  │
│       entropy:    平均策略熵,                                        │
│       hidden_norm: 平均 GRU 隐藏状态 L2 范数,                         │
│       grad_norm:  平均梯度范数                                       │
│   }                                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

#### 关键设计点

| 设计 | 原因 |
|------|------|
| **Sequential BPTT** (不 shuffle) | GRU 的状态依赖时间顺序，打乱会破坏时序关系 |
| **retain_graph=True** (chunk 内) | 同一个 chunk 内多个 step 共享 GRU 的计算图 |
| **hidden.detach()** (chunk 间) | 防止跨 chunk 的计算图爆炸 |
| **z-score return normalization** | 稳定 Critic 训练，避免不同 episode 的 return 尺度差异 |
| **cosine entropy schedule** | 前期保持高探索(0.05)，后期逐渐降低(0.01)，不坍缩 |
| **mask → -inf** | 无效动作（被占用节点）的概率强制为 0 |
| **Critic 看 hidden** | Critic 通过 GRU 状态推断 MDU 的轨迹历史 |

---

### 5. `src/agents/gnn.py` — 图编码器（可选）

#### 架构总览

```
                空间网图: 369 节点, 432 无向边 → 864 有向边

    输入                              GNN Encoder                    输出
    ────                              ───────────                    ────
                                                                      
    node_feats[N×7] ──→ node_pre ──→ h⁰[N×64]                       
      pos(3)/R           7→64→64     │                                node
      log_deg(1)                     │    conv1         conv2         embeddings
      occupied(1)                    ├──→ SAGE ──→ReLU──→SAGE──→ReLU──→ [N×64]
      local_cov(1)                   │   64→64         64→64          每个节点
      visit_norm(1)                  │    │              │            的图感知
                                     │    │              │            嵌入向量
    edge_feats[E×4] ──→ edge_pre ────┘    │              │
      edge_vec(3)/R   4→64                │              │
      edge_len(1)                    scatter_mean   scatter_mean
                                     (邻居聚合)     (邻居聚合)
```

---

#### GraphSAGEConv — 单层消息传递

```
输入: x(N, 64), edge_index(2, 864), edge_attr(864, 64)

┌──────────────────────────────────────────────────────────────┐
│ Step 1: Self-transformation                                   │
│                                                              │
│   self = W_self @ x   ← Linear(64, 64, bias=False)          │
│   → (N, 64)                                                   │
│                                                              │
│ Step 2: Neighbor message construction                         │
│                                                              │
│   src_nodes = edge_index[0]        ← (864,)  每条边的源节点   │
│   dst_nodes = edge_index[1]        ← (864,)  每条边的目标节点 │
│                                                              │
│   src_feats = x[src_nodes]         ← (864, 64)               │
│   msg = cat([src_feats, edge_attr], dim=-1)  ← (864, 128)    │
│   msg = W_neigh @ msg              ← Linear(128, 64)         │
│   → (864, 64)                                                 │
│                                                              │
│ Step 3: Mean aggregation per destination                      │
│                                                              │
│   out = zeros(N, 64)                                          │
│   for each edge: out[dst] += msg[edge]                        │
│   out = out / degree[dst]        ← 按入度归一化               │
│                                                              │
│   实际实现用 scatter_add_:                                    │
│     out.scatter_add_(0, dst_idx, msg)                        │
│     count.scatter_add_(0, dst_idx, ones)                     │
│     out = out / count.clamp(min=1)                           │
│                                                              │
│ Step 4: Combine                                               │
│                                                              │
│   h' = self + out_neigh          ← (N, 64)                   │
│                                                              │
│   即: h'_v = W_self·h_v + mean_{u∈N(v)} W_neigh·[h_u || e_uv]│
└──────────────────────────────────────────────────────────────┘
```

---

#### GNNEncoder — 完整前向传播

```
GNNEncoder.forward(node_feats, edge_index, edge_feats):

┌──────────────────────────────────────────────────────────────┐
│ Step 1: 特征预处理                                            │
│                                                              │
│   h = node_pre(node_feats)                                    │
│     → Linear(7, 64) → ReLU                                   │
│     → Linear(64, 64) → ReLU                                  │
│     → h (N, 64)                                               │
│                                                              │
│   e = edge_pre(edge_feats)                                    │
│     → Linear(4, 64) → ReLU                                   │
│     → e (E, 64)                                               │
│                                                              │
│ Step 2: 第一层消息传递 (1-hop 邻域信息)                        │
│                                                              │
│   h = conv1(h, edge_index, e)                                 │
│     = W_self1·h + mean_{u∈N(v)} W_neigh1·[h_u || e]         │
│   h = ReLU(h)    ← (N, 64)                                   │
│                                                              │
│   此时每个节点嵌入了其直接邻居的信息                            │
│                                                              │
│ Step 3: 第二层消息传递 (2-hop 邻域信息)                        │
│                                                              │
│   h = conv2(h, edge_index, e)                                 │
│     = W_self2·h + mean_{u∈N(v)} W_neigh2·[h_u || e]         │
│   h = ReLU(h)    ← (N, 64)                                   │
│                                                              │
│   此时每个节点嵌入了 2-hop 邻域信息                             │
│   (因为 conv1 输出含 1-hop 信息，conv2 聚合后 = 2-hop)        │
│                                                              │
│ Step 4: 输出 node embeddings                                  │
│                                                              │
│   return h  ← (N, 64)                                        │
│   每个节点一个 64 维嵌入，携带局部图拓扑信息                    │
└──────────────────────────────────────────────────────────────┘
```

---

#### 节点特征构建 (7 dims)

```
由 MDUCoverageEnv.get_graph_features() 每步动态构建:

┌──────────┬──────┬──────────────────────┬─────────────┐
│  索引    │ 维度 │ 内容                 │ 来源         │
│──────────┼──────┼──────────────────────┼─────────────│
│  0:3     │  3   │ 节点 3D 位置 / R     │ 静态预计算   │
│  3:4     │  1   │ log(1 + degree)      │ 静态预计算   │
│  4:5     │  1   │ 被 MDU 占用标志       │ 动态 (每步)  │
│  5:6     │  1   │ 节点本地覆盖率        │ 动态 (每步)  │
│  6:7     │  1   │ 节点访问频次 / max   │ 动态 (每步)  │
└──────────┴──────┴──────────────────────┴─────────────┘
```

#### 边特征构建 (4 dims)

```
        src ────────→ dst

┌──────────┬──────┬──────────────────────────┐
│  0:3     │  3   │ 边向量 (dst_pos - src_pos) / R │
│  3:4     │  1   │ 边长度 / R                       │
└──────────┴──────┴──────────────────────────┘
```

---

#### GNN 在 MAPPO 中的集成方式

```
当前状态: use_gnn=False, GNN 未启用, 但代码路径完整

启用后的数据流 (use_gnn=True):

Rollout 阶段 (agent.act):
┌──────────────────────────────────────────────────────────────┐
│  env.get_graph_features()                                     │
│      ↓                                                       │
│  GNN.forward(node_feats, edge_idx, edge_feats)  ← @no_grad  │
│      ↓                                                       │
│  all_embs (369, 64)  →  取 MDU 所在节点的嵌入  → node_emb(4,64) │
│      ↓                                                       │
│  Actor.forward(obs, node_emb=node_emb)                       │
│      ↓                                                       │
│  body_feats + node_emb → fusion_net → 融合特征                │
│      ↓                                                       │
│  Critic(state, hidden, gnn_mean) ← GNN 嵌入均值作为额外上下文  │
└──────────────────────────────────────────────────────────────┘

Update 阶段 (agent.update):
┌──────────────────────────────────────────────────────────────┐
│  for t in 0..T-1:                                            │
│      GNN.forward(node_feats[t], ...)  ← @no_grad, detached  │
│      → gnn_embs[t] (369, 64)                                 │
│                                                              │
│  PPO BPTT 循环中:                                             │
│      nemb_t = gnn_embs[t][mdu_node_indices]  ← (4, 64)      │
│      Actor(obs[t], node_emb=nemb_t)  ← 直接用, GNN 不参与梯度 │
└──────────────────────────────────────────────────────────────┘

设计意图:
  - GNN 提供图拓扑感知 (节点在网中的位置、连通性)
  - Actor 通过 fusion_net 学会如何利用图结构信息
  - Critic 通过 GNN 嵌入均值了解 MDU 的图空间分布
  - GNN 不参与 PPO 梯度更新 (detached) ← 当前简化策略
  - 未来: 添加辅助损失训练 GNN (预测覆盖价值)
```

---

#### scatter_mean 实现

```
def scatter_mean(src, index, dim_size):
    """
    src:   (E, D)    每条边的消息
    index: (E,)      每条边的目标节点
    → out: (N, D)    每个节点的聚合均值
    
    等价于 PyTorch Geometric 的 scatter_mean
    """
    out = zeros(N, D)           # 输出缓冲区
    count = zeros(N, 1)         # 归一化计数
    
    # 逐边累加到目标节点
    out.scatter_add_(0, index.unsqueeze(-1).expand(-1, D), src)
    count.scatter_add_(0, index.unsqueeze(-1), ones(E, 1))
    
    return out / count.clamp(min=1)
    
    示例 (4 节点, 6 条边):
      edge_index = [[0,0,1,2,2,3], [1,2,2,1,3,2]]
      src 消息指向: dst=1(来自0), dst=2(来自0,1,3), dst=1(来自2), dst=2(来自2)
      聚合: node[1] = mean(msg[0], msg[2])
            node[2] = mean(msg[1], msg[3], msg[5])
```

---

## 辅助脚本

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `train.py` | 主训练 | Config + CLI | run 目录 |
| `generate_trajectory.py` | 用训练好的模型生成轨迹 | `--tag <tag>` → 自动找 checkpoint | `trajectory_*.npz/txt` |
| `visualize_trajectory.py` | 渲染轨迹为 GIF 动画 | `--tag <tag>` → 自动找 trajectory | `animation_*.gif` |
| `compare_runs.py` | 对比两个训练 run 的曲线 | `--tag a --tag2 b` | `comparison.png` |
| `run_overnight.py` | 过夜训练（简化版） | Config + CLI | run 目录 + log.txt |
| `render_animation.py` | 独立渲染器（遗留） | 硬编码路径 | GIF |
| `test_env_smoke.py` | 环境冒烟测试 | - | PASS/FAIL |

---

## 常用命令

```bash
conda activate comm_python_env
set KMP_DUPLICATE_LIB_OK=TRUE

# 训练
python src/train.py --mdus 4 --episodes 500 --save-plot --tag 4mdu

# 生成轨迹 + 动画（自动找到对应 run）
python src/generate_trajectory.py --mode mappo --tag 4mdu
python src/visualize_trajectory.py --tag 4mdu

# 对比两个 run
python compare_runs.py --tag baseline --tag2 gnn

# 冒烟测试
python tests/test_env_smoke.py
```

## 关键设计决策

1. **No stay action**: MDU 必须移动，防止策略坍缩
2. **Physical-candidate Actor**: 动作按物理位置打分，不是按索引
3. **GRU temporal memory**: 防止 MDU 在 2-3 个节点间振荡
4. **Sticky coverage**: 面片一旦覆盖永不丢失
5. **Per-face reward**: 按面片数奖励，不是按面积
6. **Completion bonus**: 首次达到 75% 覆盖时给予时间奖励
7. **Single config source**: `src/config.py` 是唯一参数来源
8. **Unified path management**: `src/run_manager.py` 是唯一路径入口
