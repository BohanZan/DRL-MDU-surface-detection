# SNN 时序轨迹规划 — 文献综述与实现方案（中文版）

> 收集日期：2026-06-10 | 来源：arXiv API | 项目：小行星绳网表面覆盖 DRL

---

## 一、核心论文（按相关度排列）

### 第一梯队：SNN 路径规划与控制（最相关）

**1. [2404.15524] Espino et al. 2024 — "基于SNN Wavefront Planner的移动机器人路径规划"**
- 链接：https://arxiv.org/abs/2404.15524
- **方法：** SNN Wavefront Planner + E-prop 在线学习
- **核心贡献：** 用SNN脉冲活动编码遍历代价，同时建图与规划路径
- **学习规则：** E-prop（基于资格迹的在线学习，适合神经形态芯片部署）
- **对我们的意义：** ⭐⭐⭐⭐⭐ 最接近的工作——SNN在图/网格上的路径规划

**2. [2010.09635] Tang et al. 2020 — "基于群体编码SNN的深度强化学习连续控制"**
- 链接：https://arxiv.org/abs/2010.09635
- **方法：** 群体编码（Population Coding）将连续值映射为脉冲序列
- **核心贡献：** 首次证明 SNN + DRL 可以解决连续机器人控制任务
- **学习：** 兼容 surrogate gradient + PPO 训练流程
- **对我们的意义：** ⭐⭐⭐⭐⭐ 验证了SNN+PPO管线可行

**3. [2310.02361] Wang et al. 2023 — "事件增强多模态SNN动态避障"**
- 链接：https://arxiv.org/abs/2310.02361
- **方法：** 事件相机 + SNN + 深度强化学习实现避障导航
- **核心贡献：** 多模态融合（事件流+传统帧）与脉冲神经元结合
- **对我们的意义：** ⭐⭐⭐⭐ SNN + DRL 导航的成功案例

**4. [2501.17172] Casanueva-Morato et al. 2025 — "基于SNN的轨迹插值闭环控制"**
- 链接：https://arxiv.org/abs/2501.17172
- **方法：** 移位WTA电路实现脉冲轨迹插值
- **核心贡献：** 闭环神经形态控制机械臂，纯模拟硬件实现
- **对我们的意义：** ⭐⭐⭐ 展示了SNN闭环控制的硬件可行性

### 第二梯队：SNN 训练方法与理论

**5. [2406.19645] Li et al. 2024 — "稀疏surrogate gradient直接训练时序SNN"**
- 链接：https://arxiv.org/abs/2406.19645
- **方法：** 仅在需要处激活的稀疏surrogate gradient
- **意义：** ⭐⭐⭐⭐ 降低训练计算量而不损失精度，直接适用于我们的PPO训练

**6. [2201.10879] Suetake et al. 2022 — "S3NN：基于Surrogate Gradient的时间步缩减"**
- 链接：https://arxiv.org/abs/2201.10879
- **方法：** 通过surrogate gradient优化减少编码时间步
- **意义：** ⭐⭐⭐⭐ 可将 T_enc=5 缩减到 T_enc=1-2，大幅提升推理速度

**7. [2603.13478] Dold 2026 — "单脉冲 vs 多脉冲：哪种更好？"**
- 链接：https://arxiv.org/abs/2603.13478
- **核心结论：** 单脉冲和多脉冲编码在计算上等价，多脉冲单神经元表达能力更强但不会改变网络整体容量
- **意义：** ⭐⭐⭐ 理论基石，指导SNN架构设计

**8. [2504.14015] Dold 2025 — "Causal Pieces：分析和改进SNN的概念框架"**
- 链接：https://arxiv.org/abs/2504.14015
- **方法：** 因果分析框架识别SNN中哪些神经元/脉冲对决策有贡献
- **意义：** ⭐⭐⭐ 可用于调试和分析SNN策略的可解释性

### 第三梯队：神经元模型变体

**9. [2402.04663] Huang et al. 2024 — "CLIF：互补LIF神经元"**
- 链接：https://arxiv.org/abs/2402.04663
- **方法：** 双通路LIF（兴奋+抑制）
- **意义：** ⭐⭐⭐ 时序特征提取能力优于标准LIF

**10. [2210.13768] Yao et al. 2022 — "GLIF：门控LIF神经元"**
- 链接：https://arxiv.org/abs/2210.13768
- **方法：** LIF上加门控机制（类似GRU之于RNN）
- **意义：** ⭐⭐⭐ 表达能力更强，计算开销增加极小

### 第四梯队：空间应用与神经形态硬件

**11. [2501.02916] Courtois et al. 2025 — "基于脉冲单目事件的航天器6D位姿估计"**
- 链接：https://arxiv.org/abs/2501.02916
- **方法：** SNN + 事件相机实现航天器位姿估计
- **意义：** ⭐⭐⭐⭐ 证明SNN在空间应用中的可行性（在轨服务、碎片清除）

**12. [2506.14138] Gautam et al. 2025 — "NeuroCoreX：开源FPGA脉冲神经网络模拟器"**
- 链接：https://arxiv.org/abs/2506.14138
- **方法：** FPGA上的SNN仿真器，支持片上学习
- **意义：** ⭐⭐⭐ 硬件部署参考方案

**13. [2006.09985] Massa et al. 2020 — "基于DVS相机的手势识别在Loihi上的SNN实现"**
- **方法：** 在Intel Loihi神经形态芯片上部署SNN
- **意义：** ⭐⭐ SNN在专用芯片上的部署验证

### 第五梯队：多智能体与图网络

**14. [2509.05397] Lai et al. 2025 — "RoboBallet：基于GNN+RL的多机器人运动规划"**
- 链接：https://arxiv.org/abs/2509.05397
- **方法：** GNN + RL 实现多机器人协调
- **意义：** ⭐⭐⭐ 非SNN，但GNN部分可为我们的图拓扑建模提供参考

**15. [2006.15482] Huang & Liu 2020 — "异构多机器人系统的内部注意力建模"**
- **方法：** 注意力机制用于多机器人团队协作
- **意义：** ⭐⭐ 注意力机制可为多MDU协调提供思路

---

## 二、现有SNN库对比

| 库 | PyPI包名 | 特点 | 推荐度 |
|---|---------|------|--------|
| **snnTorch** | `snntorch` | 完整LIF/RLIF实现、surrogate gradient、GPU加速、文档完善 | ⭐⭐⭐⭐⭐ |
| Norse | `norse` | LIF、 surrogate gradient、与PyTorch Lightning兼容 | ⭐⭐⭐ |
| PySNN | `pysnn` | 轻量但维护不活跃 | ⭐⭐ |

**推荐：snnTorch** — 它是我们GRU→SNN替换的首选库。

### snnTorch 核心特性

```python
import snntorch as snn
from snntorch import surrogate

# LIF神经元（替代GRU）
lif = snn.Leaky(
    beta=0.5,                    # 膜电位时间常数
    learn_beta=True,             # 可学习衰减率
    spike_grad=surrogate.fast_sigmoid(slope=25),  # surrogate gradient
)

# 使用方式
spk, mem = lif(current, mem)  # spk=脉冲(0/1), mem=膜电位
```

---

## 三、实现方案（假设SNN能跑通）

### 3.1 总体策略

**GPU上训练 → 神经形态芯片部署**

```
训练阶段 (GPU + snnTorch):
  PyTorch + surrogate gradient + PPO → 训练SNN策略

部署阶段 (FPGA / Loihi):
  纯推理 + 片上自适应（STDP / E-prop）
```

### 3.2 网络架构

```
观测(46维)
    │
    ▼
┌──────────────────────────────┐
│  脉冲编码器 (Rate Coding)      │
│  连续值 → T_enc步脉冲序列      │
│  T_enc = 5 (可缩减优化)        │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  SNN 体网络 (LIF神经元)        │
│  输入: encoded_obs(46)→64     │
│  膜电位在时间步间传递作为记忆   │
│  (替代当前GRU)                │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  候选编码器 + 评分器 (ANN)    │
│  与当前架构一致                │
│  使用膜电位作为"body features" │
└──────────────────────────────┘
    │
    ▼
  动作 (4选1, 离散)
```

### 3.3 ANN到SNN的迁移策略

**关键技巧：** 先训练ANN权重，再转换到SNN

```
Step 1: 用当前GRU架构训练一个高性能策略（已做，88.04%）
Step 2: 复制权重到SNN（fc层结构相同）
Step 3: 用surrogate gradient微调50-100 episodes
Step 4: 部署为纯SNN（去掉surrogate）
```

这样避免了SNN从零训练的困难，利用了已有的高性能权重。

### 3.4 训练超参数

| 参数 | 当前(GRU) | SNN | 原因 |
|------|----------|-----|------|
| 隐藏层维度 | 64 | 64 | 相同容量 |
| 时间编码 | 1步/环境步 | T_enc=5 | SNN需要时间积分 |
| 学习率 | 3e-4 | 1e-4 | Surrogate gradient更嘈杂 |
| 熵系数 | 0.05→0.001 | 0.05→0.001 | 相同探索退火 |
| 截断BPTT | 16步 | 16×T_enc | 更长因内部时间步 |

### 3.5 代码改动量预估

| 文件 | 改动内容 | 代码量 |
|------|---------|--------|
| `src/agents/snn_actor.py` | 新建：SNN Actor类 | ~150行 |
| `src/agents/mappo.py` | 增加SNN选项，保持GRU兼容 | ~50行 |
| `src/train.py` | 增加`--snn`参数开关 | ~10行 |
| 测试脚本 | SNN vs GRU对比测试 | ~50行 |
| **总计** | | **~260行** |

### 3.6 实验验证路线

```
Phase 1: snnTorch安装验证
  → pip install snntorch + 跑MNIST示例

Phase 2: SNN Actor实现 + 5 episodes Quick Test
  → 替换GRU→LIF，检查覆盖率、脉冲发放率、膜电位

Phase 3: 完整训练200 episodes
  → 对比GRU vs SNN的收敛曲线、最终覆盖率

Phase 4: T_enc优化
  → 5→4→3→2→1步，找最小有效时间步

Phase 5: ANN→SNN权重迁移
  → 加载GRU权重，微调，部署
```

---

## 四、SNN vs GRU 对比总结

| 维度 | GRU（当前） | SNN（目标） | 差异 |
|------|-----------|------------|------|
| 记忆机制 | 隐藏状态向量 | LIF膜电位 | 膜电位可解释性更好 |
| 参数量 | ~38K | ~39K | 几乎相同 |
| GPU训练速度 | 17s/ep | 预计 30-50s/ep | 慢2-3倍（仿真开销） |
| 推理能耗 | ~10W（GPU） | ~10mW（FPGA） | 低1000倍 ⚡ |
| 空间平台适用性 | ❌ 功耗太高 | ✅ 航天级FPGA可行 | **关键优势** |
| 收敛性 | ✅ 88.04% | 待验证 | 预期持平或更好 |
| 粒子辐射耐受 | ❌ 未优化 | ✅ 脉冲信号天然鲁棒 | 理论优势 |

---

## 五、参考文献

1. Espino et al. (2024). "A Rapid Adapting and Continual Learning SNN Path Planning Algorithm for Mobile Robots." arXiv:2404.15524.
2. Tang et al. (2020). "Deep RL with Population-Coded SNN for Continuous Control." arXiv:2010.09635.
3. Wang et al. (2023). "Event-Enhanced Multi-Modal SNN for Dynamic Obstacle Avoidance." arXiv:2310.02361.
4. Casanueva-Morato et al. (2025). "Towards spiking analog hardware trajectory interpolation." arXiv:2501.17172.
5. Li et al. (2024). "Directly Training Temporal SNN with Sparse Surrogate Gradient." arXiv:2406.19645.
6. Suetake et al. (2022). "S3NN: Time Step Reduction of Spiking Surrogate Gradients." arXiv:2201.10879.
7. Dold (2026). "One spike vs multiple spikes." arXiv:2603.13478.
8. Dold (2025). "Causal pieces for SNN." arXiv:2504.14015.
9. Huang et al. (2024). "CLIF: Complementary LIF Neuron." arXiv:2402.04663.
10. Yao et al. (2022). "GLIF: Gated LIF Neuron." arXiv:2210.13768.
11. Courtois et al. (2025). "Spiking monocular event based 6D pose for space." arXiv:2501.02916.
12. Gautam et al. (2025). "NeuroCoreX: FPGA SNN Emulator." arXiv:2506.14138.
13. Massa et al. (2020). "SNN gesture recognition on Loihi." arXiv:2006.09985.
14. Lai et al. (2025). "RoboBallet: GNN+RL multi-robot." arXiv:2509.05397.
15. Huang & Liu (2020). "Robot Inner Attention for Multi-Robot." arXiv:2006.15482.
16. Suetake et al. (2022). "S3NN: Time Step Reduction." arXiv:2201.10879.
