# exp7_varfix — 方差修复验证:UPDATE_EVERY=25 跨 episode 累积更新

## 诊断(为什么原训练方差大)

1. **micro-batch PPO(根源)**:episode 仅 5-10 步,MINIBATCH_SIZE=64 > 样本数
   → 每次更新只有 ~8 个样本,梯度估计被噪声支配;
2. **逐 episode 归一化放大噪声**:在 8 个样本上做 advantage/returns 标准化,
   把纯噪声拉伸为单位方差的满幅梯度;
3. **策略随机游走**:success(阈值型)不灵敏,reward(连续型)随训练越漂越散,
   无 LR 衰减,不收敛;
4. 团队奖励 + 共享 critic → 5 个 actor 拿到相同 advantage(次要)。

## 修复

`train_mappo_phase0.py` 新增 `UPDATE_EVERY`(默认 1 = 原行为):
每 N 个 episode 累积(~8N 样本)后再做 PPO 更新,归一化改在累积批次上;
GAE 仍按 episode 分段计算(边界正确)。本组 UPDATE_EVERY=25(~200 样本/更新)。

## 结果(CH=3, none, 4000 eps, 10 种子,贪心评估末 5 点)

| 设定 | reward | success | 种子 reward 范围 |
|---|---|---|---|
| UPDATE_EVERY=1(旧) | 5.29 ± 11.58(CV 219%) | 13.10 ± 1.18 | −9.0 ~ +22.8 |
| **UPDATE_EVERY=25(新)** | **41.70 ± 5.74(CV 14%)** | **16.80 ± 0.50** | 30.4 ~ 49.8 |

- 方差比 4.07(Welch t 检验 p<0.0001);
- **不止方差降了:均值从 5.3 → 41.7,success 13.1 → 16.8**——之前 CH=3 下
  MAPPO 基本没学起来,所谓"拥塞导致训练双峰"大部分是 micro-batch 破坏训练;
- 修复后 CH=3(success 16.8)接近 CH=6 旧值(18.7)。

## 影响

- 后续所有实验改用 UPDATE_EVERY=25;
- 此前 exp3/exp4 的"延迟无效应"结论需在稳定基线上重测(exp8):
  之前策略太差、可能根本没利用干扰观测,延迟无损不代表信息不重要;
- 运行:`python scripts/run_experiments.py --group exp7_varfix`;
  对比:`python scripts/compare_varfix.py`。
