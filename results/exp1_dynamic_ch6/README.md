# ⚠️ 本组数据已弃用(2026-09-21)

原因:逐 episode micro-batch PPO 训练器缺陷(每次更新仅 ~8 样本,噪声支配梯度),策略未真正学起。详见 results/实验数据状态总览.md 与 results/exp7_varfix/README.md。

---

# exp1_dynamic_ch6 — 第一轮:动态拓扑延迟敏感性(已废弃的初步实验)

## 条件
- 拓扑:动态(每 episode 随机生成,seed = 1 + episode_count),CH=6
- 观测:34 维,**干扰维度未做 log 归一化(原始量纲)**——注意与 exp2/exp3 不同
- 臂:none / fixed_d1 / fixed_d2 / fixed_d3 / unfixed_0-2 / unfixed_1-3,各 1 个种子
- 训练:4000 episodes,无独立贪心评估(以训练 reward 末段 1000 eps 均值比较)

## 结论(为何废弃)
所有延迟臂 reward ≥ none 臂(fixed_d3 甚至 +41%)。
原因:①单种子噪声淹没(reward std≈25);②动态拓扑下学习本身偏弱;
③干扰观测原始量纲 1e-8~1e-3,网络实际未使用该特征。

## 后续
exp2 起干扰观测改为 log 归一化、加固定拓扑与贪心评估、3 种子。


