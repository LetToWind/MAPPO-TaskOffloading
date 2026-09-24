# ⚠️ 本组数据已弃用(2026-09-21)

原因:逐 episode micro-batch PPO 训练器缺陷(每次更新仅 ~8 样本,噪声支配梯度),策略未真正学起。详见 results/实验数据状态总览.md 与 results/exp7_varfix/README.md。

---

# exp2_fixed_ch6 — 第二轮:固定拓扑 6 信道(资源宽裕,延迟无效应)

## 条件
- 拓扑:固定 variant 25(场景经种子 20240921 确定化,所有臂/种子完全同场景)
- 信道:CH=6(默认,资源宽裕)
- 观测:34 维,干扰 log 归一化 (log10(x)+8)/6
- 臂:none / fixed_d1 / fixed_d3 / unfixed_1-3,各 3 种子(s1-s3)
- 训练:3000 episodes;每 100 eps 做 20 episode 贪心评估(_eval.csv)
- 指标:eval 末 5 个评估点平均

## 结论
延迟无退化:none 61.73±17.42,fixed_d3 72.16(+16.9%,噪声内),unfixed_1-3 66.64(+8.0%)。
根因:**干扰信号 episode 内高度自相关**(任务静态绑定 + 信道分配收敛后基本不变),
z(t-d) ≈ z(t);且 6 信道下错选信道代价低 → 延迟信息非决策关键。
该结果与 exp3 一起构成 2×2 因子设计的"低拥塞"侧。

## 文件
- phase0_fx_{arm}_s{1,2,3}.csv — 训练日志(episode, sum_reward, success, fail, mean_delay)
- phase0_fx_{arm}_s{1,2,3}_eval.csv — 贪心评估(episode, eval_reward, eval_success, eval_fail)


