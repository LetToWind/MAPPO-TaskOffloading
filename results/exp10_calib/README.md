# exp10_calib — Phase 0' 场景校准(forced-local / oracle / VoI 三门槛)

## 场景(动态卸载 v1,校准后参数)
- 热点泊松到达:λ0=0.15,λ_hot=1.5(热点每 ~8 步马尔可夫漂移),CH=3,
  截止期 U[5,11] 步,数据量 U[1e5,3e5],episode 30 步(到达止于 25),
  holding cost 0.02,路由不可撤回,转发 +1 步在途;
- 无线电模型(C-RAN 抽象):车辆上传永远在原 BS 接收,路由转移服务责任
  (队列+频谱),干扰按各接收方 BS 的同频能量计算——修复了"转发后
  无线电链路被 500m 地板压死"的结构缺陷;
- 校准探针结果(oracle 调度):自由路由 76.7% vs 强制本地 50.8%
  (gap 25.9 个百分点),自由率落在 [60,80] 目标区。

## 臂设计(各 5 种子,UPDATE_EVERY=25,3000 episodes)
| 臂 | 设定 | 测什么 |
|---|---|---|
| fl_s* | 路由头 mask 仅本地 | Level 0(全本地)训练上限 |
| none_s* | 自由路由,邻居负载实时 | 可学习的 Level 2 |
| frozen_s* | 自由路由,邻居负载冻结在 episode 开头 | 惯性上限(VoI 下界) |
| oracle | JSQ 贪心(真值负载,已跑:success 43.57/56.9≈76.6%) | 上界参照 |

## 门槛(analyze_exp10.py 自动判定)
- G1 可玩性:oracle − fl ≥ 15 任务/episode;
- G2 VoI:none − frozen 显著(p<0.05);
- G3 可学习性:none ≥ fl + 0.5×(oracle − fl)。

## 运行
```
conda activate py37tf
cd C:\Users\zengd\Desktop\论文\W-MADDPG\MADDPG_Discrete
python scripts/run_experiments.py --group exp10_calib --max-parallel 8
python scripts/analyze_exp10.py
```
15 个 run(8 并行,约 3-4 小时);oracle 已写入 oracle_result.csv。

## 结果
(待运行后填写;若 G1 失败 → 跑 exp10b_hotspot 扫 λ_hot;
G2 失败 → 邻居负载信息不具决策价值,需再审视观测;G3 失败 →
按 §9.0 升级梯:熵 → 课程 → oracle 模仿预热)
