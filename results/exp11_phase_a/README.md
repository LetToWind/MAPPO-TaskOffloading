# exp11 Phase A — v1.1 无训练因果校准

## 结论

Phase A 已通过。最终配置在 400 条共同随机轨迹上同时满足 A1~A5：

```text
n_hotspots=2
lambda0=0.10
lambda_hot=0.60
hotspot_dwell=10
capacity_markov=true
capacity_stay=0.85
hotspot_capacity_penalty=1
deadline=U[5,11]
data=U[1e5,3e5]
channels=3
```

| 路由信息臂 | success rate | reward |
|---|---:|---:|
| forced-local | 47.00% | -9.54 |
| live | 64.33% | 61.69 |
| frozen | 49.54% | 5.44 |
| fixed d=1 | 60.57% | 48.45 |
| fixed d=2 | 57.69% | 36.42 |
| fixed d=3 | 55.75% | 28.81 |
| unfixed [1,3] | 57.92% | 37.71 |

门槛：

- A1 路由空间：`live - local = +17.33 pp`，通过；
- A2 信息价值：`live - frozen = +14.79 pp`，bootstrap 95% CI
  `[13.55, 16.03]`，通过；
- A3 延迟致病：fixed d=2 相对下降 10.3%，通过；
- A4 剂量关系：`live > d1 > d2 > d3`，通过；
- A5 非饱和：live success 64.33%，通过。

所有臂逐 episode 使用相同环境 seed；400 个 episode 的到达任务数逐臂完全一致，
不存在外生轨迹不匹配。

## 文件

- `scan_ranking.csv`：第一轮 24 组参数筛选；
- `../exp11_phase_a_final_c085/validation_episodes.csv`：最终 400 条逐 episode 数据；
- `../exp11_phase_a_final_c085/validation_summary.json`：最终汇总；
- `scripts/calibrate_v11.py`：可复现实验入口。

下一阶段为 Phase B：forced-local / live / frozen MAPPO 可学习性筛选。
