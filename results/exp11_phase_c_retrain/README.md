# exp11 Phase C：延迟环境重训 Base MAPPO

日期：2026-09-27

## 设置

- 架构：参数共享 actor + 随机邻居顺序 + 置换等变结构化路由头；
- 训练：1500 episodes，10 seeds/arm；
- 延迟臂：`fixed d=2`、`unfixed [1,3]`；
- 评估：每个模型 20 条共同环境轨迹；随机延迟使用逐 episode 固定随机种子；
- 直接延迟对照：`results/exp11_phase_c_direct.csv`。

## 结果

| 模式 | none 参考 | 未适配模型直接受延迟 | 延迟重训 Base | 适配增益 | 恢复率 |
|---|---:|---:|---:|---:|---:|
| fixed d=2 | 23.63 | 21.12 | 21.54 | +0.43 | 16.9% |
| unfixed [1,3] | 23.63 | 21.21 | 21.51 | +0.30 | 12.4% |

适配增益的配对 bootstrap 95% CI：

- fixed d=2：`[-0.57, 1.43]` tasks/episode；
- unfixed [1,3]：`[-0.73, 1.24]` tasks/episode。

两者均跨 0。结论是普通 MAPPO 的延迟分布适配没有显著恢复性能；Phase C 的
“延迟确实有害”结论仍成立，同时 Phase E 应把这里保存的模型作为 delayed-base
基线，RDC 必须相对该基线计算恢复率。

## 复现

```powershell
& 'D:\tool\anaconda\envs\py37tf\python.exe' scripts/run_experiments.py --group exp11_phase_c_retrain --max-parallel 8
& 'D:\tool\anaconda\envs\py37tf\python.exe' scripts/eval_exp11_phase_c_retrained.py --episodes 20
& 'D:\tool\anaconda\envs\py37tf\python.exe' scripts/analyze_exp11_phase_c_retrained.py
```

共同轨迹评估明细位于 `results/exp11_phase_c_retrained_eval.csv`，训练日志、逐 episode
CSV 和 20 个 checkpoint 均保存在本目录。
