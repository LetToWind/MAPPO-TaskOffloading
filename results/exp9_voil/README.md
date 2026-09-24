# exp9_voil — 衰落校准实验:追踪 vs 惯性(信息价值测量)

## 前置设计(为什么做这个)

衰落本身不保证智能体学会追踪——若信道基线差距 ≫ 衰落幅度 σ,
"永远选基线最优信道"(惯性)仍是最优解,延迟照样无害。
所以先测**信息价值 VoI** 再跑延迟矩阵:

```
VoI = E[回报 | 干扰观测实时] − E[回报 | 干扰观测冻结在 episode 开头]
```

## 三臂设计(CH=3 固定拓扑,UPDATE_EVERY=25,各 10 种子)

| 臂 | 设定 | 检验什么 |
|---|---|---|
| (a) `fa_r07s06` | 衰落 on(ρ=0.7, σ=6dB),观测实时 | 追踪策略上限 |
| (b) `fz_r07s06` | 衰落 on,干扰观测**冻结**在 t=0 | 惯性策略近似上限 |
| (c) exp7 参考 | 无衰落 | 基线(已有数据,不重跑) |

## 衰落模块规格(fading_env.py)

- 每 (智能体 i, 信道 c) 一个独立高斯-马尔可夫过程(dB 域):
  g(t+1) = ρ·g(t) + √(1−ρ²)·ξ,d 步自相关恰为 ρ^d
  → ρ=0.7 时:零延迟(含环境 1 步内在滞后)相关性 0.49,
  延迟 1/2/4 步降至 0.34/0.24/0.12——追踪可行但延迟有代价
- 只乘干扰项(含噪声底保护),动力学与观测用同一实现值;
- 默认关闭(FADING=0 完全复现旧行为)

## 运行

```
conda activate py37tf
cd C:\Users\zengd\Desktop\论文\W-MADDPG\MADDPG_Discrete
python scripts/run_experiments.py --group exp9_voil --max-parallel 8
python scripts/analyze_exp9.py
```

20 个 run(8 并行约 35-40 分钟)。分析输出自动给 VERDICT:
- p<0.05 且 VoI>3 → 追踪成立,进入延迟矩阵(exp10)
- 否则 → 跑参数 sweep:
  `python scripts/run_experiments.py --group exp9b_sweep --max-parallel 8`
  (40 个 run,ρ∈{0.5,0.9}×σ=6 与 σ∈{3,9}×ρ=0.7,各 5 种子)

## 结果
(待运行后填写)
