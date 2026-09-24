# W-MADDPG 代码说明

## 项目结构

```
MADDPG_Discrete/
├── config/
│   ├── settings.py          ← 所有配置参数（改这里即可）
│   └── config.py            ← Dynaconf 加载器
├── init_input/
│   ├── dynamic_topology.py  ← 动态拓扑生成（匈牙利算法/贪心）
│   ├── experiment_setup.py  ← 固定场景构建器（variant 25）
│   ├── init_vehicles.py     ← 车辆轨迹加载
│   ├── init_edge_node.py    ← 边缘节点初始化
│   ├── init_task_by_time.py ← 任务生成（数据量、截止时间）
│   └── init_distance.py     ← 距离矩阵、task-BS 分配
├── Environment_marl_discrete.py  ← MEC 仿真环境
│
├── train_mappo.py           ← MAPPO 训练入口
├── mappo_agent.py           ← MAPPO Actor-Critic
├── mappo_buffer.py          ← Rollout Buffer + GAE
│
├── maddpg_discrete.py       ← MADDPG 训练入口（baseline）
├── model_agent_discrete_maddpg.py  ← MADDPG 模型
├── replay_buffer_discrete.py       ← MADDPG Replay Buffer
│
├── test_mappo.py            ← 统一测试脚本（MAPPO/MADDPG）
├── plot_training.py         ← 训练曲线绘制
│
├── model/                   ← 自动生成（checkpoint）
├── sum_reward_mappo.csv     ← 训练输出
├── sum_reward_discrete.csv  ← 训练输出
└── test_results.json        ← 测试输出
```

## 配置参数表（config/settings.py）

### 训练超参

| 参数 | 默认值 | 说明 |
|---|---|---|
| `N_EPISODES` | 18000 | 总训练 episode 数 |
| `N_STEPS` | 2000 | buffer 最大容量（每 episode 自动 GAE） |
| `K_EPOCHS` | 4 | MAPPO 每批数据训练轮数 |
| `MINIBATCH_SIZE` | 64 | MAPPO minibatch 大小 |
| `GAMMA` | 0.99 | 折扣因子 |
| `GAE_LAMBDA` | 0.95 | GAE λ 参数 |
| `CLIP_EPS` | 0.2 | PPO clip 系数 |
| `ENTROPY_COEF` | 0.01 | 熵正则化系数 |
| `MAX_GRAD_NORM` | 0.5 | 梯度裁剪阈值 |
| `BATCH_SIZE` | 256 | MADDPG 采样批量 |
| `MEMORY_SIZE` | 50000 | MADDPG replay buffer 容量 |

### 训练模式

| 参数 | 默认值 | 说明 |
|---|---|---|
| `FIXED_CURRICULUM` | 0 | `1`=课程学习(固定→匈牙利), `0`=走 USE_DYNAMIC_TOPOLOGY 分支 |
| `USE_DYNAMIC_TOPOLOGY` | 0 | `1`=动态拓扑(随机 UE+匈牙利), `0`=固定拓扑(variant 25) |

三种模式组合：

| FIXED_CURRICULUM | USE_DYNAMIC_TOPOLOGY | 效果 |
|---|---|---|
| 1 | - | 课程学习：Phase1 固定→Phase2 过渡→Phase3 匈牙利 |
| 0 | 0 | 纯固定拓扑（test_data.csv + hardcoded BS） |
| 0 | 1 | 纯动态拓扑（随机 UE + 匈牙利） |

### 课程学习控制

| 参数 | 默认值 | 说明 |
|---|---|---|
| `UE_FIXED_EPISODES` | 6000 | 固定拓扑阶段 episode 数 |
| `UE_RANDOM_RAMP` | 6000 | 过渡阶段 episode 数 |
| `MAX_UE_DISTANCE` | 400 | UE 到最近 BS 的最大距离约束（0=不约束） |

### 环境配置

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DYNAMIC_N_BS` | 5 | BS 数量 |
| `DYNAMIC_N_TASKS` | 25 | 总任务数量 |
| `DYNAMIC_AREA_WIDTH/HEIGHT` | 2000 | 区域大小 |
| `TASK_DATA_SIZE_MIN/MAX` | 1e5/5e5 | 任务数据量范围(字节) |
| `TASK_DEADLINE_MIN/MAX` | 1000/2000 | 任务截止时间范围(ms) |
| `WHITE_GAUSSIAN_NOISE` | 1e-8 | 噪声功率 |

### 固定拓扑 variant 25 硬编码数据

| 项 | 值 |
|---|---|
| BS 数量 | 5 |
| BS 位置 | [1590,1490], [896,1500], [1760,764], [882,830], [1188,272] |
| 数据源 | `../data/test_data.csv` |
| 信道数 | 6 |
| Task-BS 绑定 | 硬编码（见 `VARIANT_CONFIG[25]["task_id_list"]`） |

## 运行

### 环境要求

```bash
conda create -n maddpg python=3.7 -y
conda activate maddpg
pip install tensorflow==1.15.5 numpy pandas scipy dynaconf matplotlib
```

### MAPPO 训练

```bash
cd MADDPG_Discrete
python train_mappo.py
```

输出：
- `sum_reward_mappo.csv` — 每集 reward/success/fail（第一行含种子和超参）
- `sum_reward_mappo.png` — 训练曲线（自动生成）
- `model/mappo-{episode}` — 每 1000 集保存 checkpoint

### MADDPG 训练（baseline）

```bash
python maddpg_discrete.py
```

输出格式同上，文件名替换 mappo → maddpg。

### 测试（两模型对比）

两个模型各训练完成后，在 `model/` 下有 checkpoint：

```bash
python test_mappo.py
```

输出 `test_results.json`：

```json
{
  "mappo": {
    "avg_reward": 56.3,
    "std_reward": 12.1,
    "success_rate": 91.1,
    "n_tests": 100
  },
  "maddpg": {
    "avg_reward": 48.7,
    "std_reward": 15.3,
    "success_rate": 85.2,
    "n_tests": 100
  }
}
```

测试数据来源：`data/fill_xy.csv`（真实车辆轨迹，每隔 3 帧取一组拓扑），UE 由匈牙利算法分配至 BS，模型只 inference 不更新。

### 指定 checkpoint 测试

```bash
CHECKPOINT_MAPPO=./model/mappo-6000  CHECKPOINT_MADDPG=./model/maddpg-6000  python test_mappo.py
```

### 画图

训练后自动调用。也可手动对已有 CSV 画图：

```bash
python plot_training.py sum_reward_mappo.csv
```

输出同路径 `.png` 文件，包含 reward 折线图（带滑动平均）和 success 柱状图。

## 快速切换实验模式

所有配置在 `config/settings.py`，改一个值后直接 `python train_mappo.py`：

```python
# 固定拓扑训练
FIXED_CURRICULUM = 0
USE_DYNAMIC_TOPOLOGY = 0

# 动态拓扑训练
FIXED_CURRICULUM = 0
USE_DYNAMIC_TOPOLOGY = 1

# 课程学习（固定→匈牙利）
FIXED_CURRICULUM = 1
UE_FIXED_EPISODES = 6000
UE_RANDOM_RAMP = 6000
```

## 环境变量覆盖

所有配置支持环境变量临时覆盖（优先级高于 settings.py）：

```bash
N_EPISODES=5000  MAX_UE_DISTANCE=0  python train_mappo.py
```
