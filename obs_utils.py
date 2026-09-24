"""Observation utilities for the delay-compensation experiments (Phase 0+).

Observation layout (34 dims for n_agents=5):

  [0:10]   own task remaining data sizes, normalized      (real-time)
  [10:20]  own task deadlines, normalized                 (real-time)
  [20:26]  per-channel V2V interference                    (delayable)
  [26:34]  neighbor load, per neighbor j != i:
           (active_task_count / 6, remaining_data / 3e6)   (delayable)

The first 20 dims are the agent's self-state (kept identical to the original
26-dim `get_state` in train_mappo.py). The last 14 dims are other-source
information: 6 interference measurements (environment/other-agent action
mediated) and 8 neighbor-load features (other-agent states). Only these
14 dims are subject to the DelayFilter.
"""

import numpy as np

N_TASK_SLOTS = 10
MAX_TASKS_PER_BS = 6
TASK_DATA_MIN = 1e5
TASK_DATA_RANGE = 4e5
TASK_DEADLINE_MIN = 1000.0
TASK_DEADLINE_RANGE = 1000.0
N_INTERF_DIMS = 6
NEIGHBOR_TOTAL_NORM = 3e6  # 6 tasks * 5e5 bytes


def obs_dim(n_agents):
    return 2 * N_TASK_SLOTS + N_INTERF_DIMS + (n_agents - 1) * 2


def infer_slice(n_agents):
    return slice(2 * N_TASK_SLOTS, 2 * N_TASK_SLOTS + N_INTERF_DIMS)


def neighbor_slice(n_agents):
    start = 2 * N_TASK_SLOTS + N_INTERF_DIMS
    return slice(start, start + (n_agents - 1) * 2)


def delayed_slice(n_agents):
    """Indices of all delayable components (interference + neighbor load)."""
    start = 2 * N_TASK_SLOTS
    return slice(start, obs_dim(n_agents))


def build_obs(env, i, n_agents):
    """Delay-free observation of agent i (ground truth for the compensator)."""
    obs = np.zeros(obs_dim(n_agents), dtype=np.float32)
    tds = env.task_data_size
    tdl = env.task_deadline

    k = 0
    for j in env.task_on_base[i]:
        if j == -1:
            continue
        if k < N_TASK_SLOTS:
            obs[k] = (tds[j] - TASK_DATA_MIN) / TASK_DATA_RANGE
            k += 1
    k = N_TASK_SLOTS
    for j in env.task_on_base[i]:
        if j == -1:
            continue
        if k < 2 * N_TASK_SLOTS:
            obs[k] = (tdl[j] - TASK_DEADLINE_MIN) / TASK_DEADLINE_RANGE
            k += 1

    inf = np.asarray(env.V2V_Interference_all[i], dtype=np.float32).flatten()
    n_inf = min(len(inf), N_INTERF_DIMS)
    # log-normalize: raw interference spans ~1e-8 (noise floor) .. ~1e-3,
    # three to eight orders of magnitude below the other features; without
    # this transform the network cannot use the interference dims at all.
    inf_log = (np.log10(np.maximum(inf[:n_inf], 1e-8)) + 8.0) / 6.0
    obs[2 * N_TASK_SLOTS:2 * N_TASK_SLOTS + n_inf] = np.clip(inf_log, 0.0, 1.0)

    base = 2 * N_TASK_SLOTS + N_INTERF_DIMS
    for j in range(n_agents):
        if j == i:
            continue
        active = [t for t in env.task_on_base[j] if t != -1 and tds[t] > 0]
        total = float(sum(tds[t] for t in active))
        obs[base] = min(len(active) / float(MAX_TASKS_PER_BS), 1.0)
        obs[base + 1] = min(total / NEIGHBOR_TOTAL_NORM, 1.0)
        base += 2
    return obs
