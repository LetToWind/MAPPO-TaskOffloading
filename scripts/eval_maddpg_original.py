"""Greedy evaluation of an original maddpg_discrete.py checkpoint.

Same protocol as scripts/eval_original.py (MAPPO): identical deterministic
eval scenario (random.seed(990001)), 100 greedy episodes, so exp5 (MAPPO)
and exp6 (MADDPG) numbers are directly comparable.

Usage:
    python scripts/eval_maddpg_original.py --checkpoint model/maddpg_final \
        --episodes 100 --out seed1_eval.txt
"""

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Environment_marl_discrete import DiscreteEnviron  # noqa: E402
from model_agent_discrete_maddpg import DiscreteMADDPG  # noqa: E402
from init_input.experiment_setup import build_fixed_env_config  # noqa: E402

OBS_DIM = 26
BS_CAPACITY = 6
N_POWER = 5


def get_state(env, i):
    task_on_base = env.task_on_base
    task_state = []
    for j in range(len(task_on_base[i])):
        task_state.append((env.task_data_size[task_on_base[i][j]] - 1e5) / 4e5)
    while len(task_state) < 10:
        task_state.append(0.0)
    for j in range(len(task_on_base[i])):
        task_state.append((env.task_deadline[task_on_base[i][j]] - 1000.0) / 1000.0)
    while len(task_state) < 20:
        task_state.append(0.0)
    channel_infer = np.asarray(env.V2V_Interference_all[i], dtype=np.float32).flatten()
    if len(channel_infer) > 6:
        channel_infer = channel_infer[:6]
    elif len(channel_infer) < 6:
        channel_infer = np.pad(channel_infer, (0, 6 - len(channel_infer)),
                               constant_values=float(env.sig2))
    return np.concatenate((task_state, channel_infer)).astype(np.float32)


def build_scenario():
    for _ in range(500):
        try:
            cand = build_fixed_env_config(25)
            if len(cand["task_data_size"]) >= 25:
                return cand
        except (IndexError, ValueError):
            continue
    raise RuntimeError("scenario build failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n_agents = 5
    agents = [
        DiscreteMADDPG("agent%d" % (i + 1), i, n_agents=n_agents, obs_dim=OBS_DIM,
                       max_tasks=BS_CAPACITY, max_channels=BS_CAPACITY,
                       n_power=N_POWER)
        for i in range(n_agents)
    ]
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

    random.seed(990001)
    sim_dict = build_scenario()

    rewards, successes, fails = [], [], []
    for _ in range(args.episodes):
        env = DiscreteEnviron(copy.deepcopy(sim_dict))
        ep_r, ep_s, ep_f = 0.0, 0, 0
        steps = 0
        while (not (np.any(env.task_deadline) == 0)) and \
                (not (np.any(env.task_data_size) == 0)) and steps < 60:
            states = np.asarray([get_state(env, i) for i in range(n_agents)],
                                dtype=np.float32)
            cm, pm = env.get_action_masks()
            ch = np.zeros((n_agents, BS_CAPACITY), dtype=np.int32)
            pw = np.zeros((n_agents, BS_CAPACITY), dtype=np.int32)
            for i, agent in enumerate(agents):
                a_ch, a_pw = agent.greedy_action(
                    sess, states[i], cm[i], pm[i])
                ch[i] = a_ch[0]
                pw[i] = a_pw[0]
            r, s, f = env.act_for_training_discrete(ch, pw)
            ep_r += r
            ep_s += s
            ep_f += f
            steps += 1
        rewards.append(ep_r)
        successes.append(ep_s)
        fails.append(ep_f)
    line = "greedy_eval reward=%.3f+/-%.3f success=%.3f fail=%.3f n=%d" % (
        np.mean(rewards), np.std(rewards), np.mean(successes),
        np.mean(fails), len(rewards))
    print(line)
    if args.out:
        Path(args.out).write_text(line + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
