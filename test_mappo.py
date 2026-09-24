import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from config.config import settings
from Environment_marl_discrete import DiscreteEnviron
from init_input.dynamic_topology import build_dynamic_env_config

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

OBS_DIM = 26
BS_CAPACITY = 6
N_POWER = 5


def get_variant_bs_config(variant=25):
    cfg = settings.VARIANT_CONFIG[variant]
    bs_positions = np.column_stack([cfg["rsu_x"], cfg["rsu_y"]]).astype(np.float64)
    usable_channels = [list(range(cfg["channel_num"])) for _ in range(cfg["rsu_num"])]
    return bs_positions, usable_channels


def get_state(env, i):
    task_on_base = env.task_on_base
    task_data_size = env.task_data_size
    task_deadline = env.task_deadline
    task_state = []
    for j in range(len(task_on_base[i])):
        task_state.append((task_data_size[task_on_base[i][j]] - 1e5) / 4e5)
    while len(task_state) < 10:
        task_state.append(0.0)
    for j in range(len(task_on_base[i])):
        task_state.append((task_deadline[task_on_base[i][j]] - 1000.0) / 1000.0)
    while len(task_state) < 20:
        task_state.append(0.0)
    channel_infer = np.asarray(env.V2V_Interference_all[i], dtype=np.float32).flatten()
    if len(channel_infer) > 6:
        channel_infer = channel_infer[:6]
    elif len(channel_infer) < 6:
        channel_infer = np.pad(channel_infer, (0, 6 - len(channel_infer)), constant_values=float(env.sig2))
    return np.concatenate((task_state, channel_infer)).astype(np.float32)


def should_continue_episode(env):
    return (not (np.any(env.task_deadline) == 0)) and (not (np.any(env.task_data_size) == 0))


def load_real_ue_positions(csv_path, time_step):
    df = pd.read_csv(csv_path)
    time_df = df[df["time"] == time_step]
    return time_df[["x", "y"]].values.astype(np.float64)


def run_episode_mappo(sess, agents, sim_dict, n_agents):
    env = DiscreteEnviron(sim_dict)
    sum_reward = 0.0
    success = 0
    fail = 0
    while should_continue_episode(env):
        states = np.asarray([get_state(env, i) for i in range(n_agents)], dtype=np.float32)
        channel_masks, power_masks = env.get_action_masks()
        channel_choices = np.zeros((n_agents, BS_CAPACITY), dtype=np.int32)
        power_choices = np.zeros((n_agents, BS_CAPACITY), dtype=np.int32)
        for agent_id, agent in enumerate(agents):
            ch, pw, _ = agent.act(sess, states[agent_id], channel_masks[agent_id], power_masks[agent_id])
            channel_choices[agent_id] = ch
            power_choices[agent_id] = pw
        step_reward, s, f = env.act_for_training_discrete(channel_choices, power_choices)
        sum_reward += step_reward
        success += s
        fail += f
    return sum_reward, success, fail


def run_episode_maddpg(sess, agents, sim_dict, n_agents):
    from maddpg_discrete import get_state as maddpg_get_state
    env = DiscreteEnviron(sim_dict)
    sum_reward = 0.0
    success = 0
    fail = 0
    while should_continue_episode(env):
        states = np.asarray([maddpg_get_state(env, i) for i in range(n_agents)], dtype=np.float32)
        channel_masks, power_masks = env.get_action_masks()
        channel_choices = np.zeros((n_agents, BS_CAPACITY), dtype=np.int32)
        power_choices = np.zeros((n_agents, BS_CAPACITY), dtype=np.int32)
        for agent_id, agent in enumerate(agents):
            ch, pw, _ = agent.act(sess, states[agent_id], channel_masks[agent_id], power_masks[agent_id])
            channel_choices[agent_id] = ch
            power_choices[agent_id] = pw
        step_reward, s, f = env.act_for_training_discrete(channel_choices, power_choices)
        sum_reward += step_reward
        success += s
        fail += f
    return sum_reward, success, fail


def test_model(model_type, checkpoint_path, test_times, bs_positions, bs_channels, n_agents):
    tf.reset_default_graph()

    if model_type == "mappo":
        from mappo_agent import MAPPOAgent
        agents = [
            MAPPOAgent("agent%d" % (i + 1), i, n_agents=n_agents, obs_dim=OBS_DIM,
                       max_tasks=BS_CAPACITY, max_channels=BS_CAPACITY, n_power=N_POWER,
                       is_first_agent=(i == 0))
            for i in range(n_agents)
        ]
        run_ep = run_episode_mappo
    elif model_type == "maddpg":
        from model_agent_discrete_maddpg import DiscreteMADDPG
        agents = [
            DiscreteMADDPG("agent%d" % (i + 1), i, n_agents=n_agents, obs_dim=OBS_DIM,
                           max_tasks=BS_CAPACITY, max_channels=BS_CAPACITY, n_power=N_POWER)
            for i in range(n_agents)
        ]
        run_ep = run_episode_maddpg
    else:
        raise ValueError("model_type must be 'mappo' or 'maddpg'")

    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    saver.restore(sess, checkpoint_path)

    csv_path = str(PROJECT_ROOT / "data" / "fill_xy.csv")
    results = []
    for t in test_times:
        ue_positions = load_real_ue_positions(csv_path, t)
        if len(ue_positions) < n_agents + 1:
            continue
        n_tasks = min(len(ue_positions), 30)
        rng = np.random.RandomState(int(t))
        ue_idx = rng.choice(len(ue_positions), n_tasks, replace=False)
        ue_pos = ue_positions[ue_idx]
        sim_dict = build_dynamic_env_config(
            n_bs=n_agents, n_tasks=n_tasks, n_channels=BS_CAPACITY,
            channel_per_bs=BS_CAPACITY, max_tasks_per_bs=BS_CAPACITY,
            max_ue_distance=500, seed=int(t),
            bs_positions=bs_positions, usable_channel_of_all_nodes=bs_channels,
            ue_positions=ue_pos)
        reward, success, fail = run_ep(sess, agents, sim_dict, n_agents)
        results.append({"time": int(t), "tasks": n_tasks, "reward": round(float(reward), 2),
                        "success": int(success), "fail": int(fail)})
    sess.close()
    return results


def main():
    variant = int(os.environ.get("TEST_VARIANT", "25"))
    n_agents = settings.VARIANT_CONFIG[variant]["rsu_num"]
    bs_positions, bs_channels = get_variant_bs_config(variant)

    csv_path = str(PROJECT_ROOT / "data" / "fill_xy.csv")
    df = pd.read_csv(csv_path)
    all_times = sorted(df["time"].unique())
    test_times = all_times[:len(all_times):3]

    model_dir = SCRIPT_DIR / "model"
    all_results = {}

    for model_type in ["mappo", "maddpg"]:
        ckpt = os.environ.get("CHECKPOINT_%s" % model_type.upper(), "")
        if not ckpt:
            ckpt = tf.train.latest_checkpoint(str(model_dir))
            if ckpt:
                print("[%s] using checkpoint: %s" % (model_type.upper(), ckpt))
            else:
                print("[%s] no checkpoint found, skipping" % model_type.upper())
                continue
        print("[%s] testing %d time steps..." % (model_type.upper(), len(test_times)))
        results = test_model(model_type, ckpt, test_times, bs_positions, bs_channels, n_agents)
        rewards = [r["reward"] for r in results]
        successes = [r["success"] for r in results]
        fails = [r["fail"] for r in results]
        all_results[model_type] = {
            "avg_reward": round(float(np.mean(rewards)), 2),
            "std_reward": round(float(np.std(rewards)), 2),
            "success_rate": round(sum(successes) / (sum(successes) + sum(fails)) * 100, 1),
            "avg_success": round(float(np.mean(successes)), 2),
            "avg_fail": round(float(np.mean(fails)), 2),
            "n_tests": len(results),
            "checkpoint": ckpt,
        }
        print("  Reward: %.1f +/- %.1f  Success: %.1f%%" %
              (all_results[model_type]["avg_reward"], all_results[model_type]["std_reward"],
               all_results[model_type]["success_rate"]))

    if all_results:
        output_file = SCRIPT_DIR / "test_results.json"
        with output_file.open("w") as f:
            json.dump(all_results, f, indent=2)
        print("\nSaved to %s" % output_file)


if __name__ == "__main__":
    main()
