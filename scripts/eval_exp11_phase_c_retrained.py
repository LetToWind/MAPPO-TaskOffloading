"""Evaluate Phase-C delay-trained Base MAPPO checkpoints on common seeds."""

from __future__ import print_function

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from delay_filter import DelayFilter  # noqa: E402
from dynamic_offload_env import DynamicOffloadEnv  # noqa: E402
from mappo_agent import MAPPOAgent  # noqa: E402
from train_mappo_dyn import (  # noqa: E402
    K_ROUTE, MAX_QUEUE, NEIGH_GROUPS, N_POWER, N_ROUTE_OPT, OBS_DIM,
    run_episode,
)


MODEL_ROOT = ROOT / "results" / "exp11_phase_c_retrain"
ARMS = (
    ("fixed2", "struct_fixed_d2", "fixed", {"d_value": 2}),
    ("unfixed1-3", "struct_unfixed_1-3", "unfixed",
     {"d_min": 1, "d_max": 3}),
)


def model_paths(prefix):
    out = []
    pattern = str(MODEL_ROOT / ("model_%s_s*" % prefix) / "model.index")
    for path in glob.glob(pattern):
        match = re.search(r"_s(\d+)[\\/]model\.index$", path)
        if match:
            out.append((int(match.group(1)), path[:-len(".index")]))
    return sorted(out)


def make_env(seed):
    return DynamicOffloadEnv(
        n_bs=5, n_channels=3, lam0=0.1, lam_hot=0.6,
        hotspot_dwell=10, n_hotspots=2, T=30, arrival_end=25,
        max_queue=MAX_QUEUE, holding_cost=0.02, ddl_min=5, ddl_max=11,
        data_min=1e5, data_max=3e5, capacity_markov=True,
        capacity_stay=0.85, hotspot_capacity_penalty=1,
        randomize_neighbor_order=True, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default=str(
        ROOT / "results" / "exp11_phase_c_retrained_eval.csv"))
    args = ap.parse_args()

    tf.reset_default_graph()
    agent = MAPPOAgent(
        "shared_agent", 0, n_agents=5, obs_dim=OBS_DIM,
        max_tasks=MAX_QUEUE, max_channels=3, n_power=N_POWER,
        is_first_agent=True, n_route_slots=K_ROUTE,
        n_route_options=N_ROUTE_OPT, structured_route=True)
    agents = [agent for _ in range(5)]
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=2, inter_op_parallelism_threads=1))

    rows = []
    for arm, prefix, delay_mode, delay_kwargs in ARMS:
        models = model_paths(prefix)
        if len(models) != 10:
            raise SystemExit("expected 10 %s models, found %d" %
                             (arm, len(models)))
        for seed_id, checkpoint in models:
            saver.restore(sess, checkpoint)
            # Report both the trained condition and a no-delay sanity check.
            for eval_mode in ("none", arm):
                rewards, successes, arrivals = [], [], []
                for ep in range(args.episodes):
                    eval_seed = 9100003 + ep * 100003
                    env = make_env(eval_seed)
                    actual_mode = "none" if eval_mode == "none" else delay_mode
                    kwargs = {} if eval_mode == "none" else delay_kwargs
                    delay = DelayFilter(
                        5, OBS_DIM, mode=actual_mode,
                        rng=np.random.RandomState(eval_seed + 17),
                        groups=NEIGH_GROUPS, **kwargs)
                    reward, success, _, arrived, _ = run_episode(
                        env, agents, sess, delay, actual_mode, greedy=True)
                    rewards.append(reward)
                    successes.append(success)
                    arrivals.append(arrived)
                row = {
                    "train_mode": arm,
                    "model_seed": seed_id,
                    "eval_mode": eval_mode,
                    "episodes": args.episodes,
                    "reward": float(np.mean(rewards)),
                    "success": float(np.mean(successes)),
                    "arrived": float(np.mean(arrivals)),
                    "success_rate": float(np.sum(successes) /
                                          max(np.sum(arrivals), 1)),
                }
                rows.append(row)
                print("train %-11s seed %2d eval %-11s success %.2f "
                      "(%.2f%%) reward %.2f" %
                      (arm, seed_id, eval_mode, row["success"],
                       100 * row["success_rate"], row["reward"]),
                      flush=True)
    sess.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
