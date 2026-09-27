"""Phase-C direct delay evaluation for saved structured MAPPO models."""

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


MODES = (
    ("none", {}),
    ("fixed1", {"mode": "fixed", "d_value": 1}),
    ("fixed2", {"mode": "fixed", "d_value": 2}),
    ("fixed3", {"mode": "fixed", "d_value": 3}),
    ("unfixed1-3", {"mode": "unfixed", "d_min": 1, "d_max": 3}),
    ("frozen", {"mode": "frozen"}),
)


def model_paths():
    roots = [ROOT / "results" / "exp11_phase_b_structured",
             ROOT / "results" / "exp11_phase_b_structured_more"]
    out = []
    for root in roots:
        for path in glob.glob(str(root / "model_struct_live_s*" / "model.index")):
            match = re.search(r"model_struct_live_s(\d+)", path)
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
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default=str(
        ROOT / "results" / "exp11_phase_c_direct.csv"))
    args = ap.parse_args()

    models = model_paths()
    if not models:
        raise SystemExit("no saved structured live models found")

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
    for seed_id, checkpoint in models:
        saver.restore(sess, checkpoint)
        for mode_name, spec in MODES:
            rewards = []
            successes = []
            arrivals = []
            for ep in range(args.episodes):
                eval_seed = 9100003 + ep * 100003
                env = make_env(eval_seed)
                kwargs = dict(spec)
                actual_mode = kwargs.pop("mode", "none")
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
                "model_seed": seed_id,
                "mode": mode_name,
                "episodes": args.episodes,
                "reward": float(np.mean(rewards)),
                "success": float(np.mean(successes)),
                "arrived": float(np.mean(arrivals)),
                "success_rate": float(np.sum(successes) /
                                      max(np.sum(arrivals), 1)),
            }
            rows.append(row)
            print("seed %d %-12s success %.2f (%.2f%%) reward %.2f" %
                  (seed_id, mode_name, row["success"],
                   100 * row["success_rate"], row["reward"]), flush=True)
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
