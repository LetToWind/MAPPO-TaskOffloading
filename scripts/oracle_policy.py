"""Hand-coded JSQ oracle for the dynamic offloading scenario (calibration).

Sees TRUE states (no delay). Routing: urgent tasks (rem<=4) stay local;
others go to the destination with the lowest alive-count score.
Scheduling: earliest-deadline tasks first, one per channel, max power;
spare channels reinforce the largest-data tasks.

Usage:
    python scripts/oracle_policy.py --episodes 200 --lam-hot 0.6 --ch 4
Writes results/exp10_calib/oracle_result.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_offload_env import DynamicOffloadEnv, POWER_LEVELS  # noqa

MAX_QUEUE = 10


def oracle_actions(env):
    n = env.n_bs
    score = env.alive_counts()            # true loads incl. unrouted + transit
    in_transit_cnt = [sum(1 for _, d, _ in env.in_transit if d == j)
                      for j in range(n)]
    rt = np.full((n, env.K if hasattr(env, "K") else 2), 5, dtype=np.int32)
    ch = np.zeros((n, env.n_ch), dtype=np.int32)
    pw = np.zeros((n, MAX_QUEUE), dtype=np.int32)
    sel = [[] for _ in range(n)]
    spare = [None] * n
    for j in range(n):
        others = env.neighbor_indices(j)
        for k in range(min(env.K, len(env.unrouted[j]))):
            tid = env.unrouted[j][k]
            tk = env.tasks[tid]
            if tk["rem"] <= 4:
                rt[j, k] = 0
                continue
            dest_scores = [score[o] + in_transit_cnt[o] for o in others]
            best = int(np.argmin(dest_scores))
            if dest_scores[best] < score[j]:
                rt[j, k] = best + 1
            else:
                rt[j, k] = 0
        # scheduling: earliest deadline first; channels assigned in a second
        # pass with global least-used coordination (see below)
        q = env.queue[j]
        order = [i for i in range(len(q)) if i < MAX_QUEUE]
        order = sorted(order,
                       key=lambda idx: (env.tasks[q[idx]]["rem"],
                                        -env.tasks[q[idx]]["data"]))
        n_tasks = min(len(order), env.n_ch)
        for c in range(n_tasks):
            slot = order[c]
            pw[j, slot] = len(POWER_LEVELS) - 1
        sel[j] = [order[c] for c in range(n_tasks)]
        spare[j] = max(order, key=lambda idx: env.tasks[q[idx]]["data"]) \
            if order else None
    # global channel assignment: (bs, slot) pairs to least-used channels
    ch_usage = np.zeros(env.n_ch, dtype=int)
    pairs = []
    for j in range(n):
        for slot in sel[j]:
            pairs.append((j, slot))
    for j, slot in pairs:
        c = int(np.argmin(ch_usage))
        ch[j, c] = slot + 1
        ch_usage[c] += 1
    for j in range(n):
        n_used = int((ch[j] > 0).sum())
        if n_used < env.n_ch and spare[j] is not None:
            big = spare[j]
            for c in range(env.n_ch):
                if ch[j, c] == 0:
                    ch[j, c] = big + 1
    return rt, ch, pw
    return rt, ch, pw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--lam-hot", type=float, default=1.5)
    ap.add_argument("--lam0", type=float, default=0.15)
    ap.add_argument("--ch", type=int, default=3)
    ap.add_argument("--ddl-min", type=int, default=5)
    ap.add_argument("--ddl-max", type=int, default=11)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rewards, succs, fails, arrives = [], [], [], []
    for ep in range(args.episodes):
        env = DynamicOffloadEnv(n_channels=args.ch, lam0=args.lam0,
                                lam_hot=args.lam_hot, ddl_min=args.ddl_min,
                                ddl_max=args.ddl_max,
                                seed=args.seed * 1000003 + 500000 + ep)
        total = 0.0
        s = f = 0
        while not env.done:
            rt, ch, pw = oracle_actions(env)
            r, info = env.step(rt, ch, pw)
            total += r
            s += info["success"]
            f += info["fail"]
        rewards.append(total)
        succs.append(s)
        fails.append(f)
        arrives.append(info["arrived"])
    line = dict(
        n=args.episodes, reward=float(np.mean(rewards)),
        reward_std=float(np.std(rewards)), success=float(np.mean(succs)),
        success_std=float(np.std(succs)), fail=float(np.mean(fails)),
        arrived=float(np.mean(arrives)), lam_hot=args.lam_hot, ch=args.ch,
        ddl="%d-%d" % (args.ddl_min, args.ddl_max))
    print("oracle: reward %.2f +/- %.2f | success %.2f +/- %.2f | "
          "fail %.2f | arrived %.1f (lam_hot=%.2f ch=%d)" %
          (line["reward"], line["reward_std"], line["success"],
           line["success_std"], line["fail"], line["arrived"],
           line["lam_hot"], line["ch"]))
    out = PROJECT_ROOT / "results" / "exp10_calib" / "oracle_result.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(line))
        if new:
            wcsv.writeheader()
        wcsv.writerow(line)
    print("appended ->", out)


if __name__ == "__main__":
    main()
