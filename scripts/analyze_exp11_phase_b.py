"""Analyze the v1.1 Phase-B MAPPO pilot."""

from __future__ import print_function

import csv
import glob
import os
import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "exp11_phase_b_pilot"
PHASE_A_FULL_INFO = 0.6433


def seed_tail(prefix, tail=3):
    values = []
    for path in sorted(glob.glob(str(BASE / ("phase0_%s_s*_eval.csv" % prefix)))):
        with open(path) as fh:
            rows = [r for r in csv.reader(fh) if r and r[0].isdigit()]
        if len(rows) < tail:
            continue
        part = rows[-tail:]
        values.append({
            "reward": np.mean([float(r[1]) for r in part]),
            "success": np.mean([float(r[2]) for r in part]),
            "fail": np.mean([float(r[3]) for r in part]),
            "routes": np.mean([float(r[4]) for r in part]),
            "file": os.path.basename(path),
        })
    return values


def train_canaries(prefix, tail=300):
    out = []
    for path in sorted(glob.glob(str(BASE / ("phase0_%s_s*.csv" % prefix)))):
        if path.endswith("_eval.csv") or path.endswith("_diag.csv"):
            continue
        with open(path) as fh:
            rows = [r for r in csv.reader(fh) if r and r[0].isdigit()]
        if not rows:
            continue
        part = rows[-tail:]
        out.append({
            "route_rate": np.mean([float(r[5]) for r in part]),
            "cond_route": np.mean([float(r[6]) for r in part]),
            "top_share": np.mean([float(r[7]) for r in part]),
            "load_std": np.mean([float(r[8]) for r in part]),
        })
    return out


def bootstrap_ci(diff, seed=20260927, n_boot=20000):
    diff = np.asarray(diff, dtype=np.float64)
    if len(diff) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    sims = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return tuple(np.percentile(sims, [2.5, 97.5]))


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", action="store_true")
    ap.add_argument("--permuted", action="store_true")
    ap.add_argument("--structured", action="store_true")
    args = ap.parse_args()
    if args.structured:
        BASE = ROOT / "results" / "exp11_phase_b_structured"
        prefixes = (("live", "struct_live"),
                    ("frozen", "struct_frozen"))
    elif args.permuted:
        BASE = ROOT / "results" / "exp11_phase_b_permuted"
        prefixes = (("live", "perm_live"),
                    ("frozen", "perm_frozen"))
    elif args.shared:
        BASE = ROOT / "results" / "exp11_phase_b_shared"
        prefixes = (("live", "shared_live"),
                    ("frozen", "shared_frozen"))
    else:
        prefixes = (("forced-local", "fl"), ("live", "live"),
                    ("frozen", "frozen"))
    arms = {name: seed_tail(prefix) for name, prefix in
            prefixes}
    print("=== exp11 Phase-B MAPPO pilot ===")
    for name, values in arms.items():
        if not values:
            print("%-14s no data" % name)
            continue
        succ = [x["success"] for x in values]
        reward = [x["reward"] for x in values]
        print("%-14s n=%d reward %7.2f +/- %5.2f success %5.2f +/- %.2f" %
              (name, len(values), np.mean(reward), np.std(reward, ddof=1)
               if len(reward) > 1 else 0.0, np.mean(succ),
               np.std(succ, ddof=1) if len(succ) > 1 else 0.0))
        print("  seeds", " ".join("%.2f" % x for x in succ))

    if not all(arms.values()):
        return
    live = np.array([x["success"] for x in arms["live"]])
    frozen = np.array([x["success"] for x in arms["frozen"]])
    n = min(len(live), len(frozen))
    live, frozen = live[:n], frozen[:n]

    # Convert Phase-A full-info success rate to tasks/episode using the mean
    # arrivals observed by the eval files (about 35 in this configuration).
    # For the pilot, the halfway rule is equivalently evaluated in the same
    # success-count scale using live eval arrival counts from training CSVs.
    arrivals = []
    live_prefix = dict(prefixes)["live"]
    for path in sorted(glob.glob(str(
            BASE / ("phase0_%s_s*.csv" % live_prefix)))):
        if path.endswith("_eval.csv") or path.endswith("_diag.csv"):
            continue
        with open(path) as fh:
            rows = [r for r in csv.reader(fh) if r and r[0].isdigit()]
        arrivals.append(np.mean([float(r[4]) for r in rows[-300:]]))
    mean_arrivals = float(np.mean(arrivals)) if arrivals else 1.0
    full_info_count = PHASE_A_FULL_INFO * mean_arrivals
    if "forced-local" in arms:
        fl = np.array([x["success"] for x in arms["forced-local"]])[:n]
        fl_mean = np.mean(fl)
    else:
        pilot_base = ROOT / "results" / "exp11_phase_b_pilot"
        old_base = BASE
        BASE = pilot_base
        fl_vals = seed_tail("fl")
        BASE = old_base
        fl_mean = np.mean([x["success"] for x in fl_vals])
    b1_need = fl_mean + 0.5 * (full_info_count - fl_mean)
    b2_diff = live - frozen
    b2_ci = bootstrap_ci(b2_diff)

    print("\n--- gates ---")
    print("B1 learnability: live %.2f vs halfway %.2f [%s]" %
          (np.mean(live), b1_need,
           "PASS" if np.mean(live) >= b1_need else "FAIL"))
    print("B2 live-frozen: %+.2f tasks CI[%.2f, %.2f] [%s]" %
          (np.mean(b2_diff), b2_ci[0], b2_ci[1],
           "PASS" if np.mean(b2_diff) >= 0.03 * mean_arrivals and
           b2_ci[0] > 0 else "FAIL"))

    print("\n--- canaries (train tail-300) ---")
    canary_prefixes = prefixes
    for name, prefix in canary_prefixes:
        vals = train_canaries(prefix)
        if vals:
            print("%-14s route %.3f cond %.3f top %.3f load_std %.3f" %
                  (name,
                   np.mean([x["route_rate"] for x in vals]),
                   np.mean([x["cond_route"] for x in vals]),
                   np.mean([x["top_share"] for x in vals]),
                   np.mean([x["load_std"] for x in vals])))


if __name__ == "__main__":
    main()
