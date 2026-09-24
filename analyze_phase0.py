"""Analyze Phase 0 delay-sensitivity results.

Fixed-topology modes: aggregates phase0_{prefix}_{arm}_s{seed}[_eval].csv
  - final greedy-eval reward / success, mean +/- std across seeds
  - relative degradation vs the `none` arm
Dynamic mode: previous per-tag behavior via --dynamic.
"""

import csv
import glob
import os
import sys

import numpy as np

EVAL_TAIL = 5          # use the last N eval points per seed
PREFIXES = {
    "fx": ["none", "fixed_d1", "fixed_d3", "unfixed_1-3"],
    "c3": ["none", "fixed_d3", "fixed_d5", "unfixed_1-4"],
}


def load_rows(path, cols):
    out = []
    with open(path) as f:
        for r in csv.reader(f):
            if not r or r[0].startswith("#") or not r[0].isdigit():
                continue
            out.append([float(r[c]) for c in cols])
    return out


def analyze_fixed(base, prefix):
    arms = PREFIXES[prefix]
    print("=== greedy evaluation (prefix %s, last %d eval points) ===" %
          (prefix, EVAL_TAIL))
    print("%-13s %5s | %16s | %16s | %10s" %
          ("arm", "n", "eval_reward", "eval_success", "vs none"))
    summary = {}
    n_seeds = 0
    for arm in arms:
        per_seed_r, per_seed_s = [], []
        for path in sorted(glob.glob(
                os.path.join(base, "phase0_%s_%s_s*_eval.csv" % (prefix, arm)))):
            rows = load_rows(path, [1, 2])
            if len(rows) < 3:
                continue
            tail = rows[-EVAL_TAIL:]
            per_seed_r.append(np.mean([r[0] for r in tail]))
            per_seed_s.append(np.mean([r[1] for r in tail]))
        if not per_seed_r:
            print("%-13s  no data" % arm)
            continue
        n_seeds = len(per_seed_r)
        summary[arm] = (np.mean(per_seed_r), np.std(per_seed_r),
                        np.mean(per_seed_s), np.std(per_seed_s))
    oracle = summary.get("none", (float("nan"),))[0]
    for arm in arms:
        if arm not in summary:
            continue
        m, sd, ms, ssd = summary[arm]
        print("%-13s %5d | %7.2f +/- %5.2f | %5.2f +/- %4.2f | %+9.1f%%" %
              (arm, n_seeds, m, sd, ms, ssd,
               (m - oracle) / abs(oracle) * 100.0 if oracle == oracle else float("nan")))

    print("\n=== eval success over training (avg across seeds, per 500 eps) ===")
    marks = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    print("%-13s %s" % ("arm", " ".join("%7d" % m for m in marks)))
    for arm in arms:
        curves = []
        for path in sorted(glob.glob(
                os.path.join(base, "phase0_%s_%s_s*_eval.csv" % (prefix, arm)))):
            rows = load_rows(path, [0, 2])
            if len(rows) < 3:
                continue
            curves.append(np.array([r[1] for r in rows]))
        if not curves:
            continue
        n = min(len(c) for c in curves)
        mat = np.vstack([c[:n] for c in curves])
        idx = [min(int(m / 100) - 1, n - 1) for m in marks if m <= n * 100]
        print("%-13s %s" % (arm, " ".join("%7.2f" % mat[:, i].mean() for i in idx)))


def analyze_dynamic(base):
    paths = sorted(glob.glob(os.path.join(base, "phase0_*.csv")))
    tags = [os.path.basename(p)[7:-4] for p in paths
            if "smoke" not in os.path.basename(p) and "_s" not in os.path.basename(p)
            and not os.path.basename(p).endswith("_eval.csv")]
    window = 1000
    stats = {}
    for tag in tags:
        rows = load_rows(os.path.join(base, "phase0_%s.csv" % tag), [1, 2, 3])
        if len(rows) < window // 2:
            continue
        tail = rows[-window:]
        stats[tag] = (np.mean([r[0] for r in tail]),
                      np.mean([r[1] for r in tail]),
                      np.mean([r[2] for r in tail]))
    oracle = stats.get("none", (float("nan"),))[0]
    print("%-14s %8s %8s %8s %10s" % ("tag", "reward", "success", "fail", "vs none"))
    for tag, (r, s, f) in sorted(stats.items()):
        print("%-14s %8.2f %8.2f %8.2f %+9.1f%%" %
              (tag, r, s, f, (r - oracle) / abs(oracle) * 100.0 if oracle == oracle else float("nan")))


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    if "--dynamic" in sys.argv:
        analyze_dynamic(base)
    else:
        prefix = "fx"
        for arg in sys.argv[1:]:
            if arg in PREFIXES:
                prefix = arg
        analyze_fixed(base, prefix)
