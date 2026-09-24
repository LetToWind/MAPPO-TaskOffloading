"""exp8: delay effect under the STABLE trainer (UPDATE_EVERY=25, CH=3).

Arms: none (exp7) vs fixed_d3 / fixed_d5 / unfixed_1-4 (exp8), 10 seeds each.
Metric: greedy eval mean of last 5 eval points.
"""
import csv
import glob
import os

import numpy as np
from scipy import stats

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
EVAL_TAIL = 5

ARMS = {
    "none": ("exp7_varfix", "u25_none"),
    "fixed_d3": ("exp8_delay_stable", "u25_fixed_d3"),
    "fixed_d5": ("exp8_delay_stable", "u25_fixed_d5"),
    "unfixed_1-4": ("exp8_delay_stable", "u25_unfixed_1-4"),
}


def seed_stats(path):
    rows = []
    for r in csv.reader(open(path)):
        if r and r[0].isdigit():
            rows.append([float(x) for x in r])
    if len(rows) < 3:
        return None
    tail = rows[-EVAL_TAIL:]
    return np.mean([r[1] for r in tail]), np.mean([r[2] for r in tail])


data = {}
for arm, (folder, tag) in ARMS.items():
    vals = []
    for path in sorted(glob.glob(os.path.join(base, folder,
                                              "phase0_%s_s*_eval.csv" % tag))):
        st = seed_stats(path)
        if st:
            vals.append(st)
    data[arm] = vals

nr = [x[0] for x in data["none"]]
ns = [x[1] for x in data["none"]]
print("=== exp8: delay effect, stable trainer (UE=25, CH=3, n=%d seeds/arm) ===" % len(nr))
print("%-13s %18s %18s %10s %10s" % ("arm", "eval_reward", "eval_success",
                                     "vs none", "p(Welch)"))
print("%-13s %8.2f +/- %5.2f %8.2f +/- %4.2f" %
      ("none", np.mean(nr), np.std(nr, ddof=1), np.mean(ns), np.std(ns, ddof=1)))
for arm in ("fixed_d3", "fixed_d5", "unfixed_1-4"):
    r = [x[0] for x in data[arm]]
    s = [x[1] for x in data[arm]]
    t_r, p_r = stats.ttest_ind(nr, r, equal_var=False)
    t_s, p_s = stats.ttest_ind(ns, s, equal_var=False)
    rel = (np.mean(r) - np.mean(nr)) / abs(np.mean(nr)) * 100
    print("%-13s %8.2f +/- %5.2f %8.2f +/- %4.2f %+9.1f%%  r_p=%.4f s_p=%.4f" %
          (arm, np.mean(r), np.std(r, ddof=1), np.mean(s), np.std(s, ddof=1),
           rel, p_r, p_s))

print("\nper-seed reward:")
for arm in ARMS:
    print("%-13s %s" % (arm, " ".join("%6.1f" % x[0] for x in data[arm])))
