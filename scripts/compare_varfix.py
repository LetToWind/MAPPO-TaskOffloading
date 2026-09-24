"""Compare seed variance: UPDATE_EVERY=1 (exp3/4/4b) vs UPDATE_EVERY=25 (exp7)."""
import csv
import glob
import os

import numpy as np
from scipy import stats

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
EVAL_TAIL = 5


def seed_stats(path):
    rows = []
    for r in csv.reader(open(path)):
        if r and r[0].isdigit():
            rows.append([float(x) for x in r])
    if len(rows) < 3:
        return None
    tail = rows[-EVAL_TAIL:]
    return np.mean([r[1] for r in tail]), np.mean([r[2] for r in tail])


def collect(pattern, folders):
    out = []
    for f in folders:
        for path in sorted(glob.glob(os.path.join(base, f, pattern))):
            st = seed_stats(path)
            if st:
                out.append(st)
    return out


ue1 = collect("phase0_c3_none_s*_eval.csv",
              ["exp3_fixed_ch3", "exp4_sealing", "exp4b_seeds"])
ue25 = collect("phase0_u25_none_s*_eval.csv", ["exp7_varfix"])

for name, data in (("UPDATE_EVERY=1 (old, n=%d)" % len(ue1), ue1),
                   ("UPDATE_EVERY=25 (fix, n=%d)" % len(ue25), ue25)):
    r = [x[0] for x in data]
    s = [x[1] for x in data]
    print("%-28s reward %7.2f +/- %5.2f (CV %4.0f%%) | success %.2f +/- %.2f"
          % (name, np.mean(r), np.std(r, ddof=1), 100 * np.std(r, ddof=1) / max(abs(np.mean(r)), 1e-9),
             np.mean(s), np.std(s, ddof=1)))
    print("%-28s   seeds reward: %s" % ("", " ".join("%6.1f" % x for x in r)))

if len(ue1) > 1 and len(ue25) > 1:
    r1 = [x[0] for x in ue1]
    r2 = [x[0] for x in ue25]
    t, p = stats.ttest_ind(r1, r2, equal_var=False)
    print("\nWelch t-test reward UE1 vs UE25: t=%.2f p=%.4f" % (t, p))
    v1 = np.var(r1, ddof=1)
    v2 = np.var(r2, ddof=1)
    print("variance ratio (UE1/UE25): %.2f" % (v1 / max(v2, 1e-9)))
