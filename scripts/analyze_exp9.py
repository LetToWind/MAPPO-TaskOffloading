"""exp9 VoI calibration analysis: tracking vs inertia under fading.

Arms (CH=3, fixed topology, UPDATE_EVERY=25, 10 seeds):
  (a) fa_* : fading on, interference obs live        -> tracking upper bound
  (b) fz_* : fading on, interference obs frozen       -> inertia proxy
  (c) exp7 u25_none : no fading (reference, existing data)

Decision rule: VoI = mean(a) - mean(b); tracking is learnable iff
Welch p(a vs b) < 0.05 and VoI exceeds the seed noise floor (~ +/- 6).
"""

import csv
import glob
import os
import sys

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


def collect(folder, pattern):
    out = []
    for path in sorted(glob.glob(os.path.join(base, folder, pattern))):
        st = seed_stats(path)
        if st:
            out.append(st)
    return out


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "exp9_voil"
    fa = collect(folder, "phase0_fa_*_s*_eval.csv")
    fz = collect(folder, "phase0_fz_*_s*_eval.csv")
    ref = collect("exp7_varfix", "phase0_u25_none_s*_eval.csv")

    def show(name, data):
        if not data:
            print("%-28s no data" % name)
            return None, None
        r = [x[0] for x in data]
        s = [x[1] for x in data]
        print("%-28s reward %7.2f +/- %5.2f | success %5.2f +/- %4.2f | n=%d"
              % (name, np.mean(r), np.std(r, ddof=1),
                 np.mean(s), np.std(s, ddof=1), len(r)))
        print("%-28s   seeds: %s" % ("", " ".join("%6.1f" % x for x in r)))
        return r, s

    print("=== exp9 VoI calibration (%s) ===" % folder)
    fa_r, fa_s = show("(a) fading + live obs", fa)
    fz_r, fz_s = show("(b) fading + frozen obs", fz)
    show("(c) no fading (exp7 ref)", ref)

    if fa_r and fz_r:
        t, p = stats.ttest_ind(fa_r, fz_r, equal_var=False)
        voi = np.mean(fa_r) - np.mean(fz_r)
        print("\nVoI (a-b) = %+0.2f reward | Welch t=%.2f p=%.4f" % (voi, t, p))
        t_s, p_s = stats.ttest_ind(fa_s, fz_s, equal_var=False)
        print("success gap = %+0.2f | p=%.4f" %
              (np.mean(fa_s) - np.mean(fz_s), p_s))
        if p < 0.05 and voi > 3:
            print("VERDICT: tracking is learned and pays -> proceed to the "
                  "delay matrix (exp10).")
        elif p >= 0.05:
            print("VERDICT: no significant tracking value -> run the sweep:")
            print("  python scripts/run_experiments.py --group exp9b_sweep "
                  "--max-parallel 8")
        else:
            print("VERDICT: tracking value marginal -> consider larger sigma "
                  "or different rho (sweep).")


if __name__ == "__main__":
    main()
