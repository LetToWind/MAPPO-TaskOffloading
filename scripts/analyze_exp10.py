"""exp10 Phase 0' calibration analysis.

Arms (results/exp10_calib/):
  fl_s*     forced-local training (Level-0 ceiling)
  none_s*   free routing, live neighbor obs (learned Level-2 attempt)
  frozen_s* free routing, neighbor obs frozen at episode start (inertia/VoI)
  oracle_result.csv   JSQ oracle upper bound (true loads)

Gates (doc section 9.0):
  G1 playability : oracle.success - fl.success >= 15 tasks/episode
  G2 VoI         : none.success - frozen.success significant (p<0.05)
  G3 learnability: none >= fl + 0.5*(oracle - fl)
Also prints canary curves (route rate / cond route rate) per arm.
"""

import csv
import glob
import os
import sys

import numpy as np
from scipy import stats

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "results", "exp10_calib")


def eval_stats(pattern, tail=3):
    out = []
    for path in sorted(glob.glob(os.path.join(base, pattern))):
        rows = []
        for r in csv.reader(open(path)):
            if r and r[0].isdigit():
                rows.append([float(x) for x in r])
        if len(rows) < 3:
            continue
        t = rows[-tail:]
        out.append((np.mean([r[1] for r in t]), np.mean([r[2] for r in t]),
                    np.mean([r[4] for r in t])))   # reward, success, route
    return out


def diag_tail(pattern, tail=300):
    """last-N-episode canaries from the train csv (exclude _eval/_diag)"""
    rates = {"rr": [], "cr": [], "ts": []}
    for path in sorted(glob.glob(os.path.join(base, pattern))):
        if path.endswith("_eval.csv") or path.endswith("_diag.csv"):
            continue
        rows = [r for r in csv.reader(open(path)) if r and r[0].isdigit()]
        if not rows:
            continue
        t = rows[-tail:]
        rates["rr"].append(np.mean([float(r[5]) for r in t]))
        rates["cr"].append(np.mean([float(r[6]) for r in t]))
        rates["ts"].append(np.mean([float(r[7]) for r in t]))
    return rates


def main():
    fl = eval_stats("phase0_fl_s*_eval.csv")
    free = eval_stats("phase0_none_s*_eval.csv")
    fz = eval_stats("phase0_frozen_s*_eval.csv")
    oracle = None
    opath = os.path.join(base, "oracle_result.csv")
    if os.path.exists(opath):
        rows = list(csv.DictReader(open(opath)))
        if rows:
            oracle = float(rows[-1]["success"])

    def show(name, data):
        if not data:
            print("%-12s no data" % name)
            return None, None
        r = [x[0] for x in data]
        s = [x[1] for x in data]
        rt = [x[2] for x in data]
        print("%-12s n=%d reward %7.2f+/-%5.2f success %5.2f+/-%4.2f "
              "eval_route %.1f" % (name, len(r), np.mean(r),
                                   np.std(r, ddof=1) if len(r) > 1 else 0,
                                   np.mean(s),
                                   np.std(s, ddof=1) if len(s) > 1 else 0,
                                   np.mean(rt)))
        return np.mean(s), s

    print("=== exp10 Phase 0' calibration ===")
    fl_m, fl_s = show("forced-local", fl)
    fr_m, fr_s = show("free(none)", free)
    fz_m, fz_s = show("frozen", fz)
    print("oracle      success %.2f" % oracle if oracle is not None
          else "oracle      no data (run scripts/oracle_policy.py)")

    print("\n--- gates ---")
    if oracle is not None and fl_m is not None:
        gap = oracle - fl_m
        print("G1 playability: oracle - fl = %.2f tasks  [%s] (need >= 15)"
              % (gap, "PASS" if gap >= 15 else "FAIL"))
    if fr_m is not None and fz_m is not None and fr_s and fz_s:
        t, p = stats.ttest_ind(fr_s, fz_s, equal_var=False)
        print("G2 VoI (none vs frozen): +%.2f success, p=%.4f [%s]"
              % (fr_m - fz_m, p, "PASS" if p < 0.05 else "FAIL"))
    if oracle is not None and fl_m is not None and fr_m is not None:
        need = fl_m + 0.5 * (oracle - fl_m)
        print("G3 learnability: none %.2f vs need %.2f [%s]"
              % (fr_m, need, "PASS" if fr_m >= need else "FAIL"))

    print("\n--- canaries (train tail-300) ---")
    for name, pat in (("fl", "phase0_fl_s*.csv"), ("none", "phase0_none_s*.csv"),
                      ("frozen", "phase0_frozen_s*.csv")):
        d = diag_tail(pat)
        if d["rr"]:
            print("%-8s route_rate %.2f | cond_route_rate %.2f | top_share %.2f"
                  % (name, np.mean(d["rr"]), np.mean(d["cr"]),
                     np.mean(d["ts"])))


if __name__ == "__main__":
    main()
