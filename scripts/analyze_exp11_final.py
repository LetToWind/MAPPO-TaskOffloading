"""Aggregate exp11 structured Phase-B results across seeds 1..10."""

from __future__ import print_function

import csv
import glob
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DIRS = [ROOT / "results" / "exp11_phase_b_structured",
        ROOT / "results" / "exp11_phase_b_structured_more"]
PHASE_A_FULL_INFO = 0.6433


def collect(prefix, eval_file=True, tail=3):
    out = {}
    suffix = "_eval.csv" if eval_file else ".csv"
    for base in DIRS:
        for path in glob.glob(str(base / ("phase0_%s_s*%s" %
                                           (prefix, suffix)))):
            if not eval_file and (path.endswith("_eval.csv") or
                                  path.endswith("_diag.csv")):
                continue
            match = re.search(r"_s(\d+)", Path(path).name)
            if not match:
                continue
            seed = int(match.group(1))
            with open(path) as fh:
                rows = [r for r in csv.reader(fh) if r and r[0].isdigit()]
            if eval_file and len(rows) >= tail:
                part = rows[-tail:]
                out[seed] = {
                    "reward": np.mean([float(r[1]) for r in part]),
                    "success": np.mean([float(r[2]) for r in part]),
                }
            elif not eval_file and rows:
                part = rows[-300:]
                out[seed] = {
                    "arrived": np.mean([float(r[4]) for r in part]),
                    "route": np.mean([float(r[5]) for r in part]),
                    "cond": np.mean([float(r[6]) for r in part]),
                    "top": np.mean([float(r[7]) for r in part]),
                    "load_std": np.mean([float(r[8]) for r in part]),
                }
    return out


def bootstrap_ci(values, seed=20260927, n_boot=50000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    sims = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(1)
    return tuple(np.percentile(sims, [2.5, 97.5]))


def main():
    live = collect("struct_live")
    frozen = collect("struct_frozen")
    live_train = collect("struct_live", eval_file=False)
    frozen_train = collect("struct_frozen", eval_file=False)
    seeds = sorted(set(live) & set(frozen))
    if not seeds:
        raise SystemExit("no paired results")
    ls = np.array([live[s]["success"] for s in seeds])
    fs = np.array([frozen[s]["success"] for s in seeds])
    lr = np.array([live[s]["reward"] for s in seeds])
    fr = np.array([frozen[s]["reward"] for s in seeds])
    diff = ls - fs
    ci = bootstrap_ci(diff)
    arrivals = np.mean([live_train[s]["arrived"] for s in seeds
                        if s in live_train])

    # Forced-local MAPPO mean from the three-seed pilot.
    fl_files = glob.glob(str(ROOT / "results" / "exp11_phase_b_pilot" /
                             "phase0_fl_s*_eval.csv"))
    fl = []
    for path in fl_files:
        with open(path) as fh:
            rows = [r for r in csv.reader(fh) if r and r[0].isdigit()]
        fl.append(np.mean([float(r[2]) for r in rows[-3:]]))
    fl_mean = float(np.mean(fl))
    b1_need = fl_mean + 0.5 * (PHASE_A_FULL_INFO * arrivals - fl_mean)

    print("=== exp11 structured Phase-B aggregate ===")
    print("paired seeds:", seeds)
    print("live   reward %.2f +/- %.2f success %.2f +/- %.2f" %
          (lr.mean(), lr.std(ddof=1), ls.mean(), ls.std(ddof=1)))
    print("frozen reward %.2f +/- %.2f success %.2f +/- %.2f" %
          (fr.mean(), fr.std(ddof=1), fs.mean(), fs.std(ddof=1)))
    print("B1 live %.2f vs halfway %.2f [%s]" %
          (ls.mean(), b1_need, "PASS" if ls.mean() >= b1_need else "FAIL"))
    print("B2 diff %+.2f tasks (%.2f pp) CI[%.2f, %.2f] [%s]" %
          (diff.mean(), 100 * diff.mean() / arrivals, ci[0], ci[1],
           "PASS" if diff.mean() >= 0.03 * arrivals and ci[0] > 0 else
           "FAIL"))
    for name, data in (("live", live_train), ("frozen", frozen_train)):
        vals = [data[s] for s in seeds if s in data]
        print("%-7s route %.3f cond %.3f top %.3f load_std %.3f" %
              (name, np.mean([x["route"] for x in vals]),
               np.mean([x["cond"] for x in vals]),
               np.mean([x["top"] for x in vals]),
               np.mean([x["load_std"] for x in vals])))


if __name__ == "__main__":
    main()
