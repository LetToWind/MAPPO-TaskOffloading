"""Analyze direct delay injection on the 10 structured MAPPO models."""

from __future__ import print_function

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "results" / "exp11_phase_c_direct.csv"
ORDER = ("none", "fixed1", "fixed2", "fixed3", "unfixed1-3", "frozen")


def bootstrap_ci(values, seed=20260927, n_boot=50000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    sims = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(1)
    return tuple(np.percentile(sims, [2.5, 97.5]))


def main():
    with open(PATH) as fh:
        rows = list(csv.DictReader(fh))
    data = {}
    for row in rows:
        data[(int(row["model_seed"]), row["mode"])] = {
            "success": float(row["success"]),
            "rate": float(row["success_rate"]),
            "reward": float(row["reward"]),
        }
    seeds = sorted(set(seed for seed, _ in data))
    print("=== exp11 Phase-C direct delay test ===")
    print("models=%d, eval episodes/model/mode=%s" %
          (len(seeds), rows[0]["episodes"] if rows else "?"))
    base = np.array([data[(s, "none")]["success"] for s in seeds])
    for mode in ORDER:
        vals = np.array([data[(s, mode)]["success"] for s in seeds])
        rates = np.array([data[(s, mode)]["rate"] for s in seeds])
        rewards = np.array([data[(s, mode)]["reward"] for s in seeds])
        diff = base - vals
        ci = bootstrap_ci(diff)
        relative = diff.mean() / max(base.mean(), 1e-9)
        print("%-12s success %5.2f +/- %.2f (%5.2f%%) reward %6.2f | "
              "drop %5.1f%% CI_tasks[%.2f, %.2f]" %
              (mode, vals.mean(), vals.std(ddof=1), 100 * rates.mean(),
               rewards.mean(), 100 * relative, ci[0], ci[1]))
    means = {mode: np.mean([data[(s, mode)]["success"] for s in seeds])
             for mode in ORDER}
    monotone = (means["none"] >= means["fixed1"] >= means["fixed2"] >=
                means["fixed3"])
    d2_drop = (means["none"] - means["fixed2"]) / means["none"]
    d2_ci = bootstrap_ci([
        data[(s, "none")]["success"] - data[(s, "fixed2")]["success"]
        for s in seeds])
    print("\nC gate: dose_monotone=%s, d2_drop=%.1f%%, d2_CI=[%.2f, %.2f] [%s]" %
          (monotone, 100 * d2_drop, d2_ci[0], d2_ci[1],
           "PASS" if monotone and d2_drop >= 0.10 and d2_ci[0] > 0 else
           "FAIL"))


if __name__ == "__main__":
    main()
