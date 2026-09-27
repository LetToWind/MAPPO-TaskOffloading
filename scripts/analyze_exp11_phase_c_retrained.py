"""Compare direct delay damage with delay-trained Base MAPPO."""

from __future__ import print_function

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DIRECT = ROOT / "results" / "exp11_phase_c_direct.csv"
RETRAINED = ROOT / "results" / "exp11_phase_c_retrained_eval.csv"
ARMS = ("fixed2", "unfixed1-3")


def read_rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def bootstrap_ci(values, seed=20260927, n_boot=50000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    sims = rng.choice(values, size=(n_boot, len(values)),
                      replace=True).mean(axis=1)
    return tuple(np.percentile(sims, [2.5, 97.5]))


def main():
    direct_rows = read_rows(DIRECT)
    retrain_rows = read_rows(RETRAINED)
    direct = {(int(r["model_seed"]), r["mode"]): r for r in direct_rows}
    retrained = {(int(r["model_seed"]), r["train_mode"], r["eval_mode"]): r
                 for r in retrain_rows}

    print("=== exp11 Phase-C delayed retraining ===")
    for arm in ARMS:
        seeds = sorted(set(s for s, mode in direct if mode in ("none", arm)) &
                       set(s for s, train, mode in retrained
                           if train == arm and mode == arm))
        if len(seeds) != 10:
            raise SystemExit("expected 10 paired %s seeds, got %d" %
                             (arm, len(seeds)))
        clean = np.array([float(direct[(s, "none")]["success"])
                          for s in seeds])
        deployed = np.array([float(direct[(s, arm)]["success"])
                             for s in seeds])
        adapted = np.array([float(retrained[(s, arm, arm)]["success"])
                            for s in seeds])
        gain = adapted - deployed
        gap = clean - deployed
        gain_ci = bootstrap_ci(gain)
        recovery = gain.mean() / gap.mean() if gap.mean() > 0 else np.nan
        arrived = np.mean([float(direct[(s, arm)]["arrived"])
                           for s in seeds])
        print("%-11s none %.2f direct %.2f retrained %.2f" %
              (arm, clean.mean(), deployed.mean(), adapted.mean()))
        print("  adaptation gain %+.2f tasks (%+.2f pp), "
              "CI[%.2f, %.2f], recovery %.1f%%" %
              (gain.mean(), 100 * gain.mean() / arrived,
               gain_ci[0], gain_ci[1], 100 * recovery))
        print("  delay-trained Base vs none gap %+.2f tasks" %
              (adapted.mean() - clean.mean()))


if __name__ == "__main__":
    main()
