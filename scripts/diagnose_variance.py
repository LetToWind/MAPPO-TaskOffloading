"""Diagnose seed-variance patterns in exp3/exp4 c3_none runs (CH=3, no delay).

For each seed: success averaged over 5 training stages; classify as
collapsed / mid / converged by final-stage success; print divergence point.
"""
import csv
import glob
import os

import numpy as np

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
rows_by_seed = {}
for folder in ("exp3_fixed_ch3", "exp4_sealing", "exp4b_seeds"):
    for path in sorted(glob.glob(os.path.join(base, folder, "phase0_c3_none_s*.csv"))):
        if path.endswith("_eval.csv"):
            continue
        tag = os.path.basename(path)[len("phase0_"):-4]
        rows = []
        for r in csv.reader(open(path)):
            if r and r[0].isdigit():
                rows.append((int(r[0]), float(r[1]), int(r[2]), int(r[3])))
        if rows:
            rows_by_seed[tag] = rows

stages = [(1, 500), (501, 1000), (1001, 2000), (2001, 3000), (3001, 4000)]
print("%-20s %s | final" % ("seed", " ".join("s%4d" % s[1] for s in stages)))
finals = {}
for tag, rows in sorted(rows_by_seed.items()):
    vals = []
    for lo, hi in stages:
        seg = [x[2] for x in rows if lo <= x[0] <= hi]
        vals.append(np.mean(seg) if seg else float("nan"))
    finals[tag] = vals[-1]
    print("%-20s %s | %5.1f" % (tag, " ".join("%5.1f" % v for v in vals), vals[-1]))

good = [t for t, v in finals.items() if v >= 13]
bad = [t for t, v in finals.items() if v < 13]
print("\ncollapsed (<13 success @end): %d seeds %s" % (len(bad), sorted(bad)))
print("converged (>=13): %d seeds %s" % (len(good), sorted(good)))

# first episode where good-group and bad-group success curves separate by >1.5
g = {t: rows_by_seed[t] for t in good}
b = {t: rows_by_seed[t] for t in bad}
print("\nstage-mean success per group:")
for lo, hi in stages:
    gs = np.mean([x[2] for t in g for x in g[t] if lo <= x[0] <= hi])
    bs = np.mean([x[2] for t in b for x in b[t] if lo <= x[0] <= hi])
    print("  ep %4d-%4d: good %5.2f  bad %5.2f  gap %5.2f" % (lo, hi, gs, bs, gs - bs))

# reward check: do bad seeds go NEGATIVE (idle collapse => fewer -5 fail penalties)?
print("\nstage-mean reward per group:")
for lo, hi in stages:
    gr = np.mean([x[1] for t in g for x in g[t] if lo <= x[0] <= hi])
    br = np.mean([x[1] for t in b for x in b[t] if lo <= x[0] <= hi])
    print("  ep %4d-%4d: good %7.2f  bad %7.2f" % (lo, hi, gr, br))
