import csv
import glob
import os

import numpy as np

base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "results", "exp5_original_baseline")
rows_out = []
for path in sorted(glob.glob(os.path.join(base, "seed[0-9].csv"))):
    rows = []
    for r in csv.reader(open(path)):
        if r and r[0].isdigit():
            rows.append((int(r[0]), float(r[1]), int(r[2]), int(r[3])))
    tail = rows[-500:]
    tr_r = np.mean([x[1] for x in tail])
    tr_s = np.mean([x[2] for x in tail])
    ev = open(path.replace(".csv", "_eval.txt")).read()
    ev_r = float(ev.split("reward=")[1].split("+")[0])
    ev_s = float(ev.split("success=")[1].split()[0])
    rows_out.append((os.path.basename(path), tr_r, tr_s, ev_r, ev_s))

print("%-10s %18s %18s" % ("seed", "train_tail_reward", "greedy_eval_reward"))
for name, tr_r, tr_s, ev_r, ev_s in rows_out:
    print("%-10s %8.2f (s=%.1f) %10.2f (s=%.1f)" % (name, tr_r, tr_s, ev_r, ev_s))
tr = [x[1] for x in rows_out]
ev = [x[3] for x in rows_out]
evs = [x[4] for x in rows_out]
print("\ntrain_tail : %.2f +/- %.2f" % (np.mean(tr), np.std(tr, ddof=1)))
print("greedy_eval: %.2f +/- %.2f  (success %.2f +/- %.2f)" %
      (np.mean(ev), np.std(ev, ddof=1), np.mean(evs), np.std(evs, ddof=1)))
