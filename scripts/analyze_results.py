"""Analysis for organized Phase-0 result folders.

Usage:
    python scripts/analyze_results.py seal
        Full sealing report:
          A: 5-seed Welch t-test, CH=3 none vs unfixed_1-4 (exp3+exp4 merged)
          B: dose-response table + png (unfixed ranges at CH=3)
          C: strict 2x2 interaction table (CH 6/3 x none/unfixed_1-4)
    python scripts/analyze_results.py group exp3_fixed_ch3 c3
        Generic per-folder summary (arms auto-detected from file names).
"""

import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
EVAL_TAIL = 5


def seed_eval_stats(path):
    rows = []
    with open(path) as f:
        for r in csv.reader(f):
            if r and r[0].isdigit():
                rows.append([float(x) for x in r])
    if len(rows) < 3:
        return None
    tail = rows[-EVAL_TAIL:]
    return (float(np.mean([r[1] for r in tail])),
            float(np.mean([r[2] for r in tail])))


def collect(folders, pattern):
    """Return {tag: [(reward, success), ...]} for files matching
    results/<folder>/phase0_<pattern>_s*_eval.csv"""
    out = {}
    for folder in folders:
        for path in sorted(glob.glob(str(RESULTS / folder / ("phase0_" + pattern)))):
            tag = Path(path).stem[len("phase0_"):]
            if tag.endswith("_eval"):
                tag = tag[:-5]
            st = seed_eval_stats(path)
            if st:
                out.setdefault(tag, []).append(st)
    return out


def welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return t, p


def analyze_seal():
    lines = []

    def w(s=""):
        print(s)
        lines.append(s)

    # ---------------- A: 5-seed main comparison ----------------
    w("=" * 72)
    w("A. CH=3, none vs unfixed[1,4] — 5 seeds (exp3 s1-3 + exp4 s4-5)")
    w("=" * 72)
    none = collect(["exp3_fixed_ch3", "exp4_sealing", "exp4b_seeds"], "c3_none_s*_eval.csv")
    uf14 = collect(["exp3_fixed_ch3", "exp4_sealing", "exp4b_seeds"], "c3_unfixed_1-4_s*_eval.csv")
    for name, d in (("none", none), ("unfixed_1-4", uf14)):
        for tag in sorted(d):
            w("  %-22s reward=%7.2f success=%5.2f" % (tag, d[tag][0][0], d[tag][0][1]))
    nr = [v[0] for vs in none.values() for v in vs]
    ns = [v[1] for vs in none.values() for v in vs]
    ur = [v[0] for vs in uf14.values() for v in vs]
    us = [v[1] for vs in uf14.values() for v in vs]
    if nr and ur:
        t_r, p_r = welch(nr, ur)
        t_s, p_s = welch(ns, us)
        w("")
        w("  reward  : none %.2f +/- %.2f  vs  unfixed %.2f +/- %.2f   "
          "(%.1f%%)  t=%.2f p=%.4f" %
          (np.mean(nr), np.std(nr), np.mean(ur), np.std(ur),
           (np.mean(ur) - np.mean(nr)) / abs(np.mean(nr)) * 100, t_r, p_r))
        w("  success : none %.2f +/- %.2f  vs  unfixed %.2f +/- %.2f   "
          "(%.1f%%)  t=%.2f p=%.4f" %
          (np.mean(ns), np.std(ns), np.mean(us), np.std(us),
           (np.mean(us) - np.mean(ns)) / np.mean(ns) * 100, t_s, p_s))
        w("  significance (alpha=0.05): reward %s | success %s" %
          ("YES" if p_r < 0.05 else "no", "YES" if p_s < 0.05 else "no"))

    # ---------------- B: dose-response ----------------
    w("")
    w("=" * 72)
    w("B. Dose-response at CH=3 — unfixed delay range vs degradation")
    w("=" * 72)
    ranges = [("1-2", "c3_unfixed_1-2_s*_eval.csv"),
              ("1-3", "c3_unfixed_1-3_s*_eval.csv"),
              ("1-4", "c3_unfixed_1-4_s*_eval.csv"),
              ("2-4", "c3_unfixed_2-4_s*_eval.csv")]
    base_r = np.mean(nr) if nr else float("nan")
    base_s = np.mean(ns) if ns else float("nan")
    w("  %-10s %5s %18s %18s" % ("range", "n", "reward (vs none)", "success (vs none)"))
    dose_x, dose_y = ["none\n0"], [0.0]
    for label, pat in ranges:
        d = collect(["exp3_fixed_ch3", "exp4_sealing", "exp4b_seeds"], pat)
        rs = [v[0] for vs in d.values() for v in vs]
        ss = [v[1] for vs in d.values() for v in vs]
        if not rs:
            w("  %-10s no data" % label)
            continue
        rel_r = (np.mean(rs) - base_r) / abs(base_r) * 100
        rel_s = (np.mean(ss) - base_s) / base_s * 100
        w("  %-10s %5d %8.2f %+7.1f%% %8.2f %+7.1f%%" %
          (label, len(rs), np.mean(rs), rel_r, np.mean(ss), rel_s))
        mean_d = (int(label[0]) + int(label[2])) / 2.0
        dose_x.append(label.replace("-", "\n") + "\nmu=%.1f" % mean_d)
        dose_y.append(rel_r)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(range(len(dose_y)), dose_y, color=["gray"] + ["crimson"] * (len(dose_y) - 1))
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(dose_x)))
        ax.set_xticklabels(dose_x, fontsize=8)
        ax.set_ylabel("eval reward vs none (%)")
        ax.set_title("Dose-response: unfixed delay range (CH=3, fixed topology)")
        ax.grid(axis="y", alpha=0.3)
        png = RESULTS / "exp4_sealing" / "dose_response.png"
        fig.tight_layout()
        fig.savefig(png, dpi=150)
        w("  plot -> %s" % png)
    except Exception as e:  # noqa: BLE001
        w("  (plot skipped: %s)" % e)

    # ---------------- C: strict 2x2 ----------------
    w("")
    w("=" * 72)
    w("C. 2x2 interaction: congestion (CH 6/3) x delay (none / unfixed[1,4])")
    w("=" * 72)
    ch6_none = collect(["exp2_fixed_ch6"], "fx_none_s*_eval.csv")
    ch6_uf = collect(["exp4_sealing"], "ch6_unfixed_1-4_s*_eval.csv")
    ch3_none, ch3_uf = none, uf14

    def cell(d):
        rs = [v[0] for vs in d.values() for v in vs]
        ss = [v[1] for vs in d.values() for v in vs]
        if not rs:
            return "      no data"
        return "%7.2f (n=%d) success %5.2f" % (np.mean(rs), len(rs), np.mean(ss))

    w("  %-22s %-12s %-12s" % ("", "no delay", "unfixed [1,4]"))
    w("  %-22s %s" % ("CH=6 (uncongested)", cell(ch6_none)))
    w("  %-34s%s" % ("", cell(ch6_uf)))
    w("  %-22s %s" % ("CH=3 (congested)", cell(ch3_none)))
    w("  %-34s%s" % ("", cell(ch3_uf)))
    w("")
    w("  note: CH=6 none cell is from exp2 (3000 eps); others 4000 eps.")

    report = RESULTS / "exp4_sealing" / "seal_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\nreport -> %s" % report)


def analyze_group(folder, prefix):
    base = RESULTS / folder
    paths = sorted(glob.glob(str(base / ("phase0_%s_*_s*_eval.csv" % prefix))))
    arms = {}
    for path in paths:
        stem = Path(path).stem[len("phase0_"):]
        stem = stem[:-len("_eval")] if stem.endswith("_eval") else stem
        arm = stem.rsplit("_s", 1)[0]
        st = seed_eval_stats(path)
        if st:
            arms.setdefault(arm, []).append(st)
    if not arms:
        print("no eval data under", base)
        return
    oracle = arms.get("none")
    oracle_r = np.mean([v[0] for v in oracle]) if oracle else float("nan")
    print("%-18s %4s %18s %14s %10s" %
          ("arm", "n", "eval_reward", "eval_success", "vs none"))
    for arm in sorted(arms):
        rs = [v[0] for v in arms[arm]]
        ss = [v[1] for v in arms[arm]]
        print("%-18s %4d %8.2f +/- %5.2f %6.2f +/- %3.2f %+9.1f%%" %
              (arm, len(rs), np.mean(rs), np.std(rs), np.mean(ss), np.std(ss),
               (np.mean(rs) - oracle_r) / abs(oracle_r) * 100.0))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "seal":
        analyze_seal()
    elif len(sys.argv) > 2 and sys.argv[1] == "group":
        analyze_group(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
    else:
        print(__doc__)

