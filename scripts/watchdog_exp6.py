"""Watchdog for exp6: when each remaining MADDPG seed reaches TARGET_EPS
episodes, kill the process (saving wall time), move its CSV, and run the
greedy evaluation against the checkpoint saved at the 1000-episode boundary.

Usage:
    python scripts/watchdog_exp6.py --target 2000 --seeds 1,2,3
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "exp6_original_maddpg"


def csv_rows(path):
    n = 0
    try:
        with open(path) as f:
            for line in f:
                if line[:1].isdigit():
                    n += 1
    except OSError:
        pass
    return n


def kill_tree(pid):
    try:
        import psutil
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except Exception:
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(pid)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--seeds", default="1,2,3")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    # python processes sorted by start time, excluding self
    import os
    me = os.getpid()
    procs = [p for p in subprocess_objs() if p.pid != me]
    if len(procs) != len(seeds):
        print("WARNING: expected %d training processes, found %d" %
              (len(seeds), len(procs)), flush=True)
    proc_by_seed = dict(zip(seeds, procs))

    done = set()
    t0 = time.time()
    while len(done) < len(seeds):
        time.sleep(60)
        for k in seeds:
            if k in done:
                continue
            wd = OUT_DIR / ("wd%d" % k)
            csvp = wd / "sum_reward_discrete.csv"
            n = csv_rows(csvp)
            if n < args.target:
                continue
            proc = proc_by_seed.get(k)
            if proc is not None and proc.poll() is None:
                kill_tree(proc.pid)
                time.sleep(3)
            shutil.move(str(csvp), str(OUT_DIR / ("seed%d.csv" % k)))
            ckpt = wd / "model" / ("maddpg-%d" % (args.target // 1000 * 1000))
            subprocess.call(
                [sys.executable, "-X", "utf8", "-W", "ignore",
                 str(PROJECT_ROOT / "scripts" / "eval_maddpg_original.py"),
                 "--checkpoint", str(ckpt),
                 "--episodes", "100",
                 "--out", str(OUT_DIR / ("seed%d_eval.txt" % k))],
                cwd=str(PROJECT_ROOT),
                stdout=open(OUT_DIR / ("seed%d_eval.log" % k), "w"),
                stderr=subprocess.STDOUT)
            done.add(k)
            print("[%6.1fmin] seed %d harvested at %d eps" %
                  ((time.time() - t0) / 60, k, n), flush=True)
    print("ALL DONE in %.1f min" % ((time.time() - t0) / 60), flush=True)


def subprocess_objs():
    """Return live python Popen-like objects (use psutil if available)."""
    class P:  # minimal shim
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            r = subprocess.call(
                ["powershell", "-NoProfile", "-Command",
                 "if (Get-Process -Id %d -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" % self.pid])
            return None if r == 0 else 1

    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-Process python -ErrorAction SilentlyContinue | Sort-Object StartTime | ForEach-Object { $_.Id }"],
        capture_output=True, text=True)
    pids = [int(x) for x in out.stdout.split()]
    return [P(pid) for pid in pids]


if __name__ == "__main__":
    main()
