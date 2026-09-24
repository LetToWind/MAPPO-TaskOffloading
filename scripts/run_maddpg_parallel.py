"""Parallel multi-seed runner for the ORIGINAL maddpg_discrete.py baseline.

Each seed runs in an isolated working-directory copy (wd{k}) of the project
sources so the hardcoded output paths (sum_reward_discrete.csv, model/) do
not collide. A directory junction <out>/data -> ../../../data provides the
"../data/*.csv" lookups the original code performs relative to CWD.

Protocol matches exp5 (MAPPO): unmodified training code, fixed variant-25
topology (USE_DYNAMIC_TOPOLOGY=0), N_EPISODES per seed, crash-retry,
greedy eval of the final checkpoint on the common deterministic scenario.

Usage:
    python scripts/run_maddpg_parallel.py --seeds 5 --episodes 4000
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "exp6_original_maddpg"
SRC = PROJECT_ROOT
COPY_FILES = [
    "maddpg_discrete.py", "Environment_marl_discrete.py",
    "model_agent_discrete_maddpg.py", "replay_buffer_discrete.py",
    "plot_training.py", "__init__.py",
]
COPY_DIRS = ["config", "init_input", "experiment"]


def csv_done(path, episodes):
    if not path.exists():
        return False
    n = 0
    with open(path) as f:
        for r in csv.reader(f):
            if r and r[0].isdigit():
                n += 1
    return n >= episodes * 0.95


def make_wd(k):
    wd = OUT_DIR / ("wd%d" % k)
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    for f in COPY_FILES:
        shutil.copy(str(SRC / f), str(wd / f))
    for d in COPY_DIRS:
        shutil.copytree(str(SRC / d), str(wd / d),
                        ignore=shutil.ignore_patterns("__pycache__"))
    return wd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=4000)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    junction = OUT_DIR / "data"
    if not junction.exists():
        os.system('mklink /J "%s" "%s"' % (junction, PROJECT_ROOT.parent / "data"))

    env = dict(os.environ)
    env.update({
        "USE_DYNAMIC_TOPOLOGY": "0",
        "N_EPISODES": str(args.episodes),
    })

    t0 = time.time()
    procs = {}
    for k in range(1, args.seeds + 1):
        wd = make_wd(k)
        logf = open(OUT_DIR / ("seed%d.log" % k), "w")
        p = subprocess.Popen(
            [sys.executable, "-X", "utf8", "-W", "ignore", "maddpg_discrete.py"],
            cwd=str(wd), env=env, stdout=logf, stderr=subprocess.STDOUT)
        procs[k] = (p, wd, logf, 0)
        print("[%6.1fmin] launched seed %d in %s" %
              ((time.time() - t0) / 60, k, wd), flush=True)

    while procs:
        time.sleep(30)
        for k in list(procs):
            p, wd, logf, attempts = procs[k]
            if p.poll() is None:
                continue
            logf.close()
            csvp = wd / "sum_reward_discrete.csv"
            if csv_done(csvp, args.episodes):
                shutil.move(str(csvp), str(OUT_DIR / ("seed%d.csv" % k)))
                subprocess.call(
                    [sys.executable, "-X", "utf8", "-W", "ignore",
                     str(PROJECT_ROOT / "scripts" / "eval_maddpg_original.py"),
                     "--checkpoint", str(wd / "model" / "maddpg_final"),
                     "--episodes", "100",
                     "--out", str(OUT_DIR / ("seed%d_eval.txt" % k))],
                    cwd=str(PROJECT_ROOT), env=env,
                    stdout=open(OUT_DIR / ("seed%d_eval.log" % k), "w"),
                    stderr=subprocess.STDOUT)
                print("[%6.1fmin] seed %d DONE" % ((time.time() - t0) / 60, k),
                      flush=True)
                del procs[k]
            elif attempts < 5:
                attempts += 1
                logf = open(OUT_DIR / ("seed%d.log" % k), "a")
                p = subprocess.Popen(
                    [sys.executable, "-X", "utf8", "-W", "ignore",
                     "maddpg_discrete.py"],
                    cwd=str(wd), env=env, stdout=logf, stderr=subprocess.STDOUT)
                procs[k] = (p, wd, logf, attempts)
                print("[%6.1fmin] seed %d retry %d" %
                      ((time.time() - t0) / 60, k, attempts), flush=True)
            else:
                print("[%6.1fmin] seed %d FAILED" % ((time.time() - t0) / 60, k),
                      flush=True)
                del procs[k]
    print("ALL DONE in %.1f min" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
