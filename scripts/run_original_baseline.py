"""Sequential multi-seed runner for the ORIGINAL train_mappo.py baseline.

Does not modify train_mappo.py. For each seed:
  1. run `python train_mappo.py` with USE_SPMARL=0 USE_DYNAMIC_TOPOLOGY=0
     FIXED_CURRICULUM=0 N_EPISODES=...  (fixed variant-25 topology, 26-dim
     obs, CH=6 -- the original baseline as-is; scenario is re-drawn randomly
     per run because the original builder is unseeded, which is part of the
     original code's run-to-run variance);
  2. retry on startup crash (build_fixed_env_config fails ~1/3 of the time);
  3. move sum_reward_mappo.csv -> results/exp5_original_baseline/seed{k}.csv
     (also the png), keep model/mappo_final for greedy evaluation;
  4. run scripts/eval_original.py for a 100-episode greedy evaluation.

Usage:
    python scripts/run_original_baseline.py --seeds 5 --episodes 4000
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "exp5_original_baseline"
ORIG = PROJECT_ROOT / "train_mappo.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=4000)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "USE_SPMARL": "0",
        "USE_DYNAMIC_TOPOLOGY": "0",
        "FIXED_CURRICULUM": "0",
        "N_EPISODES": str(args.episodes),
    })

    t0 = time.time()
    for k in range(1, args.seeds + 1):
        csv_path = PROJECT_ROOT / "sum_reward_mappo.csv"
        if csv_path.exists():
            csv_path.unlink()
        rc = 1
        for attempt in range(6):
            print("[%6.1fmin] seed %d attempt %d ..." %
                  ((time.time() - t0) / 60, k, attempt + 1), flush=True)
            rc = subprocess.call(
                [sys.executable, "-X", "utf8", "-W", "ignore", str(ORIG)],
                cwd=str(PROJECT_ROOT), env=env,
                stdout=open(OUT_DIR / ("seed%d_run%d.log" % (k, attempt + 1)), "w"),
                stderr=subprocess.STDOUT)
            if rc == 0:
                break
            print("  rc=%d, retrying (scenario builder crash is expected "
                  "~1/3 of the time)" % rc, flush=True)
        if rc != 0:
            print("seed %d FAILED after retries" % k, flush=True)
            continue
        shutil.move(str(csv_path), str(OUT_DIR / ("seed%d.csv" % k)))
        png = PROJECT_ROOT / "sum_reward_mappo.png"
        if png.exists():
            shutil.move(str(png), str(OUT_DIR / ("seed%d.png" % k)))
        # greedy evaluation of the final checkpoint
        subprocess.call(
            [sys.executable, "-X", "utf8", "-W", "ignore",
             str(PROJECT_ROOT / "scripts" / "eval_original.py"),
             "--checkpoint", str(PROJECT_ROOT / "model" / "mappo_final"),
             "--episodes", "100", "--out", str(OUT_DIR / ("seed%d_eval.txt" % k))],
            cwd=str(PROJECT_ROOT), env=env,
            stdout=open(OUT_DIR / ("seed%d_eval.log" % k), "w"),
            stderr=subprocess.STDOUT)
        print("[%6.1fmin] seed %d done" % ((time.time() - t0) / 60, k), flush=True)
    print("ALL DONE in %.1f min" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
