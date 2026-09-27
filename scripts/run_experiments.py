"""Experiment runner: launches Phase-0 training runs grouped into folders.

Each run writes phase0_{tag}.csv / phase0_{tag}_eval.csv into
results/<group>/ and its console log into results/<group>/logs/.

Usage:
    python scripts/run_experiments.py --list
    python scripts/run_experiments.py --group exp4_sealing --max-parallel 8
    python scripts/run_experiments.py --group exp4_sealing --only A --max-parallel 4

Runs are defined declaratively below. Conditions are passed to
train_mappo_phase0.py via environment variables (see that file's docstring).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
RUNNER = PROJECT_ROOT / "train_mappo_phase0.py"
N_EPISODES_DEFAULT = 4000


def run(tag, ch, mode, seed, d_value=None, d_min=None, d_max=None,
        episodes=N_EPISODES_DEFAULT, update_every=25, fading=False,
        rho=0.7, sigma=6.0):
    env = {
        "TAG": tag,
        "TOPOLOGY": "fixed",
        "CHANNEL_NUM": str(ch),
        "DELAY_MODE": mode,
        "SEED": str(seed),
        "N_EPISODES": str(episodes),
        "EVAL_EVERY": "100",
        "EVAL_EPS": "20",
        "TF_INTRA": "2",
        "TF_INTER": "1",
        "UPDATE_EVERY": str(update_every),
    }
    if fading:
        env["FADING"] = "1"
        env["FADING_RHO"] = str(rho)
        env["FADING_SIGMA_DB"] = str(sigma)
    if d_value is not None:
        env["DELAY_VALUE"] = str(d_value)
    if d_min is not None:
        env["DELAY_MIN"] = str(d_min)
    if d_max is not None:
        env["DELAY_MAX"] = str(d_max)
    return env


def build_exp4_sealing():
    runs = []
    # A: statistical power — extra seeds for the two critical arms
    for s in (4, 5):
        runs.append(("A", run("c3_none_s%d" % s, 3, "none", s)))
        runs.append(("A", run("c3_unfixed_1-4_s%d" % s, 3, "unfixed", s, d_min=1, d_max=4)))
    # B: dose-response — delay-range scaling at CH=3
    for rng in ((1, 2), (1, 3), (2, 4)):
        for s in (1, 2, 3):
            runs.append(("B", run("c3_unfixed_%d-%d_s%d" % (rng[0], rng[1], s),
                                  3, "unfixed", s, d_min=rng[0], d_max=rng[1])))
    # C: complete the strict 2x2 — unfixed [1,4] at CH=6
    for s in (1, 2, 3):
        runs.append(("C", run("ch6_unfixed_1-4_s%d" % s, 6, "unfixed", s, d_min=1, d_max=4)))
    return runs


def dynrun(tag, mode, seed, route_mask="", lam_hot=1.5, ch=3,
           episodes=3000):
    env = {
        "_SCRIPT": "train_mappo_dyn.py",
        "TAG": tag,
        "DELAY_MODE": mode,
        "SEED": str(seed),
        "N_EPISODES": str(episodes),
        "EVAL_EVERY": "200",
        "EVAL_EPS": "20",
        "TF_INTRA": "2",
        "TF_INTER": "1",
        "UPDATE_EVERY": "25",
        "DYN_LAM_HOT": str(lam_hot),
        "DYN_CH": str(ch),
    }
    if route_mask:
        env["ROUTE_MASK"] = route_mask
    return env


def dynrun_v11(tag, mode, seed, route_mask="", episodes=1500,
               share_actor=False, save_model=False,
               random_neighbor_order=False, structured_route=False,
               d_value=None, d_min=None, d_max=None):
    """Phase-B MAPPO pilot on the Phase-A-qualified v1.1 scenario."""
    env = dynrun(tag, mode, seed, route_mask=route_mask, lam_hot=0.6,
                 ch=3, episodes=episodes)
    env.update({
        "DYN_LAM0": "0.1",
        "DYN_DWELL": "10",
        "DYN_N_HOTSPOTS": "2",
        "DYN_CAPACITY_MARKOV": "1",
        "DYN_CAPACITY_STAY": "0.85",
        "DYN_HOTSPOT_CAP_PENALTY": "1",
        "DYN_DDL_MIN": "5",
        "DYN_DDL_MAX": "11",
        "DYN_DATA_MIN": "100000",
        "DYN_DATA_MAX": "300000",
    })
    if share_actor:
        env["SHARE_ACTOR"] = "1"
    if save_model:
        env["SAVE_MODEL"] = "1"
    if random_neighbor_order:
        env["DYN_RANDOM_NEIGHBOR_ORDER"] = "1"
    if structured_route:
        env["STRUCTURED_ROUTE"] = "1"
    if d_value is not None:
        env["DELAY_VALUE"] = str(d_value)
    if d_min is not None:
        env["DELAY_MIN"] = str(d_min)
    if d_max is not None:
        env["DELAY_MAX"] = str(d_max)
    return env


GROUPS = {
    # Phase B pilot after exp11 Phase-A gates passed.  Three seeds are a
    # screening run; expand to >=10 only if the live-vs-frozen effect survives.
    "exp11_phase_b_pilot": lambda: [
        ("FL", dynrun_v11("fl_s%d" % s, "none", s,
                          route_mask="local"))
        for s in range(1, 4)
    ] + [
        ("LIVE", dynrun_v11("live_s%d" % s, "none", s))
        for s in range(1, 4)
    ] + [
        ("FROZEN", dynrun_v11("frozen_s%d" % s, "frozen", s))
        for s in range(1, 4)
    ],
    "exp11_phase_b_shared": lambda: [
        ("LIVE", dynrun_v11("shared_live_s%d" % s, "none", s,
                            share_actor=True, save_model=True))
        for s in range(1, 4)
    ] + [
        ("FROZEN", dynrun_v11("shared_frozen_s%d" % s, "frozen", s,
                              share_actor=True))
        for s in range(1, 4)
    ],
    "exp11_phase_b_permuted": lambda: [
        ("LIVE", dynrun_v11("perm_live_s%d" % s, "none", s,
                            share_actor=True, save_model=True,
                            random_neighbor_order=True))
        for s in range(1, 4)
    ] + [
        ("FROZEN", dynrun_v11("perm_frozen_s%d" % s, "frozen", s,
                              share_actor=True,
                              random_neighbor_order=True))
        for s in range(1, 4)
    ],
    "exp11_phase_b_structured": lambda: [
        ("LIVE", dynrun_v11("struct_live_s%d" % s, "none", s,
                            share_actor=True, save_model=True,
                            random_neighbor_order=True,
                            structured_route=True))
        for s in range(1, 4)
    ] + [
        ("FROZEN", dynrun_v11("struct_frozen_s%d" % s, "frozen", s,
                              share_actor=True,
                              random_neighbor_order=True,
                              structured_route=True))
        for s in range(1, 4)
    ],
    "exp11_phase_b_structured_more": lambda: [
        ("LIVE", dynrun_v11("struct_live_s%d" % s, "none", s,
                            share_actor=True, save_model=True,
                            random_neighbor_order=True,
                            structured_route=True))
        for s in range(4, 11)
    ] + [
        ("FROZEN", dynrun_v11("struct_frozen_s%d" % s, "frozen", s,
                              share_actor=True,
                              random_neighbor_order=True,
                              structured_route=True))
        for s in range(4, 11)
    ],
    # Phase-C second half: retrain the qualified structured Base MAPPO under
    # representative fixed and stochastic observation delays.  These runs
    # separate irreducible delay damage from damage recoverable by adaptation.
    "exp11_phase_c_retrain": lambda: [
        ("D2", dynrun_v11("struct_fixed_d2_s%d" % s, "fixed", s,
                           share_actor=True, save_model=True,
                           random_neighbor_order=True,
                           structured_route=True, d_value=2))
        for s in range(1, 11)
    ] + [
        ("U13", dynrun_v11("struct_unfixed_1-3_s%d" % s, "unfixed", s,
                            share_actor=True, save_model=True,
                            random_neighbor_order=True,
                            structured_route=True, d_min=1, d_max=3))
        for s in range(1, 11)
    ],
    # Phase 0' calibration: forced-local vs free vs frozen (VoI), no delay
    "exp10_calib": lambda: [
        ("FL", dynrun("fl_s%d" % s, "none", s, route_mask="local"))
        for s in range(1, 6)
    ] + [
        ("FREE", dynrun("none_s%d" % s, "none", s))
        for s in range(1, 6)
    ] + [
        ("FZ", dynrun("frozen_s%d" % s, "frozen", s))
        for s in range(1, 6)
    ],
    # optional hotspot sweep
    "exp10b_hotspot": lambda: [
        ("SW", dynrun("lh%02d_none_s%d" % (int(lh * 10), s), "none", s,
                      lam_hot=lh))
        for lh in (0.4, 0.8)
        for s in range(1, 4)
    ] + [
        ("SW", dynrun("lh%02d_fl_s%d" % (int(lh * 10), s), "none", s,
                      route_mask="local", lam_hot=lh))
        for lh in (0.4, 0.8)
        for s in range(1, 4)
    ],
    "exp4b_seeds": lambda: [
        ("A", run("c3_none_s%d" % s, 3, "none", s))
        for s in (6, 7, 8, 9, 10)
    ] + [
        ("A", run("c3_unfixed_1-4_s%d" % s, 3, "unfixed", s, d_min=1, d_max=4))
        for s in (6, 7, 8, 9, 10)
    ],
    "exp7_varfix": lambda: [
        ("UE", run("u25_none_s%d" % s, 3, "none", s, update_every=25))
        for s in range(1, 11)
    ],
    "exp8_delay_stable": lambda: [
        ("UE", run("u25_fixed_d3_s%d" % s, 3, "fixed", s, d_value=3,
                   update_every=25))
        for s in range(1, 11)
    ] + [
        ("UE", run("u25_fixed_d5_s%d" % s, 3, "fixed", s, d_value=5,
                   update_every=25))
        for s in range(1, 11)
    ] + [
        ("UE", run("u25_unfixed_1-4_s%d" % s, 3, "unfixed", s, d_min=1,
                   d_max=4, update_every=25))
        for s in range(1, 11)
    ],
    # VoI calibration: (a) fading + full obs  vs  (b) fading + frozen
    # interference obs; no-fading reference (c) is exp7. 10 seeds each.
    "exp9_voil": lambda: [
        ("A", run("fa_r07s06_s%d" % s, 3, "none", s, fading=True,
                  rho=0.7, sigma=6.0))
        for s in range(1, 11)
    ] + [
        ("B", run("fz_r07s06_s%d" % s, 3, "frozen", s, fading=True,
                  rho=0.7, sigma=6.0))
        for s in range(1, 11)
    ],
    # optional parameter sweep (run only if exp9 shows VoI ~ 0)
    "exp9b_sweep": lambda: [
        ("SW", run("fa_r%02ds%02d_s%d" % (int(rho * 10), int(sigma), s),
                   3, "none", s, fading=True, rho=rho, sigma=sigma))
        for (rho, sigma) in ((0.5, 6.0), (0.9, 6.0), (0.7, 3.0), (0.7, 9.0))
        for s in range(1, 6)
    ] + [
        ("SW", run("fz_r%02ds%02d_s%d" % (int(rho * 10), int(sigma), s),
                   3, "frozen", s, fading=True, rho=rho, sigma=sigma))
        for (rho, sigma) in ((0.5, 6.0), (0.9, 6.0), (0.7, 3.0), (0.7, 9.0))
        for s in range(1, 6)
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="exp10_calib", choices=sorted(GROUPS))
    ap.add_argument("--only", default=None,
                    help="restrict to sub-group label (e.g. A, B, C)")
    ap.add_argument("--max-parallel", type=int, default=8)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    runs = GROUPS[args.group]()
    if args.only:
        runs = [r for r in runs if r[0] == args.only]
    if args.list:
        for sub, env in runs:
            print(sub, env["TAG"])
        print("total:", len(runs))
        return

    out_dir = RESULTS_ROOT / args.group
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = list(runs)
    running = []  # (pop, tag, logfh)
    t0 = time.time()
    done = 0
    print("launching %d runs into %s (max_parallel=%d)" %
          (len(pending), out_dir, args.max_parallel), flush=True)
    while pending or running:
        while pending and len(running) < args.max_parallel:
            sub, env = pending.pop(0)
            script = env.pop("_SCRIPT", "train_mappo_phase0.py")
            tag = env["TAG"]
            full_env = dict(os.environ)
            full_env.update(env)
            full_env["OUT_DIR"] = str(out_dir)
            lf = open(log_dir / ("%s.log" % tag), "w")
            pop = subprocess.Popen(
                [sys.executable, "-X", "utf8", "-W", "ignore", str(PROJECT_ROOT / script)],
                cwd=str(PROJECT_ROOT), env=full_env,
                stdout=lf, stderr=subprocess.STDOUT)
            running.append((pop, tag, lf))
            print("[%6.1fmin] started %s (%s) — %d running, %d pending" %
                  ((time.time() - t0) / 60, tag, sub, len(running), len(pending)),
                  flush=True)
        time.sleep(5)
        still = []
        for pop, tag, lf in running:
            if pop.poll() is None:
                still.append((pop, tag, lf))
            else:
                lf.close()
                done += 1
                print("[%6.1fmin] finished %s (rc=%d) — %d/%d done" %
                      ((time.time() - t0) / 60, tag, pop.returncode, done,
                       len(runs)), flush=True)
        running = still
    print("ALL DONE in %.1f min" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
