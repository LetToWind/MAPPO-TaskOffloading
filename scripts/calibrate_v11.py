"""Phase-A causal calibration for dynamic offloading v1.1.

This experiment deliberately avoids RL.  Every arm uses the same EDF radio
scheduler and the same exogenous random seed; only routing information differs.
It tests whether fresh neighbor workload has value before RDC-MAPPO is built.

Examples:
  python scripts/calibrate_v11.py --episodes 200
  python scripts/calibrate_v11.py --scan --episodes 80
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_offload_env import (  # noqa: E402
    DynamicOffloadEnv, NOMINAL_BYTES_PER_CHANNEL, POWER_LEVELS,
)


ARMS = ("local", "live", "frozen", "fixed1", "fixed2", "fixed3",
        "unfixed1-3")


def schedule_actions(env):
    """Identical EDF/max-power scheduler used by every routing arm."""
    n = env.n_bs
    ch = np.zeros((n, env.n_ch), dtype=np.int32)
    pw = np.zeros((n, env.max_queue), dtype=np.int32)
    selected = [[] for _ in range(n)]
    caps = env.effective_caps()
    for j in range(n):
        q = env.queue[j]
        order = list(range(min(len(q), env.max_queue)))
        order.sort(key=lambda idx: (env.tasks[q[idx]]["rem"],
                                    -env.tasks[q[idx]]["data"]))
        count = min(len(order), int(caps[j]))
        selected[j] = order[:count]
        for slot in selected[j]:
            pw[j, slot] = len(POWER_LEVELS) - 1

    usage = np.zeros(env.n_ch, dtype=np.int32)
    for j in range(n):
        available = list(range(int(caps[j])))
        for slot in selected[j]:
            c = min(available, key=lambda x: usage[x])
            ch[j, c] = slot + 1
            usage[c] += 1
            available.remove(c)
    return ch, pw


class InformationView(object):
    """Per-source delayed views of entity effective workloads."""

    def __init__(self, arm, n_agents, rng):
        self.arm = arm
        self.n_agents = n_agents
        self.rng = rng
        self.history = []
        self.delays = np.zeros((n_agents, n_agents), dtype=np.int32)
        if arm == "unfixed1-3":
            self.delays = rng.randint(1, 4, size=(n_agents, n_agents))
            np.fill_diagonal(self.delays, 0)

    def observe(self, true_loads, true_caps):
        snapshot = (np.asarray(true_loads, dtype=np.float64).copy(),
                    np.asarray(true_caps, dtype=np.float64).copy())
        self.history.append(snapshot)
        t = len(self.history) - 1
        out = np.tile(self.history[t][0], (self.n_agents, 1))
        caps = np.tile(self.history[t][1], (self.n_agents, 1))
        if self.arm == "live" or self.arm == "local":
            return out, caps
        if self.arm == "frozen":
            for i in range(self.n_agents):
                out[i] = self.history[0][0]
                caps[i] = self.history[0][1]
                out[i, i] = self.history[t][0][i]
                caps[i, i] = self.history[t][1][i]
            return out, caps
        if self.arm.startswith("fixed"):
            delay = int(self.arm[-1])
            past = self.history[max(t - delay, 0)]
            for i in range(self.n_agents):
                out[i] = past[0]
                caps[i] = past[1]
                out[i, i] = self.history[t][0][i]
                caps[i, i] = self.history[t][1][i]
            return out, caps
        if self.arm == "unfixed1-3":
            target = self.rng.randint(
                1, 4, size=(self.n_agents, self.n_agents))
            self.delays = np.where(
                target > self.delays,
                np.minimum(self.delays + 1, target), target)
            np.fill_diagonal(self.delays, 0)
            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    d = int(self.delays[i, j])
                    past = self.history[max(t - d, 0)]
                    out[i, j] = past[0][j]
                    caps[i, j] = past[1][j]
            return out, caps
        raise ValueError("unknown arm %s" % self.arm)


def route_actions(env, arm, perceived, perceived_caps, forwarding_penalty=0.75):
    """Workload-aware routing; only perceived neighbor load varies by arm."""
    rt = np.full((env.n_bs, env.K), 5, dtype=np.int32)
    assigned = np.zeros(env.n_bs, dtype=np.float64)
    destinations = []
    # A small forwarding penalty represents the extra transit step.  Urgent
    # tasks remain local regardless of load.
    for source in range(env.n_bs):
        others = env.neighbor_indices(source)
        for k in range(min(env.K, len(env.unrouted[source]))):
            tid = env.unrouted[source][k]
            task = env.tasks[tid]
            if arm == "local" or task["rem"] <= 4:
                rt[source, k] = 0
                assigned[source] += task["data"] / (
                    max(env.effective_caps()[source], 1) *
                    NOMINAL_BYTES_PER_CHANNEL)
                destinations.append(source)
                continue

            scores = perceived[source].copy() + assigned
            for dest in range(env.n_bs):
                if dest != source:
                    scores[dest] += forwarding_penalty
            dest = int(np.argmin(scores))
            if dest == source:
                rt[source, k] = 0
            else:
                rt[source, k] = others.index(dest) + 1
            # Use the advertised effective service rate only through the
            # perceived load; this increment merely prevents same-step pileup.
            assigned[dest] += task["data"] / (
                max(perceived_caps[source, dest], 1) *
                NOMINAL_BYTES_PER_CHANNEL)
            destinations.append(dest)
    return rt, destinations


def run_episode(arm, seed, cfg):
    env = DynamicOffloadEnv(
        n_channels=3, lam0=cfg["lam0"], lam_hot=cfg["lam_hot"],
        hotspot_dwell=cfg["dwell"], n_hotspots=cfg["n_hotspots"],
        lam_near=cfg.get("lam_near", 0.0), ddl_min=cfg["ddl_min"],
        ddl_max=cfg["ddl_max"], data_min=cfg.get("data_min", 1e5),
        data_max=cfg.get("data_max", 3e5), capacity_markov=True,
        capacity_stay=cfg["capacity_stay"],
        hotspot_capacity_penalty=cfg.get("hotspot_capacity_penalty", 0),
        seed=seed)
    view = InformationView(arm, env.n_bs,
                           np.random.RandomState(seed + 7919))
    total_reward = 0.0
    success = fail = 0
    routes = forwards = 0
    dest_counts = np.zeros(env.n_bs, dtype=np.int32)
    while not env.done:
        perceived, perceived_caps = view.observe(
            env.effective_loads(), env.effective_caps())
        rt, destinations = route_actions(
            env, arm, perceived, perceived_caps,
            forwarding_penalty=cfg.get("forwarding_penalty", 0.75))
        ch, pw = schedule_actions(env)
        r, info = env.step(rt, ch, pw)
        total_reward += r
        success += info["success"]
        fail += info["fail"]
        for source in range(env.n_bs):
            for k in range(min(env.K, len(destinations))):
                pass
        n_now = len(destinations)
        routes += n_now
        forwards += sum(1 for source in range(env.n_bs)
                        for k in range(env.K)
                        if 1 <= int(rt[source, k]) <= 4)
        for dest in destinations:
            dest_counts[dest] += 1
    arrived = int(info["arrived"])
    hhi = float(np.sum((dest_counts / float(max(dest_counts.sum(), 1))) ** 2))
    return {
        "reward": total_reward,
        "success": success,
        "fail": fail,
        "arrived": arrived,
        "success_rate": success / float(max(arrived, 1)),
        "route_rate": forwards / float(max(routes, 1)),
        "dest_hhi": hhi,
    }


def bootstrap_ci(diff, seed=20260927, n_boot=10000):
    diff = np.asarray(diff, dtype=np.float64)
    if len(diff) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    samples = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return tuple(np.percentile(samples, [2.5, 97.5]))


def evaluate_config(cfg, episodes, base_seed, arms=ARMS):
    rows = []
    by_arm = {a: [] for a in arms}
    for ep in range(episodes):
        seed = base_seed + ep * 100003
        for arm in arms:
            result = run_episode(arm, seed, cfg)
            result.update({"episode": ep, "seed": seed, "arm": arm})
            rows.append(result)
            by_arm[arm].append(result)
    return rows, by_arm


def summarize(cfg, by_arm):
    summary = {"config": dict(cfg), "arms": {}}
    for arm, values in by_arm.items():
        summary["arms"][arm] = {
            key: float(np.mean([v[key] for v in values]))
            for key in ("reward", "success", "fail", "arrived",
                        "success_rate", "route_rate", "dest_hhi")
        }

    live = np.array([x["success_rate"] for x in by_arm["live"]])
    local = np.array([x["success_rate"] for x in by_arm["local"]])
    frozen = np.array([x["success_rate"] for x in by_arm["frozen"]])
    d2 = np.array([x["success_rate"] for x in by_arm["fixed2"]])
    live_frozen = 100.0 * (live - frozen)
    summary["gates"] = {
        "A1_route_gap_pp": float(100.0 * np.mean(live - local)),
        "A2_voi_pp": float(np.mean(live_frozen)),
        "A2_ci95_pp": [float(x) for x in bootstrap_ci(live_frozen)],
        "A3_d2_relative_drop": float(
            (np.mean(live) - np.mean(d2)) / max(np.mean(live), 1e-9)),
        "A4_dose_monotone": bool(
            summary["arms"]["live"]["success_rate"] + 1e-12 >=
            summary["arms"]["fixed1"]["success_rate"] >=
            summary["arms"]["fixed2"]["success_rate"] >=
            summary["arms"]["fixed3"]["success_rate"] - 1e-12),
        "A5_live_success": float(np.mean(live)),
    }
    g = summary["gates"]
    g["A1_pass"] = g["A1_route_gap_pp"] >= 15.0
    g["A2_pass"] = g["A2_voi_pp"] >= 5.0 and g["A2_ci95_pp"][0] > 0.0
    g["A3_pass"] = g["A3_d2_relative_drop"] >= 0.10
    g["A4_pass"] = g["A4_dose_monotone"]
    g["A5_pass"] = 0.55 <= g["A5_live_success"] <= 0.85
    g["all_pass"] = all(g[k] for k in
                         ("A1_pass", "A2_pass", "A3_pass", "A4_pass", "A5_pass"))
    return summary


def print_summary(summary):
    cfg = summary["config"]
    print("config", json.dumps(cfg, sort_keys=True))
    for arm in ARMS:
        if arm not in summary["arms"]:
            continue
        x = summary["arms"][arm]
        print("  %-12s success=%6.2f%% reward=%7.2f route=%5.1f%% HHI=%.3f" %
              (arm, 100 * x["success_rate"], x["reward"],
               100 * x["route_rate"], x["dest_hhi"]))
    g = summary["gates"]
    print("  A1 route gap: %+.2f pp [%s]" %
          (g["A1_route_gap_pp"], "PASS" if g["A1_pass"] else "FAIL"))
    print("  A2 VoI:       %+.2f pp CI[%.2f, %.2f] [%s]" %
          (g["A2_voi_pp"], g["A2_ci95_pp"][0], g["A2_ci95_pp"][1],
           "PASS" if g["A2_pass"] else "FAIL"))
    print("  A3 d2 drop:   %+.1f%% [%s]" %
          (100 * g["A3_d2_relative_drop"], "PASS" if g["A3_pass"] else "FAIL"))
    print("  A4 monotone:  %s [%s]" %
          (g["A4_dose_monotone"], "PASS" if g["A4_pass"] else "FAIL"))
    print("  A5 live:      %.1f%% [%s]" %
          (100 * g["A5_live_success"], "PASS" if g["A5_pass"] else "FAIL"))
    print("  ALL: %s" % ("PASS" if g["all_pass"] else "FAIL"))


def write_results(out_dir, name, rows, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_dir / (name + "_episodes.csv"), "w", newline="") as fh:
            fields = list(rows[0].keys())
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with open(out_dir / (name + "_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1103)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--n-hotspots", type=int, default=2)
    ap.add_argument("--lam0", type=float, default=0.20)
    ap.add_argument("--lam-hot", type=float, default=0.90)
    ap.add_argument("--dwell", type=int, default=6)
    ap.add_argument("--capacity-stay", type=float, default=0.70)
    ap.add_argument("--ddl-min", type=int, default=5)
    ap.add_argument("--ddl-max", type=int, default=11)
    ap.add_argument("--forwarding-penalty", type=float, default=0.75)
    ap.add_argument("--data-scale", type=float, default=1.0)
    ap.add_argument("--hotspot-capacity-penalty", type=int, default=0)
    ap.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "results" / "exp11_phase_a"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not args.scan:
        cfg = {
            "n_hotspots": args.n_hotspots, "lam0": args.lam0,
            "lam_hot": args.lam_hot, "lam_near": 0.0,
            "dwell": args.dwell, "capacity_stay": args.capacity_stay,
            "ddl_min": args.ddl_min, "ddl_max": args.ddl_max,
            "forwarding_penalty": args.forwarding_penalty,
            "data_min": 1e5 * args.data_scale,
            "data_max": 3e5 * args.data_scale,
            "hotspot_capacity_penalty": args.hotspot_capacity_penalty,
        }
        rows, by_arm = evaluate_config(cfg, args.episodes, args.seed)
        summary = summarize(cfg, by_arm)
        print_summary(summary)
        write_results(out_dir, "validation", rows, summary)
        return

    # Cheap screening uses the three decisive arms plus d=2.  The best
    # candidates should subsequently be rerun without --scan for dose curves.
    configs = []
    for nh in (2, 3):
        for lam0 in (0.10, 0.20):
            for lh in (0.60, 0.90, 1.20):
                for dwell in (4, 6):
                    configs.append({
                        "n_hotspots": nh, "lam0": lam0, "lam_hot": lh,
                        "lam_near": 0.0, "dwell": dwell,
                        "capacity_stay": args.capacity_stay,
                        "ddl_min": args.ddl_min, "ddl_max": args.ddl_max,
                        "forwarding_penalty": args.forwarding_penalty,
                        "data_min": 1e5 * args.data_scale,
                        "data_max": 3e5 * args.data_scale,
                        "hotspot_capacity_penalty": args.hotspot_capacity_penalty,
                    })
    ranking = []
    scan_arms = ("local", "live", "frozen", "fixed2")
    for idx, cfg in enumerate(configs):
        rows, by_arm = evaluate_config(
            cfg, args.episodes, args.seed + idx * 10000019, arms=scan_arms)
        live = np.mean([x["success_rate"] for x in by_arm["live"]])
        local = np.mean([x["success_rate"] for x in by_arm["local"]])
        frozen = np.mean([x["success_rate"] for x in by_arm["frozen"]])
        d2 = np.mean([x["success_rate"] for x in by_arm["fixed2"]])
        rec = dict(cfg)
        rec.update({
            "route_gap_pp": 100 * (live - local),
            "voi_pp": 100 * (live - frozen),
            "d2_drop": (live - d2) / max(live, 1e-9),
            "live_success": live,
        })
        # Rank for information value and delay effect while penalizing
        # saturated/collapsed scenarios.
        band_penalty = 100 * max(0.55 - live, 0, live - 0.85)
        rec["score"] = rec["voi_pp"] + 20 * rec["d2_drop"] - band_penalty
        ranking.append(rec)
        print("[%02d/%02d] live %.1f%% gap %+.1fpp VoI %+.1fpp d2 %+.1f%%" %
              (idx + 1, len(configs), 100 * live, rec["route_gap_pp"],
               rec["voi_pp"], 100 * rec["d2_drop"]))
    ranking.sort(key=lambda x: x["score"], reverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scan_ranking.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ranking[0].keys()))
        writer.writeheader()
        writer.writerows(ranking)
    print("\nTop candidates:")
    for x in ranking[:8]:
        print(json.dumps(x, sort_keys=True))


if __name__ == "__main__":
    main()
