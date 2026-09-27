"""MAPPO training for the dynamic task-arrival offloading scenario.

Env vars (beyond those of train_mappo_phase0.py):
  ROUTE_MASK      "" | "local"   (local = forced-local calibration arm)
  DYN_LAM0=0.15  DYN_LAM_HOT=0.6  DYN_DWELL=8
  DYN_T=30  DYN_ARRIVAL_END=25  DYN_CH=4  DYN_HOLD=0.02
  DELAY_MODE acts on the neighbor block (12 dims) only; groups built for
  the 47-dim layout. UPDATE_EVERY defaults to 25.

Outputs (OUT_DIR): phase0_{TAG}.csv (episode log),
phase0_{TAG}_eval.csv (greedy eval), phase0_{TAG}_diag.csv (canaries).
"""

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dynamic_offload_env import DynamicOffloadEnv
from mappo_agent import MAPPOAgent
from mappo_buffer import RolloutBuffer
from delay_filter import DelayFilter

OBS_DIM = 47
MAX_QUEUE = 10
N_POWER = 5
K_ROUTE = 2
N_ROUTE_OPT = 6
NEIGH_OFF = 28
NEIGH_GROUPS = [np.arange(NEIGH_OFF + 3 * j, NEIGH_OFF + 3 * j + 3)
                for j in range(4)]


def env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def env_int(name, d):
    return int(os.environ.get(name, str(d)))


def env_float(name, d):
    return float(os.environ.get(name, str(d)))


class RouteBuffer(RolloutBuffer):

    def __init__(self, n_steps, n_agents, obs_dim, max_channels, max_tasks,
                 n_power):
        RolloutBuffer.__init__(self, n_steps, n_agents, obs_dim, max_channels,
                               max_tasks, n_power)
        S, G, K, O = self.n_steps, self.n_agents, K_ROUTE, N_ROUTE_OPT
        self.route_choices = np.zeros((S, G, K), dtype=np.int32)
        self.route_masks = np.zeros((S, G, K, O), dtype=np.float32)

    def add(self, obs, global_obs, channel_choices, power_choices,
            route_choices, channel_mask, power_mask, route_mask,
            log_probs, reward, values, done):
        p = self.pos
        RolloutBuffer.add(self, obs, global_obs, channel_choices,
                          power_choices, channel_mask, power_mask,
                          log_probs, reward, values, done)
        self.route_choices[p] = route_choices
        self.route_masks[p] = route_mask


def gae_segment(buf, start, end, n_agents, gamma, lam):
    n = end - start
    adv = np.zeros((n, n_agents), dtype=np.float32)
    ret = np.zeros(n, dtype=np.float32)
    for a in range(n_agents):
        g = 0.0
        nxt = 0.0
        ra = np.zeros(n, dtype=np.float32)
        rr = np.zeros(n, dtype=np.float32)
        for t in reversed(range(n)):
            mask = 1.0 - buf.dones[start + t]
            delta = (buf.rewards[start + t] + gamma * nxt * mask
                     - buf.values[start + t, a])
            g = delta + gamma * lam * mask * g
            ra[t] = g
            rr[t] = g + buf.values[start + t, a]
            nxt = buf.values[start + t, a]
        adv[:, a] = ra
        if a == 0:
            ret = rr
    return adv, ret


def run_episode(env, agents, sess, delay_filter, mode, greedy=False,
                route_mask_mode="", collect_diag=None):
    obs_true_all = []
    total_r = 0.0
    succ = fail = arrived = 0
    delay_filter.reset() if delay_filter is not None else None
    n_agents = env.n_bs
    while not env.done:
        obs_true = np.asarray([env.build_obs(i) for i in range(n_agents)],
                              dtype=np.float32)
        if delay_filter is not None and mode != "none":
            delay_filter.step_delays()
            delay_filter.append(obs_true)
            obs_actor = delay_filter.observe()
        else:
            obs_actor = obs_true
        route_m, ch_m, pw_m = env.get_masks(route_mask_mode)
        rt = np.zeros((n_agents, K_ROUTE), dtype=np.int32)
        ch = np.zeros((n_agents, env.n_ch), dtype=np.int32)
        pw = np.zeros((n_agents, MAX_QUEUE), dtype=np.int32)
        if greedy:
            for i, ag in enumerate(agents):
                c_, p_, r_ = ag.greedy_all(sess, obs_actor[i], ch_m[i],
                                           pw_m[i], route_m[i])
                ch[i] = c_[:env.n_ch]
                pw[i] = p_
                rt[i] = r_
        else:
            for i, ag in enumerate(agents):
                c_, p_, r_, _ = ag.act_route(sess, obs_actor[i], ch_m[i],
                                             pw_m[i], route_m[i])
                ch[i] = c_[:env.n_ch]
                pw[i] = p_
                rt[i] = r_
        if collect_diag is not None:
            collect_diag.append((obs_actor.copy(), rt.copy()))
        r, info = env.step(rt, ch, pw)
        total_r += r
        succ += info["success"]
        fail += info["fail"]
        arrived = info["arrived"]
        obs_true_all.append((obs_actor, ch, pw, rt, ch_m, pw_m, route_m, r))
    return total_r, succ, fail, arrived, obs_true_all


def main():
    tf.set_random_seed(env_int("SEED", 1))
    tag = os.environ.get("TAG", "dyn")
    mode = os.environ.get("DELAY_MODE", "none")
    d_value = env_int("DELAY_VALUE", 0)
    d_min = env_int("DELAY_MIN", 0)
    d_max = env_int("DELAY_MAX", 0)
    route_mask_mode = os.environ.get("ROUTE_MASK", "")
    n_episode = env_int("N_EPISODES", 3000)
    base_seed = env_int("SEED", 1)
    eval_every = env_int("EVAL_EVERY", 200)
    eval_eps = env_int("EVAL_EPS", 20)
    update_every = max(1, env_int("UPDATE_EVERY", 25))
    k_epochs = env_int("K_EPOCHS", 4)
    minibatch = env_int("MINIBATCH_SIZE", 64)
    gamma = env_float("GAMMA", 0.99)
    lam = env_float("GAE_LAMBDA", 0.95)

    env_kwargs = dict(
        n_bs=5,
        n_channels=env_int("DYN_CH", 3),
        lam0=env_float("DYN_LAM0", 0.15),
        lam_hot=env_float("DYN_LAM_HOT", 1.5),
        hotspot_dwell=env_int("DYN_DWELL", 8),
        n_hotspots=env_int("DYN_N_HOTSPOTS", 1),
        lam_near=env_float("DYN_LAM_NEAR", 0.0),
        T=env_int("DYN_T", 30),
        arrival_end=env_int("DYN_ARRIVAL_END", 25),
        max_queue=MAX_QUEUE,
        holding_cost=env_float("DYN_HOLD", 0.02),
        ddl_min=env_int("DYN_DDL_MIN", 5),
        ddl_max=env_int("DYN_DDL_MAX", 11),
        data_min=env_float("DYN_DATA_MIN", 1e5),
        data_max=env_float("DYN_DATA_MAX", 3e5),
        capacity_markov=env_flag("DYN_CAPACITY_MARKOV", "0"),
        capacity_stay=env_float("DYN_CAPACITY_STAY", 0.70),
        hotspot_capacity_penalty=env_int("DYN_HOTSPOT_CAP_PENALTY", 0),
        randomize_neighbor_order=env_flag("DYN_RANDOM_NEIGHBOR_ORDER", "0"),
    )

    n_agents = 5
    agents = []
    if env_flag("SHARE_ACTOR", "0"):
        shared_agent = MAPPOAgent(
            "shared_agent", 0, n_agents=n_agents, obs_dim=OBS_DIM,
            max_tasks=MAX_QUEUE, max_channels=env_kwargs["n_channels"],
            n_power=N_POWER, clip_eps=env_float("CLIP_EPS", 0.2),
            entropy_coef=env_float("ENTROPY_COEF", 0.01),
            max_grad_norm=env_float("MAX_GRAD_NORM", 0.5),
            is_first_agent=True, n_route_slots=K_ROUTE,
            n_route_options=N_ROUTE_OPT,
            structured_route=env_flag("STRUCTURED_ROUTE", "0"))
        agents = [shared_agent for _ in range(n_agents)]
    else:
        for i in range(n_agents):
            agents.append(MAPPOAgent(
                "agent%d" % (i + 1), i, n_agents=n_agents, obs_dim=OBS_DIM,
                max_tasks=MAX_QUEUE, max_channels=env_kwargs["n_channels"],
                n_power=N_POWER, clip_eps=env_float("CLIP_EPS", 0.2),
                entropy_coef=env_float("ENTROPY_COEF", 0.01),
                max_grad_norm=env_float("MAX_GRAD_NORM", 0.5),
                is_first_agent=(i == 0), n_route_slots=K_ROUTE,
                n_route_options=N_ROUTE_OPT,
                structured_route=env_flag("STRUCTURED_ROUTE", "0")))
    buffer = RouteBuffer(env_int("N_STEPS", 2000), n_agents, OBS_DIM,
                         env_kwargs["n_channels"], MAX_QUEUE, N_POWER)

    cfg = tf.ConfigProto(intra_op_parallelism_threads=env_int("TF_INTRA", 2),
                         inter_op_parallelism_threads=env_int("TF_INTER", 1))
    sess = tf.Session(config=cfg)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1) if env_flag("SAVE_MODEL", "0") else None

    rng = np.random.RandomState(base_seed)
    delay_filter = DelayFilter(n_agents, OBS_DIM, mode=mode,
                               d_value=d_value, d_min=d_min, d_max=d_max,
                               rng=rng, groups=NEIGH_GROUPS)

    out_dir = Path(os.environ.get(
        "OUT_DIR", str(SCRIPT_DIR / "results" / "exp10_calib")))
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / ("phase0_%s.csv" % tag), "w", newline="")
    w = csv.writer(log)
    w.writerow(["# dyn mode=%s d=%d [%d,%d] route_mask=%s lam_hot=%.2f ch=%d "
                "seed=%d eps=%d ue=%d" % (mode, d_value, d_min, d_max,
                                          route_mask_mode or "free",
                                          env_kwargs["lam_hot"],
                                          env_kwargs["n_channels"], base_seed,
                                          n_episode, update_every)])
    w.writerow(["episode", "reward", "success", "fail", "arrived",
                "route_rate", "cond_route_rate", "top_share", "load_std"])
    elog = open(out_dir / ("phase0_%s_eval.csv" % tag), "w", newline="")
    ew = csv.writer(elog)
    ew.writerow(["episode", "eval_reward", "eval_success", "eval_fail",
                 "eval_route_rate"])
    dlog = open(out_dir / ("phase0_%s_diag.csv" % tag), "w", newline="")
    dw = csv.writer(dlog)
    dw.writerow(["episode", "fail_fwd", "fail_local"])

    def make_env(seed):
        return DynamicOffloadEnv(seed=seed, **env_kwargs)

    accum_adv = np.zeros_like(buffer.values)
    accum_ret = np.zeros(buffer.rewards.shape, dtype=np.float32)
    ep_in_buf = 0

    def ppo_update(total):
        adv = accum_adv[:total].copy()
        ret = accum_ret[:total]
        for a in range(n_agents):
            x = adv[:, a]
            if x.std() > 1e-8:
                adv[:, a] = (x - x.mean()) / (x.std() + 1e-8)
        ret_n = (ret - ret.mean()) / (ret.std() + 1e-8)
        for _ in range(k_epochs):
            for idx in buffer.get_batches(minibatch):
                for a, ag in enumerate(agents):
                    if a == 0:
                        ag.evaluate_actions_route(
                            sess, buffer.obs[idx, a], buffer.global_obs[idx],
                            buffer.channel_choices[idx, a],
                            buffer.power_choices[idx, a],
                            buffer.route_choices[idx, a],
                            buffer.channel_masks[idx, a],
                            buffer.power_masks[idx, a],
                            buffer.route_masks[idx, a],
                            buffer.log_probs[idx, a], buffer.values[idx, a],
                            adv[idx, a], ret_n[idx])
                    else:
                        ag.train_actor_only_route(
                            sess, buffer.obs[idx, a], buffer.global_obs[idx],
                            buffer.channel_choices[idx, a],
                            buffer.power_choices[idx, a],
                            buffer.route_choices[idx, a],
                            buffer.channel_masks[idx, a],
                            buffer.power_masks[idx, a],
                            buffer.route_masks[idx, a],
                            buffer.log_probs[idx, a], adv[idx, a])

    t0 = time.time()
    for ep in range(1, n_episode + 1):
        env = make_env(base_seed * 1000003 + ep)
        delay_filter.reset()
        ep_start = buffer.pos
        succ = fail = 0
        total_r = 0.0
        while not env.done:
            obs_true = np.asarray([env.build_obs(i) for i in range(n_agents)],
                                  dtype=np.float32)
            if mode != "none":
                delay_filter.step_delays()
                delay_filter.append(obs_true)
                obs_actor = delay_filter.observe()
            else:
                obs_actor = obs_true
            route_m, ch_m, pw_m = env.get_masks(route_mask_mode)
            rt = np.zeros((n_agents, K_ROUTE), dtype=np.int32)
            ch = np.zeros((n_agents, env_kwargs["n_channels"]), dtype=np.int32)
            pw = np.zeros((n_agents, MAX_QUEUE), dtype=np.int32)
            lps = np.zeros(n_agents, dtype=np.float32)
            vals = np.zeros(n_agents, dtype=np.float32)
            # CTDE: the critic receives the delay-free global state.  Only
            # decentralized actors are exposed to delayed neighbor reports.
            glob = obs_true.flatten()
            for i, ag in enumerate(agents):
                c_, p_, r_, lp = ag.act_route(sess, obs_actor[i], ch_m[i],
                                              pw_m[i], route_m[i])
                ch[i] = c_
                pw[i] = p_
                rt[i] = r_
                lps[i] = lp
                vals[i] = ag.get_value(sess, glob)
            r, info = env.step(rt, ch, pw)
            total_r += r
            succ += info["success"]
            fail += info["fail"]
            done = 1.0 if env.done else 0.0
            buffer.add(obs_actor, glob, ch, pw, rt, ch_m, pw_m, route_m,
                       lps, r, vals, done)
        adv_ep, ret_ep = gae_segment(buffer, ep_start, buffer.pos, n_agents,
                                     gamma, lam)
        accum_adv[ep_start:buffer.pos] = adv_ep
        accum_ret[ep_start:buffer.pos] = ret_ep
        ep_in_buf += 1
        if ep_in_buf >= update_every or buffer.pos + 64 > buffer.capacity:
            ppo_update(buffer.pos)
            buffer.clear()
            ep_in_buf = 0

        # canaries from env.diag
        evs = env.diag["route_events"]
        n_dec = len(evs)
        n_fwd = sum(1 for e in evs if 1 <= e["opt"] <= 4)
        n_dec_p = sum(1 for e in evs if e["own_q"] >= 3)
        n_fwd_p = sum(1 for e in evs if e["own_q"] >= 3 and 1 <= e["opt"] <= 4)
        rr = n_fwd / n_dec if n_dec else 0.0
        cr = n_fwd_p / n_dec_p if n_dec_p else 0.0
        dest_counts = [0] * 5
        for e in evs:
            if 1 <= e["opt"] <= 4:
                others = [x for x in range(5) if x != e["agent"]]
                dest_counts[others[e["opt"] - 1]] += 1
        top = max(dest_counts) / sum(dest_counts) if sum(dest_counts) else 0.0
        loads = np.asarray(env.diag["loads"], dtype=np.float64)
        lstd = float(loads.std(axis=1).mean()) if len(loads) else 0.0
        w.writerow([ep, round(total_r, 3), succ, fail, info["arrived"],
                    round(rr, 3), round(cr, 3), round(top, 3), round(lstd, 2)])
        dw.writerow([ep, env.diag["fail_fwd"], env.diag["fail_local"]])
        if ep % 50 == 0 or ep == 1:
            print("ep %d/%d | r=%.1f s=%d f=%d | route=%.2f cond=%.2f | "
                  "%.2fs/ep" % (ep, n_episode, total_r, succ, fail, rr, cr,
                                (time.time() - t0) / ep), flush=True)
        if eval_every > 0 and ep % eval_every == 0:
            er = es = ef = evt = 0.0
            for k in range(eval_eps):
                eenv = make_env(base_seed * 1000003 + 500000 + k)
                delay_filter.reset()
                rr_, s_, f_, a_, _ = run_episode(
                    eenv, agents, sess, delay_filter, mode, greedy=True,
                    route_mask_mode=route_mask_mode)
                er += rr_
                es += s_
                ef += f_
                evt += sum(1 for e in eenv.diag["route_events"]
                           if 1 <= e["opt"] <= 4)
            ew.writerow([ep, round(er / eval_eps, 3), round(es / eval_eps, 2),
                         round(ef / eval_eps, 2), round(evt / eval_eps, 2)])
            elog.flush()
        log.flush()
        dlog.flush()
    log.close()
    elog.close()
    dlog.close()
    if saver is not None:
        model_dir = out_dir / ("model_%s" % tag)
        model_dir.mkdir(parents=True, exist_ok=True)
        saver.save(sess, str(model_dir / "model"))
    sess.close()
    print("DONE tag=%s" % tag, flush=True)


if __name__ == "__main__":
    main()
