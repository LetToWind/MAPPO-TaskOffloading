"""Phase 0 experiment: delay sensitivity of Base MAPPO with the 34-dim obs.

Runs MAPPO under a configurable DelayFilter setting and logs reward /
success / fail per episode. The actor acts on the DELAYED observation; the
critic input follows USE_DR (0: concatenation of delayed views = Base,
1: delay-free true state = Base+DR).

Environment variables (all optional):
  DELAY_MODE     none | fixed | partial | unfixed   (default none)
  DELAY_VALUE    fixed/partial delay steps          (default 0)
  DELAY_MIN      unfixed lower bound                (default 0)
  DELAY_MAX      unfixed upper bound                (default 0)
  DELAY_PARTIAL_P partial mode probability          (default 0.5)
  USE_DR         1 = delay-reconciled critic        (default 0)
  TAG            output file suffix                 (default auto)
  N_EPISODES                                        (default 6000)
  SEED           dynamic topology base seed         (default 1)
  UE_FIXED_EPISODES                                 (default 0)
  TF_INTRA / TF_INTER  thread limits                (default 2 / 1)
"""

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from config.config import settings
from Environment_marl_discrete import DiscreteEnviron
from fading_env import FadingEnviron
from mappo_agent import MAPPOAgent
from mappo_buffer import RolloutBuffer
from delay_filter import DelayFilter
import obs_utils

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from init_input.dynamic_topology import build_dynamic_env_config, generate_fixed_bs_config  # noqa: E402


def greedy_act(sess, agent, obs, channel_mask, power_mask):
    """Greedy (argmax) action selection for evaluation episodes."""
    obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    ch_probs, pw_probs = sess.run(
        [agent.channel_probs, agent.power_probs],
        {agent.local_obs: obs,
         agent.channel_mask: np.asarray(channel_mask, dtype=np.float32).reshape(1, agent.max_channels, agent.max_tasks + 1),
         agent.power_mask: np.asarray(power_mask, dtype=np.float32).reshape(1, agent.max_tasks, agent.n_power)})
    ch = np.argmax(ch_probs[0], axis=-1).astype(np.int32)
    pw = np.argmax(pw_probs[0], axis=-1).astype(np.int32)
    return ch, pw

BS_CAPACITY = 6
N_POWER = 5
N_CHANNELS = obs_utils.N_INTERF_DIMS


def env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _cfg(name, settings_attr, cast=str):
    env = os.environ.get(name, "")
    return cast(env) if env else cast(getattr(settings, settings_attr))


def dynamic_topology_kwargs(n_bs, seed=None):
    kwargs = {
        "n_bs": n_bs,
        "n_tasks": env_int("DYNAMIC_N_TASKS", settings.DYNAMIC_N_TASKS),
        "n_channels": BS_CAPACITY, "channel_per_bs": BS_CAPACITY,
        "max_tasks_per_bs": BS_CAPACITY,
        "area_size": (env_float("DYNAMIC_AREA_WIDTH", settings.DYNAMIC_AREA_WIDTH),
                      env_float("DYNAMIC_AREA_HEIGHT", settings.DYNAMIC_AREA_HEIGHT)),
    }
    max_ue = env_float("MAX_UE_DISTANCE", settings.MAX_UE_DISTANCE)
    if max_ue > 0:
        kwargs["max_ue_distance"] = max_ue
    if seed is not None:
        kwargs["seed"] = seed
    return kwargs


def should_continue_episode(env):
    return (not (np.any(env.task_deadline) == 0)) and (not (np.any(env.task_data_size) == 0))


def auto_tag():
    mode = os.environ.get("DELAY_MODE", "none")
    fade = os.environ.get("FADING", "0")
    prefix = "fa" if fade in ("1", "true", "yes", "on") else ""
    if mode == "none":
        return (prefix + "none") if prefix else "none"
    if mode == "frozen":
        return prefix + "frozen" if prefix else "frozen"
    if mode in ("fixed", "partial"):
        p = os.environ.get("DELAY_PARTIAL_P", "0.5")
        return "%s%s_d%s%s" % (prefix, mode, os.environ.get("DELAY_VALUE", "0"),
                               "" if mode == "fixed" else "_p" + p)
    return "%sunfixed_%s-%s" % (prefix, os.environ.get("DELAY_MIN", "0"),
                                os.environ.get("DELAY_MAX", "0"))


def build_sim_dict(topology, n_agents, base_seed, episode_count,
                   bs_positions, bs_channels, fixed_sim_dict, kwargs_fn):
    """Topology schedule: fixed (variant 25) or dynamic (random per episode)."""
    if topology == "fixed":
        import copy
        return copy.deepcopy(fixed_sim_dict)
    return build_dynamic_env_config(
        **kwargs_fn(seed=base_seed + episode_count),
        bs_positions=bs_positions, usable_channel_of_all_nodes=bs_channels)


def run_eval_episode(env, n_agents, agents, sess, delay_filter, mode):
    """One greedy evaluation episode under the arm's delay setting."""
    sum_reward, success, fail = 0.0, 0, 0
    delay_filter.reset()
    steps = 0
    while should_continue_episode(env) and steps < 60:
        states_true = np.asarray(
            [obs_utils.build_obs(env, i, n_agents) for i in range(n_agents)],
            dtype=np.float32)
        delay_filter.step_delays()
        delay_filter.append(states_true)
        states_actor = delay_filter.observe() if mode != "none" else states_true
        channel_masks, power_masks = env.get_action_masks()
        ch = np.zeros((n_agents, agents[0].max_channels), dtype=np.int32)
        pw = np.zeros((n_agents, agents[0].max_channels), dtype=np.int32)
        for i, agent in enumerate(agents):
            ch[i], pw[i] = greedy_act(sess, agent, states_actor[i],
                                      channel_masks[i], power_masks[i])
        r, s, f = env.act_for_training_discrete(ch, pw)
        sum_reward += r
        success += s
        fail += f
        steps += 1
    return sum_reward, success, fail


def main():
    tf.set_random_seed(env_int("SEED", 1))
    mode = os.environ.get("DELAY_MODE", "none")
    d_value = env_int("DELAY_VALUE", 0)
    d_min = env_int("DELAY_MIN", 0)
    d_max = env_int("DELAY_MAX", 0)
    partial_p = env_float("DELAY_PARTIAL_P", 0.5)
    use_dr = env_flag("USE_DR")
    tag = os.environ.get("TAG", auto_tag())
    n_episode = env_int("N_EPISODES", 6000)
    n_steps = env_int("N_STEPS", settings.N_STEPS)
    base_seed = env_int("SEED", 1)
    ue_fixed_episodes = env_int("UE_FIXED_EPISODES", 0)

    n_agents = env_int("DYNAMIC_N_BS", settings.DYNAMIC_N_BS)
    obs_d = obs_utils.obs_dim(n_agents)
    topology = os.environ.get("TOPOLOGY", "fixed").strip().lower()
    eval_every = env_int("EVAL_EVERY", 100)
    eval_eps = env_int("EVAL_EPS", 20)
    use_fading = env_flag("FADING")
    fading_rho = env_float("FADING_RHO", 0.7)
    fading_sigma = env_float("FADING_SIGMA_DB", 6.0)
    update_every = max(1, env_int("UPDATE_EVERY", 1))

    def make_env(sim_dict, seed=None):
        if use_fading:
            return FadingEnviron(sim_dict, rho=fading_rho,
                                 sigma_db=fading_sigma, seed=seed)
        return DiscreteEnviron(sim_dict)

    bs_positions, bs_channels = generate_fixed_bs_config(
        n_bs=n_agents, n_channels=BS_CAPACITY, channel_per_bs=BS_CAPACITY,
        area_size=(env_float("DYNAMIC_AREA_WIDTH", settings.DYNAMIC_AREA_WIDTH),
                   env_float("DYNAMIC_AREA_HEIGHT", settings.DYNAMIC_AREA_HEIGHT)),
        seed=base_seed)
    _ch = env_int("CHANNEL_NUM", 0)
    if 0 < _ch <= BS_CAPACITY:
        bs_channels = [list(range(_ch)) for _ in bs_channels]

    fixed_sim_dict = None
    if topology == "fixed":
        import random as _random
        from init_input.experiment_setup import build_fixed_env_config
        # Deterministic scenario across ALL arms/seeds: seed the module-level
        # RNG and retry until a valid 25-task scenario is built (the builder
        # randomly samples edge vehicles and intermittently yields < 25 tasks).
        for attempt in range(500):
            _random.seed(20240921 + attempt)
            try:
                cand = build_fixed_env_config(25)
            except (IndexError, ValueError):
                continue
            if len(cand["task_data_size"]) >= 25:
                fixed_sim_dict = cand
                break
        if fixed_sim_dict is None:
            raise RuntimeError("could not build a valid fixed variant-25 scenario")
        # congestion control: restrict the number of usable channels per node
        ch_num = env_int("CHANNEL_NUM", 0)
        if 0 < ch_num <= len(fixed_sim_dict["usable_channel_of_all_nodes"][0]):
            for k in range(len(fixed_sim_dict["usable_channel_of_all_nodes"])):
                fixed_sim_dict["usable_channel_of_all_nodes"][k] = list(range(ch_num))
        n_agents = len(fixed_sim_dict["task_on_base"])
        obs_d = obs_utils.obs_dim(n_agents)

    agents = [
        MAPPOAgent("agent%d" % (i + 1), i, n_agents=n_agents, obs_dim=obs_d,
                   max_tasks=BS_CAPACITY, max_channels=BS_CAPACITY, n_power=N_POWER,
                   clip_eps=env_float("CLIP_EPS", settings.CLIP_EPS),
                   entropy_coef=env_float("ENTROPY_COEF", settings.ENTROPY_COEF),
                   max_grad_norm=env_float("MAX_GRAD_NORM", settings.MAX_GRAD_NORM),
                   is_first_agent=(i == 0))
        for i in range(n_agents)
    ]
    buffer = RolloutBuffer(n_steps, n_agents, obs_d, BS_CAPACITY, BS_CAPACITY, N_POWER)

    tf_config = tf.ConfigProto(
        intra_op_parallelism_threads=env_int("TF_INTRA", 2),
        inter_op_parallelism_threads=env_int("TF_INTER", 1))
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())

    rng = np.random.RandomState(base_seed)
    delay_filter = DelayFilter(
        n_agents, obs_d, n_channels=N_CHANNELS, mode=mode,
        d_value=d_value, d_min=d_min, d_max=d_max, partial_p=partial_p,
        rng=rng)

    out_dir = Path(os.environ.get(
        "OUT_DIR",
        str(Path(__file__).resolve().parent / "results" / "default")))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / ("phase0_%s.csv" % tag)
    reward_log = out_csv.open("w", newline="")
    reward_writer = csv.writer(reward_log)
    reward_writer.writerow(["# mode=%s d_value=%d d=[%d,%d] p=%.2f dr=%d seed=%d "
                            "topology=%s ch=%d fade=%d(rho=%.2f,sig=%.1f) "
                            "n_agents=%d obs_dim=%d n_episodes=%d ue=%d"
                            % (mode, d_value, d_min, d_max, partial_p, int(use_dr),
                               base_seed, topology, env_int("CHANNEL_NUM", 0),
                               int(use_fading), fading_rho, fading_sigma,
                               n_agents, obs_d, n_episode, update_every)])
    reward_writer.writerow(["episode", "sum_reward", "success", "fail", "mean_delay"])

    eval_csv = out_dir / ("phase0_%s_eval.csv" % tag)
    eval_log = eval_csv.open("w", newline="")
    eval_writer = csv.writer(eval_log)
    eval_writer.writerow(["# greedy eval | topology=%s eval_eps=%d" % (topology, eval_eps)])
    eval_writer.writerow(["episode", "eval_reward", "eval_success", "eval_fail"])

    def make_eval_env(k):
        if topology == "fixed":
            import copy
            return make_env(copy.deepcopy(fixed_sim_dict),
                            seed=base_seed + 500000 + k)
        return make_env(build_dynamic_env_config(
            **dynamic_topology_kwargs(n_bs=n_agents, seed=base_seed + 100000 + k),
            bs_positions=bs_positions, usable_channel_of_all_nodes=bs_channels))

    gamma = env_float("GAMMA", settings.GAMMA)
    gae_lambda = env_float("GAE_LAMBDA", settings.GAE_LAMBDA)
    k_epochs = env_int("K_EPOCHS", settings.K_EPOCHS)
    minibatch_size = env_int("MINIBATCH_SIZE", settings.MINIBATCH_SIZE)

    def gae_segment(start, end):
        """GAE for the contiguous episode rows [start, end) of the buffer.
        Same math as RolloutBuffer._compute_gae_for_agent, unnormalized."""
        n = end - start
        adv = np.zeros((n, n_agents), dtype=np.float32)
        ret = np.zeros(n, dtype=np.float32)
        for agent_id in range(n_agents):
            g = 0.0
            nxt = 0.0
            r_adv = np.zeros(n, dtype=np.float32)
            r_ret = np.zeros(n, dtype=np.float32)
            for t in reversed(range(n)):
                mask = 1.0 - buffer.dones[start + t]
                delta = (buffer.rewards[start + t]
                         + gamma * nxt * mask
                         - buffer.values[start + t, agent_id])
                g = delta + gamma * gae_lambda * mask * g
                r_adv[t] = g
                r_ret[t] = g + buffer.values[start + t, agent_id]
                nxt = buffer.values[start + t, agent_id]
            adv[:, agent_id] = r_adv
            if agent_id == 0:
                ret = r_ret
        return adv, ret

    def ppo_update(total_rows):
        advantages = accum_adv[:total_rows]
        returns = accum_ret[:total_rows]
        for agent_id in range(n_agents):
            a = advantages[:, agent_id]
            if a.std() > 1e-8:
                advantages[:, agent_id] = (a - a.mean()) / (a.std() + 1e-8)
        returns_n = (returns - returns.mean()) / (returns.std() + 1e-8)
        for _ in range(k_epochs):
            for batch_indices in buffer.get_batches(minibatch_size):
                for agent_id, agent in enumerate(agents):
                    if agent_id == 0:
                        agent.evaluate_actions(
                            sess,
                            buffer.obs[batch_indices, agent_id],
                            buffer.global_obs[batch_indices],
                            buffer.channel_choices[batch_indices, agent_id],
                            buffer.power_choices[batch_indices, agent_id],
                            buffer.channel_masks[batch_indices, agent_id],
                            buffer.power_masks[batch_indices, agent_id],
                            buffer.log_probs[batch_indices, agent_id],
                            buffer.values[batch_indices, agent_id],
                            advantages[batch_indices, agent_id],
                            returns_n[batch_indices])
                    else:
                        agent.train_actor_only(
                            sess,
                            buffer.obs[batch_indices, agent_id],
                            buffer.global_obs[batch_indices],
                            buffer.channel_choices[batch_indices, agent_id],
                            buffer.power_choices[batch_indices, agent_id],
                            buffer.channel_masks[batch_indices, agent_id],
                            buffer.power_masks[batch_indices, agent_id],
                            buffer.log_probs[batch_indices, agent_id],
                            advantages[batch_indices, agent_id])

    episode_count = 0
    episodes_in_buffer = 0
    accum_adv = np.zeros_like(buffer.values)
    accum_ret = np.zeros(buffer.rewards.shape, dtype=np.float32)
    t_start = time.time()
    recent = []

    while episode_count < n_episode:
        sim_dict = build_sim_dict(topology, n_agents, base_seed, episode_count,
                                  bs_positions, bs_channels, fixed_sim_dict,
                                  lambda seed=None: dynamic_topology_kwargs(n_agents, seed))
        env = make_env(sim_dict)

        sum_reward = 0.0
        epsi_success = 0
        fail_sum = 0
        ep_delay_sum = 0.0
        ep_delay_n = 0
        delay_filter.reset()
        ep_start = buffer.pos

        while should_continue_episode(env):
            states_true = np.asarray(
                [obs_utils.build_obs(env, i, n_agents) for i in range(n_agents)],
                dtype=np.float32)
            delay_filter.step_delays()
            delay_filter.append(states_true)
            states_delayed = delay_filter.observe()
            states_actor = states_delayed if mode != "none" else states_true

            if use_dr:
                global_obs = states_true.flatten()
            else:
                global_obs = states_actor.flatten()

            channel_masks, power_masks = env.get_action_masks()
            channel_choices = np.zeros((n_agents, BS_CAPACITY), dtype=np.int32)
            power_choices = np.zeros((n_agents, BS_CAPACITY), dtype=np.int32)
            log_probs = np.zeros(n_agents, dtype=np.float32)
            values = np.zeros(n_agents, dtype=np.float32)

            for agent_id, agent in enumerate(agents):
                ch, pw, lp = agent.act(
                    sess, states_actor[agent_id],
                    channel_masks[agent_id], power_masks[agent_id])
                v = agent.get_value(sess, global_obs)
                channel_choices[agent_id] = ch
                power_choices[agent_id] = pw
                log_probs[agent_id] = lp
                values[agent_id] = v

            step_reward, success, fail = env.act_for_training_discrete(
                channel_choices, power_choices)
            sum_reward += step_reward
            epsi_success += success
            fail_sum += fail
            ep_delay_sum += delay_filter.mean_delay()
            ep_delay_n += 1

            done = 0.0 if should_continue_episode(env) else 1.0
            buffer.add(states_actor, global_obs, channel_choices, power_choices,
                       channel_masks, power_masks, log_probs, step_reward,
                       values, done)

        adv_ep, ret_ep = gae_segment(ep_start, buffer.pos)
        accum_adv[ep_start:buffer.pos] = adv_ep
        accum_ret[ep_start:buffer.pos] = ret_ep
        episodes_in_buffer += 1
        if episodes_in_buffer >= update_every or \
                buffer.pos + 64 > buffer.capacity:
            ppo_update(buffer.pos)
            buffer.clear()
            episodes_in_buffer = 0

        episode_count += 1
        recent.append((sum_reward, epsi_success, fail_sum))
        if len(recent) > 200:
            recent.pop(0)

        if episode_count % 50 == 0 or episode_count == 1:
            r = np.mean([x[0] for x in recent])
            s = np.mean([x[1] for x in recent])
            f = np.mean([x[2] for x in recent])
            elapsed = time.time() - t_start
            print("ep %d/%d | reward(MA200)=%.2f success=%.2f fail=%.2f | "
                  "%.2fs/ep | delay=%.2f" %
                  (episode_count, n_episode, r, s, f, elapsed / episode_count,
                   ep_delay_sum / max(ep_delay_n, 1)), flush=True)
        reward_writer.writerow([episode_count, sum_reward, epsi_success, fail_sum,
                                ep_delay_sum / max(ep_delay_n, 1)])
        if episode_count % 1000 == 0:
            reward_log.flush()

        if eval_every > 0 and episode_count % eval_every == 0:
            ev_r, ev_s, ev_f = [], [], []
            for k in range(eval_eps):
                e_env = make_eval_env(k)
                er, es, ef = run_eval_episode(
                    e_env, n_agents, agents, sess, delay_filter, mode)
                ev_r.append(er)
                ev_s.append(es)
                ev_f.append(ef)
            eval_writer.writerow([episode_count,
                                  float(np.mean(ev_r)), float(np.mean(ev_s)),
                                  float(np.mean(ev_f))])
            eval_log.flush()

    reward_log.close()
    eval_log.close()
    sess.close()
    print("DONE tag=%s -> %s" % (tag, out_csv), flush=True)


if __name__ == "__main__":
    main()
