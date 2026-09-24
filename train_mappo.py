import copy
import csv
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from config.config import settings
from Environment_marl_discrete import DiscreteEnviron
from mappo_agent import MAPPOAgent
from mappo_buffer import RolloutBuffer
from curriculum_optimizer import SPMARLCurriculum

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXPERIMENT_OUTPUT_DIR = PROJECT_ROOT / "experiment_output"
EXPERIMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from init_input.dynamic_topology import build_dynamic_env_config, generate_fixed_bs_config  # noqa: E402
OBS_DIM = 26
BS_CAPACITY = 6
N_POWER = 5


def env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _cfg(name, settings_attr, cast=str):
    env = os.environ.get(name, "")
    return cast(env) if env else cast(getattr(settings, settings_attr))


def get_state(env, i):
    task_on_base = env.task_on_base
    task_data_size = env.task_data_size
    task_deadline = env.task_deadline
    task_state = []
    for j in range(len(task_on_base[i])):
        task_state.append((task_data_size[task_on_base[i][j]] - 1e5) / 4e5)
    while len(task_state) < 10:
        task_state.append(0.0)
    for j in range(len(task_on_base[i])):
        task_state.append((task_deadline[task_on_base[i][j]] - 1000.0) / 1000.0)
    while len(task_state) < 20:
        task_state.append(0.0)
    channel_infer = np.asarray(env.V2V_Interference_all[i], dtype=np.float32).flatten()
    if len(channel_infer) > 6:
        channel_infer = channel_infer[:6]
    elif len(channel_infer) < 6:
        channel_infer = np.pad(channel_infer, (0, 6 - len(channel_infer)), constant_values=float(env.sig2))
    return np.concatenate((task_state, channel_infer)).astype(np.float32)


def should_continue_episode(env):
    return (not (np.any(env.task_deadline) == 0)) and (not (np.any(env.task_data_size) == 0))


def dynamic_topology_kwargs(n_bs, seed=None):
    kwargs = {
        "n_bs": n_bs,
        "n_tasks": _cfg("DYNAMIC_N_TASKS", "DYNAMIC_N_TASKS", int),
        "n_channels": BS_CAPACITY, "channel_per_bs": BS_CAPACITY, "max_tasks_per_bs": BS_CAPACITY,
    }
    kwargs["area_size"] = (_cfg("DYNAMIC_AREA_WIDTH", "DYNAMIC_AREA_WIDTH", float),
                           _cfg("DYNAMIC_AREA_HEIGHT", "DYNAMIC_AREA_HEIGHT", float))
    max_ue = _cfg("MAX_UE_DISTANCE", "MAX_UE_DISTANCE", float)
    if max_ue > 0:
        kwargs["max_ue_distance"] = max_ue
    if seed is not None:
        kwargs["seed"] = seed
    return kwargs


def get_variant_bs_config(variant=25):
    cfg = settings.VARIANT_CONFIG[variant]
    bs_positions = np.column_stack([cfg["rsu_x"], cfg["rsu_y"]]).astype(np.float64)
    usable_channels = [list(range(cfg["channel_num"])) for _ in range(cfg["rsu_num"])]
    return bs_positions, usable_channels


def main():
    use_dynamic_topology = env_flag("USE_DYNAMIC_TOPOLOGY", default=str(settings.USE_DYNAMIC_TOPOLOGY))
    use_fixed_curriculum = env_flag("FIXED_CURRICULUM", default=str(settings.FIXED_CURRICULUM))
    use_spmarl = env_flag("USE_SPMARL", default=str(
        int(getattr(settings, "USE_SPMARL", 0)) if hasattr(settings, "USE_SPMARL") else 0))
    dynamic_seed = int(os.environ.get("DYNAMIC_TOPOLOGY_SEED", "1"))
    fixed_variant = int(os.environ.get("FIXED_VARIANT", "0"))

    if use_spmarl:
        n_agents = int(os.environ.get("DYNAMIC_N_BS", str(settings.DYNAMIC_N_BS)))
        sim_dict_init = None
        bs_positions, bs_channels = generate_fixed_bs_config(
            n_bs=n_agents, n_channels=BS_CAPACITY, channel_per_bs=BS_CAPACITY,
            area_size=(_cfg("DYNAMIC_AREA_WIDTH", "DYNAMIC_AREA_WIDTH", float),
                       _cfg("DYNAMIC_AREA_HEIGHT", "DYNAMIC_AREA_HEIGHT", float)),
            seed=dynamic_seed)
        from init_input.experiment_setup import build_fixed_env_config  # noqa: E402
        sim_dict_fixed = build_fixed_env_config(25)
        curriculum = SPMARLCurriculum(
            init_mean=_cfg("SPMARL_INIT_MEAN", "SPMARL_INIT_MEAN", float),
            init_var=_cfg("SPMARL_INIT_VAR", "SPMARL_INIT_VAR", float),
            target_mean=_cfg("SPMARL_TARGET_MEAN", "SPMARL_TARGET_MEAN", float),
            target_var=_cfg("SPMARL_TARGET_VAR", "SPMARL_TARGET_VAR", float),
            context_lower=_cfg("SPMARL_CONTEXT_LOWER", "SPMARL_CONTEXT_LOWER", float),
            context_upper=_cfg("SPMARL_CONTEXT_UPPER", "SPMARL_CONTEXT_UPPER", float),
            max_kl=_cfg("SPMARL_MAX_KL", "SPMARL_MAX_KL", float),
            perf_lb=_cfg("SPMARL_PERF_LB", "SPMARL_PERF_LB", float),
            std_lower_bound=_cfg("SPMARL_STD_LOWER_BOUND", "SPMARL_STD_LOWER_BOUND", float),
            max_ue_distance_easy=_cfg("SPMARL_MAX_UE_EASY", "SPMARL_MAX_UE_EASY", float),
            window_size=_cfg("SPMARL_WINDOW_SIZE", "SPMARL_WINDOW_SIZE", int),
        )
    elif use_fixed_curriculum:
        from init_input.experiment_setup import build_fixed_env_config  # noqa: E402
        n_agents = settings.VARIANT_CONFIG[25]["rsu_num"]
        sim_dict_init = None
        sim_dict_fixed = build_fixed_env_config(25)
        bs_positions, bs_channels = get_variant_bs_config(25)
    elif use_dynamic_topology:
        n_agents = int(os.environ.get("DYNAMIC_N_BS", str(settings.DYNAMIC_N_BS)))
        sim_dict_init = None
        bs_positions, bs_channels = generate_fixed_bs_config(
            n_bs=n_agents, n_channels=BS_CAPACITY, channel_per_bs=BS_CAPACITY,
            area_size=(_cfg("DYNAMIC_AREA_WIDTH", "DYNAMIC_AREA_WIDTH", float),
                       _cfg("DYNAMIC_AREA_HEIGHT", "DYNAMIC_AREA_HEIGHT", float)),
            seed=dynamic_seed)
        sim_dict_fixed = build_dynamic_env_config(
            **dynamic_topology_kwargs(n_bs=n_agents, seed=dynamic_seed),
            bs_positions=bs_positions, usable_channel_of_all_nodes=bs_channels)
    else:
        from init_input.experiment_setup import build_fixed_env_config  # noqa: E402
        variant = fixed_variant or 25
        sim_dict_init = build_fixed_env_config(variant)
        n_agents = len(sim_dict_init["task_on_base"])

    n_episode = _cfg("N_EPISODES", "N_EPISODES", int)
    n_steps = _cfg("N_STEPS", "N_STEPS", int)
    ue_fixed_episodes = _cfg("UE_FIXED_EPISODES", "UE_FIXED_EPISODES", int)
    ue_random_ramp = _cfg("UE_RANDOM_RAMP", "UE_RANDOM_RAMP", int)

    agents = [
        MAPPOAgent("agent%d" % (i + 1), i, n_agents=n_agents, obs_dim=OBS_DIM,
                   max_tasks=BS_CAPACITY, max_channels=BS_CAPACITY, n_power=N_POWER,
                   clip_eps=_cfg("CLIP_EPS", "CLIP_EPS", float),
                   entropy_coef=_cfg("ENTROPY_COEF", "ENTROPY_COEF", float),
                   max_grad_norm=_cfg("MAX_GRAD_NORM", "MAX_GRAD_NORM", float),
                   is_first_agent=(i == 0))
        for i in range(n_agents)
    ]

    buffer = RolloutBuffer(n_steps, n_agents, OBS_DIM, BS_CAPACITY, BS_CAPACITY, N_POWER)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())

    model_dir = SCRIPT_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    saver = tf.train.Saver(max_to_keep=5)

    reward_log = (SCRIPT_DIR / "sum_reward_mappo.csv").open("w", newline="")
    reward_writer = csv.writer(reward_log)
    seed = dynamic_seed if use_dynamic_topology else (24 if use_fixed_curriculum else (fixed_variant or 25))
    reward_writer.writerow(["# seed=%d  n_agents=%d  n_episodes=%d  n_steps=%d" %
                            (seed, n_agents, n_episode, n_steps)])
    reward_writer.writerow(["episode", "sum_reward", "success", "fail"])

    curriculum_log = None
    curriculum_writer = None
    if use_spmarl:
        curriculum_log = (SCRIPT_DIR / "curriculum_mappo.csv").open("w", newline="")
        curriculum_writer = csv.writer(curriculum_log)
        curriculum_writer.writerow(["episode", "context_c", "mu", "sigma", "stage",
                                     "critic_loss_avg", "success_rate"])

    episode_count = 0

    while episode_count < n_episode:
        if use_spmarl:
            context_c = curriculum.sample_context()
            max_ue_dist = curriculum.context_to_max_ue_distance(context_c)
            topo_kw = dynamic_topology_kwargs(n_bs=n_agents, seed=dynamic_seed + episode_count)
            if max_ue_dist is not None:
                topo_kw["max_ue_distance"] = max_ue_dist
            sim_dict = build_dynamic_env_config(
                **topo_kw, bs_positions=bs_positions,
                usable_channel_of_all_nodes=bs_channels)
        elif use_fixed_curriculum:
            if episode_count < ue_fixed_episodes:
                sim_dict = copy.deepcopy(sim_dict_fixed)
            elif episode_count < ue_fixed_episodes + ue_random_ramp:
                p_random = (episode_count - ue_fixed_episodes) / float(ue_random_ramp)
                if np.random.random() < p_random:
                    sim_dict = build_dynamic_env_config(
                        **dynamic_topology_kwargs(n_bs=n_agents, seed=dynamic_seed + episode_count),
                        bs_positions=bs_positions, usable_channel_of_all_nodes=bs_channels)
                else:
                    sim_dict = copy.deepcopy(sim_dict_fixed)
            else:
                sim_dict = build_dynamic_env_config(
                    **dynamic_topology_kwargs(n_bs=n_agents, seed=dynamic_seed + episode_count),
                    bs_positions=bs_positions, usable_channel_of_all_nodes=bs_channels)
        elif use_dynamic_topology:
            if ue_fixed_episodes > 0:
                if episode_count < ue_fixed_episodes:
                    sim_dict = copy.deepcopy(sim_dict_fixed)
                elif episode_count < ue_fixed_episodes + ue_random_ramp:
                    p_random = (episode_count - ue_fixed_episodes) / float(ue_random_ramp)
                    if np.random.random() < p_random:
                        sim_dict = build_dynamic_env_config(
                            **dynamic_topology_kwargs(n_bs=n_agents, seed=dynamic_seed + episode_count),
                            bs_positions=bs_positions, usable_channel_of_all_nodes=bs_channels)
                    else:
                        sim_dict = copy.deepcopy(sim_dict_fixed)
                else:
                    sim_dict = build_dynamic_env_config(
                        **dynamic_topology_kwargs(n_bs=n_agents, seed=dynamic_seed + episode_count),
                        bs_positions=bs_positions, usable_channel_of_all_nodes=bs_channels)
            else:
                sim_dict = build_dynamic_env_config(
                    **dynamic_topology_kwargs(n_bs=n_agents, seed=dynamic_seed + episode_count),
                    bs_positions=bs_positions, usable_channel_of_all_nodes=bs_channels)
        else:
            sim_dict = copy.deepcopy(sim_dict_init)
        env = DiscreteEnviron(sim_dict)

        sum_reward = 0.0
        epsi_success = 0
        fail_sum = 0
        episode_critic_loss = 0.0
        episode_critic_count = 0
        buffer.clear()

        while should_continue_episode(env):
            states = np.asarray([get_state(env, i) for i in range(n_agents)], dtype=np.float32)
            channel_masks, power_masks = env.get_action_masks()
            global_obs = states.flatten()

            channel_choices = np.zeros((n_agents, BS_CAPACITY), dtype=np.int32)
            power_choices = np.zeros((n_agents, BS_CAPACITY), dtype=np.int32)
            log_probs = np.zeros(n_agents, dtype=np.float32)
            values = np.zeros(n_agents, dtype=np.float32)

            for agent_id, agent in enumerate(agents):
                ch, pw, lp = agent.act(sess, states[agent_id], channel_masks[agent_id], power_masks[agent_id])
                v = agent.get_value(sess, global_obs)
                channel_choices[agent_id] = ch
                power_choices[agent_id] = pw
                log_probs[agent_id] = lp
                values[agent_id] = v

            step_reward, success, fail = env.act_for_training_discrete(channel_choices, power_choices)
            sum_reward += step_reward
            epsi_success += success
            fail_sum += fail

            done = 0.0 if should_continue_episode(env) else 1.0

            buffer.add(states, global_obs, channel_choices, power_choices,
                       channel_masks, power_masks, log_probs, step_reward, values, done)

        # --- episode ended, train on full episode ---
        gamma = _cfg("GAMMA", "GAMMA", float)
        gae_lambda = _cfg("GAE_LAMBDA", "GAE_LAMBDA", float)
        k_epochs = _cfg("K_EPOCHS", "K_EPOCHS", int)
        minibatch_size = _cfg("MINIBATCH_SIZE", "MINIBATCH_SIZE", int)
        advantages, returns = buffer.compute_gae(gamma=gamma, gae_lambda=gae_lambda)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        for epoch in range(k_epochs):
            for batch_indices in buffer.get_batches(minibatch_size):
                for agent_id, agent in enumerate(agents):
                    idx = batch_indices
                    if agent_id == 0:
                        a_loss, c_loss, ent = agent.evaluate_actions(
                            sess,
                            buffer.obs[idx, agent_id],
                            buffer.global_obs[idx],
                            buffer.channel_choices[idx, agent_id],
                            buffer.power_choices[idx, agent_id],
                            buffer.channel_masks[idx, agent_id],
                            buffer.power_masks[idx, agent_id],
                            buffer.log_probs[idx, agent_id],
                            buffer.values[idx, agent_id],
                            advantages[idx, agent_id],
                            returns[idx])
                        if use_spmarl and c_loss is not None:
                            episode_critic_loss += float(c_loss)
                            episode_critic_count += 1
                    else:
                        agent.train_actor_only(
                            sess,
                            buffer.obs[idx, agent_id],
                            buffer.global_obs[idx],
                            buffer.channel_choices[idx, agent_id],
                            buffer.power_choices[idx, agent_id],
                            buffer.channel_masks[idx, agent_id],
                            buffer.power_masks[idx, agent_id],
                            buffer.log_probs[idx, agent_id],
                            advantages[idx, agent_id])
        buffer.clear()

        if use_spmarl:
            avg_c_loss = (episode_critic_loss / max(episode_critic_count, 1))
            total = epsi_success + fail_sum
            perf = epsi_success / max(total, 1)
            curriculum.record(context_c, avg_c_loss, perf)
            update_interval = int(_cfg("SPMARL_UPDATE_INTERVAL", "SPMARL_UPDATE_INTERVAL", int))
            if episode_count > 0 and episode_count % update_interval == 0:
                curriculum.update_distribution()

        episode_count += 1
        if use_spmarl:
            s = curriculum.get_state()
            print("Episode: %d, Reward: %.4f, S: %d, F: %d | c=%.3f mu=%.3f std=%.3f stage=%d Lp=%.4f perf=%.3f" %
                  (episode_count, sum_reward, epsi_success, fail_sum,
                   context_c, s["mean"], s["std"], s["stage"],
                   s["recent_lp_mean"], s["recent_perf"]))
            curriculum_writer.writerow([episode_count, context_c,
                                         s["mean"], s["std"], s["stage"],
                                         avg_c_loss, perf])
        else:
            print("Episode: %d, Reward: %.4f, success: %d, fail: %d" % (episode_count, sum_reward, epsi_success, fail_sum))
        reward_writer.writerow([episode_count, sum_reward, epsi_success, fail_sum])

        if episode_count % 1000 == 0:
            saver.save(sess, str(model_dir / "mappo"), global_step=episode_count)

    saver.save(sess, str(model_dir / "mappo_final"))
    reward_log.close()
    if curriculum_log is not None:
        curriculum_log.close()
    sess.close()

    from plot_training import plot_training_curve  # noqa: E402
    plot_training_curve(str(SCRIPT_DIR / "sum_reward_mappo.csv"), title="MAPPO Training")
    if use_spmarl:
        plot_training_curve(str(SCRIPT_DIR / "curriculum_mappo.csv"), title="SPMARL Curriculum")


if __name__ == "__main__":
    main()
