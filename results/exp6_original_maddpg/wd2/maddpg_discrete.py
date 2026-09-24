import copy
import csv
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from config.config import settings
from Environment_marl_discrete import DiscreteEnviron
from model_agent_discrete_maddpg import DiscreteMADDPG
from replay_buffer_discrete import DiscreteReplayBuffer

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
GAMMA = 0.9
TAU = 0.01


def create_init_update(online_name, target_name, tau=TAU):
    online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=online_name)
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_name)
    online_vars = sorted(online_vars, key=lambda var: var.name)
    target_vars = sorted(target_vars, key=lambda var: var.name)
    target_init = [tf.assign(target, online) for online, target in zip(online_vars, target_vars)]
    target_update = [tf.assign(target, tau * online + (1.0 - tau) * target) for online, target in zip(online_vars, target_vars)]
    return target_init, target_update


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


def env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _cfg(name, settings_attr, cast=str):
    env = os.environ.get(name, "")
    return cast(env) if env else cast(getattr(settings, settings_attr))


def dynamic_topology_kwargs(n_bs, seed=None):
    kwargs = {
        "n_bs": n_bs,
        "n_tasks": _cfg("DYNAMIC_N_TASKS", "DYNAMIC_N_TASKS", int),
        "n_channels": BS_CAPACITY,
        "channel_per_bs": BS_CAPACITY,
        "max_tasks_per_bs": BS_CAPACITY,
    }
    kwargs["area_size"] = (_cfg("DYNAMIC_AREA_WIDTH", "DYNAMIC_AREA_WIDTH", float),
                           _cfg("DYNAMIC_AREA_HEIGHT", "DYNAMIC_AREA_HEIGHT", float))
    max_ue = _cfg("MAX_UE_DISTANCE", "MAX_UE_DISTANCE", float)
    if max_ue > 0:
        kwargs["max_ue_distance"] = max_ue
    if seed is not None:
        kwargs["seed"] = seed
    return kwargs


def flatten_obs(obs_batch):
    return obs_batch.reshape((obs_batch.shape[0], -1))


def should_continue_episode(env):
    return (not (np.any(env.task_deadline) == 0)) and (not (np.any(env.task_data_size) == 0))


def train_agents(agents, target_agents, replay_buffer, sess, target_updates, batch_size, temperature):
    (obs_batch, act_batch, rew_batch, next_obs_batch, done_batch, channel_masks, power_masks, next_channel_masks, next_power_masks) = replay_buffer.sample(batch_size)

    global_obs = flatten_obs(obs_batch)
    global_next_obs = flatten_obs(next_obs_batch)
    joint_action = act_batch.reshape((batch_size, -1))
    n_agents = len(agents)

    target_actions = []
    for other_id, target_agent in enumerate(target_agents):
        next_action = target_agent.greedy_action(
            sess,
            next_obs_batch[:, other_id, :],
            next_channel_masks[:, other_id, :, :],
            next_power_masks[:, other_id, :, :],
        )
        target_actions.append(next_action)
    target_joint_action = np.concatenate(target_actions, axis=1)

    target_q_batch = rew_batch.reshape(-1, 1) + GAMMA * (1.0 - done_batch.reshape(-1, 1)) * target_agents[0].Q(sess, global_next_obs, target_joint_action)

    actor_losses = []
    critic_losses = []
    for agent_id in range(n_agents):
        target_q = target_q_batch  # same TD target for all agents (cooperative)
        critic_loss = agents[agent_id].train_critic(sess, global_obs, joint_action, target_q)
        actor_loss = agents[agent_id].train_actor(
            sess,
            global_obs,
            obs_batch[:, agent_id, :],
            joint_action,
            channel_masks[:, agent_id, :, :],
            power_masks[:, agent_id, :, :],
            temperature=temperature,
        )
        sess.run(target_updates[agent_id])
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
    return actor_losses, critic_losses


def main():
    use_dynamic_topology = env_flag("USE_DYNAMIC_TOPOLOGY", default="1")
    dynamic_seed = int(os.environ.get("DYNAMIC_TOPOLOGY_SEED", "1"))
    fixed_variant = int(os.environ.get("FIXED_VARIANT", "0"))

    if use_dynamic_topology:
        n_agents = int(os.environ.get("DYNAMIC_N_BS", str(settings.DYNAMIC_N_BS)))
        sim_dict_init = None
        area_width = _cfg("DYNAMIC_AREA_WIDTH", "DYNAMIC_AREA_WIDTH", float)
        area_height = _cfg("DYNAMIC_AREA_HEIGHT", "DYNAMIC_AREA_HEIGHT", float)
        bs_positions, bs_channels = generate_fixed_bs_config(
            n_bs=n_agents,
            n_channels=BS_CAPACITY,
            channel_per_bs=BS_CAPACITY,
            area_size=(area_width, area_height),
            seed=dynamic_seed,
        )
        sim_dict_fixed = build_dynamic_env_config(
            **dynamic_topology_kwargs(n_bs=n_agents, seed=dynamic_seed),
            bs_positions=bs_positions,
            usable_channel_of_all_nodes=bs_channels,
        )
    else:
        from init_input.experiment_setup import build_fixed_env_config  # noqa: E402
        variant = fixed_variant or 25
        sim_dict_init = build_fixed_env_config(variant)
        n_agents = len(sim_dict_init["task_on_base"])

    n_episode = _cfg("N_EPISODES", "N_EPISODES", int)
    batch_size = _cfg("BATCH_SIZE", "BATCH_SIZE", int)
    memory_size = _cfg("MEMORY_SIZE", "MEMORY_SIZE", int)
    epsi_final = 0.01
    epsi_anneal_length = int(0.8 * n_episode)
    ue_fixed_episodes = _cfg("UE_FIXED_EPISODES", "UE_FIXED_EPISODES", int)
    ue_random_ramp = _cfg("UE_RANDOM_RAMP", "UE_RANDOM_RAMP", int)

    agents = [
        DiscreteMADDPG("agent%d" % (i + 1), i, n_agents=n_agents, obs_dim=OBS_DIM,
                       max_tasks=BS_CAPACITY, max_channels=BS_CAPACITY, n_power=N_POWER)
        for i in range(n_agents)
    ]
    target_agents = [
        DiscreteMADDPG("agent%d_target" % (i + 1), i, n_agents=n_agents, obs_dim=OBS_DIM,
                       max_tasks=BS_CAPACITY, max_channels=BS_CAPACITY, n_power=N_POWER)
        for i in range(n_agents)
    ]

    target_inits = []
    target_updates = []
    for i in range(n_agents):
        init_ops, update_ops = create_init_update("agent%d" % (i + 1), "agent%d_target" % (i + 1))
        target_inits.extend(init_ops)
        target_updates.append(update_ops)

    replay_buffer = DiscreteReplayBuffer(memory_size)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())
    sess.run(target_inits)

    model_dir = SCRIPT_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    saver = tf.train.Saver(max_to_keep=5)

    seed = dynamic_seed if use_dynamic_topology else (fixed_variant or 25)
    reward_log = (SCRIPT_DIR / "sum_reward_discrete.csv").open("w", newline="")
    reward_writer = csv.writer(reward_log)
    reward_writer.writerow(["# seed=%d  n_agents=%d  n_episodes=%d  batch_size=%d  memory_size=%d" %
                            (seed, n_agents, n_episode, batch_size, memory_size)])
    reward_writer.writerow(["episode", "sum_reward", "success", "fail"])

    for i_episode in range(n_episode):
        if use_dynamic_topology:
            if ue_fixed_episodes > 0:
                if i_episode < ue_fixed_episodes:
                    sim_dict = copy.deepcopy(sim_dict_fixed)
                elif i_episode < ue_fixed_episodes + ue_random_ramp:
                    p_random = (i_episode - ue_fixed_episodes) / float(ue_random_ramp)
                    if np.random.random() < p_random:
                        sim_dict = build_dynamic_env_config(
                            **dynamic_topology_kwargs(n_bs=n_agents, seed=dynamic_seed + i_episode),
                            bs_positions=bs_positions, usable_channel_of_all_nodes=bs_channels)
                    else:
                        sim_dict = copy.deepcopy(sim_dict_fixed)
                else:
                    sim_dict = build_dynamic_env_config(
                        **dynamic_topology_kwargs(n_bs=n_agents, seed=dynamic_seed + i_episode),
                        bs_positions=bs_positions, usable_channel_of_all_nodes=bs_channels)
            else:
                sim_dict = build_dynamic_env_config(
                    **dynamic_topology_kwargs(n_bs=n_agents, seed=dynamic_seed + i_episode),
                    bs_positions=bs_positions, usable_channel_of_all_nodes=bs_channels)
        else:
            sim_dict = copy.deepcopy(sim_dict_init)
        env = DiscreteEnviron(sim_dict)

        if i_episode < epsi_anneal_length:
            epsilon = 1.0 - i_episode * (1.0 - epsi_final) / (epsi_anneal_length - 1)
        elif i_episode < 10000:
            epsilon = epsi_final
        else:
            epsilon = 0.0
        temperature = max(0.1, 1.5 - i_episode / float(max(epsi_anneal_length, 1)))

        sum_reward = 0.0
        epsi_success = 0
        fail_sum = 0

        while should_continue_episode(env):
            states = np.asarray([get_state(env, i) for i in range(n_agents)], dtype=np.float32)
            channel_masks, power_masks = env.get_action_masks()

            channel_choices = np.zeros((n_agents, BS_CAPACITY), dtype=np.int32)
            power_choices = np.zeros((n_agents, BS_CAPACITY), dtype=np.int32)
            action_onehots = []
            for agent_id, agent in enumerate(agents):
                agent_channel, agent_power, agent_action = agent.action(
                    sess,
                    states[agent_id],
                    channel_masks[agent_id],
                    power_masks[agent_id],
                    epsilon=epsilon,
                    temperature=temperature,
                    greedy=False,
                )
                channel_choices[agent_id] = agent_channel[0]
                power_choices[agent_id] = agent_power[0]
                action_onehots.append(agent_action[0])

            train_reward, success, fail = env.act_for_training_discrete(channel_choices, power_choices)
            sum_reward += train_reward
            epsi_success += success
            fail_sum += fail

            next_states = np.asarray(
                [get_state(env, i) for i in range(n_agents)],
                dtype=np.float32,
            )
            next_channel_masks, next_power_masks = env.get_action_masks()
            done = 0.0 if should_continue_episode(env) else 1.0

            replay_buffer.add(
                states,
                np.asarray(action_onehots, dtype=np.float32),
                train_reward,
                next_states,
                done,
                (channel_masks, power_masks),
                (next_channel_masks, next_power_masks),
            )
            if len(replay_buffer) >= batch_size:
                train_agents(agents, target_agents, replay_buffer, sess, target_updates, batch_size, temperature)

        print(
            "Episode: "
            + str(i_episode)
            + ", Explore: "
            + str(round(epsilon, 4))
            + ", Temperature: "
            + str(round(temperature, 4))
            + ", Reward: "
            + str(round(sum_reward, 4))
            + ", success: "
            + str(epsi_success)
            + ", fail: "
            + str(fail_sum)
        )
        reward_writer.writerow([i_episode, sum_reward, epsi_success, fail_sum])

        if i_episode % 1000 == 0:
            saver.save(sess, str(model_dir / "maddpg"), global_step=i_episode)

    saver.save(sess, str(model_dir / "maddpg_final"))
    reward_log.close()
    sess.close()

    from plot_training import plot_training_curve  # noqa: E402
    plot_training_curve(str(SCRIPT_DIR / "sum_reward_discrete.csv"), title="MADDPG Training")


if __name__ == "__main__":
    main()
