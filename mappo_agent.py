import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc


def _dense_norm_relu(x, units, name):
    x = tf.layers.dense(x, units, name=name)
    x = tc.layers.layer_norm(x, center=True, scale=True, scope=name + "_ln")
    return tf.nn.relu(x)


class MAPPOAgent(object):
    """MAPPO agent: per-agent actor with shared centralized critic."""

    def __init__(self, name, agent_index, n_agents=5, obs_dim=26,
                 max_tasks=6, max_channels=6, n_power=5,
                 clip_eps=0.2, entropy_coef=0.01, lr_actor=3e-4, lr_critic=1e-3,
                 max_grad_norm=0.5, is_first_agent=True,
                 n_route_slots=0, n_route_options=6,
                 structured_route=False):
        self.name = name
        self.agent_index = int(agent_index)
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.max_tasks = int(max_tasks)
        self.max_channels = int(max_channels)
        self.n_power = int(n_power)
        self.n_route_slots = int(n_route_slots)
        self.n_route_options = int(n_route_options)
        self.structured_route = bool(structured_route)
        self.global_state_dim = self.n_agents * self.obs_dim
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef

        # ---- placeholders ----
        self.local_obs = tf.placeholder(tf.float32, [None, self.obs_dim], name=name + "_obs")
        self.global_obs = tf.placeholder(tf.float32, [None, self.global_state_dim], name=name + "_gobs")
        self.channel_mask = tf.placeholder(tf.float32, [None, self.max_channels, self.max_tasks + 1], name=name + "_cmask")
        self.power_mask = tf.placeholder(tf.float32, [None, self.max_tasks, self.n_power], name=name + "_pmask")
        self.action_channel = tf.placeholder(tf.int32, [None, self.max_channels], name=name + "_ach")
        self.action_power = tf.placeholder(tf.int32, [None, self.max_tasks], name=name + "_apw")
        self.old_log_probs = tf.placeholder(tf.float32, [None], name=name + "_old_lp")
        self.advantages = tf.placeholder(tf.float32, [None], name=name + "_adv")
        self.returns = tf.placeholder(tf.float32, [None], name=name + "_ret")
        self.old_value = tf.placeholder(tf.float32, [None], name=name + "_old_v")

        # ---- actor (per-agent) ----
        with tf.variable_scope(name + "/actor"):
            self.channel_logits, self.power_logits, self.route_logits = self._actor(self.local_obs)
        self.route_enabled = self.n_route_slots > 0
        if self.route_enabled:
            self.route_mask = tf.placeholder(
                tf.float32, [None, self.n_route_slots, self.n_route_options],
                name=name + "_rtmask")
            self.action_route = tf.placeholder(
                tf.int32, [None, self.n_route_slots], name=name + "_aroute")

        # ---- critic (shared across agents) ----
        with tf.variable_scope("critic", reuse=not is_first_agent):
            self.value = tf.squeeze(self._critic(self.global_obs), axis=-1)

        # ---- distribution helpers ----
        self.channel_probs = self._masked_softmax(self.channel_logits, self.channel_mask)
        self.power_probs = self._masked_softmax(self.power_logits, self.power_mask)

        self.new_log_probs = self._log_prob(self.channel_logits, self.action_channel, self.channel_mask) \
                           + self._log_prob(self.power_logits, self.action_power, self.power_mask)

        self.entropy = self._entropy(self.channel_probs, self.channel_mask) \
                     + self._entropy(self.power_probs, self.power_mask)

        if self.route_enabled:
            self.route_probs = self._masked_softmax(self.route_logits, self.route_mask)
            self.new_log_probs += self._log_prob(self.route_logits, self.action_route, self.route_mask)
            self.entropy += self._entropy(self.route_probs, self.route_mask)

        # ---- PPO actor loss ----
        ratio = tf.exp(self.new_log_probs - self.old_log_probs)
        surr1 = ratio * self.advantages
        surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * self.advantages
        self.actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        # ---- PPO critic loss with value clipping ----
        if is_first_agent:
            v_clipped = self.old_value + tf.clip_by_value(self.value - self.old_value, -self.clip_eps, self.clip_eps)
            v_loss1 = tf.square(self.value - self.returns)
            v_loss2 = tf.square(v_clipped - self.returns)
            self.critic_loss = 0.5 * tf.reduce_mean(tf.maximum(v_loss1, v_loss2))
        else:
            self.critic_loss = None

        # ---- optimizers ----
        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + "/actor")
        self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic")
        self.train_actor = self._clipped_optimizer(lr_actor, self.actor_loss, self.actor_vars, max_grad_norm)
        if is_first_agent:
            self.train_critic = self._clipped_optimizer(lr_critic, self.critic_loss, self.critic_vars, max_grad_norm)
        else:
            self.train_critic = None

    # ========== network builders ==========

    def _actor(self, obs):
        x = _dense_norm_relu(obs, 256, "fc1")
        x = _dense_norm_relu(x, 128, "fc2")
        x = _dense_norm_relu(x, 64, "fc3")
        channel_logits = tf.layers.dense(x, self.max_channels * (self.max_tasks + 1), name="ch_logits")
        power_logits = tf.layers.dense(x, self.max_tasks * self.n_power, name="pw_logits")
        route_logits = None
        if self.n_route_slots > 0:
            if self.structured_route:
                route_logits = self._structured_route_head(obs, x)
            else:
                route_logits = tf.layers.dense(
                    x, self.n_route_slots * self.n_route_options,
                    name="rt_logits")
                route_logits = tf.reshape(
                    route_logits,
                    [-1, self.n_route_slots, self.n_route_options])
        return (tf.reshape(channel_logits, [-1, self.max_channels, self.max_tasks + 1]),
                tf.reshape(power_logits, [-1, self.max_tasks, self.n_power]),
                route_logits)

    def _structured_route_head(self, obs, trunk):
        """Permutation-equivariant scorer for local + four neighbor targets.

        The v1.1 observation layout stores own destination features at
        (effective load, min remaining time, capacity) and four aligned
        neighbor blocks at obs[28:40].  The same scorer is applied to every
        destination, preventing the route head from memorizing option IDs.
        """
        own_dest = tf.stack([obs[:, 4], obs[:, 6], obs[:, 7]], axis=1)
        own_dest = tf.expand_dims(own_dest, axis=1)
        neighbors = tf.reshape(obs[:, 28:40], [-1, 4, 3])
        destinations = tf.concat([own_dest, neighbors], axis=1)
        noop_logits = tf.layers.dense(
            trunk, self.n_route_slots, name="rt_noop_logits")
        slot_logits = []
        for k in range(self.n_route_slots):
            context = tf.concat(
                [obs[:, 2 * k:2 * k + 2], obs[:, 4:8], obs[:, 46:47]],
                axis=1)
            context = tf.tile(tf.expand_dims(context, 1), [1, 5, 1])
            features = tf.concat([context, destinations], axis=2)
            flat = tf.reshape(features, [-1, 10])
            with tf.variable_scope("route_scorer", reuse=tf.AUTO_REUSE):
                h = tf.layers.dense(flat, 64, activation=tf.nn.relu,
                                    name="fc1")
                score = tf.layers.dense(h, 1, name="score")
            score = tf.reshape(score, [-1, 5])
            slot_logits.append(tf.concat(
                [score, noop_logits[:, k:k + 1]], axis=1))
        return tf.stack(slot_logits, axis=1)

    def _critic(self, global_obs):
        x = _dense_norm_relu(global_obs, 256, "fc1")
        x = _dense_norm_relu(x, 128, "fc2")
        x = _dense_norm_relu(x, 64, "fc3")
        return tf.layers.dense(x, 1, name="v")

    # ========== distribution ops ==========

    def _masked_softmax(self, logits, mask):
        masked = logits + (1.0 - tf.cast(mask, tf.float32)) * (-1e9)
        return tf.nn.softmax(masked, axis=-1)

    def _log_prob(self, logits, actions, mask):
        masked = logits + (1.0 - tf.cast(mask, tf.float32)) * (-1e9)
        lp = tf.nn.log_softmax(masked, axis=-1)
        oh = tf.one_hot(actions, depth=tf.shape(logits)[-1])
        return tf.reduce_sum(tf.reduce_sum(lp * oh, axis=-1), axis=-1)

    def _entropy(self, probs, mask):
        lp = tf.log(tf.clip_by_value(probs, 1e-10, 1.0))
        h = -tf.reduce_sum(probs * lp * tf.cast(mask, tf.float32), axis=-1)
        return tf.reduce_sum(h, axis=-1)

    # ========== optimizer helper ==========

    @staticmethod
    def _clipped_optimizer(lr, loss, var_list, max_grad_norm):
        opt = tf.train.AdamOptimizer(lr)
        grads_and_vars = opt.compute_gradients(loss, var_list=var_list)
        grads, vars_ = zip(*grads_and_vars)
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        return opt.apply_gradients(zip(grads, vars_))

    # ========== session methods ==========

    def act(self, sess, obs, channel_mask, power_mask):
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        channel_mask = np.asarray(channel_mask, dtype=np.float32).reshape(1, self.max_channels, self.max_tasks + 1)
        power_mask = np.asarray(power_mask, dtype=np.float32).reshape(1, self.max_tasks, self.n_power)
        ch_logits, pw_logits, ch_probs, pw_probs = sess.run(
            [self.channel_logits, self.power_logits, self.channel_probs, self.power_probs],
            {self.local_obs: obs, self.channel_mask: channel_mask, self.power_mask: power_mask})
        ch_choices = np.array([np.random.choice(ch_probs.shape[2], p=ch_probs[0, i]) for i in range(self.max_channels)], dtype=np.int32)
        pw_choices = np.array([np.random.choice(pw_probs.shape[2], p=pw_probs[0, i]) for i in range(self.max_tasks)], dtype=np.int32)
        log_prob = self._numpy_log_prob(ch_logits[0], ch_choices, channel_mask[0]) \
                 + self._numpy_log_prob(pw_logits[0], pw_choices, power_mask[0])
        return ch_choices, pw_choices, log_prob

    def get_value(self, sess, global_obs):
        return sess.run(self.value, {self.global_obs: np.asarray(global_obs, dtype=np.float32).reshape(1, -1)})

    def evaluate_actions(self, sess, obs_batch, global_obs_batch, channel_choices, power_choices,
                         channel_masks, power_masks, old_log_probs, old_values, advantages, returns):
        feeds = {
            self.local_obs: np.asarray(obs_batch, dtype=np.float32),
            self.global_obs: np.asarray(global_obs_batch, dtype=np.float32),
            self.action_channel: np.asarray(channel_choices, dtype=np.int32),
            self.action_power: np.asarray(power_choices, dtype=np.int32),
            self.channel_mask: np.asarray(channel_masks, dtype=np.float32),
            self.power_mask: np.asarray(power_masks, dtype=np.float32),
            self.old_log_probs: np.asarray(old_log_probs, dtype=np.float32),
            self.old_value: np.asarray(old_values, dtype=np.float32),
            self.advantages: np.asarray(advantages, dtype=np.float32),
            self.returns: np.asarray(returns, dtype=np.float32),
        }
        _, _, a_loss, c_loss, ent = sess.run(
            [self.train_actor, self.train_critic, self.actor_loss, self.critic_loss, self.entropy], feeds)
        return a_loss, c_loss, ent

    def train_actor_only(self, sess, obs_batch, global_obs_batch, channel_choices, power_choices,
                         channel_masks, power_masks, old_log_probs, advantages):
        feeds = {
            self.local_obs: np.asarray(obs_batch, dtype=np.float32),
            self.global_obs: np.asarray(global_obs_batch, dtype=np.float32),
            self.action_channel: np.asarray(channel_choices, dtype=np.int32),
            self.action_power: np.asarray(power_choices, dtype=np.int32),
            self.channel_mask: np.asarray(channel_masks, dtype=np.float32),
            self.power_mask: np.asarray(power_masks, dtype=np.float32),
            self.old_log_probs: np.asarray(old_log_probs, dtype=np.float32),
            self.advantages: np.asarray(advantages, dtype=np.float32),
        }
        _, a_loss = sess.run([self.train_actor, self.actor_loss], feeds)
        return a_loss

    # ========== route-enabled session methods ==========

    def act_route(self, sess, obs, channel_mask, power_mask, route_mask):
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        cm = np.asarray(channel_mask, dtype=np.float32).reshape(
            1, self.max_channels, self.max_tasks + 1)
        pm = np.asarray(power_mask, dtype=np.float32).reshape(
            1, self.max_tasks, self.n_power)
        rm = np.asarray(route_mask, dtype=np.float32).reshape(
            1, self.n_route_slots, self.n_route_options)
        fetch = [self.channel_probs, self.power_probs, self.route_probs]
        ch_p, pw_p, rt_p = sess.run(fetch, {
            self.local_obs: obs, self.channel_mask: cm,
            self.power_mask: pm, self.route_mask: rm})
        ch = np.array([np.random.choice(ch_p.shape[2], p=ch_p[0, i])
                       for i in range(self.max_channels)], dtype=np.int32)
        pw = np.array([np.random.choice(pw_p.shape[2], p=pw_p[0, i])
                       for i in range(self.max_tasks)], dtype=np.int32)
        rt = np.array([np.random.choice(rt_p.shape[2], p=rt_p[0, i])
                       for i in range(self.n_route_slots)], dtype=np.int32)
        lp = self._numpy_log_prob_probs(ch_p[0], ch, cm[0]) \
            + self._numpy_log_prob_probs(pw_p[0], pw, pm[0]) \
            + self._numpy_log_prob_probs(rt_p[0], rt, rm[0])
        return ch, pw, rt, lp

    @staticmethod
    def _numpy_log_prob_probs(probs, choices, mask):
        probs = np.asarray(probs, dtype=np.float64) * (np.asarray(mask) > 0)
        probs = probs / np.maximum(probs.sum(axis=-1, keepdims=True), 1e-12)
        lp = 0.0
        for i in range(choices.shape[0]):
            lp += np.log(max(probs[i, choices[i]], 1e-12))
        return float(lp)

    def greedy_all(self, sess, obs, channel_mask, power_mask, route_mask):
        """Argmax actions across all heads (evaluation)."""
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        cm = np.asarray(channel_mask, dtype=np.float32).reshape(
            1, self.max_channels, self.max_tasks + 1)
        pm = np.asarray(power_mask, dtype=np.float32).reshape(
            1, self.max_tasks, self.n_power)
        rm = np.asarray(route_mask, dtype=np.float32).reshape(
            1, self.n_route_slots, self.n_route_options)
        fetch = [self.channel_probs, self.power_probs]
        feeds = {self.local_obs: obs, self.channel_mask: cm, self.power_mask: pm}
        if self.route_enabled:
            fetch.append(self.route_probs)
            feeds[self.route_mask] = rm
        outs = sess.run(fetch, feeds)
        ch = np.argmax(outs[0][0], axis=-1).astype(np.int32)
        pw = np.argmax(outs[1][0], axis=-1).astype(np.int32)
        rt = np.argmax(outs[2][0], axis=-1).astype(np.int32) if self.route_enabled else None
        return ch, pw, rt

    def evaluate_actions_route(self, sess, obs_batch, global_obs_batch,
                               channel_choices, power_choices, route_choices,
                               channel_masks, power_masks, route_masks,
                               old_log_probs, old_values, advantages, returns):
        feeds = {
            self.local_obs: np.asarray(obs_batch, dtype=np.float32),
            self.global_obs: np.asarray(global_obs_batch, dtype=np.float32),
            self.action_channel: np.asarray(channel_choices, dtype=np.int32),
            self.action_power: np.asarray(power_choices, dtype=np.int32),
            self.action_route: np.asarray(route_choices, dtype=np.int32),
            self.channel_mask: np.asarray(channel_masks, dtype=np.float32),
            self.power_mask: np.asarray(power_masks, dtype=np.float32),
            self.route_mask: np.asarray(route_masks, dtype=np.float32),
            self.old_log_probs: np.asarray(old_log_probs, dtype=np.float32),
            self.old_value: np.asarray(old_values, dtype=np.float32),
            self.advantages: np.asarray(advantages, dtype=np.float32),
            self.returns: np.asarray(returns, dtype=np.float32),
        }
        _, _, a_loss, c_loss, ent = sess.run(
            [self.train_actor, self.train_critic, self.actor_loss,
             self.critic_loss, self.entropy], feeds)
        return a_loss, c_loss, ent

    def train_actor_only_route(self, sess, obs_batch, global_obs_batch,
                               channel_choices, power_choices, route_choices,
                               channel_masks, power_masks, route_masks,
                               old_log_probs, advantages):
        feeds = {
            self.local_obs: np.asarray(obs_batch, dtype=np.float32),
            self.global_obs: np.asarray(global_obs_batch, dtype=np.float32),
            self.action_channel: np.asarray(channel_choices, dtype=np.int32),
            self.action_power: np.asarray(power_choices, dtype=np.int32),
            self.action_route: np.asarray(route_choices, dtype=np.int32),
            self.channel_mask: np.asarray(channel_masks, dtype=np.float32),
            self.power_mask: np.asarray(power_masks, dtype=np.float32),
            self.route_mask: np.asarray(route_masks, dtype=np.float32),
            self.old_log_probs: np.asarray(old_log_probs, dtype=np.float32),
            self.advantages: np.asarray(advantages, dtype=np.float32),
        }
        _, a_loss = sess.run([self.train_actor, self.actor_loss], feeds)
        return a_loss

    # ========== numpy helpers ==========

    @staticmethod
    def _numpy_log_prob(logits, choices, mask):
        logits = np.asarray(logits, dtype=np.float64)
        mask = np.asarray(mask, dtype=np.float64)
        logits = np.where(mask > 0, logits, -1e9)
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(logits) * mask
        probs = probs / np.maximum(np.sum(probs, axis=-1, keepdims=True), 1e-12)
        lp = 0.0
        for i in range(choices.shape[0]):
            lp += np.log(max(probs[i, choices[i]], 1e-12))
        return float(lp)
