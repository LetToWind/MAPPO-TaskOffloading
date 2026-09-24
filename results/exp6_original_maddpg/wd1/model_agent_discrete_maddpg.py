import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc


class DiscreteMADDPG(object):
    """MADDPG agent with decomposed discrete actions and Gumbel-Softmax."""

    def __init__(
        self,
        name,
        agent_index,
        n_agents=5,
        obs_dim=26,
        max_tasks=6,
        max_channels=6,
        n_power=5,
        action_embed_dim=64,
        layer_norm=True,
    ):
        self.name = name
        self.agent_index = int(agent_index)
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.max_tasks = int(max_tasks)
        self.max_channels = int(max_channels)
        self.n_power = int(n_power)
        self.channel_dim = self.max_channels * (self.max_tasks + 1)
        self.power_dim = self.max_tasks * self.n_power
        self.action_dim = self.channel_dim + self.power_dim
        self.joint_action_dim = self.n_agents * self.action_dim
        self.global_state_dim = self.n_agents * self.obs_dim
        self.action_embed_dim = int(action_embed_dim)
        self.layer_norm = layer_norm

        self.local_obs = tf.placeholder(tf.float32, [None, self.obs_dim], name=name + "_local_obs")
        self.global_obs = tf.placeholder(tf.float32, [None, self.global_state_dim], name=name + "_global_obs")
        self.joint_action = tf.placeholder(tf.float32, [None, self.joint_action_dim], name=name + "_joint_action")
        self.target_q = tf.placeholder(tf.float32, [None, 1], name=name + "_target_q")
        self.channel_mask = tf.placeholder(
            tf.float32,
            [None, self.max_channels, self.max_tasks + 1],
            name=name + "_channel_mask",
        )
        self.power_mask = tf.placeholder(
            tf.float32,
            [None, self.max_tasks, self.n_power],
            name=name + "_power_mask",
        )
        self.temperature = tf.placeholder_with_default(1.0, shape=(), name=name + "_temperature")

        with tf.variable_scope(name):
            self.channel_logits, self.power_logits = self._actor_network("actor", self.local_obs)
            self.action_output = self._gumbel_action(self.channel_logits, self.power_logits)
            self.greedy_action_output = self._greedy_action(self.channel_logits, self.power_logits)

            self.critic_output = self._critic_network("critic", self.global_obs, self.joint_action, reuse=False)
            actor_joint_action = self._replace_own_action(self.joint_action, self.action_output)
            self.actor_q = self._critic_network("critic", self.global_obs, actor_joint_action, reuse=True)

        self.actor_loss = -tf.reduce_mean(self.actor_q)
        self.critic_loss = tf.reduce_mean(tf.square(self.target_q - self.critic_output))

        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + "/actor")
        self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + "/critic")
        self.actor_train = tf.train.AdamOptimizer(1e-3).minimize(self.actor_loss, var_list=self.actor_vars)
        self.critic_train = tf.train.AdamOptimizer(1e-3).minimize(self.critic_loss, var_list=self.critic_vars)

    def _dense_norm_relu(self, x, units, name):
        x = tf.layers.dense(x, units, name=name)
        if self.layer_norm:
            x = tc.layers.layer_norm(x, center=True, scale=True, scope=name + "_ln")
        return tf.nn.relu(x)

    def _actor_network(self, scope, obs):
        with tf.variable_scope(scope):
            x = self._dense_norm_relu(obs, 256, "fc1")
            x = self._dense_norm_relu(x, 128, "fc2")
            x = self._dense_norm_relu(x, 64, "fc3")
            channel_logits = tf.layers.dense(
                x,
                self.channel_dim,
                name="channel_logits",
                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
            )
            power_logits = tf.layers.dense(
                x,
                self.power_dim,
                name="power_logits",
                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
            )
            channel_logits = tf.reshape(channel_logits, [-1, self.max_channels, self.max_tasks + 1])
            power_logits = tf.reshape(power_logits, [-1, self.max_tasks, self.n_power])
            return channel_logits, power_logits

    def _masked_logits(self, logits, mask):
        return logits + (1.0 - mask) * (-1e9)

    def _gumbel_softmax_st(self, logits, mask):
        masked_logits = self._masked_logits(logits, mask)
        uniform = tf.random_uniform(tf.shape(masked_logits), minval=1e-6, maxval=1.0 - 1e-6)
        gumbel = -tf.log(-tf.log(uniform))
        soft = tf.nn.softmax((masked_logits + gumbel) / self.temperature, axis=-1)
        hard = tf.one_hot(tf.argmax(soft, axis=-1), tf.shape(soft)[-1], dtype=tf.float32)
        return tf.stop_gradient(hard - soft) + soft

    def _gumbel_action(self, channel_logits, power_logits):
        channel_action = self._gumbel_softmax_st(channel_logits, self.channel_mask)
        power_action = self._gumbel_softmax_st(power_logits, self.power_mask)
        return tf.concat(
            [
                tf.reshape(channel_action, [-1, self.channel_dim]),
                tf.reshape(power_action, [-1, self.power_dim]),
            ],
            axis=1,
        )

    def _greedy_action(self, channel_logits, power_logits):
        channel_logits = self._masked_logits(channel_logits, self.channel_mask)
        power_logits = self._masked_logits(power_logits, self.power_mask)
        channel_hard = tf.one_hot(tf.argmax(channel_logits, axis=-1), self.max_tasks + 1, dtype=tf.float32)
        power_hard = tf.one_hot(tf.argmax(power_logits, axis=-1), self.n_power, dtype=tf.float32)
        return tf.concat(
            [tf.reshape(channel_hard, [-1, self.channel_dim]), tf.reshape(power_hard, [-1, self.power_dim])],
            axis=1,
        )

    def _encode_agent_action(self, action_slice, reuse=False):
        with tf.variable_scope("action_encoder", reuse=reuse):
            x = self._dense_norm_relu(action_slice, self.action_embed_dim, "fc1")
            x = self._dense_norm_relu(x, self.action_embed_dim // 2, "fc2")
            return x

    def _critic_network(self, scope, global_obs, joint_action, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            action_embeddings = []
            for agent_id in range(self.n_agents):
                start = agent_id * self.action_dim
                end = (agent_id + 1) * self.action_dim
                action_slice = joint_action[:, start:end]
                action_embeddings.append(self._encode_agent_action(action_slice, reuse=(reuse or agent_id > 0)))
            x = tf.concat([global_obs] + action_embeddings, axis=1)
            x = self._dense_norm_relu(x, 512, "fc1")
            x = self._dense_norm_relu(x, 256, "fc2")
            x = self._dense_norm_relu(x, 64, "fc3")
            return tf.layers.dense(
                x,
                1,
                name="q",
                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
            )

    def _replace_own_action(self, joint_action, own_action):
        start = self.agent_index * self.action_dim
        end = (self.agent_index + 1) * self.action_dim
        return tf.concat([joint_action[:, :start], own_action, joint_action[:, end:]], axis=1)

    def _onehot_from_choices(self, channel_choices, power_choices):
        channel_choices = np.asarray(channel_choices, dtype=np.int32)
        power_choices = np.asarray(power_choices, dtype=np.int32)
        batch_size = channel_choices.shape[0]
        channel_onehot = np.zeros((batch_size, self.max_channels, self.max_tasks + 1), dtype=np.float32)
        power_onehot = np.zeros((batch_size, self.max_tasks, self.n_power), dtype=np.float32)
        b_idx, c_idx = np.indices((batch_size, self.max_channels))
        channel_onehot[b_idx, c_idx, np.clip(channel_choices, 0, self.max_tasks)] = 1.0
        b_idx, t_idx = np.indices((batch_size, self.max_tasks))
        power_onehot[b_idx, t_idx, np.clip(power_choices, 0, self.n_power - 1)] = 1.0
        return np.concatenate(
            [channel_onehot.reshape(batch_size, -1), power_onehot.reshape(batch_size, -1)],
            axis=1,
        )

    @staticmethod
    def _masked_probs(logits, mask, temperature):
        logits = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
        mask = np.asarray(mask, dtype=np.float64)
        logits = np.where(mask > 0.0, logits, -1e9)
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(logits) * mask
        denom = np.sum(probs, axis=-1, keepdims=True)
        return probs / np.maximum(denom, 1e-12)

    @staticmethod
    def _sample_choices(probs, mask, greedy):
        if greedy:
            return np.argmax(probs, axis=-1).astype(np.int32)
        cdf = np.cumsum(probs, axis=-1)
        r = np.random.random(probs.shape[:-1] + (1,))
        return np.argmax(r < cdf, axis=-1).astype(np.int32)

    @staticmethod
    def _random_choices(mask):
        rand_scores = np.where(mask > 0.0, np.random.random(mask.shape), -1.0)
        return np.argmax(rand_scores, axis=-1).astype(np.int32)

    def action(self, sess, obs, channel_mask, power_mask, epsilon=0.0, temperature=1.0, greedy=False):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        channel_mask = np.asarray(channel_mask, dtype=np.float32)
        power_mask = np.asarray(power_mask, dtype=np.float32)
        if channel_mask.ndim == 2:
            channel_mask = channel_mask.reshape(1, self.max_channels, self.max_tasks + 1)
        if power_mask.ndim == 2:
            power_mask = power_mask.reshape(1, self.max_tasks, self.n_power)

        if np.random.random() < epsilon:
            channel_choices = self._random_choices(channel_mask)
            power_choices = self._random_choices(power_mask)
        else:
            channel_logits, power_logits = sess.run(
                [self.channel_logits, self.power_logits],
                {self.local_obs: obs, self.channel_mask: channel_mask, self.power_mask: power_mask},
            )
            channel_probs = self._masked_probs(channel_logits, channel_mask, temperature)
            power_probs = self._masked_probs(power_logits, power_mask, temperature)
            channel_choices = self._sample_choices(channel_probs, channel_mask, greedy)
            power_choices = self._sample_choices(power_probs, power_mask, greedy)

        action_onehot = self._onehot_from_choices(channel_choices, power_choices)
        return channel_choices, power_choices, action_onehot

    def greedy_action(self, sess, obs, channel_mask, power_mask):
        """Greedy one-hot joint action computed fully in-graph (no Python loops).

        Equivalent to ``action(..., epsilon=0.0, greedy=True)[2]`` but avoids
        the GPU->host logits transfer and per-(batch, head) Python loops, which
        is the dominant cost when computing target actions during training.
        """
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        channel_mask = np.asarray(channel_mask, dtype=np.float32)
        power_mask = np.asarray(power_mask, dtype=np.float32)
        if channel_mask.ndim == 2:
            channel_mask = channel_mask.reshape(1, self.max_channels, self.max_tasks + 1)
        if power_mask.ndim == 2:
            power_mask = power_mask.reshape(1, self.max_tasks, self.n_power)
        return sess.run(
            self.greedy_action_output,
            {self.local_obs: obs, self.channel_mask: channel_mask, self.power_mask: power_mask},
        )

    def train_actor(self, sess, global_obs, local_obs, joint_action, channel_mask, power_mask, temperature=1.0):
        return sess.run(
            [self.actor_train, self.actor_loss],
            {
                self.global_obs: global_obs,
                self.local_obs: local_obs,
                self.joint_action: joint_action,
                self.channel_mask: channel_mask,
                self.power_mask: power_mask,
                self.temperature: temperature,
            },
        )[1]

    def train_critic(self, sess, global_obs, joint_action, target_q):
        return sess.run(
            [self.critic_train, self.critic_loss],
            {self.global_obs: global_obs, self.joint_action: joint_action, self.target_q: target_q},
        )[1]

    def Q(self, sess, global_obs, joint_action):
        return sess.run(self.critic_output, {self.global_obs: global_obs, self.joint_action: joint_action})
