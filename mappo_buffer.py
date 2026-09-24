import numpy as np


class RolloutBuffer(object):

    def __init__(self, n_steps, n_agents, obs_dim, max_channels, max_tasks, n_power):
        self.n_steps = int(n_steps)
        self.n_agents = int(n_agents)
        self.capacity = self.n_steps
        self.pos = 0
        self.full = False

        G = self.n_agents
        S = self.n_steps
        D = int(obs_dim)
        C, T, P = int(max_channels), int(max_tasks), int(n_power)

        self.obs = np.zeros((S, G, D), dtype=np.float32)
        self.global_obs = np.zeros((S, G * D), dtype=np.float32)
        self.channel_choices = np.zeros((S, G, C), dtype=np.int32)
        self.power_choices = np.zeros((S, G, T), dtype=np.int32)
        self.channel_masks = np.zeros((S, G, C, T + 1), dtype=np.float32)
        self.power_masks = np.zeros((S, G, T, P), dtype=np.float32)
        self.log_probs = np.zeros((S, G), dtype=np.float32)
        self.rewards = np.zeros(S, dtype=np.float32)
        self.values = np.zeros((S, G), dtype=np.float32)
        self.dones = np.zeros(S, dtype=np.float32)

    def add(self, obs, global_obs, channel_choices, power_choices,
            channel_mask, power_mask, log_probs, reward, values, done):
        p = self.pos
        self.obs[p] = obs
        self.global_obs[p] = global_obs
        self.channel_choices[p] = channel_choices
        self.power_choices[p] = power_choices
        self.channel_masks[p] = channel_mask
        self.power_masks[p] = power_mask
        self.log_probs[p] = log_probs
        self.rewards[p] = reward
        self.values[p] = values
        self.dones[p] = done
        self.pos += 1
        if self.pos >= self.capacity:
            self.full = True

    def compute_gae(self, gamma=0.99, gae_lambda=0.95):
        n = self.pos if not self.full else self.capacity
        advantages_agent = np.zeros((n, self.n_agents), dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        for agent_id in range(self.n_agents):
            adv, ret = self._compute_gae_for_agent(agent_id, n, gamma, gae_lambda)
            advantages_agent[:, agent_id] = adv
            if agent_id == 0:
                returns = ret
        return advantages_agent, returns

    def _compute_gae_for_agent(self, agent_id, n, gamma, gae_lambda):
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(n)):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t, agent_id]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t, agent_id]
            next_value = self.values[t, agent_id]
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def get_batches(self, batch_size):
        n = self.pos if not self.full else self.capacity
        indices = np.arange(n)
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            yield indices[start:start + batch_size]

    def clear(self):
        self.pos = 0
        self.full = False
