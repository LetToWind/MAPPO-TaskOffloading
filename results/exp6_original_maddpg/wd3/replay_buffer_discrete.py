import numpy as np


class DiscreteReplayBuffer(object):
    """Shared replay buffer for centralized-training MADDPG."""

    def __init__(self, size):
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs, action, reward, next_obs, done, masks, next_masks):
        data = (
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_obs, dtype=np.float32),
            float(done),
            tuple(np.asarray(mask, dtype=np.float32) for mask in masks),
            tuple(np.asarray(mask, dtype=np.float32) for mask in next_masks),
        )
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obs, actions, rewards, next_obs, dones = [], [], [], [], []
        channel_masks, power_masks = [], []
        next_channel_masks, next_power_masks = [], []

        for idx in idxes:
            data = self._storage[idx]
            obs_t, action, reward, obs_tp1, done, masks, next_masks = data
            obs.append(obs_t)
            actions.append(action)
            rewards.append(reward)
            next_obs.append(obs_tp1)
            dones.append(done)
            channel_masks.append(masks[0])
            power_masks.append(masks[1])
            next_channel_masks.append(next_masks[0])
            next_power_masks.append(next_masks[1])

        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            np.asarray(channel_masks, dtype=np.float32),
            np.asarray(power_masks, dtype=np.float32),
            np.asarray(next_channel_masks, dtype=np.float32),
            np.asarray(next_power_masks, dtype=np.float32),
        )

    def sample(self, batch_size):
        idxes = [np.random.randint(0, len(self._storage)) for _ in range(batch_size)]
        return self._encode_sample(idxes)
