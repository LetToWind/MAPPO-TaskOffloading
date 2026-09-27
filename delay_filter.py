"""Delay filter implementing DSID-POMDP style per-component observation delays.

Semantics follow RDC (NeurIPS 2025) Appendix C.1: the environment keeps a
history of delay-free observations; each agent's actually-received
observation replaces selected components with values retrieved from past
steps according to per-(agent, component) delay processes.

Component groups per agent i (see obs_utils for the layout):
  - one group per interference channel  (1 dim each, 6 groups)
  - one group per neighbor node         (2 dims each, n_agents-1 groups)

Modes:
  none    : no delay (oracle observation)
  fixed   : every group delayed by DELAY_VALUE steps
  partial : a random subset of groups (prob partial_p) delayed by
            DELAY_VALUE steps, subset resampled each episode
  unfixed : each group's delay re-draws a target ~ U[d_min, d_max] every
            step and moves toward it under the temporal-consistency
            constraint  d_t <= d_{t-1} + 1  (information never gets older
            than one extra step per step), matching the DSID-POMDP
            constraint d_t^ij < min(d_{t-1}^ij + 1, T).
"""

import numpy as np

import obs_utils


class DelayFilter(object):

    MODES = ("none", "fixed", "partial", "unfixed", "frozen")

    def __init__(self, n_agents, obs_d, n_channels=obs_utils.N_INTERF_DIMS,
                 mode="none", d_value=0, d_min=0, d_max=0, partial_p=0.5,
                 max_delay=None, rng=None, groups=None):
        assert mode in self.MODES, "unknown delay mode: %s" % mode
        self.n_agents = int(n_agents)
        self.obs_d = int(obs_d)
        self.n_channels = int(n_channels)
        self.n_neighbors = self.n_agents - 1
        self.mode = mode
        self.d_value = int(d_value)
        self.d_min = int(d_min)
        self.d_max = int(d_max)
        self.partial_p = float(partial_p)
        self.rng = rng if rng is not None else np.random
        self.max_delay = int(max_delay) if max_delay is not None else max(
            self.d_value, self.d_max, 1)

        if groups is not None:
            # explicit group index arrays (e.g. neighbor-block-only layout)
            self.groups = [np.asarray(g, dtype=np.int64) for g in groups]
        else:
            # legacy 34-dim layout: per-channel groups + per-neighbor groups
            infer_start = 2 * obs_utils.N_TASK_SLOTS
            gs = [np.array([infer_start + c], dtype=np.int64)
                  for c in range(self.n_channels)]
            neigh_start = infer_start + self.n_channels
            for j in range(self.n_neighbors):
                gs.append(np.array([neigh_start + 2 * j,
                                    neigh_start + 2 * j + 1], dtype=np.int64))
            self.groups = gs
        self.n_groups = len(self.groups)

        self._sample_initial_delays()
        self.history = []

    # ------------------------------------------------------------------
    def _sample_initial_delays(self):
        n = (self.n_agents, self.n_groups)
        if self.mode == "none" or self.mode == "frozen":
            self.delays = np.zeros(n, dtype=np.int64)
        elif self.mode == "fixed":
            self.delays = np.full(n, self.d_value, dtype=np.int64)
        elif self.mode == "partial":
            mask = self.rng.random_sample(n) < self.partial_p
            self.delays = (mask * self.d_value).astype(np.int64)
        else:  # unfixed
            if self.d_max < self.d_min:
                self.d_min, self.d_max = self.d_max, self.d_min
            self.delays = self.rng.randint(
                self.d_min, self.d_max + 1, size=n).astype(np.int64)
        self._partial_mask = (self.delays > 0) if self.mode == "partial" else None

    # ------------------------------------------------------------------
    def reset(self):
        """Call at episode start (before first observation)."""
        self._sample_initial_delays()
        self.history = []

    def append(self, obs_true):
        """Record the delay-free joint observation of the current step."""
        self.history.append(np.array(obs_true, dtype=np.float32, copy=True))

    def step_delays(self):
        """Evolve per-step delay processes (call once per env step)."""
        if self.mode == "unfixed":
            target = self.rng.randint(
                self.d_min, self.d_max + 1,
                size=(self.n_agents, self.n_groups)).astype(np.int64)
            prev = self.delays
            self.delays = np.where(
                target > prev, np.minimum(prev + 1, target), target)
        # fixed / partial / none keep constant delays within an episode

    def observe(self):
        """Return the delayed joint observation for the current step.

        Must be called after `append` for the current step. Delay 0 returns
        the current value; delay d retrieves history[t-d] (clamped to the
        start of the episode).
        """
        t = len(self.history) - 1
        out = self.history[t].copy()
        if self.mode == "frozen":
            # VoI calibration: ALL delayed groups pinned to the episode's
            # first frame (best-inertia proxy); real-time dims untouched
            first = self.history[0]
            for i in range(self.n_agents):
                for g, idx in enumerate(self.groups):
                    out[i, idx] = first[i, idx]
            return out
        for i in range(self.n_agents):
            for g, idx in enumerate(self.groups):
                d = int(self.delays[i, g])
                if d > 0:
                    src = self.history[max(t - min(d, t), 0)]
                    out[i, idx] = src[i, idx]
        return out

    def delay_vector(self):
        """Current delay matrix, shape (n_agents, n_groups)."""
        return self.delays.copy()

    def mean_delay(self):
        if self.mode in ("none", "frozen"):
            return 0.0
        return float(self.delays.mean())


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # self test
    rng = np.random.RandomState(0)
    n_agents, od = 5, obs_utils.obs_dim(5)
    f = DelayFilter(n_agents, od, mode="fixed", d_value=2, rng=rng)
    f.reset()
    seq = []
    for t in range(6):
        obs = np.zeros((n_agents, od), dtype=np.float32)
        obs[:, obs_utils.infer_slice(n_agents)] = float(t)      # interference = t
        obs[:, obs_utils.neighbor_slice(n_agents)] = 10.0 + t   # neighbors = 10+t
        seq.append(obs)
        f.step_delays()
        f.append(obs)
        delayed = f.observe()
        assert np.allclose(delayed[:, obs_utils.infer_slice(n_agents)], max(t - 2, 0)), \
            "fixed delay wrong at t=%d" % t
        assert np.allclose(delayed[:, obs_utils.neighbor_slice(n_agents)], 10.0 + max(t - 2, 0))
        assert np.allclose(delayed[:, :20], 0.0), "self-state must stay real-time"

    f = DelayFilter(n_agents, od, mode="unfixed", d_min=1, d_max=3, rng=rng)
    f.reset()
    prev = f.delay_vector().copy()
    for t in range(50):
        obs = rng.randn(n_agents, od).astype(np.float32)
        f.step_delays()
        f.append(obs)
        _ = f.observe()
        cur = f.delay_vector()
        assert np.all(cur >= 1) and np.all(cur <= 3)
        assert np.all(cur <= prev + 1), "temporal consistency violated"
        prev = cur
    print("DelayFilter self-test OK")
