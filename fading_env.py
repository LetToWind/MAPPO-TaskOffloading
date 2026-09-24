"""Gauss-Markov log-normal fast fading for the MEC environment.

Design (see 实验规划文档.md / exp9 calibration):
  - one independent fading process per (agent i, channel c)
  - state g[i,c] in dB, stationary marginal N(0, sigma_db^2), evolution
        g(t+1) = rho * g(t) + sqrt(1-rho^2) * xi,  xi ~ N(0, sigma_db^2)
    so the autocorrelation after d steps is exactly rho^d:
    one-step predictability is rho (tracking is possible), multi-step
    staleness decays as rho^d (delay is costly).
  - linear multiplier h = 10^(g/10) applied ONLY to the aggregate
    interference of each (agent, channel) (v1: own signal unfaded), in
    both the physics (SINR via get_inference) and the observation
    (Compute_Interference_2 -> V2V_Interference_all), with the same h.
  - fades are redrawn i.i.d. at episode start; pass `seed` for
    deterministic evaluation trajectories.
"""

import numpy as np

from Environment_marl_discrete import DiscreteEnviron, Environ


class FadingEnviron(DiscreteEnviron):

    def __init__(self, sim_dict_init, rho=0.7, sigma_db=6.0, seed=None):
        super().__init__(sim_dict_init)
        self.rho = float(rho)
        self.sigma_db = float(sigma_db)
        self._rng = (np.random.RandomState(seed) if seed is not None
                     else np.random.RandomState())
        n_ch = max(self.n_channels, self.n_channel_ids)
        self._g = self._rng.normal(0.0, self.sigma_db,
                                   size=(self.n_agents, n_ch))
        self._update_h()

    def _update_h(self):
        self._h = 10.0 ** (self._g / 10.0)

    def step_fading(self):
        """Advance all fading processes by one step (call once per env step)."""
        xi = self._rng.normal(0.0, self.sigma_db, size=self._g.shape)
        self._g = self.rho * self._g + np.sqrt(1.0 - self.rho ** 2) * xi
        self._update_h()

    # ------------------------------------------------------------------
    def step(self, channel_strategy, power_strategy):
        self.step_fading()
        return Environ.step(self, channel_strategy, power_strategy)

    def act_for_training_discrete(self, channel_choices, power_choices):
        self.step_fading()
        return DiscreteEnviron.act_for_training_discrete(
            self, channel_choices, power_choices)

    # ------------------------------------------------------------------
    def Compute_Interference_2(self, distance_matrix, channel_index,
                               strategy_channel_allocation, power_every_task):
        Environ.Compute_Interference_2(
            self, distance_matrix, channel_index,
            strategy_channel_allocation, power_every_task)
        # fade the aggregate interference of each (agent, channel),
        # keeping the noise floor unfaded
        arr = np.asarray(self.V2V_Interference_all, dtype=np.float64)
        faded = (arr - self.sig2) * self._h[:arr.shape[0], :arr.shape[1]] + self.sig2
        self.V2V_Interference_all = faded

    def get_inference(self, distance_matrix, channel_index,
                      strategy_channel_allocation, i, task_no, power_every_task):
        inf_sum, task_on_channel = Environ.get_inference(
            self, distance_matrix, channel_index, strategy_channel_allocation,
            i, task_no, power_every_task)
        # apply the same per-(i, channel) fade used in the observation so
        # the physics matches what the agent observes
        for t, ch in enumerate(task_on_channel):
            if 0 <= ch < self._h.shape[1]:
                inf_sum[t] *= self._h[i, ch]
        return inf_sum, task_on_channel
