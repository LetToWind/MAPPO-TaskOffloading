"""
SPMARL Curriculum Optimizer for MAPPO training.

Implements the TD-error-based learning progress curriculum from:
"Learning Progress Driven Multi-Agent Curriculum" (Zhao et al., ICML 2025)

Core idea: replace hard-coded curriculum schedules with an adaptive
context distribution that is updated via trust-region optimization,
using critic loss (TD-error) as the learning progress signal instead
of sparse episode returns.

Context c in [0, 1] maps to environment difficulty:
  c=0 -> easy   (UE constrained near BS, max_ue_distance large positive)
  c=1 -> target (no distance constraint, full dynamic topology)
"""

import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
except ImportError:
    _scipy_minimize = None


def _gaussian_log_pdf(x, mean, var):
    """Log-probability of x under N(mean, var)."""
    return -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)


def _gaussian_pdf(x, mean, var):
    return np.exp(_gaussian_log_pdf(x, mean, var))


def _gaussian_kl(mean1, var1, mean2, var2):
    """KL(N(mean1,var1) || N(mean2,var2))."""
    return 0.5 * (
        np.log(max(var2, 1e-12) / max(var1, 1e-12))
        + var1 / max(var2, 1e-12)
        + (mean1 - mean2) ** 2 / max(var2, 1e-12)
        - 1.0
    )


class SPMARLCurriculum:
    """Adaptive curriculum that controls task difficulty via a
    Gaussian context distribution updated by learning progress."""

    def __init__(
        self,
        init_mean=0.15,
        init_var=0.05,
        target_mean=1.0,
        target_var=0.004,
        context_lower=0.0,
        context_upper=1.0,
        max_kl=0.05,
        perf_lb=0.55,
        std_lower_bound=0.02,
        max_ue_distance_easy=300.0,
        window_size=20,
        use_stage2=True,
    ):
        # ---- distribution parameters ----
        self.mean = float(init_mean)
        self.var = float(init_var)
        self.target_mean = float(target_mean)
        self.target_var = float(target_var)
        self.lower = float(context_lower)
        self.upper = float(context_upper)
        self.max_kl = float(max_kl)
        self.perf_lb = float(perf_lb)
        self.std_lower_bound = float(std_lower_bound)

        # ---- environment mapping ----
        self.max_ue_distance_easy = float(max_ue_distance_easy)

        # ---- history buffers ----
        self.window_size = int(window_size)
        self.context_history = []
        self.lp_history = []         # learning progress (critic loss)
        self.performance_history = []  # success rate

        # ---- stage control ----
        self.use_stage2 = use_stage2
        self.kl_updates_since_stage2 = 0
        self.kl_threshold_for_stage2 = 20
        self.step_counter = 0
        self.is_stage2 = False

    # ==================================================================
    #  Context sampling
    # ==================================================================

    def sample_context(self):
        """Sample a context value from current Gaussian, clipped to bounds."""
        c = np.random.normal(self.mean, np.sqrt(max(self.var, 1e-8)))
        c = np.clip(c, self.lower, self.upper)
        return c

    def sample_contexts(self, n=1):
        return np.array([self.sample_context() for _ in range(int(n))], dtype=np.float64)

    # ==================================================================
    #  Context -> Environment parameters
    # ==================================================================

    def context_to_max_ue_distance(self, c):
        """Map context c to MAX_UE_DISTANCE.

        c=0 -> max_ue_distance_easy (UE constrained near BS)
        c=1 -> None (no constraint, full dynamic = target)

        Uses linear interpolation; when c >= 1 we return None.
        """
        if c >= 1.0:
            return None
        d = self.max_ue_distance_easy * (1.0 - c)
        if d < 1.0:
            return None
        return float(d)

    def context_to_task_data_size_range(self, c):
        """Optionally widen task data size with difficulty."""
        base_min, base_max = 5e4, 2e5
        hard_min, hard_max = 1e5, 5e5
        tmin = int(base_min + (hard_min - base_min) * c)
        tmax = int(base_max + (hard_max - base_max) * c)
        return (tmin, tmax)

    def context_to_task_deadline_range(self, c):
        """Optionally tighten deadlines with difficulty."""
        base_min, base_max = 1500, 2500
        hard_min, hard_max = 500, 1500
        dmin = int(base_min + (hard_min - base_min) * c)
        dmax = int(base_max + (hard_max - base_max) * c)
        return (dmin, dmax)

    # ==================================================================
    #  Recording & history
    # ==================================================================

    def record(self, context, learning_progress, performance):
        """Store one data point for later distribution update."""
        self.context_history.append(float(context))
        self.lp_history.append(float(learning_progress))
        self.performance_history.append(float(performance))
        if len(self.context_history) > self.window_size * 5:
            self.context_history = self.context_history[-self.window_size * 2:]
            self.lp_history = self.lp_history[-self.window_size * 2:]
            self.performance_history = self.performance_history[-self.window_size * 2:]

    def recent_contexts(self):
        return np.array(self.context_history[-self.window_size:], dtype=np.float64)

    def recent_lps(self):
        return np.array(self.lp_history[-self.window_size:], dtype=np.float64)

    def recent_performance(self):
        if len(self.performance_history) < 5:
            return 0.0
        return float(np.mean(self.performance_history[-5:]))

    # ==================================================================
    #  Distribution update (two-stage optimization)
    # ==================================================================

    def update_distribution(self):
        """Run one step of two-stage context-distribution optimization."""
        if len(self.context_history) < self.window_size:
            return  # not enough data

        contexts = self.recent_contexts()
        lps = self.recent_lps()
        recent_perf = self.recent_performance()
        old_mean, old_var = self.mean, self.var

        if self.use_stage2 and recent_perf >= self.perf_lb:
            self.is_stage2 = True
            self._optimize_stage2(contexts, old_mean, old_var)
        else:
            self.is_stage2 = False
            self._optimize_stage1(contexts, lps, old_mean, old_var)

        # enforce lower bound on standard deviation
        min_var = self.std_lower_bound ** 2
        if self.var < min_var:
            self.var = min_var

        self.step_counter += 1

    # ------------------------------------------------------------------
    #  Stage 1: maximize learning progress under KL constraint
    # ------------------------------------------------------------------

    def _optimize_stage1(self, contexts, lps, old_mean, old_var):
        """max_{mean,log_var}  E_{c~N(mean,var)}[LP(c)]
           s.t.  KL(N(mean,var) || N(old_mean,old_var)) <= max_kl
        """

        def objective(params):
            mean, log_var = params[0], params[1]
            var = np.exp(log_var) + 1e-8
            weights = np.array([
                _gaussian_pdf(c, mean, var) / max(_gaussian_pdf(c, old_mean, old_var), 1e-12)
                for c in contexts
            ], dtype=np.float64)
            weights = np.clip(weights, 0.0, 10.0)
            expected_lp = np.mean(weights * lps)
            return -expected_lp  # minimize negative = maximize

        def kl_constraint(params):
            mean, log_var = params[0], params[1]
            var = np.exp(log_var) + 1e-8
            kl = _gaussian_kl(mean, var, old_mean, old_var)
            return float(self.max_kl - kl)  # must be >= 0

        self._solve_optimization(
            objective, kl_constraint, old_mean, old_var
        )

    # ------------------------------------------------------------------
    #  Stage 2: minimize KL to target under performance constraint
    # ------------------------------------------------------------------

    def _optimize_stage2(self, contexts, old_mean, old_var):
        """min_{mean,log_var}  KL(N(mean,var) || N(target_mean,target_var))
           s.t.  E[R(c)] >= perf_lb
        """

        def objective(params):
            mean, log_var = params[0], params[1]
            var = np.exp(log_var) + 1e-8
            return float(_gaussian_kl(mean, var, self.target_mean, self.target_var))

        def perf_constraint(params):
            mean, log_var = params[0], params[1]
            var = np.exp(log_var) + 1e-8
            weights = np.array([
                _gaussian_pdf(c, mean, var) / max(_gaussian_pdf(c, old_mean, old_var), 1e-12)
                for c in contexts
            ], dtype=np.float64)
            weights = np.clip(weights, 0.0, 10.0)
            expected_perf = np.mean(weights * self.recent_performance())
            return float(expected_perf - self.perf_lb)  # must be >= 0

        self._solve_optimization(
            objective, perf_constraint, old_mean, old_var, is_stage2=True
        )

    # ------------------------------------------------------------------
    #  Shared solver
    # ------------------------------------------------------------------

    def _solve_optimization(self, objective, constraint, old_mean, old_var,
                            is_stage2=False):
        if _scipy_minimize is None:
            self._simple_gradient_step(old_mean, old_var, is_stage2)
            return

        constraints = [{"type": "ineq", "fun": constraint}]
        bounds = [
            (self.lower, self.upper),
            (np.log(1e-6), np.log(1.0)),
        ]
        x0 = [old_mean, np.log(max(old_var, 1e-6))]

        try:
            result = _scipy_minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 80, "ftol": 1e-8},
            )
            if result.success:
                self.mean = float(np.clip(result.x[0], self.lower, self.upper))
                self.var = float(max(np.exp(result.x[1]), 1e-8))
            else:
                self._simple_gradient_step(old_mean, old_var, is_stage2)
        except Exception:
            self._simple_gradient_step(old_mean, old_var, is_stage2)

    def _simple_gradient_step(self, old_mean, old_var, is_stage2=False):
        """Fallback: one step towards LP-weighted mean, respecting KL bound."""
        contexts = self.recent_contexts()
        lps = self.recent_lps()

        if is_stage2:
            # Move towards target mean
            target_dir = self.target_mean - old_mean
            step_size = np.clip(abs(target_dir), 0.001, 0.08)
            new_mean = old_mean + np.sign(target_dir) * step_size
        else:
            # Move towards LP-weighted mean of contexts
            lps_shifted = lps - np.min(lps) + 1e-6
            weights = lps_shifted / (np.sum(lps_shifted) + 1e-8)
            weighted_mean = np.sum(weights * contexts)
            direction = weighted_mean - old_mean
            step_size = np.clip(abs(direction), 0.001, 0.08)
            new_mean = old_mean + np.sign(direction) * step_size

        # KL constraint: (Δμ)^2 / (2*var) <= max_kl
        max_delta = np.sqrt(2.0 * self.max_kl * max(old_var, 1e-6))
        delta = np.clip(new_mean - old_mean, -max_delta, max_delta)
        new_mean = old_mean + delta

        # Towards target: reduce variance
        if is_stage2:
            new_var = old_var * 0.9 + self.target_var * 0.1
        else:
            new_var = old_var

        new_mean = float(np.clip(new_mean, self.lower, self.upper))
        new_var = float(max(new_var, 1e-8))

        self.mean = new_mean
        self.var = new_var

    # ==================================================================
    #  Diagnostics
    # ==================================================================

    def get_state(self):
        return {
            "mean": self.mean,
            "var": self.var,
            "std": np.sqrt(max(self.var, 1e-8)),
            "stage": 2 if self.is_stage2 else 1,
            "recent_perf": self.recent_performance(),
            "recent_lp_mean": float(np.mean(self.recent_lps())) if self.lp_history else 0.0,
        }

    def __repr__(self):
        s = self.get_state()
        return (
            f"SPMARL(mu={s['mean']:.3f}, sigma={s['std']:.3f}, "
            f"stage={s['stage']}, perf={s['recent_perf']:.3f})"
        )
