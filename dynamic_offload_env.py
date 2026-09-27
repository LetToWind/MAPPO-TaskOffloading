"""Dynamic task-arrival offloading environment (设计文档 v1.0 implementation).

5 BS agents, hotspot-wandering Poisson arrivals, one-shot routing
(local / neighbor+1 transit step), per-BS servable queues, channel+power
scheduling with the original uplink physics (path loss 3.7, gain 64,
noise 1e-8, bandwidth 5e4, SINR>2 gate, rate log2(1+SINR), 200ms steps).

Episode: T steps (default 30), arrivals stop at arrival_end (default 25).
Reward: +5 success, -5 timeout, throughput term, energy term,
holding cost eps_h per alive task per step (dense congestion signal).

Obs layout (47 dims, see doc section 3):
  [0:4]   candidates 2 x (data_norm, rem_norm)      <- routing head
  [4:8]   own queue summary
  [8:28]  own queue detail 10 x (data_norm, rem_norm) <- channel/power heads
  [28:40] neighbors 4 x (n/10, data_norm, min_rem)   <- DELAY TARGET
  [40:46] interference 6 x log-normalized            <- real-time (v1)
  [46]    t/T

Actions:
  route   (n_bs, 2)   in {0 local, 1..4 neighbor, 5 no-op}
  channel (n_bs, n_ch) in {0 idle, 1..10 queue slot}
  power   (n_bs, 10)  in {0..4} power level index
"""

import numpy as np

BS_POSITIONS = np.array([[1590.0, 1490.0], [896.0, 1500.0], [1760.0, 764.0],
                         [882.0, 830.0], [1188.0, 272.0]])
POWER_LEVELS = (0.0, 100.0, 200.0, 300.0, 400.0)

ALPHA = 3.7
GAIN2 = 64.0
NOISE = 1e-8
BANDWIDTH = 5e4
DT = 0.2
DFLOOR = 500.0

DATA_MIN, DATA_MAX = 1e5, 3e5
DDL_MIN, DDL_MAX = 6, 14
DATA_RANGE = DATA_MAX - DATA_MIN
NEIGH_TOTAL_NORM = 3.0e6
EFFECTIVE_LOAD_NORM = 20.0
NOMINAL_BYTES_PER_CHANNEL = BANDWIDTH * np.log2(1.0 + 3.0) * DT


def _d_floor(dist):
    return dist if dist < DFLOOR else DFLOOR


class DynamicOffloadEnv(object):

    def __init__(self, n_bs=5, n_channels=4, lam0=0.15, lam_hot=0.6,
                 hotspot_dwell=8, n_hotspots=1, lam_near=0.0, near_k=2,
                 T=30, arrival_end=25, max_queue=10, max_candidates=2,
                 holding_cost=0.02, pos_radius=300.0, ddl_min=6, ddl_max=14,
                 data_min=1e5, data_max=3e5, bs_positions=None, seed=None,
                 ch_flip=0.0, lam_mode="hotspot", walk_sigma=0.15,
                 lam_min=0.05, lam_max=1.0, capacity_markov=False,
                 capacity_stay=0.70, unrouted_weight=0.5,
                 hotspot_capacity_penalty=0,
                 randomize_neighbor_order=False):
        self.ch_flip = float(ch_flip)
        self.lam_mode = lam_mode
        self.walk_sigma = float(walk_sigma)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.capacity_markov = bool(capacity_markov)
        self.capacity_stay = float(capacity_stay)
        self.unrouted_weight = float(unrouted_weight)
        self.hotspot_capacity_penalty = int(hotspot_capacity_penalty)
        self.randomize_neighbor_order = bool(randomize_neighbor_order)
        self.n_hotspots = max(1, min(int(n_hotspots), n_bs - 1))
        self.lam_near = float(lam_near)
        self.near_k = int(near_k)
        self.ddl_min = int(ddl_min)
        self.ddl_max = int(ddl_max)
        self.data_min = float(data_min)
        self.data_max = float(data_max)
        self.n_bs = n_bs
        self.n_ch = int(n_channels)
        self.lam0 = lam0
        self.lam_hot = lam_hot
        self.dwell = int(hotspot_dwell)
        self.T = int(T)
        self.arrival_end = int(arrival_end)
        self.max_queue = int(max_queue)
        self.K = int(max_candidates)
        self.eps_h = float(holding_cost)
        self.pos_radius = pos_radius
        self.bs_pos = np.asarray(bs_positions, dtype=np.float64) \
            if bs_positions is not None else BS_POSITIONS[:n_bs].copy()
        # geometric nearest-BS adjacency (for the hotspot spill gradient)
        D = np.zeros((n_bs, n_bs))
        for a in range(n_bs):
            for b in range(n_bs):
                D[a, b] = np.linalg.norm(self.bs_pos[a] - self.bs_pos[b])
        self._near = []
        for a in range(n_bs):
            order = np.argsort(D[a])
            self._near.append([int(x) for x in order[1:1 + self.near_k]])
        self.rng = np.random.RandomState(seed) if seed is not None \
            else np.random.RandomState()
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.t = 0
        self._next_id = 0
        self.tasks = {}                      # id -> task dict
        self.unrouted = [[] for _ in range(self.n_bs)]
        self.queue = [[] for _ in range(self.n_bs)]        # servable, FIFO
        self.in_transit = []                                 # (task_id, dest, arrival_step)
        self.hotspots = list(self.rng.choice(
            self.n_bs, size=self.n_hotspots, replace=False))
        self._neighbor_order = []
        for i in range(self.n_bs):
            order = [j for j in range(self.n_bs) if j != i]
            if self.randomize_neighbor_order:
                self.rng.shuffle(order)
            self._neighbor_order.append(order)
        self.hotspot_timers = [self.rng.randint(self.dwell // 2, self.dwell + 1)
                               for _ in range(self.n_hotspots)]
        self.interf = np.full((self.n_bs, 6), NOISE)        # last measured
        self.avail = np.ones((self.n_bs, self.n_ch), dtype=np.float32)
        if self.capacity_markov:
            self.service_cap = self.rng.randint(
                1, self.n_ch + 1, size=self.n_bs).astype(np.int32)
        else:
            self.service_cap = np.full(self.n_bs, self.n_ch, dtype=np.int32)
        self.rates = np.full(self.n_bs, self.lam0)
        self.diag = {"route_events": [], "loads": [], "forwards": 0,
                     "fail_fwd": 0, "fail_local": 0}
        self.ep_arrived = 0
        return self

    # ------------------------------------------------------------------
    def _new_task(self, bs):
        ang = self.rng.rand() * 2 * np.pi
        rad = np.sqrt(self.rng.rand()) * self.pos_radius
        self._next_id += 1
        tid = self._next_id
        self.tasks[tid] = {
            "id": tid,
            "data": self.rng.uniform(self.data_min, self.data_max),
            "rem": self.rng.randint(self.ddl_min, self.ddl_max + 1),
            "origin": bs,
            "loc": bs,
            "pos": self.bs_pos[bs] + rad * np.array([np.cos(ang), np.sin(ang)]),
            "routed": False, "forwarded": False,
        }
        self.ep_arrived += 1
        return tid

    def _step_arrivals(self):
        if self.lam_mode == "static_hetero":
            # BS 0 permanently overloaded, others permanently light:
            # the user's "5/3/7" world, static forever
            for j in range(self.n_bs):
                lam = self.lam0 + (self.lam_hot if j == 0 else 0.0)
                n = self.rng.poisson(lam)
                for _ in range(n):
                    self.unrouted[j].append(self._new_task(j))
            return
        if self.lam_mode == "walk":
            # per-BS arrival rates drift on a bounded random walk: the load
            # profile is graded AND fluid, so who-is-busy changes over time
            self.rates = np.clip(
                self.rates + self.rng.normal(0.0, self.walk_sigma, self.n_bs),
                self.lam_min, self.lam_max)
            for j in range(self.n_bs):
                n = self.rng.poisson(self.rates[j])
                for _ in range(n):
                    self.unrouted[j].append(self._new_task(j))
            return
        for h in range(self.n_hotspots):
            if self.hotspot_timers[h] <= 0:
                choices = [j for j in range(self.n_bs)
                           if j not in self.hotspots]
                self.hotspots[h] = choices[self.rng.randint(len(choices))]
                self.hotspot_timers[h] = self.rng.randint(self.dwell // 2,
                                                          self.dwell + 1)
            self.hotspot_timers[h] -= 1
        lam = np.full(self.n_bs, self.lam0)
        for h in self.hotspots:
            lam[h] += self.lam_hot
            for nb in self._near[h]:
                lam[nb] += self.lam_near
        for j in range(self.n_bs):
            n = self.rng.poisson(lam[j])
            for _ in range(n):
                self.unrouted[j].append(self._new_task(j))

    # ------------------------------------------------------------------
    def alive_counts(self):
        return [len(self.queue[j]) + len(self.unrouted[j]) +
                sum(1 for _, d, _ in self.in_transit if d == j)
                for j in range(self.n_bs)]

    def reserved_data(self, j):
        """Bytes already routed to j but not yet admitted to its queue."""
        return sum(self.tasks[tid]["data"] for tid, dest, _ in self.in_transit
                   if dest == j and tid in self.tasks)

    def effective_workload(self, j, extra_data=0.0):
        """Estimated drain time in steps for work currently committed to BS j."""
        queued = sum(self.tasks[tid]["data"] for tid in self.queue[j]
                     if tid in self.tasks)
        reserved = self.reserved_data(j)
        unrouted = sum(self.tasks[tid]["data"] for tid in self.unrouted[j]
                       if tid in self.tasks)
        work = queued + reserved + self.unrouted_weight * unrouted + extra_data
        rate = max(float(self.effective_caps()[j]) *
                   NOMINAL_BYTES_PER_CHANNEL, 1.0)
        return work / rate

    def effective_loads(self):
        return [self.effective_workload(j) for j in range(self.n_bs)]

    def effective_caps(self):
        caps = self.service_cap.copy()
        if self.hotspot_capacity_penalty > 0:
            for j in self.hotspots:
                caps[j] = max(1, caps[j] - self.hotspot_capacity_penalty)
        return caps

    def _min_remaining(self, j):
        ids = list(self.queue[j]) + list(self.unrouted[j])
        ids += [tid for tid, dest, _ in self.in_transit if dest == j]
        rem = [self.tasks[tid]["rem"] for tid in ids if tid in self.tasks]
        return min(rem) if rem else 0

    def _step_service_capacity(self):
        if not self.capacity_markov:
            return
        move_p = max((1.0 - self.capacity_stay) / 2.0, 0.0)
        draws = self.rng.random_sample(self.n_bs)
        for j in range(self.n_bs):
            if draws[j] < move_p:
                self.service_cap[j] = max(1, self.service_cap[j] - 1)
            elif draws[j] > 1.0 - move_p:
                self.service_cap[j] = min(self.n_ch, self.service_cap[j] + 1)

    def build_obs(self, i):
        obs = np.zeros(47, dtype=np.float32)
        dmin = self.data_min
        drange = max(self.data_max - self.data_min, 1.0)
        dmax = float(self.ddl_max)
        cands = self.unrouted[i][:self.K]
        for k, tid in enumerate(cands[:2]):
            tk = self.tasks[tid]
            obs[2 * k] = np.clip((tk["data"] - dmin) / drange, 0.0, 1.0)
            obs[2 * k + 1] = tk["rem"] / dmax
        q = self.queue[i]
        obs[4] = np.clip(self.effective_workload(i) / EFFECTIVE_LOAD_NORM, 0, 1)
        obs[5] = np.clip(self.reserved_data(i) / NEIGH_TOTAL_NORM, 0, 1)
        obs[6] = self._min_remaining(i) / dmax
        caps = self.effective_caps()
        obs[7] = caps[i] / float(max(self.n_ch, 1))
        for k in range(min(len(q), self.max_queue)):
            tk = self.tasks[q[k]]
            obs[8 + 2 * k] = np.clip((tk["data"] - dmin) / drange, 0.0, 1.0)
            obs[9 + 2 * k] = tk["rem"] / dmax
        base = 28
        for jj, j in enumerate(self.neighbor_indices(i)):
            obs[base + 3 * jj] = np.clip(
                self.effective_workload(j) / EFFECTIVE_LOAD_NORM, 0, 1)
            obs[base + 3 * jj + 1] = self._min_remaining(j) / dmax
            obs[base + 3 * jj + 2] = (
                caps[j] / float(max(self.n_ch, 1)))
        obs[40:46] = np.clip((np.log10(np.maximum(self.interf[i][:6], NOISE)) + 8.0) / 6.0, 0, 1)
        obs[46] = self.t / float(self.T)
        return obs

    def neighbor_indices(self, i):
        return list(self._neighbor_order[i])

    # ------------------------------------------------------------------
    def get_masks(self, route_mask_mode=""):
        nq = len(self.queue) and [len(q) for q in self.queue]
        caps = self.effective_caps()
        route = np.zeros((self.n_bs, self.K, 6), dtype=np.float32)
        for j in range(self.n_bs):
            for k in range(self.K):
                if k < len(self.unrouted[j]):
                    if route_mask_mode == "local":
                        route[j, k, 0] = 1.0
                    else:
                        route[j, k, :5] = 1.0
                else:
                    route[j, k, 5] = 1.0
        channel = np.zeros((self.n_bs, self.n_ch, self.max_queue + 1), dtype=np.float32)
        power = np.zeros((self.n_bs, self.max_queue, len(POWER_LEVELS)), dtype=np.float32)
        for j in range(self.n_bs):
            n_valid = min(nq[j], self.max_queue)
            for c in range(self.n_ch):
                channel[j, c, 0] = 1.0
                if c < caps[j] and self.avail[j, c] >= 0.5:
                    channel[j, c, 1:1 + n_valid] = 1.0
            for k in range(n_valid):
                power[j, k, :] = 1.0
            for k in range(n_valid, self.max_queue):
                power[j, k, 0] = 1.0
        return route, channel, power

    # ------------------------------------------------------------------
    def step(self, route_choices, channel_choices, power_choices):
        r = 0.0
        success = fail = 0
        # ---- per-channel availability flips (exogenous Markov process)
        if self.ch_flip > 0:
            flips = self.rng.random((self.n_bs, self.n_ch)) < self.ch_flip
            self.avail = np.logical_xor(self.avail, flips).astype(np.float32)
        # ---- in-transit arrivals (join dest queue, servable this step)
        arrived_now = [x for x in self.in_transit if x[2] <= self.t]
        self.in_transit = [x for x in self.in_transit if x[2] > self.t]
        for tid, dest, _ in arrived_now:
            self.tasks[tid]["loc"] = dest
            self.queue[dest].append(tid)

        # ---- transmissions (channel/power scheduling)
        # Physical model (C-RAN style): the vehicle's uplink is ALWAYS
        # received at its ORIGIN BS (radio stays local); routing transfers
        # the servicing responsibility (queue + spectrum grant) to the
        # destination BS, and the 1-step transit is the backhaul cost.
        # Interference is measured at each receiver (origin BS) per channel.
        alloc = [[] for _ in range(self.n_bs)]        # bs -> [(task, channel)]
        powers = {}
        caps = self.effective_caps()
        for j in range(self.n_bs):
            q = self.queue[j]
            for c in range(self.n_ch):
                if c >= caps[j] or self.avail[j, c] < 0.5:
                    continue
                slot = int(channel_choices[j][c]) if c < len(channel_choices[j]) else 0
                if 0 < slot <= len(q):
                    tid = q[slot - 1]
                    alloc[j].append((tid, c))
                    if tid not in powers:
                        lvl = int(power_choices[j][slot - 1]) \
                            if slot - 1 < len(power_choices[j]) else 0
                        powers[tid] = POWER_LEVELS[min(max(lvl, 0), len(POWER_LEVELS) - 1)]
        # energy at receiver BS j on channel c: from all transmitters on c
        # whose receiver is not j (own signals are desired, not interference)
        interf = np.zeros((self.n_bs, self.n_ch))
        for j2 in range(self.n_bs):
            for tid, c in alloc[j2]:
                if powers.get(tid, 0.0) <= 0:
                    continue
                for j in range(self.n_bs):
                    if self.tasks[tid]["origin"] == j:
                        continue
                    d = np.linalg.norm(self.tasks[tid]["pos"] - self.bs_pos[j])
                    interf[j, c] += powers[tid] * GAIN2 * _d_floor(d) ** (-ALPHA)
        # rates: each task's signal strength uses its ORIGIN BS distance
        last_data = {tid: self.tasks[tid]["data"] for tid in self.tasks}
        for j in range(self.n_bs):
            by_task = {}
            for tid, c in alloc[j]:
                by_task.setdefault(tid, []).append(c)
            for tid, chans in by_task.items():
                P = powers.get(tid, 0.0)
                if P <= 0:
                    continue
                o = self.tasks[tid]["origin"]
                d_self = np.linalg.norm(self.tasks[tid]["pos"] - self.bs_pos[o])
                S = P * GAIN2 * _d_floor(d_self) ** (-ALPHA)
                for c in chans:
                    sinr = S / (interf[o, c] + NOISE)
                    if sinr > 2:
                        self.tasks[tid]["data"] -= BANDWIDTH * np.log2(1 + sinr) * DT
                r -= P * 200.0 * len(chans) / 5e5
        interf6 = np.full((self.n_bs, 6), NOISE)
        interf6[:, :self.n_ch] = interf + NOISE
        self.interf = np.minimum(interf6, 1e-2)
        # throughput reward
        delta = sum(last_data[tid] - self.tasks[tid]["data"] for tid in last_data)
        r += delta / 5e5

        # ---- completions
        for j in range(self.n_bs):
            done_ids = [tid for tid in self.queue[j] if self.tasks[tid]["data"] <= 0]
            for tid in done_ids:
                self.queue[j].remove(tid)
                del self.tasks[tid]
                r += 5.0
                success += 1

        # ---- routing decisions (FIFO candidates, up to K)
        for j in range(self.n_bs):
            others = self.neighbor_indices(j)
            for k in range(min(self.K, len(self.unrouted[j]))):
                opt = int(route_choices[j][k]) if k < len(route_choices[j]) else 5
                tid = self.unrouted[j][k]
                ev = {"agent": j, "opt": opt,
                      "own_q": len(self.queue[j]),
                      "rem": self.tasks[tid]["rem"], "t": self.t}
                if 1 <= opt <= len(others):
                    dest = others[opt - 1]
                    self.in_transit.append((tid, dest, self.t + 2))
                    self.tasks[tid]["forwarded"] = True
                    ev["dest_true_q"] = len(self.queue[dest])
                    self.diag["forwards"] += 1
                else:
                    self.queue[j].append(tid)          # local, servable next step
                self.tasks[tid]["routed"] = True
                self.diag["route_events"].append(ev)
            self.unrouted[j] = self.unrouted[j][self.K:]

        # ---- deadlines
        for tid in list(self.tasks.keys()):
            self.tasks[tid]["rem"] -= 1
            if self.tasks[tid]["rem"] <= 0 and self.tasks[tid]["data"] > 0:
                if self.tasks[tid]["forwarded"]:
                    self.diag["fail_fwd"] += 1
                else:
                    self.diag["fail_local"] += 1
                fail += 1
                r -= 5.0
                for j in range(self.n_bs):
                    if tid in self.queue[j]:
                        self.queue[j].remove(tid)
                    if tid in self.unrouted[j]:
                        self.unrouted[j].remove(tid)
                self.in_transit = [x for x in self.in_transit if x[0] != tid]
                del self.tasks[tid]

        # ---- holding cost on all alive tasks
        r -= self.eps_h * len(self.tasks)

        # ---- arrivals for next step + hotspot drift
        if self.t < self.arrival_end:
            self._step_arrivals()
        self._step_service_capacity()
        self.diag["loads"].append(self.effective_loads())
        self.t += 1
        info = {"success": success, "fail": fail, "reward": r,
                "arrived": self.ep_arrived, "alive": len(self.tasks),
                "service_cap": self.effective_caps()}
        return r, info

    @property
    def done(self):
        return self.t >= self.T
