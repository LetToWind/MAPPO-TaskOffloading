import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _hungarian
except ImportError:
    _hungarian = None


DEFAULT_AREA_SIZE = (2000.0, 2000.0)
DEFAULT_TASK_DATA_SIZE_RANGE = (1e5, 5e5)
DEFAULT_TASK_DEADLINE_RANGE = (1000, 2000)


def _as_random_state(seed=None):
    if isinstance(seed, np.random.RandomState):
        return seed
    return np.random.RandomState(seed)


def _pairwise_distance(bs_positions, ue_positions):
    bs = np.asarray(bs_positions, dtype=np.float64)
    ue = np.asarray(ue_positions, dtype=np.float64)
    return np.sqrt(np.sum((bs[:, None, :] - ue[None, :, :]) ** 2, axis=2))


def _sample_bs_positions(rng, n_bs, area_size):
    width, height = area_size
    margin_x = width * 0.15
    margin_y = height * 0.15
    x = rng.uniform(margin_x, width - margin_x, size=n_bs)
    y = rng.uniform(margin_y, height - margin_y, size=n_bs)
    return np.stack([x, y], axis=1)


def _sample_ue_positions(rng, n_tasks, area_size, bs_positions=None, max_ue_distance=None):
    width, height = area_size
    max_retries = 100
    positions = []
    for _ in range(n_tasks):
        for _ in range(max_retries):
            x = rng.uniform(0.0, width)
            y = rng.uniform(0.0, height)
            if bs_positions is None or max_ue_distance is None:
                break
            dists = np.sqrt(np.sum((bs_positions - np.array([x, y])) ** 2, axis=1))
            if np.min(dists) <= max_ue_distance:
                break
        else:
            x = rng.uniform(0.0, width)
            y = rng.uniform(0.0, height)
        positions.append([x, y])
    return np.array(positions)


def _associate_greedy(distance_matrix, max_tasks_per_bs=6):
    """Greedy assignment: each UE takes the nearest BS with remaining capacity."""
    distances = np.asarray(distance_matrix, dtype=np.float64)
    n_bs, n_tasks = distances.shape
    K = int(max_tasks_per_bs)
    capacities = np.full(n_bs, K, dtype=np.int32)
    task_on_base = [[] for _ in range(n_bs)]
    order = np.argsort(distances.T, axis=1)  # [n_tasks, n_bs] sorted nearest first
    for task_id in range(n_tasks):
        for bs_id in order[task_id]:
            if capacities[bs_id] > 0:
                task_on_base[bs_id].append(task_id)
                capacities[bs_id] -= 1
                break
    return task_on_base


def associate_ues_to_bs(distance_matrix, max_tasks_per_bs=6):
    """Assign each UE to one BS via Hungarian algorithm (scipy) or greedy fallback.

    Returns
    -------
    task_on_base : list[list[int]]
        Outer list indexed by BS id; each inner list contains the task ids
        assigned to that BS.
    """
    distances = np.asarray(distance_matrix, dtype=np.float64)
    n_bs, n_tasks = distances.shape
    K = int(max_tasks_per_bs)

    if _hungarian is not None:
        n_slots = n_bs * K
        cost = np.zeros((n_slots, n_slots), dtype=np.float64)
        for bs in range(n_bs):
            for slot in range(K):
                row = bs * K + slot
                cost[row, :n_tasks] = distances[bs]
        row_ind, col_ind = _hungarian(cost)
        task_on_base = [[] for _ in range(n_bs)]
        for row, col in zip(row_ind, col_ind):
            if col < n_tasks:
                task_on_base[row // K].append(int(col))
        return task_on_base

    return _associate_greedy(distance_matrix, max_tasks_per_bs=K)


def generate_dynamic_topology(
    n_bs=5,
    n_tasks=25,
    area_size=DEFAULT_AREA_SIZE,
    n_channels=6,
    channel_per_bs=6,
    max_tasks_per_bs=6,
    max_ue_distance=None,
    task_data_size_range=DEFAULT_TASK_DATA_SIZE_RANGE,
    task_deadline_range=DEFAULT_TASK_DEADLINE_RANGE,
    seed=None,
    bs_positions=None,
    usable_channel_of_all_nodes=None,
    ue_positions=None,
):
    """Generate a random topology and optimal UE-BS association.

    BSs and UEs are sampled uniformly across the area (full coverage).
    Association is solved via Hungarian algorithm or greedy fallback.

    If ``bs_positions`` is provided, BS placement and channel assignment are
    reused (only UEs and association are re-generated per call).
    """
    rng = _as_random_state(seed)

    if bs_positions is None:
        bs_positions = _sample_bs_positions(rng, n_bs, area_size)
    if ue_positions is not None:
        ue_positions = np.asarray(ue_positions, dtype=np.float64)
        n_tasks = len(ue_positions)
    else:
        ue_positions = _sample_ue_positions(rng, n_tasks, area_size, bs_positions, max_ue_distance)
    fixed_distance_matrix = _pairwise_distance(bs_positions, ue_positions)

    task_on_base = associate_ues_to_bs(fixed_distance_matrix, max_tasks_per_bs=max_tasks_per_bs)

    task_on_base_matrix = np.zeros((n_bs, n_tasks), dtype=np.float32)
    for bs_id, task_ids in enumerate(task_on_base):
        for task_id in task_ids:
            task_on_base_matrix[bs_id, task_id] = 1.0

    if usable_channel_of_all_nodes is None:
        usable_channel_of_all_nodes = []
        for _bs_id in range(n_bs):
            channels = np.arange(n_channels)
            rng.shuffle(channels)
            usable_channel_of_all_nodes.append(channels[:channel_per_bs].astype(int).tolist())

    task_data_size = rng.randint(
        int(task_data_size_range[0]),
        int(task_data_size_range[1]) + 1,
        size=n_tasks,
    ).astype(float).tolist()
    task_deadline = rng.randint(
        int(task_deadline_range[0]),
        int(task_deadline_range[1]) + 1,
        size=n_tasks,
    ).astype(float).tolist()

    return {
        "task_on_base": task_on_base,
        "task_on_base_matrix": task_on_base_matrix,
        "fixed_distance_matrix": fixed_distance_matrix,
        "usable_channel_of_all_nodes": usable_channel_of_all_nodes,
        "task_deadline": task_deadline,
        "task_data_size": task_data_size,
        "bs_positions": bs_positions,
        "ue_positions": ue_positions,
    }


def generate_fixed_bs_config(n_bs, n_channels, channel_per_bs, area_size, seed):
    """Pre-generate BS positions and channel assignments (fixed for an experiment run)."""
    rng = _as_random_state(seed)
    bs_positions = _sample_bs_positions(rng, n_bs, area_size)
    usable_channels = []
    for _ in range(n_bs):
        channels = np.arange(n_channels)
        rng.shuffle(channels)
        usable_channels.append(channels[:channel_per_bs].astype(int).tolist())
    return bs_positions, usable_channels


def build_dynamic_env_config(**kwargs):
    topology = generate_dynamic_topology(**kwargs)
    return {
        "n_veh": int(kwargs.get("n_bs", 5)),
        "n_neighbor": 1,
        "sub_channel_bandwidth": 5e4,
        "MBS_transmission_power_max": 100,
        "RSU_transmission_power_max": 20,
        "current_busy": 1e-8,
        "MBS_gain": 8,
        "RSU_gain": 2,
        "dt": 20,
        "N_agents": int(kwargs.get("n_bs", 5)),
        "PATH_LOSS_EXPONENT": 3.7,
        "task_on_base": topology["task_on_base"],
        "task_on_base_matrix": topology["task_on_base_matrix"],
        "fixed_distance_matrix": topology["fixed_distance_matrix"],
        "usable_channel_of_all_nodes": topology["usable_channel_of_all_nodes"],
        "task_deadline": topology["task_deadline"],
        "task_data_size": topology["task_data_size"],
        "topology": topology,
    }