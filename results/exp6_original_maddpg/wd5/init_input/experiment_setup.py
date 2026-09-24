import numpy as np
import pandas as pd

from config.config import settings
from init_input.init_edge_node import init_fixed_edge_node
from init_input.init_vehicles import get_vehicle_id, get_customer_vehicle_id, get_edge_vehicle_id
from init_input.init_task_by_time import init_task_by_time
from init_input.init_distance import (
    get_fixed_distance_matrix,
    get_task_id_under_edge_node,
    get_task_time_limitation_under_edge_node,
)


def build_fixed_env_config(variant):
    """Build a sim_dict for the fixed scenario from a variant number (10/15/20/25)."""
    cfg = settings.VARIANT_CONFIG[variant]

    edge_node_list = init_fixed_edge_node(variant=variant)
    vehicle_ids = get_vehicle_id(variant=variant)
    edge_ids = get_edge_vehicle_id()
    customer_ids = get_customer_vehicle_id(edge_ids, vehicle_ids)
    task_list = init_task_by_time(customer_ids, variant=variant)

    distance_matrix = get_fixed_distance_matrix(edge_node_list, task_list)
    n_tasks = len(task_list)

    task_id_under_each_node = []
    task_time_limitation_of_all_nodes = []
    usable_channel_of_all_nodes = []

    for node_id, node in enumerate(edge_node_list):
        task_ids = get_task_id_under_edge_node(
            node_type=settings.NODE_TYPE_RSU if variant else "BaseStation",
            node_id=node_id,
            variant=variant,
        )
        task_id_under_each_node.append(task_ids)

        time_limit = get_task_time_limitation_under_edge_node(
            iteration=0,
            node_type=settings.NODE_TYPE_RSU,
            node_id=node_id,
            distance_matrix_list=distance_matrix,
            task_list=task_list,
            variant=variant,
        )
        task_time_limitation_of_all_nodes.append(time_limit)

        usable_channel_of_all_nodes.append(
            list(range(cfg["channel_num"]))
        )

    task_data_size = np.asarray([t["data_size"] for t in task_list], dtype=np.float64)
    task_deadline = np.asarray([t["deadline"] for t in task_list], dtype=np.float64)

    task_on_base_matrix = np.zeros((len(edge_node_list), n_tasks))
    for node_id, task_ids in enumerate(task_id_under_each_node):
        for tid in task_ids:
            task_on_base_matrix[node_id][tid] = 1

    task_to_base = np.zeros(n_tasks, dtype=np.int32)
    for node_id, task_ids in enumerate(task_id_under_each_node):
        for tid in task_ids:
            task_to_base[tid] = node_id

    return {
        "n_veh": 5,
        "n_neighbor": 1,
        "sub_channel_bandwidth": 5e4,
        "MBS_transmission_power_max": 100,
        "RSU_transmission_power_max": 20,
        "current_busy": 1e-8,
        "MBS_gain": 8,
        "RSU_gain": 2,
        "dt": 20,
        "N_agents": len(edge_node_list),
        "PATH_LOSS_EXPONENT": 3.7,
        "task_on_base": task_id_under_each_node,
        "task_on_base_matrix": task_on_base_matrix,
        "fixed_distance_matrix": distance_matrix,
        "usable_channel_of_all_nodes": usable_channel_of_all_nodes,
        "task_deadline": task_deadline,
        "task_data_size": task_data_size,
    }
