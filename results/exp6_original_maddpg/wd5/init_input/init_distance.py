from config.config import settings
import numpy as np


def get_distance_between_two_nodes(x1, y1, x2, y2):
    distance = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
    return distance


def get_fixed_distance_matrix(fixed_edge_node, task_list):
    edge_node_length = len(fixed_edge_node)
    task_length = len(task_list)
    fixed_distance_matrix = np.zeros((edge_node_length, task_length))
    for i in range(edge_node_length):
        x1 = fixed_edge_node[i]["x"]
        y1 = fixed_edge_node[i]["y"]
        for j in range(task_length):
            x2 = task_list[j]["x"]
            y2 = task_list[j]["y"]
            fixed_distance_matrix[i][j] = get_distance_between_two_nodes(x1, y1, x2, y2)
    return fixed_distance_matrix


def get_mobile_distance_matrix(time, edge_vehicle_node, task_list):
    edge_vehicle_length = len(edge_vehicle_node)
    task_length = len(task_list)
    time_no = time - settings.experiment_start_time
    mobile_distance_matrix = np.zeros((edge_vehicle_length, task_length))
    for i in range(edge_vehicle_length):
        x1 = edge_vehicle_node[i]["x_list"][time_no]
        y1 = edge_vehicle_node[i]["y_list"][time_no]
        for j in range(task_length):
            x2 = task_list[j]["x"]
            y2 = task_list[j]["y"]
            mobile_distance_matrix[i][j] = get_distance_between_two_nodes(x1, y1, x2, y2)
    return mobile_distance_matrix


_TASK_ID_LISTS = {
    10: [[4, 5, 6, 7, 8, 9], [0, 1, 2, 3]],
    15: [[4, 5, 6, 7, 9, 10], [0, 1, 2, 3], [8, 11, 12, 13, 14]],
    20: [[4, 5, 6, 7, 9, 10], [0, 1, 2, 3], [8, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
    25: [[0, 1, 2, 3, 4, 5], [6, 7, 8, 18], [10, 11, 14, 15, 16, 17], [9, 12, 13, 24, 19], [20, 21, 22, 23]],
}

_TASK_TIME_LISTS = {
    10: [[4, 5, 6, 7, 8, 9], [0, 1, 2, 3]],
    15: [[4, 5, 6, 7, 9, 10], [0, 1, 2, 3], [8, 11, 12, 13, 14]],
    20: [[4, 5, 6, 7, 9, 10], [0, 1, 2, 3], [8, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
    25: [[0, 1, 2, 3, 4, 5], [6, 7, 8], [10, 11, 14, 15, 16, 17, 18, 19], [9, 12, 13, 24], [20, 21, 22, 23]],
}


def get_task_id_under_edge_node(node_type, node_id, distance_matrix=None, variant=None):
    if variant is not None:
        task_id_list_all = _TASK_ID_LISTS[variant]
    else:
        task_id_list_all = [
            [26, 25, 27], [4, 0, 2, 5, 3, 1], [15, 18, 16, 19, 17],
            [21, 20, 24, 23, 22], [13, 9, 14, 7, 8, 6], [10, 12, 11],
        ]
    return task_id_list_all[node_id]


def get_task_time_limitation_under_edge_node(iteration, node_type, node_id, distance_matrix_list, task_list, variant=None):
    if variant is not None:
        task_id_list_all = _TASK_TIME_LISTS[variant]
    else:
        task_id_list_all = [
            [26, 25, 27], [4, 0, 2, 5, 3, 1], [15, 18, 16, 19, 17],
            [21, 20, 24, 23, 22], [13, 9, 14, 7, 8, 6], [10, 12, 11],
        ]
    task_id_list = task_id_list_all[node_id]
    task_time_limitation = np.zeros(len(task_id_list))
    for i in range(len(task_id_list)):
        task_id = task_id_list[i]
        task = task_list[task_id]
        task_time_limitation[i] = task["deadline"]
    return task_time_limitation
