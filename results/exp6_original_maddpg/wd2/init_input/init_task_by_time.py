from config.config import settings
import random
import pandas as pd


def get_random_task_data_size():
    return random.randint(settings.task_data_size_min, settings.task_data_size_max)


def get_random_task_deadline():
    return random.randint(settings.TASK_DEADLINE_MIN, settings.TASK_DEADLINE_MAX)


def init_task_by_time(customer_vehicle_id, variant=None):
    csv_path = settings.FILE_XY_DEL
    if variant is not None and variant in settings.VARIANT_CONFIG:
        csv_path = settings.VARIANT_CONFIG[variant]["csv"]
    df = pd.read_csv(csv_path)
    vehicle_id_x = df["x"].tolist()
    vehicle_id_y = df["y"].tolist()
    task_list = []
    for id in customer_vehicle_id:
        task = {
            "task_id": id,
            "x": vehicle_id_x[id - 1],
            "y": vehicle_id_y[id - 1],
            "data_size": get_random_task_data_size(),
            "deadline": get_random_task_deadline(),
        }
        task_list.append(task)
    return task_list
