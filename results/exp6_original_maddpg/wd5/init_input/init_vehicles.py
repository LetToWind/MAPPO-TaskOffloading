from config.config import settings
import random
import pandas as pd


def get_vehicle_trace():
    df = pd.read_csv(settings.fill_xy_csv_name)
    start_df = df[(df["time"] >= settings.experiment_start_time)]
    experiment_df = start_df[(start_df["time"] <= settings.experiment_start_time + settings.experiment_last_time - 1)]
    return experiment_df


def get_vehicle_id(variant=None):
    csv_path = settings.FILE_XY_DEL
    if variant is not None and variant in settings.VARIANT_CONFIG:
        csv_path = settings.VARIANT_CONFIG[variant]["csv"]
    df = pd.read_csv(csv_path)
    vehicle_id_list = df["id"].drop_duplicates().tolist()
    return vehicle_id_list


def get_edge_vehicle_id():
    time_id_set = []
    for i in range(settings.experiment_start_time, settings.experiment_start_time + settings.experiment_last_time):
        time_df = get_vehicle_trace_in_time(i)
        time_id = set(time_df["id"].drop_duplicates().tolist())
        time_id_set.append(time_id)
    intersection_id_set = time_id_set[0]
    for id_set in time_id_set:
        intersection_id_set = intersection_id_set & id_set
    id = list(intersection_id_set)
    for i in id:
        vehicle_location = get_vehicle_location(i, settings.experiment_start_time)
        if vehicle_location[0] < 500 or vehicle_location[0] > 2500:
            id.remove(i)
        else:
            if vehicle_location[1] < 500 or vehicle_location[1] > 2500:
                id.remove(i)
    if len(id) >= settings.EDGE_VEHICLE_NUM:
        edge_vehicle_id = random.sample(id, settings.EDGE_VEHICLE_NUM)
        return edge_vehicle_id
    else:
        raise ValueError("from init_vehicles.get_edge_vehicle_id() 满足条件的边缘节点车辆数量少于需求数量")


def get_customer_vehicle_id(edge_vehicle_id, id):
    customer_vehicles_id = []
    for vehicle_id in id:
        if vehicle_id in edge_vehicle_id:
            pass
        else:
            customer_vehicles_id.append(vehicle_id)
    return customer_vehicles_id


def get_vehicle_location(vehicle_id, time):
    df = get_vehicle_trace()
    vehicle_df = df[(df["id"] == vehicle_id)]
    if time in vehicle_df["time"].tolist():
        time_df = vehicle_df[(vehicle_df["time"] == time)]
        x = float(time_df["x"])
        y = float(time_df["y"])
        return x, y
    else:
        return


def get_vehicle_trace_in_time(time):
    df = get_vehicle_trace()
    time_df = df[(df["time"] == time)]
    return time_df


if __name__ == '__main__':
    print("*" * 32)
    print("Vehicle Trace")
    print(get_vehicle_trace())
    df = get_vehicle_trace()
    id = get_vehicle_id()
    print("*" * 32)
    print("Vehicle ID")
    print(id)
    print(type(id))
    edge_id = get_edge_vehicle_id()
    customer_id = get_customer_vehicle_id(edge_id, id)
    print("*" * 32)
    print("Edge Vehicle ID")
    print(edge_id)
    print(type(edge_id))
    print("*" * 32)
    print("Customer Vehicle ID")
    print(customer_id)
    print(type(customer_id))
