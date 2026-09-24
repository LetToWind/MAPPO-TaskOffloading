from config.config import settings
import random
from init_input.init_vehicles import get_vehicle_location, get_edge_vehicle_id


def generate_random_sub_channel(sub_channel_num, need_sub_channel):
    sub_channel = random.sample(range(sub_channel_num), need_sub_channel)
    return sub_channel


def get_sub_channel_transmission_power(transmission_power_max, channel_num):
    sub_channel_transmission_power = transmission_power_max / channel_num
    return sub_channel_transmission_power


def _make_rsu_node(idx, x, y, channel_num, sub_channel_num):
    return {
        "id": idx,
        "x": x,
        "y": y,
        "radius": settings.rsu_communication_radius,
        "channel_num": channel_num,
        "sub_channel": generate_random_sub_channel(
            sub_channel_num=sub_channel_num,
            need_sub_channel=channel_num,
        ),
        "channel_power": get_sub_channel_transmission_power(
            transmission_power_max=250,
            channel_num=channel_num,
        ),
    }


def init_fixed_edge_node(variant=None):
    fixed_edge_node = []
    if variant is None:
        for i in range(settings.base_station_num):
            base_station = {
                "id": i,
                "x": settings.base_station_x[i],
                "y": settings.base_station_y[i],
                "radius": 1000,
                "channel_num": settings.base_station_sub_channel_num,
                "sub_channel": generate_random_sub_channel(
                    sub_channel_num=settings.sub_channel_num,
                    need_sub_channel=settings.base_station_sub_channel_num,
                ),
                "channel_power": get_sub_channel_transmission_power(
                    transmission_power_max=settings.base_station_transmission_power_max,
                    channel_num=settings.base_station_sub_channel_num,
                ),
            }
            fixed_edge_node.append(base_station)
        for j in range(settings.rsu_num):
            rsu = {
                "id": settings.base_station_num + j,
                "x": settings.rsu_x[j],
                "y": settings.rsu_y[j],
                "radius": settings.rsu_communication_radius,
                "channel_num": settings.rsu_sub_channel_num,
                "sub_channel": generate_random_sub_channel(
                    sub_channel_num=settings.sub_channel_num,
                    need_sub_channel=settings.rsu_sub_channel_num,
                ),
                "channel_power": get_sub_channel_transmission_power(
                    transmission_power_max=settings.rsu_transmission_power_max,
                    channel_num=settings.rsu_sub_channel_num,
                ),
            }
            fixed_edge_node.append(rsu)
    else:
        cfg = settings.VARIANT_CONFIG[variant]
        for j in range(cfg["rsu_num"]):
            rsu = _make_rsu_node(j, cfg["rsu_x"][j], cfg["rsu_y"][j], cfg["channel_num"], cfg["sub_channel_num"])
            fixed_edge_node.append(rsu)
    return fixed_edge_node


def init_edge_vehicle_node(edge_vehicle_num, edge_vehicle_id):
    edge_vehicles_node = []
    for i in range(edge_vehicle_num):
        x_list = []
        y_list = []
        for time in range(settings.experiment_start_time, settings.experiment_start_time + settings.experiment_last_time):
            vehicle_location = get_vehicle_location(vehicle_id=edge_vehicle_id[i], time=time)
            x_list.append(vehicle_location[0])
            y_list.append(vehicle_location[1])
        edge_vehicle = {
            "vehicle_id": edge_vehicle_id,
            "x_list": x_list,
            "y_list": y_list,
            "radius": settings.edge_vehicle_communication_radius,
            "channel_num": settings.edge_vehicle_sub_channel_num,
            "sub_channel": generate_random_sub_channel(
                sub_channel_num=settings.sub_channel_num,
                need_sub_channel=settings.edge_vehicle_sub_channel_num,
            ),
            "channel_power": get_sub_channel_transmission_power(
                transmission_power_max=settings.edge_vehicle_transmission_power_max,
                channel_num=settings.edge_vehicle_sub_channel_num,
            ),
        }
        edge_vehicles_node.append(edge_vehicle)
    return edge_vehicles_node
