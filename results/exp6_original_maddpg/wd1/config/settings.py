FILL_XY_CSV_NAME = "../data/fill_xy.csv"

"""##################################
#  experiment  settings
##################################"""

"""##################################
#  edge node  settings
##################################"""
BASE_STATION_NUM = 1
RSU_NUM = 5
SUB_CHANNEL_NUM = 10

BASE_STATION_X = [1000]
BASE_STATION_Y = [1000]

RSU_X = [1590, 1379, 1062, 547, 1760]
RSU_Y = [1490, 1225, 1530, 820, 764]

VARIANT_CONFIG = {
    10: {
        "rsu_num": 2,
        "rsu_x": [1700, 1000],
        "rsu_y": [1000, 1500],
        "csv": "../data/fill_xy_10.csv",
        "channel_num": 10,
        "sub_channel_num": 10,
        "task_id_list": [[4, 5, 6, 7, 8, 9], [0, 1, 2, 3]],
        "task_time_list": [[4, 5, 6, 7, 8, 9], [0, 1, 2, 3]],
    },
    15: {
        "rsu_num": 3,
        "rsu_x": [1700, 1000, 1096],
        "rsu_y": [1000, 1500, 533],
        "csv": "../data/fill_xy_15.csv",
        "channel_num": 10,
        "sub_channel_num": 10,
        "task_id_list": [[4, 5, 6, 7, 9, 10], [0, 1, 2, 3], [8, 11, 12, 13, 14]],
        "task_time_list": [[4, 5, 6, 7, 9, 10], [0, 1, 2, 3], [8, 11, 12, 13, 14]],
    },
    20: {
        "rsu_num": 4,
        "rsu_x": [1700, 1000, 1096, 1774],
        "rsu_y": [1000, 1500, 533, 2098],
        "csv": "../data/fill_xy_20.csv",
        "channel_num": 10,
        "sub_channel_num": 10,
        "task_id_list": [[4, 5, 6, 7, 9, 10], [0, 1, 2, 3], [8, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
        "task_time_list": [[4, 5, 6, 7, 9, 10], [0, 1, 2, 3], [8, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
    },
    25: {
        "rsu_num": 5,
        "rsu_x": [1590, 896, 1760, 882, 1188],
        "rsu_y": [1490, 1500, 764, 830, 272],
        "csv": "../data/test_data.csv",
        "channel_num": 6,
        "sub_channel_num": 6,
        "task_id_list": [[0, 1, 2, 3, 4, 5], [6, 7, 8, 18], [10, 11, 14, 15, 16, 17], [9, 12, 13, 24, 19], [20, 21, 22, 23]],
        "task_time_list": [[0, 1, 2, 3, 4, 5], [6, 7, 8], [10, 11, 14, 15, 16, 17, 18, 19], [9, 12, 13, 24], [20, 21, 22, 23]],
    },
}

RSU_COMMUNICATION_RADIUS = 500
EDGE_VEHICLE_COMMUNICATION_RADIUS = 300

BASE_STATION_TRANSMISSION_POWER_MAX = 400
RSU_TRANSMISSION_POWER_MAX = 400
EDGE_VEHICLE_TRANSMISSION_POWER_MAX = 1

BASE_STATION_SUB_CHANNEL_NUM = 10
RSU_SUB_CHANNEL_NUM = 10
EDGE_VEHICLE_SUB_CHANNEL_NUM = 10

SUB_CHANNEL_BANDWIDTH = 5e4

"""##################################
#  vehicular transmission task settings
##################################"""
TASK_DATA_SIZE_MIN = 1e5
TASK_DATA_SIZE_MAX = 5e5

TASK_DEADLINE_MIN = 1000
TASK_DEADLINE_MAX = 2000

"""##################################
#  wireless communication parameters value settings
##################################"""
CHANNEL_FADING_GAIN_EX = 8
CHANNEL_FADING_GAIN_DX = 0.4

ANTENNA_CONSTANT = 1

PATH_LOSS_EXPONENT = 3.6

WHITE_GAUSSIAN_NOISE = 1e-8

"""##################################
#  algorithm parameters value settings
##################################"""
LEARNING_RATE = 10

"""##################################
#  training hyperparameters
##################################"""
N_EPISODES = 18000
N_STEPS = 2000           # buffer max capacity (episode-based, no truncation)
K_EPOCHS = 4
MINIBATCH_SIZE = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
FIXED_CURRICULUM = 0      # 1=课程学习, 0=固定拓扑
USE_DYNAMIC_TOPOLOGY = 0   # 1=动态拓扑, 0=固定拓扑(variant 25)
UE_FIXED_EPISODES = 6000
UE_RANDOM_RAMP = 6000
MAX_UE_DISTANCE = 400  # 0 = no constraint

"""##################################
#  SPMARL curriculum hyperparameters
#  (Learning Progress Driven Multi-Agent Curriculum, ICML 2025)
##################################"""
USE_SPMARL = 1                # 1=自适应课程, 0=使用原有 FIXED_CURRICULUM
SPMARL_INIT_MEAN = 0.15       # context 初始均值 (0=最简单)
SPMARL_INIT_VAR = 0.05        # context 初始方差
SPMARL_TARGET_MEAN = 1.0      # 目标 context (1=最难, 无距离约束)
SPMARL_TARGET_VAR = 0.004     # 目标方差
SPMARL_MAX_KL = 0.05          # KL 散度约束
SPMARL_PERF_LB = 0.55         # Stage2 触发阈值 (成功率的 performance lower bound)
SPMARL_UPDATE_INTERVAL = 20   # 每 N 个 episode 更新一次 context 分布
SPMARL_CONTEXT_LOWER = 0.0    # context 下界
SPMARL_CONTEXT_UPPER = 1.0    # context 上界
SPMARL_STD_LOWER_BOUND = 0.02 # sigma 最小值
SPMARL_MAX_UE_EASY = 300.0    # c=0 时的 MAX_UE_DISTANCE (UE 距 BS 最近约束)
SPMARL_WINDOW_SIZE = 20       # 用于估计 LP 和 perf 的滑动窗口大小
DYNAMIC_N_BS = 5
DYNAMIC_N_TASKS = 25
DYNAMIC_AREA_WIDTH = 2000.0
DYNAMIC_AREA_HEIGHT = 2000.0
BATCH_SIZE = 256
MEMORY_SIZE = 50000

"""##################################
#  experiment parameters value settings
##################################"""
EXPERIMENT_START_TIME = 1
EXPERIMENT_LAST_TIME = 10

EXPERIMENT_FILE_NAME = "../experiment_data/experiment_file_name.txt"
EXPERIMENT_MEDIAN_FILE_NAME = "../experiment_data/experiment_median_file_name.txt"
ITERATION_MEDIAN_FILE_NAME = "../experiment_data/iteration_median_file_name.txt"

NODE_TYPE_BASE_STATION = "BaseStation"
NODE_TYPE_RSU = "RSU"
NODE_TYPE_VEHICLE = "Vehicle"

NODE_TYPE_FIXED = "Fixed_Edge_Node"
NODE_TYPE_MOBILE = "Mobile_Edge_Node"

FILE_XY_DEL = "../data/test_data.csv"
EDGE_VEHICLE_NUM = 3
