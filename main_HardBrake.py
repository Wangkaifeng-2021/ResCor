import os
import sys
import traci
import random
import torch
import numpy as np
from numpy import ndarray
from numpy.typing import NDArray
from sumolib import checkBinary
from Experiment.PPO import create_ppo
from Experiment.PPO_ResCor import create_ppo_rescor
from Experiment.PPO_Lag import create_ppo_lag
import datetime
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import math
from typing import Dict, List, Tuple
from Utils.utils import get_options, execute_commands, compute_target_pose, find_closest_vehicle, get_corners
from Utils.collision_detector import CollisionDetector
import xml.etree.ElementTree as ET
from filelock import FileLock

# ------ Experiment Param Config ------ #
Enable_Graph = True
# Enable_Graph = False

RENDER = True
# RENDER = False

Noise_Level = 0  # do not add noise during training
Noise_Sigma = {
    'x': 10,  # m
    'y': 1,  # m
    'theta': math.degrees(0.1),  # rad
    'v': 2,  # m/s
}

GNN = 'GCN'
# GNN = 'GAT'
# GNN = 'GraphSAGE'

# DRL = 'PPO'
# DRL = 'PPO_Lag'
# DRL = 'PPO_SafetyLayer'
DRL = 'PPO_ResCor'

max_speed = 55.56
num_av = 2
num_hv = 7
num_vehicles = num_hv + num_av
road_length = 150
road_angle = 90
road_speed_limit = 30.0
lane_width = 3.2
num_lane = 3
n_episodes = 300
horizon = 400
step_length = 0.1
sense_range = 50.0
action_space = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),  # accelerate
    2: (2.0, 0.0),
    3: (3.0, 0.0),
    4: (4.0, 0.0),
    5: (5.0, 0.0),
    6: (-1.0, 0.0),  # decelerate
    7: (-2.0, 0.0),
    8: (-3.0, 0.0),
    9: (-4.0, 0.0),
    10: (-5.0, 0.0),
    11: (0.0, 5.0),  # right turn
    12: (0.0, 10.0),
    13: (0.0, 15.0),
    14: (0.0, 20.0),
    15: (0.0, 25.0),
    16: (0.0, 30.0),
    17: (0.0, -5.0),  # left turn
    18: (0.0, -10.0),
    19: (0.0, -15.0),
    20: (0.0, -20.0),
    21: (0.0, -25.0),
    22: (0.0, -30.0),
}
accel = 5.0
decel = -5.0
vehicle_length = 5.0
vehicle_width = 1.8
wheelbase = 3.2
min_gap = 2.5
random_seeds = list(range(1000, 1000+n_episodes))

now_time = datetime.datetime.now()
now_time = datetime.datetime.strftime(now_time, '%Y_%m_%d-%H_%M_%S')
if Enable_Graph:
    save_dir = 'TrainedModels/HardBrake' + '/' + GNN + '_' + DRL + '/' + now_time
else:
    save_dir = 'TrainedModels/HardBrake' + '/' + DRL + '/' + now_time

sim_params = {'n_vehicles': num_vehicles, 'n_hv': num_hv, 'n_av': num_av, 'n_lanes': num_lane,
              'max_speed': max_speed, 'action_dim': 23, 'road_length': road_length, 'lane_width': lane_width,
              'feature_dim': 22, 'Graph': Enable_Graph, 'horizon': horizon, 'step_length': step_length,
              'GNN': GNN, 'DRL': DRL, 'road_speed_limit': road_speed_limit, 'sense_range': sense_range,
              'action_space': action_space, 'road_angle': road_angle, 'accel': accel, 'decel': decel,
              'veh_length': vehicle_length, 'veh_width': vehicle_width, 'wheelbase': wheelbase, 'min_gap': min_gap,
              'Noise_Level': Noise_Level, 'Noise_Sigma': Noise_Sigma}
list_AV = ["AV_0.0", "AV_1.0"]
list_HV = ["HV_0.0", "HV_1.0", "HV_1.1", "HV_1.2", "HV_1.3", "HV_1.4", "HV_1.5"]
vehicles_ids = list_AV + list_HV
sim_params['state_space_av_ids'] = list_AV
sim_params['state_space_ids'] = vehicles_ids


def start_sumo_env(seed: int, is_render: bool, params: Dict) -> None:
    random.seed(seed)
    # random scenario
    random_scenarios = [(0, 10.00, 0, 30.00, 1, 5.00), (1, 10.00, 1, 30.00, 0, 5.00)]
    selected = random.choice(random_scenarios)
    av0_lane, av0_pos, hv0_lane, hv0_pos, av1_lane, av1_pos = selected
    params['hv0_lane'] = hv0_lane

    # random HVs' pos
    positions = []
    lanes = {0: [], 1: [], 2: []}
    num_cars = 6
    while len(positions) < num_cars:
        lane = random.choice([0, 1, 2])
        if lane == hv0_lane:
            position = random.uniform(50, params['road_length']-30)
        else:
            position = random.uniform(30, params['road_length']-30)
        if all(abs(position - pos) >= 20 for pos in lanes[lane]):
            lanes[lane].append(position)
            positions.append((lane, round(position, 2)))

    file_path = 'Env_Net/HardBrake/HardBrake.rou.xml'
    lock_path = file_path + '.lock'
    lock = FileLock(lock_path)
    with lock:
        tree = ET.parse(file_path)
        root = tree.getroot()

        position_index = 0
        for vehicle in root.findall('vehicle'):
            vehicle_id = vehicle.attrib['id']
            if vehicle_id == 'AV_0.0' or vehicle_id == 'AV_1.0' or vehicle_id == 'HV_0.0' or vehicle_id == 'AV_2.0':
                depart_speed = round(random.uniform(0, 10), 2)  # random speed 0-10 m/s
                vehicle.attrib['departSpeed'] = str(depart_speed)
                if vehicle_id == 'AV_0.0':
                    vehicle.attrib['departLane'] = str(av0_lane)
                    vehicle.attrib['departPos'] = str(av0_pos)
                elif vehicle_id == 'HV_0.0':
                    vehicle.attrib['departLane'] = str(hv0_lane)
                    vehicle.attrib['departPos'] = str(hv0_pos)
                elif vehicle_id == 'AV_1.0':
                    vehicle.attrib['departLane'] = str(av1_lane)
                    vehicle.attrib['departPos'] = str(av1_pos)
            if 'HV_1' in vehicle_id:
                if position_index < len(positions):
                    lane, pos = positions[position_index]
                    position_index += 1
                    vehicle.attrib['departLane'] = str(lane)
                    vehicle.attrib['departPos'] = str(pos)
                    depart_speed = round(random.uniform(0, 10), 2)  # random speed 0-10 m/s
                    vehicle.attrib['departSpeed'] = str(depart_speed)
                else:
                    raise ValueError("The number of random locations generated is less than the number of HV_1")

        tree.write(file_path)

        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        options = get_options()
        if not options.nogui and is_render:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')

        time_to_teleport = params['horizon']
        traci.start([sumoBinary, "-c", "Env_Net/HardBrake/HardBrake.sumocfg", "--seed", str(seed),
                     "--time-to-teleport", str(time_to_teleport)])


def remove_arrived_vehicles(params: Dict) -> List[str]:
    arrived_vehs = []
    ids = traci.vehicle.getIDList()
    for ID in ids:
        x, y = traci.vehicle.getPosition(vehID=ID)
        if x >= params['road_length']:
            arrived_vehs.append(ID)
            traci.vehicle.remove(vehID=ID)

    return arrived_vehs


def get_valid_actions_mask(observation: Tuple[ndarray, ndarray, ndarray], environment_info: Dict, time_step: int, params: Dict):
    """ Constrain the action space """
    n_vehicles = params['n_vehicles']
    n_av = params['n_av']
    road_angle = params['road_angle']
    lane_width = params['lane_width']
    state_space_ids = params['state_space_ids']
    action_dim = params['action_dim']
    action_space_dict = params['action_space']
    target_lane_boundary = {0: {'upper': -2, 'lower': -3},
                            1: {'upper': -1, 'lower': -2},
                            2: {'upper': 0, 'lower': -1}}
    t_h = 1.2

    actions_mask = torch.ones(n_vehicles, action_dim, dtype=torch.float32)
    node_feature, adjacency, mask = observation
    avs_info = environment_info['avs_info']
    for i in range(n_av):
        if mask[i] == 1:
            veh_id = state_space_ids[i]
            front_bumper_x = node_feature[i][0]
            front_bumper_y = node_feature[i][1]
            angle = node_feature[i][2]
            speed = node_feature[i][3]

            is_forbid_lc = False
            last_lc_step = params['AVs_lc'][veh_id]['step']
            target_lane = params['AVs_lc'][veh_id]['target_lane']
            if last_lc_step > 0 and time_step <= last_lc_step + 50:
                is_forbid_lc = True
                upper = target_lane_boundary[target_lane]['upper']
                lower = target_lane_boundary[target_lane]['lower']

            decel_s = (0 - speed ** 2) / (2 * params['decel'])
            decel_x = front_bumper_x + decel_s * math.sin(math.radians(angle))
            decel_y = front_bumper_y + decel_s * math.cos(math.radians(angle))

            decel_corners = get_corners(decel_x, decel_y, angle, params)
            decel_lf_x, decel_lf_y = decel_corners[0]
            decel_rf_x, decel_rf_y = decel_corners[1]
            if decel_lf_y >= 0:
                actions_mask[i][:11] = torch.tensor(0.0, dtype=torch.float32)
                actions_mask[i][17:] = torch.tensor(0.0, dtype=torch.float32)
            if decel_rf_y <= (-3 * lane_width):
                actions_mask[i][:17] = torch.tensor(0.0, dtype=torch.float32)

            if is_forbid_lc:
                if angle < road_angle and decel_lf_y >= (upper * lane_width):
                    actions_mask[i][:11] = torch.tensor(0.0, dtype=torch.float32)
                    actions_mask[i][17:] = torch.tensor(0.0, dtype=torch.float32)
                if angle > road_angle and decel_rf_y <= (lower * lane_width):
                    actions_mask[i][:17] = torch.tensor(0.0, dtype=torch.float32)

            if avs_info:
                ttc = avs_info[veh_id]['observe_ttc']
                if ttc <= t_h:
                    actions_mask[i][:6] = torch.tensor(0.0, dtype=torch.float32)

            for j in range(action_dim):
                if actions_mask[i][j]:
                    acc, steer = action_space_dict[j]
                    target_front_bumper_x, target_front_bumper_y, target_angle, target_speed = compute_target_pose(
                        vehicle_id=veh_id,
                        acceleration=acc,
                        steering=steer,
                        params=params)
                    # 1. Constrain actions beyond the roadway boundary
                    target_corners = get_corners(target_front_bumper_x, target_front_bumper_y, target_angle, params)
                    target_lf_x, target_lf_y = target_corners[0]
                    target_rf_x, target_rf_y = target_corners[1]
                    if target_lf_y >= 0 or target_rf_y <= (-3 * lane_width):
                        actions_mask[i][j] = torch.tensor(0.0, dtype=torch.float32)
                        continue
                    # Prohibition of continuous lane changing
                    if is_forbid_lc:
                        if (target_angle < road_angle and target_lf_y >= (upper * lane_width)) or \
                                (target_angle > road_angle and target_rf_y <= (lower * lane_width)):
                            actions_mask[i][j] = torch.tensor(0.0, dtype=torch.float32)
                            continue
                    # 2. Constrain the reverse action
                    if acc != 0.0:
                        if target_speed < 0:
                            actions_mask[i][j] = torch.tensor(0.0, dtype=torch.float32)
                            continue
                    # 3. Constrain the U-turn action
                    angle_diff_threshold = 30
                    if steer != 0.0:
                        if target_angle < (road_angle - angle_diff_threshold) \
                                or target_angle > (road_angle + angle_diff_threshold):
                            actions_mask[i][j] = torch.tensor(0.0, dtype=torch.float32)
                            continue
                    # 4. Constrain actions to steer towards the road boundary
                    if front_bumper_y <= -2.5 * lane_width:
                        if steer > 0:
                            actions_mask[i][j] = torch.tensor(0.0, dtype=torch.float32)
                            continue
                    elif front_bumper_y >= -0.5 * lane_width:
                        if steer < 0:
                            actions_mask[i][j] = torch.tensor(0.0, dtype=torch.float32)
                            continue

            if torch.sum(actions_mask[i]) == 0:
                if front_bumper_y > -1 * lane_width:
                    actions_mask[i][16] = torch.tensor(1.0, dtype=torch.float32)
                elif front_bumper_y < -2 * lane_width:
                    actions_mask[i][22] = torch.tensor(1.0, dtype=torch.float32)

            if torch.sum(actions_mask[i]) == 0 and is_forbid_lc:
                if angle < road_angle:
                    if front_bumper_y < (upper + lower) / 2.0 * lane_width:
                        actions_mask[i][:11] = torch.tensor(1.0, dtype=torch.float32)
                    else:
                        actions_mask[i][16] = torch.tensor(1.0, dtype=torch.float32)
                elif angle > road_angle:
                    if front_bumper_y > (upper + lower) / 2.0 * lane_width:
                        actions_mask[i][:11] = torch.tensor(1.0, dtype=torch.float32)
                    else:
                        actions_mask[i][22] = torch.tensor(1.0, dtype=torch.float32)

            if torch.sum(actions_mask[i]) == 0:
                raise ValueError(f"The action space is fully constrained!")

    return actions_mask


def get_neighbors(vehicle_id: str, params: Dict) -> List:
    _sense_range = params['sense_range']
    gap = params['min_gap']
    noise_level = params['Noise_Level']
    noise_sigma = params['Noise_Sigma']

    f_rel_x = lf_rel_x = rf_rel_x = _sense_range
    r_rel_x = lr_rel_x = rr_rel_x = -_sense_range
    f_rel_v = r_rel_v = lf_rel_v = lr_rel_v = rf_rel_v = rr_rel_v = 0.0

    ego_v = traci.vehicle.getSpeed(vehID=vehicle_id)

    # 1. F
    f_veh = traci.vehicle.getLeader(vehID=vehicle_id, dist=300.0)
    if f_veh:
        f_id, f_dis = f_veh
        f_real_dis = f_dis + gap
        if f_real_dis <= _sense_range:
            x_noise = noise_level * np.random.normal(0, noise_sigma['x'])
            v_noise = noise_level * np.random.normal(0, noise_sigma['v'])
            f_rel_x = min(f_real_dis + x_noise, _sense_range)
            f_v = traci.vehicle.getSpeed(vehID=f_id) + v_noise
            f_rel_v = f_v - ego_v

    # 2. R
    r_veh = traci.vehicle.getFollower(vehID=vehicle_id, dist=300.0)
    if r_veh[0]:
        r_id, r_dis = r_veh
        r_real_dis = r_dis + gap
        if r_real_dis <= _sense_range:
            x_noise = noise_level * np.random.normal(0, noise_sigma['x'])
            v_noise = noise_level * np.random.normal(0, noise_sigma['v'])
            r_rel_x = -min(r_real_dis + x_noise, _sense_range)
            r_v = traci.vehicle.getSpeed(vehID=r_id) + v_noise
            r_rel_v = r_v - ego_v

    # 3. LF
    lf_vehs = traci.vehicle.getLeftLeaders(vehID=vehicle_id)
    if lf_vehs:
        lf_id, lf_dis = find_closest_vehicle(lf_vehs)
        lf_real_dis = lf_dis + gap
        if lf_real_dis <= _sense_range:
            x_noise = noise_level * np.random.normal(0, noise_sigma['x'])
            v_noise = noise_level * np.random.normal(0, noise_sigma['v'])
            lf_rel_x = min(lf_real_dis + x_noise, _sense_range)
            lf_v = traci.vehicle.getSpeed(vehID=lf_id) + v_noise
            lf_rel_v = lf_v - ego_v

    # 4. LR
    lr_vehs = traci.vehicle.getLeftFollowers(vehID=vehicle_id)
    if lr_vehs:
        lr_id, lr_dis = find_closest_vehicle(lr_vehs)
        lr_real_dis = lr_dis + gap
        if lr_real_dis <= _sense_range:
            x_noise = noise_level * np.random.normal(0, noise_sigma['x'])
            v_noise = noise_level * np.random.normal(0, noise_sigma['v'])
            lr_rel_x = -min(lr_real_dis + x_noise, _sense_range)
            lr_v = traci.vehicle.getSpeed(vehID=lr_id) + v_noise
            lr_rel_v = lr_v - ego_v

    # 5. RF
    rf_vehs = traci.vehicle.getRightLeaders(vehID=vehicle_id)
    if rf_vehs:
        rf_id, rf_dis = find_closest_vehicle(rf_vehs)
        rf_real_dis = rf_dis + gap
        if rf_real_dis <= _sense_range:
            x_noise = noise_level * np.random.normal(0, noise_sigma['x'])
            v_noise = noise_level * np.random.normal(0, noise_sigma['v'])
            rf_rel_x = min(rf_real_dis + x_noise, _sense_range)
            rf_v = traci.vehicle.getSpeed(vehID=rf_id) + v_noise
            rf_rel_v = rf_v - ego_v

    # 6. RR
    rr_vehs = traci.vehicle.getRightFollowers(vehID=vehicle_id)
    if rr_vehs:
        rr_id, rr_dis = find_closest_vehicle(rr_vehs)
        rr_real_dis = rr_dis + gap
        if rr_real_dis <= _sense_range:
            x_noise = noise_level * np.random.normal(0, noise_sigma['x'])
            v_noise = noise_level * np.random.normal(0, noise_sigma['v'])
            rr_rel_x = -min(rr_real_dis + x_noise, _sense_range)
            rr_v = traci.vehicle.getSpeed(vehID=rr_id) + v_noise
            rr_rel_v = rr_v - ego_v

    neighbors = [f_rel_x, f_rel_v,
                 r_rel_x, r_rel_v,
                 lf_rel_x, lf_rel_v,
                 lr_rel_x, lr_rel_v,
                 rf_rel_x, rf_rel_v,
                 rr_rel_x, rr_rel_v]

    return neighbors


def calculate_avoid_heading(x0: float, y0: float, x1: float, y1: float, params: Dict) -> float:
    """
    Calculate the heading angle required for avoidance

    :param x0: rear center x
    :param y0: rear center y
    :param x1: front LR x
    :param y1: front LR y
    :param params: parameters
    :return: heading
    """
    half_width = params['veh_width'] / 2.0

    A = y0 - y1
    B = x1 - x0
    C = -half_width

    phi = math.atan2(B, A)
    sin_value = -C / math.sqrt(A**2 + B**2)

    if abs(sin_value) > 1:
        raise ValueError(f"No valid solution! x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}")

    theta_1_rad = math.asin(sin_value) - phi
    theta_2_rad = (math.pi - math.asin(sin_value)) - phi

    theta_1_deg = math.degrees(theta_1_rad) % 360
    theta_2_deg = math.degrees(theta_2_rad) % 360

    if theta_1_deg < 180 and theta_2_deg < 180:
        raise ValueError(f"There are 2 solutions: {theta_1_deg}, {theta_2_deg}, x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}")
    if theta_1_deg >= 180 and theta_2_deg >= 180:
        raise ValueError(f"No valid solution! x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}")

    if theta_1_deg < 180:
        return theta_1_deg
    if theta_2_deg < 180:
        return theta_2_deg


def take_actions(rl_actions: NDArray[np.int32], params: Dict, preheat: bool, time_step: int) -> Dict:
    state_space_ids = params['state_space_ids']
    gap = params['min_gap']
    noise_level = params['Noise_Level']
    noise_sigma = params['Noise_Sigma']
    action_space_dict = params['action_space']
    avs_info = {}

    # 1. Take actions
    take_action_ids = traci.vehicle.getIDList()

    last_step_types = []
    for ID in take_action_ids:
        last_step_types.append(traci.vehicle.getTypeID(vehID=ID))
    last_step_has_av = any(elem == 'AV' for elem in last_step_types)

    if preheat:
        for ID in take_action_ids:
            execute_commands(vehicle_id=ID, acceleration=0.0, steering=0.0, params=params)
    else:
        for ID in take_action_ids:
            # front veh hard brake
            if ID == 'HV_0.0':
                if time_step <= 30:
                    acc = -10.0  # m/s²
                else:
                    acc = 2.0
                steer = 0
                execute_commands(vehicle_id=ID, acceleration=acc, steering=steer, params=sim_params)

            if traci.vehicle.getTypeID(vehID=ID) == 'AV':
                last_x, last_y = traci.vehicle.getPosition(vehID=ID)
                last_angle = traci.vehicle.getAngle(vehID=ID)
                last_speed = traci.vehicle.getSpeed(vehID=ID)
                last_lane_index = traci.vehicle.getLaneIndex(vehID=ID)
                index = state_space_ids.index(ID)
                _action = rl_actions[index]
                acc, steer = action_space_dict[_action]
                target_speed, target_angle = execute_commands(vehicle_id=ID, acceleration=acc, steering=steer,
                                                              params=params)

                if steer != 0.0:
                    _r = params['wheelbase'] / math.tan(math.radians(abs(steer) * params['step_length']))
                    lat_acc = (target_speed ** 2) / _r
                else:
                    lat_acc = 0.0
                avs_info[ID] = {'acc': acc, 'steer': steer, 'speed': target_speed, 'angle': target_angle,
                                'last_x': last_x, 'last_y': last_y, 'last_angle': last_angle,
                                'last_speed': last_speed, 'last_lane_index': last_lane_index, 'lat_acc': lat_acc}

                last_leader_id = None
                last_leader = traci.vehicle.getLeader(vehID=ID, dist=300.0)
                if last_leader:
                    last_leader_id = last_leader[0]
                    last_leader_dis = last_leader[1]
                    last_real_leader_dis = last_leader_dis + gap
                    avs_info[ID]['last_leader_dis'] = min(last_real_leader_dis, params['sense_range'])
                    last_leader_x, last_leader_y = traci.vehicle.getPosition(vehID=last_leader_id)
                    avs_info[ID]['last_leader_x'] = last_leader_x
                    avs_info[ID]['last_leader_y'] = last_leader_y
                    avs_info[ID]['last_leader_speed'] = traci.vehicle.getSpeed(vehID=last_leader_id)
                    avs_info[ID]['last_leader_angle'] = traci.vehicle.getAngle(vehID=last_leader_id)
                avs_info[ID]['last_leader_id'] = last_leader_id
    traci.simulationStep()

    # Remove vehicles arriving at the target point
    arrived_vehs_by_move = remove_arrived_vehicles(params)
    arrived_vehs_by_sumo = list(traci.simulation.getArrivedIDList())
    arrived_vehs_all = arrived_vehs_by_move + arrived_vehs_by_sumo

    # 2. Get information for collision detection and reward calculation
    vehs_info = {}
    for ID in take_action_ids:
        if ID not in arrived_vehs_all:
            x, y = traci.vehicle.getPosition(vehID=ID)
            angle = traci.vehicle.getAngle(vehID=ID)
            length = traci.vehicle.getLength(typeID=ID)
            width = traci.vehicle.getWidth(typeID=ID)
            _type = traci.vehicle.getTypeID(vehID=ID)
            vehs_info[ID] = {'id': ID, 'x': x, 'y': y, 'angle': angle, 'length': length, 'width': width, 'type': _type}
        if ID[:2] == 'AV':
            leader_id = None
            real_leader_dis = -1
            ttc = float('inf')
            observe_ttc = float('inf')
            if ID not in arrived_vehs_all:
                avs_info[ID]['lane_index'] = traci.vehicle.getLaneIndex(vehID=ID)
                avs_info[ID]['lat_lane_pos'] = traci.vehicle.getLateralLanePosition(vehID=ID)
                leader = traci.vehicle.getLeader(vehID=ID, dist=300.0)
                if leader:
                    leader_id = leader[0]
                    leader_dis = leader[1]
                    real_leader_dis = leader_dis + gap
                    leader_x, leader_y = traci.vehicle.getPosition(vehID=leader_id)
                    avs_info[ID]['leader_x'] = leader_x
                    avs_info[ID]['leader_y'] = leader_y
                    avs_info[ID]['leader_speed'] = traci.vehicle.getSpeed(vehID=leader_id)
                    avs_info[ID]['leader_acc'] = traci.vehicle.getAcceleration(vehID=leader_id)
                    avs_info[ID]['leader_angle'] = traci.vehicle.getAngle(vehID=leader_id)
                    relative_speed = avs_info[ID]['speed'] * math.sin(math.radians(avs_info[ID]['angle'])) - \
                                     avs_info[ID]['leader_speed'] * math.sin(math.radians(avs_info[ID]['leader_angle']))
                    if relative_speed > 0:
                        ttc = max(real_leader_dis / relative_speed, 0.0)
                        x_noise = noise_level * np.random.normal(0, noise_sigma['x'])
                        v_noise = noise_level * np.random.normal(0, noise_sigma['v'])
                        theta_noise = noise_level * np.random.normal(0, noise_sigma['theta'])
                        real_leader_dis_noise = real_leader_dis + x_noise
                        relative_speed_noise = avs_info[ID]['speed'] * math.sin(math.radians(avs_info[ID]['angle'])) - \
                                               (avs_info[ID]['leader_speed'] + v_noise) * math.sin(math.radians(avs_info[ID]['leader_angle'] + theta_noise))
                        observe_ttc = max(real_leader_dis_noise / relative_speed_noise, 0.0)
                # Calibration SUMO Detection of front vehicle
                if ID == 'AV_1.0':
                    if 'AV_0.0' in take_action_ids and 'AV_0.0' not in arrived_vehs_all and leader_id != 'AV_0.0':
                        av0_x, av0_y = traci.vehicle.getPosition(vehID='AV_0.0')
                        av1_x, av1_y = traci.vehicle.getPosition(vehID='AV_1.0')
                        lf_x, lf_y = get_corners(av0_x, av0_y, traci.vehicle.getAngle(vehID='AV_0.0'), params)[0]
                        av0_lane_index = traci.vehicle.getLaneIndex(vehID='AV_0.0')
                        av1_lane_index = traci.vehicle.getLaneIndex(vehID=ID)
                        if av1_lane_index == 1 and av0_lane_index == 0 and av0_x > av1_x:
                            if lf_y > -2 * lane_width:
                                leader_id = 'AV_0.0'
                                real_leader_dis = av0_x - params['veh_length'] - av1_x
                                avs_info[ID]['leader_x'] = av0_x
                                avs_info[ID]['leader_y'] = av0_y
                                avs_info[ID]['leader_speed'] = traci.vehicle.getSpeed(vehID=leader_id)
                                avs_info[ID]['leader_acc'] = traci.vehicle.getAcceleration(vehID=leader_id)
                                avs_info[ID]['leader_angle'] = traci.vehicle.getAngle(vehID=leader_id)
                                relative_speed = avs_info[ID]['speed'] * math.sin(math.radians(avs_info[ID]['angle'])) - \
                                                 avs_info[ID]['leader_speed'] * math.sin(math.radians(avs_info[ID]['leader_angle']))
                                ttc = float('inf')
                                observe_ttc = float('inf')
                                if relative_speed > 0:
                                    ttc = max(real_leader_dis / relative_speed, 0.0)
                                    x_noise = noise_level * np.random.normal(0, noise_sigma['x'])
                                    v_noise = noise_level * np.random.normal(0, noise_sigma['v'])
                                    theta_noise = noise_level * np.random.normal(0, noise_sigma['theta'])
                                    real_leader_dis_noise = real_leader_dis + x_noise
                                    relative_speed_noise = avs_info[ID]['speed'] * math.sin(math.radians(avs_info[ID]['angle'])) - \
                                                           (avs_info[ID]['leader_speed'] + v_noise) * math.sin(math.radians(avs_info[ID]['leader_angle'] + theta_noise))
                                    observe_ttc = max(real_leader_dis_noise / relative_speed_noise, 0.0)
            avs_info[ID]['leader_id'] = leader_id
            avs_info[ID]['leader_dis'] = min(real_leader_dis, params['sense_range'])
            avs_info[ID]['ttc'] = ttc
            avs_info[ID]['observe_ttc'] = observe_ttc

    # 3. Detect collisions and remove colliding vehicles
    detector = CollisionDetector(vehs_info)
    collisions = detector.detect_collisions()
    collision_num = len(collisions)
    col_primary_av_ids = []
    road_angle = params['road_angle']
    for collision in collisions:
        collider_id = collision.collider
        collider_type = collision.colliderType
        victim_id = collision.victim
        victim_type = collision.victimType
        print(f"There is a collision: Rear({collider_id}), Front({victim_id})")
        if collider_type == 'AV' and victim_type == 'HV':
            col_primary_av_ids.append(collider_id)
        elif collider_type == 'HV' and victim_type == 'AV':
            col_primary_av_ids.append(victim_id)
        elif collider_type == 'AV' and victim_type == 'AV':
            collision_num += 1
            is_lc = False
            collider_angle = collision.colliderAngle
            if abs(collider_angle - road_angle) >= 10:
                is_lc = True
                col_primary_av_ids.append(collider_id)
            victim_angle = collision.victimAngle
            if abs(victim_angle - road_angle) >= 10:
                is_lc = True
                col_primary_av_ids.append(victim_id)
            if not is_lc:
                col_primary_av_ids.append(collider_id)
        else:
            raise ValueError(f"Special collision situation: {collision}")
        # Remove colliding vehicles
        now_ids = traci.vehicle.getIDList()
        if collider_id in now_ids:
            traci.vehicle.remove(vehID=collider_id)
        if victim_id in now_ids:
            traci.vehicle.remove(vehID=victim_id)

    # 4. Get env information for graph representation
    ids = traci.vehicle.getIDList()
    x_positions = []
    y_positions = []
    speeds = []
    lane_indexes = []
    angles = []
    types = []
    neighbor_vehs = []
    av_ids = []
    av_speeds = []
    av_lat_accs = []
    av_lon_accs = []
    av_leader_dis = []
    av_actions = []
    for ID in ids:
        speed = traci.vehicle.getSpeed(vehID=ID)
        speeds.append(speed)
        if ID in avs_info:
            av_lat_accs.append(avs_info[ID]['lat_acc'])
            av_lon_accs.append(avs_info[ID]['acc'])
        _type = traci.vehicle.getTypeID(vehID=ID)
        types.append(_type)
        x, y = traci.vehicle.getPosition(vehID=ID)
        x_positions.append(x)
        y_positions.append(y)
        lane_index = traci.vehicle.getLaneIndex(vehID=ID)
        lane_indexes.append(lane_index)
        angles.append(traci.vehicle.getAngle(vehID=ID))
        neighbors = get_neighbors(vehicle_id=ID, params=params)
        neighbor_vehs.append(neighbors)
        if _type == 'AV':
            if ID == 'AV_0.0' and not params['AV00_is_avoided'] and lane_index != params['hv0_lane']:
                params['AV00_is_avoided'] = True
            if not preheat and lane_index != avs_info[ID]['last_lane_index']:
                params['AVs_lc'][ID]['step'] = time_step
                params['AVs_lc'][ID]['target_lane'] = lane_index
            av_ids.append(ID)
            av_speeds.append(speed)
    has_av = any(elem == 'AV' for elem in types)

    avs_done = False
    if (last_step_has_av and not has_av) or time_step == params['horizon']-1:
        avs_done = True

    environment_info = {'ids': ids, 'av_ids': av_ids, 'x_positions': x_positions, 'y_positions': y_positions,
                        'speeds': speeds, 'av_speeds': av_speeds, 'types': types, 'lane_indexes': lane_indexes,
                        'collision_num': collision_num, 'av_leader_dis': av_leader_dis, 'has_av': has_av,
                        'col_primary_av_ids': col_primary_av_ids, 'av_actions': av_actions, 'avs_info': avs_info,
                        'angles': angles, 'neighbor_vehs': neighbor_vehs, 'arrived_vehs_all': arrived_vehs_all,
                        'avs_done': avs_done, 'av_lat_accs': av_lat_accs, 'av_lon_accs': av_lon_accs}

    return environment_info


def graph_representation(environment_info: Dict, params: Dict) -> Tuple[ndarray, ndarray, ndarray]:
    """ Graph representation: node feature matrix, adjacency matrix, avs mask matrix """
    N = params['n_vehicles']
    n_av = params['n_av']
    num_lanes = params['n_lanes']
    ids = environment_info['ids']
    av_ids = environment_info['av_ids']
    x_positions = environment_info['x_positions']
    y_positions = environment_info['y_positions']
    angles = environment_info['angles']
    speeds = environment_info['speeds']
    lane_indexes = environment_info['lane_indexes']
    types = environment_info['types']
    neighbor_vehs = environment_info['neighbor_vehs']
    state_space_ids = params['state_space_ids']
    noise_level = params['Noise_Level']
    noise_sigma = params['Noise_Sigma']

    node_feature = np.zeros([N, params['feature_dim']])
    adjacency = np.zeros([N, N])
    mask = np.zeros(N)

    if ids:
        noisy_x, noisy_y, noisy_theta, noisy_v = [], [], [], []
        for i in range(len(ids)):
            if noise_level != 0 and types[i] != 'AV':
                x_noise = noise_level * np.random.normal(0, noise_sigma['x'])
                y_noise = noise_level * np.random.normal(0, noise_sigma['y'])
                theta_noise = noise_level * np.random.normal(0, noise_sigma['theta'])
                v_noise = noise_level * np.random.normal(0, noise_sigma['v'])

                noisy_x.append(x_positions[i] + x_noise)
                noisy_y.append(y_positions[i] + y_noise)
                noisy_theta.append(angles[i] + theta_noise)
                noisy_v.append(speeds[i] + v_noise)
            else:
                noisy_x.append(x_positions[i])
                noisy_y.append(y_positions[i])
                noisy_theta.append(angles[i])
                noisy_v.append(speeds[i])

        # 1. node feature matrix
        xs = np.array(noisy_x).reshape(-1, 1)
        ys = np.array(noisy_y).reshape(-1, 1)
        thetas = np.array(noisy_theta).reshape(-1, 1)
        vs = np.array(noisy_v).reshape(-1, 1)
        lanes_column = np.array(lane_indexes)
        lanes = np.zeros([len(ids), num_lanes])
        lanes[np.arange(len(ids)), lanes_column] = 1.0
        intention_indexes = [0 if elem == 'AV' else 1 if elem == 'HV' else 2 for elem in types]
        intention_column = np.array(intention_indexes)
        intentions = np.zeros([len(ids), 3])
        intentions[np.arange(len(ids)), intention_column] = 1.0
        neighbors = np.array(neighbor_vehs).reshape(-1, 12)

        observed_states = np.c_[xs, ys, thetas, vs, lanes, intentions, neighbors]

        indexes = [state_space_ids.index(elem) for elem in ids]
        node_feature[indexes, :] = observed_states

        # 2. adjacency matrix
        dist_matrix_x = euclidean_distances(xs)
        dist_matrix_y = euclidean_distances(ys)
        dist_matrix = np.sqrt(dist_matrix_x * dist_matrix_x + dist_matrix_y * dist_matrix_y)
        adjacency_small = np.zeros_like(dist_matrix)
        adjacency_small[dist_matrix <= params['sense_range']] = 1.0
        for cnt, index in enumerate(indexes):
            adjacency[index, indexes] = adjacency_small[cnt, :]
        for index in indexes:
            if index >= n_av:
                if adjacency[index][:n_av].sum() > 0.0:
                    adjacency[index, :] = 0.0
                    adjacency[index, index] = 1.0
                else:
                    node_feature[index, :] = 0.0
                    adjacency[index, :] = 0.0
        av_indexes = [state_space_ids.index(elem) for elem in av_ids]
        for av_index in av_indexes:
            adjacency[av_index, av_indexes] = 1.0

        # 3. avs mask matrix
        mask[av_indexes] = 1.0

    return node_feature, adjacency, mask


def reward_function(environment_info: Dict, params: Dict) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    w_collision = 100.0
    w_velocity = 4.0
    w_distance_keep = 3.0
    w_comfort = 0.5
    w_avoid = 0.0
    w_lat = 0.5

    N = params['n_vehicles']
    state_space_ids = params['state_space_ids']
    col_primary_av_ids = environment_info['col_primary_av_ids']
    avs_info = environment_info['avs_info']

    total_rewards = np.zeros(N)
    collision_rewards = np.zeros(N)
    velocity_rewards = np.zeros(N)
    distance_keep_rewards = np.zeros(N)
    comfort_rewards = np.zeros(N)
    avoid_rewards = np.zeros(N)
    lane_change_rewards = np.zeros(N)

    for ID, info in avs_info.items():
        collision_reward = 0.0
        velocity_reward = 0.0
        distance_keep_reward = 0.0
        comfort_reward = 0.0
        avoid_reward = 0.0

        if ID in col_primary_av_ids:
            # 1. collision
            collision_reward = -1.0
        else:
            # 2. speed
            av_speed = avs_info[ID]['speed']
            acc = avs_info[ID]['acc']
            if acc <= 2.0:
                velocity_reward = acc / 5.0

            # 3. distance keep
            leader_dis = avs_info[ID]['leader_dis']
            if leader_dis > 0:
                distance_keep_reward = -min(1 / leader_dis - 1 / params['sense_range'], 2)

            # 4. comfort
            lon_comfort_reward = 0.0
            lat_comfort_reward = 0.0
            # longitudinal
            steer = avs_info[ID]['steer']
            if steer == 0.0:
                if -2.0 <= acc <= 2.0:
                    lon_comfort_reward = 1.0
                elif acc <= -4.5:
                    lon_comfort_reward = -10.0
            # lateral
            if steer != 0.0:
                r = params['wheelbase'] / math.tan(math.radians(abs(steer) * params['step_length']))
                a_lat = (av_speed ** 2) / r
                if a_lat < 0.5:
                    lat_comfort_reward = 1.0
                elif a_lat < 2.0:
                    lat_comfort_reward = 0.5
                else:
                    lat_comfort_reward = -10.0
            # sum
            comfort_reward = lon_comfort_reward + lat_comfort_reward * w_lat

        weighted_collision_reward = w_collision * collision_reward
        weighted_velocity_reward = w_velocity * velocity_reward
        weighted_distance_keep_reward = w_distance_keep * distance_keep_reward
        weighted_comfort_reward = w_comfort * comfort_reward
        weighted_avoid_reward = w_avoid * avoid_reward
        # total reward
        total_reward = weighted_collision_reward + weighted_velocity_reward + \
                       weighted_distance_keep_reward + weighted_comfort_reward + weighted_avoid_reward

        index = state_space_ids.index(ID)
        total_rewards[index] = total_reward
        collision_rewards[index] = weighted_collision_reward
        velocity_rewards[index] = weighted_velocity_reward
        distance_keep_rewards[index] = weighted_distance_keep_reward
        comfort_rewards[index] = weighted_comfort_reward
        avoid_rewards[index] = weighted_avoid_reward

    return total_rewards, collision_rewards, velocity_rewards, distance_keep_rewards, comfort_rewards, avoid_rewards, lane_change_rewards


def safety_reward_function(environment_info: Dict, params: Dict) -> Tuple[ndarray, ndarray, ndarray]:
    w_collision = 100.0
    w_velocity = 4.0
    w_distance_keep = 3.0
    w_comfort = 0.5
    w_avoid = 20.0
    w_lat = 0.5

    N = params['n_vehicles']
    state_space_ids = params['state_space_ids']
    veh_length = params['veh_length']
    safe_avoid_angle = 0.3
    lambda_param = 10.0
    col_primary_av_ids = environment_info['col_primary_av_ids']
    avs_info = environment_info['avs_info']

    total_rewards = np.zeros(N)
    collision_rewards = np.zeros(N)
    velocity_rewards = np.zeros(N)
    distance_keep_rewards = np.zeros(N)
    comfort_rewards = np.zeros(N)
    avoid_rewards = np.zeros(N)

    for ID, info in avs_info.items():
        collision_reward = 0.0
        velocity_reward = 0.0
        distance_keep_reward = 0.0
        comfort_reward = 0.0
        avoid_reward = 0.0

        if ID in col_primary_av_ids:
            # 1. collision
            collision_reward = -1.0
        else:
            # 2. speed
            av_speed = avs_info[ID]['speed']
            acc = avs_info[ID]['acc']
            if acc <= 2.0:
                velocity_reward = acc / 5.0

            # 3. distance keep
            leader_dis = avs_info[ID]['leader_dis']
            if leader_dis > 0:
                distance_keep_reward = -min(1 / leader_dis - 1 / params['sense_range'], 2)

            # 4. comfort
            lon_comfort_reward = 0.0
            lat_comfort_reward = 0.0
            # longitudinal
            steer = avs_info[ID]['steer']
            if steer == 0.0:
                if -2.0 <= acc <= 2.0:
                    lon_comfort_reward = 1.0
                elif acc <= -4.5:
                    lon_comfort_reward = -10.0
            # lateral
            if steer != 0.0:
                r = params['wheelbase'] / math.tan(math.radians(abs(steer) * params['step_length']))
                a_lat = (av_speed ** 2) / r
                if a_lat < 0.5:
                    lat_comfort_reward = 1.0
                elif a_lat < 2.0:
                    lat_comfort_reward = 0.5
                else:
                    lat_comfort_reward = -10.0
            # sum
            comfort_reward = lon_comfort_reward + lat_comfort_reward * w_lat

            # 5. avoid
            last_leader_id = avs_info[ID]['last_leader_id']
            if ID == 'AV_0.0' and not params['AV00_is_avoided'] and last_leader_id == 'HV_0.0':
                last_x = avs_info[ID]['last_x']
                last_y = avs_info[ID]['last_y']
                last_angle = avs_info[ID]['last_angle']
                last_leader_x = avs_info[ID]['last_leader_x']
                last_leader_y = avs_info[ID]['last_leader_y']
                last_leader_angle = avs_info[ID]['last_leader_angle']

                if last_x < (last_leader_x - veh_length):
                    # Calculate the avoidance angle
                    half_length = veh_length / 2.0
                    last_x0 = last_x - half_length * math.sin(math.radians(last_angle))
                    last_y0 = last_y - half_length * math.cos(math.radians(last_angle))

                    last_leader_corners = get_corners(last_leader_x, last_leader_y, last_leader_angle, params)
                    last_x1, last_y1 = last_leader_corners[3]

                    avoid_angle = calculate_avoid_heading(last_x0, last_y0, last_x1, last_y1, params)
                    avoid_angle -= safe_avoid_angle

                    if steer < 0.0 and last_angle >= avoid_angle:
                        need_steer = last_angle - avoid_angle
                        contribution = min(abs(steer * params['step_length']) / need_steer, 1.0)
                        last_speed = avs_info[ID]['last_speed']
                        last_leader_speed = avs_info[ID]['last_leader_speed']
                        relative_speed = last_speed - last_leader_speed
                        last_leader_dis = avs_info[ID]['last_leader_dis']
                        avoid_reward = math.exp(-last_leader_dis / lambda_param) * relative_speed * contribution
                        avoid_reward = max(-10.0, min(10.0, avoid_reward))
                    if steer > 0.0:
                        avoid_reward = -1.0

        weighted_collision_reward = w_collision * collision_reward
        weighted_velocity_reward = w_velocity * velocity_reward
        weighted_distance_keep_reward = w_distance_keep * distance_keep_reward
        weighted_comfort_reward = w_comfort * comfort_reward
        weighted_avoid_reward = w_avoid * avoid_reward
        # total reward
        total_reward = weighted_collision_reward + weighted_velocity_reward + \
                       weighted_distance_keep_reward + weighted_comfort_reward + weighted_avoid_reward

        index = state_space_ids.index(ID)
        total_rewards[index] = total_reward
        collision_rewards[index] = weighted_collision_reward
        velocity_rewards[index] = weighted_velocity_reward
        distance_keep_rewards[index] = weighted_distance_keep_reward
        comfort_rewards[index] = weighted_comfort_reward
        avoid_rewards[index] = weighted_avoid_reward

    return total_rewards, collision_rewards, avoid_rewards


def cost_function(environment_info: Dict, params: Dict) -> Tuple[ndarray, ndarray, ndarray]:
    """ The cost function for the PPO-Lagrangian algorithm """
    w_collision = 100.0
    w_ttc = 20.0

    N = params['n_vehicles']
    state_space_ids = params['state_space_ids']
    col_primary_av_ids = environment_info['col_primary_av_ids']
    avs_info = environment_info['avs_info']
    ttc_threshold = 1.2

    total_costs = np.zeros(N)
    collision_costs = np.zeros(N)
    ttc_costs = np.zeros(N)

    for ID, info in avs_info.items():
        collision_cost = 0.0
        _ttc_cost = 0.0

        # 1. avoid collision
        if ID in col_primary_av_ids:
            collision_cost = 1.0

        # 2. safe distance
        ttc = avs_info[ID]['ttc']
        if ttc < ttc_threshold:
            _ttc_cost = (1 - ttc / ttc_threshold) ** 2

        weighted_collision_cost = w_collision * collision_cost
        weighted_ttc_cost = w_ttc * _ttc_cost
        # total cost
        total_cost = weighted_collision_cost + weighted_ttc_cost

        index = state_space_ids.index(ID)
        total_costs[index] = total_cost
        collision_costs[index] = weighted_collision_cost
        ttc_costs[index] = weighted_ttc_cost

    return total_costs, collision_costs, ttc_costs


def safety_layer(observation: Tuple[ndarray, ndarray, ndarray], environment_info: Dict,
                 origin_actions: ndarray, params: Dict) -> ndarray:
    """ The safety layer for the PPO-SafetyLayer algorithm  """
    n_av = params['n_av']
    state_space_ids = params['state_space_ids']
    action_space_dict = params['action_space']
    dt = params['step_length']
    noise_level = params['Noise_Level']
    noise_sigma = params['Noise_Sigma']
    safe_action = 10  # set the safe action

    safety_actions = origin_actions.copy()
    node_feature, adjacency, mask = observation
    avs_info = environment_info['avs_info']
    for i in range(n_av):
        if mask[i] == 1 and avs_info:
            ID = state_space_ids[i]
            leader_id = avs_info[ID]['leader_id']
            current_action = origin_actions[i]
            acc, steer = action_space_dict[current_action]
            # longitudinal collision check
            if leader_id and acc != action_space_dict[safe_action][0]:
                real_leader_dis = avs_info[ID]['leader_dis']
                # The safety layer is also not able to observe accurate information.
                # So if there is noise, it needs to be added.
                x_noise = noise_level * np.random.normal(0, noise_sigma['x'])
                v_noise = noise_level * np.random.normal(0, noise_sigma['v'])
                theta_noise = noise_level * np.random.normal(0, noise_sigma['theta'])
                real_leader_dis_noise = real_leader_dis + x_noise
                relative_speed_noise = avs_info[ID]['speed'] * math.sin(math.radians(avs_info[ID]['angle'])) - \
                                       (avs_info[ID]['leader_speed'] + v_noise) * math.sin(math.radians(avs_info[ID]['leader_angle'] + theta_noise))
                if relative_speed_noise > 0:
                    # predict collisions using the constant velocity kinematic model
                    predict_dis = real_leader_dis_noise - relative_speed_noise * dt - 0.5 * acc * dt**2
                    if predict_dis <= 0:
                        safety_actions[i] = safe_action

    return safety_actions


def risk_quantification_function(environment_info: Dict, params: Dict) -> ndarray:
    w_emergency = 0.5
    w_ttc = 1.0

    N = params['n_vehicles']
    state_space_av_ids = params['state_space_av_ids']

    risk_list = np.full(N, 0.1)
    id_to_index = {}
    for cnt, ID in enumerate(state_space_av_ids):
        id_to_index[ID] = cnt

    avs_info = environment_info['avs_info']
    if environment_info['has_av']:
        if avs_info:
            for ID, info in avs_info.items():
                # 1. dangerous events
                # 1.1 leading vehicle emergency braking
                emergency = 0.0
                leader_acc = info.get('leader_acc', 0.0)
                if leader_acc <= -4.0:
                    emergency += 1.0
                # 1.2 pedestrian intrusion
                leader_ped = info.get('leader_ped', False)
                if leader_ped:
                    emergency += 1.0
                # 1.3 vehicle cut-in
                leader_start = info.get('leader_start', False)
                if leader_start:
                    emergency += 0.7
                # 2. TTC
                ttc = info['observe_ttc']
                ttc_normalized = np.clip(1 / (ttc + 1e-5), 0, 1)
                # total risk
                av_risk = w_emergency * emergency + w_ttc * ttc_normalized
                av_risk = np.clip(av_risk, 0.1, 1)
                risk_list[id_to_index[ID]] = av_risk

    return risk_list


def compute_metrics(environment_info: Dict):
    # 1. average speed  FIXME: consider collisions
    av_speeds = environment_info['av_speeds']
    average_speed = 0
    if av_speeds:
        average_speed = sum(av_speeds) / len(av_speeds)

    # 2. comfort
    av_lat_accs = environment_info['av_lat_accs']
    av_lon_accs = environment_info['av_lon_accs']
    average_lat_acc = 0
    average_lon_acc = 0
    if av_lat_accs:
        average_lat_acc = sum(av_lat_accs) / len(av_lat_accs)
    if av_lon_accs:
        average_lon_acc = sum(abs(a) for a in av_lon_accs) / len(av_lon_accs)

    return average_speed, average_lat_acc, average_lon_acc


if __name__ == "__main__":
    Rewards = []
    Col_Rewards = []
    Vel_Rewards = []
    Dis_Keep_Rewards = []
    Comfort_Rewards = []
    Avoid_Rewards = []
    LC_Rewards = []
    if DRL == 'PPO_ResCor':
        Safety_Rewards = []
        Safety_Col_Rewards = []
        Safety_Avoid_Rewards = []
    Loss = []
    Episode_Steps = []
    Avg_Speeds = []
    Avg_Lat_Accs = []
    Avg_Lon_Accs = []
    Col_Nums = []

    if DRL == 'PPO' or DRL == 'PPO_SafetyLayer':
        GRL_Net, GRL_model = create_ppo(sim_params)
    elif DRL == 'PPO_ResCor':
        GRL_Net, GRL_model = create_ppo_rescor(sim_params)
    elif DRL == 'PPO_Lag':
        GRL_Net, GRL_model = create_ppo_lag(sim_params)
    else:
        raise ValueError(f"Unexpected value for DRL: {DRL}")

    print("#-------------HardBrake--------------#")
    print("#----------Training Begins-----------#")
    print("#------------------------------------#")
    print(f"Enable_Graph: {Enable_Graph}, GNN: {GNN}, DRL: {DRL}")

    for i in range(1, n_episodes + 1):
        sim_params['AV00_is_avoided'] = False
        sim_params['AVs_lc'] = {'AV_0.0': {'step': -1, 'target_lane': -1},
                                'AV_1.0': {'step': -1, 'target_lane': -1},
                                'AV_2.0': {'step': -1, 'target_lane': -1}}
        random_seed = random_seeds[i - 1]
        print(f"random_seed = {random_seed}")
        start_sumo_env(seed=random_seed, is_render=RENDER, params=sim_params)
        # warm-up environment
        action = np.zeros(GRL_Net.num_agents, dtype=int)
        env_info = take_actions(action, sim_params, preheat=True, time_step=0)
        state = graph_representation(env_info, sim_params)

        Reward = np.zeros(num_vehicles)
        Col_Reward = np.zeros(num_vehicles)
        Vel_Reward = np.zeros(num_vehicles)
        Dis_Keep_Reward = np.zeros(num_vehicles)
        Comfort_Reward = np.zeros(num_vehicles)
        Avoid_Reward = np.zeros(num_vehicles)
        LC_Reward = np.zeros(num_vehicles)
        if DRL == 'PPO_ResCor':
            Safety_Reward = np.zeros(num_vehicles)
            Safety_Col_Reward = np.zeros(num_vehicles)
            Safety_Avoid_Reward = np.zeros(num_vehicles)
        Step_Avg_Speeds = []
        Step_Avg_Lat_Accs = []
        Step_Avg_Lon_Accs = []
        Collision_Num = 0
        steps = 0
        for step in range(horizon):
            risk = risk_quantification_function(env_info, sim_params)
            actions_mask = get_valid_actions_mask(state, env_info, step, sim_params)
            # ------ choose actions ------ #
            if DRL == 'PPO' or DRL == 'PPO_SafetyLayer':
                action, prob, value = GRL_model.choose_action(state, actions_mask)
            elif DRL == 'PPO_ResCor':
                action, origin_prob, alpha, \
                    origin_action, origin_action_prob, origin_value, \
                    safety_action, safety_action_prob, safety_value = GRL_model.choose_action(state, actions_mask, risk)
            elif DRL == 'PPO_Lag':
                action, prob, value, cost_value = GRL_model.choose_action(state, actions_mask)
            else:
                action = GRL_model.choose_action(state, actions_mask)
            if isinstance(action, torch.Tensor):
                action = action.numpy()
            if DRL == 'PPO_SafetyLayer':
                action = safety_layer(state, env_info, action, sim_params)
            env_info = take_actions(action, sim_params, preheat=False, time_step=step)
            steps += 1
            next_state = graph_representation(env_info, sim_params)
            # ------ reward function ------ #
            reward, col_reward, vel_reward, dis_keep_reward, com_reward, avd_reward, lc_reward \
                = reward_function(env_info, sim_params)
            if DRL == 'PPO_ResCor':
                safety_reward, safety_col_reward, safety_avd_reward = safety_reward_function(env_info, sim_params)
            if DRL == 'PPO_Lag':
                cost, col_cost, ttc_cost = cost_function(env_info, sim_params)
                avd_reward = col_cost  # for convenience, directly replace
                lc_reward = ttc_cost
            avg_speed, avg_lat_acc, avg_lon_acc = compute_metrics(env_info)
            col_num = env_info['collision_num']
            done = False
            remain_number = traci.simulation.getMinExpectedNumber()
            if remain_number == 0:
                print(f"simulation over, Episode: {i}, steps: {steps}")
                done = True
            else:
                veh_ids = traci.vehicle.getIDList()
                ids_set = set(veh_ids)
                if ids_set == {'HV_0.0'}:
                    done = True
            # ------ store transition ------ #
            avs_done = env_info['avs_done']
            if env_info['has_av'] or avs_done:
                if DRL == 'PPO' or DRL == 'PPO_SafetyLayer':
                    GRL_model.store_transition(state, actions_mask, action, prob, value, reward, avs_done)
                elif DRL == 'PPO_ResCor':
                    GRL_model.store_transition(state, actions_mask, origin_prob, avs_done, risk, alpha,
                                               origin_action, origin_action_prob, origin_value, reward,
                                               safety_action, safety_action_prob, safety_value, safety_reward)
                elif DRL == 'PPO_Lag':
                    GRL_model.store_transition(state, actions_mask, action, prob, value, cost_value, reward, cost, avs_done)
            # episode reward
            Reward += reward
            Col_Reward += col_reward
            Vel_Reward += vel_reward
            Dis_Keep_Reward += dis_keep_reward
            Comfort_Reward += com_reward
            Avoid_Reward += avd_reward
            LC_Reward += lc_reward
            if DRL == 'PPO_ResCor':
                Safety_Reward += safety_reward
                Safety_Col_Reward += safety_col_reward
                Safety_Avoid_Reward += safety_avd_reward
            Collision_Num += col_num
            # ------ update policy ------ #
            if env_info['has_av'] or avs_done:
                Step_Avg_Speeds.append(avg_speed)
                Step_Avg_Lat_Accs.append(avg_lat_acc)
                Step_Avg_Lon_Accs.append(avg_lon_acc)
                GRL_model.learn()
            state = next_state
            if done:
                break
        traci.close()
        # ------ record training data ------ #
        training_data = GRL_model.get_statistics()
        loss = training_data[0]
        Loss.append(loss)
        Episode_Steps.append(steps)
        # normalize reward
        Reward /= steps
        Col_Reward /= steps
        Vel_Reward /= steps
        Dis_Keep_Reward /= steps
        Comfort_Reward /= steps
        Avoid_Reward /= steps
        LC_Reward /= steps
        if DRL == 'PPO_ResCor':
            Safety_Reward /= steps
            Safety_Col_Reward /= steps
            Safety_Avoid_Reward /= steps
        Reward_Sum_AVs = np.sum(Reward[:num_av])
        Col_Reward_Sum_AVs = np.sum(Col_Reward[:num_av])
        Vel_Reward_Sum_AVs = np.sum(Vel_Reward[:num_av])
        Dis_Keep_Reward_Sum_AVs = np.sum(Dis_Keep_Reward[:num_av])
        Comfort_Reward_Sum_AVs = np.sum(Comfort_Reward[:num_av])
        Avoid_Reward_Sum_AVs = np.sum(Avoid_Reward[:num_av])
        LC_Reward_Sum_AVs = np.sum(LC_Reward[:num_av])
        if DRL == 'PPO_ResCor':
            Safety_Reward_Sum_AVs = np.sum(Safety_Reward[:num_av])
            Safety_Col_Reward_Sum_AVs = np.sum(Safety_Col_Reward[:num_av])
            Safety_Avoid_Reward_Sum_AVs = np.sum(Safety_Avoid_Reward[:num_av])
        Rewards.append(Reward_Sum_AVs)
        Col_Rewards.append(Col_Reward_Sum_AVs)
        Vel_Rewards.append(Vel_Reward_Sum_AVs)
        Dis_Keep_Rewards.append(Dis_Keep_Reward_Sum_AVs)
        Comfort_Rewards.append(Comfort_Reward_Sum_AVs)
        Avoid_Rewards.append(Avoid_Reward_Sum_AVs)
        LC_Rewards.append(LC_Reward_Sum_AVs)
        if DRL == 'PPO_ResCor':
            Safety_Rewards.append(Safety_Reward_Sum_AVs)
            Safety_Col_Rewards.append(Safety_Col_Reward_Sum_AVs)
            Safety_Avoid_Rewards.append(Safety_Avoid_Reward_Sum_AVs)
        Episode_Avg_Speed = 0.0
        Episode_Avg_Lat_Acc = 0.0
        Episode_Avg_Lon_Acc = 0.0
        if len(Step_Avg_Speeds) > 0:
            Episode_Avg_Speed = sum(Step_Avg_Speeds) / len(Step_Avg_Speeds)
        if len(Step_Avg_Lat_Accs) > 0:
            Episode_Avg_Lat_Acc = sum(Step_Avg_Lat_Accs) / len(Step_Avg_Lat_Accs)
        if len(Step_Avg_Lon_Accs) > 0:
            Episode_Avg_Lon_Acc = sum(Step_Avg_Lon_Accs) / len(Step_Avg_Lon_Accs)
        Avg_Speeds.append(Episode_Avg_Speed)
        Avg_Lat_Accs.append(Episode_Avg_Lat_Acc)
        Avg_Lon_Accs.append(Episode_Avg_Lon_Acc)
        Col_Nums.append(Collision_Num)
        print(f'Episode: {i}, steps: {steps}, '
              f'Reward: {Reward_Sum_AVs:.1f}, '
              f'Episode_Avg_Speed: {Episode_Avg_Speed:.1f}, Collision_Num: {Collision_Num}')
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        col_reward_color = RED if Col_Reward_Sum_AVs < 0 else GREEN
        print(f'Col_Reward: {col_reward_color}{Col_Reward_Sum_AVs:.1f}{RESET}, '
              f'Vel_Reward: {Vel_Reward_Sum_AVs:.1f}, '
              f'Dis_Keep_Reward: {Dis_Keep_Reward_Sum_AVs:.2f}, '
              f'Comfort_Reward: {Comfort_Reward_Sum_AVs:.1f}, '
              f'Avoid_Reward: {Avoid_Reward_Sum_AVs:.2f}, '
              f'LC_Reward: {LC_Reward_Sum_AVs:.1f}')
        if DRL == 'PPO_ResCor':
            print(f'Safety_Reward: {Safety_Reward_Sum_AVs:.1f}, '
                  f'Safety_Col_Reward: {Safety_Col_Reward_Sum_AVs:.1f}, '
                  f'Safety_Avoid_Reward: {Safety_Avoid_Reward_Sum_AVs:.2f}')
        print('=' * 115)
        # plot
        if DRL == 'PPO_ResCor':
            plt.figure(num=1, figsize=(16, 7))
            plt.subplot(2, 5, 1)
            plt.title('Rewards')
            plt.plot(Rewards)
            plt.subplot(2, 5, 2)
            plt.title('Episode_Steps')
            plt.plot(Episode_Steps)
            plt.subplot(2, 5, 3)
            plt.title('Col_Rewards')
            plt.plot(Col_Rewards)
            plt.subplot(2, 5, 4)
            plt.title('Vel_Rewards')
            plt.plot(Vel_Rewards)
            plt.subplot(2, 5, 5)
            plt.title('Dis_Keep_Rewards')
            plt.plot(Dis_Keep_Rewards)
            plt.subplot(2, 5, 6)
            plt.title('Comfort_Rewards')
            plt.plot(Comfort_Rewards)
            plt.subplot(2, 5, 7)
            plt.title('Avoid_Rewards')
            plt.plot(Avoid_Rewards)
            plt.subplot(2, 5, 8)
            plt.title('Safety_Rewards')
            plt.plot(Safety_Rewards)
            plt.subplot(2, 5, 9)
            plt.title('Safety_Col_Rewards')
            plt.plot(Safety_Col_Rewards)
            plt.subplot(2, 5, 10)
            plt.title('Safety_Avoid_Rewards')
            plt.plot(Safety_Avoid_Rewards)
        else:
            plt.figure(num=1, figsize=(13, 7))
            plt.subplot(2, 4, 1)
            plt.title('Rewards')
            plt.plot(Rewards)
            plt.subplot(2, 4, 2)
            plt.title('Episode_Steps')
            plt.plot(Episode_Steps)
            plt.subplot(2, 4, 3)
            plt.title('Col_Rewards')
            plt.plot(Col_Rewards)
            plt.subplot(2, 4, 4)
            plt.title('Vel_Rewards')
            plt.plot(Vel_Rewards)
            plt.subplot(2, 4, 5)
            plt.title('Dis_Keep_Rewards')
            plt.plot(Dis_Keep_Rewards)
            plt.subplot(2, 4, 6)
            plt.title('Comfort_Rewards')
            plt.plot(Comfort_Rewards)
            plt.subplot(2, 4, 7)
            plt.title('Avoid_Rewards (Col_Costs)')
            plt.plot(Avoid_Rewards)
            plt.subplot(2, 4, 8)
            plt.title('LC_Rewards (TTC_Costs)')
            plt.plot(LC_Rewards)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.show(block=False)
        plt.pause(1)
    print('Training Finished.')

    # save model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    GRL_model.save_model(save_dir)
    np.save(save_dir + "/Loss", Loss)
    np.save(save_dir + "/Episode_Steps", Episode_Steps)
    np.save(save_dir + "/Rewards", Rewards)
    np.save(save_dir + "/Col_Rewards", Col_Rewards)
    np.save(save_dir + "/Vel_Rewards", Vel_Rewards)
    np.save(save_dir + "/Dis_Keep_Rewards", Dis_Keep_Rewards)
    np.save(save_dir + "/Comfort_Rewards", Comfort_Rewards)
    np.save(save_dir + "/Avoid_Rewards", Avoid_Rewards)
    np.save(save_dir + "/LC_Rewards", LC_Rewards)
    if DRL == 'PPO_ResCor':
        np.save(save_dir + "/Safety_Rewards", Safety_Rewards)
        np.save(save_dir + "/Safety_Col_Rewards", Safety_Col_Rewards)
        np.save(save_dir + "/Safety_Avoid_Rewards", Safety_Avoid_Rewards)
    np.save(save_dir + "/Avg_Speeds", Avg_Speeds)
    np.save(save_dir + "/Col_Nums", Col_Nums)

    plt.figure(1)
    plt.savefig(save_dir + '/Data_Figure.png')
