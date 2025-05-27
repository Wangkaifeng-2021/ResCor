import traci
import optparse
import math
import numpy as np
from typing import Dict, List, Tuple


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()

    return options


def execute_commands(vehicle_id: str, acceleration: float, steering: float, params: Dict) -> Tuple[float, float]:
    step_length = params['step_length']
    max_speed = params['max_speed']

    # 1. turn
    front_bumper_x, front_bumper_y = traci.vehicle.getPosition(vehID=vehicle_id)
    angle = traci.vehicle.getAngle(vehID=vehicle_id)
    length = params['veh_length']
    d = length / 2.0
    center_x = front_bumper_x - d * math.sin(math.radians(angle))
    center_y = front_bumper_y - d * math.cos(math.radians(angle))

    new_angle = (angle + steering * step_length) % 360

    # 2. move
    speed = traci.vehicle.getSpeed(vehID=vehicle_id)
    new_speed = speed + acceleration * step_length
    target_speed = np.clip(new_speed, a_min=0.0, a_max=max_speed).item()
    traci.vehicle.setSpeedMode(vehID=vehicle_id, speedMode=0b000000)
    traci.vehicle.setSpeed(vehID=vehicle_id, speed=target_speed)

    ds = target_speed * step_length
    new_center_x = center_x + ds * math.sin(math.radians(new_angle))
    new_center_y = center_y + ds * math.cos(math.radians(new_angle))

    new_front_bumper_x = new_center_x + d * math.sin(math.radians(new_angle))
    new_front_bumper_y = new_center_y + d * math.cos(math.radians(new_angle))
    traci.vehicle.moveToXY(vehID=vehicle_id, edgeID="", laneIndex=-1, x=new_front_bumper_x, y=new_front_bumper_y,
                           angle=new_angle)
    return target_speed, new_angle


def execute_commands_bicycle(vehicle_id: str, acceleration: float, steering: float, params: Dict) -> Tuple[float, float]:
    L = params['wheelbase']
    step_length = params['step_length']
    max_speed = params['max_speed']

    # 1. turn
    front_bumper_x, front_bumper_y = traci.vehicle.getPosition(vehID=vehicle_id)
    angle = traci.vehicle.getAngle(vehID=vehicle_id)
    d = 4.1
    rear_center_x = front_bumper_x - d * math.sin(math.radians(angle))
    rear_center_y = front_bumper_y - d * math.cos(math.radians(angle))

    speed = traci.vehicle.getSpeed(vehID=vehicle_id)
    angle_rad = math.radians(angle)
    steering_rad = math.radians(steering)
    new_angle_rad = angle_rad + (speed / L) * math.tan(steering_rad) * step_length
    new_angle = math.degrees(new_angle_rad) % 360

    # 2. move
    new_speed = speed + acceleration * step_length
    target_speed = np.clip(new_speed, a_min=0.0, a_max=max_speed).item()
    traci.vehicle.setSpeedMode(vehID=vehicle_id, speedMode=0b000000)
    traci.vehicle.setSpeed(vehID=vehicle_id, speed=target_speed)

    ds = target_speed * step_length
    new_rear_center_x = rear_center_x + ds * math.sin(math.radians(new_angle))
    new_rear_center_y = rear_center_y + ds * math.cos(math.radians(new_angle))

    new_front_bumper_x = new_rear_center_x + d * math.sin(math.radians(new_angle))
    new_front_bumper_y = new_rear_center_y + d * math.cos(math.radians(new_angle))
    traci.vehicle.moveToXY(vehID=vehicle_id, edgeID="", laneIndex=-1, x=new_front_bumper_x, y=new_front_bumper_y,
                           angle=new_angle)
    return target_speed, new_angle


def compute_target_pose(vehicle_id: str, acceleration: float, steering: float, params: Dict) -> Tuple:
    step_length = params['step_length']

    front_bumper_x, front_bumper_y = traci.vehicle.getPosition(vehID=vehicle_id)
    angle = traci.vehicle.getAngle(vehID=vehicle_id)
    length = params['veh_length']
    d = length / 2.0

    center_x = front_bumper_x - d * math.sin(math.radians(angle))
    center_y = front_bumper_y - d * math.cos(math.radians(angle))

    target_angle = (angle + steering * step_length) % 360

    speed = traci.vehicle.getSpeed(vehID=vehicle_id)
    target_speed = speed + acceleration * step_length

    ds = target_speed * step_length
    target_center_x = center_x + ds * math.sin(math.radians(target_angle))
    target_center_y = center_y + ds * math.cos(math.radians(target_angle))

    target_front_bumper_x = target_center_x + d * math.sin(math.radians(target_angle))
    target_front_bumper_y = target_center_y + d * math.cos(math.radians(target_angle))

    return target_front_bumper_x, target_front_bumper_y, target_angle, target_speed


def find_closest_vehicle(lf_vehs: Tuple) -> Tuple:
    if not lf_vehs:
        return None, None

    min_distance = float('inf')
    closest_vehicle_id = None

    for vehicle_id, distance in lf_vehs:
        if distance < min_distance:
            min_distance = distance
            closest_vehicle_id = vehicle_id

    return closest_vehicle_id, min_distance


def get_corners(front_bumper_x: float, front_bumper_y: float, angle: float, params: Dict) -> List:

    half_length = params['veh_length'] / 2.0
    half_width = params['veh_width'] / 2.0
    angle_rad = math.radians(angle)

    center_x = front_bumper_x - half_length * math.sin(angle_rad)
    center_y = front_bumper_y - half_length * math.cos(angle_rad)

    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    corners = [
        (+half_length, +half_width),  # lf
        (+half_length, -half_width),  # rf
        (-half_length, -half_width),  # rr
        (-half_length, +half_width),  # lr
    ]

    rotated_corners = []
    for dx, dy in corners:
        x = center_x + dx * sin_angle - dy * cos_angle
        y = center_y + dx * cos_angle + dy * sin_angle
        rotated_corners.append((x, y))

    return rotated_corners
