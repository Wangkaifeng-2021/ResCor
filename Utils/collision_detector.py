import math
from traci._simulation import tc


class Collision(object):
    def __init__(self, collider, victim, collider_type, victim_type, collider_angle, victim_angle):
        self.collider = collider
        self.victim = victim
        self.colliderType = collider_type
        self.victimType = victim_type
        self.colliderAngle = collider_angle
        self.victimAngle = victim_angle

    def __attr_repr__(self, attrname, default=""):
        if getattr(self, attrname) == default:
            return ""
        else:
            val = getattr(self, attrname)
            if val == tc.INVALID_DOUBLE_VALUE:
                val = "INVALID"
            return "%s=%s" % (attrname, val)

    def __repr__(self):
        return "Collision(%s)" % ', '.join([v for v in [
            self.__attr_repr__("collider"),
            self.__attr_repr__("colliderType"),
            self.__attr_repr__("colliderAngle"),
            self.__attr_repr__("victim"),
            self.__attr_repr__("victimType"),
            self.__attr_repr__("victimAngle"),
        ] if v != ""])


class CollisionDetector:
    def __init__(self, vehs_info):
        self.vehs_info = vehs_info

    @staticmethod
    def get_center(vehicle):
        x_front = vehicle["x"]
        y_front = vehicle["y"]
        angle_rad = math.radians(vehicle["angle"])

        center_x = x_front - (vehicle["length"] / 2) * math.sin(angle_rad)
        center_y = y_front - (vehicle["length"] / 2) * math.cos(angle_rad)

        return center_x, center_y

    def get_corners(self, vehicle):
        center_x, center_y = self.get_center(vehicle)
        angle_rad = math.radians(vehicle["angle"])
        length = vehicle["length"]
        width = vehicle["width"]

        half_length = length / 2
        half_width = width / 2

        corners = []
        for dx, dy in [(-half_length, -half_width), (-half_length, half_width),
                       (half_length, half_width), (half_length, -half_width)]:
            corner_x = center_x + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
            corner_y = center_y + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            corners.append((corner_x, corner_y))

        return corners

    @staticmethod
    def project_polygon(axis, corners):
        min_proj = float('inf')
        max_proj = -float('inf')
        for corner in corners:
            proj = corner[0] * axis[0] + corner[1] * axis[1]
            min_proj = min(min_proj, proj)
            max_proj = max(max_proj, proj)
        return min_proj, max_proj

    def is_separating_axis(self, axis, corners1, corners2):
        min1, max1 = self.project_polygon(axis, corners1)
        min2, max2 = self.project_polygon(axis, corners2)
        return max1 < min2 or max2 < min1

    def check_collision(self, corners1, corners2):
        axes = []
        for i in range(4):
            next_i = (i + 1) % 4
            edge = (corners1[next_i][0] - corners1[i][0], corners1[next_i][1] - corners1[i][1])
            axis = (-edge[1], edge[0])
            norm = math.sqrt(axis[0] ** 2 + axis[1] ** 2)
            axes.append((axis[0] / norm, axis[1] / norm))

        for i in range(4):
            next_i = (i + 1) % 4
            edge = (corners2[next_i][0] - corners2[i][0], corners2[next_i][1] - corners2[i][1])
            axis = (-edge[1], edge[0])
            norm = math.sqrt(axis[0] ** 2 + axis[1] ** 2)
            axes.append((axis[0] / norm, axis[1] / norm))

        for axis in axes:
            if self.is_separating_axis(axis, corners1, corners2):
                return False

        return True

    def determine_collider_victim(self, veh1, veh2):
        center1 = self.get_center(veh1)
        center2 = self.get_center(veh2)

        direction_vector = (center2[0] - center1[0], center2[1] - center1[1])

        angle_rad = math.radians(veh1["angle"])
        veh1_direction = (math.sin(angle_rad), math.cos(angle_rad))

        dot_product = direction_vector[0] * veh1_direction[0] + direction_vector[1] * veh1_direction[1]
        magnitude_direction = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
        magnitude_veh1 = math.sqrt(veh1_direction[0] ** 2 + veh1_direction[1] ** 2)
        angle = math.acos(dot_product / (magnitude_direction * magnitude_veh1))

        if angle < math.pi / 2:
            return veh1["id"], veh2["id"]
        else:
            return veh2["id"], veh1["id"]

    def detect_collisions(self):
        veh_ids = list(self.vehs_info.keys())
        collisions = []

        for i in range(len(veh_ids)):
            for j in range(i + 1, len(veh_ids)):
                veh1 = self.vehs_info[veh_ids[i]]
                veh2 = self.vehs_info[veh_ids[j]]
                if veh1["type"] == 'DEFAULT_PEDTYPE' and veh2["type"] == 'DEFAULT_PEDTYPE':
                    continue

                corners1 = self.get_corners(veh1)
                corners2 = self.get_corners(veh2)

                if self.check_collision(corners1, corners2):
                    collider, victim = self.determine_collider_victim(veh1, veh2)
                    collider_type = self.vehs_info[collider]['type']
                    victim_type = self.vehs_info[victim]['type']
                    collider_angle = self.vehs_info[collider]['angle']
                    victim_angle = self.vehs_info[victim]['angle']
                    _collision = Collision(collider=collider, victim=victim,
                                           collider_type=collider_type, victim_type=victim_type,
                                           collider_angle=collider_angle, victim_angle=victim_angle)
                    collisions.append(_collision)

        return collisions


if __name__ == "__main__":
    # example
    vehs_info = {
        "AV.0": {"id": "AV.0", "x": 47.0, "y": -4.8, "angle": 90.0, "length": 5.0, "width": 1.8, 'type': 'AV'},
        "AV.1": {"id": "AV.1", "x": 40.0, "y": -1.6, "angle": 90.0, "length": 5.0, "width": 1.8, 'type': 'AV'},
        "HV.0": {"id": "HV.0", "x": 49.0, "y": -4.8, "angle": 90.0, "length": 5.0, "width": 1.8, 'type': 'HV'},
        "HV.1": {"id": "HV.1", "x": 49.0, "y": -8.0, "angle": 270.0, "length": 5.0, "width": 1.8, 'type': 'HV'},
        "HV.2": {"id": "HV.2", "x": 50.0, "y": -8.0, "angle": 270.0, "length": 5.0, "width": 1.8, 'type': 'HV'}
        # more vehs info
    }

    detector = CollisionDetector(vehs_info)
    collisions = detector.detect_collisions()
    if collisions:
        print("collision:", collisions)
    else:
        print("no collision")
