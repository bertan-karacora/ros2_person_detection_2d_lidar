import numpy as np
from scipy.spatial.transform import Rotation

import ros2_person_detection_2d_lidar.libs.utils_geometry as utils_geometry


class Detection:
    def __init__(
        self,
        position,
        size,
        orientation,
        label,
        confidence,
        features_appearance=None,
        name=None,
        id_track=None,
        is_lying=False,
        is_sitting=False,
        is_pointing_left=False,
        is_pointing_right=False,
        is_standing=False,
        is_waving_left=False,
        is_waving_right=False,
    ):
        self.confidence = confidence
        self.features_appearance = features_appearance
        self.is_lying = is_lying
        self.is_sitting = is_sitting
        self.is_pointing_left = is_pointing_left
        self.is_pointing_right = is_pointing_right
        self.is_standing = is_standing
        self.is_waving_left = is_waving_left
        self.is_waving_right = is_waving_right
        self.id_track = id_track
        self.label = label
        self.name = name
        self.orientation = orientation
        self.position = position
        self.size = size

    def __repr__(self):
        s = f"""Detection({self.position}, {self.size}, {self.orientation}, {self.label}, {self.name}, {self.confidence}, {self.id_track})"""
        return s

    def lift_2d_to_3d(self, height_sensor, height_bbox=1.8):
        position_3d = np.zeros(3)
        position_3d[:2] = self.position
        position_3d[2] = height_bbox / 2.0 - height_sensor
        self.position = position_3d

        size_3d = np.zeros(3)
        if np.any(self.size > 0.0):
            size_3d[:2] = self.size
            size_3d[2] = height_bbox
        self.size = size_3d

        self.orientation = utils_geometry.complex_to_quaternion(self.orientation[0], self.orientation[1])

    def transform(self, rotation, vec_t):
        self.position = rotation.apply(self.position) + vec_t
        self.orientation = (rotation * Rotation.from_quat(self.orientation)).as_quat()
