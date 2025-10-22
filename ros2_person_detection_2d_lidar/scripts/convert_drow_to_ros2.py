# TODO: clean
import argparse
import math
import json

import numpy as np

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Pose, TransformStamped, Quaternion, Vector3
import rclpy
import rclpy.serialization as serialization
import rosbag2_py as rosbag2
from rosbag2_py import StorageOptions, ConverterOptions, SequentialWriter
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from tf2_msgs.msg import TFMessage


def parse_args():
    parser = argparse.ArgumentParser(description="Create ROS 2 bag from DROW data")
    parser.add_argument("--path_csv", type=str, required=True, help="Path to DROW sequence csv file")
    parser.add_argument("--path_out", type=str, required=False, default="./out.bag")
    args = parser.parse_args()

    return args.path_csv, args.path_out


def load_scans(fname):
    data = np.genfromtxt(fname, delimiter=",")
    seqs, times, scans = data[:, 0].astype(np.uint32), data[:, 1].astype(np.float32), data[:, 2:].astype(np.float32)
    return seqs, times, scans


def load_odoms(fname):
    data = np.genfromtxt(fname, delimiter=",")
    seqs, times = data[:, 0].astype(np.uint32), data[:, 1].astype(np.float32)
    odoms = data[:, 2:].astype(np.float32)  # x, y, phi
    return seqs, times, odoms


def load_dets(name):
    def _doload(fname):
        seqs, dets = [], []
        with open(fname) as f:
            for line in f:
                seq, tail = line.split(",", 1)
                seqs.append(int(seq))
                dets.append(json.loads(tail))
        return seqs, dets

    s1, wcs = _doload(name + "wc")
    s2, was = _doload(name + "wa")
    s3, wps = _doload(name + "wp")

    assert all(a == b == c for a, b, c in zip(s1, s2, s3)), "Uhhhh?"
    return np.array(s1), wcs, was, wps


def convert_drow_to_ros2(seq_name, bag_name):
    rclpy.init()
    node = rclpy.create_node("drow_data_converter")

    writer = SequentialWriter()
    storage_options = StorageOptions(uri=bag_name, storage_id="sqlite3")
    converter_options = ConverterOptions("", "")
    writer.open(storage_options, converter_options)

    # Odometry
    topic_info = rosbag2._storage.TopicMetadata(name="/tf", type="tf2_msgs/msg/TFMessage", serialization_format="cdr")
    writer.create_topic(topic_info)

    tran = TransformStamped()
    tran.header.frame_id = "base_footprint"
    tran.child_frame_id = "sick_laser_front"

    seqs, times, odoms = load_odoms(seq_name[:-3] + "odom2")
    for time, odom in zip(times, odoms):
        time_msg = Time()
        time_msg.sec = int(time)
        time_msg.nanosec = int((time - int(time)) * 1e9)
        tran.header.stamp = time_msg
        tran.transform.translation.x = float(odom[0])
        tran.transform.translation.y = float(odom[1])
        tran.transform.translation.z = 0.0
        tran.transform.rotation.x = 0.0
        tran.transform.rotation.y = 0.0
        tran.transform.rotation.z = math.sin(odom[2] * 0.5)
        tran.transform.rotation.w = math.cos(odom[2] * 0.5)
        tf_msg = TFMessage(transforms=[tran])
        writer.write("/tf", serialization.serialize_message(tf_msg), int(time_msg.sec * 1e9) + time_msg.nanosec)

    # Scans
    topic_info = rosbag2._storage.TopicMetadata(name="/sick_laser_front/scan", type="sensor_msgs/msg/LaserScan", serialization_format="cdr")
    writer.create_topic(topic_info)

    scan_msg = LaserScan()
    scan_msg.header.frame_id = "sick_laser_front"
    scan_msg.angle_min = np.radians(-225.0 / 2)
    scan_msg.angle_max = np.radians(225.0 / 2)
    scan_msg.range_min = 0.005
    scan_msg.range_max = 100.0
    scan_msg.scan_time = 0.066667
    scan_msg.time_increment = 0.000062
    scan_msg.angle_increment = (scan_msg.angle_max - scan_msg.angle_min) / 450

    seqs_scans, times_scans, scans = load_scans(seq_name)
    for time, scan in zip(times_scans, scans):
        time_msg = Time()
        time_msg.sec = int(time)
        time_msg.nanosec = int((time - int(time)) * 1e9)
        scan_msg.header.stamp = time_msg
        scan_msg.ranges = scan.tolist()
        writer.write("/sick_laser_front/scan", serialization.serialize_message(scan_msg), int(time_msg.sec * 1e9) + time_msg.nanosec)

    # Marker
    topic_info = rosbag2._storage.TopicMetadata(name="/marker_gt", type="visualization_msgs/msg/Marker", serialization_format="cdr")
    writer.create_topic(topic_info)

    seqs, wcs, was, wps = load_dets(seq_name[:-3])
    id_to_time = dict(zip(seqs_scans, times_scans))
    times = [id_to_time[seq] for seq in seqs]
    for time, wc, wa, wp in zip(times, wcs, was, wps):
        time_msg = Time()
        time_msg.sec = int(time)
        time_msg.nanosec = int((time - int(time)) * 1e9)

        w = wc + wa + wp
        message_marker = detections_to_marker_message(w, header=scan_msg.header)
        message_marker.header.stamp = time_msg

        writer.write("/marker_gt", serialization.serialize_message(message_marker), int(time_msg.sec * 1e9) + time_msg.nanosec)

    node.destroy_node()
    rclpy.shutdown()


def detections_to_marker_message(detections, header=None):
    radius = 0.1
    angles = np.linspace(0.0, 2.0 * np.pi, 10)
    offsets_xy = radius * np.stack((np.cos(angles), np.sin(angles)), axis=1)

    messages_points = []
    for detection in detections:
        position = detection
        for i in range(offsets_xy.shape[0] - 1):
            message_point = Point(x=position[0] + offsets_xy[i, 0], y=position[1] + offsets_xy[i, 1], z=0.0)
            messages_points.append(message_point)

            message_point = Point(x=position[0] + offsets_xy[i + 1, 0], y=position[1] + offsets_xy[i + 1, 1], z=0.0)
            messages_points.append(message_point)

    message_marker = Marker(
        header=header,
        ns="dr_spaam_ros",
        id=0,
        action=Marker.ADD,
        type=Marker.LINE_LIST,
        scale=Vector3(x=0.03, y=0.0, z=0.0),
        pose=Pose(position=Point(x=0.0, y=0.0, z=0.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
        color=ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
        frame_locked=False,
        points=messages_points,
    )

    return message_marker


def main():
    path_csv, path_out = parse_args()
    convert_drow_to_ros2(path_csv, path_out)


if __name__ == "__main__":
    main()
