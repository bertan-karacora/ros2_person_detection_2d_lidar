import ros2_utils.node as utils_node

from ros2_person_detection_2d_lidar.ros import NodePersonDetection2DLiDAR


def main():
    utils_node.start_and_spin_node(NodePersonDetection2DLiDAR)


if __name__ == "__main__":
    main()
