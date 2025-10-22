# Person Detection in 2D LiDAR Scans

ROS 2 package for Person Detection in 2D LiDAR Scans.

Note: This is a public, stripped-down version of a private repository. It may depend on other repositories which might not have a public version. Some paths, configurations, dependencies, have been removed or altered, so the code may not run out of the box.

## Setup

```bash
git clone https://github.com/bertan-karacora/ros2_person_detection_2d_lidar.git
cd ros2_person_detection_2d_lidar
git submodule update --init --recursive
```

## Installation

### Build container

```bash
container/build.sh
```

## Usage

### Run in container

```bash
container/run.sh
```

You may provide any command with arguments directly, e.g.:

```bash
container/run.sh -a scripts/start_all.sh use_full_fov:=False name_topic_scan:=/sick_laser_front/scan use_filter_region:=False
```

## Links

- [Code of DROW3, DR-SPAAM, and Self-Supervised Person Detection in 2D Range Data using a Calibrated Camera](https://github.com/VisualComputingInstitute/2D_lidar_person_detection)
- [DROW](https://arxiv.org/pdf/1603.02636)
- [DROW3x](https://arxiv.org/pdf/1804.02463)
- [DR-SPAAM](https://arxiv.org/pdf/2004.14079)
- [Self-Supervised Person Detection in 2D Range Data using a Calibrated Camera](https://arxiv.org/pdf/2012.08890)
- [2D vs. 3D LiDAR-based Person Detection on Mobile Robots](https://arxiv.org/pdf/2106.11239)

## TODO

- Add TensorRT acceleration
