from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args_launch = [
        DeclareLaunchArgument("name_config", description="Name of the model", default_value="jrdb_drspaam_bbox"),
        DeclareLaunchArgument("factor_downsampling", description="Downsampling factor of input scans", default_value="1"),
        DeclareLaunchArgument("height_bbox", description="Height of detected persons", default_value="1.8"),
        DeclareLaunchArgument("height_sensor", description="Height of the sensor in world frame", default_value="0.20864"),
        DeclareLaunchArgument("distance_nms", description="Distance defining the neighborhood to apply non-maximum suppression in", default_value="0.5"),
        DeclareLaunchArgument("threshold_confidence", description="Confidence threshold for detections to be considered valid", default_value="0.8"),
        DeclareLaunchArgument("use_full_fov", description="Consideration of scan to be a full fov 360 degree scan", default_value="True"),
        DeclareLaunchArgument("use_filter_region", description="Usage of region filter", default_value="False"),
        DeclareLaunchArgument("use_transform_frame", description="Usage of transform to target frame", default_value="False"),
        DeclareLaunchArgument("name_markers", description="Name of the markers file", default_value="ais_arena"),
        DeclareLaunchArgument("name_region", description="Name of the region for filtering", default_value="arena"),
        DeclareLaunchArgument("name_frame_target", description="Name of the target frame for detections", default_value="map"),
        DeclareLaunchArgument("name_topic_scan", description="Name of the 2D LiDAR scans topic (for subscriber)", default_value="/scan"),
        DeclareLaunchArgument("name_topic_detections", description="Name of the detections topic (for publisher)", default_value="/person_detection_2d_lidar/detections"),
        DeclareLaunchArgument("name_topic_marker", description="Name of the marker topic (for publisher)", default_value="/person_detection_2d_lidar/marker"),
        DeclareLaunchArgument("use_service_only", description="Usage of service-only mode", default_value="False"),
    ]
    launch_description = LaunchDescription(args_launch)

    action_person_detection_2d_lidar = Node(
        package="ros2_person_detection_2d_lidar",
        namespace="",
        executable="spin",
        name="ros2_person_detection_2d_lidar",
        output="screen",
        parameters=[
            {
                "name_config": LaunchConfiguration("name_config"),
                "factor_downsampling": LaunchConfiguration("factor_downsampling"),
                "height_bbox": LaunchConfiguration("height_bbox"),
                "height_sensor": LaunchConfiguration("height_sensor"),
                "distance_nms": LaunchConfiguration("distance_nms"),
                "threshold_confidence": LaunchConfiguration("threshold_confidence"),
                "use_full_fov": LaunchConfiguration("use_full_fov"),
                "use_filter_region": LaunchConfiguration("use_filter_region"),
                "use_transform_frame": LaunchConfiguration("use_transform_frame"),
                "name_markers": LaunchConfiguration("name_markers"),
                "name_region": LaunchConfiguration("name_region"),
                "name_frame_target": LaunchConfiguration("name_frame_target"),
                "name_topic_scan": LaunchConfiguration("name_topic_scan"),
                "name_topic_detections": LaunchConfiguration("name_topic_detections"),
                "name_topic_marker": LaunchConfiguration("name_topic_marker"),
                "use_service_only": LaunchConfiguration("use_service_only"),
            }
        ],
        respawn=True,
        respawn_delay=5.0,
    )
    launch_description.add_action(action_person_detection_2d_lidar)

    return launch_description
