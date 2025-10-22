import time

import numpy as np
from scipy.spatial.transform import Rotation

from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor, ParameterType
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
import rclpy.qos as qos
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker

from ros2_interfaces.msg import DetectionsPerson
import ros2_utils.geometry as utils_geometry_ros
from ros2_utils.markers_client import MarkersClient
from ros2_utils.parameter_handler import ParameterHandler
from ros2_utils.tf_oracle import TFOracle

from ros2_person_detection_2d_lidar.detection import Detector
import ros2_person_detection_2d_lidar.ros.utils as utils_ros


class NodePersonDetection2DLiDAR(Node):
    def __init__(
        self,
        name_config="jrdb_drspaam_bbox",
        factor_downsampling=1,
        height_bbox=1.8,
        height_sensor=0.20864,
        distance_nms=0.5,
        threshold_confidence=0.95,
        use_full_fov=True,
        name_frame_target="map",
        name_topic_detections="/person_detection_2d_lidar/detections",
        name_topic_marker="/person_detection_2d_lidar/marker",
        name_topic_scan="/scan",
        use_filter_region=True,
        use_transform_frame=False,
        name_markers="ais_arena",
        name_region="arena",
        use_service_only=False,
    ):
        super().__init__(node_name="person_detection_2d_lidar")

        self.client_markers = None
        self.detector = None
        self.distance_nms = distance_nms
        self.factor_downsampling = factor_downsampling
        self.handler_parameters = None
        self.height_bbox = height_bbox
        self.height_sensor = height_sensor
        self.lock = None
        self.name_config = name_config
        self.name_frame_target = name_frame_target
        self.name_markers = name_markers
        self.name_region = name_region
        self.name_topic_detections = name_topic_detections
        self.name_topic_marker = name_topic_marker
        self.name_topic_scan = name_topic_scan
        self.offsets_marker = None
        self.publisher_detections = None
        self.publisher_marker = None
        self.subscriber_scan = None
        self.tf_broadcaster = None
        self.tf_buffer = None
        self.tf_listener = None
        self.tf_oracle = None
        self.threshold_confidence = threshold_confidence
        self.use_filter_region = use_filter_region
        self.use_transform_frame = use_transform_frame
        self.use_full_fov = use_full_fov
        self.use_service_only = use_service_only

        self._init()

    def _init(self):
        self._init_offsets_marker()
        self._init_tf_oracle()

        self.handler_parameters = ParameterHandler(self, verbose=False)
        self._init_parameters()

        self._init_client_markers()
        self._init_detector()

        self._del_publishers()
        self._init_publishers()
        self._del_services()
        self._init_services()
        self._del_subscribers()
        self._init_subscribers()

    def _init_offsets_marker(self):
        radius = 0.4
        angles = np.linspace(0.0, 2.0 * np.pi, 20)
        self.offsets_marker = radius * np.stack((np.cos(angles), np.sin(angles)), axis=1)

    def _init_tf_oracle(self):
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_oracle = TFOracle(self)

    def _init_client_markers(self):
        if self.name_markers:
            self.client_markers = MarkersClient(self, markers_name=self.name_markers)

    def _del_client_markers(self):
        self.client_markers = None

    def _init_detector(self):
        self.detector = Detector(name_config=self.name_config, factor_downsampling=self.factor_downsampling, threshold_confidence=self.threshold_confidence, use_full_fov=self.use_full_fov, distance_nms=self.distance_nms)

    def _init_publishers(self):
        profile_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.publisher_detections = self.create_publisher(
            msg_type=DetectionsPerson,
            topic=self.name_topic_detections,
            qos_profile=profile_qos,
            callback_group=ReentrantCallbackGroup(),
        )
        self.publisher_marker = self.create_publisher(
            msg_type=Marker,
            topic=self.name_topic_marker,
            qos_profile=profile_qos,
            callback_group=ReentrantCallbackGroup(),
        )

    def _del_publishers(self):
        names_publisher = ["publisher_detections", "publisher_marker"]
        for name_publisher in names_publisher:
            publisher = getattr(self, name_publisher)
            if publisher is not None:
                self.destroy_publisher(publisher)
                setattr(self, name_publisher, None)

    def _init_subscribers(self):
        if self.use_service_only:
            return

        profile_qos = qos.qos_profile_sensor_data

        self.subscriber_scan = self.create_subscription(
            msg_type=LaserScan,
            topic=self.name_topic_scan,
            callback=self.on_scan_received,
            qos_profile=profile_qos,
            callback_group=ReentrantCallbackGroup(),
        )

    def _del_subscribers(self):
        names_subscriber = ["subscriber_scan"]
        for name_subscriber in names_subscriber:
            subscriber = getattr(self, name_subscriber)
            if subscriber is not None:
                self.destroy_subscription(subscriber)
                setattr(self, name_subscriber, None)

    def _init_services(self): ...

    def _del_services(self):
        names_service = []
        for name_service in names_service:
            service = name_service
            if service is not None:
                self.destroy_service(service)
                setattr(self, name_service, None)

    def on_scan_received(self, msg_scan):
        time_start = time.time()

        scan = utils_ros.msg_to_scan(msg_scan)
        time_scan_ros = Time.from_msg(msg_scan.header.stamp)

        detections = self.detector(scan)

        detections = self.lift_detections_to_3d(detections)
        if self.use_transform_frame:
            detections = self.transform_detections(detections, name_frame_source=msg_scan.header.frame_id, time_source=time_scan_ros)
        if self.use_filter_region:
            detections = self.filter_detections(detections, name_frame=self.name_frame_target if self.use_transform_frame else msg_scan.header.frame_id, time_ros=time_scan_ros)
        self.publish_detections(detections, msg_scan)

        time_end = time.time()
        duration_inference = time_end - time_start
        self.get_logger().info(f"Number of persons: {len(detections)} | Duration of inference: {duration_inference:.3f}", throttle_duration_sec=1.0)

    def lift_detections_to_3d(self, detections):
        if not detections:
            return detections

        for detection in detections:
            detection.lift_2d_to_3d(height_sensor=self.height_sensor, height_bbox=self.height_bbox)

        return detections

    def transform_detections(self, detections, name_frame_source, time_source_ros=None):
        if not detections:
            return detections

        is_success, info, transform = self.tf_oracle.get_transform(name_frame_source, self.name_frame_target, time=time_source_ros, target_format="default", timeout=1.0)
        if not is_success:
            self.get_logger().warn(f"{info}")
            return []

        vec_t = transform[:3]
        vec_q = transform[3:]
        rotation = Rotation.from_quat(vec_q)

        for detection in detections:
            detection.transform(rotation, vec_t)

        return detections

    def filter_detections(self, detections, name_frame, time_ros=None):
        if not detections:
            return detections

        if self.client_markers is None:
            self.get_logger().warn("Region filtering skipped because the markers client is not initialized")
            return detections

        for i, detection in enumerate(detections):
            msg_point_stamped = utils_geometry_ros.create_point(position_arr=detection.position, frame=name_frame, stamp=time_ros)

            is_success, info, is_in_region = self.client_markers.in_region(msg_point_stamped, self.name_region)
            if not is_success:
                self.get_logger().warn(f"{info}")
                return detections

            if not is_in_region:
                del detections[i]

        return detections

    def publish_detections(self, detections, msg_scan):
        msg_detections = utils_ros.detections_to_msg(detections, header=msg_scan.header, use_transform_frame=self.use_transform_frame, name_frame_target=self.name_frame_target)
        self.publisher_detections.publish(msg_detections)

        msg_marker = utils_ros.detections_to_marker_msg(
            detections,
            offsets=self.offsets_marker,
            header=msg_detections.header,
            use_transform_frame=self.use_transform_frame,
            name_frame_target=self.name_frame_target,
            namespace=self.get_name(),
        )
        self.publisher_marker.publish(msg_marker)

    def _init_parameters(self):
        self.add_on_set_parameters_callback(self.handler_parameters.parameter_callback)

        self._init_parameter_name_config()
        self._init_parameter_factor_downsampling()
        self._init_parameter_height_bbox()
        self._init_parameter_height_sensor()
        self._init_parameter_distance_nms()
        self._init_parameter_threshold_confidence()
        self._init_parameter_use_full_fov()
        self._init_parameter_use_filter_region()
        self._init_parameter_use_transform_frame()
        self._init_parameter_name_markers()
        self._init_parameter_name_region()
        self._init_parameter_name_topic_detections()
        self._init_parameter_name_topic_scan()
        self._init_parameter_name_topic_marker()
        self._init_parameter_use_service_only()

        self.handler_parameters.all_declared()

    def _init_parameter_name_config(self):
        descriptor = ParameterDescriptor(
            name="name_config",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the config",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.name_config, descriptor)

    def _init_parameter_factor_downsampling(self):
        descriptor = ParameterDescriptor(
            name="factor_downsampling",
            type=ParameterType.PARAMETER_INTEGER,
            description="Downsampling factor of input scans",
            read_only=False,
            integer_range=(IntegerRange(from_value=1, to_value=16, step=1),),
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.factor_downsampling, descriptor)

    def _init_parameter_height_bbox(self):
        descriptor = ParameterDescriptor(
            name="height_bbox",
            type=ParameterType.PARAMETER_DOUBLE,
            description="Height of detected persons",
            read_only=False,
            floating_point_range=(FloatingPointRange(from_value=0.0, to_value=2.0, step=0.0),),
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.height_bbox, descriptor)

    def _init_parameter_height_sensor(self):
        descriptor = ParameterDescriptor(
            name="height_sensor",
            type=ParameterType.PARAMETER_DOUBLE,
            description="Height of the sensor in world frame",
            read_only=False,
            floating_point_range=(FloatingPointRange(from_value=0.0, to_value=1.0, step=0.0),),
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.height_sensor, descriptor)

    def _init_parameter_distance_nms(self):
        descriptor = ParameterDescriptor(
            name="distance_nms",
            type=ParameterType.PARAMETER_BOOL,
            description="Distance defining the neighborhood to apply non-maximum suppression in",
            read_only=False,
            floating_point_range=(FloatingPointRange(from_value=0.0, to_value=1.0, step=0.0),),
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.distance_nms, descriptor)

    def _init_parameter_threshold_confidence(self):
        descriptor = ParameterDescriptor(
            name="threshold_confidence",
            type=ParameterType.PARAMETER_DOUBLE,
            description="Confidence threshold for detections to be considered valid",
            read_only=False,
            floating_point_range=(FloatingPointRange(from_value=0.0, to_value=1.0, step=0.0),),
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.threshold_confidence, descriptor)

    def _init_parameter_use_full_fov(self):
        descriptor = ParameterDescriptor(
            name="use_full_fov",
            type=ParameterType.PARAMETER_BOOL,
            description="Consideration of scan to be a full fov 360 degree scan",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.detector.use_full_fov, descriptor)

    def _init_parameter_use_filter_region(self):
        descriptor = ParameterDescriptor(
            name="use_filter_region",
            type=ParameterType.PARAMETER_BOOL,
            description="Usage of region filter",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.use_filter_region, descriptor)

    def _init_parameter_use_transform_frame(self):
        descriptor = ParameterDescriptor(
            name="use_transform_frame",
            type=ParameterType.PARAMETER_BOOL,
            description="Usage of transform to target frame",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.use_transform_frame, descriptor)

    def _init_parameter_name_frame_target(self):
        descriptor = ParameterDescriptor(
            name="name_frame_target",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the target frame for detections",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.name_frame_target, descriptor)

    def _init_parameter_name_markers(self):
        descriptor = ParameterDescriptor(
            name="name_markers",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the markers file",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.name_markers, descriptor)

    def _init_parameter_name_region(self):
        descriptor = ParameterDescriptor(
            name="name_region",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the region for filtering",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.name_region, descriptor)

    def _init_parameter_name_topic_detections(self):
        descriptor = ParameterDescriptor(
            name="name_topic_detections",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the detections topic (for publisher)",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.name_topic_detections, descriptor)

    def _init_parameter_name_topic_scan(self):
        descriptor = ParameterDescriptor(
            name="name_topic_scan",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the 2D LiDAR scans topic (for subscriber)",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.name_topic_scan, descriptor)

    def _init_parameter_name_topic_marker(self):
        descriptor = ParameterDescriptor(
            name="name_topic_marker",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the marker topic (for publisher)",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.name_topic_marker, descriptor)

    def _init_parameter_use_service_only(self):
        descriptor = ParameterDescriptor(
            name="use_service_only",
            type=ParameterType.PARAMETER_BOOL,
            description="Usage of service-only mode",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.use_service_only, descriptor)

    # Rename call from parameter handler
    def parameter_changed(self, parameter):
        is_success, info = self.update_parameter(name=parameter.name, value=parameter.value)
        return is_success, info

    def update_parameter(self, name, value):
        try:
            func_update = getattr(self, f"update_{name}")
            is_success, info = func_update(value)
        except Exception as exception:
            is_success = False
            info = f"{exception}"
            self.get_logger().error(f"{exception}")

        return is_success, info

    def update_name_config(self, name_config):
        self.name_config = name_config
        self._init_detector()

        is_success = True
        info = ""
        return is_success, info

    def update_factor_downsampling(self, factor_downsampling):
        self.factor_downsampling = factor_downsampling
        self._init_detector()

        is_success = True
        info = ""
        return is_success, info

    def update_height_bbox(self, height_bbox):
        self.height_bbox = height_bbox

        is_success = True
        info = ""
        return is_success, info

    def update_height_sensor(self, height_sensor):
        self.height_sensor = height_sensor

        is_success = True
        info = ""
        return is_success, info

    def update_distance_nms(self, distance_nms):
        self.distance_nms = distance_nms
        self._init_detector()

        is_success = True
        info = ""
        return is_success, info

    def update_threshold_confidence(self, threshold_confidence):
        self.threshold_confidence = threshold_confidence
        self._init_detector()

        is_success = True
        info = ""
        return is_success, info

    def update_use_full_fov(self, use_full_fov):
        self.use_full_fov = use_full_fov
        self._init_detector()

        is_success = True
        info = ""
        return is_success, info

    def update_use_filter_region(self, use_filter_region):
        self.use_filter_region = use_filter_region

        is_success = True
        info = ""
        return is_success, info

    def update_use_transform_frame(self, use_transform_frame):
        self.use_transform_frame = use_transform_frame

        is_success = True
        info = ""
        return is_success, info

    def update_name_frame_target(self, name_frame_target):
        self.name_frame_target = name_frame_target

        is_success = True
        info = ""
        return is_success, info

    def update_name_markers(self, name_markers):
        self._del_client_markers()
        self.name_markers = name_markers
        self._init_client_markers()

        is_success = True
        info = ""
        return is_success, info

    def update_name_region(self, name_region):
        self.name_region = name_region

        is_success = True
        info = ""
        return is_success, info

    def update_name_topic_scan(self, name_topic_scan):
        self.name_topic_scan = name_topic_scan

        is_success = True
        info = ""
        return is_success, info

    def update_name_topic_detections(self, name_topic_detections):
        self._del_publishers()
        self.name_topic_detections = name_topic_detections
        self._init_publishers()

        is_success = True
        info = ""
        return is_success, info

    def update_name_topic_marker(self, name_topic_marker):
        self._del_publishers()
        self.name_topic_marker = name_topic_marker
        self._init_publishers()

        is_success = True
        info = ""
        return is_success, info

    def update_use_service_only(self, use_service_only):
        self._del_subscribers()
        self.use_service_only = use_service_only
        self._init_subscribers()

        is_success = True
        info = ""
        return is_success, info
