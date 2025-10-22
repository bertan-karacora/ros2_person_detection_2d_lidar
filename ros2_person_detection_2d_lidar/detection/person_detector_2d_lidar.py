import math
from pathlib import Path

import numpy as np
import torch

import ros2_person_detection_2d_lidar.config as config
from ros2_person_detection_2d_lidar.detection.detection import Detection
import ros2_person_detection_2d_lidar.libs.utils_import as utils_import
import ros2_person_detection_2d_lidar.libs.utils_geometry as utils_geometry


class Detector:
    def __init__(self, name_config, factor_downsampling=1, threshold_confidence=0.5, use_full_fov=False, distance_nms=0.5):
        self.device = None
        self.distance_nms = distance_nms
        self.factor_downsampling = factor_downsampling
        self.name_config = name_config
        self.threshold_confidence = threshold_confidence
        self.use_full_fov = use_full_fov

        self._init()

    def _init(self):
        config.apply_config_preset(self.name_config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._init_model()

    @torch.inference_mode()
    def _init_model(self):
        class_model = utils_import.import_model(config.MODEL["name"])
        self.model = class_model(
            shape_input=config.MODEL["shape_input"],
            use_full_fov=self.use_full_fov,
            **config.MODEL["kwargs"],
        )
        self.model.eval()

        name_checkpoint = config.CHECKPOINT["name"]
        path_checkpoint = Path(config._PATH_DIR_CHECKPOINTS) / f"{name_checkpoint}.pth"
        checkpoint = torch.load(path_checkpoint, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model = self.model.to(self.device)

    def preprocess(self, scan):
        # Convert to tensor
        ranges_scan = torch.from_numpy(np.ascontiguousarray(scan["ranges"]))[None, :]
        angles_scan = torch.from_numpy(np.ascontiguousarray(scan["angles"]))[None, :]

        # Replace invalid input
        ranges_scan = replace_invalid(ranges_scan, **config.DATA["inference"]["transform"]["transforms"][0]["kwargs"])

        # Convert to dtype
        # NOTE: Original implementation uses double
        ranges_scan = ranges_scan.float()
        angles_scan = angles_scan.float()

        # Downsample
        ranges_scan_downsampled = downsample(ranges_scan, self.factor_downsampling)
        angles_scan_downsampled = downsample(angles_scan, self.factor_downsampling)

        # Cutout
        cutouts = cutout_scans(ranges_scan, ranges_scan_downsampled, angles_scan, angles_scan_downsampled, **config.DATA["inference"]["transform"]["transforms"][2]["kwargs"])

        return cutouts, ranges_scan_downsampled, angles_scan_downsampled

    @torch.inference_mode()
    def __call__(self, scan):
        cutouts, ranges_scan_downsampled, angles_scan_downsampled = self.preprocess(scan)

        # Move to GPU
        cutouts = cutouts.to(self.device)
        ranges_scan_downsampled = ranges_scan_downsampled.to(self.device)
        angles_scan_downsampled = angles_scan_downsampled.to(self.device)

        with torch.autocast(device_type=self.device.type, dtype=torch.half):
            output = self.model(cutouts.permute((2, 0, 1))[None, :, :, :])

            # Remove batch dimension
            for key in output:
                output[key] = output[key][0]
            ranges_scan_downsampled = ranges_scan_downsampled[0]
            angles_scan_downsampled = angles_scan_downsampled[0]

            # Assume single class
            output["confidence"] = output["confidence"][:, 0]

            # Threshold confidence
            mask_confidence = output["confidence"] >= self.threshold_confidence
            for key in output:
                output[key] = output[key][mask_confidence]
            ranges_scan_downsampled = ranges_scan_downsampled[mask_confidence]
            angles_scan_downsampled = angles_scan_downsampled[mask_confidence]

            # Transform from cutout frame to sensor frame
            rs, phis = utils_geometry.cartesian_cutout_to_polar_sensor(rs_cutout=ranges_scan_downsampled, phis_cutout=angles_scan_downsampled, xs=output["position"][:, 0], ys=output["position"][:, 1])
            output["position"][:, 0], output["position"][:, 1] = utils_geometry.polar_to_cartesian(rs, phis)

        # Move to CPU
        for key in output:
            output[key] = output[key].detach().cpu().float()

        # Sort by descending confidence
        idxs = torch.argsort(output["confidence"], descending=True)
        for key in output:
            output[key] = output[key][idxs]

        # Apply non-maximum suppression
        mask_nms = nms_position(output["position"], distance_min=self.distance_nms)
        for key in output:
            output[key] = output[key][mask_nms]

        # Normalize orientation
        if "bbox" in output:
            output["bbox"][:, 2:] /= torch.linalg.norm(output["bbox"][:, 2:], axis=1, keepdims=True)
            # I observed that the regressed values for the size do not make sense (they are very small and some are even negative).
            # The code for the BEV bbox predicting model was added when the paper on the comparison of 2d and 3d LiDAR-based person detection was published.
            # In this paper, they compare DRSPAAM with Centerpoint.
            # I read somewhere in the Centerpoint paper that they output the size in logarithmic scale.
            # As a guess, I tried to apply the xponential function here and the result seems very reasonable now.
            output["bbox"][:, :2] = torch.exp(output["bbox"][:, :2])

        # Sort by distance
        distances_detections = torch.linalg.norm(output["position"], axis=1)
        idxs = distances_detections.argsort()
        for key in output:
            output[key] = output[key][idxs]

        # To numpy
        for key in output:
            output[key] = output[key].numpy().astype(float)

        # Create detections
        detections = []
        if not "bbox" in output:
            output["bbox"] = np.tile([0.0, 0.0, 0.0, 1.0], (len(output["confidence"]), 1))
        for position, confidence, bbox in zip(output["position"], output["confidence"], output["bbox"]):
            detection = Detection(
                position=position,
                size=bbox[:2],
                orientation=bbox[2:],
                confidence=confidence,
                label="person",
            )
            detections.append(detection)

        return detections


def replace_invalid_numpy(input, distance_padding=29.99, inplace=False):
    if not inplace:
        input = np.copy(input)
    output = input

    invalid_mask = np.logical_or.reduce((output < 0.0, np.isinf(output), np.isnan(output)))
    output[invalid_mask] = distance_padding

    return output


def replace_invalid(input, distance_padding=29.99, inplace=False):
    if not inplace:
        input = input.clone()
    output = input

    invalid_mask = (output < 0.0) | torch.isinf(output) | torch.isnan(output)
    output.masked_fill_(invalid_mask, distance_padding)

    return output


def downsample_numpy(input, factor):
    output = input[::factor]
    return output


def downsample(input, factor):
    output = input[:, ::factor]
    return output


def cutout_scans_numpy(ranges_scan, phis_scan, factor_downsampling=1, width_window=1.66, depth_window=1.0, num_points=48, distance_padding=29.99, use_centered=True, use_fixed=False, use_area_downsampling=False):
    # Size (width) of the window
    dists = ranges_scan[:, ::factor_downsampling] if use_fixed else np.tile(ranges_scan[-1, ::factor_downsampling], ranges_scan.size(0)).reshape(ranges_scan.size(0), -1)
    half_alpha = np.arctan(0.5 * width_window / np.maximum(dists, 1e-2))

    # Cutout indices
    delta_alpha = 2.0 * half_alpha / (num_points - 1)
    ang_ct = phis_scan[::factor_downsampling] - half_alpha + np.arange(num_points).reshape(num_points, 1, 1) * delta_alpha
    # Warp angle
    ang_ct = (ang_ct + np.pi) % (2.0 * np.pi) - np.pi
    inds_ct = (ang_ct - phis_scan[0]) / (phis_scan[1] - phis_scan[0])
    outbound_mask = np.logical_or(inds_ct < 0, inds_ct > ranges_scan.size(1) - 1)

    # Cutout (linear interp)
    inds_ct_low = np.clip(np.floor(inds_ct), 0, ranges_scan.size(1) - 1).astype(int)
    inds_ct_high = np.clip(inds_ct_low + 1, 0, ranges_scan.size(1) - 1).astype(int)
    inds_ct_ratio = np.clip(inds_ct - inds_ct_low, 0.0, 1.0)
    # Because np.take flattens array
    inds_offset = np.arange(ranges_scan.size(0)).reshape(1, ranges_scan.size(0), 1) * ranges_scan.size(1)
    ct_low = np.take(ranges_scan, inds_ct_low + inds_offset)
    ct_high = np.take(ranges_scan, inds_ct_high + inds_offset)
    ct = ct_low + inds_ct_ratio * (ct_high - ct_low)

    # Use area sampling for down-sampling (close points)
    if use_area_downsampling:
        num_in_window = inds_ct[-1] - inds_ct[0]
        area_mask = num_in_window > num_points
        if np.sum(area_mask) > 0:
            # Sample the window with more points than the actual number of points
            s_area = int(math.ceil(np.max(num_in_window) / num_points))
            num_ct_pts_area = s_area * num_points
            delta_alpha_area = 2.0 * half_alpha / (num_ct_pts_area - 1)
            ang_ct_area = phis_scan[::factor_downsampling] - half_alpha + np.arange(num_ct_pts_area).reshape(num_ct_pts_area, 1, 1) * delta_alpha_area
            # Warp angle
            ang_ct_area = (ang_ct_area + np.pi) % (2.0 * np.pi) - np.pi

            inds_ct_area = (ang_ct_area - phis_scan[0]) / (phis_scan[1] - phis_scan[0])
            inds_ct_area = np.rint(np.clip(inds_ct_area, 0, ranges_scan.size(1) - 1)).astype(np.int32)
            ct_area = np.take(ranges_scan, inds_ct_area + inds_offset)
            ct_area = ct_area.reshape(num_points, s_area, ranges_scan.size(0), dists.size(1)).mean(axis=1)
            ct[:, area_mask] = ct_area[:, area_mask]

    # Normalize cutout
    ct[outbound_mask] = distance_padding
    ct = np.clip(ct, dists - depth_window, dists + depth_window)
    if use_centered:
        ct = ct - dists
        ct = ct / depth_window

    return ct.transpose((1, 0, 2))


def cutout_scans(ranges_scan, ranges_scan_downsampled, angles_scan, angles_scan_downsampled, width_window=1.66, depth_window=1.0, num_points=48, distance_padding=29.99, use_centered=True, use_area_downsampling=False):
    # Precompute angles per window
    half_alpha = torch.atan(0.5 * width_window / torch.clamp(ranges_scan_downsampled, min=1e-2))
    delta_alpha = 2.0 * half_alpha / (num_points - 1)
    angles_cutout = angles_scan_downsampled - half_alpha + torch.arange(num_points, device=ranges_scan.device)[:, None] * delta_alpha
    angles_cutout = (angles_cutout + math.pi) % (2.0 * math.pi) - math.pi

    # Map to index space
    angle_step = angles_scan[0, 1] - angles_scan[0, 0]
    idxs_cutout = (angles_cutout - angles_scan[0, 0]) / angle_step
    idxs_low = torch.floor(idxs_cutout).clamp(0, ranges_scan.size(1) - 1).long()
    idxs_high = (idxs_low + 1).clamp(0, ranges_scan.size(1) - 1)
    alpha = (idxs_cutout - idxs_low).clamp(0.0, 1.0)

    # Gather range values
    vals_low = ranges_scan[:, idxs_low]
    vals_high = ranges_scan[:, idxs_high]
    cutouts = vals_low + alpha * (vals_high - vals_low)

    if use_area_downsampling:
        num_in_window = idxs_cutout[-1, :] - idxs_cutout[0, :]
        area_mask = num_in_window > num_points
        if area_mask.any():
            # Compute the area sampling scale factor
            s_area = int(math.ceil(num_in_window[area_mask].max().item() / num_points))
            num_ct_pts_area = s_area * num_points
            delta_alpha_area = 2.0 * half_alpha / (num_ct_pts_area - 1)

            angles_cutout_area = angles_scan_downsampled - half_alpha + torch.arange(num_ct_pts_area, device=angles_scan.device)[:, None] * delta_alpha_area
            angles_cutout_area = (angles_cutout_area + math.pi) % (2.0 * math.pi) - math.pi

            idxs_cutout_area = ((angles_cutout_area - angles_scan[0, 0]) / angle_step).clamp(0, ranges_scan.size(1) - 1).round().long()
            vals_low_area = ranges_scan[:, idxs_cutout_area]
            cutouts_area = vals_low_area.view(num_points, s_area, -1).mean(dim=1)[None, :, :]

            cutouts[:, :, area_mask] = cutouts_area[:, :, area_mask]

    # Out-of-bounds padding
    mask_oob = (idxs_cutout < 0) | (idxs_cutout > ranges_scan.size(1) - 1)
    cutouts = torch.where(mask_oob, distance_padding, cutouts)

    # Normalize
    min_val = (ranges_scan_downsampled - depth_window)[:, None]
    max_val = (ranges_scan_downsampled + depth_window)[:, None]
    cutouts = torch.clamp(cutouts, min_val, max_val)
    if use_centered:
        cutouts -= ranges_scan_downsampled[:, None]
        cutouts /= depth_window

    return cutouts


def nms_position_numpy(positions, distance_min=0.5):
    distances = np.sqrt(np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=-1))

    mask_keep = np.ones(positions.size(0), dtype=bool)
    for i in range(positions.size(0)):
        if not mask_keep[i]:
            continue

        mask_keep &= distances[i] >= distance_min
        mask_keep[i] = True

    return mask_keep


def nms_position(positions, distance_min=0.5):
    distances = torch.cdist(positions, positions, p=2)

    mask_keep = torch.ones(positions.size(0), device=positions.device, dtype=torch.bool)
    for i in range(positions.size(0)):
        if not mask_keep[i]:
            continue

        mask_is_close = distances[i] < distance_min
        mask_is_close[i] = False
        mask_keep[mask_is_close] = False

    return mask_keep
