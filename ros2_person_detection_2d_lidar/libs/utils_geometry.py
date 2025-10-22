import numpy as np
import torch


def cartesian_cutout_to_polar_sensor_numpy(rs_cutout, phis_cutout, xs, ys):
    """
    Apply local Cartesian offsets to polar coordinates of cutouts.

    Offsets are defined in each cutouts's local [forward, left] frame.
    Returns updated (rs, phis) in the global sensor frame.
    """
    ys_tmp = rs_cutout + ys
    # This is correct due to rotated reference frame, it gives the angle from the local forward axis to the offset vector
    phis_tmp = np.arctan2(xs, ys_tmp)

    rs = ys_tmp / np.cos(phis_tmp)
    phis = phis_cutout + phis_tmp

    return rs, phis


def cartesian_cutout_to_polar_sensor(rs_cutout, phis_cutout, xs, ys):
    """
    Apply local Cartesian offsets to polar coordinates of cutouts.

    Offsets are defined in each cutouts's local [forward, left] frame.
    Returns updated (rs, phis) in the global sensor frame.
    """
    ys_tmp = rs_cutout + ys
    # This is correct due to rotated reference frame, it gives the angle from the local forward axis to the offset vector
    phis_tmp = torch.arctan2(xs, ys_tmp)

    rs = ys_tmp / torch.cos(phis_tmp)
    phis = phis_cutout + phis_tmp

    return rs, phis


def polar_to_cartesian_numpy(rs, phis):
    xs = rs * np.cos(phis)
    ys = rs * np.sin(phis)
    return xs, ys


def polar_to_cartesian(rs, phis):
    xs = rs * torch.cos(phis)
    ys = rs * torch.sin(phis)
    return xs, ys


def complex_to_quaternion(real, imag):
    # Assume rotation in xy-plane
    quaternion = np.array([0.0, 0.0, real, imag])
    return quaternion
