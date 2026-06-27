"""Runtime constants for the GO2 leg-entanglement detector.

Self-contained (no dependency on the research package). Mirrors the channel
layout / ordering the TCN was trained with. Python 3.8 compatible.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

LEG_ORDER = ("FR", "FL", "RR", "RL")          # type: Tuple[str, ...]
JOINT_ORDER = ("hip", "thigh", "calf")        # type: Tuple[str, ...]
CSV_FOOT_ORDER = ("FL", "FR", "RL", "RR")     # order of foot_* channels
# Unitree GO2 /lowstate motor_state indices
MOTOR_INDEX = {"FR": (0, 1, 2), "FL": (3, 4, 5), "RR": (6, 7, 8), "RL": (9, 10, 11)}  # type: Dict[str, Tuple[int, int, int]]
IMU_COLS = ("roll", "pitch", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z")  # yaw dropped
EFFORT_DQ_EPSILON = 0.05

TARGET_HZ = 500
WINDOW_SAMPLES = 200          # 0.40 s @ 500 Hz
N_LEGS = 4


def raw_channel_names():
    # type: () -> List[str]
    """48 raw channels: 36 motor (leg-major, joint, q/dq/tau) + 4 foot + 8 IMU."""
    names = []  # type: List[str]
    for leg in LEG_ORDER:
        for joint in JOINT_ORDER:
            for kind in ("q", "dq", "tau"):
                names.append("{}_{}_{}".format(leg, joint, kind))
    for leg in CSV_FOOT_ORDER:
        names.append("foot_{}".format(leg))
    names.extend(IMU_COLS)
    return names


def engineered_channel_names():
    # type: () -> List[str]
    """12 engineered channels: per-leg thigh_tau_down, thigh_down_effort, tau_sum."""
    names = []  # type: List[str]
    for leg in LEG_ORDER:
        names.append("{}_thigh_tau_down".format(leg))
        names.append("{}_thigh_down_effort".format(leg))
        names.append("{}_tau_sum".format(leg))
    return names


def channel_names():
    # type: () -> List[str]
    return raw_channel_names() + engineered_channel_names()


N_RAW = len(raw_channel_names())      # 48
N_CHANNELS = len(channel_names())     # 60
