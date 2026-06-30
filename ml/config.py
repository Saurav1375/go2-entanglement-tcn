"""Single source of truth for the leg-entanglement ML pipeline.

All leg/joint/foot ordering constants follow the Unitree GO2 `/lowstate` convention
(motor index FR(0-2), FL(3-5), RR(6-8), RL(9-11); foot order FL,FR,RL,RR) so the
offline pipeline and a future live `/lowstate` node share one contract.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------- paths
ML_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(ML_DIR)
CSV_LABELLED_DIR = os.path.join(PROJECT_DIR, "csv_labelled")
ARTIFACTS_DIR = os.path.join(ML_DIR, "artifacts")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")

# ---------------------------------------------------------------- ordering (mirror reuse-source)
LEG_ORDER = ("FR", "FL", "RR", "RL")
JOINT_ORDER = ("hip", "thigh", "calf")
CSV_FOOT_ORDER = ("FL", "FR", "RL", "RR")            # order of foot_* columns in the CSV
MOTOR_INDEX = {"FR": (0, 1, 2), "FL": (3, 4, 5), "RR": (6, 7, 8), "RL": (9, 10, 11)}
EFFORT_DQ_EPSILON = 0.05                              # reuse-source value

# IMU columns kept (yaw dropped: unbounded integrator)
IMU_COLS = ("roll", "pitch", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z")
STATUS_COL = "Status"

# ---------------------------------------------------------------- channel-set flags
# (mutated by ablation.py; defaults reproduce the shipped raw+engineered, C=60)
INCLUDE_FOOT = True       # 4 foot-force channels
INCLUDE_IMU = True        # 8 IMU channels (roll,pitch,gyro xyz,acc xyz)
ENGINEERED_ONLY = False   # if True: drop all raw channels, keep only the 12 engineered

# ---------------------------------------------------------------- resample / window
TARGET_HZ = 500
WINDOW_SAMPLES = 200          # 0.40 s @ 500 Hz
HOP_TRAIN = 25                # 0.05 s, 87.5% overlap
HOP_EVAL = 1                  # dense replay
POS_FRACTION = 0.5            # window is "entangled" if >= this fraction of rows are Entangled
TRANSITION_LOW = 0.2          # windows with entangled-fraction in (LOW, POS_FRACTION) excluded from TRAINING
USE_ENGINEERED = True         # add 12 engineered physics channels -> C=60 (False -> C=48 ablation)

# ---------------------------------------------------------------- model
ENCODER_CHANNELS = 64
TCN_DILATIONS = (1, 2, 4, 8, 16)
TCN_KERNEL = 3
TCN_DROPOUT = 0.1
N_LEGS = 4

# loss weights
W_BIN = 1.0
W_LEGS = 1.0
W_INTENSITY = 0.3
POS_WEIGHT_CAP = 10.0

# ---------------------------------------------------------------- training
SEED = 1234
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda"               # falls back to cpu in code if unavailable

# 3-window persistence/debounce mirrors the heuristic (DEFAULT_PERSISTENCE_WINDOWS)
PERSISTENCE_WINDOWS = 3

# ---------------------------------------------------------------- filename -> affected legs
# back_ -> rear pool (RR/RL); front_ -> front pool (FR/FL)
# _left_ -> left of pair (RL/FL); _right_ -> right of pair (RR/FR); _both_ -> both
def parse_legs(filename: str) -> set[str]:
    """Return the set of entangled legs implied by a recording's filename.

    Walking/Stop (and anything without a position+side token) -> empty set.
    """
    stem = os.path.basename(filename)
    stem = stem[:-4] if stem.endswith(".csv") else stem
    stem = stem.lower()
    if "back" in stem:
        left, right = "RL", "RR"
    elif "front" in stem:
        left, right = "FL", "FR"
    else:
        return set()                       # walking / stop
    if "both" in stem:
        return {left, right}
    if "left" in stem:
        return {left}
    if "right" in stem:
        return {right}
    return set()


# ---------------------------------------------------------------- fixed leakage-safe split
# v2 adds 5 GO2 deployment recordings (Lock/Stop/backward-walk + a dedicated front-right
# entanglement). Each recording lives in exactly one split (leakage-safe, grouped by file).
#   front_right_wire -> TRAIN  (so the SHIPPED model learns dedicated FR; held-out FR via LORO)
#   walk_stop_back   -> TRAIN  (teaches backward-Walking + Lock + Stop negatives)
#   back_right_intensity -> TRAIN (RR positive whose post-event Lock is a key negative)
#   back_right_stop  -> VAL    (RR positive; entanglement-while-standing case)
#   lock_stop        -> TEST   (held-out pure stand-up->Lock: clean Lock false-alarm measure)
SPLIT = {
    "test": ["back_left_hand2", "back_right_wire1", "front_left_hand2", "front_both_wire2",
             "go2_lowstate_lock_stop"],
    "val":  ["back_both_wire2", "back_right_hand2", "walking4",
             "go2_lowstate_entaglement_back_right_stop"],
    "train": ["back_both_wire1", "back_left_hand1", "back_left_wire1", "back_right_hand1",
              "front_both_wire1", "front_left_hand1",
              "stop1", "stop2", "stop3", "walking1", "walking2", "walking3", "walking5",
              "go2_lowstate_entaglement_front_right_wire",
              "go2_lowstate_entaglement_back_right_intensity",
              "go2_lowstate_walk_stop_back"],
}

# The 15 positive recordings (one contiguous Entangled event each) for LORO-CV.
# (The 3 new positives add the first dedicated front-right event and two RR events.)
POSITIVE_FILES = [
    "back_both_wire1", "back_both_wire2", "back_left_hand1", "back_left_hand2",
    "back_left_wire1", "back_right_hand1", "back_right_hand2", "back_right_wire1",
    "front_both_wire1", "front_both_wire2", "front_left_hand1", "front_left_hand2",
    "go2_lowstate_entaglement_front_right_wire",
    "go2_lowstate_entaglement_back_right_stop",
    "go2_lowstate_entaglement_back_right_intensity",
]


# ---------------------------------------------------------------- channel layout helpers
def raw_signal_columns() -> list[str]:
    """All raw CSV signal columns that ever load from disk (36 motor + 4 foot + 8 IMU).

    This is fixed regardless of ablation flags — io_load/resample validate against it.
    """
    names: list[str] = []
    for leg in LEG_ORDER:
        for joint in JOINT_ORDER:
            for kind in ("q", "dq", "tau"):
                names.append(f"{leg}_{joint}_{kind}")
    for leg in CSV_FOOT_ORDER:
        names.append(f"foot_{leg}")
    names.extend(IMU_COLS)
    return names


def raw_channel_names() -> list[str]:
    """Raw model-input channels, honoring the ablation flags (motors always kept)."""
    names: list[str] = []
    for leg in LEG_ORDER:
        for joint in JOINT_ORDER:
            for kind in ("q", "dq", "tau"):
                names.append(f"{leg}_{joint}_{kind}")
    if INCLUDE_FOOT:
        for leg in CSV_FOOT_ORDER:
            names.append(f"foot_{leg}")
    if INCLUDE_IMU:
        names.extend(IMU_COLS)
    return names


def engineered_channel_names() -> list[str]:
    """12 engineered channels: per-leg thigh_tau_down, thigh_down_effort, tau_sum."""
    names: list[str] = []
    for leg in LEG_ORDER:
        names.append(f"{leg}_thigh_tau_down")
        names.append(f"{leg}_thigh_down_effort")
        names.append(f"{leg}_tau_sum")
    return names


def channel_names() -> list[str]:
    if ENGINEERED_ONLY:
        return engineered_channel_names()
    names = raw_channel_names()
    if USE_ENGINEERED:
        names = names + engineered_channel_names()
    return names


def n_channels() -> int:
    return len(channel_names())
