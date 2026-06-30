"""Verified Unitree Go2 high-level Sport API constants + ROS 2 topic names.

Source-verified 2026-06-30 against unitree_ros2 `ros2_sport_client.h` (master) and the
unitree_api `.msg` definitions. Pure module (no ROS imports) so it is safe to import in the
unit-testable FSM and in tests. Do NOT reference api_ids 1011-1014 — they were removed from
the V2 motion interface and do not exist in the current header.
"""
from __future__ import annotations

# --- high-level Sport API ids (int64 api_id in unitree_api/Request) ---
SPORT_API_ID = {
    "DAMP": 1001,           # soft e-stop: release joint torque (limp)
    "BALANCE_STAND": 1002,  # actively-balanced upright stance (presumes upright)
    "STOP_MOVE": 1003,      # halt locomotion in place
    "STAND_UP": 1004,       # prone -> stand
    "STAND_DOWN": 1005,     # stand -> prone
    "RECOVERY_STAND": 1006,  # right the body from a FALLEN state (face up/down)
    "MOVE": 1008,           # velocity command (must be streamed)
    "SIT": 1009,
    "RISE_SIT": 1010,
}

# --- ROS 2 topics (configurable in recovery.yaml; these are the documented defaults) ---
TOPIC_SPORT_REQUEST = "/api/sport/request"     # unitree_api/msg/Request  (publish)
TOPIC_SPORT_RESPONSE = "/api/sport/response"   # unitree_api/msg/Response (subscribe)
TOPIC_SPORT_MODE_STATE = "/sportmodestate"     # unitree_go/msg/SportModeState (subscribe)
TOPIC_LOW_STATE = "/lowstate"                  # unitree_go/msg/LowState (subscribe, optional)

# --- SportModeState.mode integer mapping (README-verified) ---
MODE = {
    0: "idle",          # default stand
    1: "balanceStand",
    2: "pose",
    3: "locomotion",
    5: "lieDown",
    6: "jointLock",
    7: "damping",
    8: "recoveryStand",
    10: "sit",
    11: "frontFlip",
    12: "frontJump",
    13: "frontPounce",
}
# modes that mean "the robot is NOT a normally-upright, ready stance"
MODE_FALLEN = {5}                      # lieDown
MODE_LOCOMOTION = 3
MODE_UPRIGHT_STABLE = {0, 1}           # idle / balanceStand
MODE_RECOVERY_IN_PROGRESS = {8}        # recoveryStand running

# success return code on /api/sport/response (header.status.code)
RESP_OK = 0
