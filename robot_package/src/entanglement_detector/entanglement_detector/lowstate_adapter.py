"""Convert a Unitree GO2 `unitree_go/msg/LowState` into the engine's sample dict.

Duck-typed (no ROS import) so it can be unit-tested. Mapping follows the GO2
convention used to record the training data:
  motor_state[MOTOR_INDEX[leg]] -> {leg}_{hip|thigh|calf}_{q,dq,tau}
    tau uses tau_est (fallback tau)
  foot_force[i]      -> foot_{CSV_FOOT_ORDER[i]}
  imu_state.rpy[0,1] -> roll, pitch   (yaw dropped)
  imu_state.gyroscope[0:3]      -> gyro_x,y,z
  imu_state.accelerometer[0:3]  -> acc_x,y,z
"""
from __future__ import annotations

from typing import Dict

from . import constants as K


def _tau(motor):
    v = getattr(motor, "tau_est", None)
    if v is None:
        v = getattr(motor, "tau", 0.0)
    return float(v)


def lowstate_to_sample(msg):
    # type: (object) -> Dict[str, float]
    sample = {}  # type: Dict[str, float]
    motors = list(getattr(msg, "motor_state"))
    for leg, idx in K.MOTOR_INDEX.items():
        for joint, mi in zip(K.JOINT_ORDER, idx):
            m = motors[mi]
            sample["{}_{}_q".format(leg, joint)] = float(m.q)
            sample["{}_{}_dq".format(leg, joint)] = float(m.dq)
            sample["{}_{}_tau".format(leg, joint)] = _tau(m)

    foot = list(getattr(msg, "foot_force"))
    for i, leg in enumerate(K.CSV_FOOT_ORDER):
        sample["foot_{}".format(leg)] = float(foot[i]) if i < len(foot) else 0.0

    imu = getattr(msg, "imu_state")
    rpy = list(getattr(imu, "rpy"))
    gyro = list(getattr(imu, "gyroscope"))
    acc = list(getattr(imu, "accelerometer"))
    sample["roll"] = float(rpy[0])
    sample["pitch"] = float(rpy[1])
    sample["gyro_x"], sample["gyro_y"], sample["gyro_z"] = (float(gyro[0]), float(gyro[1]), float(gyro[2]))
    sample["acc_x"], sample["acc_y"], sample["acc_z"] = (float(acc[0]), float(acc[1]), float(acc[2]))
    return sample
