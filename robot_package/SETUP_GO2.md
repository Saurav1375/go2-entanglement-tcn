# GO2 Setup Instructions

Step-by-step deployment of the leg-entanglement detector on a Unitree GO2
(Python 3.8, ROS 2, CPU-only).

> **Deploying the full pipeline (detector + recovery)?** Use **[`RUNBOOK.md`](RUNBOOK.md)** — it
> covers all three packages, the network/DDS setup, and the staged bring-up (the recovery runs the
> one-shot sequence stop → back → stop → front jump → stop). This file covers the detector alone.

## 0. Assumptions
- The GO2 runs ROS 2 (e.g. Foxy on Ubuntu 20.04, Python 3.8) and publishes
  `unitree_go/msg/LowState` on `/lowstate`.
- You can `ssh` onto the robot's onboard computer and it has internet or a way to
  install Python wheels.

## 1. Copy the package to the robot
Copy the ROS 2 packages into a colcon workspace `src/` (include `entanglement_recovery` if you
also want the recovery node — see RUNBOOK.md):
```bash
# on your dev machine
scp -r robot_package/src/entanglement_interfaces   unitree@<go2-ip>:~/ros2_ws/src/
scp -r robot_package/src/entanglement_detector      unitree@<go2-ip>:~/ros2_ws/src/
scp -r robot_package/src/entanglement_recovery      unitree@<go2-ip>:~/ros2_ws/src/   # recovery (optional)
scp robot_package/requirements_robot.txt            unitree@<go2-ip>:~/ros2_ws/
```
Only `robot_package/src/**` is needed on the robot. `export_model.py` and `tools/`
are dev-side and can stay on your machine.

## 2. Install Python runtime deps
Install into the **same Python interpreter that runs ROS 2** (usually system python3):
```bash
ssh unitree@<go2-ip>
cd ~/ros2_ws
pip3 install -r requirements_robot.txt
```
- **numpy** is pinned `<1.25` for Python 3.8.
- **onnxruntime (CPU)**: the `pip` wheel works on x86_64 and most aarch64.
  - On **NVIDIA Jetson** (some GO2 variants): if the pip wheel is unavailable, install the
    Jetson-specific ONNX Runtime wheel from NVIDIA, **or** switch to the TorchScript backend:
    set `model_path: models/entanglement_tcn_ts.pt` in `config.yaml` and `pip3 install torch`
    (use the Jetson torch build).

## 3. Build
```bash
source /opt/ros/<distro>/setup.bash        # e.g. foxy
cd ~/ros2_ws
colcon build --packages-select entanglement_interfaces entanglement_detector entanglement_recovery
source install/setup.bash
```
Build `entanglement_interfaces` first (the detector depends on its message); the single
`colcon build` above handles ordering automatically.

## 4. Verify the message is registered
```bash
ros2 interface show entanglement_interfaces/msg/EntanglementState
```

## 5. Run
```bash
ros2 launch entanglement_detector entanglement.launch.py
# or directly:
ros2 run entanglement_detector entanglement_node --ros-args --params-file \
    install/entanglement_detector/share/entanglement_detector/config/config.yaml
```
You should see: `entanglement_detector up: backend=entanglement_tcn.onnx ...`.
The node waits ~0.40 s (one window) before the first publish.

## 6. Observe output
```bash
ros2 topic echo /entanglement_state
ros2 topic hz   /entanglement_state      # should track the /lowstate rate (÷ publish_every_n)
```

## 7. Tuning on hardware
Edit the installed `config.yaml` (or the source copy and rebuild):
- **CPU too slow?** Increase `publish_every_n` (e.g. 2–4) to decimate inference; or keep ONNX
  (faster than TorchScript). Check load with `top`/`htop` and `ros2 topic hz`.
- **Too many false alarms while walking?** Increase `debounce_ms` (e.g. 100) and/or raise
  `detection_threshold` slightly.
- **RR over-firing?** Its per-leg threshold is already raised to 0.9; raise further if needed.

## 8. (Optional) Autostart with systemd
Create `/etc/systemd/system/entanglement.service`:
```ini
[Unit]
Description=GO2 leg-entanglement detector
After=network.target

[Service]
User=unitree
ExecStart=/bin/bash -lc 'source /opt/ros/<distro>/setup.bash && source /home/unitree/ros2_ws/install/setup.bash && ros2 launch entanglement_detector entanglement.launch.py'
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload && sudo systemctl enable --now entanglement.service
journalctl -u entanglement.service -f
```

## Troubleshooting
| symptom | fix |
|---|---|
| `ModuleNotFoundError: unitree_go` | source the robot's Unitree ROS 2 overlay before running |
| `No module named onnxruntime` | `pip3 install` into the ROS 2 interpreter (check `which python3`) |
| no `/entanglement_state` output | confirm `/lowstate` is publishing: `ros2 topic hz /lowstate` |
| high CPU / lag | increase `publish_every_n`; ensure ONNX backend; `num_threads: 1` |
| message type not found | rebuild `entanglement_interfaces` and re-`source install/setup.bash` |
