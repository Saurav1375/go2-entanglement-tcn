#!/usr/bin/env python3
"""Standalone Unitree SDK2 worker — runs a single recovery sequence on command.

Runs as a persistent subprocess of ``recovery_node``. It initialises the Unitree
SDK2 ``SportClient`` once at startup (so the DDS handshake is done before the
first alarm), then waits for commands on stdin. This process MUST be launched
with a CLEANED environment (no ``CYCLONEDDS_URI`` / ``RMW_IMPLEMENTATION`` /
``FASTRTPS_*``): the SDK brings up its own CycloneDDS participant on the robot's
internal interface, and the ROS 2 middleware env vars would otherwise block
``ChannelFactoryInitialize``.

Recovery sequence (executed exactly once per ``recover`` line):
    StopMove  ->  Move backward for `back_duration` s  ->  StopMove  ->  FrontJump  ->  StopMove
Each step is a discrete Sport-API call (StopMove is NOT streamed, which avoids the
jitter seen when StopMove is spammed); only the backward Move is streamed, because
Move sets a velocity that must be refreshed to keep the robot walking.

Usage:
    python3 sport_worker.py [network_interface] [back_speed_mps] [back_duration_s]

stdin protocol (one command per line):
    recover  -> run the sequence above exactly once, then report the result
    quit     -> exit cleanly

stdout protocol (one line per event):
    READY          - SDK initialised; parent may now send commands
    ERROR: msg     - init or a step failed
    DONE <codes>   - sequence finished; <codes> = return codes of the API calls
"""
import sys
import time

# Fixed settle times (seconds). Small, discrete pauses between steps so each motion
# lands before the next begins; tuned to be gentle rather than abrupt.
SETTLE_AFTER_STOP = 0.3      # after the initial StopMove, before backing up
SETTLE_AFTER_BACK = 0.4      # after backing up + StopMove, before the jump
JUMP_DURATION = 3.0          # wait for the front jump to physically complete before the final StopMove
MOVE_DT = 0.05               # backward-Move refresh period (~20 Hz)


def main():
    iface = sys.argv[1] if len(sys.argv) > 1 else ""
    back_speed = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3     # m/s (magnitude)
    back_duration = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5  # s

    try:
        import unitree_sdk2py.core.channel as channel
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        channel.ChannelFactoryInitialize(0, iface)
        client = SportClient()
        client.SetTimeout(10.0)
        client.Init()
    except Exception as exc:  # SDK missing / DDS init failed / wrong interface
        sys.stdout.write("ERROR: {}\n".format(exc))
        sys.stdout.flush()
        sys.exit(1)

    sys.stdout.write("READY\n")
    sys.stdout.flush()

    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "recover":
            try:
                codes = _run_sequence(client, back_speed, back_duration)
            except Exception as exc:
                sys.stdout.write("ERROR: recovery raised {}\n".format(exc))
                sys.stdout.flush()
                continue
            sys.stdout.write("DONE {}\n".format(",".join(str(c) for c in codes)))
            sys.stdout.flush()
        elif cmd == "quit":
            break


def _run_sequence(client, back_speed, back_duration):
    """stop -> move back -> stop -> front jump -> stop. Runs once; never loops."""
    codes = []
    codes.append(client.StopMove())                    # 1. halt whatever gait is running
    time.sleep(SETTLE_AFTER_STOP)

    t_end = time.monotonic() + max(0.0, back_duration)  # 2. back up (Move must be streamed)
    while time.monotonic() < t_end:
        client.Move(-abs(back_speed), 0.0, 0.0)         # vx<0 = backward (body frame)
        time.sleep(MOVE_DT)

    codes.append(client.StopMove())                    # 3. stop the backward motion
    time.sleep(SETTLE_AFTER_BACK)

    codes.append(client.FrontJump())                   # 4. the front jump (once)
    time.sleep(JUMP_DURATION)                          # let the jump physically complete

    codes.append(client.StopMove())                    # 5. final stop
    return codes


if __name__ == "__main__":
    main()
