#!/usr/bin/env python3
"""Standalone Unitree SDK2 worker — performs a single front jump on command.

Runs as a persistent subprocess of ``recovery_node``. It initialises the Unitree
SDK2 ``SportClient`` once at startup (so the DDS handshake is done before the
first alarm), then waits for commands on stdin. This process must be launched
with a CLEANED environment (no ``CYCLONEDDS_URI`` / ``RMW_IMPLEMENTATION`` /
``FASTRTPS_*``): the SDK brings up its own CycloneDDS participant on the robot's
internal interface, and the ROS 2 middleware env vars would otherwise block
``ChannelFactoryInitialize``.

Usage:
    python3 sport_worker.py [network_interface]

stdin protocol (one command per line):
    jump   -> call SportClient.FrontJump() exactly once, report the return code
    quit   -> exit cleanly

stdout protocol (one line per event):
    READY        - SDK initialised; parent may now send commands
    ERROR: msg   - init failed; process exits with code 1
    DONE <code>  - FrontJump() returned <code> (0 = accepted by the robot)
"""
import sys


def main():
    network_interface = sys.argv[1] if len(sys.argv) > 1 else ""

    try:
        import unitree_sdk2py.core.channel as channel
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        channel.ChannelFactoryInitialize(0, network_interface)
        client = SportClient()
        client.SetTimeout(10.0)
        client.Init()
    except Exception as exc:  # SDK missing / DDS init failed / wrong interface
        sys.stdout.write("ERROR: {}\n".format(exc))
        sys.stdout.flush()
        sys.exit(1)

    sys.stdout.write("READY\n")
    sys.stdout.flush()

    # Blocking line-reader. FrontJump is a one-shot trigger, so there is no
    # streaming loop here: we call it exactly once per "jump" line and never
    # repeat it, which is what keeps the robot from jumping in a loop.
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "jump":
            try:
                code = client.FrontJump()
            except Exception as exc:
                sys.stdout.write("ERROR: FrontJump raised {}\n".format(exc))
                sys.stdout.flush()
                continue
            sys.stdout.write("DONE {}\n".format(code))
            sys.stdout.flush()
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
