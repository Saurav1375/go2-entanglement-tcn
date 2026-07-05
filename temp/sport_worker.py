#!/usr/bin/env python3
"""Standalone stop-move daemon with standby protocol — no ROS 2.

Runs as a persistent subprocess of recovery_node. Initialises the Unitree
SDK2 SportClient immediately at startup, then waits for commands on stdin.
The parent node writes "start" when the latch activates and "stop" to return
to standby.  This means the SDK handshake happens at node startup, so there
is zero additional latency between alarm and first StopMove call.

Usage:
    python3 sport_worker.py [network_interface]

stdin protocol (one command per line):
    start   → begin sending StopMove at 100 Hz
    stop    → stop sending, return to standby (stay alive for next alarm)
    quit    → exit cleanly

stdout protocol (one line, then silence):
    READY       — SDK initialised; parent can now send start/stop
    ERROR: msg  — init failed; process exits with code 1
"""
import sys
import time
import select


def main():
    network_interface = sys.argv[1] if len(sys.argv) > 1 else ""

    try:
        import unitree_sdk2py.core.channel as channel
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        channel.ChannelFactoryInitialize(0, network_interface)
        client = SportClient()
        client.SetTimeout(10.0)
        client.Init()
    except Exception as exc:
        sys.stdout.write("ERROR: {}\n".format(exc))
        sys.stdout.flush()
        sys.exit(1)

    sys.stdout.write("READY\n")
    sys.stdout.flush()

    active = False
    while True:
        # When active: poll stdin every 10 ms, call StopMove in between.
        # When standby: block on stdin for up to 100 ms.
        timeout = 0.01 if active else 0.1
        r, _, _ = select.select([sys.stdin], [], [], timeout)

        if r:
            line = sys.stdin.readline()
            if not line:   # EOF — parent process died
                break
            line = line.strip()
            if line == "start":
                active = True
            elif line == "stop":
                active = False
            elif line == "quit":
                break

        if active:
            try:
                client.StopMove()
            except Exception:
                pass


if __name__ == "__main__":
    main()
