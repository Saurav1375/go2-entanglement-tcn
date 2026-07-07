#!/usr/bin/env python3
"""Standalone recovery-sequence daemon with standby protocol — no ROS 2.

Runs as a persistent subprocess of recovery_node. Initialises the Unitree
SDK2 SportClient immediately at startup, then waits for commands on stdin.
The parent node writes "start" when the latch activates and "stop" to return
to standby.  Because the SDK handshake happens at node startup, there is zero
additional latency between alarm and the first motion command.

Recovery sequence (runs ONCE per "start", non-blocking so "stop"/"quit"
are honoured mid-sequence):

    STOP  ->  MOVE BACKWARD (0.5 s)  ->  STOP  ->  FRONT JUMP (once)  ->  STOP (hold)

Tunable timings/velocity are the constants below.

Usage:
    python3 sport_worker.py [network_interface]

stdin protocol (one command per line):
    start   -> run the recovery sequence once, then hold a stop
    stop    -> abort/return to standby (stay alive for next alarm)
    quit    -> exit cleanly

stdout protocol (one line, then silence):
    READY       -- SDK initialised; parent can now send start/stop
    ERROR: msg  -- init failed; process exits with code 1
"""
import sys
import time
import select

# --- Tunables --------------------------------------------------------------
BACKWARD_VX = -0.6        # m/s, negative = backward (Go2: vx>0 forward)
BACKWARD_DURATION = 2.0   # s, how long to drive backward
STOP_SETTLE = 0.4         # s, pause on each StopMove so motion fully settles
JUMP_SETTLE = 2.5         # s, time to let FrontJump physically complete
TICK_ACTIVE = 0.02        # s, command period while a sequence runs (50 Hz)
TICK_IDLE = 0.1           # s, stdin poll period in standby
HOLD_PERIOD = 0.1         # s, StopMove re-issue period while holding after seq

# --- Sequence phases -------------------------------------------------------
IDLE = "IDLE"          # standby, waiting for "start"
STOP1 = "STOP1"        # initial stop + settle
BACKWARD = "BACKWARD"  # drive backward, continuous Move()
STOP2 = "STOP2"        # stop + settle before the jump
JUMP = "JUMP"          # issue FrontJump exactly once, then wait for it
HOLD = "HOLD"          # sequence done: keep issuing StopMove until reset


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

    phase = IDLE
    phase_start = 0.0        # monotonic timestamp when current phase began
    jump_issued = False      # ensures FrontJump is called exactly once
    last_hold_cmd = 0.0      # last time StopMove was re-issued during HOLD

    def enter(new_phase, now):
        nonlocal phase, phase_start, jump_issued
        phase = new_phase
        phase_start = now
        if new_phase == JUMP:
            jump_issued = False

    def safe(fn):
        try:
            fn()
        except Exception:
            pass

    while True:
        # Poll stdin more tightly while a sequence is active so backward
        # velocity commands stay smooth; block longer in standby.
        timeout = TICK_IDLE if phase == IDLE else TICK_ACTIVE
        r, _, _ = select.select([sys.stdin], [], [], timeout)

        if r:
            line = sys.stdin.readline()
            if not line:            # EOF — parent process died
                break
            cmd = line.strip()
            if cmd == "start":
                # Begin the sequence. Ignore if already running so a stray
                # "start" can never re-trigger the jump.
                if phase == IDLE:
                    safe(client.StopMove)
                    enter(STOP1, time.monotonic())
            elif cmd == "stop":
                safe(client.StopMove)
                enter(IDLE, time.monotonic())
            elif cmd == "quit":
                safe(client.StopMove)
                break

        now = time.monotonic()
        elapsed = now - phase_start

        if phase == STOP1:
            # Held stopped; when settled, start driving backward.
            if elapsed >= STOP_SETTLE:
                enter(BACKWARD, now)

        elif phase == BACKWARD:
            if elapsed < BACKWARD_DURATION:
                safe(lambda: client.Move(BACKWARD_VX, 0.0, 0.0))
            else:
                safe(client.StopMove)
                enter(STOP2, now)

        elif phase == STOP2:
            if elapsed >= STOP_SETTLE:
                enter(JUMP, now)

        elif phase == JUMP:
            if not jump_issued:
                safe(client.FrontJump)   # exactly once
                jump_issued = True
            elif elapsed >= JUMP_SETTLE:
                safe(client.StopMove)
                enter(HOLD, now)

        elif phase == HOLD:
            # Sequence complete. Keep the robot stopped (overriding the remote
            # controller) until the parent sends "stop"/"quit" on reset.
            if now - last_hold_cmd >= HOLD_PERIOD:
                safe(client.StopMove)
                last_hold_cmd = now


if __name__ == "__main__":
    main()