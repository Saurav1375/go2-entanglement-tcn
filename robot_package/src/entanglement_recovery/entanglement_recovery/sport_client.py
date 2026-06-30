"""ROS 2 adapter: publish verified Unitree Sport API requests + track response codes.

Publishes `unitree_api/msg/Request` on `/api/sport/request` and reads `/api/sport/response`.
Honors the `enable_actuation` safety gate: when False (default) it NEVER publishes — it only
logs the intended command (dry-run). All api_ids come from the source-verified `sport_api` table.
Python 3.8 compatible.
"""
from __future__ import annotations

import json
from typing import Optional

from unitree_api.msg import Request, Response  # provided by the robot's ROS 2 env

from .states import Command
from . import sport_api as API

# abstract FSM Command -> Sport API name (for the direct-emit path: e.g. Damp from ESTOP/FAULT)
_CMD_TO_API = {
    Command.STOP_MOVE: "STOP_MOVE",
    Command.BALANCE_STAND: "BALANCE_STAND",
    Command.RECOVERY_STAND: "RECOVERY_STAND",
    Command.DAMP: "DAMP",
}


class SportClient:
    def __init__(self, node, request_topic, response_topic, enable_actuation, logger):
        # type: (object, str, str, bool, object) -> None
        self.node = node
        self.enable_actuation = bool(enable_actuation)
        self.log = logger
        self.pub = node.create_publisher(Request, request_topic, 10)
        self.sub = node.create_subscription(Response, response_topic, self._on_response, 10)
        self._inflight = {}        # api_id -> name (for response matching)
        self._error_pending = False
        self._req_id = 0

    # ---- core: send any verified api by name ----
    def send_api(self, name, params=None):
        # type: (str, Optional[dict]) -> Optional[int]
        api_id = API.SPORT_API_ID.get(name)
        if api_id is None:
            self.log.warn("SportClient: unknown api '{}'".format(name))
            return None
        if not self.enable_actuation:
            self.log.warn("[DRY-RUN] would send {} (api_id={}){}".format(
                name, api_id, " " + json.dumps(params) if params else ""))
            return api_id
        req = Request()
        self._req_id += 1
        req.header.identity.id = self._req_id
        req.header.identity.api_id = api_id
        if params:
            req.parameter = json.dumps(params)
        self._inflight[api_id] = name
        self.pub.publish(req)
        return api_id

    # ---- convenience: send an abstract Command (direct-emit path) ----
    def send(self, command):
        # type: (Command) -> Optional[int]
        if command == Command.NONE:
            return None
        name = _CMD_TO_API.get(command)
        if name is None:
            self.log.warn("SportClient: no direct API for {}".format(command))
            return None
        return self.send_api(name)

    def pop_error(self):
        # type: () -> bool
        """Return + clear whether any non-zero response code arrived since last call."""
        e, self._error_pending = self._error_pending, False
        return e

    def _on_response(self, msg):
        # type: (Response) -> None
        try:
            api_id = int(msg.header.identity.api_id)
            code = int(msg.header.status.code)
        except Exception as exc:
            self.log.warn("SportClient: bad response: {}".format(exc))
            return
        name = self._inflight.pop(api_id, None)
        if name is None:
            return
        if code != API.RESP_OK:
            self._error_pending = True
            self.log.warn("Sport api {} (id {}) returned code {}".format(name, api_id, code))
