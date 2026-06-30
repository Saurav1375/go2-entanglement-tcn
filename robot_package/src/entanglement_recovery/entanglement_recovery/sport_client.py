"""ROS 2 adapter: maps abstract FSM Commands to verified Unitree Sport API requests.

Publishes `unitree_api/msg/Request` on `/api/sport/request` and tracks
`/api/sport/response` return codes. Honors the `enable_actuation` safety gate: when False
(default) it NEVER publishes to the robot — it only logs the intended command (dry-run).

Python 3.8 compatible.
"""
from __future__ import annotations

import json
from typing import Callable, Dict, Optional

from unitree_api.msg import Request, Response  # provided by the robot's ROS 2 env

from .states import Command
from . import sport_api as API

# abstract command -> Sport API name (-> verified integer id in sport_api.SPORT_API_ID)
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
        self._result_cb = None  # type: Optional[Callable[[Command, bool], None]]
        self._inflight = {}     # type: Dict[int, Command]   api_id -> command
        self._req_id = 0

    def set_result_callback(self, cb):
        # type: (Callable[[Command, bool], None]) -> None
        self._result_cb = cb

    def send(self, command, parameter=None):
        # type: (Command, Optional[dict]) -> Optional[int]
        """Send a command. Returns the api_id sent, or None (unmapped / dry-run reports below)."""
        if command == Command.NONE:
            return None
        name = _CMD_TO_API.get(command)
        if name is None:
            self.log.warn("SportClient: no API mapping for {}".format(command))
            return None
        api_id = API.SPORT_API_ID[name]

        if not self.enable_actuation:
            # DRY-RUN: do not actuate; report the intended command only.
            self.log.warn("[DRY-RUN] would send {} (api_id={})".format(name, api_id))
            return api_id

        req = Request()
        self._req_id += 1
        req.header.identity.id = self._req_id
        req.header.identity.api_id = api_id
        if parameter is not None:
            req.parameter = json.dumps(parameter)
        self._inflight[api_id] = command
        self.pub.publish(req)
        self.log.info("sent {} (api_id={})".format(name, api_id))
        return api_id

    def _on_response(self, msg):
        # type: (Response) -> None
        try:
            api_id = int(msg.header.identity.api_id)
            code = int(msg.header.status.code)
        except Exception as exc:  # malformed response
            self.log.warn("SportClient: bad response: {}".format(exc))
            return
        command = self._inflight.pop(api_id, None)
        if command is None:
            return  # not one of ours
        success = (code == API.RESP_OK)
        if not success:
            self.log.warn("command {} returned code {}".format(command, code))
        if self._result_cb is not None:
            self._result_cb(command, success)
