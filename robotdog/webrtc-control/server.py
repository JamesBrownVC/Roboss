#!/usr/bin/env python3
"""
Go2 WebRTC Control Server
=========================
Bridge between a browser UI and a Unitree Go2 robot using
https://github.com/legion1581/unitree_webrtc_connect

- Connects to the robot over WebRTC (LocalSTA, direct ethernet by default)
- Exposes a web dashboard  : http://localhost:8080
- MJPEG video stream       : /video
- WebSocket (state + cmds) : /ws
- Constants dump           : /api/constants

Env vars:
  UNITREE_ROBOT_IP    robot IP (default: 192.168.123.161)
  UNITREE_AES_128_KEY per-device AES key, required on firmware >= 1.1.15
  PORT                HTTP port (default: 8080)
"""

import asyncio
import io
import json
import logging
import os
import time

from aiohttp import web, WSMsgType
from PIL import Image

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from unitree_webrtc_connect.constants import (
    RTC_TOPIC,
    SPORT_CMD,
    SPORT_CMD_MCF,
    AUDIO_API,
    OBSTACLES_AVOID_API,
)

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("go2-control")
log.setLevel(logging.INFO)
# aiortc logs a WARNING for every H264 packet it can't decode (constant noise
# while waiting for keyframes) — keep only real errors.
logging.getLogger("aiortc.codecs.h264").setLevel(logging.ERROR)
logging.getLogger("aiortc").setLevel(logging.ERROR)
logging.getLogger("aioice").setLevel(logging.ERROR)

ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "192.168.123.161")
AES_KEY = os.environ.get("UNITREE_AES_128_KEY") or None
PORT = int(os.environ.get("PORT", "8080"))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# VUI api ids (rt/api/vui/request)
VUI_API = {
    "SET_SWITCH": 1001, "GET_SWITCH": 1002,
    "SET_VOLUME": 1003, "GET_VOLUME": 1004,
    "SET_BRIGHTNESS": 1005, "GET_BRIGHTNESS": 1006,
    "SET_COLOR": 1007, "GET_COLOR": 1008,
}

# Motion switcher api ids (rt/api/motion_switcher/request)
MOTION_SWITCHER_API = {"GET_MODE": 1001, "SET_MODE": 1002, "RELEASE_MODE": 1003}

# State topics pushed to the browser automatically
DEFAULT_SUBSCRIPTIONS = [
    RTC_TOPIC["LF_SPORT_MOD_STATE"],
    RTC_TOPIC["LOW_STATE"],
    RTC_TOPIC["MULTIPLE_STATE"],
    RTC_TOPIC["ULIDAR_STATE"],
    RTC_TOPIC["SERVICE_STATE"],
    RTC_TOPIC["AUDIO_HUB_PLAY_STATE"],
    RTC_TOPIC["UWB_STATE"],
    RTC_TOPIC["GAS_SENSOR"],
    RTC_TOPIC["WIRELESS_CONTROLLER"],  # R3-1 remote (sticks + buttons)
]

# Acrobatic commands disabled for safety
ACRO_BLOCKLIST = {"FrontFlip", "BackFlip", "LeftFlip", "RightFlip", "FrontJump",
                  "FrontPounce", "Handstand", "HandStand", "BackStand", "StandOut"}

REQUEST_TIMEOUT = 8  # seconds


class RobotBridge:
    """Owns the WebRTC connection and fan-outs state to websocket clients."""

    def __init__(self):
        self.conn: UnitreeWebRTCConnection | None = None
        self.connected = False
        self.ws_clients: set[web.WebSocketResponse] = set()
        self.latest_state: dict[str, dict] = {}   # topic -> last message data
        self.dirty_topics: set[str] = set()
        self.latest_jpeg: bytes | None = None
        self.frame_event = asyncio.Event()
        self.video_viewers = 0
        self.avoid_enabled = False
        self.motion_mode = "?"
        self.lidar_points: list | None = None
        self.lidar_meta: dict | None = None
        self.lidar_dirty = False
        self._joy = {"lx": 0.0, "ly": 0.0, "rx": 0.0}
        self._joy_deadline = 0.0
        self._joy_task: asyncio.Task | None = None

    # ------------------------------------------------------------- connect

    async def connect(self):
        log.info("Connecting to Go2 @ %s ...", ROBOT_IP)
        self.conn = UnitreeWebRTCConnection(
            WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP, aes_128_key=AES_KEY
        )
        await self.conn.connect()
        self.connected = True
        log.info("WebRTC connected")

        # Video: register frame consumer + enable channel
        self.conn.video.add_track_callback(self._video_track)
        self.conn.video.switchVideoChannel(True)

        # State subscriptions
        for topic in DEFAULT_SUBSCRIPTIONS:
            self.conn.datachannel.pub_sub.subscribe(topic, self._make_state_cb(topic))

        # Make sure the LiDAR is spinning — obstacle avoidance depends on it
        self.conn.datachannel.pub_sub.publish_without_callback(
            RTC_TOPIC["ULIDAR_SWITCH"], "on")

        # Discover current motion mode
        try:
            resp = await self.request(RTC_TOPIC["MOTION_SWITCHER"],
                                      MOTION_SWITCHER_API["GET_MODE"])
            data = json.loads(resp["data"]["data"])
            self.motion_mode = data.get("name", "?")
            log.info("Motion mode: %s", self.motion_mode)
        except Exception as e:
            log.warning("Could not fetch motion mode: %s", e)

        # Sync obstacle-avoidance state from the robot
        try:
            resp = await self.request(RTC_TOPIC["OBSTACLES_AVOID"],
                                      OBSTACLES_AVOID_API["SWITCH_GET"])
            self.avoid_enabled = bool(json.loads(resp["data"]["data"]).get("enable"))
            log.info("Obstacle avoidance: %s", self.avoid_enabled)
            if self.avoid_enabled:
                self._joy_task = asyncio.create_task(self._joystick_loop())
        except Exception as e:
            log.warning("Could not fetch avoid state: %s", e)

        asyncio.create_task(self._broadcast_loop())

    def _make_state_cb(self, topic):
        def cb(message):
            self.latest_state[topic] = message.get("data")
            self.dirty_topics.add(topic)
        return cb

    # --------------------------------------------------------------- video

    async def _video_track(self, track):
        while True:
            frame = await track.recv()
            if self.video_viewers <= 0:
                continue  # drain frames but skip encoding when nobody watches
            try:
                img: Image.Image = frame.to_image()
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=80)
                self.latest_jpeg = buf.getvalue()
                self.frame_event.set()
                self.frame_event = asyncio.Event()
            except Exception:
                log.exception("frame encode failed")

    # --------------------------------------------------------------- lidar

    def _lidar_cb(self, message):
        try:
            md = message.get("data", {})
            decoded = md.get("data")
            positions = None
            if isinstance(decoded, dict):
                positions = decoded.get("positions")
                if positions is None:
                    positions = decoded.get("points")
            if positions is None or len(positions) == 0:
                return
            if hasattr(positions, "tolist"):  # numpy array -> plain list
                positions = positions.tolist()
            # downsample: cap at 5000 points (15000 floats)
            max_floats = 15000
            if len(positions) > max_floats:
                step = (len(positions) // 3) // 5000 or 1
                positions = [v for i in range(0, len(positions) - 2, 3 * step)
                             for v in positions[i:i + 3]]
            self.lidar_points = list(positions)
            # voxel-grid metadata needed to map indices -> world meters
            self.lidar_meta = {
                "resolution": md.get("resolution"),
                "origin": md.get("origin"),
            }
            self.lidar_dirty = True
        except Exception:
            log.exception("lidar decode failed")

    async def set_lidar(self, enable: bool):
        if enable:
            await self.conn.datachannel.disableTrafficSaving(True)
            self.conn.datachannel.pub_sub.subscribe(
                RTC_TOPIC["ULIDAR_ARRAY"], self._lidar_cb)
            self.conn.datachannel.pub_sub.publish_without_callback(
                RTC_TOPIC["ULIDAR_SWITCH"], "on")
        else:
            # only stop the point-cloud view — NEVER switch the lidar off,
            # obstacle avoidance depends on it
            self.conn.datachannel.pub_sub.unsubscribe(RTC_TOPIC["ULIDAR_ARRAY"])
            self.lidar_points = None

    # ------------------------------------------------------------ commands

    async def request(self, topic, api_id, parameter=None, priority=None):
        opts = {"api_id": api_id}
        if parameter is not None:
            opts["parameter"] = parameter
        if priority is not None:
            opts["priority"] = priority
        return await asyncio.wait_for(
            self.conn.datachannel.pub_sub.publish_request_new(topic, opts),
            timeout=REQUEST_TIMEOUT,
        )

    async def sport(self, cmd, parameter=None):
        if cmd in ACRO_BLOCKLIST:
            raise ValueError(f"acrobatic command '{cmd}' is disabled")
        table = SPORT_CMD_MCF if self.motion_mode == "mcf" else SPORT_CMD
        api_id = table.get(cmd) or SPORT_CMD.get(cmd)
        if api_id is None:
            raise ValueError(f"unknown sport command: {cmd}")
        return await self.request(RTC_TOPIC["SPORT_MOD"], api_id, parameter)

    async def move(self, x, y, z):
        if self.avoid_enabled and self.motion_mode != "mcf":
            # Legacy avoidance filters the (simulated) joystick stream on
            # rt/wirelesscontroller — NOT the sport Move api. Feed the
            # 50 Hz publisher with normalized stick values.
            clamp = lambda v: max(-1.0, min(1.0, v))
            self._joy = {"lx": clamp(-y), "ly": clamp(x), "rx": clamp(z)}
            self._joy_deadline = time.monotonic() + 0.45
            return {"ok": True}
        # MCF avoid mode filters sport Move commands internally.
        return await self.sport("Move", {"x": x, "y": y, "z": z})

    def _publish_joystick(self, lx=0.0, ly=0.0, rx=0.0, ry=0.0, keys=0):
        self.conn.datachannel.pub_sub.publish_without_callback(
            RTC_TOPIC["WIRELESS_CONTROLLER"],
            {"lx": lx, "ly": ly, "rx": rx, "ry": ry, "keys": keys},
        )

    async def _joystick_loop(self):
        """50 Hz simulated-joystick publisher, active while avoid mode is on."""
        was_moving = False
        while self.avoid_enabled:
            if time.monotonic() < self._joy_deadline:
                self._publish_joystick(**self._joy)
                was_moving = True
            elif was_moving:
                self._publish_joystick()  # zero once = stop
                was_moving = False
            await asyncio.sleep(0.02)
        if was_moving:
            self._publish_joystick()

    async def set_mode(self, name):
        resp = await self.request(RTC_TOPIC["MOTION_SWITCHER"],
                                  MOTION_SWITCHER_API["SET_MODE"], {"name": name})
        self.motion_mode = name
        return resp

    async def get_avoid(self) -> bool:
        g = await self.request(RTC_TOPIC["OBSTACLES_AVOID"],
                               OBSTACLES_AVOID_API["SWITCH_GET"])
        return bool(json.loads(g["data"]["data"]).get("enable"))

    async def set_avoid(self, enable: bool):
        # primary path: dedicated obstacles_avoid service
        try:
            await self.request(RTC_TOPIC["OBSTACLES_AVOID"],
                               OBSTACLES_AVOID_API["SWITCH_SET"],
                               {"enable": enable})
        except Exception as e:
            log.warning("SWITCH_SET failed: %s", e)
        await asyncio.sleep(0.3)
        state = await self.get_avoid()
        # MCF firmware fallback: avoidance toggled via sport api 2058
        if state != enable and self.motion_mode == "mcf":
            log.info("SWITCH_SET ineffective, trying MCF SwitchAvoidMode")
            await self.sport("SwitchAvoidMode", {"data": enable})
            await asyncio.sleep(0.5)
            state = await self.get_avoid()
        self.avoid_enabled = state
        if state:
            self._joy = {"lx": 0.0, "ly": 0.0, "rx": 0.0}
            self._joy_deadline = 0.0
            if self._joy_task is None or self._joy_task.done():
                self._joy_task = asyncio.create_task(self._joystick_loop())
        log.info("Obstacle avoidance requested=%s actual=%s", enable, state)
        return {"enable": state}

    # ----------------------------------------------------------- broadcast

    async def _broadcast_loop(self):
        while True:
            await asyncio.sleep(0.2)  # 5 Hz to the browser
            if not self.ws_clients:
                self.dirty_topics.clear()
                self.lidar_dirty = False
                continue
            payloads = []
            for topic in list(self.dirty_topics):
                payloads.append(json.dumps(
                    {"type": "state", "topic": topic,
                     "data": self.latest_state.get(topic)},
                    default=str))
            self.dirty_topics.clear()
            if self.lidar_dirty and self.lidar_points is not None:
                payloads.append(json.dumps(
                    {"type": "lidar", "positions": self.lidar_points,
                     "meta": self.lidar_meta}, default=str))
                self.lidar_dirty = False
            payloads.append(json.dumps({
                "type": "status",
                "connected": self.connected,
                "robot_ip": ROBOT_IP,
                "motion_mode": self.motion_mode,
                "avoid_enabled": self.avoid_enabled,
            }))
            for ws in list(self.ws_clients):
                for p in payloads:
                    try:
                        await ws.send_str(p)
                    except Exception:
                        self.ws_clients.discard(ws)
                        break


bridge = RobotBridge()

# ================================================================ handlers


async def handle_index(request):
    return web.FileResponse(os.path.join(STATIC_DIR, "index.html"))


async def handle_constants(request):
    return web.json_response({
        "RTC_TOPIC": RTC_TOPIC,
        "SPORT_CMD": SPORT_CMD,
        "SPORT_CMD_MCF": SPORT_CMD_MCF,
        "AUDIO_API": AUDIO_API,
        "OBSTACLES_AVOID_API": OBSTACLES_AVOID_API,
        "VUI_API": VUI_API,
        "MOTION_SWITCHER_API": MOTION_SWITCHER_API,
    })


async def handle_video(request):
    resp = web.StreamResponse(headers={
        "Content-Type": "multipart/x-mixed-replace; boundary=frame",
        "Cache-Control": "no-cache",
    })
    await resp.prepare(request)
    bridge.video_viewers += 1
    try:
        while True:
            event = bridge.frame_event
            await event.wait()
            jpeg = bridge.latest_jpeg
            if jpeg is None:
                continue
            await resp.write(
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                + jpeg + b"\r\n")
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        bridge.video_viewers -= 1
    return resp


async def _dispatch(msg: dict):
    """Execute a websocket command and return a result dict."""
    action = msg.get("action")
    b = bridge

    if action == "sport":
        return await b.sport(msg["cmd"], msg.get("parameter"))
    if action == "move":
        return await b.move(msg.get("x", 0), msg.get("y", 0), msg.get("z", 0))
    if action == "stop":
        b._joy_deadline = 0.0  # release simulated joystick immediately
        return await b.sport("StopMove")
    if action == "get_mode":
        resp = await b.request(RTC_TOPIC["MOTION_SWITCHER"],
                               MOTION_SWITCHER_API["GET_MODE"])
        data = json.loads(resp["data"]["data"])
        b.motion_mode = data.get("name", "?")
        return data
    if action == "set_mode":
        return await b.set_mode(msg["name"])
    if action == "request":  # generic RPC: any topic / api_id / parameter
        return await b.request(msg["topic"], msg["api_id"], msg.get("parameter"))
    if action == "publish":  # generic raw publish
        b.conn.datachannel.pub_sub.publish_without_callback(
            msg["topic"], msg.get("data"), msg.get("msg_type"))
        return {"ok": True}
    if action == "subscribe":
        topic = msg["topic"]
        b.conn.datachannel.pub_sub.subscribe(topic, b._make_state_cb(topic))
        return {"ok": True}
    if action == "unsubscribe":
        b.conn.datachannel.pub_sub.unsubscribe(msg["topic"])
        return {"ok": True}

    # --- VUI (LED / volume / brightness)
    if action == "vui_color":
        p = {"color": msg["color"]}
        if msg.get("time") is not None:
            p["time"] = msg["time"]
        if msg.get("flash_cycle") is not None:
            p["flash_cycle"] = msg["flash_cycle"]
        return await b.request(RTC_TOPIC["VUI"], VUI_API["SET_COLOR"], p)
    if action == "vui_brightness":
        return await b.request(RTC_TOPIC["VUI"], VUI_API["SET_BRIGHTNESS"],
                               {"brightness": int(msg["level"])})
    if action == "vui_volume":
        return await b.request(RTC_TOPIC["VUI"], VUI_API["SET_VOLUME"],
                               {"volume": int(msg["level"])})
    if action == "vui_get":
        out = {}
        for key, api in [("volume", "GET_VOLUME"), ("brightness", "GET_BRIGHTNESS")]:
            resp = await b.request(RTC_TOPIC["VUI"], VUI_API[api])
            out.update(json.loads(resp["data"]["data"]))
        return out

    # --- obstacle avoidance
    if action == "avoid_set":
        return await b.set_avoid(bool(msg["enable"]))
    if action == "avoid_get":
        b.avoid_enabled = await b.get_avoid()
        if b.avoid_enabled and (b._joy_task is None or b._joy_task.done()):
            b._joy_task = asyncio.create_task(b._joystick_loop())
        return {"enable": b.avoid_enabled}

    # --- lidar
    if action == "lidar":
        await b.set_lidar(bool(msg["enable"]))
        return {"ok": True}

    # --- video / audio channels
    if action == "video_channel":
        b.conn.datachannel.switchVideoChannel(bool(msg["enable"]))
        return {"ok": True}
    if action == "audio_channel":
        b.conn.datachannel.switchAudioChannel(bool(msg["enable"]))
        return {"ok": True}

    # --- audio hub
    if action == "audio_list":
        resp = await b.request(RTC_TOPIC["AUDIO_HUB_REQ"],
                               AUDIO_API["GET_AUDIO_LIST"], json.dumps({}))
        return json.loads(resp["data"]["data"])
    if action == "audio_play":
        return await b.request(RTC_TOPIC["AUDIO_HUB_REQ"],
                               AUDIO_API["SELECT_START_PLAY"],
                               json.dumps({"unique_id": msg["uuid"]}))
    if action == "audio_pause":
        return await b.request(RTC_TOPIC["AUDIO_HUB_REQ"], AUDIO_API["PAUSE"],
                               json.dumps({}))
    if action == "audio_resume":
        return await b.request(RTC_TOPIC["AUDIO_HUB_REQ"], AUDIO_API["UNSUSPEND"],
                               json.dumps({}))
    if action == "audio_play_mode":
        return await b.request(RTC_TOPIC["AUDIO_HUB_REQ"], AUDIO_API["SET_PLAY_MODE"],
                               json.dumps({"play_mode": msg["mode"]}))
    if action == "megaphone":
        api = AUDIO_API["ENTER_MEGAPHONE"] if msg["enable"] else AUDIO_API["EXIT_MEGAPHONE"]
        return await b.request(RTC_TOPIC["AUDIO_HUB_REQ"], api, json.dumps({}))

    raise ValueError(f"unknown action: {action}")


async def handle_ws(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    bridge.ws_clients.add(ws)
    # send cached state immediately
    for topic, data in bridge.latest_state.items():
        await ws.send_str(json.dumps(
            {"type": "state", "topic": topic, "data": data}, default=str))
    try:
        async for raw in ws:
            if raw.type != WSMsgType.TEXT:
                continue
            msg = json.loads(raw.data)
            req_id = msg.get("id")
            try:
                result = await _dispatch(msg)
                await ws.send_str(json.dumps(
                    {"type": "response", "id": req_id, "ok": True,
                     "result": result}, default=str))
            except Exception as e:
                log.warning("command failed: %s -> %s", msg.get("action"), e)
                await ws.send_str(json.dumps(
                    {"type": "response", "id": req_id, "ok": False,
                     "error": str(e)}))
    finally:
        bridge.ws_clients.discard(ws)
    return ws


# ==================================================================== main

async def on_startup(app):
    asyncio.create_task(bridge.connect())


def main():
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/ws", handle_ws)
    app.router.add_get("/video", handle_video)
    app.router.add_get("/api/constants", handle_constants)
    app.router.add_static("/static", STATIC_DIR)
    app.on_startup.append(on_startup)
    log.info("UI available at http://localhost:%d  (robot: %s)", PORT, ROBOT_IP)
    web.run_app(app, port=PORT, print=None)


if __name__ == "__main__":
    main()
