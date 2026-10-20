"""Rendezvous between computers ("hosts") that display results and phones that supply a camera.

Flow:
  host  /ws/host     registers; gets an id + 6-digit pairing code; picks classes/confidence
  phone /api/hosts   lists hosts on the same network (oldest first), or one host by its code
  phone /ws/phone    sends {"type": "request", "host_id" | "code"}; host shows an accept popup
  phone              after acceptance streams binary JPEG frames, one at a time (waits for "ack")
  server             runs detection with the *host's* class selection and relays frame + boxes

Messages are JSON text unless noted.
  host  -> server  {"type": "respond", "req_id", "accept"} | {"type": "config", "classes", "conf"}
                   | {"type": "rename", "name"} | {"type": "end"}
  server -> host   {"type": "registered", "id", "code", "name"} | {"type": "request", "req_id", "from"}
                   | {"type": "cancelled", "req_id"} | {"type": "session", "state", "phone"}
                   | {"type": "error", "error"}
                   | binary: 4-byte big-endian N, N bytes of JSON {"detections", "ms"}, then the JPEG
  server -> phone  {"type": "pending"} | {"type": "accepted"} | {"type": "declined", "reason"}
                   | {"type": "ack", "found", "ms"} | {"type": "ended", "reason"} | {"type": "error"}
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Mapping

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .detector import Detector
from .registry import Registry

MAX_FRAME_BYTES = 4_000_000


def client_ip(headers: Mapping[str, str], peer: str | None) -> str:
    """Network identity used to decide who is "nearby".

    Behind a proxy or tunnel the real client is in a forwarded header. It is only used to group
    devices for discovery - a request still needs the host's explicit approval - so spoofing it
    gains nothing. IPv6 addresses are collapsed to their /64 so one LAN maps to one group.
    """
    raw = (headers.get("cf-connecting-ip")
           or (headers.get("x-forwarded-for") or "").split(",")[0].strip()
           or peer or "")
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        return raw
    if ip.version == 6:
        if ip.ipv4_mapped:
            return str(ip.ipv4_mapped)
        return str(ipaddress.ip_network(f"{ip}/64", strict=False).network_address)
    return str(ip)


def _clean(value, fallback: str, limit: int = 40) -> str:
    return " ".join(str(value or "").split())[:limit] or fallback


async def _json(ws: WebSocket, obj: dict) -> None:
    try:
        await ws.send_text(json.dumps(obj))
    except Exception:  # peer already gone; its own handler cleans up
        pass


async def _bytes(ws: WebSocket, data: bytes) -> None:
    try:
        await ws.send_bytes(data)
    except Exception:
        pass


@dataclass(eq=False)
class Pending:
    req_id: str
    phone: WebSocket
    label: str


@dataclass(eq=False)
class Session:
    phone: WebSocket
    label: str


@dataclass(eq=False)
class Host:
    id: str
    code: str
    name: str
    ip: str
    ws: WebSocket
    since: float = field(default_factory=time.time)
    pending: dict[str, Pending] = field(default_factory=dict)
    session: Session | None = None
    classes: list[str] = field(default_factory=list)
    conf: float | None = None


class Hub:
    def __init__(self, registry: Registry, detector: Detector):
        self.registry = registry
        self.detector = detector
        self.hosts: dict[str, Host] = {}

    # ---- discovery -------------------------------------------------------------------------

    def list_for(self, ip: str, code: str | None = None) -> list[dict]:
        """Hosts visible to a phone: those on its network, first-come first-listed, or one by code."""
        if code:
            code = code.strip()
            hosts = [h for h in self.hosts.values() if h.code == code]
        else:
            hosts = sorted((h for h in self.hosts.values() if h.ip == ip), key=lambda h: h.since)
        return [{"id": h.id, "name": h.name, "busy": h.session is not None} for h in hosts]

    def _new_code(self) -> str:
        used = {h.code for h in self.hosts.values()}
        while True:
            code = f"{secrets.randbelow(10**6):06d}"
            if code not in used:
                return code

    # ---- host side -------------------------------------------------------------------------

    async def run_host(self, ws: WebSocket, ip: str, name: str) -> None:
        await ws.accept()
        host = Host(id=uuid.uuid4().hex[:8], code=self._new_code(),
                    name=_clean(name, "Computer"), ip=ip, ws=ws)
        self.hosts[host.id] = host
        await _json(ws, {"type": "registered", "id": host.id, "code": host.code, "name": host.name})
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("text") is None:
                    continue
                try:
                    m = json.loads(msg["text"])
                except ValueError:
                    continue
                kind = m.get("type")
                if kind == "respond":
                    await self._respond(host, m)
                elif kind == "config":
                    await self._config(host, m)
                elif kind == "rename":
                    host.name = _clean(m.get("name"), host.name)
                elif kind == "end":
                    await self._end_session(host, "host_ended")
        except WebSocketDisconnect:
            pass
        finally:
            self.hosts.pop(host.id, None)
            for p in list(host.pending.values()):
                await _json(p.phone, {"type": "declined", "reason": "host_left"})
            host.pending.clear()
            await self._end_session(host, "host_left")

    async def _respond(self, host: Host, m: dict) -> None:
        p = host.pending.pop(str(m.get("req_id")), None)
        if p is None:
            return
        if not m.get("accept"):
            await _json(p.phone, {"type": "declined", "reason": "declined"})
        elif host.session is not None:
            await _json(p.phone, {"type": "declined", "reason": "busy"})
        else:
            host.session = Session(p.phone, p.label)
            await _json(p.phone, {"type": "accepted", "host": host.name})
            await _json(host.ws, {"type": "session", "state": "started", "phone": p.label})
            # First come, first served: everyone still waiting is told the host is taken.
            for other in list(host.pending.values()):
                await _json(other.phone, {"type": "declined", "reason": "busy"})
            host.pending.clear()

    async def _config(self, host: Host, m: dict) -> None:
        classes = [str(c) for c in m.get("classes", [])]
        try:
            self.registry.group(classes)  # validate early, like /ws/detect
        except (KeyError, FileNotFoundError) as e:
            host.classes = []
            await _json(host.ws, {"type": "error", "error": str(e.args[0])})
            return
        host.classes = classes
        conf = m.get("conf")
        host.conf = float(conf) if conf is not None else None

    async def _end_session(self, host: Host, reason: str) -> None:
        s, host.session = host.session, None
        if s is None:
            return
        await _json(s.phone, {"type": "ended", "reason": reason})
        await _json(host.ws, {"type": "session", "state": "ended", "phone": s.label})

    # ---- phone side ------------------------------------------------------------------------

    async def run_phone(self, ws: WebSocket, ip: str) -> None:
        await ws.accept()
        host: Host | None = None
        req_id: str | None = None
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("text") is not None:
                    try:
                        m = json.loads(msg["text"])
                    except ValueError:
                        continue
                    if m.get("type") == "request" and host is None:
                        host, req_id = await self._request(ws, m)
                    continue
                data = msg.get("bytes")
                if data is None:
                    continue
                if not (host and host.session and host.session.phone is ws):
                    await _json(ws, {"type": "error", "error": "not connected to a computer"})
                    continue
                await self._frame(host, ws, data)
        except WebSocketDisconnect:
            pass
        finally:
            if host:
                if req_id and host.pending.pop(req_id, None) is not None:
                    await _json(host.ws, {"type": "cancelled", "req_id": req_id})
                if host.session and host.session.phone is ws:
                    await self._end_session(host, "phone_left")

    async def _request(self, ws: WebSocket, m: dict) -> tuple[Host | None, str | None]:
        if m.get("host_id"):
            target = self.hosts.get(str(m["host_id"]))
        else:
            code = str(m.get("code", "")).strip()
            target = next((h for h in self.hosts.values() if code and h.code == code), None)
        if target is None:
            await _json(ws, {"type": "declined", "reason": "gone"})
            return None, None
        if target.session is not None:
            await _json(ws, {"type": "declined", "reason": "busy"})
            return None, None
        req_id = uuid.uuid4().hex[:8]
        label = _clean(m.get("name"), "Phone")
        target.pending[req_id] = Pending(req_id, ws, label)
        await _json(ws, {"type": "pending", "host": target.name})
        await _json(target.ws, {"type": "request", "req_id": req_id, "from": label})
        return target, req_id

    async def _frame(self, host: Host, ws: WebSocket, data: bytes) -> None:
        t0 = time.perf_counter()
        dets: list[dict] = []
        ack: dict = {"type": "ack"}
        classes, conf = list(host.classes), host.conf
        if len(data) > MAX_FRAME_BYTES:
            ack["error"] = "frame too large"
        elif classes:
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                ack["error"] = "bad frame"
            else:
                try:
                    dets = await asyncio.to_thread(self.detector.detect, frame, classes, conf)
                except Exception as e:  # keep the stream alive; surface the cause to the host
                    ack["error"] = "detection failed"
                    await _json(host.ws, {"type": "error", "error": f"detection failed: {e}"})
        if "error" not in ack and host.session is not None and host.session.phone is ws:
            ms = round((time.perf_counter() - t0) * 1000)
            head = json.dumps({"detections": dets, "ms": ms}).encode()
            await _bytes(host.ws, len(head).to_bytes(4, "big") + head + data)
            ack.update(found=len(dets), ms=ms)
        await _json(ws, ack)  # always ack so the phone's send loop never stalls
