import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_ods import YAML, FakeModel

from ods.detector import Detector
from ods.hub import client_ip
from ods.registry import Registry
from ods.server import create_app

JPG = cv2.imencode(".jpg", np.zeros((100, 100, 3), np.uint8))[1].tobytes()


@pytest.fixture
def client(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(YAML)
    reg = Registry.load(p)
    # `with` keeps every connection on one event loop, like a real server.
    with TestClient(create_app(reg, Detector(reg, loader=lambda s: FakeModel()))) as c:
        yield c


def unpack(blob: bytes):
    n = int.from_bytes(blob[:4], "big")
    return json.loads(blob[4:4 + n]), blob[4 + n:]


def pair(phone, host, name="Pixel"):
    """Run request -> accept; returns once the session is live on both sides."""
    reg = host.receive_json()
    assert reg["type"] == "registered"
    phone.send_json({"type": "request", "host_id": reg["id"], "name": name})
    assert phone.receive_json()["type"] == "pending"
    req = host.receive_json()
    assert req["type"] == "request" and req["from"] == name
    host.send_json({"type": "respond", "req_id": req["req_id"], "accept": True})
    assert phone.receive_json()["type"] == "accepted"
    assert host.receive_json() == {"type": "session", "state": "started", "phone": name}
    return reg


def test_phone_stream_is_detected_with_hosts_classes_and_relayed(client):
    with client.websocket_connect("/ws/host?name=Desk") as host, \
            client.websocket_connect("/ws/phone") as phone:
        reg = pair(phone, host)
        assert reg["name"] == "Desk" and len(reg["code"]) == 6
        host.send_json({"type": "config", "classes": ["dog"], "conf": 0.3})
        phone.send_bytes(JPG)
        head, jpg = unpack(host.receive_bytes())
        assert [d["class"] for d in head["detections"]] == ["dog"]
        assert jpg == JPG
        ack = phone.receive_json()
        assert ack["type"] == "ack" and ack["found"] == 1


def test_frames_relayed_without_detection_when_no_classes_selected(client):
    with client.websocket_connect("/ws/host") as host, client.websocket_connect("/ws/phone") as phone:
        pair(phone, host)
        phone.send_bytes(JPG)
        head, jpg = unpack(host.receive_bytes())
        assert head["detections"] == [] and jpg == JPG
        assert phone.receive_json()["found"] == 0


def test_hosts_listed_oldest_first_and_flagged_busy(client):
    with client.websocket_connect("/ws/host?name=First") as h1, \
            client.websocket_connect("/ws/host?name=Second") as h2:
        id1, id2 = h1.receive_json()["id"], h2.receive_json()["id"]
        listed = client.get("/api/hosts").json()["hosts"]
        assert [(h["id"], h["busy"]) for h in listed] == [(id1, False), (id2, False)]

        with client.websocket_connect("/ws/phone") as phone:
            phone.send_json({"type": "request", "host_id": id1})
            phone.receive_json()
            req = h1.receive_json()
            h1.send_json({"type": "respond", "req_id": req["req_id"], "accept": True})
            assert phone.receive_json()["type"] == "accepted"
            h1.receive_json()
            assert client.get("/api/hosts").json()["hosts"][0]["busy"] is True

            with client.websocket_connect("/ws/phone") as late:
                late.send_json({"type": "request", "host_id": id1})
                assert late.receive_json() == {"type": "declined", "reason": "busy"}


def test_decline(client):
    with client.websocket_connect("/ws/host") as host, client.websocket_connect("/ws/phone") as phone:
        hid = host.receive_json()["id"]
        phone.send_json({"type": "request", "host_id": hid})
        phone.receive_json()
        req = host.receive_json()
        host.send_json({"type": "respond", "req_id": req["req_id"], "accept": False})
        assert phone.receive_json() == {"type": "declined", "reason": "declined"}
        phone.send_bytes(JPG)  # no session: frames are refused, not relayed
        assert phone.receive_json()["type"] == "error"


def test_accepting_one_declines_the_rest_as_busy(client):
    with client.websocket_connect("/ws/host") as host, \
            client.websocket_connect("/ws/phone") as a, client.websocket_connect("/ws/phone") as b:
        hid = host.receive_json()["id"]
        a.send_json({"type": "request", "host_id": hid, "name": "A"})
        a.receive_json()
        first = host.receive_json()
        b.send_json({"type": "request", "host_id": hid, "name": "B"})
        b.receive_json()
        host.receive_json()  # B's request
        host.send_json({"type": "respond", "req_id": first["req_id"], "accept": True})
        assert a.receive_json()["type"] == "accepted"
        assert b.receive_json() == {"type": "declined", "reason": "busy"}


def test_pairing_code_works_across_networks(client):
    with client.websocket_connect("/ws/host", headers={"x-forwarded-for": "8.8.8.8"}) as host:
        reg = host.receive_json()
        other = {"x-forwarded-for": "1.1.1.1"}
        assert client.get("/api/hosts", headers=other).json()["hosts"] == []
        found = client.get("/api/hosts", params={"code": reg["code"]}, headers=other).json()["hosts"]
        assert [h["id"] for h in found] == [reg["id"]]
        assert client.get("/api/hosts", params={"code": "000000"}, headers=other).json()["hosts"] == []

        with client.websocket_connect("/ws/phone", headers=other) as phone:
            phone.send_json({"type": "request", "code": reg["code"], "name": "Via code"})
            assert phone.receive_json()["type"] == "pending"
            assert host.receive_json()["from"] == "Via code"


def test_phone_cancelling_withdraws_request(client):
    with client.websocket_connect("/ws/host") as host:
        hid = host.receive_json()["id"]
        with client.websocket_connect("/ws/phone") as phone:
            phone.send_json({"type": "request", "host_id": hid})
            req = host.receive_json()
        assert host.receive_json() == {"type": "cancelled", "req_id": req["req_id"]}


def test_host_leaving_ends_phone_session(client):
    with client.websocket_connect("/ws/phone") as phone:
        with client.websocket_connect("/ws/host") as host:
            pair(phone, host)
        assert phone.receive_json() == {"type": "ended", "reason": "host_left"}


def test_phone_leaving_tells_host(client):
    with client.websocket_connect("/ws/host") as host:
        with client.websocket_connect("/ws/phone") as phone:
            pair(phone, host)
        assert host.receive_json() == {"type": "session", "state": "ended", "phone": "Pixel"}
        assert client.get("/api/hosts").json()["hosts"][0]["busy"] is False


def test_invalid_class_selection_reported_to_host(client):
    with client.websocket_connect("/ws/host") as host:
        host.receive_json()
        host.send_json({"type": "config", "classes": ["drone"]})
        assert host.receive_json()["type"] == "error"


def test_client_ip_grouping():
    assert client_ip({}, "10.0.0.5") == "10.0.0.5"
    assert client_ip({"x-forwarded-for": "9.9.9.9, 10.0.0.1"}, "127.0.0.1") == "9.9.9.9"
    assert client_ip({"cf-connecting-ip": "7.7.7.7", "x-forwarded-for": "9.9.9.9"}, None) == "7.7.7.7"
    # two devices on one IPv6 LAN share a /64
    a = client_ip({}, "2001:db8:1:2:aaaa::1")
    assert a == client_ip({}, "2001:db8:1:2:bbbb::9")
    assert client_ip({}, "::ffff:1.2.3.4") == "1.2.3.4"


def test_info_exposes_public_url_only_when_sharing(client):
    assert client.get("/api/info").json() == {"public_url": None}
    client.app.state.public_url = "https://abc.trycloudflare.com"
    assert client.get("/api/info").json() == {"public_url": "https://abc.trycloudflare.com"}
