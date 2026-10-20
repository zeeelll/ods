import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ods.detector import Detector
from ods.registry import Registry
from ods.server import create_app

YAML = """
default_conf: 0.3
models:
  coco: {weights: yolo.pt, classes: [person, car, dog]}
  drone: {weights: weights/does_not_exist.pt, classes: {drone: Drone}}
"""


class Box:
    def __init__(self, cls, conf, xyxy):
        self.cls, self.conf, self.xyxy = [cls], [conf], [_T(xyxy)]


class _T(list):
    def tolist(self):
        return list(self)


class FakeModel:
    names = {0: "person", 1: "car", 2: "dog"}
    calls = []

    def predict(self, img, classes, conf, imgsz, verbose):
        FakeModel.calls.append(classes)
        everything = [Box(0, .9, [0, 0, 50, 100]), Box(1, .8, [50, 50, 100, 100]), Box(2, .7, [0, 0, 10, 10])]
        return [type("R", (), {"boxes": [b for b in everything if b.cls[0] in classes]})()]


@pytest.fixture
def reg(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(YAML)
    return Registry.load(p)


def test_only_selected_classes_returned(reg):
    det = Detector(reg, loader=lambda s: FakeModel())
    out = det.detect(np.zeros((100, 100, 3), np.uint8), ["person", "car"])
    assert {d["class"] for d in out} == {"person", "car"}
    assert out[0]["box"] == [0, 0, 0.5, 1.0]  # normalised


def test_missing_weights_rejected_and_flagged(reg):
    assert {c["class"]: c["available"] for c in reg.catalog()}["drone"] is False
    with pytest.raises(FileNotFoundError):
        reg.group(["drone"])


def test_duplicate_class_rejected(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("models:\n a: {weights: a.pt, classes: [x]}\n b: {weights: b.pt, classes: [x]}\n")
    with pytest.raises(ValueError):
        Registry.load(p)


def test_websocket_flow(reg):
    client = TestClient(create_app(reg, Detector(reg, loader=lambda s: FakeModel())))
    assert any(c["class"] == "person" for c in client.get("/api/classes").json()["classes"])
    ok, jpg = cv2.imencode(".jpg", np.zeros((100, 100, 3), np.uint8))
    with client.websocket_connect("/ws/detect") as ws:
        ws.send_text('{"classes": ["dog"]}')
        ws.send_bytes(jpg.tobytes())
        assert [d["class"] for d in ws.receive_json()["detections"]] == ["dog"]
        ws.send_text('{"classes": ["drone"]}')
        assert "error" in ws.receive_json()


def test_ui_name_mapped_to_native_class(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("models:\n d: {weights: x.pt, classes: {drone: UAV}}\n")

    class M(FakeModel):
        names = {0: "uav"}
    out = Detector(Registry.load(p), loader=lambda s: M()).detect(np.zeros((10, 10, 3), np.uint8), ["drone"])
    assert [d["class"] for d in out] == ["drone"]
