"""Runs only the models/classes the user selected."""
from __future__ import annotations

import threading
from typing import Callable

import numpy as np

from .registry import ModelSpec, Registry

# (model_spec) -> object with .predict(img, classes=[ids], conf=, imgsz=) and .names
ModelLoader = Callable[[ModelSpec], object]


def _ultralytics_loader(spec: ModelSpec):
    from ultralytics import YOLO  # lazy: heavy import

    return YOLO(spec.weights_path())


class Detector:
    def __init__(self, registry: Registry, loader: ModelLoader = _ultralytics_loader):
        self.registry = registry
        self._loader = loader
        self._models: dict[str, object] = {}
        self._lock = threading.Lock()  # one inference at a time per process

    def _model(self, spec: ModelSpec):
        if spec.name not in self._models:
            self._models[spec.name] = self._loader(spec)
        return self._models[spec.name]

    def warmup(self) -> None:
        """Load every available model and run one blank inference.

        The first prediction after loading is much slower than later ones (tens of seconds on a small
        CPU host), so do it at startup instead of on a user's first frame.
        """
        for spec in self.registry.models.values():
            if not spec.available():
                continue
            try:
                with self._lock:
                    model = self._model(spec)
                    blank = np.zeros((spec.imgsz, spec.imgsz, 3), np.uint8)
                    model.predict(blank, imgsz=spec.imgsz, verbose=False)
            except Exception as e:  # a bad model must not take the server down
                print(f"warmup: model {spec.name!r} failed: {e}")

    def detect(self, frame_bgr: np.ndarray, selected: list[str],
               conf: float | None = None) -> list[dict]:
        """Return detections for the selected classes only.

        Boxes are normalised to 0..1 so the client can draw them at any size.
        """
        h, w = frame_bgr.shape[:2]
        results: list[dict] = []
        with self._lock:
            for model_name, wanted in self.registry.group(selected).items():
                spec = self.registry.models[model_name]
                model = self._model(spec)
                names = model.names  # {id: name}
                to_ui = {spec.native[c].lower(): c for c in wanted}
                ids = [i for i, n in names.items() if n.lower() in to_ui]
                if not ids:
                    continue
                # classes=ids makes the model drop everything else at NMS time.
                pred = model.predict(
                    frame_bgr, classes=ids, conf=conf if conf is not None else spec.conf,
                    imgsz=spec.imgsz, verbose=False,
                )[0]
                for box in pred.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    results.append({
                        "class": to_ui[names[int(box.cls[0])].lower()],
                        "conf": round(float(box.conf[0]), 3),
                        "box": [x1 / w, y1 / h, x2 / w, y2 / h],
                        "model": model_name,
                    })
        return results
