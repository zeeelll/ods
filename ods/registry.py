"""Loads models.yaml and answers "which model detects which class"."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelSpec:
    name: str
    weights: str
    classes: tuple[str, ...]          # names shown in the UI
    native: dict[str, str]            # UI name -> class name inside the model
    conf: float
    imgsz: int

    def available(self) -> bool:
        # Bare names like "yolo11m.pt" are auto-downloaded by ultralytics;
        # anything with a directory part must exist on disk.
        p = Path(self.weights)
        if p.parent == Path("."):
            return True
        return (p if p.is_absolute() else ROOT / p).is_file()

    def weights_path(self) -> str:
        p = Path(self.weights)
        if p.parent == Path(".") or p.is_absolute():
            return self.weights
        return str(ROOT / p)


@dataclass
class Registry:
    models: dict[str, ModelSpec]
    default_conf: float = 0.35
    class_to_model: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = ROOT / "models.yaml") -> "Registry":
        raw = yaml.safe_load(Path(path).read_text())
        default_conf = float(raw.get("default_conf", 0.35))
        default_imgsz = int(raw.get("default_imgsz", 640))
        models: dict[str, ModelSpec] = {}
        class_to_model: dict[str, str] = {}
        for name, m in raw["models"].items():
            raw_classes = m["classes"]
            # list form: UI name == model name. dict form: {ui_name: model_name}
            native = dict(raw_classes) if isinstance(raw_classes, dict) else {c: c for c in raw_classes}
            spec = ModelSpec(
                name=name,
                weights=m["weights"],
                classes=tuple(native),
                native=native,
                conf=float(m.get("conf", default_conf)),
                imgsz=int(m.get("imgsz", default_imgsz)),
            )
            models[name] = spec
            for c in spec.classes:
                if c in class_to_model:
                    raise ValueError(f"class '{c}' defined by both '{class_to_model[c]}' and '{name}'")
                class_to_model[c] = name
        return cls(models, default_conf, class_to_model)

    def catalog(self) -> list[dict]:
        return [
            {"class": c, "model": s.name, "available": s.available()}
            for s in self.models.values()
            for c in s.classes
        ]

    def group(self, selected: list[str]) -> dict[str, list[str]]:
        """Group selected class names by the model that handles them."""
        out: dict[str, list[str]] = {}
        for c in selected:
            m = self.class_to_model.get(c)
            if m is None:
                raise KeyError(f"unknown class '{c}'")
            if not self.models[m].available():
                raise FileNotFoundError(
                    f"class '{c}' needs model '{m}' but weights '{self.models[m].weights}' are missing"
                )
            out.setdefault(m, []).append(c)
        return out
