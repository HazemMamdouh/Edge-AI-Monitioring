"""Model wrappers. One code path for PyTorch (.pt), ONNX (.onnx) and TensorRT (.engine)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    box: np.ndarray  # xyxy
    conf: float
    kpts: np.ndarray | None = None  # (17, 3) x, y, conf  (COCO order)
    cls: int = 0


def backend_name(weights: str | Path) -> str:
    p = Path(weights)
    s = p.suffix.lower()
    n = p.stem.lower()
    prec = "INT8" if "int8" in n else "FP16" if "fp16" in n else "FP32"
    if s == ".engine":
        return f"TensorRT {prec}"
    if s == ".onnx":
        return f"ONNX Runtime {prec}"
    return {".pt": "PyTorch FP32"}.get(s, s)


class _Yolo:
    def __init__(self, weights, task, imgsz=640, conf=0.25, iou=0.6, device=None, half=False,
                 classes=None):
        from ultralytics import YOLO

        if not Path(weights).exists():
            raise FileNotFoundError(
                f"model not found: {weights} (run `mantrap export` or check the path)")
        self.weights = str(weights)
        self.model = YOLO(self.weights, task=task)
        self.kw = dict(imgsz=imgsz, conf=conf, iou=iou, verbose=False, classes=classes)
        if device is not None:
            self.kw["device"] = device
        if half and self.weights.endswith(".pt"):
            self.kw["half"] = True
        self.backend = backend_name(weights)

    def _predict(self, frame):
        return self.model.predict(frame, **self.kw)[0]

    def warmup(self, frame: np.ndarray, n: int = 3) -> None:
        for _ in range(n):
            self._predict(frame)

    @property
    def device(self) -> str:
        try:
            return str(self.model.predictor.device)
        except Exception:
            return "?"


class PoseDetector(_Yolo):
    def __init__(self, weights, **kw):
        super().__init__(weights, "pose", **kw)

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        r = self._predict(frame)
        if r.boxes is None or len(r.boxes) == 0:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else [None] * len(boxes)
        return [Detection(b.astype(np.float32), float(c), k) for b, c, k in zip(boxes, confs, kpts)]


class ObjectDetector(_Yolo):
    def __init__(self, weights, **kw):
        super().__init__(weights, "detect", **kw)

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        r = self._predict(frame)
        if r.boxes is None or len(r.boxes) == 0:
            return []
        return [
            Detection(b.astype(np.float32), float(c), None, int(k))
            for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                               r.boxes.cls.cpu().numpy())
        ]

    @property
    def names(self) -> dict:
        return self.model.names
