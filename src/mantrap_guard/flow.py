"""Dense optical flow -> one motion vector per tracked person.

Flow is computed once per frame on a downscaled grey image (DIS by default: fast enough for
an embedded CPU core while the GPU runs the networks). For each track we take the median
flow of the pixels that actually move inside the box, so the static background inside the
box does not drag the estimate towards zero.
"""

from __future__ import annotations

import cv2
import numpy as np


class FlowEstimator:
    def __init__(self, method="dis", width=480, moving_px=0.35, min_moving_frac=0.12, **_):
        self.width, self.moving_px, self.min_frac = width, moving_px, min_moving_frac
        self.method = method
        self._dis = None
        if method == "dis":
            self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        elif method != "farneback":
            raise ValueError(f"unknown flow method: {method}")
        self.prev: np.ndarray | None = None
        self.flow: np.ndarray | None = None
        self.scale = 1.0

    def update(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        self.scale = min(1.0, self.width / w)
        small = cv2.resize(frame, (int(w * self.scale), int(h * self.scale)),
                           interpolation=cv2.INTER_AREA) if self.scale < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self.prev is None or self.prev.shape != gray.shape:
            self.flow = None
        elif self._dis is not None:
            self.flow = self._dis.calc(self.prev, gray, None)
        else:
            self.flow = cv2.calcOpticalFlowFarneback(self.prev, gray, None, 0.5, 3, 15, 3, 5,
                                                     1.2, 0)
        self.prev = gray

    def vector(self, box) -> np.ndarray:
        """Median motion (px/frame, full-resolution pixels) of the moving pixels in `box`."""
        if self.flow is None:
            return np.zeros(2, np.float32)
        fh, fw = self.flow.shape[:2]
        x1, y1, x2, y2 = (np.asarray(box, np.float32) * self.scale)
        # shrink the box a little: its border is mostly background
        mx, my = 0.12 * (x2 - x1), 0.06 * (y2 - y1)
        x1, x2 = int(max(0, x1 + mx)), int(min(fw, x2 - mx))
        y1, y2 = int(max(0, y1 + my)), int(min(fh, y2 - my))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return np.zeros(2, np.float32)
        roi = self.flow[y1:y2, x1:x2].reshape(-1, 2)
        moving = roi[np.hypot(roi[:, 0], roi[:, 1]) > self.moving_px]
        if len(moving) < self.min_frac * len(roi):
            return np.zeros(2, np.float32)
        return (np.median(moving, axis=0) / self.scale).astype(np.float32)
