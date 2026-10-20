"""A compact ByteTrack-style tracker: constant-velocity Kalman filter + two-stage IoU matching.

Stage 1 matches confident detections to tracks. Stage 2 gives the still-unmatched tracks a
second chance against low-score detections, which is what keeps IDs alive through partial
occlusion (a person half hidden by the gate, or crouching).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detection


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _xyxy_to_cxcywh(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]], np.float64)


def _cxcywh_to_xyxy(s):
    return np.array([s[0] - s[2] / 2, s[1] - s[3] / 2, s[0] + s[2] / 2, s[1] + s[3] / 2],
                    np.float32)


class KalmanBox:
    """State [cx, cy, w, h, vcx, vcy, vw, vh]; noise scales with the box height."""

    F = np.eye(8)
    F[:4, 4:] = np.eye(4)
    H = np.eye(4, 8)

    def __init__(self, box):
        self.x = np.zeros(8)
        self.x[:4] = _xyxy_to_cxcywh(box)
        h = self.x[3]
        self.P = np.diag([(0.1 * h) ** 2] * 4 + [(0.2 * h) ** 2] * 4)

    def predict(self):
        h = max(self.x[3], 1.0)
        q = np.diag([(0.05 * h) ** 2] * 4 + [(0.02 * h) ** 2] * 4)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + q
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)

    def update(self, box):
        z = _xyxy_to_cxcywh(box)
        h = max(self.x[3], 1.0)
        r = np.diag([(0.05 * h) ** 2] * 4)
        s = self.H @ self.P @ self.H.T + r
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ (z - self.H @ self.x)
        self.P = (np.eye(8) - k @ self.H) @ self.P

    @property
    def box(self):
        return _cxcywh_to_xyxy(self.x)


@dataclass
class Track:
    id: int
    kf: KalmanBox
    conf: float
    kpts: np.ndarray | None = None
    cls: int = 0
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    box: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.box is None:
            self.box = self.kf.box


class ByteTracker:
    def __init__(self, high_thresh=0.5, low_thresh=0.15, new_track_thresh=0.55, match_iou=0.2,
                 match_iou_low=0.4, max_age=30, min_hits=2):
        self.high, self.low, self.new = high_thresh, low_thresh, new_track_thresh
        self.match_iou, self.match_iou_low = match_iou, match_iou_low
        self.max_age, self.min_hits = max_age, min_hits
        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame = 0

    @staticmethod
    def _match(tracks: list[Track], dets: list[Detection], min_iou: float):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        iou = iou_matrix(np.stack([t.kf.box for t in tracks]), np.stack([d.box for d in dets]))
        rows, cols = linear_sum_assignment(1.0 - iou)
        pairs = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]
        mr, mc = {r for r, _ in pairs}, {c for _, c in pairs}
        return (pairs, [i for i in range(len(tracks)) if i not in mr],
                [j for j in range(len(dets)) if j not in mc])

    def update(self, dets: list[Detection]) -> list[Track]:
        self.frame += 1
        for t in self.tracks:
            t.kf.predict()
            t.age += 1
            t.time_since_update += 1

        high = [d for d in dets if d.conf >= self.high]
        low = [d for d in dets if self.low <= d.conf < self.high]

        def apply(t: Track, d: Detection):
            t.kf.update(d.box)
            t.box, t.conf, t.kpts, t.cls = d.box, d.conf, d.kpts, d.cls
            t.hits += 1
            t.time_since_update = 0

        pairs, un_t, un_d = self._match(self.tracks, high, self.match_iou)
        for ti, di in pairs:
            apply(self.tracks[ti], high[di])

        rest = [self.tracks[i] for i in un_t]
        pairs2, _, _ = self._match(rest, low, self.match_iou_low)
        for ti, di in pairs2:
            apply(rest[ti], low[di])

        for di in un_d:
            d = high[di]
            if d.conf >= self.new:
                self.tracks.append(Track(self._next_id, KalmanBox(d.box), d.conf, d.kpts, d.cls,
                                         box=d.box))
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks
                if t.time_since_update == 0 and (t.hits >= self.min_hits or self.frame <= self.min_hits)]
