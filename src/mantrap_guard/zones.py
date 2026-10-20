"""Areas of interest: the mantrap polygon, the entry line and the exit line."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Zones:
    mantrap: np.ndarray  # (N, 2) float32 polygon
    entry_line: np.ndarray  # (2, 2)
    exit_line: np.ndarray  # (2, 2)
    frame_size: tuple[int, int]  # (w, h) the annotation was made on

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path, frame_size: tuple[int, int] | None = None) -> "Zones":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        z = cls(
            mantrap=np.asarray(d["mantrap"], np.float32).reshape(-1, 2),
            entry_line=np.asarray(d["entry_line"], np.float32).reshape(2, 2),
            exit_line=np.asarray(d["exit_line"], np.float32).reshape(2, 2),
            frame_size=tuple(d["frame_size"]),
        )
        if len(z.mantrap) < 3:
            raise ValueError("mantrap polygon needs at least 3 points")
        return z.scaled_to(frame_size) if frame_size else z

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        d = {
            "frame_size": list(self.frame_size),
            "mantrap": np.round(self.mantrap, 1).tolist(),
            "entry_line": np.round(self.entry_line, 1).tolist(),
            "exit_line": np.round(self.exit_line, 1).tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    def scaled_to(self, frame_size: tuple[int, int]) -> "Zones":
        """Rescale an annotation made at one resolution to another."""
        if tuple(frame_size) == tuple(self.frame_size):
            return self
        s = np.array(
            [frame_size[0] / self.frame_size[0], frame_size[1] / self.frame_size[1]], np.float32
        )
        return Zones(self.mantrap * s, self.entry_line * s, self.exit_line * s, tuple(frame_size))

    # ------------------------------------------------------------ geometry
    def contains(self, pt) -> bool:
        return cv2.pointPolygonTest(self.mantrap, (float(pt[0]), float(pt[1])), False) >= 0

    @property
    def centroid(self) -> np.ndarray:
        return self.mantrap.mean(axis=0)

    def inward_normal(self, line: np.ndarray) -> np.ndarray:
        """Unit normal of `line` pointing towards the inside of the mantrap."""
        d = line[1] - line[0]
        n = np.array([-d[1], d[0]], np.float32)
        n /= np.linalg.norm(n) + 1e-9
        if np.dot(self.centroid - line.mean(axis=0), n) < 0:
            n = -n
        return n


def nearest_on_segment(pt, seg: np.ndarray) -> np.ndarray:
    a, b = seg[0], seg[1]
    ab = b - a
    t = float(np.dot(np.asarray(pt, np.float32) - a, ab) / (np.dot(ab, ab) + 1e-9))
    return a + np.clip(t, 0.0, 1.0) * ab


def signed_distance(pt, line: np.ndarray, normal: np.ndarray) -> float:
    return float(np.dot(np.asarray(pt, np.float32) - line[0], normal))


def crosses_inward(p0, p1, line: np.ndarray, normal: np.ndarray, margin: float = 0.15) -> bool:
    """True if the move p0 -> p1 crosses `line` from its outer side to its inner side."""
    d0, d1 = signed_distance(p0, line, normal), signed_distance(p1, line, normal)
    if not (d0 < 0 <= d1):
        return False
    t = d0 / (d0 - d1)
    x = np.asarray(p0, np.float32) + t * (np.asarray(p1, np.float32) - np.asarray(p0, np.float32))
    ab = line[1] - line[0]
    u = float(np.dot(x - line[0], ab) / (np.dot(ab, ab) + 1e-9))
    return -margin <= u <= 1 + margin
