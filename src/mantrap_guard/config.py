"""Configuration: built-in defaults deep-merged with an optional YAML file."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "model": {
        "pose": "models/yolo11n-pose.pt",  # .pt | .onnx | .engine
        "objects": None,  # optional COCO detector for left-behind objects
        "imgsz": 640,
        "conf": 0.15,  # kept low on purpose: the tracker's 2nd stage uses low-score boxes
        "iou": 0.6,
        "device": None,  # None = auto, "cpu", 0, ...
        "half": False,
        "object_classes": [24, 26, 28],  # COCO: backpack, handbag, suitcase
        "object_conf": 0.25,
    },
    "zone": {
        "kpt_conf": 0.35,  # min confidence for a leg keypoint to count
        "min_leg_points": 1,  # how many leg keypoints must be inside the mantrap
        "fallback_to_bbox": True,  # no confident leg keypoint -> use bbox bottom-centre
        "exit_debounce_frames": 5,  # frames outside before a track leaves the zone
    },
    "tracker": {
        "high_thresh": 0.5,
        "low_thresh": 0.15,
        "new_track_thresh": 0.55,
        "match_iou": 0.2,
        "match_iou_low": 0.4,
        "max_age": 30,
        "min_hits": 2,
    },
    "flow": {
        "method": "dis",  # dis | farneback
        "width": 480,  # flow is computed on a frame downscaled to this width
        "moving_px": 0.35,  # px/frame (at flow scale) for a pixel to count as moving
        "min_moving_frac": 0.12,  # fraction of ROI pixels that must move
        "ema": 0.5,  # smoothing of the per-track velocity
    },
    "rules": {
        # speeds are in body-heights per second, so they hold across the perspective
        "static": {"enabled": True, "seconds": 3.0, "speed": 0.12, "grace": 0.4},
        "wrong_way": {"enabled": True, "speed": 0.22, "cos": 0.5, "seconds": 0.5},
        "exit_intrusion": {"enabled": True, "latch": 3.0},
        "abandoned_object": {"enabled": True, "seconds": 5.0, "move_frac": 0.5},
        "alert_hold": 1.0,  # seconds the red overlay stays after the last alert
    },
    "viz": {
        "skeleton": True,
        "flow_arrows": True,
        "show_outside": True,  # draw (greyed) people who are not in the mantrap
        "overlay_alpha": 0.2,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg
