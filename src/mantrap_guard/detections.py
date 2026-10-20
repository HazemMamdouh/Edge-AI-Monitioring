"""Full per-frame detection record of a run: `detections.jsonl`.

JSON Lines: line 1 is a header, then one JSON object per processed frame. One line per
frame keeps the file streamable (a long video never has to fit in memory) and every line
is valid JSON on its own.

Header  {"type": "header", "schema": 1, "video", "start", "fps", "size", "model", "backend",
         "zones": {...}, "config": {...}, "keypoints": [17 COCO names], ...}

Frame   {"type": "frame", "frame": i, "source_frame": start + i, "t": seconds,
         "detections": [{"box": [x1,y1,x2,y2], "conf", "kpts": [[x,y,c] x17]}],  raw pose output
         "objects_raw": [{"box", "conf", "cls"}],                               raw object output
         "persons": [{"id", "box", "conf", "kpts", "foot", "in_zone", "dwell", "static",
                      "speed", "velocity", "toward_entry", "flow", "status", "rules"}],
         "objects": [{"id", "box", "conf", "cls", "in_zone", "static", "alert"}],
         "alarm", "alarm_rules", "alerts": [[rule, id]], "events": [...], "visits": [...]}

Coordinates are pixels of the source video. `flow` is the raw optical-flow vector
(px/frame) of the track: together with the tracks it is everything the rule engine needs,
so the rules can be re-run offline with other thresholds (`mantrap render --reapply-rules`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .rules import FrameState, ObjectState, PersonState
from .tracker import KalmanBox, Track
from .zones import Zones

SCHEMA = 1
KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
             "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
             "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]


def _r(a, nd=1):
    return np.round(np.asarray(a, np.float64), nd).tolist()


def _kpts(k):
    if k is None:
        return None
    k = np.asarray(k, np.float64)
    return np.column_stack([np.round(k[:, :2], 1), np.round(k[:, 2], 3)]).tolist()


def zones_dict(z: Zones) -> dict:
    return {"frame_size": list(z.frame_size), "mantrap": _r(z.mantrap), "entry_line": _r(z.entry_line),
            "exit_line": _r(z.exit_line)}


class DetectionWriter:
    def __init__(self, path: str | Path, header: dict):
        self.f = open(path, "w", encoding="utf-8")
        self._write({"type": "header", "schema": SCHEMA, "keypoints": KEYPOINTS, **header})

    def _write(self, obj):
        self.f.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def write(self, i: int, source_frame: int, t: float, dets, odets, state: FrameState,
              flow_vecs: dict) -> None:
        self._write({
            "type": "frame", "frame": i, "source_frame": source_frame, "t": round(t, 3),
            "detections": [{"box": _r(d.box), "conf": round(d.conf, 3), "kpts": _kpts(d.kpts)}
                           for d in dets],
            "objects_raw": [{"box": _r(d.box), "conf": round(d.conf, 3), "cls": d.cls} for d in odets],
            "persons": [{
                "id": p.track.id, "box": _r(p.track.box), "conf": round(p.track.conf, 3),
                "kpts": _kpts(p.track.kpts), "foot": _r(p.foot), "in_zone": p.in_zone,
                "dwell": round(p.dwell_time, 2), "static": round(p.static_time, 2),
                "speed": round(p.speed, 3), "velocity": _r(p.velocity, 2),
                "toward_entry": round(p.toward_entry, 3),
                "flow": _r(flow_vecs.get(p.track.id, (0, 0)), 3),
                "status": p.status, "rules": p.rules} for p in state.persons],
            "objects": [{"id": o.track.id, "box": _r(o.track.box), "conf": round(o.track.conf, 3),
                         "cls": o.track.cls, "in_zone": o.in_zone, "static": round(o.static_time, 2),
                         "alert": o.alert} for o in state.objects],
            "alarm": state.alarm, "alarm_rules": state.alarm_rules,
            "alerts": [list(a) for a in state.alerts],
            "events": [e.as_dict() for e in state.events],
            "visits": [v.as_dict() for v in state.visits],
        })

    def close(self):
        self.f.close()


# ------------------------------------------------------------------ reading
def read_detections(path: str | Path) -> tuple[dict, Iterator[dict]]:
    """-> (header, iterator over frame records)"""
    f = open(path, "r", encoding="utf-8")
    header = json.loads(f.readline())
    if header.get("type") != "header":
        raise ValueError(f"{path}: first line is not a detections header")

    def frames():
        with f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    return header, frames()


def zones_from_header(header: dict) -> Zones:
    z = header["zones"]
    return Zones(np.asarray(z["mantrap"], np.float32), np.asarray(z["entry_line"], np.float32),
                 np.asarray(z["exit_line"], np.float32), tuple(z["frame_size"]))


def _track(d: dict) -> Track:
    box = np.asarray(d["box"], np.float32)
    k = None if d.get("kpts") is None else np.asarray(d["kpts"], np.float32)
    return Track(int(d["id"]), KalmanBox(box), float(d["conf"]), k, int(d.get("cls", 0)), box=box)


def tracks_from_record(rec: dict) -> tuple[list[Track], dict, list[Track]]:
    """Tracks, flow vectors and object tracks of one frame: the rule engine's inputs."""
    persons = [_track(p) for p in rec["persons"]]
    flow = {int(p["id"]): np.asarray(p["flow"], np.float32) for p in rec["persons"]}
    objects = [_track(o) for o in rec["objects"]]
    return persons, flow, objects


def state_from_record(rec: dict) -> FrameState:
    """Rebuild the FrameState the visualiser needs, exactly as recorded during the run."""
    persons = [PersonState(_track(p), np.asarray(p["foot"], np.float32), p["in_zone"],
                           np.asarray(p["velocity"], np.float32), p["speed"], p["toward_entry"],
                           p["static"], p["status"], list(p["rules"]), p["dwell"])
               for p in rec["persons"]]
    objects = [ObjectState(_track(o), o["in_zone"], o["static"], o["alert"]) for o in rec["objects"]]
    return FrameState(rec["t"], rec["frame"], persons, objects,
                      [tuple(a) for a in rec["alerts"]], rec["alarm"], list(rec["alarm_rules"]), [])
