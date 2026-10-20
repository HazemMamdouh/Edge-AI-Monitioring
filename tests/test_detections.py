import numpy as np

from mantrap_guard.config import load_config
from mantrap_guard.detections import (DetectionWriter, read_detections, state_from_record,
                                      tracks_from_record, zones_dict, zones_from_header)
from mantrap_guard.detector import Detection
from mantrap_guard.rules import STATIC, RuleEngine
from mantrap_guard.tracker import KalmanBox, Track
from mantrap_guard.zones import Zones

ZONES = Zones(np.array([[0, 0], [200, 0], [200, 200], [0, 200]], np.float32),
              np.array([[20, 20], [180, 20]], np.float32),
              np.array([[0, 180], [200, 180]], np.float32), (200, 200))


def person(x=100.0, y=100.0):
    b = np.array([x - 20, y - 100, x + 20, y], np.float32)
    k = np.zeros((17, 3), np.float32)
    k[13:17] = (x, y, 0.9)
    return Track(1, KalmanBox(b), 0.9, k, box=b)


def write_run(path, n=40):
    eng = RuleEngine(ZONES, load_config(), 10.0)
    w = DetectionWriter(path, {"video": "v.mp4", "fps": 10.0, "zones": zones_dict(ZONES),
                               "model": "m", "backend": "b", "config": load_config()})
    states = []
    for i in range(n):
        tr = person()
        st = eng.update([tr], {1: np.zeros(2)}, i / 10)
        w.write(i, i, i / 10, [Detection(tr.box, 0.9, tr.kpts)], [], st, {1: np.zeros(2)})
        states.append(st)
    w.close()
    return states


def test_roundtrip_keeps_what_the_visualiser_needs(tmp_path):
    states = write_run(tmp_path / "d.jsonl")
    header, frames = read_detections(tmp_path / "d.jsonl")
    assert header["schema"] == 1 and len(header["keypoints"]) == 17
    assert np.allclose(zones_from_header(header).mantrap, ZONES.mantrap)
    recs = list(frames)
    assert len(recs) == 40 and len(recs[0]["detections"]) == 1
    s = state_from_record(recs[35])
    p, q = s.persons[0], states[35].persons[0]
    assert (p.track.id, p.status, p.rules, p.in_zone) == (q.track.id, q.status, q.rules, q.in_zone)
    assert abs(p.dwell_time - q.dwell_time) < 0.01 and s.alarm == states[35].alarm
    assert np.allclose(p.track.kpts, q.track.kpts, atol=0.05)


def test_rules_can_be_rerun_offline_with_other_thresholds(tmp_path):
    write_run(tmp_path / "d.jsonl")
    _, frames = read_detections(tmp_path / "d.jsonl")
    cfg = load_config(overrides={"rules": {"static": {"seconds": 1.0}}})
    eng = RuleEngine(ZONES, cfg, 10.0)
    fired = []
    for rec in frames:
        persons, flow, objects = tracks_from_record(rec)
        st = eng.update(persons, flow, rec["t"], objects)
        fired += [(e.rule, e.frame) for e in st.events]
    assert fired == [(STATIC, 11)]  # 1 s instead of 3 s
