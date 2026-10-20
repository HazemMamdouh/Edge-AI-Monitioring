import numpy as np

from mantrap_guard.detector import Detection
from mantrap_guard.tracker import ByteTracker, iou_matrix


def box(x, y, w=40, h=100):
    return np.array([x, y, x + w, y + h], np.float32)


def test_iou():
    m = iou_matrix(np.stack([box(0, 0)]), np.stack([box(0, 0), box(200, 0)]))
    assert m[0, 0] > 0.99 and m[0, 1] == 0


def test_ids_stay_stable_for_two_people_crossing_slowly():
    tr = ByteTracker(min_hits=1)
    ids = []
    for i in range(30):
        dets = [Detection(box(10 + 4 * i, 50), 0.9), Detection(box(400 - 4 * i, 50), 0.9)]
        out = sorted(tr.update(dets), key=lambda t: t.box[0])
        ids.append([t.id for t in out])
    assert all(x == ids[0] for x in ids) and len(set(ids[0])) == 2


def test_low_score_detection_keeps_track_alive():
    tr = ByteTracker(min_hits=1)
    tr.update([Detection(box(10, 10), 0.9)])
    out = tr.update([Detection(box(12, 10), 0.2)])  # below high_thresh, recovered in stage 2
    assert [t.id for t in out] == [1]


def test_low_score_detection_does_not_start_a_track():
    tr = ByteTracker(min_hits=1)
    assert tr.update([Detection(box(10, 10), 0.2)]) == []


def test_track_dropped_after_max_age():
    tr = ByteTracker(max_age=3, min_hits=1)
    tr.update([Detection(box(10, 10), 0.9)])
    for _ in range(5):
        tr.update([])
    assert tr.tracks == []
