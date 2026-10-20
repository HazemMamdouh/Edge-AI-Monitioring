import numpy as np

from mantrap_guard.config import load_config
from mantrap_guard.rules import EXIT_INTRUSION, STATIC, WRONG_WAY, RuleEngine, legs_in_zone
from mantrap_guard.tracker import KalmanBox, Track
from mantrap_guard.zones import Zones

FPS = 10.0
# mantrap: 200x200 square; entry line at the top (y=20), exit line at the bottom (y=180)
ZONES = Zones(np.array([[0, 0], [200, 0], [200, 200], [0, 200]], np.float32),
              np.array([[20, 20], [180, 20]], np.float32),
              np.array([[0, 180], [200, 180]], np.float32), (200, 200))


def person(tid, foot_x, foot_y, kpt_conf=0.9, height=100):
    b = np.array([foot_x - 20, foot_y - height, foot_x + 20, foot_y], np.float32)
    k = np.zeros((17, 3), np.float32)
    for i in (13, 14, 15, 16):
        k[i] = (foot_x, foot_y - (30 if i < 15 else 0), kpt_conf)
    return Track(tid, KalmanBox(b), 0.9, k, box=b)


def run(engine, frames):
    """frames: list of (foot_x, foot_y, flow_vec_px_per_frame)"""
    out = []
    for i, (x, y, v) in enumerate(frames):
        out.append(engine.update([person(1, x, y)], {1: np.array(v, np.float32)}, i / FPS))
    return out


def rules_of(state):
    return set(state.persons[0].rules)


def test_legs_decide_membership_not_the_box():
    cfg = load_config()["zone"]
    inside = person(1, 100, 100)
    assert legs_in_zone(inside, ZONES, cfg["kpt_conf"], 1, False)
    outside = person(1, 300, 100)
    assert not legs_in_zone(outside, ZONES, cfg["kpt_conf"], 1, False)
    hidden = person(1, 100, 100, kpt_conf=0.05)  # legs not visible
    assert not legs_in_zone(hidden, ZONES, cfg["kpt_conf"], 1, False)
    assert legs_in_zone(hidden, ZONES, cfg["kpt_conf"], 1, True)  # bbox fallback


def test_static_fires_after_three_seconds_not_before():
    eng = RuleEngine(ZONES, load_config(), FPS)
    states = run(eng, [(100, 100, (0, 0))] * 40)
    assert STATIC not in rules_of(states[25])  # 2.5 s
    assert STATIC in rules_of(states[31])  # 3.1 s
    assert states[31].alarm
    assert [e.rule for s in states for e in s.events] == [STATIC]  # one event, on the rising edge


def test_static_ignored_outside_mantrap():
    eng = RuleEngine(ZONES, load_config(), FPS)
    states = run(eng, [(400, 100, (0, 0))] * 40)
    assert not any(s.alarm for s in states)
    assert states[-1].persons[0].status == "outside"


def test_moving_towards_entry_line_fires_wrong_way():
    eng = RuleEngine(ZONES, load_config(), FPS)
    # walking up (towards y=20) at 5 px/frame = 50 px/s = 0.5 body heights/s
    states = run(eng, [(100, 150 - 5 * i, (0, -5)) for i in range(15)])
    assert WRONG_WAY in rules_of(states[-1])
    assert STATIC not in rules_of(states[-1])


def test_moving_towards_exit_line_is_allowed():
    eng = RuleEngine(ZONES, load_config(), FPS)
    states = run(eng, [(100, 60 + 5 * i, (0, 5)) for i in range(15)])
    assert not any(s.alarm for s in states)
    assert states[-1].persons[0].status == "ok"


def test_entering_through_exit_line_is_flagged():
    eng = RuleEngine(ZONES, load_config(), FPS)
    states = run(eng, [(100, 195, (0, 0)), (100, 188, (0, 0)), (100, 172, (0, 0)), (100, 170, (0, 0))])
    assert EXIT_INTRUSION in rules_of(states[2])


def test_alarm_holds_then_clears():
    cfg = load_config()
    eng = RuleEngine(ZONES, cfg, FPS)
    frames = [(100, 100, (0, 0))] * 35 + [(400, 100, (0, 0))] * 30  # stands, then leaves the area
    states = run(eng, frames)
    assert states[34].alarm
    assert not states[-1].alarm


def test_object_left_in_mantrap():
    from mantrap_guard.rules import ABANDONED_OBJECT

    eng = RuleEngine(ZONES, load_config(), FPS)
    b = np.array([90, 90, 120, 120], np.float32)
    bag = Track(7, KalmanBox(b), 0.8, None, cls=24, box=b)
    states = [eng.update([], {}, i / FPS, [bag]) for i in range(60)]
    assert not states[40].alarm  # 4 s
    assert (ABANDONED_OBJECT, 7) in states[55].alerts  # 5.5 s


def test_dwell_time_counts_while_inside_and_is_reported_on_exit():
    eng = RuleEngine(ZONES, load_config(), FPS)
    frames = [(400, 100, (0, 0))] * 5 + [(100, 100, (0, 0))] * 20 + [(400, 100, (0, 0))] * 10
    states = run(eng, frames)
    assert states[4].persons[0].dwell_time == 0  # still outside
    assert abs(states[15].persons[0].dwell_time - 1.0) < 1e-6  # entered at frame 5, 10 frames on
    visits = [v for s in states for v in s.visits]
    assert len(visits) == 1 and visits[0].track_id == 1 and visits[0].reason == "left"
    assert abs(visits[0].duration - 1.9) < 1e-6  # frames 5..24 inside
    assert states[-1].persons[0].dwell_time == 0


def test_dwell_restarts_on_second_visit_and_open_visit_closes_at_video_end():
    eng = RuleEngine(ZONES, load_config(), FPS)
    frames = ([(100, 100, (0, 0))] * 10 + [(400, 100, (0, 0))] * 10 + [(100, 100, (0, 0))] * 5)
    states = run(eng, frames)
    assert abs(states[-1].persons[0].dwell_time - 0.4) < 1e-6
    end = eng.finish()
    assert [v.reason for v in end] == ["video_end"] and abs(end[0].duration - 0.4) < 1e-6


def test_visit_closed_when_track_is_lost_inside():
    eng = RuleEngine(ZONES, load_config(), FPS)
    run(eng, [(100, 100, (0, 0))] * 10)
    closed = []
    for i in range(10, 60):
        closed += eng.update([], {}, i / FPS).visits
    assert [v.reason for v in closed] == ["track_lost"] and abs(closed[0].duration - 0.9) < 1e-6
