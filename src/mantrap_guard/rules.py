"""Rule engine: turns tracks + optical-flow vectors + zones into warnings.

All timing uses video time (frame index / fps), so results do not depend on how fast the
machine processes the video. Speeds are normalised by the person's box height
("body-heights per second") so one threshold works near and far from the camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tracker import Track
from .zones import Zones, crosses_inward, nearest_on_segment

KNEES, ANKLES = (13, 14), (15, 16)

STATIC = "STATIC"
WRONG_WAY = "WRONG_WAY"
EXIT_INTRUSION = "EXIT_INTRUSION"
ABANDONED_OBJECT = "ABANDONED_OBJECT"

MESSAGES = {
    STATIC: "stationary in mantrap",
    WRONG_WAY: "moving towards entry line",
    EXIT_INTRUSION: "entered through exit line",
    ABANDONED_OBJECT: "object left in mantrap",
}


def foot_point(track: Track, kpt_conf: float) -> tuple[np.ndarray, bool]:
    """Ground contact point: mean of confident ankles, else knees, else bbox bottom-centre."""
    k = track.kpts
    if k is not None:
        for idx in (ANKLES, KNEES):
            pts = [k[i, :2] for i in idx if k[i, 2] >= kpt_conf]
            if pts:
                return np.mean(pts, axis=0).astype(np.float32), True
    b = track.box
    return np.array([(b[0] + b[2]) / 2, b[3]], np.float32), False


def legs_in_zone(track: Track, zones: Zones, kpt_conf: float, min_points: int,
                 fallback: bool) -> bool:
    """The membership test: are the person's leg keypoints inside the mantrap polygon?"""
    k = track.kpts
    legs = [] if k is None else [k[i, :2] for i in KNEES + ANKLES if k[i, 2] >= kpt_conf]
    if legs:
        return sum(zones.contains(p) for p in legs) >= min_points
    if fallback:
        b = track.box
        return zones.contains(((b[0] + b[2]) / 2, b[3]))
    return False


@dataclass
class Event:
    t: float
    frame: int
    rule: str
    track_id: int
    message: str

    def as_dict(self):
        return {"t": round(self.t, 2), "frame": self.frame, "rule": self.rule,
                "track_id": self.track_id, "message": self.message}


@dataclass
class Visit:
    """One stay of one tracked person inside the mantrap."""

    track_id: int
    enter_t: float
    exit_t: float
    enter_frame: int
    exit_frame: int
    reason: str  # left | track_lost | video_end

    @property
    def duration(self) -> float:
        return self.exit_t - self.enter_t

    def as_dict(self):
        return {"track_id": self.track_id, "enter_t": round(self.enter_t, 2),
                "exit_t": round(self.exit_t, 2), "duration": round(self.duration, 2),
                "enter_frame": self.enter_frame, "exit_frame": self.exit_frame,
                "reason": self.reason}


@dataclass
class PersonState:
    track: Track
    foot: np.ndarray
    in_zone: bool
    velocity: np.ndarray  # px / s
    speed: float  # body-heights / s
    toward_entry: float  # cosine between motion and the direction to the entry line
    static_time: float
    status: str  # outside | ok | watch | alert
    rules: list[str] = field(default_factory=list)
    dwell_time: float = 0.0  # seconds inside the mantrap during the current visit


@dataclass
class ObjectState:
    track: Track
    in_zone: bool
    static_time: float
    alert: bool


@dataclass
class FrameState:
    t: float
    frame: int
    persons: list[PersonState]
    objects: list[ObjectState]
    alerts: list[tuple[str, int]]  # (rule, track id)
    alarm: bool  # red overlay on/off (includes hold time)
    alarm_rules: list[str]  # rules behind the overlay (kept during the hold time)
    events: list[Event]  # events that started on this frame
    visits: list[Visit] = field(default_factory=list)  # mantrap visits that ended on this frame


@dataclass
class _Mem:
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    prev_foot: np.ndarray | None = None
    in_zone: bool = False
    out_count: int = 0
    static_t0: float | None = None
    moving_t0: float | None = None
    toward_t0: float | None = None
    intrusion_until: float = -1.0
    active: set = field(default_factory=set)
    last_seen: int = 0
    enter_t: float | None = None  # start of the current mantrap visit
    enter_frame: int = 0
    last_in_t: float = 0.0  # last time the legs were seen inside
    last_in_frame: int = 0


@dataclass
class _ObjMem:
    anchor: np.ndarray
    t0: float
    active: bool = False
    last_seen: int = 0


class RuleEngine:
    def __init__(self, zones: Zones, cfg: dict, fps: float):
        self.z, self.fps = zones, float(fps)
        self.zc, self.rc, self.ema = cfg["zone"], cfg["rules"], cfg["flow"]["ema"]
        self.mem: dict[int, _Mem] = {}
        self.omem: dict[int, _ObjMem] = {}
        self.exit_normal = zones.inward_normal(zones.exit_line)
        self.alarm_until = -1.0
        self.alarm_rules: list[str] = []
        self.frame = 0
        self.lost_frames = cfg["tracker"]["max_age"]  # unseen this long -> the visit is closed
        self._closed: list[Visit] = []

    def _close_visit(self, tid: int, m: _Mem, reason: str) -> None:
        if m.enter_t is not None:
            self._closed.append(Visit(tid, m.enter_t, m.last_in_t, m.enter_frame,
                                      m.last_in_frame, reason))
        m.enter_t, m.in_zone, m.out_count = None, False, 0

    def finish(self) -> list[Visit]:
        """Close the visits still open when the video ends."""
        self._closed = []
        for tid, m in self.mem.items():
            self._close_visit(tid, m, "video_end")
        return self._closed

    # ---------------------------------------------------------------- people
    def _person(self, tr: Track, flow_vec: np.ndarray, t: float, events: list[Event]) -> PersonState:
        m = self.mem.setdefault(tr.id, _Mem())
        m.last_seen = self.frame
        rc = self.rc
        foot, _ = foot_point(tr, self.zc["kpt_conf"])
        body_h = max(float(tr.box[3] - tr.box[1]), 1.0)

        # zone membership from leg keypoints, with a short debounce on leaving
        inside = legs_in_zone(tr, self.z, self.zc["kpt_conf"], self.zc["min_leg_points"],
                              self.zc["fallback_to_bbox"])
        if inside:
            if m.enter_t is None:  # a new visit starts
                m.enter_t, m.enter_frame = t, self.frame
            m.in_zone, m.out_count = True, 0
            m.last_in_t, m.last_in_frame = t, self.frame
        elif m.in_zone:
            m.out_count += 1
            if m.out_count > self.zc["exit_debounce_frames"]:
                self._close_visit(tr.id, m, "left")  # ends at the last frame seen inside
        dwell = t - m.enter_t if m.enter_t is not None else 0.0

        # motion from optical flow (px/frame -> px/s), smoothed
        m.vel = self.ema * m.vel + (1 - self.ema) * np.asarray(flow_vec, np.float32)
        vel = m.vel * self.fps
        speed = float(np.linalg.norm(vel)) / body_h
        to_entry = nearest_on_segment(foot, self.z.entry_line) - foot
        cos = float(np.dot(vel, to_entry) /
                    (np.linalg.norm(vel) * np.linalg.norm(to_entry) + 1e-6))

        # exit-line crossing is checked for everyone: it happens on the way in
        if rc["exit_intrusion"]["enabled"] and m.prev_foot is not None and crosses_inward(
                m.prev_foot, foot, self.z.exit_line, self.exit_normal):
            m.intrusion_until = t + rc["exit_intrusion"]["latch"]
        m.prev_foot = foot

        rules: list[str] = []
        static_time = 0.0
        watch = False
        if m.in_zone:
            st = rc["static"]
            if speed < st["speed"]:
                m.moving_t0 = None
                if m.static_t0 is None:
                    m.static_t0 = t
            elif m.static_t0 is not None:
                if m.moving_t0 is None:
                    m.moving_t0 = t
                if t - m.moving_t0 > st["grace"]:
                    m.static_t0 = m.moving_t0 = None
            if m.static_t0 is not None:
                static_time = t - m.static_t0
                watch |= static_time > 0.3 * st["seconds"]
                if st["enabled"] and static_time >= st["seconds"]:
                    rules.append(STATIC)

            ww = rc["wrong_way"]
            if speed >= ww["speed"] and cos >= ww["cos"]:
                if m.toward_t0 is None:
                    m.toward_t0 = t
                watch = True
                if ww["enabled"] and t - m.toward_t0 >= ww["seconds"]:
                    rules.append(WRONG_WAY)
            else:
                m.toward_t0 = None

            if t <= m.intrusion_until:
                rules.append(EXIT_INTRUSION)
        else:
            m.static_t0 = m.moving_t0 = m.toward_t0 = None

        for r in rules:
            if r not in m.active:
                events.append(Event(t, self.frame, r, tr.id, f"person #{tr.id} {MESSAGES[r]}"))
        m.active = set(rules)

        status = "outside" if not m.in_zone else "alert" if rules else "watch" if watch else "ok"
        return PersonState(tr, foot, m.in_zone, vel, speed, cos, static_time, status, rules, dwell)

    # --------------------------------------------------------------- objects
    def _object(self, tr: Track, t: float, events: list[Event]) -> ObjectState:
        oc = self.rc["abandoned_object"]
        b = tr.box
        c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], np.float32)
        size = max(float(max(b[2] - b[0], b[3] - b[1])), 1.0)
        m = self.omem.get(tr.id)
        if m is None or np.linalg.norm(c - m.anchor) > oc["move_frac"] * size:
            m = self.omem[tr.id] = _ObjMem(c, t)
        m.last_seen = self.frame
        in_zone = self.z.contains(((b[0] + b[2]) / 2, b[3]))
        static_time = t - m.t0 if in_zone else 0.0
        if not in_zone:
            m.t0 = t
        alert = bool(oc["enabled"] and in_zone and static_time >= oc["seconds"])
        if alert and not m.active:
            events.append(Event(t, self.frame, ABANDONED_OBJECT, tr.id,
                                f"object #{tr.id} {MESSAGES[ABANDONED_OBJECT]}"))
        m.active = alert
        return ObjectState(tr, in_zone, static_time, alert)

    # ------------------------------------------------------------------ step
    def update(self, tracks: list[Track], flow_vecs: dict[int, np.ndarray], t: float,
               object_tracks: list[Track] | None = None) -> FrameState:
        self.frame += 1
        self._closed = []
        events: list[Event] = []
        persons = [self._person(tr, flow_vecs.get(tr.id, np.zeros(2)), t, events) for tr in tracks]
        objects = [self._object(tr, t, events) for tr in (object_tracks or [])]

        alerts = [(r, p.track.id) for p in persons for r in p.rules]
        alerts += [(ABANDONED_OBJECT, o.track.id) for o in objects if o.alert]
        if alerts:
            self.alarm_until = t + self.rc["alert_hold"]
            self.alarm_rules = sorted({r for r, _ in alerts})

        # a person whose track disappeared inside the mantrap: close the visit at the last sighting
        for tid, m in self.mem.items():
            if m.enter_t is not None and self.frame - m.last_seen > self.lost_frames:
                self._close_visit(tid, m, "track_lost")

        # forget tracks that have been gone for a while
        for d in (self.mem, self.omem):
            for k in [k for k, v in d.items() if self.frame - v.last_seen > 150]:
                del d[k]
        alarm = t <= self.alarm_until
        return FrameState(t, self.frame, persons, objects, alerts, alarm,
                          self.alarm_rules if alarm else [], events, self._closed)
