"""Per-frame visualisation: zones, skeletons, IDs with time in mantrap, flow arrows, rule states
and the red warning overlay. Dark-green HUD style; text is rendered with TrueType fonts
(see theme.py) in one pass at the end of each frame."""

from __future__ import annotations

import cv2
import numpy as np

from . import theme as T
from .rules import MESSAGES, FrameState, PersonState
from .zones import Zones

B = T.hex_to_bgr
C_ZONE, C_ZONE_EDGE = B("#0f5a3e"), B(T.ACCENT_DIM)
C_ENTRY, C_EXIT = B(T.ACCENT), B(T.AMBER)
C_RED, C_PLATE, C_TEXT2 = B(T.RED), B(T.SURFACE), B(T.TEXT_2)
STATUS = {"outside": (140, 150, 145), "ok": B(T.ACCENT), "watch": B(T.AMBER), "alert": C_RED}

SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
            (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
LEG_KPTS = {13, 14, 15, 16}


def fmt_duration(sec: float) -> str:
    """12.3s below one minute, 1:05 above."""
    return f"{sec:.1f}s" if sec < 60 else f"{int(sec // 60)}:{int(sec % 60):02d}"


def _blend_rect(img, x1, y1, x2, y2, color, alpha):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0, dst=roi)


def _brackets(img, x1, y1, x2, y2, color, t, frac=0.22):
    """Corner brackets instead of a full rectangle: the HUD look."""
    lx, ly = int((x2 - x1) * frac), int((y2 - y1) * frac)
    lx, ly = max(lx, 6), max(min(ly, lx * 2), 6)
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * lx, cy), color, t, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * ly), color, t, cv2.LINE_AA)


_GLYPHS = T.GlyphCache()


class _Text:
    """Collects text while drawing and renders it at the end of the frame, so text drawn
    under the red tint stays crisp. Uses cached glyphs (fast enough for every frame)."""

    def __init__(self):
        self.ops: list[tuple] = []

    @staticmethod
    def size(text, f):
        return _GLYPHS.size(text, f)

    def add(self, xy, text, f, color_bgr):
        self.ops.append((xy, text, f, color_bgr))

    def render(self, img):
        for xy, text, f, col in self.ops:
            _GLYPHS.draw(img, xy, text, f, col)
        return img


class Visualizer:
    def __init__(self, zones: Zones, cfg: dict, static_seconds: float):
        self.z, self.cfg, self.static_seconds = zones, cfg, static_seconds
        h = zones.frame_size[1]
        self.s = max(0.9, h / 480.0)  # scale drawing with resolution
        px = lambda v: max(10, int(round(v * self.s)))  # noqa: E731
        self.f_tag = T.font("display", px(15))
        self.f_tag_small = T.font("display_medium", px(12))
        self.f_hud_big = T.font("display", px(22))
        self.f_hud = T.font("display_medium", px(13))
        self.f_banner = T.font("display", px(17))

    # ---------------------------------------------------------------- pieces
    def _tag(self, txt: _Text, img, x, y_bottom, lines, accent, align_right=False):
        """Dark translucent plate with a coloured accent bar; lines = [(text, font, bgr)]."""
        pad, gap = int(5 * self.s), int(2 * self.s)
        sizes = [txt.size(s, f) for s, f, _ in lines]
        w = max(s[0] for s in sizes) + 2 * pad + int(3 * self.s)
        h = sum(s[1] for s in sizes) + gap * (len(lines) - 1) + 2 * pad
        x = int(x - w) if align_right else int(x)
        x = max(0, min(x, img.shape[1] - w))
        y = max(0, int(y_bottom - h))
        _blend_rect(img, x, y, x + w, y + h, C_PLATE, 0.78)
        cv2.rectangle(img, (x, y), (x + max(2, int(3 * self.s)) - 1, y + h - 1), accent, -1)
        cy = y + pad
        for (s, f, col), (tw, th, top) in zip(lines, sizes):
            txt.add((x + pad + int(3 * self.s), cy - top), s, f, col)
            cy += th + gap
        return x, y, w, h

    def _zones(self, img, txt: _Text, alarm: bool):
        poly = self.z.mantrap.astype(np.int32)
        if not alarm:  # the full-frame red tint replaces the zone fill during a warning
            layer = img.copy()
            cv2.fillPoly(layer, [poly], C_ZONE)
            cv2.addWeighted(layer, 0.32, img, 0.68, 0, img)
        edge = C_RED if alarm else C_ZONE_EDGE
        cv2.polylines(img, [poly], True, edge, max(1, int(1.5 * self.s)), cv2.LINE_AA)
        for p in poly:  # small diamonds on the vertices
            r = int(4 * self.s)
            d = np.array([[p[0], p[1] - r], [p[0] + r, p[1]], [p[0], p[1] + r], [p[0] - r, p[1]]])
            cv2.fillConvexPoly(img, d, edge, cv2.LINE_AA)
        for line, col, name in ((self.z.entry_line, C_ENTRY, "ENTRY"),
                                (self.z.exit_line, C_EXIT, "EXIT")):
            a, b = line.astype(int)
            t = max(2, int(2.5 * self.s))
            cv2.line(img, tuple(a), tuple(b), col, t, cv2.LINE_AA)
            n = np.array([-(b - a)[1], (b - a)[0]], np.float32)
            n = (n / (np.linalg.norm(n) + 1e-6) * 7 * self.s).astype(int)
            for e in (a, b):  # end caps
                cv2.line(img, tuple(e - n), tuple(e + n), col, t, cv2.LINE_AA)
            self._tag(txt, img, a[0], a[1] - 5 * self.s, [(name, self.f_tag_small, B(T.TEXT))], col)

    def _person(self, img, txt: _Text, p: PersonState):
        if p.status == "outside" and not self.cfg["show_outside"]:
            return
        col = STATUS[p.status]
        t = max(1, int(1.6 * self.s))
        x1, y1, x2, y2 = p.track.box.astype(int)
        k = p.track.kpts
        if self.cfg["skeleton"] and k is not None:
            for a, b in SKELETON:
                if k[a, 2] > 0.35 and k[b, 2] > 0.35:
                    leg = bool({a, b} & LEG_KPTS)
                    cv2.line(img, tuple(k[a, :2].astype(int)), tuple(k[b, :2].astype(int)),
                             col if leg else C_TEXT2, t, cv2.LINE_AA)
            for i in LEG_KPTS:
                if k[i, 2] > 0.35:
                    c = tuple(k[i, :2].astype(int))
                    cv2.circle(img, c, int(3.5 * self.s), (10, 20, 15), -1, cv2.LINE_AA)
                    cv2.circle(img, c, int(3.5 * self.s), col, max(1, t - 1), cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), tuple(int(c * 0.45) for c in col), 1, cv2.LINE_AA)
        _brackets(img, x1, y1, x2, y2, col, t + 1)

        # tag: id and time in mantrap; second line says what the rules see
        head = f"ID {p.track.id:02d}"
        if p.in_zone:
            head += f"   {fmt_duration(p.dwell_time)}"
        lines = [(head, self.f_tag, B(T.TEXT))]
        if p.status == "alert":
            lines.append((" + ".join(MESSAGES[r].upper() for r in p.rules), self.f_tag_small, C_RED))
        elif p.in_zone and p.static_time > 0.5:
            lines.append((f"STATIC {p.static_time:.1f}s", self.f_tag_small, B(T.AMBER)))
        elif p.in_zone:
            lines.append(("IN MANTRAP", self.f_tag_small, B(T.ACCENT)))
        self._tag(txt, img, x1, y1 - 3 * self.s, lines, col)

        if p.in_zone:
            fx, fy = p.foot.astype(int)
            # static progress: thin bar under the box, fills up to the static-rule threshold
            if p.static_time > 0.3:
                frac = min(1.0, p.static_time / self.static_seconds)
                yb, hb = y2 + int(4 * self.s), max(2, int(3 * self.s))
                _blend_rect(img, x1, yb, x2, yb + hb, C_PLATE, 0.8)
                cv2.rectangle(img, (x1, yb), (int(x1 + frac * (x2 - x1)), yb + hb - 1), col, -1)
            # optical-flow direction arrow (0.5 s of travel at the current velocity)
            if self.cfg["flow_arrows"] and p.speed > 0.1:
                tip = (p.foot + 0.5 * p.velocity).astype(int)
                cv2.arrowedLine(img, (fx, fy), tuple(tip), (10, 20, 15), t + 3, cv2.LINE_AA,
                                tipLength=0.3)
                cv2.arrowedLine(img, (fx, fy), tuple(tip), col, t + 1, cv2.LINE_AA, tipLength=0.3)
            r = int(5 * self.s)
            cv2.circle(img, (fx, fy), r + 3, col, 1, cv2.LINE_AA)
            cv2.circle(img, (fx, fy), r - 1, col, -1, cv2.LINE_AA)

    def _tint(self, img):
        """Red tint over the whole frame, a thin red border and heavy corner brackets."""
        h, w = img.shape[:2]
        a = self.cfg["overlay_alpha"]
        cv2.addWeighted(img, 1 - a, np.full_like(img, C_RED), a, 0, dst=img)
        m = max(3, int(4 * self.s))
        cv2.rectangle(img, (m, m), (w - 1 - m, h - 1 - m), C_RED, max(1, int(1.5 * self.s)))
        _brackets(img, m, m, w - 1 - m, h - 1 - m, C_RED, max(3, int(5 * self.s)), frac=0.08)

    def _banner(self, img, txt: _Text, state: FrameState, right_limit: int):
        m = int(12 * self.s)
        rules = " + ".join(MESSAGES[r].upper() for r in state.alarm_rules)
        lines = [("WARNING", self.f_banner, B(T.TEXT)), (rules, self.f_tag, B(T.TEXT))]
        f = self.f_tag
        while txt.size(rules, f)[0] > right_limit - 3 * m and f.size > 9:
            f = T.font("display", f.size - 1)
            lines[1] = (rules, f, B(T.TEXT))
        pad = int(7 * self.s)
        sizes = [txt.size(s, ff) for s, ff, _ in lines]
        w = max(s[0] for s in sizes) + 2 * pad + int(4 * self.s)
        h = sum(s[1] for s in sizes) + 2 * pad + int(3 * self.s)
        _blend_rect(img, m, m, m + w, m + h, (40, 30, 150), 0.85)
        cv2.rectangle(img, (m, m), (m + int(4 * self.s), m + h - 1), C_RED, -1)
        cy = m + pad
        for (s, ff, col), (tw, th, top) in zip(lines, sizes):
            txt.add((m + pad + int(4 * self.s), cy - top), s, ff, col)
            cy += th + int(3 * self.s)

    def _hud(self, img, txt: _Text, hud: dict | None, state: FrameState) -> int:
        """Stats plate, top right. Returns its left edge."""
        if not hud:
            return img.shape[1]
        in_zone = sum(p.in_zone for p in state.persons)
        lines = [(f"{hud.get('fps', 0):.1f} FPS", self.f_hud_big, B(T.ACCENT)),
                 (f"{hud.get('latency_ms', 0):.0f} ms   {len(state.persons)} "
                  f"track{'' if len(state.persons) == 1 else 's'}   "
                  f"{in_zone} in mantrap", self.f_hud, B(T.TEXT)),
                 (hud.get("backend", ""), self.f_hud, B(T.TEXT_2))]
        m = int(12 * self.s)
        pad = int(6 * self.s)
        sizes = [txt.size(s, f) for s, f, _ in lines]
        w = max(s[0] for s in sizes) + 2 * pad
        h = sum(s[1] for s in sizes) + int(4 * self.s) * 2 + 2 * pad
        x, y = img.shape[1] - m - w, m
        _blend_rect(img, x, y, x + w, y + h, C_PLATE, 0.72)
        _brackets(img, x, y, x + w, y + h, B(T.ACCENT_DIM), 1, frac=0.15)
        cy = y + pad
        for (s, f, col), (tw, th, top) in zip(lines, sizes):
            txt.add((x + pad, cy - top), s, f, col)
            cy += th + int(4 * self.s)
        return x

    # ------------------------------------------------------------------ frame
    def draw(self, frame: np.ndarray, state: FrameState, hud: dict | None = None) -> np.ndarray:
        img = frame.copy()
        txt = _Text()
        self._zones(img, txt, state.alarm)
        for o in state.objects:
            col = C_RED if o.alert else B(T.AMBER)
            x1, y1, x2, y2 = o.track.box.astype(int)
            _brackets(img, x1, y1, x2, y2, col, max(1, int(2 * self.s)))
            lab = f"OBJ {o.track.id:02d}" + (f"   {o.static_time:.0f}s" if o.in_zone else "")
            self._tag(txt, img, x1, y1 - 3 * self.s, [(lab, self.f_tag_small, B(T.TEXT))], col)
        for p in sorted(state.persons, key=lambda q: q.status == "alert"):
            self._person(img, txt, p)
        if state.alarm:  # tint first so the HUD and banner stay crisp on top
            self._tint(img)
        hud_left = self._hud(img, txt, hud, state)
        if state.alarm:
            self._banner(img, txt, state, hud_left)
        return txt.render(img)
