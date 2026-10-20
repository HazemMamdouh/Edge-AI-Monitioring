"""Compose the portfolio showcase video from a finished run:

    +-----------------------------+-----------------------------+
    |  01 INPUT                   |  02 OUTPUT (rules + overlay)|
    +-----------------------------+-----------------------------+
    |  03 SYSTEM PERFORMANCE      |  04 TERMINAL                |
    |  fps, latency, cpu, gpu ... |  replay of the run log      |
    +-----------------------------+-----------------------------+

Every number shown comes from metrics.csv / log.jsonl written by `mantrap run` on the
machine that ran it. Nothing is simulated; a missing sensor is shown as "n/a".
Style: dark-green HUD, Rajdhani / Space Grotesk / JetBrains Mono (see theme.py).
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from . import theme as T
from .pipeline import STAGES, to_h264

W, H = 1920, 1080
PAD, HEADER = 28, 96
PANEL_W = (W - 3 * PAD) // 2
TAB = 40
VID_H = int((PANEL_W - 2) * 9 / 16)
ROW1_Y = HEADER
ROW2_Y = ROW1_Y + TAB + VID_H + 2 + 18
ROW2_H = H - ROW2_Y - PAD

TERM_BG = "#020604"
SPARK_FILL = "#0b2d21"  # accent blended into the surface
TERM_COLORS = {"info": T.TEXT_2, "stat": "#7f9c8f", "alert": T.RED, "ok": T.ACCENT}


class Fonts:
    def __init__(self):
        self.title = T.font("display", 40)
        self.tab = T.font("display", 19)
        self.tab_idx = T.font("display", 19)
        self.label = T.font("display_medium", 16)
        self.big = T.font("display", 60)
        self.big_unit = T.font("display_medium", 26)
        self.state = T.font("display", 34)
        self.body = T.font("body", 17)
        self.body_m = T.font("body_medium", 17)
        self.small = T.font("body", 14)
        self.meta = T.font("body", 15)
        self.mono = T.font("mono", 15)
        self.mono_b = T.font("mono_bold", 15)


# ------------------------------------------------------------------- helpers
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load(run_dir: Path):
    summary = json.loads((run_dir / "summary.json").read_text())
    with open(run_dir / "metrics.csv") as f:
        metrics = [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]
    log = [json.loads(l) for l in (run_dir / "log.jsonl").read_text().splitlines() if l.strip()]
    return summary, metrics, log


def _fit(frame: np.ndarray, w: int, h: int) -> Image.Image:
    fh, fw = frame.shape[:2]
    r = min(w / fw, h / fh)
    nw, nh = int(fw * r), int(fh * r)
    img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((h, w, 3), np.uint8)
    y, x = (h - nh) // 2, (w - nw) // 2
    canvas[y:y + nh, x:x + nw] = img[:, :, ::-1]
    return Image.fromarray(canvas)


def _tracked(d: ImageDraw.ImageDraw, xy, text, font, fill, tracking=1.5):
    """Text with letter-spacing (wide, uppercase labels read as 'HUD')."""
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + tracking
    return x


def _tracked_len(d, text, font, tracking=1.5):
    return sum(d.textlength(c, font=font) + tracking for c in text)


def _brackets(d, x1, y1, x2, y2, color, length=18, width=2):
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        d.line((cx, cy, cx + dx * length, cy), fill=color, width=width)
        d.line((cx, cy, cx, cy + dy * length), fill=color, width=width)


def _steps(d: ImageDraw.ImageDraw, x, y, steps, font):
    """Pipeline steps joined by arrows drawn as lines (no font glyph needed)."""
    asc, desc = font.getmetrics()
    cy = y + (asc + desc) / 2 + 1
    for i, step in enumerate(steps):
        if i:
            x0, x1 = x + 12, x + 34
            d.line((x0, cy, x1, cy), fill=T.ACCENT_DIM, width=1)
            d.polygon([(x1, cy), (x1 - 6, cy - 4), (x1 - 6, cy + 4)], fill=T.ACCENT_DIM)
            x += 46
        d.text((x, y), step, font=font, fill=T.TEXT_2)
        x += d.textlength(step, font=font)


def _panel(d, x, y, w, h, idx, title, f: Fonts, meta="", alert=False):
    edge = T.RED if alert else T.EDGE
    d.rectangle((x, y, x + w, y + h), fill=T.SURFACE, outline=edge, width=1)
    d.rectangle((x + 1, y + 1, x + w - 1, y + TAB - 1), fill=T.SURFACE_2)
    d.line((x, y + TAB, x + w, y + TAB), fill=edge, width=1)
    _brackets(d, x, y, x + w, y + h, T.RED if alert else T.ACCENT_DIM, 22, 3 if alert else 2)
    tx = _tracked(d, (x + 18, y + 9), idx, f.tab_idx, T.RED if alert else T.ACCENT, 1)
    _tracked(d, (tx + 12, y + 9), title, f.tab, T.TEXT, 2.2)
    if meta:
        d.text((x + w - 18 - d.textlength(meta, font=f.meta), y + 12), meta, font=f.meta,
               fill=T.MUTED)


def _label(d, xy, text, f: Fonts):
    _tracked(d, xy, text, f.label, T.MUTED, 1.8)


def _segment_meter(d, x, y, w, value, segments=34, h=10):
    """A segmented bar: futuristic, and the segment count makes 3% steps readable."""
    gap = 3
    sw = (w - gap * (segments - 1)) / segments
    lit = 0 if value is None else round(segments * min(max(value, 0), 100) / 100)
    for i in range(segments):
        sx = x + i * (sw + gap)
        d.rectangle((sx, y, sx + sw, y + h), fill=T.ACCENT if i < lit else T.SURFACE_2)


def _meter(d, x, y, w, label, value, text, f: Fonts):
    _label(d, (x, y), label, f)
    tw = d.textlength(text, font=f.body_m)
    d.text((x + w - tw, y - 2), text, font=f.body_m, fill=T.TEXT)
    _segment_meter(d, x, y + 28, w, value)


def _spark(d, x, y, w, h, values, label, unit, f: Fonts, vmax=None):
    _label(d, (x, y), label, f)
    vals = [v for v in values if v is not None]
    y0, y1 = y + 30, y + h
    pw = w - 64
    for gy in (y0, (y0 + y1) / 2):  # recessive grid
        d.line((x, gy, x + pw, gy), fill=T.SURFACE_2, width=1)
    d.line((x, y1, x + pw, y1), fill=T.EDGE, width=1)
    if len(vals) < 2:
        d.text((x, y0 + 8), "n/a", font=f.body, fill=T.MUTED)
        return
    top = vmax or max(max(vals) * 1.2, 1e-6)
    pts = [(x + pw * i / (len(vals) - 1), y1 - (y1 - y0) * min(v / top, 1.0))
           for i, v in enumerate(vals)]
    d.polygon(pts + [(pts[-1][0], y1), (pts[0][0], y1)], fill=SPARK_FILL)
    d.line(pts, fill=T.ACCENT, width=2, joint="curve")
    ex, ey = pts[-1]
    d.ellipse((ex - 5, ey - 5, ex + 5, ey + 5), fill=T.SURFACE, outline=T.ACCENT, width=2)
    d.text((x + pw + 12, ey - 11), f"{vals[-1]:.0f}{unit}", font=f.body_m, fill=T.TEXT)


# --------------------------------------------------------------- the panels
def _system_panel(d, x, y, w, h, m: dict, hist: list[dict], summary: dict, f: Fonts):
    alarm = bool(m["alarm"])
    _panel(d, x, y, w, h, "03", "SYSTEM PERFORMANCE", f, meta="live metrics of this run")

    ix, iy, iw = x + 26, y + TAB + 16, w - 52
    tile_w = iw / 4
    # --- stat tiles (hero numbers)
    tiles = [("PIPELINE", f"{m['fps']:.1f}", "FPS"), ("LATENCY", f"{m['latency_ms']:.0f}", "ms"),
             ("IN MANTRAP", f"{int(m['in_zone'])}", "")]
    for i, (lab, val, unit) in enumerate(tiles):
        tx = ix + i * tile_w
        if i:
            d.line((tx - 14, iy + 4, tx - 14, iy + 78), fill=T.EDGE, width=1)
        _label(d, (tx, iy), lab, f)
        d.text((tx, iy + 16), val, font=f.big, fill=T.TEXT)
        if unit:
            d.text((tx + d.textlength(val, font=f.big) + 8, iy + 44), unit, font=f.big_unit,
                   fill=T.TEXT_2)
    tx = ix + 3 * tile_w
    d.line((tx - 14, iy + 4, tx - 14, iy + 78), fill=T.EDGE, width=1)
    _label(d, (tx, iy), "STATE", f)
    col = T.RED if alarm else T.ACCENT
    cy = iy + 50
    d.ellipse((tx, cy - 9, tx + 18, cy + 9), outline=col, width=2)
    d.ellipse((tx + 5, cy - 4, tx + 13, cy + 4), fill=col)
    d.text((tx + 28, cy - 22), "WARNING" if alarm else "CLEAR", font=f.state, fill=col)

    # --- latency by stage: parts of one frame's processing time
    sy = iy + 96
    _label(d, (ix, sy), "LATENCY BY STAGE", f)
    vals = [m.get(s + "_ms") or 0.0 for s in STAGES]
    total = sum(vals) or 1.0
    bx = ix
    for v, c in zip(vals, T.STAGE_COLORS):
        seg = iw * v / total
        if seg >= 3:
            d.rectangle((bx, sy + 28, bx + seg - 3, sy + 38), fill=c)
        bx += seg
    lx = ix
    for s, v, c in zip(STAGES, vals, T.STAGE_COLORS):
        d.rectangle((lx, sy + 54, lx + 10, sy + 64), fill=c)
        txt = f"{s}  {v:.1f} ms"
        d.text((lx + 17, sy + 49), txt, font=f.small, fill=T.TEXT_2)
        lx += 17 + d.textlength(txt, font=f.small) + 26

    # --- meters
    my = sy + 84
    gap = 34
    col_w = (iw - 2 * gap) / 3
    cpu, gpu = m.get("cpu"), m.get("gpu")
    _meter(d, ix, my, col_w, "CPU", cpu, "n/a" if cpu is None else f"{cpu:.0f}%", f)
    _meter(d, ix + col_w + gap, my, col_w, "GPU", gpu, "n/a" if gpu is None else f"{gpu:.0f}%", f)
    ru, rt = m.get("ram_used_gb"), m.get("ram_total_gb")
    _meter(d, ix + 2 * (col_w + gap), my, col_w, "RAM",
           None if not (ru and rt) else 100 * ru / rt,
           "n/a" if not (ru and rt) else f"{ru:.1f} / {rt:.1f} GB", f)

    # --- sparklines (one measure each) + device facts
    py = my + 56
    ph = y + h - py - 18
    _spark(d, ix, py, col_w, ph, [r["fps"] for r in hist], "FPS  |  15 S", "", f)
    _spark(d, ix + col_w + gap, py, col_w, ph, [r["cpu"] for r in hist], "CPU  |  15 S", "%", f,
           vmax=100)
    fx = ix + 2 * (col_w + gap)
    temp, power = m.get("temp_c"), m.get("power_w")
    _label(d, (fx, py), "TEMP", f)
    d.text((fx, py + 22), "n/a" if temp is None else f"{temp:.0f} C", font=f.body_m, fill=T.TEXT)
    _label(d, (fx + col_w / 2, py), "POWER", f)
    d.text((fx + col_w / 2, py + 22), "n/a" if power is None else f"{power:.1f} W", font=f.body_m,
           fill=T.TEXT)
    _label(d, (fx, py + 48), "INPUT", f)
    w_, h_ = summary["size"]
    d.text((fx, py + 68), f"{w_} x {h_}  |  {summary['imgsz']} px model", font=f.body_m,
           fill=T.TEXT)


def _terminal_panel(d, x, y, w, h, lines: list[dict], f: Fonts, blink: bool):
    _panel(d, x, y, w, h, "04", "TERMINAL", f, meta="$ mantrap run")
    d.rectangle((x + 1, y + TAB + 1, x + w - 1, y + h - 1), fill=TERM_BG)
    _brackets(d, x, y, x + w, y + h, T.ACCENT_DIM, 22, 2)
    line_h = 22
    top = y + TAB + 14
    rows = (y + h - 16 - top) // line_h
    cw = d.textlength("M", font=f.mono)
    max_chars = int((w - 40) / cw)
    ty = top
    shown = lines[-(rows - 1):] if rows > 1 else []
    for e in shown:
        t = e["t"]
        stamp = f"{int(t // 60):02d}:{t % 60:04.1f}"
        body = e["text"]
        if len(stamp) + 3 + len(body) > max_chars:
            body = body[: max_chars - len(stamp) - 6] + "..."
        d.text((x + 20, ty), stamp, font=f.mono, fill=T.MUTED)
        d.text((x + 20 + cw * (len(stamp) + 2), ty), body,
               font=f.mono_b if e["level"] == "alert" else f.mono,
               fill=TERM_COLORS.get(e["level"], T.TEXT_2))
        ty += line_h
    # prompt with a blinking block cursor
    d.text((x + 20, ty), ">", font=f.mono_b, fill=T.ACCENT)
    if blink:
        d.rectangle((x + 20 + cw * 2, ty + 3, x + 20 + cw * 3 - 1, ty + 19), fill=T.ACCENT)


def _background(summary: dict, f: Fonts, title: str) -> Image.Image:
    """Static chrome drawn once: grid, glow, header."""
    base = Image.new("RGB", (W, H), T.BG)
    d = ImageDraw.Draw(base)
    for gx in range(0, W, 48):
        d.line((gx, 0, gx, H), fill=T.GRID, width=1)
    for gy in range(0, H, 48):
        d.line((0, gy, W, gy), fill=T.GRID, width=1)
    # soft dark-green glow behind the title
    glow = Image.new("RGB", (W, H), T.BG)
    ImageDraw.Draw(glow).ellipse((-300, -260, 820, 300), fill=T.DEEP_GREEN)
    base = ImageChops.lighter(base, glow.filter(ImageFilter.GaussianBlur(140)))
    d = ImageDraw.Draw(base)

    # header
    d.polygon([(PAD, 34), (PAD + 9, 25), (PAD + 18, 34), (PAD + 9, 43)], fill=T.ACCENT)
    tx = _tracked(d, (PAD + 32, 12), title, f.title, T.TEXT, 4)
    _steps(d, tx + 26, 23, ["pose", "legs in mantrap", "tracking", "optical flow", "rules"],
           f.body)
    chips = [("MODEL", summary["model"]), ("BACKEND", summary["backend"]),
             ("DEVICE", summary["device"])]
    x = W - PAD
    for lab, val in reversed(chips):
        vw = d.textlength(val, font=f.meta)
        lw = _tracked_len(d, lab, f.label, 1.6)
        cw = vw + lw + 34
        x -= cw
        d.rectangle((x, 18, x + cw, 50), fill=T.SURFACE, outline=T.EDGE)
        _tracked(d, (x + 12, 25), lab, f.label, T.ACCENT_DIM, 1.6)
        d.text((x + 22 + lw, 24), val, font=f.meta, fill=T.TEXT)
        x -= 10
    # header rule: a bright segment fading into the dark line
    y = 66
    d.line((PAD, y, W - PAD, y), fill=T.EDGE, width=1)
    for i in range(320):
        a = 1 - i / 320
        c = tuple(int(int(T.ACCENT[k:k + 2], 16) * a + int(T.EDGE[k:k + 2], 16) * (1 - a))
                  for k in (1, 3, 5))
        d.point((PAD + i, y), fill=c)
        d.point((PAD + i, y + 1), fill=c)
    return base


# --------------------------------------------------------------------- main
def build_showcase(run_dir: str, out: str | None = None, video: str | None = None,
                   title: str = "MANTRAP GUARD", gif: bool = True) -> Path:
    run = Path(run_dir)
    summary, metrics, log = _load(run)
    video = video or summary["video"]
    out_path = Path(out) if out else run / "showcase.mp4"
    fps = summary["fps"]
    f = Fonts()

    cap_in = cv2.VideoCapture(str(video))
    cap_out = cv2.VideoCapture(str(run / "output.mp4"))
    if not cap_in.isOpened() or not cap_out.isOpened():
        raise RuntimeError("cannot open the input video or output.mp4 of the run")
    if summary.get("start"):
        cap_in.set(cv2.CAP_PROP_POS_FRAMES, summary["start"])

    w_, h_ = summary["size"]
    meta = f"{w_} x {h_}  |  {fps:.0f} FPS"
    base = _background(summary, f, title)
    x2 = PAD * 2 + PANEL_W
    d = ImageDraw.Draw(base)
    _panel(d, PAD, ROW1_Y, PANEL_W, TAB + VID_H + 2, "01", "INPUT", f, meta=meta)
    base_alarm = base.copy()
    for img_, alert in ((base, False), (base_alarm, True)):
        _panel(ImageDraw.Draw(img_), x2, ROW1_Y, PANEL_W, TAB + VID_H + 2, "02",
               "OUTPUT", f, meta="zones, tracks, dwell time, warnings", alert=alert)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    window = int(15 * fps)
    li, shown = 0, []
    still, still_score = None, -1
    blink_period = max(1, int(fps / 2))
    for i, m in enumerate(metrics):
        ok1, fin = cap_in.read()
        ok2, fout = cap_out.read()
        if not (ok1 and ok2):
            break
        while li < len(log) and log[li]["frame"] <= m["frame"]:
            shown.append(log[li])
            li += 1
        img = (base_alarm if m["alarm"] else base).copy()
        d = ImageDraw.Draw(img)
        img.paste(_fit(fin, PANEL_W - 2, VID_H), (PAD + 1, ROW1_Y + TAB + 1))
        img.paste(_fit(fout, PANEL_W - 2, VID_H), (x2 + 1, ROW1_Y + TAB + 1))
        _system_panel(d, PAD, ROW2_Y, PANEL_W, ROW2_H, m, metrics[max(0, i - window):i + 1],
                      summary, f)
        _terminal_panel(d, x2, ROW2_Y, PANEL_W, ROW2_H, shown, f, (i // blink_period) % 2 == 0)
        frame = np.asarray(img)[:, :, ::-1]
        writer.write(np.ascontiguousarray(frame))
        score = m["alarm"] * 10 + m["in_zone"] + min(len(shown), 12) / 12
        if score > still_score:
            still, still_score = frame.copy(), score
    writer.release()
    cap_in.release()
    cap_out.release()
    to_h264(out_path)
    if still is not None:
        cv2.imwrite(str(out_path.with_suffix(".png")), still)
    if gif and shutil.which("ffmpeg"):
        g = out_path.with_suffix(".gif")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out_path), "-vf",
                        "fps=5,scale=854:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];"
                        "[b][p]paletteuse=dither=bayer:bayer_scale=4", str(g)])
    return out_path
