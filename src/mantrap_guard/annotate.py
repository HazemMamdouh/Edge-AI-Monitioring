"""Interactive (OpenCV) and headless annotation of the mantrap, entry line and exit line."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .zones import Zones

MODES = ["mantrap", "entry_line", "exit_line"]
COLORS = {"mantrap": (255, 200, 0), "entry_line": (80, 220, 80), "exit_line": (0, 150, 255)}
HELP = [
    "1 mantrap polygon   2 entry line   3 exit line",
    "left click: add point   right click / u: undo   c: clear shape",
    "n / p: next / previous frame   s: save   q: quit",
]


def read_frame(video: str, index: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index} of {video}")
    return frame


def draw_zones(img: np.ndarray, pts: dict[str, list], active: str | None = None) -> np.ndarray:
    out = img.copy()
    poly = np.asarray(pts.get("mantrap", []), np.int32)
    if len(poly) >= 3:
        layer = out.copy()
        cv2.fillPoly(layer, [poly], COLORS["mantrap"])
        out = cv2.addWeighted(layer, 0.22, out, 0.78, 0)
    if len(poly) >= 2:
        cv2.polylines(out, [poly], len(poly) >= 3, COLORS["mantrap"], 2, cv2.LINE_AA)
    for name in ("entry_line", "exit_line"):
        ln = np.asarray(pts.get(name, []), np.int32)
        if len(ln) == 2:
            cv2.line(out, tuple(ln[0]), tuple(ln[1]), COLORS[name], 3, cv2.LINE_AA)
            cv2.putText(out, name.split("_")[0].upper(), tuple(ln[0] + (6, -8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[name], 2, cv2.LINE_AA)
    for name, p in pts.items():
        for q in p:
            r = 6 if name == active else 4
            cv2.circle(out, (int(q[0]), int(q[1])), r, COLORS[name], -1, cv2.LINE_AA)
    return out


def _to_zones(pts: dict[str, list], size: tuple[int, int]) -> Zones:
    if len(pts["mantrap"]) < 3 or len(pts["entry_line"]) != 2 or len(pts["exit_line"]) != 2:
        raise ValueError("need a mantrap polygon (>=3 points) and two points for each line")
    return Zones(
        np.asarray(pts["mantrap"], np.float32),
        np.asarray(pts["entry_line"], np.float32),
        np.asarray(pts["exit_line"], np.float32),
        size,
    )


def parse_points(text: str) -> list[list[float]]:
    """'x,y x,y x,y' -> [[x, y], ...]"""
    return [[float(v) for v in p.split(",")] for p in text.replace(";", " ").split()]


def annotate_headless(video, out, mantrap, entry_line, exit_line, frame=0, preview=None) -> Zones:
    """Write a zone file from coordinates given on the command line (no display needed)."""
    img = read_frame(video, frame)
    pts = {"mantrap": parse_points(mantrap), "entry_line": parse_points(entry_line),
           "exit_line": parse_points(exit_line)}
    zones = _to_zones(pts, (img.shape[1], img.shape[0]))
    zones.save(out)
    if preview:
        Path(preview).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview), draw_zones(img, pts))
    return zones


def annotate_interactive(video, out, frame=0, preview=None) -> Zones | None:
    """Click the three shapes on a video frame. Needs a display."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    idx = frame
    img = read_frame(video, idx)
    pts: dict[str, list] = {m: [] for m in MODES}
    if Path(out).exists():  # resume editing an existing annotation
        z = Zones.load(out, (img.shape[1], img.shape[0]))
        pts = {"mantrap": z.mantrap.tolist(), "entry_line": z.entry_line.tolist(),
               "exit_line": z.exit_line.tolist()}
    state = {"mode": "mantrap"}

    def on_mouse(event, x, y, flags, param):
        m = state["mode"]
        if event == cv2.EVENT_LBUTTONDOWN:
            if m != "mantrap" and len(pts[m]) >= 2:
                pts[m] = []
            pts[m].append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN and pts[m]:
            pts[m].pop()

    win = "mantrap-guard annotate"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    saved = None
    while True:
        view = draw_zones(img, pts, state["mode"])
        cv2.rectangle(view, (0, 0), (view.shape[1], 76), (20, 20, 20), -1)
        cv2.putText(view, f"editing: {state['mode']}   frame {idx}", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[state["mode"]], 1, cv2.LINE_AA)
        for i, line in enumerate(HELP):
            cv2.putText(view, line, (10, 36 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imshow(win, view)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("1"), ord("2"), ord("3")):
            state["mode"] = MODES[k - ord("1")]
        elif k == ord("u") and pts[state["mode"]]:
            pts[state["mode"]].pop()
        elif k == ord("c"):
            pts[state["mode"]] = []
        elif k in (ord("n"), ord("p")):
            idx = int(np.clip(idx + (15 if k == ord("n") else -15), 0, max(total - 1, 0)))
            img = read_frame(video, idx)
        elif k == ord("s"):
            try:
                saved = _to_zones(pts, (img.shape[1], img.shape[0]))
                saved.save(out)
                if preview:
                    cv2.imwrite(str(preview), draw_zones(img, pts))
                print(f"saved {out}")
            except ValueError as e:
                print(f"not saved: {e}")
        elif k in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    return saved
