"""Re-create the visualisation of a run from its files, without running any model.

    mantrap render --run outputs/demo                      # redraw output.mp4 as recorded
    mantrap render --run outputs/demo --config my.yaml \\
                   --reapply-rules --out outputs/demo_tuned  # new thresholds, no inference

Inputs: detections.jsonl (+ metrics.csv for the FPS / latency shown in the frame HUD) and
the source video. With --reapply-rules the stored tracks and optical-flow vectors are fed
through the rule engine again, so rule thresholds can be tuned in seconds instead of
re-running detection; events.json and dwell.json are rewritten for the new rules.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import cv2
import yaml

from .config import deep_merge, load_config
from .detections import read_detections, state_from_record, tracks_from_record, zones_from_header
from .pipeline import to_h264
from .rules import RuleEngine
from .visualizer import Visualizer
from .zones import Zones


def _metrics(run: Path) -> dict[int, dict]:
    p = run / "metrics.csv"
    if not p.exists():
        return {}
    with open(p) as f:
        return {int(r["frame"]): r for r in csv.DictReader(f)}


def render(run_dir: str, out_dir: str | None = None, video: str | None = None,
           config: str | None = None, zones: str | None = None, reapply_rules: bool = False,
           hud: bool = True) -> Path:
    run = Path(run_dir)
    header, frames = read_detections(run / "detections.jsonl")
    out = Path(out_dir) if out_dir else run
    out.mkdir(parents=True, exist_ok=True)

    # same settings as the original run unless overridden
    cfg = load_config(None, header.get("config"))
    if config:  # a YAML file overrides only the keys it sets
        cfg = deep_merge(cfg, yaml.safe_load(Path(config).read_text()) or {})
    video = video or header["video"]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    z = Zones.load(zones, (w, h)) if zones else zones_from_header(header).scaled_to((w, h))
    fps = header["fps"]
    viz = Visualizer(z, cfg["viz"], cfg["rules"]["static"]["seconds"])
    engine = RuleEngine(z, cfg, fps) if reapply_rules else None
    metrics = _metrics(run)
    label = f"{header['model']} | {header['backend']}"

    name = "output.mp4" if out != run or reapply_rules else "output_rendered.mp4"
    path = out / name
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    events, visits = [], []
    next_src = None
    for rec in frames:
        src = rec["source_frame"]
        if next_src != src:  # seek only when needed (first frame, or a gap)
            cap.set(cv2.CAP_PROP_POS_FRAMES, src)
        ok, frame = cap.read()
        if not ok:
            break
        next_src = src + 1
        if engine:
            persons, flow, objects = tracks_from_record(rec)
            state = engine.update(persons, flow, rec["t"], objects)
            events += [e.as_dict() for e in state.events]
            visits += [v.as_dict() for v in state.visits]
        else:
            state = state_from_record(rec)
        m = metrics.get(rec["frame"])
        info = None
        if hud and m:
            info = {"backend": label, "fps": float(m["fps"]), "latency_ms": float(m["latency_ms"])}
        writer.write(viz.draw(frame, state, info))
    writer.release()
    cap.release()
    to_h264(path)
    if out != run:  # make the new folder a complete run: showcase can be built from it
        for name in ("summary.json", "metrics.csv", "log.jsonl"):
            if (run / name).exists():
                shutil.copy(run / name, out / name)
    if engine:
        visits += [v.as_dict() for v in engine.finish()]
        (out / "events.json").write_text(json.dumps(events, indent=2))
        (out / "dwell.json").write_text(json.dumps(visits, indent=2))
        _rewrite_log(run, out, events, visits)
    return path


def _rewrite_log(run: Path, out: Path, events: list[dict], visits: list[dict]) -> None:
    """The terminal log replayed by the showcase must match the re-applied rules: keep the
    run's own lines (startup, stats), replace its alert and visit lines with the new ones."""
    src = run / "log.jsonl"
    if not src.exists():
        return
    keep = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    keep = [e for e in keep if e["level"] != "alert" and "was in the mantrap" not in e["text"]
            and e["text"] != "mantrap clear"]
    new = [{"frame": e["frame"] - 1, "t": e["t"], "level": "alert",
            "text": f"ALERT {e['rule']:<16} {e['message']}"} for e in events]
    new += [{"frame": v["exit_frame"], "t": v["exit_t"], "level": "info",
             "text": f"person #{v['track_id']} was in the mantrap for {v['duration']:.1f}s "
                     f"({v['reason'].replace('_', ' ')})"} for v in visits]
    lines = sorted(keep + new, key=lambda e: (e["frame"], e["level"] != "info"))
    (out / "log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
