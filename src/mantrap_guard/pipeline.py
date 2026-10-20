"""The end-to-end pipeline: pose -> leg-in-mantrap filter -> tracking -> optical flow -> rules."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .detections import DetectionWriter, zones_dict
from .detector import ObjectDetector, PoseDetector
from .flow import FlowEstimator
from .rules import RuleEngine
from .sysmon import SystemMonitor, device_name
from .tracker import ByteTracker
from .visualizer import Visualizer
from .zones import Zones

STAGES = ["detect", "track", "flow", "rules", "draw"]
LEVEL_STYLE = {"info": "cyan", "stat": "white", "alert": "bold red", "ok": "green"}


def to_h264(path: Path) -> None:
    """Re-encode to H.264 so the file plays in browsers / on GitHub. No-op without ffmpeg."""
    if not shutil.which("ffmpeg"):
        return
    tmp = path.with_suffix(".h264.mp4")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(tmp)])
    if r.returncode == 0:
        tmp.replace(path)
    else:
        tmp.unlink(missing_ok=True)


class RunLog:
    """Terminal log that is also saved with frame numbers, so the showcase can replay it."""

    def __init__(self, path: Path, console: Console, echo: bool = True):
        self.f = open(path, "w", encoding="utf-8")
        self.console, self.echo = console, echo
        self.tail: deque[tuple[str, str]] = deque(maxlen=8)

    def __call__(self, frame: int, t: float, level: str, text: str):
        stamp = f"{int(t // 60):02d}:{t % 60:04.1f}"
        self.f.write(json.dumps({"frame": frame, "t": round(t, 2), "level": level,
                                 "text": text}) + "\n")
        line = f"[{stamp}] {text}"
        self.tail.append((level, line))
        if self.echo:
            self.console.print(line, style=LEVEL_STYLE.get(level, "white"), highlight=False,
                               soft_wrap=True)

    def close(self):
        self.f.close()


def _fmt(v, unit="", nd=0):
    return "n/a" if v is None else f"{v:.{nd}f}{unit}"


def _dashboard(meta: dict, m: dict, sysm: dict, log: RunLog) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_row("[b]model[/b]", f"{meta['model']}  [dim]{meta['backend']}[/dim]")
    t.add_row("[b]device[/b]", meta["device"])
    t.add_row("[b]fps[/b]", f"[bold green]{m['fps']:.1f}[/]   latency {m['latency_ms']:.1f} ms")
    t.add_row("[b]stages[/b]", "  ".join(f"{s} {m[s + '_ms']:.1f}" for s in STAGES) + " ms")
    t.add_row("[b]cpu[/b]", f"{sysm['cpu']:.0f}%   ram {sysm['ram_used_gb']:.1f}/"
                            f"{sysm['ram_total_gb']:.1f} GB")
    t.add_row("[b]gpu[/b]", f"{_fmt(sysm['gpu'], '%')}   temp {_fmt(sysm['temp_c'], ' C')}   "
                            f"power {_fmt(sysm['power_w'], ' W', 1)}")
    t.add_row("[b]scene[/b]", f"tracks {m['tracks']}  in mantrap {m['in_zone']}  "
                              + ("[bold red]ALARM[/]" if m["alarm"] else "[green]clear[/]"))
    t.add_row("", "")
    for level, line in log.tail:
        t.add_row("", f"[{LEVEL_STYLE.get(level, 'white')}]{line}[/]")
    return Panel(t, title=f"mantrap-guard {__version__}", border_style="red" if m["alarm"] else "cyan")


def run_pipeline(video: str, zones_path: str, out_dir: str, cfg: dict, max_frames: int | None = None,
                 start: int = 0, show: bool = False, dashboard: bool = False,
                 save_video: bool = True) -> dict:
    console = Console()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start
    if max_frames:
        total = min(total, max_frames)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    mc = cfg["model"]
    zones = Zones.load(zones_path, (w, h))
    pose = PoseDetector(mc["pose"], imgsz=mc["imgsz"], conf=mc["conf"], iou=mc["iou"],
                        device=mc["device"], half=mc["half"])
    objdet = objtracker = None
    if mc.get("objects"):
        objdet = ObjectDetector(mc["objects"], imgsz=mc["imgsz"], conf=mc["object_conf"],
                                iou=mc["iou"], device=mc["device"], half=mc["half"],
                                classes=mc["object_classes"])
        objtracker = ByteTracker(high_thresh=0.3, low_thresh=0.1, new_track_thresh=0.3,
                                 max_age=int(2 * fps), min_hits=3)
    tracker = ByteTracker(**cfg["tracker"])
    flow = FlowEstimator(**cfg["flow"])
    engine = RuleEngine(zones, cfg, fps)
    viz = Visualizer(zones, cfg["viz"], cfg["rules"]["static"]["seconds"])

    log = RunLog(out / "log.jsonl", console, echo=not dashboard)
    meta = {"model": Path(mc["pose"]).name, "backend": pose.backend, "device": device_name(),
            "imgsz": mc["imgsz"], "video": str(video), "zones": str(zones_path), "fps": fps,
            "size": [w, h], "start": start, "version": __version__}
    log(0, 0, "info", f"$ mantrap run --video {Path(video).name} --model {meta['model']}")
    log(0, 0, "info", f"device: {meta['device']}")
    log(0, 0, "info", f"pose model: {meta['model']}  backend: {pose.backend}  imgsz: {mc['imgsz']}")
    if objdet:
        log(0, 0, "info", f"object model: {Path(mc['objects']).name}  backend: {objdet.backend}")
    log(0, 0, "info", f"video: {w}x{h} @ {fps:.0f} fps, {total} frames   flow: {cfg['flow']['method']}")
    r = cfg["rules"]
    log(0, 0, "info", f"rules: static >= {r['static']['seconds']:.0f}s | moving towards entry line | "
                      f"exit-line intrusion" + (" | left object" if objdet else ""))

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("video has no frames")
    pose.warmup(first)
    if objdet:
        objdet.warmup(first)
    meta["infer_device"] = pose.device
    log(0, 0, "ok", f"warm-up done, inference on {pose.device}")

    writer = None
    if save_video:
        writer = cv2.VideoWriter(str(out / "output.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    detw = DetectionWriter(out / "detections.jsonl", {
        **meta, "objects_model": Path(mc["objects"]).name if mc.get("objects") else None,
        "object_names": objdet.names if objdet else None, "zones": zones_dict(zones),
        "config": cfg, "total_frames": total})
    csv_f = open(out / "metrics.csv", "w", newline="")
    fields = (["frame", "t"] + [s + "_ms" for s in STAGES] + ["latency_ms", "fps", "tracks",
              "in_zone", "alarm", "cpu", "ram_used_gb", "ram_total_gb", "gpu", "gpu_mem_gb",
              "temp_c", "power_w"])
    cw = csv.DictWriter(csv_f, fieldnames=fields)
    cw.writeheader()

    events, visits, lat_hist = [], [], []
    fps_ema, last_stat, alarm_prev = None, -1.0, False
    alarm_frames = 0
    frame, i = first, 0
    live = Live(console=console, refresh_per_second=8) if dashboard else None
    t_wall = time.perf_counter()
    with SystemMonitor() as mon:
        if live:
            live.start()
        try:
            while ok and (not max_frames or i < max_frames):
                t = i / fps
                tm = {}
                t0 = time.perf_counter()
                dets = pose(frame)
                odets = objdet(frame) if objdet else []
                t1 = time.perf_counter()
                tracks = tracker.update(dets)
                otracks = objtracker.update(odets) if objtracker else []
                t2 = time.perf_counter()
                flow.update(frame)
                vecs = {tr.id: flow.vector(tr.box) for tr in tracks}
                t3 = time.perf_counter()
                state = engine.update(tracks, vecs, t, otracks)
                t4 = time.perf_counter()
                tm["detect"], tm["track"], tm["flow"], tm["rules"] = (
                    (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3)
                latency = (t4 - t0) * 1e3
                fps_ema = 1e3 / latency if fps_ema is None else 0.9 * fps_ema + 0.1 * 1e3 / latency
                img = viz.draw(frame, state, {"backend": f"{meta['model']} | {pose.backend}",
                                              "fps": fps_ema, "latency_ms": latency})
                tm["draw"] = (time.perf_counter() - t4) * 1e3
                detw.write(i, start + i, t, dets, odets, state, vecs)
                lat_hist.append(latency)
                alarm_frames += state.alarm

                s = mon.latest
                in_zone = sum(p.in_zone for p in state.persons)
                row = {"frame": i, "t": round(t, 3), "latency_ms": round(latency, 2),
                       "fps": round(fps_ema, 2), "tracks": len(state.persons), "in_zone": in_zone,
                       "alarm": int(state.alarm)}
                row.update({k + "_ms": round(v, 2) for k, v in tm.items()})
                for k in ("cpu", "ram_used_gb", "ram_total_gb", "gpu", "gpu_mem_gb", "temp_c",
                          "power_w"):
                    row[k] = "" if s.get(k) is None else round(s[k], 2)
                cw.writerow(row)

                for e in state.events:
                    events.append(e.as_dict())
                    log(i, t, "alert", f"ALERT {e.rule:<16} {e.message}")
                for v in state.visits:
                    visits.append(v.as_dict())
                    log(i, t, "info", f"person #{v.track_id} was in the mantrap for "
                                      f"{v.duration:.1f}s ({v.reason.replace('_', ' ')})")
                if alarm_prev and not state.alarm:
                    log(i, t, "ok", "mantrap clear")
                alarm_prev = state.alarm
                if t - last_stat >= 2.0:
                    last_stat = t
                    log(i, t, "stat",
                        f"fps {fps_ema:5.1f} | det {tm['detect']:5.1f} flow {tm['flow']:4.1f} ms | "
                        f"cpu {s['cpu']:3.0f}% gpu {_fmt(s['gpu'], '%')} | tracks {len(state.persons)} "
                        f"in-zone {in_zone}")
                if live:
                    live.update(_dashboard(meta, {**row, **{k + "_ms": v for k, v in tm.items()}},
                                           s, log))
                if writer:
                    writer.write(img)
                if show:
                    cv2.imshow("mantrap-guard", img)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                i += 1
                ok, frame = cap.read()
        finally:
            if live:
                live.stop()
        wall = time.perf_counter() - t_wall
        sysm = mon.summary()

    for v in engine.finish():  # people still inside when the video ends
        visits.append(v.as_dict())
        log(i, i / fps, "info", f"person #{v.track_id} was in the mantrap for {v.duration:.1f}s "
                                f"({v.reason.replace('_', ' ')})")
    durations = [v["duration"] for v in visits]
    lat = np.asarray(lat_hist) if lat_hist else np.zeros(1)
    summary = {**meta, "frames": i, "wall_seconds": round(wall, 2),
               "latency_ms": {"mean": round(float(lat.mean()), 2),
                              "p50": round(float(np.percentile(lat, 50)), 2),
                              "p95": round(float(np.percentile(lat, 95)), 2)},
               "pipeline_fps": round(1e3 / float(lat.mean()), 2) if lat.mean() > 0 else 0,
               "events": len(events), "alarm_frames": alarm_frames,
               "mantrap_visits": {"count": len(visits),
                                  "mean_s": round(float(np.mean(durations)), 2) if durations else 0,
                                  "max_s": round(max(durations), 2) if durations else 0},
               "system": sysm}
    log(i, i / fps, "ok", f"done: {i} frames, {summary['pipeline_fps']:.1f} FPS mean, "
                          f"p95 {summary['latency_ms']['p95']:.1f} ms, {len(events)} alert events")
    log.close()
    csv_f.close()
    detw.close()
    cap.release()
    if writer:
        writer.release()
        to_h264(out / "output.mp4")
    if show:
        cv2.destroyAllWindows()
    (out / "events.json").write_text(json.dumps(events, indent=2))
    (out / "dwell.json").write_text(json.dumps(visits, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    console.print(f"[green]saved[/] {out}/output.mp4, metrics.csv, events.json, dwell.json, detections.jsonl, summary.json")
    return summary
