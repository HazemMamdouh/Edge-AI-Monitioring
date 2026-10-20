"""Benchmark model variants (.pt, ONNX FP32/INT8, TensorRT FP32/FP16/INT8) on real video frames.

Each variant runs in its own child process, so its memory figures are its own and not
left over from the previous model. Measured per variant:
  - latency (mean, p50, p95) and FPS, batch 1, pre- and post-processing included
  - processing unit (CPU threads / CUDA / TensorRT) and its utilisation
  - memory: process RAM (peak RSS) and, on discrete GPUs, the process's GPU memory
  - power, where the board exposes it (Jetson)
Each variant is also compared against the FP32 .pt of the same network on the same
frames, so the cost of quantisation is visible without a labelled dataset:
  - person recall  : share of reference people that the variant also finds (IoU >= 0.5)
  - box IoU        : mean IoU of the matched boxes
  - keypoint error : mean distance between matched keypoints, as % of the box height
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

from .detector import PoseDetector, backend_name
from .sysmon import SystemMonitor, device_name, gpu_name, gpu_process_memory_mb, is_jetson
from .tracker import iou_matrix


def load_frames(video: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, max(total - 1, 0), min(n, total)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {video}")
    return frames


def agreement(ref: list, got: list, conf: float = 0.5) -> dict:
    """Compare per-frame detections of a variant against the reference model."""
    n_ref = n_hit = 0
    ious, kerr = [], []
    for r, g in zip(ref, got):
        r = [d for d in r if d.conf >= conf]
        g = [d for d in g if d.conf >= conf * 0.5]
        n_ref += len(r)
        if not r or not g:
            continue
        iou = iou_matrix(np.stack([d.box for d in r]), np.stack([d.box for d in g]))
        for i, d in enumerate(r):
            j = int(iou[i].argmax())
            if iou[i, j] < 0.5:
                continue
            n_hit += 1
            ious.append(float(iou[i, j]))
            a, b = d.kpts, g[j].kpts
            if a is not None and b is not None:
                vis = (a[:, 2] > 0.5) & (b[:, 2] > 0.5)
                if vis.any():
                    h = max(float(d.box[3] - d.box[1]), 1.0)
                    kerr.append(float(np.linalg.norm(a[vis, :2] - b[vis, :2], axis=1).mean() / h))
    return {"recall": round(n_hit / n_ref, 3) if n_ref else None,
            "box_iou": round(float(np.mean(ious)), 3) if ious else None,
            "kpt_err_pct": round(100 * float(np.mean(kerr)), 2) if kerr else None}


def family(path: str) -> str:
    """yolo11n-pose_int8.engine -> yolo11n-pose : variants of one network share a family."""
    stem = Path(path).stem
    for suffix in ("_fp16", "_int8", "_fp32"):
        stem = stem.removesuffix(suffix)
    return stem


def _unit(model: str, infer_device: str) -> str:
    if model.endswith(".engine"):
        return "GPU (TensorRT)"
    if "cuda" in infer_device:
        return "GPU (CUDA)"
    return f"CPU ({psutil.cpu_count()} threads)"


def _run_one(model: str, video: str, imgsz: int, frames: int, warmup: int, device, q) -> None:
    """Child process: time one variant and report back through the queue."""
    try:
        imgs = load_frames(video, frames)
        proc = psutil.Process(os.getpid())
        peak = {"rss": proc.memory_info().rss, "gpu": None}
        stop = threading.Event()

        def sample():  # memory sampler, every 50 ms
            while not stop.wait(0.05):
                peak["rss"] = max(peak["rss"], proc.memory_info().rss)
                g = gpu_process_memory_mb(os.getpid())
                if g is not None:
                    peak["gpu"] = max(peak["gpu"] or 0, g)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        det = PoseDetector(model, imgsz=imgsz, conf=0.15, device=device)
        det.warmup(imgs[0], warmup)
        lat, dets = [], []
        with SystemMonitor(interval=0.25) as mon:
            for img in imgs:
                t0 = time.perf_counter()
                dets.append(det(img))
                lat.append((time.perf_counter() - t0) * 1e3)
            sysm = mon.summary()
        stop.set()
        sampler.join(timeout=1)
        lat = np.asarray(lat)
        q.put({"ok": True, "dets": dets, "row": dict(
            latency_ms=round(float(lat.mean()), 2),
            p50_ms=round(float(np.percentile(lat, 50)), 2),
            p95_ms=round(float(np.percentile(lat, 95)), 2),
            fps=round(1e3 / float(lat.mean()), 1),
            unit=_unit(model, det.device), infer_device=det.device,
            cpu=sysm["cpu_avg"], gpu=sysm["gpu_avg"], power_w=sysm["power_w_avg"],
            ram_mb=round(peak["rss"] / 2**20), gpu_mem_mb=None if peak["gpu"] is None
            else round(peak["gpu"]))})
    except Exception as e:  # noqa: BLE001
        q.put({"ok": False, "error": str(e).splitlines()[0][:200]})


def benchmark(models: list[str], video: str, imgsz: int = 640, frames: int = 150, warmup: int = 15,
              device=None, reference: str | None = None, out: str | None = None,
              label: str | None = None) -> dict:
    """Time every model. Each one is compared with the reference of its own family (the .pt
    with the same base name, or `reference` if given)."""
    refs = {family(m): m for m in models if m.endswith(".pt")}
    order = sorted(models, key=lambda m: (family(m), m not in refs.values() and m != reference))
    ctx = mp.get_context("spawn")
    rows, ref_dets = [], {}
    for m in order:
        fam = "*" if reference else family(m)
        is_ref = m == (reference or refs.get(fam))
        row = {"model": Path(m).name, "family": family(m), "backend": backend_name(m),
               "size_mb": round(Path(m).stat().st_size / 2**20, 1)}
        q = ctx.Queue()
        p = ctx.Process(target=_run_one, args=(m, video, imgsz, frames, warmup, device, q))
        p.start()
        res = q.get()
        p.join()
        if not res["ok"]:
            row["error"] = res["error"]
        else:
            row.update(res["row"])
            if is_ref:
                ref_dets[fam] = res["dets"]
                row["reference"] = True
            elif fam in ref_dets:
                row.update(agreement(ref_dets[fam], res["dets"]))
        rows.append(row)
    vm = psutil.virtual_memory()
    result = {"label": label or device_name(), "device": device_name(), "gpu": gpu_name(),
              "jetson": is_jetson(), "ram_total_gb": round(vm.total / 2**30, 1),
              "video": str(video), "imgsz": imgsz, "frames": frames, "results": rows}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).with_suffix(".json").write_text(json.dumps(result, indent=2))
        Path(out).with_suffix(".md").write_text(to_markdown(result))
    return result


def _v(x, suffix=""):
    return "n/a" if x is None else f"{x}{suffix}"


def to_markdown(res: dict) -> str:
    gpu = f", GPU: {res['gpu']}" if res.get("gpu") else ""
    lines = [f"**{res['label']}** ({res['device']}{gpu}, {res['ram_total_gb']} GB RAM)  ",
             f"**Input:** {res['imgsz']}x{res['imgsz']}, {res['frames']} frames of "
             f"`{Path(res['video']).name}`, batch 1, pre/post-processing included  ",
             "**Agreement columns:** each variant against the FP32 `.pt` of the same network (`ref`)",
             "",
             "| Model | Backend | Unit | Size (MB) | Mean (ms) | p95 (ms) | FPS | CPU % | GPU % "
             "| RAM (MB) | GPU mem (MB) | Power (W) | Recall | Box IoU | Kpt err (%) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in res["results"]:
        if "error" in r:
            lines.append(f"| {r['model']} | {r['backend']} | | {r['size_mb']} | failed: {r['error']} "
                         "| | | | | | | | | | |")
            continue
        ref = r.get("reference")
        lines.append(
            f"| {r['model']} | {r['backend']} | {r['unit']} | {r['size_mb']} | {r['latency_ms']} | "
            f"{r['p95_ms']} | **{r['fps']}** | {_v(r['cpu'])} | {_v(r['gpu'])} | {r['ram_mb']} | "
            f"{_v(r['gpu_mem_mb'])} | {_v(r['power_w'])} | "
            f"{'ref' if ref else _v(r.get('recall'))} | {'ref' if ref else _v(r.get('box_iou'))} | "
            f"{'ref' if ref else _v(r.get('kpt_err_pct'))} |")
    return "\n".join(lines) + "\n"
