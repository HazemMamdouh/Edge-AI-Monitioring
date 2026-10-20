"""Merge benchmark results from several machines into one comparison table.

    mantrap compare --inputs docs/benchmarks/cpu.json docs/benchmarks/orin-nx-16gb.json \
                             docs/benchmarks/a10.json --out docs/comparison.md

Every variant x device cell comes from a `mantrap benchmark` JSON. A device that has not
been measured yet, or a variant a device cannot run (TensorRT on a CPU-only machine), is
shown as such, never filled in.
"""

from __future__ import annotations

import json
from pathlib import Path

# (row label, how to recognise the backend in a benchmark result)
VARIANTS = [
    ("Original (PyTorch FP32)", "PyTorch FP32"),
    ("ONNX FP32", "ONNX Runtime FP32"),
    ("ONNX INT8 (CPU)", "ONNX Runtime INT8"),
    ("TensorRT FP32", "TensorRT FP32"),
    ("TensorRT FP16", "TensorRT FP16"),
    ("TensorRT INT8", "TensorRT INT8"),
]
TRT = {"TensorRT FP32", "TensorRT FP16", "TensorRT INT8"}


def _mem(r: dict) -> str:
    s = f"{r['ram_mb']} MB RAM"
    if r.get("gpu_mem_mb"):
        s += f" + {r['gpu_mem_mb']} MB GPU"
    return s


def _cell(r: dict | None, has_gpu: bool, backend: str, measured: bool) -> list[str]:
    if not measured:
        return ["not measured yet"] + [""] * 3
    if r is None:
        if backend in TRT and not has_gpu:
            return ["n/a (no NVIDIA GPU)"] + [""] * 3
        return ["not run"] + [""] * 3
    if "error" in r:
        return [f"failed: {r['error'][:40]}"] + [""] * 3
    util = r["gpu"] if "GPU" in r["unit"] else r["cpu"]
    util = "" if util is None else f", {util:.0f}% busy"
    return [f"**{r['fps']}**", f"{r['latency_ms']} / {r['p95_ms']}", f"{r['unit']}{util}", _mem(r)]


def compare(inputs: list[str], labels: list[str] | None = None, out: str | None = None) -> str:
    runs = [json.loads(Path(p).read_text()) if Path(p).exists() else None for p in inputs]
    labels = labels or [r["label"] if r else Path(p).stem for r, p in zip(runs, inputs)]
    families = sorted({x["family"] for r in runs if r for x in r["results"]})
    lines = ["# Mantrap Guard: device and precision comparison", "",
             "Pose model only, batch 1, 640 px, pre- and post-processing included. "
             "Each cell: **FPS**, latency mean / p95 (ms), processing unit and how busy it was, "
             "peak memory of the process. Produced by `mantrap compare` from `mantrap benchmark` "
             "runs; nothing is estimated.", ""]
    lines += ["## Devices", "", "| Device | Hardware | RAM |", "|---|---|---|"]
    for lab, r in zip(labels, runs):
        if r:
            hw = r["device"] + (f" / {r['gpu']}" if r.get("gpu") and not r.get("jetson") else "")
            lines.append(f"| {lab} | {hw} | {r['ram_total_gb']} GB |")
        else:
            lines.append(f"| {lab} | not measured yet | |")
    for fam in families:
        lines += ["", f"## {fam}", "",
                  "| Variant | Device | FPS | Latency mean / p95 (ms) | Processing unit | Memory "
                  "| Recall vs FP32 | Kpt err (%) |",
                  "|---|---|---:|---:|---|---|---:|---:|"]
        for name, backend in VARIANTS:
            for lab, run in zip(labels, runs):
                r = None
                if run:
                    r = next((x for x in run["results"]
                              if x["family"] == fam and x["backend"] == backend), None)
                has_gpu = bool(run and run.get("gpu"))
                c = _cell(r, has_gpu, backend, run is not None)
                acc = ["", ""]
                if r and "error" not in r:
                    acc = (["ref", "ref"] if r.get("reference") else
                           [str(r.get("recall", "")), str(r.get("kpt_err_pct", ""))])
                lines.append(f"| {name} | {lab} | " + " | ".join(c + acc) + " |")
    md = "\n".join(lines) + "\n"
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(md)
    return md
