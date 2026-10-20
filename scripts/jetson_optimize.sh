#!/usr/bin/env bash
# Run ON THE JETSON (Orin NX 16GB, JetPack 6). Builds ONNX + TensorRT FP16 + TensorRT INT8
# for every model, benchmarks all variants and renders the showcase with the INT8 engine.
# TensorRT engines are specific to the device and TensorRT version: do not copy them
# from another machine.
set -euo pipefail
cd "$(dirname "$0")/.."
VIDEO=${1:-data/one-by-one-person-detection.mp4}
ZONES=${2:-configs/zones/demo.json}

# max performance while measuring (sudo): MAXN power mode + fixed clocks
if command -v nvpmodel >/dev/null; then sudo nvpmodel -m 0 || true; fi
if command -v jetson_clocks >/dev/null; then sudo jetson_clocks || true; fi

# pose models: every variant + benchmark + rebuild docs/comparison.md
bash scripts/benchmark_device.sh "Jetson Orin NX 16GB" orin-nx-16gb "$VIDEO"

# object model for the left-object rule
mantrap export --weights models/yolo11n.pt --formats onnx int8 --calib-video "$VIDEO"

mantrap run --video "$VIDEO" --zones "$ZONES" --model models/yolo11s-pose_int8.engine \
  --objects-model models/yolo11n_int8.engine --out outputs/jetson_int8 --showcase
