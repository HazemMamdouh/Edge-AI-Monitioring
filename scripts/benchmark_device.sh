#!/usr/bin/env bash
# Fill one column of docs/comparison.md: export every variant this machine can run,
# benchmark them, and rebuild the comparison table.
#
#   bash scripts/benchmark_device.sh "Jetson Orin NX 16GB" orin-nx-16gb   # on the Jetson
#   bash scripts/benchmark_device.sh "NVIDIA A10" a10                     # on the A10 server
#   bash scripts/benchmark_device.sh "CPU (2 vCPU x86)" cpu               # CPU-only machine
#
# TensorRT engines are built on the machine that runs them (they are not portable).
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL=${1:?device label}
NAME=${2:?file name, e.g. a10}
VIDEO=${3:-data/one-by-one-person-detection.mp4}
MODELS="yolo11n-pose yolo11s-pose"

FORMATS="onnx onnx-int8"
if python3 -c "import tensorrt" 2>/dev/null; then FORMATS="$FORMATS fp32 fp16 int8"; fi
if command -v nvpmodel >/dev/null; then sudo nvpmodel -m 0 || true; sudo jetson_clocks || true; fi

for m in $MODELS; do
  mantrap export --weights models/$m.pt --formats $FORMATS --calib-video "$VIDEO" || true
done

LIST=""
for m in $MODELS; do
  for f in models/$m.pt models/$m.onnx models/${m}_int8.onnx \
           models/${m}_fp32.engine models/${m}_fp16.engine models/${m}_int8.engine; do
    [ -f "$f" ] && LIST="$LIST $f"
  done
done
mantrap benchmark --label "$LABEL" --video "$VIDEO" --frames 300 \
  --out "docs/benchmarks/$NAME" --models $LIST

mantrap compare \
  --inputs docs/benchmarks/cpu.json docs/benchmarks/orin-nx-16gb.json docs/benchmarks/a10.json \
  --labels "CPU (2 vCPU x86)" "Jetson Orin NX 16GB" "NVIDIA A10" --out docs/comparison.md
