#!/usr/bin/env bash
# Reproduce the README demo end to end on any machine (CPU is enough).
set -euo pipefail
cd "$(dirname "$0")/.."
VIDEO=data/one-by-one-person-detection.mp4
bash scripts/download_assets.sh

mantrap export --weights models/yolo11n-pose.pt --formats onnx
mantrap run --video "$VIDEO" --zones configs/zones/demo.json \
  --model models/yolo11n-pose.onnx --max-frames 480 --out outputs/demo --showcase
echo "open outputs/demo/showcase.mp4"
