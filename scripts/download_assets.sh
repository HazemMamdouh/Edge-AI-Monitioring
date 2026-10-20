#!/usr/bin/env bash
# Fetch the published weights and the demo clip used in the README.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models data

REL=https://github.com/ultralytics/assets/releases/download/v8.3.0
for m in yolo11n-pose.pt yolo11s-pose.pt yolo11n.pt; do
  [ -f "models/$m" ] || curl -L --fail -o "models/$m" "$REL/$m"
done

# Demo clip: Intel IoT DevKit sample videos, CC BY 4.0
# https://github.com/intel-iot-devkit/sample-videos
V=one-by-one-person-detection.mp4
[ -f "data/$V" ] || curl -L --fail -o "data/$V" \
  "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/$V"
ls -lh models data
