**CPU (2 vCPU x86)** (Intel(R) Xeon(R) Processor @ 2.80GHz (2 threads), 7.8 GB RAM)  
**Input:** 640x640, 120 frames of `one-by-one-person-detection.mp4`, batch 1, pre/post-processing included  
**Agreement columns:** each variant against the FP32 `.pt` of the same network (`ref`)

| Model | Backend | Unit | Size (MB) | Mean (ms) | p95 (ms) | FPS | CPU % | GPU % | RAM (MB) | GPU mem (MB) | Power (W) | Recall | Box IoU | Kpt err (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo11n-pose.pt | PyTorch FP32 | CPU (2 threads) | 6.0 | 96.81 | 127.23 | **10.3** | 48.9 | n/a | 894 | n/a | n/a | ref | ref | ref |
| yolo11n-pose.onnx | ONNX Runtime FP32 | CPU (2 threads) | 11.3 | 76.24 | 92.63 | **13.1** | 94.9 | n/a | 973 | n/a | n/a | 1.0 | 0.996 | 0.26 |
| yolo11n-pose_int8.onnx | ONNX Runtime INT8 | CPU (2 threads) | 5.4 | 72.2 | 91.06 | **13.9** | 94.6 | n/a | 917 | n/a | n/a | 1.0 | 0.992 | 0.72 |
| yolo11s-pose.pt | PyTorch FP32 | CPU (2 threads) | 19.4 | 212.33 | 242.12 | **4.7** | 50.2 | n/a | 992 | n/a | n/a | ref | ref | ref |
| yolo11s-pose.onnx | ONNX Runtime FP32 | CPU (2 threads) | 38.1 | 158.58 | 181.57 | **6.3** | 97.5 | n/a | 1072 | n/a | n/a | 1.0 | 0.998 | 0.18 |
| yolo11s-pose_int8.onnx | ONNX Runtime INT8 | CPU (2 threads) | 13.9 | 110.8 | 135.54 | **9.0** | 96.4 | n/a | 945 | n/a | n/a | 1.0 | 0.995 | 0.39 |
