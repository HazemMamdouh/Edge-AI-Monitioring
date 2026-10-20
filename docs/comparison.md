# Mantrap Guard: device and precision comparison

Pose model only, batch 1, 640 px, pre- and post-processing included. Each cell: **FPS**, latency mean / p95 (ms), processing unit and how busy it was, peak memory of the process. Produced by `mantrap compare` from `mantrap benchmark` runs; nothing is estimated.

## Devices

| Device | Hardware | RAM |
|---|---|---|
| CPU (2 vCPU x86) | Intel(R) Xeon(R) Processor @ 2.80GHz (2 threads) | 7.8 GB |
| Jetson Orin NX 16GB | not measured yet | |
| NVIDIA A10 | not measured yet | |

## yolo11n-pose

| Variant | Device | FPS | Latency mean / p95 (ms) | Processing unit | Memory | Recall vs FP32 | Kpt err (%) |
|---|---|---:|---:|---|---|---:|---:|
| Original (PyTorch FP32) | CPU (2 vCPU x86) | **10.3** | 96.81 / 127.23 | CPU (2 threads), 49% busy | 894 MB RAM | ref | ref |
| Original (PyTorch FP32) | Jetson Orin NX 16GB | not measured yet |  |  |  |  |  |
| Original (PyTorch FP32) | NVIDIA A10 | not measured yet |  |  |  |  |  |
| ONNX FP32 | CPU (2 vCPU x86) | **13.1** | 76.24 / 92.63 | CPU (2 threads), 95% busy | 973 MB RAM | 1.0 | 0.26 |
| ONNX FP32 | Jetson Orin NX 16GB | not measured yet |  |  |  |  |  |
| ONNX FP32 | NVIDIA A10 | not measured yet |  |  |  |  |  |
| ONNX INT8 (CPU) | CPU (2 vCPU x86) | **13.9** | 72.2 / 91.06 | CPU (2 threads), 95% busy | 917 MB RAM | 1.0 | 0.72 |
| ONNX INT8 (CPU) | Jetson Orin NX 16GB | not measured yet |  |  |  |  |  |
| ONNX INT8 (CPU) | NVIDIA A10 | not measured yet |  |  |  |  |  |
| TensorRT FP32 | CPU (2 vCPU x86) | n/a (no NVIDIA GPU) |  |  |  |  |  |
| TensorRT FP32 | Jetson Orin NX 16GB | not measured yet |  |  |  |  |  |
| TensorRT FP32 | NVIDIA A10 | not measured yet |  |  |  |  |  |
| TensorRT FP16 | CPU (2 vCPU x86) | n/a (no NVIDIA GPU) |  |  |  |  |  |
| TensorRT FP16 | Jetson Orin NX 16GB | not measured yet |  |  |  |  |  |
| TensorRT FP16 | NVIDIA A10 | not measured yet |  |  |  |  |  |
| TensorRT INT8 | CPU (2 vCPU x86) | n/a (no NVIDIA GPU) |  |  |  |  |  |
| TensorRT INT8 | Jetson Orin NX 16GB | not measured yet |  |  |  |  |  |
| TensorRT INT8 | NVIDIA A10 | not measured yet |  |  |  |  |  |

## yolo11s-pose

| Variant | Device | FPS | Latency mean / p95 (ms) | Processing unit | Memory | Recall vs FP32 | Kpt err (%) |
|---|---|---:|---:|---|---|---:|---:|
| Original (PyTorch FP32) | CPU (2 vCPU x86) | **4.7** | 212.33 / 242.12 | CPU (2 threads), 50% busy | 992 MB RAM | ref | ref |
| Original (PyTorch FP32) | Jetson Orin NX 16GB | not measured yet |  |  |  |  |  |
| Original (PyTorch FP32) | NVIDIA A10 | not measured yet |  |  |  |  |  |
| ONNX FP32 | CPU (2 vCPU x86) | **6.3** | 158.58 / 181.57 | CPU (2 threads), 98% busy | 1072 MB RAM | 1.0 | 0.18 |
| ONNX FP32 | Jetson Orin NX 16GB | not measured yet |  |  |  |  |  |
| ONNX FP32 | NVIDIA A10 | not measured yet |  |  |  |  |  |
| ONNX INT8 (CPU) | CPU (2 vCPU x86) | **9.0** | 110.8 / 135.54 | CPU (2 threads), 96% busy | 945 MB RAM | 1.0 | 0.39 |
| ONNX INT8 (CPU) | Jetson Orin NX 16GB | not measured yet |  |  |  |  |  |
| ONNX INT8 (CPU) | NVIDIA A10 | not measured yet |  |  |  |  |  |
| TensorRT FP32 | CPU (2 vCPU x86) | n/a (no NVIDIA GPU) |  |  |  |  |  |
| TensorRT FP32 | Jetson Orin NX 16GB | not measured yet |  |  |  |  |  |
| TensorRT FP32 | NVIDIA A10 | not measured yet |  |  |  |  |  |
| TensorRT FP16 | CPU (2 vCPU x86) | n/a (no NVIDIA GPU) |  |  |  |  |  |
| TensorRT FP16 | Jetson Orin NX 16GB | not measured yet |  |  |  |  |  |
| TensorRT FP16 | NVIDIA A10 | not measured yet |  |  |  |  |  |
| TensorRT INT8 | CPU (2 vCPU x86) | n/a (no NVIDIA GPU) |  |  |  |  |  |
| TensorRT INT8 | Jetson Orin NX 16GB | not measured yet |  |  |  |  |  |
| TensorRT INT8 | NVIDIA A10 | not measured yet |  |  |  |  |  |
