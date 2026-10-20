# Running on NVIDIA Jetson Orin NX 16GB

Target: JetPack 6.x (TensorRT 8.6 or 10.3). The builder in `export.py` supports TensorRT 8.5 to 10.x.

## 1. Environment

TensorRT, CUDA and cuDNN come with JetPack and live in the system Python. Create the virtual
environment **with** access to them:

```bash
sudo apt install -y python3-venv ffmpeg
python3 -m venv --system-site-packages ~/venvs/mantrap && source ~/venvs/mantrap/bin/activate
python3 -c "import tensorrt; print(tensorrt.__version__)"     # must work before going on
```

Install the **NVIDIA build of PyTorch for your JetPack version** (the generic PyPI wheel has no
CUDA support on Jetson), then the project:

```bash
# PyTorch + torchvision: use the wheels NVIDIA publishes for your JetPack release
pip install -e ".[export]"
sudo pip3 install jetson-stats        # optional: jtop gives GPU %, temperature and power
```

If `import tensorrt` or `cv2` fails after installing, NumPy 2 is the usual cause on JetPack
images built against NumPy 1: `pip install "numpy<2"`.

## 2. Optimise, benchmark, showcase

```bash
bash scripts/download_assets.sh
bash scripts/jetson_optimize.sh data/site.mp4 configs/zones/site.json
```

Step by step, this is what the script does:

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks      # MAXN, fixed clocks: stable measurements

mantrap export --weights models/yolo11s-pose.pt models/yolo11n.pt \
  --formats onnx fp16 int8 --calib-video data/site.mp4 --calib-images 300

bash scripts/benchmark_device.sh "Jetson Orin NX 16GB" orin-nx-16gb data/site.mp4
#   -> builds ONNX, ONNX INT8, TensorRT FP32 / FP16 / INT8 for both pose models,
#      benchmarks them and rebuilds docs/comparison.md

mantrap run --video data/site.mp4 --zones configs/zones/site.json \
  --model models/yolo11s-pose_int8.engine --objects-model models/yolo11n_int8.engine \
  --out outputs/jetson_int8 --showcase
```

Building an engine takes a few minutes per model; INT8 takes longer because of calibration.

## 3. Notes

- **Build on the device.** A TensorRT engine is valid only for the GPU and TensorRT version it
  was built with.
- **Calibration data.** Use 200 to 500 frames from the real camera, covering empty, busy, day
  and night conditions. The calibration cache (`*_int8_minmax.calib`) is reused on the next
  build; delete it after changing the calibration set.
- **If INT8 hurts keypoints.** Check the "Keypoint error" column of the benchmark. Try
  `--calib-algo entropy`, more calibration frames, or stay on FP16 for the pose model and use
  INT8 only for the object detector.
- **ONNX on Jetson** runs on the CPU with the stock `onnxruntime` wheel. It is in the benchmark
  as a portability baseline, not as a deployment option.
- **GPU statistics** come from `jtop` if installed, otherwise from the GPU load node in sysfs;
  board power is read from the INA3221 sensor. A value that cannot be read is reported as `n/a`.
- **Memory.** Orin NX 16GB shares RAM between CPU and GPU. The RAM figure in the system panel
  is therefore the number to watch, not a separate GPU-memory figure.

## Filling the A10 column

On any Linux server with an NVIDIA A10 (cloud instances are fine): install a CUDA build of
PyTorch, then `pip install tensorrt pynvml -e ".[export]"`, and run

```bash
bash scripts/benchmark_device.sh "NVIDIA A10" a10
```

On a discrete GPU the memory column also reports the process's GPU memory (from NVML).
