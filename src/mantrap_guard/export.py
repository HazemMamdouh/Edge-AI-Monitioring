"""Model optimisation: PyTorch -> ONNX -> TensorRT FP16 / TensorRT INT8.

ONNX export uses the Ultralytics exporter. The TensorRT engines are built here with the
TensorRT Python API so that INT8 calibration runs on frames from *your own camera*
(`--calib-video`), which matters far more for INT8 accuracy than calibrating on COCO.

TensorRT engines are tied to the GPU and TensorRT version they were built with:
build them on the target device (the Jetson), not on a workstation.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------------- ONNX
def export_onnx(weights: str | Path, imgsz: int = 640, opset: int | None = None,
                simplify: bool = True) -> Path:
    from ultralytics import YOLO

    weights = Path(weights)
    kw = dict(format="onnx", imgsz=imgsz, simplify=simplify, dynamic=False, half=False)
    if opset:
        kw["opset"] = opset
    out = Path(YOLO(str(weights)).export(**kw))
    target = weights.with_suffix(".onnx")
    if out.resolve() != target.resolve():
        out.replace(target)
    return target


def onnx_metadata(onnx_path: str | Path) -> dict:
    """Ultralytics stores task / names / kpt_shape / imgsz in the ONNX metadata."""
    import onnx

    m = onnx.load(str(onnx_path), load_external_data=False)
    return {p.key: p.value for p in m.metadata_props}


# ------------------------------------------------------------ calibration
def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas


def preprocess(img: np.ndarray, size: int) -> np.ndarray:
    """Same preprocessing the runtime applies: letterbox, BGR->RGB, 0..1, CHW float32."""
    x = letterbox(img, size)[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(x, np.float32) / 255.0


def calibration_frames(video: str | None, image_dir: str | None, n: int, size: int):
    """Yield up to `n` preprocessed frames, evenly spread over the video / directory."""
    if image_dir:
        files = sorted(p for p in Path(image_dir).iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        for p in files[:: max(1, len(files) // n)][:n]:
            img = cv2.imread(str(p))
            if img is not None:
                yield preprocess(img, size)
        return
    if not video:
        raise ValueError("INT8 needs calibration data: pass --calib-video or --calib-dir")
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in np.linspace(0, max(total - 1, 0), min(n, max(total, 1))).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            yield preprocess(frame, size)
    cap.release()


class TensorRTUnavailable(RuntimeError):
    """TensorRT is not installed on this machine (expected on a laptop / CI box)."""


def _make_calibrator(trt, frames, batch: int, cache: Path, algo: str):
    import torch  # device memory for the calibration batches

    base = trt.IInt8MinMaxCalibrator if algo == "minmax" else trt.IInt8EntropyCalibrator2

    class Calibrator(base):
        def __init__(self):
            super().__init__()
            self.it = iter(frames)
            self.buf = None  # keep a reference: TensorRT reads the pointer after get_batch returns
            self.count = 0

        def get_batch_size(self):
            return batch

        def get_batch(self, names):
            chunk = []
            for x in self.it:
                chunk.append(x)
                if len(chunk) == batch:
                    break
            if len(chunk) < batch:
                return None
            self.buf = torch.from_numpy(np.stack(chunk)).cuda().contiguous()
            self.count += batch
            return [int(self.buf.data_ptr())]

        def read_calibration_cache(self):
            return cache.read_bytes() if cache.exists() else None

        def write_calibration_cache(self, data):
            cache.write_bytes(bytes(data))

    return Calibrator()


def _protect_head(trt, network, config) -> int:
    """Mixed precision for INT8: keep the head's Sigmoid in FP32 and its output convolutions /
    DFL in FP16. Quantising these to 8 bits is what wrecks confidence scores and keypoint
    regression, while they are a negligible share of the compute. (Same policy as the
    Ultralytics exporter; layer names follow the ONNX node names, e.g. /model.23/...)"""
    names = [network.get_layer(i).name for i in range(network.num_layers)]
    idx = [int(m.group(1)) for n in names if (m := re.search(r"/model\.(\d+)/", n))]
    head = f"/model.{max(idx)}/" if idx else "/"
    count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if head not in layer.name:
            continue
        if layer.type == trt.LayerType.ACTIVATION and "sigmoid" in layer.name.lower():
            dtype = trt.float32
        elif layer.type == trt.LayerType.CONVOLUTION and (
                layer.name.endswith(".2/Conv") or "/dfl/" in layer.name):
            dtype = trt.float16
        else:
            continue
        layer.precision = dtype
        for j in range(layer.num_outputs):
            layer.set_output_type(j, dtype)
        count += 1
    if count:
        config.set_flag(getattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS", None)
                        or trt.BuilderFlag.STRICT_TYPES)
    return count


# ---------------------------------------------------------------- TensorRT
def build_engine(onnx_path: str | Path, precision: str = "fp16", workspace_gb: float = 4.0,
                 calib_video: str | None = None, calib_dir: str | None = None,
                 calib_images: int = 300, calib_algo: str = "minmax", protect_head: bool = True,
                 verbose: bool = False) -> Path:
    """Build a TensorRT engine from ONNX.

    precision 'fp32': plain TensorRT FP32 (the speed-up from TensorRT alone, no quantisation).
    precision 'fp16': FP16 where it is faster, FP32 elsewhere.
    precision 'int8': INT8 with post-training calibration, FP16 fallback for layers without an
    INT8 kernel, detection/pose head kept out of INT8 (see `_protect_head`).
    Written for TensorRT 8.5 - 10.x (JetPack 5 / 6).
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise TensorRTUnavailable(
            "TensorRT not installed (it ships with JetPack; build engines on the Jetson)") from e
    if precision not in ("fp32", "fp16", "int8"):
        raise ValueError("precision must be fp32, fp16 or int8")
    major = int(trt.__version__.split(".", 1)[0])
    if major >= 11:
        raise RuntimeError(f"TensorRT {trt.__version__}: builder flags for FP16/INT8 were removed "
                           "in TensorRT 11; this builder supports 8.5 - 10.x")

    onnx_path = Path(onnx_path)
    meta = onnx_metadata(onnx_path)
    engine_path = onnx_path.with_name(f"{onnx_path.stem}_{precision}.engine")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    ws = int(workspace_gb * (1 << 30))
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws)
    else:  # TensorRT < 8.4
        config.max_workspace_size = ws
    flag = 0 if major >= 10 else 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errs}")

    inp = network.get_input(0)
    batch, size = int(inp.shape[0]), int(inp.shape[2])
    # platform_has_fast_* were removed from the builder in TensorRT 10
    if precision != "fp32" and getattr(builder, "platform_has_fast_fp16", True):
        config.set_flag(trt.BuilderFlag.FP16)
    calibrator = None
    if precision == "int8":
        if not getattr(builder, "platform_has_fast_int8", True):
            raise RuntimeError("this GPU has no fast INT8 support")
        config.set_flag(trt.BuilderFlag.INT8)
        cache = onnx_path.with_name(f"{onnx_path.stem}_int8_{calib_algo}.calib")
        frames = calibration_frames(calib_video, calib_dir, calib_images, size)
        calibrator = _make_calibrator(trt, frames, max(batch, 1), cache, calib_algo)
        config.int8_calibrator = calibrator
        if protect_head:
            _protect_head(trt, network, config)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed (see log above)")

    # Same container Ultralytics uses: [4-byte length][json metadata][engine]. This lets the
    # runtime recover task / class names / keypoint shape straight from the .engine file.
    header = json.dumps({**meta, "precision": precision}).encode()
    with open(engine_path, "wb") as f:
        f.write(len(header).to_bytes(4, byteorder="little", signed=True))
        f.write(header)
        f.write(bytes(serialized))
    return engine_path


# ------------------------------------------------------------ ONNX INT8 (CPU)
def quantize_onnx_int8(onnx_path: str | Path, calib_video: str | None = None,
                       calib_dir: str | None = None, calib_images: int = 200,
                       protect_head: bool = True) -> Path:
    """Static INT8 quantisation for ONNX Runtime (QDQ, per-channel weights), calibrated on
    frames of your own video. This is the CPU counterpart of the TensorRT INT8 engine.
    Same policy as on the GPU: the detection/pose head stays in float."""
    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    onnx_path = Path(onnx_path)
    out = onnx_path.with_name(f"{onnx_path.stem}_int8.onnx")
    model = onnx.load(str(onnx_path))
    inp = model.graph.input[0]
    size = inp.type.tensor_type.shape.dim[2].dim_value or 640
    meta = {p.key: p.value for p in model.metadata_props}

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calibration_frames(calib_video, calib_dir, calib_images, size))

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {inp.name: x[None]}

    exclude = []
    if protect_head:
        idx = [int(m.group(1)) for n in model.graph.node if (m := re.search(r"/model\.(\d+)/", n.name))]
        if idx:
            head = f"/model.{max(idx)}/"
            exclude = [n.name for n in model.graph.node if head in n.name]
    pre = onnx_path.with_name(f"{onnx_path.stem}_pre.onnx")
    try:
        quant_pre_process(str(onnx_path), str(pre), skip_symbolic_shape=True)
        src = pre
    except Exception:  # noqa: BLE001 - pre-processing is an optimisation, not required
        src = onnx_path
    quantize_static(str(src), str(out), Reader(), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, nodes_to_exclude=exclude)
    pre.unlink(missing_ok=True)
    q = onnx.load(str(out))  # keep the Ultralytics metadata (task, names, kpt_shape)
    del q.metadata_props[:]
    for k, v in {**meta, "precision": "int8"}.items():
        q.metadata_props.add(key=k, value=v)
    onnx.save(q, str(out))
    return out


# --------------------------------------------------------------- top level
def export_all(weights: list[str], formats: list[str], imgsz: int = 640, **trt_kw) -> list[dict]:
    """Export every model to every requested format. Failures are reported, not hidden."""
    report = []
    for w in weights:
        w = Path(w)
        row = {"model": w.name, "pt_mb": round(w.stat().st_size / 2**20, 1)}
        onnx_path = w.with_suffix(".onnx")
        if "onnx" in formats or not onnx_path.exists():
            t0 = time.time()
            try:
                onnx_path = export_onnx(w, imgsz)
                row["onnx"] = {"path": str(onnx_path), "mb": round(onnx_path.stat().st_size / 2**20, 1),
                               "seconds": round(time.time() - t0, 1)}
            except Exception as e:  # noqa: BLE001
                row["onnx"] = {"error": str(e)}
                report.append(row)
                continue
        if "onnx-int8" in formats:
            t0 = time.time()
            try:
                p = quantize_onnx_int8(onnx_path, trt_kw.get("calib_video"), trt_kw.get("calib_dir"),
                                       min(trt_kw.get("calib_images", 200), 200),
                                       trt_kw.get("protect_head", True))
                row["onnx-int8"] = {"path": str(p), "mb": round(p.stat().st_size / 2**20, 1),
                                    "seconds": round(time.time() - t0, 1)}
            except Exception as e:  # noqa: BLE001
                row["onnx-int8"] = {"error": str(e).splitlines()[0]}
        for prec in ("fp32", "fp16", "int8"):
            if prec not in formats:
                continue
            t0 = time.time()
            try:
                p = build_engine(onnx_path, prec, **trt_kw)
                row[prec] = {"path": str(p), "mb": round(p.stat().st_size / 2**20, 1),
                             "seconds": round(time.time() - t0, 1)}
            except TensorRTUnavailable as e:
                row[prec] = {"unavailable": str(e)}
            except Exception as e:  # noqa: BLE001
                row[prec] = {"error": str(e).splitlines()[0]}
        report.append(row)
    return report
