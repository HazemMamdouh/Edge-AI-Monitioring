"""Background sampler for CPU / RAM / GPU / temperature / power.

GPU sources, tried in order:
  1. jtop (jetson-stats) on Jetson, if installed
  2. Jetson sysfs GPU load node (no extra package needed)
  3. NVML (pynvml) or `nvidia-smi` on desktop/server GPUs
If none works the GPU fields stay None and are shown as "n/a" (never guessed).
"""

from __future__ import annotations

import glob
import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path

import psutil

_JETSON_GPU_LOAD = [
    "/sys/devices/platform/bus@0/17000000.gpu/load",  # Orin, JetPack 6
    "/sys/devices/platform/17000000.gpu/load",
    "/sys/devices/17000000.ga10b/load",  # Orin, JetPack 5
    "/sys/devices/platform/gpu.0/load",
    "/sys/devices/gpu.0/load",
]


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def device_name() -> str:
    model = _read("/proc/device-tree/model")
    if model:
        return model.replace("\x00", "").strip()
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{cpu} ({psutil.cpu_count()} threads)"


def is_jetson() -> bool:
    return os.path.exists("/etc/nv_tegra_release") or "jetson" in device_name().lower()


class SystemMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.latest: dict = {"cpu": 0.0, "cpu_per_core": [], "ram_used_gb": 0.0, "ram_total_gb": 0.0,
                             "ram_pct": 0.0, "gpu": None, "gpu_mem_gb": None, "temp_c": None,
                             "power_w": None}
        self.history: list[dict] = []
        self.gpu_source = "none"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._jtop = None
        self._nvml = None
        self._gpu_node = next((p for p in _JETSON_GPU_LOAD if os.path.exists(p)), None)
        if self._gpu_node is None:
            hits = [h for pat in ("/sys/devices/*gpu*/load", "/sys/devices/platform/*gpu*/load",
                                  "/sys/devices/platform/bus@0/*gpu*/load")
                    for h in glob.glob(pat)]
            self._gpu_node = hits[0] if hits else None
        self._setup_gpu()

    # ------------------------------------------------------------------ gpu
    def _setup_gpu(self):
        if is_jetson():
            try:
                from jtop import jtop  # type: ignore

                self._jtop = jtop(interval=self.interval)
                self._jtop.start()
                self.gpu_source = "jtop"
                return
            except Exception:
                self._jtop = None
        if self._gpu_node:
            self.gpu_source = "sysfs"
            return
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
            self.gpu_source = "nvml"
            return
        except Exception:
            self._nvml = None
        if shutil.which("nvidia-smi"):
            self.gpu_source = "nvidia-smi"

    def _gpu(self) -> tuple[float | None, float | None, float | None]:
        """-> (util %, memory GB, power W)"""
        try:
            if self._jtop is not None and self._jtop.ok(spin=True):
                s = self._jtop.stats
                power = s.get("Power TOT")
                return float(s.get("GPU", 0)), None, (power / 1000.0 if power else None)
            if self.gpu_source == "sysfs":
                v = _read(self._gpu_node)
                return (float(v) / 10.0 if v else None), None, self._jetson_power()
            if self._nvml is not None:
                nv, h = self._nvml
                u = nv.nvmlDeviceGetUtilizationRates(h).gpu
                m = nv.nvmlDeviceGetMemoryInfo(h).used / 2**30
                try:
                    p = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:
                    p = None
                return float(u), float(m), p
            if self.gpu_source == "nvidia-smi":
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2
                ).stdout.strip().splitlines()[0].split(",")
                return float(out[0]), float(out[1]) / 1024.0, float(out[2])
        except Exception:
            pass
        return None, None, None

    @staticmethod
    def _jetson_power() -> float | None:
        """VDD_IN from the on-board INA3221 (Orin): channel 1 voltage (mV) x current (mA)."""
        for d in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"):
            v, c = _read(f"{d}/in1_input"), _read(f"{d}/curr1_input")
            if v and c:
                return float(v) * float(c) / 1e6
        return None

    @staticmethod
    def _temp() -> float | None:
        best = None
        for zone in glob.glob("/sys/devices/virtual/thermal/thermal_zone*"):
            v = _read(f"{zone}/temp")
            try:
                t = float(v) / 1000.0
            except (TypeError, ValueError):
                continue
            if 0 < t < 150:
                best = t if best is None else max(best, t)
        if best is None:
            try:
                temps = psutil.sensors_temperatures()
                vals = [x.current for v in temps.values() for x in v]
                best = max(vals) if vals else None
            except Exception:
                best = None
        return best

    # --------------------------------------------------------------- thread
    def sample(self) -> dict:
        vm = psutil.virtual_memory()
        per_core = psutil.cpu_percent(percpu=True)
        gpu, gmem, power = self._gpu()
        self.latest = {
            "t": time.time(),
            "cpu": sum(per_core) / max(len(per_core), 1),
            "cpu_per_core": per_core,
            "ram_used_gb": (vm.total - vm.available) / 2**30,
            "ram_total_gb": vm.total / 2**30,
            "ram_pct": vm.percent,
            "gpu": gpu,
            "gpu_mem_gb": gmem,
            "temp_c": self._temp(),
            "power_w": power,
        }
        self.history.append(self.latest)
        return self.latest

    def _loop(self):
        while not self._stop.wait(self.interval):
            self.sample()

    def start(self) -> "SystemMonitor":
        psutil.cpu_percent(percpu=True)  # prime the counters, then take a first real sample
        time.sleep(0.2)
        self.sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._jtop is not None:
            try:
                self._jtop.close()
            except Exception:
                pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()

    def summary(self) -> dict:
        def stat(key):
            vals = [h[key] for h in self.history if h.get(key) is not None]
            return (round(sum(vals) / len(vals), 1), round(max(vals), 1)) if vals else (None, None)

        out = {"device": device_name(), "gpu_source": self.gpu_source, "samples": len(self.history)}
        for key in ("cpu", "ram_used_gb", "gpu", "gpu_mem_gb", "temp_c", "power_w"):
            out[f"{key}_avg"], out[f"{key}_max"] = stat(key)
        return out


def gpu_name() -> str | None:
    """Name of the first CUDA GPU, or the Jetson module name, or None."""
    if is_jetson():
        return device_name()
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        return n.decode() if isinstance(n, bytes) else n
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        try:
            return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=3).stdout.splitlines()[0]
        except Exception:
            return None
    return None


def gpu_process_memory_mb(pid: int) -> float | None:
    """GPU memory used by one process on a discrete GPU (NVML). None on Jetson, where CPU
    and GPU share RAM and the process RSS already covers it, and on machines without NVML."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
            if p.pid == pid and p.usedGpuMemory:
                return p.usedGpuMemory / 2**20
    except Exception:
        return None
    return None
