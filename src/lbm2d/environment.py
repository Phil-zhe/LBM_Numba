from __future__ import annotations

import importlib.metadata as metadata
import os
import platform
import subprocess
import sys
import time
import numpy as np


def inspect_environment(run_smoke=True):
    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "not installed"
    data = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "numba-cuda-mlir": version("numba-cuda-mlir"),
        "cuda-core": version("cuda-core"),
        "cuda-bindings": version("cuda-bindings"),
        "cuda-toolkit": version("cuda-toolkit"),
        "thread_environment": {k: os.environ[k] for k in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                               if k in os.environ},
    }
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
                                "--format=csv,noheader"], capture_output=True, text=True,
                               check=True).stdout.strip()
        data["nvidia_smi"] = query
    except Exception as exc:
        data["nvidia_smi"] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        from threadpoolctl import threadpool_info
        data["threadpools"] = threadpool_info()
    except Exception as exc:
        data["threadpools"] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        from numba_cuda_mlir import cuda
        data["cuda_available"] = bool(cuda.is_available())
        if data["cuda_available"]:
            dev = cuda.get_current_device()
            data["gpu"] = str(dev)
            data["compute_capability"] = str(dev.compute_capability)
        if run_smoke and data["cuda_available"]:
            @cuda.jit
            def add(a, b, out):
                i = cuda.grid(1)
                if i < out.size:
                    out[i] = a[i] + b[i]
            a = np.arange(256, dtype=np.float32)
            da, db = cuda.to_device(a), cuda.to_device(a)
            out = cuda.device_array_like(a)
            start = time.perf_counter()
            add[1, 256](da, db, out)
            cuda.synchronize()
            got = out.copy_to_host()
            data["minimal_kernel"] = {"status": "pass" if np.array_equal(got, a + a) else "fail",
                                      "seconds": time.perf_counter() - start}
    except Exception as exc:
        data["cuda_available"] = False
        data["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return data

