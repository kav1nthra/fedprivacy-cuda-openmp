"""fedkernels.py -- build + call the serial C / CUDA kernels from Python.

The notebook only needs:  K = Kernels(); K.build(); K.dp_grad(...); K.secagg(...)
Everything degrades gracefully: no gcc -> numpy fallback, no GPU/nvcc -> CUDA skipped.
"""
import ctypes
import os
import platform
import shutil
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
U32 = np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")
I64, I32, F, U = ctypes.c_int64, ctypes.c_int, ctypes.c_float, ctypes.c_uint32
SCALE = 65536.0  # fixed-point scale (2^16) used by secure aggregation


def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except (FileNotFoundError, OSError) as e:
        return False, str(e)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def find_nvcc():
    p = shutil.which("nvcc")
    if p:
        return p
    for c in ("/usr/local/cuda/bin/nvcc", "/usr/local/cuda-12/bin/nvcc", "/usr/lib/nvidia-cuda-toolkit/bin/nvcc"):
        if os.path.exists(c):
            return c
    return None


def system_info():
    """Hardware / software summary for the report."""
    cpu = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    info = {"cpu": cpu, "logical_cores": os.cpu_count(), "os": platform.platform(),
            "python": platform.python_version(), "numpy": np.__version__}
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        info["ram_gb"] = None
    ok, out = _run(["gcc", "--version"])
    info["gcc"] = out.splitlines()[0] if ok else "not found"
    nv = find_nvcc()
    ok, out = _run([nv, "--version"]) if nv else (False, "")
    info["nvcc"] = out.splitlines()[-1] if ok else "not found"
    return info


class Kernels:
    def __init__(self, build_dir=None):
        self.build_dir = build_dir or os.path.join(HERE, "build")
        os.makedirs(self.build_dir, exist_ok=True)
        self.cpu = None
        self.cuda = None
        self.gpu_name = None
        self.gpu_mem_gb = None
        self.log = []

    # ---------------------------------------------------------------- build
    def build(self, verbose=True):
        self._build_cpu()
        self._build_cuda()
        if verbose:
            print("\n".join(self.log))
            print(f"\nSerial C kernels : {'READY' if self.cpu else 'not available (numpy fallback)'}")
            print(f"CUDA kernels     : {('READY on ' + self.gpu_name) if self.cuda else 'not available (CPU-only run)'}")
        return self

    def _build_cpu(self):
        so = os.path.join(self.build_dir, "libfl_cpu.so")
        src = os.path.join(HERE, "fl_cpu.c")
        if not shutil.which("gcc"):
            self.log.append("[cpu] gcc not found")
            return
        for flags in (["-O3", "-march=native", "-ffast-math"], ["-O3", "-ffast-math"]):
            ok, out = _run(["gcc", *flags, "-shared", "-fPIC", "-o", so, src, "-lm"])
            if ok:
                self.log.append(f"[cpu] compiled with: gcc {' '.join(flags)}")
                break
            self.log.append(f"[cpu] build failed with {' '.join(flags)}: {out[:200]}")
        else:
            return
        lib = ctypes.CDLL(so)
        lib.dp_grad_serial.argtypes = [F32, F32, F32, F32, I64, I64, F, F32, F32]
        lib.dp_grad_serial.restype = None
        lib.secagg_serial.argtypes = [F32, I32, I64, U, F, F32]
        lib.secagg_serial.restype = None
        lib.secagg_share.argtypes = [F32, I32, I64, U, F, I32, U32]
        lib.secagg_share.restype = None
        self.cpu = lib

    def _build_cuda(self):
        nvcc = find_nvcc()
        if not nvcc:
            self.log.append("[cuda] nvcc not found -> CUDA skipped (on Colab: Runtime > Change runtime type > GPU)")
            return
        so = os.path.join(self.build_dir, "libfl_cuda.so")
        src = os.path.join(HERE, "fl_cuda.cu")
        for arch in (["-arch=native"], []):
            ok, out = _run([nvcc, "-O3", *arch, "-shared", "-Xcompiler", "-fPIC", "-o", so, src])
            if ok:
                self.log.append(f"[cuda] compiled with: nvcc -O3 {' '.join(arch)}")
                break
            self.log.append(f"[cuda] build failed ({' '.join(arch) or 'default arch'}): {out[:300]}")
        else:
            return
        try:
            lib = ctypes.CDLL(so)
            lib.gpu_info.argtypes = [ctypes.c_char_p, I32, ctypes.POINTER(ctypes.c_double)]
            lib.gpu_info.restype = I32
            name = ctypes.create_string_buffer(256)
            mem = ctypes.c_double()
            if lib.gpu_info(name, 256, ctypes.byref(mem)) < 1:
                self.log.append("[cuda] compiled but no usable GPU found at runtime")
                return
            self.gpu_name, self.gpu_mem_gb = name.value.decode(), mem.value
            lib.gpu_upload.argtypes = [F32, F32, F32, I64, I64]
            lib.gpu_upload.restype = I32
            lib.gpu_dp_grad.argtypes = [F32, F, I32, F32, ctypes.POINTER(ctypes.c_float)]
            lib.gpu_dp_grad.restype = I32
            lib.gpu_secagg.argtypes = [F32, I32, I64, U, F, F32,
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
            lib.gpu_secagg.restype = I32
            lib.gpu_free.restype = None
            self.cuda = lib
        except OSError as e:
            self.log.append(f"[cuda] could not load library: {e}")

    # ------------------------------------------------------------ utilities
    @property
    def backends_dp(self):
        b = []
        if self.cpu:
            b += ["serial"]
        if self.cuda:
            b += ["cuda_naive", "cuda_opt"]
        return b

    @property
    def backends_secagg(self):
        b = []
        if self.cpu:
            b += ["serial"]
        if self.cuda:
            b += ["cuda"]
        return b

    # --------------------------------------------------------- DP gradient
    @staticmethod
    def prep(X, y):
        """Return contiguous float32 arrays plus row norms."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        xn = np.ascontiguousarray(np.linalg.norm(X, axis=1), dtype=np.float32)
        return X, y, xn

    def upload(self, X, y, xn):
        """Copy data to the GPU (one-time cost). Returns seconds."""
        t = time.perf_counter()
        if self.cuda.gpu_upload(X, y, xn, X.shape[0], X.shape[1]) != 0:
            raise RuntimeError("gpu_upload failed (out of GPU memory?)")
        return time.perf_counter() - t

    def dp_grad(self, backend, X, y, xn, w, C):
        """Sum of per-sample clipped logistic-regression gradients.
        Returns (g, kernel_seconds). For CUDA, call upload() first (X,y,xn only used for shapes)."""
        w = np.ascontiguousarray(w, dtype=np.float32)
        N, D = X.shape
        g = np.empty(D, dtype=np.float32)
        if backend == "serial":
            s = np.empty(N, dtype=np.float32)
            t = time.perf_counter()
            self.cpu.dp_grad_serial(X, y, w, xn, N, D, C, g, s)
            return g, time.perf_counter() - t
        if backend in ("cuda_naive", "cuda_opt"):
            ms = ctypes.c_float()
            t = time.perf_counter()
            rc = self.cuda.gpu_dp_grad(w, C, 0 if backend == "cuda_naive" else 1, g, ctypes.byref(ms))
            wall = time.perf_counter() - t
            if rc != 0:
                raise RuntimeError("gpu_dp_grad failed")
            self.last_kernel_s = ms.value / 1e3          # kernel only
            return g, wall                                 # wall = kernel + w/g copies + launch
        if backend == "numpy":
            return dp_grad_numpy(X, y, w, C).astype(np.float32), 0.0
        raise ValueError(backend)

    # ------------------------------------------------- secure aggregation
    def secagg(self, backend, U_mat, seed=12345):
        """Sum of the K rows of U_mat computed through additive masking mod 2^32.
        Returns (sum, seconds, kernel_seconds). kernel_seconds differs from seconds only on CUDA."""
        K, D = U_mat.shape
        out = np.empty(D, dtype=np.float32)
        if backend == "serial":
            t = time.perf_counter()
            self.cpu.secagg_serial(U_mat, K, D, seed, SCALE, out)
            dt = time.perf_counter() - t
            return out, dt, dt
        if backend == "cuda":
            kms, tms = ctypes.c_float(), ctypes.c_float()
            t = time.perf_counter()
            rc = self.cuda.gpu_secagg(U_mat, K, D, seed, SCALE, out, ctypes.byref(kms), ctypes.byref(tms))
            wall = time.perf_counter() - t
            if rc != 0:
                raise RuntimeError("gpu_secagg failed")
            return out, wall, kms.value / 1e3
        raise ValueError(backend)

    def secagg_share(self, U_mat, client, seed=12345):
        K, D = U_mat.shape
        share = np.empty(D, dtype=np.uint32)
        self.cpu.secagg_share(U_mat, K, D, seed, SCALE, client, share)
        return share


# ------------------------------------------------------------ numpy reference
def dp_grad_numpy(X, y, w, C):
    """float64 reference implementation (used for correctness checks / no-compiler fallback)."""
    X = X.astype(np.float64)
    z = X @ w.astype(np.float64)
    r = 1.0 / (1.0 + np.exp(-z)) - y
    gn = np.abs(r) * np.linalg.norm(X, axis=1)
    sc = np.where(gn > C, C / np.maximum(gn, 1e-30), 1.0)
    return (r * sc) @ X


def secagg_reference(U_mat):
    return U_mat.astype(np.float64).sum(axis=0)


def dp_epsilon_zcdp(sigma, steps, delta=1e-5):
    """(epsilon, delta)-DP bound for `steps` full-batch Gaussian-mechanism releases with noise
    multiplier `sigma` (noise std = sigma*C, sensitivity C for add/remove-one-record).
    Uses zCDP composition (Bun & Steinke 2016): rho = steps/(2 sigma^2),
    eps = rho + 2*sqrt(rho*ln(1/delta)).  Returns inf for sigma == 0."""
    if sigma <= 0:
        return float("inf")
    rho = steps / (2.0 * sigma ** 2)
    return rho + 2.0 * np.sqrt(rho * np.log(1.0 / delta))
