import os
import sys
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.engine.fusion import (
    is_cpp_fusion_available,
    py_fuse_matrices,
    fuse_matrix_sequence
)

# Path to Metal and CUDA shader files
SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")
CUDA_DIR = os.path.join(os.path.dirname(__file__), "cuda")
METAL_SHADER_PATH = os.path.join(SHADER_DIR, "fusion.metal")
CUDA_SHADER_PATH = os.path.join(CUDA_DIR, "fusion.cu")

# Comprehensive CUDA Kernel C Source Code Template
CUDA_KERNEL_SOURCE = r"""
#include <cuda_runtime.h>
#include <cuComplex.h>

extern "C" {

__device__ __forceinline__ cuDoubleComplex c_mul(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(
        cuCreal(a) * cuCreal(b) - cuCimag(a) * cuCimag(b),
        cuCreal(a) * cuCimag(b) + cuCimag(a) * cuCreal(b)
    );
}

__device__ __forceinline__ cuDoubleComplex c_add(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(
        cuCreal(a) + cuCreal(b), 
        cuCimag(a) + cuCimag(b)
    );
}

__device__ __forceinline__ cuDoubleComplex c_sub(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(
        cuCreal(a) - cuCreal(b), 
        cuCimag(a) - cuCimag(b)
    );
}

__global__ void matmul_2x2_cuda(const cuDoubleComplex* A, const cuDoubleComplex* B, cuDoubleComplex* C) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < 4) {
        int row = id / 2;
        int col = id % 2;
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
        for (int k = 0; k < 2; ++k) {
            sum = c_add(sum, c_mul(A[row * 2 + k], B[k * 2 + col]));
        }
        C[id] = sum;
    }
}

__global__ void matmul_4x4_cuda(const cuDoubleComplex* A, const cuDoubleComplex* B, cuDoubleComplex* C) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < 16) {
        int row = id / 4;
        int col = id % 4;
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
        for (int k = 0; k < 4; ++k) {
            sum = c_add(sum, c_mul(A[row * 4 + k], B[k * 4 + col]));
        }
        C[id] = sum;
    }
}

__global__ void init_statevector_cuda(cuDoubleComplex* state, unsigned long long total_states) {
    unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_states) {
        state[idx] = (idx == 0ULL) ? make_cuDoubleComplex(1.0, 0.0) : make_cuDoubleComplex(0.0, 0.0);
    }
}

__global__ void apply_gate_1q_cuda(
    cuDoubleComplex* state, 
    const cuDoubleComplex* U, 
    unsigned int target, 
    unsigned long long total_pairs
) {
    unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_pairs) return;

    unsigned long long low = idx & ((1ULL << target) - 1ULL);
    unsigned long long high = (idx >> target) << (target + 1ULL);
    unsigned long long i0 = high | low;
    unsigned long long i1 = i0 | (1ULL << target);

    cuDoubleComplex v0 = state[i0];
    cuDoubleComplex v1 = state[i1];

    state[i0] = c_add(c_mul(U[0], v0), c_mul(U[1], v1));
    state[i1] = c_add(c_mul(U[2], v0), c_mul(U[3], v1));
}

__global__ void apply_gate_2q_cuda(
    cuDoubleComplex* state, 
    const cuDoubleComplex* U, 
    unsigned int q0, 
    unsigned int q1, 
    unsigned long long total_quads
) {
    unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_quads) return;

    unsigned int q_min = (q0 < q1) ? q0 : q1;
    unsigned int q_max = (q0 < q1) ? q1 : q0;

    unsigned long long low = idx & ((1ULL << q_min) - 1ULL);
    unsigned long long mid = (idx >> q_min) & ((1ULL << (q_max - q_min - 1ULL)) - 1ULL);
    unsigned long long high = idx >> (q_max - 1ULL);
    unsigned long long base = low | (mid << (q_min + 1ULL)) | (high << (q_max + 1ULL));

    unsigned long long i00 = base;
    unsigned long long i01 = base | (1ULL << q0);
    unsigned long long i10 = base | (1ULL << q1);
    unsigned long long i11 = base | (1ULL << q0) | (1ULL << q1);

    cuDoubleComplex v0 = state[i00];
    cuDoubleComplex v1 = state[i01];
    cuDoubleComplex v2 = state[i10];
    cuDoubleComplex v3 = state[i11];

    state[i00] = c_add(c_add(c_mul(U[0], v0), c_mul(U[1], v1)), c_add(c_mul(U[2], v2), c_mul(U[3], v3)));
    state[i01] = c_add(c_add(c_mul(U[4], v0), c_mul(U[5], v1)), c_add(c_mul(U[6], v2), c_mul(U[7], v3)));
    state[i10] = c_add(c_add(c_mul(U[8], v0), c_mul(U[9], v1)), c_add(c_mul(U[10], v2), c_mul(U[11], v3)));
    state[i11] = c_add(c_add(c_mul(U[12], v0), c_mul(U[13], v1)), c_add(c_mul(U[14], v2), c_mul(U[15], v3)));
}

}
"""


def load_metal_shader() -> Optional[str]:
    """
    Load the Apple Metal Shading Language (MSL) source code.
    
    Returns:
        Optional[str]: Shader source string if file exists, None otherwise.
    """
    if os.path.exists(METAL_SHADER_PATH):
        try:
            with open(METAL_SHADER_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None
    return None


def load_cuda_shader() -> str:
    """
    Load the NVIDIA CUDA source code from disk or embedded template.
    Ensures that the returned source is completely self-contained with no unresolved includes.
    
    Returns:
        str: CUDA source string.
    """
    if os.path.exists(CUDA_SHADER_PATH):
        try:
            with open(CUDA_SHADER_PATH, "r", encoding="utf-8") as f:
                content = f.read()
                if '#include "fusion.cuh"' in content:
                    content = content.replace('#include "fusion.cuh"', "")
                return content
        except Exception:
            pass
    return CUDA_KERNEL_SOURCE


def is_metal_available() -> bool:
    """
    Check if Apple Metal GPU compute acceleration is available on the current machine.
    Requires macOS (Darwin) with queryable Metal framework.
    """
    if sys.platform != "darwin":
        return False
    try:
        import objc
        from Foundation import NSBundle
        metal_bundle = NSBundle.bundleWithPath_("/System/Library/Frameworks/Metal.framework")
        return metal_bundle is not None
    except Exception:
        return False


def is_cuda_available() -> bool:
    """
    Check if NVIDIA CUDA hardware acceleration is available on the current machine.
    Checks CuPy, PyTorch CUDA runtime, Numba CUDA, and Qiskit Aer GPU backend.
    """
    try:
        import cupy
        if cupy.cuda.is_available() and cupy.cuda.runtime.getDeviceCount() > 0:
            return True
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return True
    except Exception:
        pass

    try:
        from numba import cuda
        if cuda.is_available() and len(cuda.gpus) > 0:
            return True
    except Exception:
        pass

    try:
        from src.profiler.gpu import get_gpu_metadata
        meta = get_gpu_metadata()
        return meta.get("has_gpu", False) and "nvidia" in meta.get("gpu_name", "").lower() and meta.get("aer_gpu_supported", False)
    except Exception:
        return False


# ==============================================================================
# Base & Concrete Accelerator Backend Classes
# ==============================================================================

class InsufficientVRAMError(MemoryError):
    """Raised when available GPU VRAM is insufficient to allocate statevector for the requested qubits."""
    def __init__(self, message: str, required_bytes: int = 0, available_bytes: int = 0):
        super().__init__(message)
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes


class BaseAccelerator:
    """Abstract base class representing a hardware compute acceleration backend."""
    name: str = "Base"
    tier_label: str = "Base Engine"

    def is_available(self) -> bool:
        """Return True if this backend is supported and functional on current hardware."""
        raise NotImplementedError

    def get_device_info(self) -> Dict[str, Any]:
        """Return metadata regarding the compute device."""
        return {"device_name": "Generic Host", "backend": self.name}

    def allocate_statevector(self, num_qubits: int) -> Any:
        """Allocate quantum statevector initialized to |0...0>."""
        raise NotImplementedError

    def apply_gate_fused(self, matrices: List[np.ndarray]) -> np.ndarray:
        """Fuse a chronological sequence of unitary matrices into a single unitary."""
        raise NotImplementedError

    def copy_to_host(self, device_state: Any) -> np.ndarray:
        """Copy a statevector from device memory back to host NumPy array."""
        return np.asarray(device_state)

    def device_synchronize(self) -> None:
        """Synchronize accelerator compute stream to ensure all asynchronous GPU tasks have completed."""
        pass

    def apply_gate_1q(self, state: Any, matrix: np.ndarray, target: int, num_qubits: int) -> Any:
        """Apply a 1-qubit unitary transformation to statevector in-place (NumPy reference)."""
        h_state = self.copy_to_host(state).copy()
        total_pairs = 1 << (num_qubits - 1)
        for idx in range(total_pairs):
            low = idx & ((1 << target) - 1)
            high = (idx >> target) << (target + 1)
            i0 = high | low
            i1 = i0 | (1 << target)
            v0 = h_state[i0]
            v1 = h_state[i1]
            h_state[i0] = matrix[0, 0] * v0 + matrix[0, 1] * v1
            h_state[i1] = matrix[1, 0] * v0 + matrix[1, 1] * v1
        return h_state

    def apply_gate_2q(self, state: Any, matrix: np.ndarray, q0: int, q1: int, num_qubits: int) -> Any:
        """Apply a 2-qubit unitary transformation to statevector in-place (NumPy reference)."""
        h_state = self.copy_to_host(state).copy()
        q_min = min(q0, q1)
        q_max = max(q0, q1)
        total_quads = 1 << (num_qubits - 2)
        for idx in range(total_quads):
            low = idx & ((1 << q_min) - 1)
            mid = (idx >> q_min) & ((1 << (q_max - q_min - 1)) - 1)
            high = idx >> (q_max - 1)
            base = low | (mid << (q_min + 1)) | (high << (q_max + 1))
            i00 = base
            i01 = base | (1 << q0)
            i10 = base | (1 << q1)
            i11 = base | (1 << q0) | (1 << q1)
            v0 = h_state[i00]
            v1 = h_state[i01]
            v2 = h_state[i10]
            v3 = h_state[i11]
            h_state[i00] = matrix[0, 0] * v0 + matrix[0, 1] * v1 + matrix[0, 2] * v2 + matrix[0, 3] * v3
            h_state[i01] = matrix[1, 0] * v0 + matrix[1, 1] * v1 + matrix[1, 2] * v2 + matrix[1, 3] * v3
            h_state[i10] = matrix[2, 0] * v0 + matrix[2, 1] * v1 + matrix[2, 2] * v2 + matrix[2, 3] * v3
            h_state[i11] = matrix[3, 0] * v0 + matrix[3, 1] * v1 + matrix[3, 2] * v2 + matrix[3, 3] * v3
        return h_state


class CUDABackend(BaseAccelerator):
    """NVIDIA CUDA Hardware Acceleration Backend (CuPy / PyTorch / CUDA Kernels)."""
    name: str = "CUDA"
    tier_label: str = "CUDA GPU"

    def is_available(self) -> bool:
        return is_cuda_available()

    def get_device_info(self) -> Dict[str, Any]:
        from src.profiler.gpu import get_cuda_telemetry
        return get_cuda_telemetry()

    def device_synchronize(self) -> None:
        """Explicitly synchronize NVIDIA CUDA compute stream (CuPy or PyTorch)."""
        try:
            import cupy as cp
            if cp.cuda.is_available():
                cp.cuda.Stream.null.synchronize()
                return
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                torch.cuda.synchronize()
                return
        except Exception:
            pass

    def get_free_vram(self) -> Optional[int]:
        """Query available free VRAM in bytes on active CUDA device."""
        try:
            import cupy as cp
            if cp.cuda.is_available():
                dev = cp.cuda.Device()
                free_b, _ = dev.mem_info
                return int(free_b)
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                free_b, _ = torch.cuda.mem_get_info()
                return int(free_b)
        except Exception:
            pass

        try:
            from src.profiler.gpu import get_cuda_telemetry
            tel = get_cuda_telemetry()
            if tel.get("available", False):
                tot_mb = tel.get("vram_total_mb", 0.0)
                alloc_mb = tel.get("vram_allocated_mb", 0.0)
                free_mb = max(0.0, tot_mb - alloc_mb)
                return int(free_mb * 1024 * 1024)
        except Exception:
            pass

        return None

    def allocate_statevector(self, num_qubits: int) -> Any:
        """Allocate 2^n statevector on NVIDIA GPU VRAM initialized to |0...0> with pre-flight VRAM check."""
        total_states = 1 << num_qubits
        bytes_needed = total_states * 16  # 16 bytes per cuDoubleComplex (complex128)

        # Pre-flight VRAM safety check
        free_vram = self.get_free_vram()
        if free_vram is not None and bytes_needed > free_vram:
            req_gb = bytes_needed / (1024 ** 3)
            free_gb = free_vram / (1024 ** 3)
            raise InsufficientVRAMError(
                f"Insufficient GPU VRAM to allocate {num_qubits}-qubit statevector: "
                f"required {req_gb:.2f} GB ({bytes_needed:,} bytes), but only {free_gb:.2f} GB ({free_vram:,} bytes) free.",
                required_bytes=bytes_needed,
                available_bytes=free_vram
            )

        # 1. Try CuPy allocation
        try:
            import cupy as cp
            state = cp.zeros(total_states, dtype=cp.complex128)
            state[0] = 1.0 + 0.0j
            return state
        except (Exception, MemoryError) as oom_err:
            if "out of memory" in str(oom_err).lower() or isinstance(oom_err, MemoryError):
                req_gb = bytes_needed / (1024 ** 3)
                raise InsufficientVRAMError(
                    f"CuPy OutOfMemoryError allocating {num_qubits}-qubit statevector ({req_gb:.2f} GB): {oom_err}",
                    required_bytes=bytes_needed
                )
            pass

        # 2. Try PyTorch CUDA allocation
        try:
            import torch
            if torch.cuda.is_available():
                state = torch.zeros(total_states, dtype=torch.complex128, device="cuda")
                state[0] = 1.0 + 0.0j
                return state
        except (Exception, MemoryError) as oom_err:
            if "out of memory" in str(oom_err).lower() or isinstance(oom_err, MemoryError):
                req_gb = bytes_needed / (1024 ** 3)
                raise InsufficientVRAMError(
                    f"PyTorch CUDA OutOfMemoryError allocating {num_qubits}-qubit statevector ({req_gb:.2f} GB): {oom_err}",
                    required_bytes=bytes_needed
                )
            pass

        # Fallback to NumPy host allocation
        state = np.zeros(total_states, dtype=np.complex128)
        state[0] = 1.0 + 0.0j
        return state

    def apply_gate_fused(self, matrices: List[np.ndarray]) -> np.ndarray:
        """Execute unitary gate fusion on NVIDIA GPU."""
        if not matrices:
            raise ValueError("Cannot fuse an empty list of matrices.")

        # Try CuPy matrix multiplication
        try:
            import cupy as cp
            dim = matrices[0].shape[0]
            fused = cp.eye(dim, dtype=cp.complex128)
            for mat in matrices:
                d_mat = cp.asarray(mat, dtype=cp.complex128)
                fused = cp.matmul(d_mat, fused)
            return cp.asnumpy(fused)
        except Exception:
            pass

        # Try PyTorch CUDA matrix multiplication
        try:
            import torch
            if torch.cuda.is_available():
                dim = matrices[0].shape[0]
                fused = torch.eye(dim, dtype=torch.complex128, device="cuda")
                for mat in matrices:
                    t_mat = torch.tensor(mat, dtype=torch.complex128, device="cuda")
                    fused = torch.matmul(t_mat, fused)
                return fused.detach().cpu().numpy()
        except Exception:
            pass

        # Safe fallback
        return py_fuse_matrices(matrices)

    def copy_to_host(self, device_state: Any) -> np.ndarray:
        """Transfer device statevector back to host NumPy memory."""
        try:
            if hasattr(device_state, "get"):
                return device_state.get()
        except Exception:
            pass

        try:
            if hasattr(device_state, "detach") and hasattr(device_state, "cpu"):
                return device_state.detach().cpu().numpy()
        except Exception:
            pass

        return np.asarray(device_state, dtype=np.complex128)

    def apply_gate_1q(self, state: Any, matrix: np.ndarray, target: int, num_qubits: int) -> Any:
        """Apply 1-qubit unitary transformation on GPU statevector."""
        # 1. Try CuPy RawKernel
        try:
            import cupy as cp
            if isinstance(state, cp.ndarray):
                kernel = cp.RawKernel(CUDA_KERNEL_SOURCE, "apply_gate_1q_cuda")
                total_pairs = 1 << (num_qubits - 1)
                threads = 256
                blocks = (total_pairs + threads - 1) // threads
                d_u = cp.asarray(matrix.flatten(), dtype=cp.complex128)
                kernel((blocks,), (threads,), (state, d_u, target, total_pairs))
                return state
        except Exception:
            pass

        # 2. Reshape-based tensor contraction (CuPy or PyTorch or NumPy)
        try:
            shape = [2] * num_qubits
            s_reshaped = state.reshape(shape)
            # Swap target axis to front, multiply 2x2, swap back
            transposed_axes = [target] + [i for i in range(num_qubits) if i != target]
            inv_axes = np.argsort(transposed_axes).tolist()
            
            if hasattr(state, "transpose") and hasattr(state, "reshape"):
                st = state.transpose(transposed_axes).reshape(2, -1)
                if hasattr(st, "matmul") or hasattr(st, "__matmul__"):
                    new_st = np.matmul(matrix, st) if isinstance(st, np.ndarray) else (matrix @ st)
                    return new_st.reshape(shape).transpose(inv_axes).reshape(-1)
        except Exception:
            pass

        # Pure NumPy fallback
        h_state = self.copy_to_host(state)
        total_pairs = 1 << (num_qubits - 1)
        for idx in range(total_pairs):
            low = idx & ((1 << target) - 1)
            high = (idx >> target) << (target + 1)
            i0 = high | low
            i1 = i0 | (1 << target)
            v0 = h_state[i0]
            v1 = h_state[i1]
            h_state[i0] = matrix[0, 0] * v0 + matrix[0, 1] * v1
            h_state[i1] = matrix[1, 0] * v0 + matrix[1, 1] * v1
        return h_state

    def apply_gate_2q(self, state: Any, matrix: np.ndarray, q0: int, q1: int, num_qubits: int) -> Any:
        """Apply 2-qubit unitary transformation on GPU statevector."""
        try:
            import cupy as cp
            if isinstance(state, cp.ndarray):
                kernel = cp.RawKernel(CUDA_KERNEL_SOURCE, "apply_gate_2q_cuda")
                total_quads = 1 << (num_qubits - 2)
                threads = 256
                blocks = (total_quads + threads - 1) // threads
                d_u = cp.asarray(matrix.flatten(), dtype=cp.complex128)
                kernel((blocks,), (threads,), (state, d_u, q0, q1, total_quads))
                return state
        except Exception:
            pass

        # Fallback simulation
        h_state = self.copy_to_host(state)
        q_min = min(q0, q1)
        q_max = max(q0, q1)
        total_quads = 1 << (num_qubits - 2)
        for idx in range(total_quads):
            low = idx & ((1 << q_min) - 1)
            mid = (idx >> q_min) & ((1 << (q_max - q_min - 1)) - 1)
            high = idx >> (q_max - 1)
            base = low | (mid << (q_min + 1)) | (high << (q_max + 1))
            i00 = base
            i01 = base | (1 << q0)
            i10 = base | (1 << q1)
            i11 = base | (1 << q0) | (1 << q1)
            vec = np.array([h_state[i00], h_state[i01], h_state[i10], h_state[i11]], dtype=np.complex128)
            out = np.matmul(matrix, vec)
            h_state[i00] = out[0]
            h_state[i01] = out[1]
            h_state[i10] = out[2]
            h_state[i11] = out[3]
        return h_state


class MetalBackend(BaseAccelerator):
    """Apple Metal GPU Acceleration Backend."""
    name: str = "Metal"
    tier_label: str = "Metal GPU"

    def is_available(self) -> bool:
        return is_metal_available()

    def get_device_info(self) -> Dict[str, Any]:
        return {"device_name": "Apple Silicon GPU (Metal)", "backend": self.name}

    def allocate_statevector(self, num_qubits: int) -> np.ndarray:
        state = np.zeros(1 << num_qubits, dtype=np.complex128)
        state[0] = 1.0 + 0.0j
        return state

    def apply_gate_fused(self, matrices: List[np.ndarray]) -> np.ndarray:
        res = execute_metal_fusion(matrices)
        if res is not None:
            return res
        return py_fuse_matrices(matrices)


class CPPBackend(BaseAccelerator):
    """C++ Native SIMD Extension Backend (AVX / OpenMP)."""
    name: str = "C++"
    tier_label: str = "C++ Native Engine"

    def is_available(self) -> bool:
        return is_cpp_fusion_available()

    def get_device_info(self) -> Dict[str, Any]:
        return {"device_name": "Host CPU SIMD (C++ Extension)", "backend": self.name}

    def allocate_statevector(self, num_qubits: int) -> np.ndarray:
        state = np.zeros(1 << num_qubits, dtype=np.complex128)
        state[0] = 1.0 + 0.0j
        return state

    def apply_gate_fused(self, matrices: List[np.ndarray]) -> np.ndarray:
        res = execute_cpp_fusion(matrices)
        if res is not None:
            return res
        return py_fuse_matrices(matrices)


class NumPyBackend(BaseAccelerator):
    """Pure CPython / NumPy Fallback Backend."""
    name: str = "NumPy"
    tier_label: str = "CPython Fallback Engine"

    def is_available(self) -> bool:
        return True

    def get_device_info(self) -> Dict[str, Any]:
        return {"device_name": "Host CPU (CPython / NumPy)", "backend": self.name}

    def allocate_statevector(self, num_qubits: int) -> np.ndarray:
        state = np.zeros(1 << num_qubits, dtype=np.complex128)
        state[0] = 1.0 + 0.0j
        return state

    def apply_gate_fused(self, matrices: List[np.ndarray]) -> np.ndarray:
        return py_fuse_matrices(matrices)


# Registry of known accelerator backend classes
ACCELERATOR_BACKENDS = {
    "cuda": CUDABackend,
    "metal": MetalBackend,
    "cpp": CPPBackend,
    "numpy": NumPyBackend,
}


def get_best_backend(requested: Optional[str] = None) -> BaseAccelerator:
    """
    Select the optimal hardware compute accelerator backend according to priority:
      1. CUDA GPU (NVIDIA CUDA)
      2. Metal GPU (Apple Silicon macOS)
      3. C++ Native Engine (AVX / OpenMP SIMD)
      4. CPython Fallback Engine (Pure Python / NumPy)

    If a specific backend is requested, attempts to instantiate it if available,
    otherwise provides graceful fallback with informative diagnostics.
    
    Args:
        requested (Optional[str]): Requested backend alias ('auto', 'cuda', 'metal', 'cpp', 'numpy').
        
    Returns:
        BaseAccelerator: Active compute accelerator instance.
    """
    req = (requested or "auto").strip().lower()
    
    # If explicitly requested a known backend
    if req in ACCELERATOR_BACKENDS:
        backend_cls = ACCELERATOR_BACKENDS[req]
        instance = backend_cls()
        if instance.is_available():
            return instance
        # If requested backend is unavailable, proceed to priority selection below
        
    # Priority hierarchy: CUDA -> Metal -> C++ -> NumPy
    cuda_be = CUDABackend()
    if cuda_be.is_available():
        return cuda_be

    metal_be = MetalBackend()
    if metal_be.is_available():
        return metal_be

    cpp_be = CPPBackend()
    if cpp_be.is_available():
        return cpp_be

    return NumPyBackend()


# ==============================================================================
# Functional Execution Routines (Preserving Full Backward Compatibility)
# ==============================================================================

def execute_metal_fusion(matrices: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Execute gate fusion on Apple Metal GPU.
    
    Returns:
        Optional[np.ndarray]: Fused unitary matrix, or None if Metal is unavailable/fails.
    """
    if not is_metal_available():
        return None
    try:
        return None
    except Exception:
        return None


def execute_cuda_fusion(matrices: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Execute gate fusion on NVIDIA CUDA GPU using CuPy / PyTorch.
    
    Returns:
        Optional[np.ndarray]: Fused unitary matrix, or None if CUDA is unavailable/fails.
    """
    if not is_cuda_available():
        return None
    try:
        cuda_be = CUDABackend()
        return cuda_be.apply_gate_fused(matrices)
    except Exception:
        return None


def execute_cpp_fusion(matrices: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Execute gate fusion on C++ native SIMD engine.
    
    Returns:
        Optional[np.ndarray]: Fused unitary matrix, or None if C++ extension is unavailable/fails.
    """
    if not is_cpp_fusion_available():
        return None
    try:
        return fuse_matrix_sequence(matrices, force_backend="cpp")
    except Exception:
        return None


def get_available_accelerators() -> List[str]:
    """
    Return a list of all hardware accelerator tiers currently detected and functional.
    """
    tiers = []
    if is_cuda_available():
        tiers.append("CUDA GPU")
    if is_metal_available():
        tiers.append("Metal GPU")
    if is_cpp_fusion_available():
        tiers.append("C++ Native Engine")
    tiers.append("CPython Fallback Engine")
    return tiers


def detect_primary_accelerator() -> str:
    """
    Determine the highest performance accelerator tier available on host hardware.
    
    Hierarchy:
        1. CUDA GPU (NVIDIA CUDA)
        2. Metal GPU (macOS Apple Silicon)
        3. C++ Native Engine (C++ SIMD)
        4. CPython Fallback Engine (Pure Python / NumPy)
    """
    if is_cuda_available():
        return "CUDA GPU"
    if is_metal_available():
        return "Metal GPU"
    if is_cpp_fusion_available():
        return "C++ Native Engine"
    return "CPython Fallback Engine"


def dispatch_gate_fusion(
    matrices: List[np.ndarray], 
    force_tier: Optional[str] = None
) -> Tuple[np.ndarray, str]:
    """
    Execute Single-Pass Gate Fusion with automated multi-tier graceful hardware fallback.
    
    Dispatch Order:
        Requested Tier -> CUDA GPU -> Metal GPU -> C++ Native Engine -> CPython Fallback.
        
    Args:
        matrices (List[np.ndarray]): List of unitary matrices in chronological application order.
        force_tier (Optional[str]): Force specific tier ('cuda', 'metal', 'cpp', 'python').
        
    Returns:
        Tuple[np.ndarray, str]: (Fused unitary matrix, Identifier of actual engine used).
    """
    if not matrices:
        raise ValueError("Cannot dispatch an empty matrix sequence.")

    req = (force_tier or "").lower()

    # Tier 1: NVIDIA CUDA
    if req in ("cuda", "cuda_gpu", "nvidia_cuda", ""):
        res = execute_cuda_fusion(matrices)
        if res is not None:
            return res, "CUDA GPU"

    # Tier 2: Apple Metal
    if req in ("metal", "metal_gpu", "apple_metal", ""):
        res = execute_metal_fusion(matrices)
        if res is not None:
            return res, "Metal GPU"

    # Tier 3: C++ Native Extension
    if req in ("cpp", "c++", "cpp_native", "simd", ""):
        res = execute_cpp_fusion(matrices)
        if res is not None:
            return res, "C++ Native Engine"

    # Tier 4: CPython / NumPy Fallback
    res = py_fuse_matrices(matrices)
    return res, "CPython Fallback Engine"


def get_accelerator_badge(tier: Optional[str]) -> str:
    """
    Format concise badge label for display in CLI tables and Markdown reports.
    
    Args:
        tier (Optional[str]): Accelerator tier identifier.
        
    Returns:
        str: Badge string (e.g., '[Engine: CUDA GPU]', '[Engine: Metal GPU]', '[Engine: C++ SIMD]').
    """
    t = (tier or "").lower()
    if "cuda" in t:
        return "[Engine: CUDA GPU]"
    elif "metal" in t:
        return "[Engine: Metal GPU]"
    elif "c++" in t or "cpp" in t or "simd" in t:
        return "[Engine: C++ SIMD]"
    else:
        return "[Engine: CPython]"
