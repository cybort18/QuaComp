import platform
import os
import subprocess
from typing import Any, Dict
import psutil

def get_cpu_name() -> str:
    """
    Get the CPU processor brand name in a platform-agnostic way.
    """
    system = platform.system()
    if system == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(val).strip()
        except Exception:
            pass
    elif system == "Darwin":
        try:
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            return brand
        except Exception:
            pass
    elif system == "Linux":
        try:
            if os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":", 1)[1].strip()
        except Exception:
            pass

    # Fallback to standard platform info
    processor = platform.processor()
    if processor:
        return processor
    return platform.machine()

def get_system_metadata() -> Dict[str, Any]:
    """
    Collect system hardware and environment metadata.

    Returns:
        dict: A dictionary containing:
            - "cpu_name" (str): Brand name of the CPU.
            - "total_ram_bytes" (int): Total system physical RAM in bytes.
            - "total_ram_gb" (float): Total system physical RAM in GB.
            - "os_name" (str): OS name (Windows, Linux, Darwin).
            - "os_release" (str): OS release version.
            - "os_version" (str): OS details version.
            - "python_version" (str): Python interpreter version.
    """
    vm = psutil.virtual_memory()
    from src.profiler.gpu import get_gpu_metadata
    gpu_meta = get_gpu_metadata()

    return {
        "cpu_name": get_cpu_name(),
        "cpu_count_physical": psutil.cpu_count(logical=False) or 1,
        "cpu_count_logical": psutil.cpu_count(logical=True) or 1,
        "gpu_name": gpu_meta.get("gpu_name", "None detected"),
        "has_gpu": gpu_meta.get("has_gpu", False),
        "gpu_count": gpu_meta.get("gpu_count", 1 if gpu_meta.get("has_gpu") else 0),
        "multi_gpu_supported": gpu_meta.get("multi_gpu_supported", False),
        "aer_gpu_supported": gpu_meta.get("aer_gpu_supported", False),
        "total_vram_gb": gpu_meta.get("total_vram_gb", 0.0),
        "total_ram_bytes": vm.total,
        "total_ram_gb": vm.total / (1024 ** 3),
        "os_name": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "python_version": platform.python_version()
    }

def get_cpu_utilization() -> Dict[str, Any]:
    """
    Collect current CPU utilization metrics.

    Returns:
        dict: A dictionary containing:
            - "overall_percent" (float): Overall CPU usage percentage.
            - "per_core_percent" (list): CPU usage percentage per logical core.
            - "core_count" (int): Total number of logical cores.
    """
    # 0.1s interval to get a quick sample without stalling execution
    overall = psutil.cpu_percent(interval=0.1)
    per_core = psutil.cpu_percent(interval=0.1, percpu=True)
    return {
        "overall_percent": overall,
        "per_core_percent": per_core,
        "core_count": len(per_core)
    }


def calculate_effective_memory_bandwidth(
    num_qubits: int,
    total_gates: int,
    execution_time: Any = None,
    sim_method: str = "statevector",
    precision: str = "fp64",
    active_bond_dim: int = 16,
    duration_sec: Any = None
) -> Dict[str, Any]:
    """
    Calculate solver-aware and precision-aware effective memory bandwidth (GB/s)
    and arithmetic intensity (FLOPs/Byte) for quantum circuit simulation.

    Args:
        num_qubits (int): Total simulated qubits.
        total_gates (int): Total quantum gates executed.
        execution_time (float): Simulation execution duration in seconds.
        sim_method (str): Active simulation paradigm ('statevector', 'mps', or 'stabilizer').
        precision (str): Floating-point precision ('fp64' for complex128, 'fp32' for complex64).
        active_bond_dim (int): Active bond dimension chi_act for MPS contractions.
        duration_sec (Optional[float]): Alias for execution_time.

    Returns:
        dict: Memory bandwidth and roofline metrics dictionary.

    Raises:
        TypeError: If argument types are invalid.
        ValueError: If num_qubits, total_gates, or execution_time are negative.
    """
    if execution_time is None and duration_sec is not None:
        execution_time = duration_sec

    if not isinstance(num_qubits, int) or isinstance(num_qubits, bool):
        raise TypeError("num_qubits must be an integer.")
    if not isinstance(total_gates, int) or isinstance(total_gates, bool):
        raise TypeError("total_gates must be an integer.")
    if not isinstance(execution_time, (int, float)) or isinstance(execution_time, bool):
        raise TypeError("execution_time must be a real number.")

    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative.")
    if total_gates < 0:
        raise ValueError("total_gates must be non-negative.")
    if execution_time < 0.0:
        raise ValueError("execution_time must be non-negative.")

    prec_norm = str(precision).lower().strip()
    bytes_per_amplitude = 8 if prec_norm in ("fp32", "single", "complex64") else 16
    method_norm = str(sim_method).lower().strip()
    if method_norm == "matrix_product_state":
        method_norm = "mps"

    safe_time = max(float(execution_time), 1e-9)

    if method_norm == "mps":
        chi = max(1, int(active_bond_dim) if active_bond_dim is not None else 16)
        bytes_per_gate = 2 * 4 * (chi ** 2) * bytes_per_amplitude * 2
        arithmetic_intensity = round(float(2 * chi) / float(bytes_per_amplitude), 4)
    elif method_norm == "stabilizer":
        bytes_per_gate = (2 * num_qubits + 1) * 4
        arithmetic_intensity = 0.75
    else:
        method_norm = "statevector"
        bytes_per_gate = (1 << num_qubits) * bytes_per_amplitude * 2
        arithmetic_intensity = round(14.0 / float(bytes_per_amplitude * 2), 4)

    total_bytes = float(total_gates * bytes_per_gate)
    raw_bw_gbps = total_bytes / (safe_time * 1e9)
    bw_rounded = round(float(raw_bw_gbps), 4)
    if bw_rounded == 0.0 and raw_bw_gbps > 0.0:
        bw_rounded = round(float(raw_bw_gbps), 8)

    return {
        "num_qubits": num_qubits,
        "total_gates": total_gates,
        "execution_time_seconds": safe_time,
        "sim_method": method_norm,
        "precision": "fp32" if bytes_per_amplitude == 8 else "fp64",
        "bytes_per_gate": float(bytes_per_gate),
        "total_bytes_transferred": total_bytes,
        "effective_bandwidth_gbps": bw_rounded,
        "arithmetic_intensity": arithmetic_intensity,
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity
    }
