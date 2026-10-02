import os
import platform
import subprocess
from typing import Dict, Any, List, Tuple

def get_available_aer_devices() -> List[str]:
    """
    Query available compute simulation devices supported by the installed Qiskit Aer backend.
    
    Returns:
        List of device name strings, e.g. ['CPU'] or ['CPU', 'GPU'].
    """
    try:
        from qiskit_aer import AerSimulator
        devices = AerSimulator().available_devices()
        if isinstance(devices, (list, tuple)):
            return [str(d).upper() for d in devices]
        return ["CPU"]
    except Exception:
        return ["CPU"]

def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is supported and available in the current Qiskit Aer runtime.
    
    Returns:
        bool: True if 'GPU' is listed in AerSimulator available devices, False otherwise.
    """
    return "GPU" in get_available_aer_devices()

def get_gpu_metadata() -> Dict[str, Any]:
    """
    Probe the local hardware to identify all available GPUs (NVIDIA, AMD, Apple, Intel),
    multi-GPU inventory, and aggregate VRAM capacity.
    
    Returns:
        Dictionary containing GPU metadata:
            - 'gpu_name' (str): Primary brand and model of detected GPU.
            - 'has_gpu' (bool): Whether at least one GPU is physically present.
            - 'gpu_count' (int): Total number of detected GPU cards.
            - 'gpus' (list): List of detailed GPU device descriptors.
            - 'aer_gpu_supported' (bool): Whether Qiskit Aer has GPU runtime enabled.
            - 'total_vram_gb' (float): Aggregate VRAM across all detected GPUs in GB.
            - 'multi_gpu_supported' (bool): True if more than 1 GPU is present.
            - 'backend_devices' (list): Available Qiskit Aer devices.
    """
    aer_devices = get_available_aer_devices()
    aer_gpu = "GPU" in aer_devices
    gpu_name = "None detected"
    total_vram_gb = 0.0
    has_gpu = False
    gpus_list: List[Dict[str, Any]] = []
    
    # 1. Try NVIDIA-SMI if available (NVIDIA CUDA cards)
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if res.returncode == 0 and res.stdout.strip():
            lines = [l.strip() for l in res.stdout.strip().split("\n") if l.strip()]
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    idx = int(parts[0]) if parts[0].isdigit() else len(gpus_list)
                    g_name = parts[1]
                    vram_mb = float(parts[2]) if len(parts) > 2 and parts[2].replace('.', '', 1).isdigit() else 0.0
                    vram_gb = vram_mb / 1024.0
                    gpus_list.append({
                        "index": idx,
                        "name": g_name,
                        "vram_gb": round(vram_gb, 2)
                    })
                    total_vram_gb += vram_gb
            if gpus_list:
                has_gpu = True
                gpu_name = gpus_list[0]["name"]
                if len(gpus_list) > 1:
                    gpu_name = f"{len(gpus_list)}x {gpus_list[0]['name']}"
    except Exception:
        pass
        
    # 2. Platform-specific fallback if not detected by nvidia-smi
    if not has_gpu:
        sys_os = platform.system()
        if sys_os == "Windows":
            try:
                res = subprocess.run(
                    ["powershell", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                if res.returncode == 0 and res.stdout.strip():
                    names = [n.strip() for n in res.stdout.strip().split("\n") if n.strip()]
                    for i, name in enumerate(names):
                        gpus_list.append({
                            "index": i,
                            "name": name,
                            "vram_gb": 0.0
                        })
                    if names:
                        gpu_name = ", ".join(names)
                        has_gpu = True
            except Exception:
                pass
        elif sys_os == "Darwin":
            try:
                res = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                if res.returncode == 0 and res.stdout:
                    names = []
                    for line in res.stdout.split("\n"):
                        if "Chipset Model:" in line:
                            nm = line.split(":", 1)[1].strip()
                            names.append(nm)
                            gpus_list.append({
                                "index": len(gpus_list),
                                "name": nm,
                                "vram_gb": 0.0
                            })
                    if names:
                        gpu_name = ", ".join(names)
                        has_gpu = True
            except Exception:
                pass
        elif sys_os == "Linux":
            try:
                res = subprocess.run(
                    ["lspci"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if res.returncode == 0 and res.stdout:
                    names = []
                    for line in res.stdout.split("\n"):
                        if "VGA compatible controller" in line or "3D controller" in line:
                            nm = line.split(":", 2)[-1].strip()
                            names.append(nm)
                            gpus_list.append({
                                "index": len(gpus_list),
                                "name": nm,
                                "vram_gb": 0.0
                            })
                    if names:
                        gpu_name = ", ".join(names)
                        has_gpu = True
            except Exception:
                pass

    return {
        "gpu_name": gpu_name,
        "has_gpu": has_gpu,
        "gpu_count": len(gpus_list) if gpus_list else (1 if has_gpu else 0),
        "gpus": gpus_list,
        "aer_gpu_supported": aer_gpu,
        "total_vram_gb": round(total_vram_gb, 2),
        "multi_gpu_supported": len(gpus_list) > 1,
        "backend_devices": aer_devices
    }

def check_gpu_vram_safety(
    qubits: int, 
    method: str = 'statevector', 
    device: str = 'GPU',
    multi_gpu: bool = False,
    state_slicing: bool = False
) -> Tuple[bool, str]:
    """
    Check if the GPU / Multi-GPU VRAM capacity is safe for the requested qubit simulation.
    
    Args:
        qubits: Number of qubits.
        method: Simulation method ('statevector' or 'mps').
        device: Device backend ('GPU', 'MULTI_GPU', etc.).
        multi_gpu: Whether multi-GPU VRAM pooling is active.
        state_slicing: Whether distributed statevector slicing model parallelism is active.
        
    Returns:
        Tuple[bool, str]: (is_safe, message)
    """
    if not isinstance(qubits, int):
        raise TypeError("qubits must be an integer")
    if not isinstance(method, str):
        raise TypeError("method must be a string")
        
    if not is_gpu_available():
        return False, f"GPU acceleration requested for {qubits} qubits, but Qiskit Aer does not have GPU/CUDA backend support on this environment. Available devices: {get_available_aer_devices()}"
        
    # If method is MPS, VRAM is minimal
    if method.lower() in ('mps', 'matrix_product_state'):
        return True, f"MPS simulation on GPU is memory efficient for {qubits} qubits."
        
    # Statevector theoretical VRAM requirement: 2^n * 16 bytes
    req_bytes = (2 ** qubits) * 16
    req_gb = req_bytes / (1024 ** 3)
    
    gpu_meta = get_gpu_metadata()
    total_vram = gpu_meta.get("total_vram_gb", 0.0)
    gpu_count = gpu_meta.get("gpu_count", 1)
    
    is_pooling = multi_gpu or state_slicing
    effective_vram = total_vram if (is_pooling and gpu_count > 1) else (total_vram / max(1, gpu_count) if total_vram > 0 else 0.0)
    
    if effective_vram > 0.0:
        if req_gb > (effective_vram * 0.85):
            target_lbl = "Multi-GPU Aggregate VRAM (Distributed Slicing)" if is_pooling else "GPU VRAM"
            return False, f"CRITICAL: {qubits} qubits requires ~{req_gb:.2f} GB VRAM, exceeding 85% of {target_lbl} ({effective_vram:.2f} GB)."
        if req_gb > (effective_vram * 0.70):
            return True, f"WARNING: {qubits} qubits requires ~{req_gb:.2f} GB VRAM (Effective VRAM: {effective_vram:.2f} GB)."
            
    # For large statevector on GPU (>28 qubits is >4GB VRAM)
    if qubits >= 30 and not is_pooling:
        return False, f"CRITICAL: {qubits} qubits requires ~{req_gb:.2f} GB VRAM, which exceeds standard single-GPU VRAM capacity without Distributed Statevector Slicing."
        
    return True, f"SAFE: {qubits} qubits statevector requires ~{req_gb:.4f} GB VRAM on GPU."


def get_cuda_telemetry() -> Dict[str, Any]:
    """
    Query real-time NVIDIA CUDA hardware telemetry including VRAM usage,
    core utilization, temperature, power, and compute capability.
    
    Uses multi-tier probing:
      1. pynvml (NVIDIA Management Library Python bindings)
      2. PyTorch / CuPy CUDA runtime APIs
      3. nvidia-smi CLI fallback
      4. Safe non-blocking default when NVIDIA hardware is unavailable
      
    Returns:
        Dict[str, Any]: Dictionary of telemetry metrics.
    """
    telemetry: Dict[str, Any] = {
        "available": False,
        "device_name": "None detected",
        "device_count": 0,
        "compute_capability": "None",
        "vram_total_mb": 0.0,
        "vram_allocated_mb": 0.0,
        "vram_peak_mb": 0.0,
        "gpu_utilization_pct": 0.0,
        "temperature_c": None,
        "power_w": None,
        "telemetry_source": "None"
    }

    # 1. Try pynvml (NVIDIA Management Library)
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            try:
                temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                temp = None
            try:
                power = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0  # mW -> W
            except Exception:
                power = None

            # Compute capability if available via nvml
            cap_str = "None"
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                cap_str = f"{major}.{minor}"
            except Exception:
                pass

            telemetry.update({
                "available": True,
                "device_name": name,
                "device_count": count,
                "compute_capability": cap_str,
                "vram_total_mb": round(mem.total / (1024.0 * 1024.0), 2),
                "vram_allocated_mb": round(mem.used / (1024.0 * 1024.0), 2),
                "vram_peak_mb": round(mem.used / (1024.0 * 1024.0), 2),
                "gpu_utilization_pct": float(util.gpu),
                "temperature_c": temp,
                "power_w": power,
                "telemetry_source": "pynvml"
            })
            return telemetry
    except Exception:
        pass

    # 2. Try PyTorch CUDA runtime
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            dev_idx = 0
            dev_name = torch.cuda.get_device_name(dev_idx)
            major, minor = torch.cuda.get_device_capability(dev_idx)
            props = torch.cuda.get_device_properties(dev_idx)
            total_mb = props.total_memory / (1024.0 * 1024.0)
            alloc_mb = torch.cuda.memory_allocated(dev_idx) / (1024.0 * 1024.0)
            peak_mb = torch.cuda.max_memory_allocated(dev_idx) / (1024.0 * 1024.0)
            telemetry.update({
                "available": True,
                "device_name": dev_name,
                "device_count": torch.cuda.device_count(),
                "compute_capability": f"{major}.{minor}",
                "vram_total_mb": round(total_mb, 2),
                "vram_allocated_mb": round(alloc_mb, 2),
                "vram_peak_mb": round(peak_mb, 2),
                "telemetry_source": "torch.cuda"
            })
            return telemetry
    except Exception:
        pass

    # 3. Try CuPy runtime
    try:
        import cupy
        if cupy.cuda.is_available() and cupy.cuda.runtime.getDeviceCount() > 0:
            dev = cupy.cuda.Device(0)
            free_b, total_b = dev.mem_info
            cap = dev.compute_capability
            alloc_b = total_b - free_b
            telemetry.update({
                "available": True,
                "device_name": f"NVIDIA Device (CuPy Dev 0)",
                "device_count": cupy.cuda.runtime.getDeviceCount(),
                "compute_capability": str(cap),
                "vram_total_mb": round(total_b / (1024.0 * 1024.0), 2),
                "vram_allocated_mb": round(alloc_b / (1024.0 * 1024.0), 2),
                "vram_peak_mb": round(alloc_b / (1024.0 * 1024.0), 2),
                "telemetry_source": "cupy.cuda"
            })
            return telemetry
    except Exception:
        pass

    # 4. Try nvidia-smi CLI
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if res.returncode == 0 and res.stdout.strip():
            first_line = res.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) >= 3:
                name = parts[0]
                tot_mb = float(parts[1]) if parts[1].replace('.', '', 1).isdigit() else 0.0
                used_mb = float(parts[2]) if parts[2].replace('.', '', 1).isdigit() else 0.0
                util_pct = float(parts[3]) if len(parts) > 3 and parts[3].replace('.', '', 1).isdigit() else 0.0
                temp = float(parts[4]) if len(parts) > 4 and parts[4].replace('.', '', 1).isdigit() else None
                power = float(parts[5]) if len(parts) > 5 and parts[5].replace('.', '', 1).isdigit() else None
                telemetry.update({
                    "available": True,
                    "device_name": name,
                    "device_count": len(res.stdout.strip().split("\n")),
                    "compute_capability": "Detected",
                    "vram_total_mb": tot_mb,
                    "vram_allocated_mb": used_mb,
                    "vram_peak_mb": used_mb,
                    "gpu_utilization_pct": util_pct,
                    "temperature_c": temp,
                    "power_w": power,
                    "telemetry_source": "nvidia-smi"
                })
                return telemetry
    except Exception:
        pass

    return telemetry

