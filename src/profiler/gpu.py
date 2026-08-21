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
    Probe the local hardware to identify any GPU (NVIDIA, AMD, Apple, Intel) and VRAM capacity.
    
    Returns:
        Dictionary containing GPU metadata:
            - 'gpu_name' (str): Brand and model of detected GPU.
            - 'has_gpu' (bool): Whether a GPU is physically present.
            - 'aer_gpu_supported' (bool): Whether Qiskit Aer has GPU runtime enabled.
            - 'total_vram_gb' (float): Total VRAM in GB if detectable, else 0.0.
            - 'backend_devices' (list): Available Qiskit Aer devices.
    """
    aer_devices = get_available_aer_devices()
    aer_gpu = "GPU" in aer_devices
    gpu_name = "None detected"
    total_vram_gb = 0.0
    has_gpu = False
    
    # 1. Try NVIDIA-SMI if available (NVIDIA CUDA cards)
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if res.returncode == 0 and res.stdout.strip():
            lines = res.stdout.strip().split("\n")
            first = lines[0].split(",")
            gpu_name = first[0].strip()
            if len(first) > 1:
                total_vram_gb = float(first[1].strip()) / 1024.0
            has_gpu = True
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
                    for line in res.stdout.split("\n"):
                        if "Chipset Model:" in line:
                            gpu_name = line.split(":", 1)[1].strip()
                            has_gpu = True
                            break
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
                    for line in res.stdout.split("\n"):
                        if "VGA compatible controller" in line or "3D controller" in line:
                            gpu_name = line.split(":", 2)[-1].strip()
                            has_gpu = True
                            break
            except Exception:
                pass

    return {
        "gpu_name": gpu_name,
        "has_gpu": has_gpu,
        "aer_gpu_supported": aer_gpu,
        "total_vram_gb": total_vram_gb,
        "backend_devices": aer_devices
    }

def check_gpu_vram_safety(qubits: int, method: str = 'statevector') -> Tuple[bool, str]:
    """
    Check if the GPU VRAM capacity is safe for the requested qubit simulation.
    
    Args:
        qubits: Number of qubits.
        method: Simulation method ('statevector' or 'mps').
        
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
    
    if total_vram > 0.0:
        if req_gb > (total_vram * 0.85):
            return False, f"CRITICAL: {qubits} qubits requires ~{req_gb:.2f} GB VRAM, exceeding 85% of GPU VRAM ({total_vram:.2f} GB)."
        if req_gb > (total_vram * 0.70):
            return True, f"WARNING: {qubits} qubits requires ~{req_gb:.2f} GB VRAM (GPU VRAM: {total_vram:.2f} GB)."
            
    # For large statevector on GPU (>28 qubits is >4GB VRAM)
    if qubits >= 30:
        return False, f"CRITICAL: {qubits} qubits requires ~{req_gb:.2f} GB VRAM, which typically exceeds standard GPU VRAM capacity."
        
    return True, f"SAFE: {qubits} qubits statevector requires ~{req_gb:.4f} GB VRAM on GPU."
