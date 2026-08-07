import platform
import os
import sys
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
    return {
        "cpu_name": get_cpu_name(),
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
