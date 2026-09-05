import sys
import os
import time
import threading
from typing import Dict, Any, Optional
import psutil

def estimate_cpu_tdp(cpu_name: Optional[str] = None) -> float:
    """
    Estimate the Thermal Design Power (TDP) in Watts for the host CPU.
    
    Args:
        cpu_name (Optional[str]): Host CPU brand string.
        
    Returns:
        float: Estimated TDP in Watts.
    """
    name = (cpu_name or "").lower()
    if not name:
        try:
            from src.profiler.telemetry import get_system_metadata
            meta = get_system_metadata()
            name = meta.get("cpu_name", "").lower()
        except Exception:
            name = ""
            
    import re
    if "threadripper" in name or "epyc" in name or "xeon" in name:
        return 180.0
    elif "m1" in name or "m2" in name or "m3" in name or "m4" in name:
        return 30.0  # Apple Silicon SoC package
    elif "i9" in name or "ryzen 9" in name:
        return 105.0
    elif "5300u" in name or re.search(r'\b\d{4,5}u\b', name):
        return 15.0  # Mobile U-series 15W
    elif "5800h" in name or re.search(r'\b\d{4,5}h(x|s)?\b', name):
        return 45.0  # Mobile H-series 45W
    elif "i7" in name or "ryzen 7" in name:
        return 65.0
    elif "i5" in name or "ryzen 5" in name:
        return 45.0
    elif "i3" in name or "ryzen 3" in name:
        return 25.0
    else:
        return 35.0  # Generic balanced baseline


class EnergyProfiler:
    """
    Cross-platform Energy & Power Consumption Profiler.
    
    Supports:
    - Linux Running Average Power Limit (RAPL) microjoule hardware counter.
    - Windows / macOS Dynamic TDP & Utilization integration model.
    - Energy per Quantum Operation (EQO) calculation in Joules/gate.
    """
    def __init__(self, tdp_watts: Optional[float] = None, sample_interval: float = 0.05):
        self.tdp = tdp_watts if tdp_watts is not None else estimate_cpu_tdp()
        self.idle_power = max(2.0, self.tdp * 0.15)  # 15% TDP idle power floor
        self.sample_interval = sample_interval
        self._stop_event = threading.Event()
        self._samples = []
        self._thread = None
        self.start_time = 0.0
        self.end_time = 0.0
        self.backend = "Dynamic TDP Model"
        
        # Check Linux RAPL availability
        self.rapl_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
        self.use_rapl = os.path.exists(self.rapl_path) and os.access(self.rapl_path, os.R_OK)
        self.rapl_start_uj = 0
        self.rapl_end_uj = 0
        
        if self.use_rapl:
            self.backend = "Linux RAPL Hardware Interface"
        elif sys.platform == "win32":
            self.backend = "Windows Dynamic TDP Model"
        elif sys.platform == "darwin":
            self.backend = "macOS Dynamic TDP Model"
            
    def _sampling_worker(self):
        while not self._stop_event.is_set():
            try:
                cpu_p = psutil.cpu_percent(interval=None)
                self._samples.append(cpu_p)
            except Exception:
                pass
            time.sleep(self.sample_interval)
            
    def __enter__(self):
        self._samples = []
        self._stop_event.clear()
        self.start_time = time.perf_counter()
        
        if self.use_rapl:
            try:
                with open(self.rapl_path, "r") as f:
                    self.rapl_start_uj = int(f.read().strip())
            except Exception:
                self.use_rapl = False
                
        if not self.use_rapl:
            self._thread = threading.Thread(target=self._sampling_worker, daemon=True)
            self._thread.start()
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        
        if self.use_rapl:
            try:
                with open(self.rapl_path, "r") as f:
                    self.rapl_end_uj = int(f.read().strip())
            except Exception:
                pass
        else:
            self._stop_event.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=0.2)
                
    def get_metrics(self, num_gates: int = 1) -> Dict[str, Any]:
        """
        Calculate total energy (Joules), average power (Watts), and Energy per Quantum Operation (EQO).
        
        Args:
            num_gates (int): Total quantum operations executed.
            
        Returns:
            dict: Energy metrics breakdown.
        """
        duration = max(1e-6, self.end_time - self.start_time)
        effective_gates = max(1, num_gates)
        
        if self.use_rapl and self.rapl_end_uj > self.rapl_start_uj:
            total_energy = float((self.rapl_end_uj - self.rapl_start_uj) / 1e6)
            avg_power = float(total_energy / duration)
        else:
            # Dynamic TDP Integration: P(t) = P_idle + (U(t)/100) * (TDP - P_idle)
            if self._samples:
                mean_cpu = float(sum(self._samples) / len(self._samples))
            else:
                try:
                    mean_cpu = float(psutil.cpu_percent(interval=None))
                except Exception:
                    mean_cpu = 50.0
                    
            dynamic_range = self.tdp - self.idle_power
            avg_power = float(self.idle_power + (mean_cpu / 100.0) * dynamic_range)
            total_energy = float(avg_power * duration)
            
        eqo_joules = float(total_energy / effective_gates)
        
        return {
            "energy_backend": self.backend,
            "duration_seconds": round(duration, 4),
            "estimated_tdp_watts": round(self.tdp, 1),
            "average_power_watts": round(avg_power, 2),
            "total_energy_joules": round(total_energy, 6),
            "energy_per_quantum_op_joules": round(eqo_joules, 8),  # EQO
            "eqo_microjoules": round(eqo_joules * 1e6, 4)
        }
