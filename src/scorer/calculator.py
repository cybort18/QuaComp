import math
from typing import Dict, Any, Optional

def calculate_qsim_score(
    max_qubits: int, 
    total_gates: int, 
    execution_time: float, 
    fidelity: float = 100.0
) -> float:
    """
    Calculate the QuaComp Synthetic Index (QSI) - a balanced logarithmic heuristic score.
    
    Formula:
        QSI = round(n * [w1 + w2 * log10(max(T, 1.0)) + w3 * (fidelity / 100.0)], 2)
        where:
            n = max_qubits
            T = gate throughput (total_gates / safe_time in gates/second)
            w1 = 100.0 (state-space capacity scaling base)
            w2 = 50.0  (gate throughput dynamic sensitivity)
            w3 = 25.0  (quantum state fidelity weight)
            
    Note:
        QSI balances state-space scaling with execution efficiency.
        Unlike naive exponential metrics (2^n) which dwarf execution speed, QSI's logarithmic
        throughput formulation ensures that systems with faster gate throughput on the same
        qubit count achieve clear, calibrated differentiation.
        
    Args:
        max_qubits (int): Maximum number of qubits simulated.
        total_gates (int): Total quantum gates executed.
        execution_time (float): Execution latency in seconds.
        fidelity (float, optional): Quantum state fidelity percentage [0.0 - 100.0]. Default: 100.0.
        
    Returns:
        float: Calibrated QuaComp Synthetic Index (QSI) score.
        
    Raises:
        TypeError: If input arguments are not numbers.
        ValueError: If max_qubits or total_gates is negative.
    """
    breakdown = calculate_scoring_breakdown(max_qubits, total_gates, execution_time, fidelity=fidelity)
    return breakdown["composite_score"]


# Semantic alias for the new scoring architecture
calculate_qsi_score = calculate_qsim_score


def calculate_gate_throughput_kgps(total_gates: int, execution_time: float) -> float:
    """
    Calculate normalized gate throughput in thousands of gates per second (kGates/s).
    
    Standard metric reported in Quantum HPC simulator benchmarks (e.g. Qiskit Aer GPU, cuQuantum).
    
    Args:
        total_gates (int): Total quantum gates executed.
        execution_time (float): Execution latency in seconds.
        
    Returns:
        float: Throughput in kGates/s.
    """
    if not isinstance(total_gates, int):
        raise TypeError("total_gates must be an integer.")
    if not isinstance(execution_time, (int, float)):
        raise TypeError("execution_time must be a number.")
    if total_gates < 0:
        raise ValueError("total_gates must be non-negative.")
        
    safe_time = max(float(execution_time), 1e-9)
    throughput_gps = total_gates / safe_time
    return round(float(throughput_gps / 1000.0), 4)


def calculate_energy_delay_product(latency_seconds: float, energy_joules: Optional[float]) -> Optional[float]:
    """
    Calculate the Energy-Delay Product (EDP) in Joule-seconds (J·s).
    
    A standard figure-of-merit in green computing and computer systems architecture:
        EDP = Latency (seconds) * Energy (Joules)
    Lower values indicate higher combined energy efficiency and execution speed.
    
    Args:
        latency_seconds (float): Execution duration in seconds.
        energy_joules (Optional[float]): Total energy consumed in Joules.
        
    Returns:
        Optional[float]: EDP in J·s, or None if energy data is unavailable.
    """
    if latency_seconds is None or energy_joules is None:
        return None
    if not isinstance(latency_seconds, (int, float)) or not isinstance(energy_joules, (int, float)):
        raise TypeError("latency_seconds and energy_joules must be numbers.")
    if latency_seconds < 0 or energy_joules < 0:
        raise ValueError("latency_seconds and energy_joules must be non-negative.")
        
    return round(float(latency_seconds * energy_joules), 6)


def verify_quantum_volume_status(
    heavy_output_prob: float, 
    confidence_2sigma: Optional[float] = None, 
    threshold: float = 2.0 / 3.0
) -> Dict[str, Any]:
    """
    Verify Quantum Volume certification status based on Heavy Output Probability.
    
    Per Cross et al. (2019) / IBM Quantum Volume protocol:
    A system passes depth d if h_prob > 2/3 with >97.7% statistical confidence (h_prob - 2*sigma > 2/3).
    
    Args:
        heavy_output_prob (float): Measured heavy output probability h_prob.
        confidence_2sigma (Optional[float]): Lower 2-sigma confidence bound (h_prob - 2*sigma).
        threshold (float): Target threshold (default 2/3 ≈ 0.6667).
        
    Returns:
        dict: Quantum Volume certification status, margin, and boolean certified flag.
    """
    if not isinstance(heavy_output_prob, (int, float)):
        raise TypeError("heavy_output_prob must be a number.")
    if not (0.0 <= heavy_output_prob <= 1.0):
        raise ValueError("heavy_output_prob must be between 0.0 and 1.0.")
        
    bound = confidence_2sigma if confidence_2sigma is not None else heavy_output_prob
    is_certified = bool(bound > threshold)
    margin = round(float(bound - threshold), 4)
    
    return {
        "heavy_output_probability": round(float(heavy_output_prob), 4),
        "lower_bound_2sigma": round(float(bound), 4) if confidence_2sigma is not None else None,
        "threshold": round(float(threshold), 4),
        "margin": margin,
        "qv_certified": is_certified,
        "status": "PASSED" if is_certified else "FAILED"
    }


def calculate_scoring_breakdown(
    max_qubits: int, 
    total_gates: int, 
    execution_time: float,
    fidelity: float = 100.0,
    energy_joules: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate detailed breakdown of QuaComp scoring: QSI composite score,
    state-space capacity metric, throughput metric, and formal academic indicators.
    
    Args:
        max_qubits (int): The maximum number of qubits successfully simulated.
        total_gates (int): The total number of gates executed in that simulation.
        execution_time (float): The mean execution time in seconds.
        fidelity (float, optional): Quantum state fidelity percentage [0.0 - 100.0]. Default: 100.0.
        energy_joules (float, optional): Energy consumed in Joules during execution.
        
    Returns:
        dict: Detailed scoring metrics breakdown dictionary.
    """
    if not isinstance(max_qubits, int) or not isinstance(total_gates, int):
        raise TypeError("max_qubits and total_gates must be integers.")
    if not isinstance(execution_time, (int, float)):
        raise TypeError("execution_time must be a number.")
    if not isinstance(fidelity, (int, float)):
        raise TypeError("fidelity must be a number.")
        
    if max_qubits < 0:
        raise ValueError("max_qubits must be non-negative.")
    if total_gates < 0:
        raise ValueError("total_gates must be non-negative.")
        
    # Prevent division by zero or negative times with safe floor
    safe_time = max(float(execution_time), 1e-6)
    
    capacity_metric = float(2 ** max_qubits)
    throughput_metric = float(total_gates / safe_time)
    throughput_kgps = calculate_gate_throughput_kgps(total_gates, safe_time)
    
    # Calculate EDP if energy data provided
    edp = calculate_energy_delay_product(safe_time, energy_joules) if energy_joules is not None else None
    
    # Normalized fidelity factor [0.0 - 1.0]
    fid_clamped = max(0.0, min(100.0, float(fidelity)))
    fid_factor = fid_clamped / 100.0
    
    # QSI balanced formulation
    # Base per qubit: 100.0, Throughput log10 factor: 50.0, Fidelity weight: 25.0
    w1 = 100.0
    w2 = 50.0
    w3 = 25.0
    
    if max_qubits == 0 and total_gates == 0:
        composite_score = 0.0
    elif max_qubits == 0:
        log_t = math.log10(max(throughput_metric, 1.0))
        composite_score = round(w2 * log_t, 2)
    else:
        log_t = math.log10(max(throughput_metric, 1.0))
        qubit_weight = float(max_qubits)
        score_val = qubit_weight * (w1 + (w2 * log_t) + (w3 * fid_factor))
        composite_score = round(score_val, 2)
        
    return {
        "composite_score": composite_score,
        "qsi_score": composite_score,
        "score_type": "QuaComp Synthetic Index (QSI)",
        "capacity_metric": capacity_metric,
        "throughput_metric": throughput_metric,
        "throughput_kgps": throughput_kgps,
        "fidelity_factor": fid_factor,
        "energy_delay_product": edp
    }


def categorize_score(score: float) -> str:
    """
    Categorize the QuaComp Synthetic Index (QSI) into system performance tiers.
    
    Tiers:
        - Entry-Level: < 2,500 pts (Up to ~10-12 Qubits)
        - Mid-Range: 2,500 - 6,000 pts (Up to ~14-22 Qubits)
        - High-Performance: 6,000 - 10,000 pts (Up to ~24-30 Qubits)
        - Extreme Workstation: >= 10,000 pts (30+ Qubits / Advanced Tensor Networks)
        
    Args:
        score (float): The calculated QSI benchmark score.
        
    Returns:
        str: The performance tier category name.
    """
    if score < 2500.0:
        return "Entry-Level"
    elif score < 6000.0:
        return "Mid-Range"
    elif score < 10000.0:
        return "High-Performance"
    else:
        return "Extreme Workstation"
