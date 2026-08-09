from typing import Optional, Dict, Any
import numpy as np
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error

def get_noise_model(level: str = 'none') -> Optional[NoiseModel]:
    """
    Generate a Qiskit Aer NoiseModel based on the specified noise preset level.
    
    Presets:
        - 'none': Returns None (ideal simulation).
        - 'low': T1=100µs, T2=120µs, gate error rate 0.001 (0.1%).
        - 'medium': T1=50µs, T2=70µs, gate error rate 0.005 (0.5%).
        - 'high': T1=20µs, T2=30µs, gate error rate 0.02 (2.0%).
        
    Args:
        level (str): Noise level ('none', 'low', 'medium', 'high').
        
    Returns:
        Optional[NoiseModel]: Built Qiskit Aer NoiseModel or None if level is 'none'.
        
    Raises:
        TypeError: If level is not a string.
        ValueError: If level is not one of the valid options.
    """
    if not isinstance(level, str):
        raise TypeError("Noise level must be a string.")
        
    norm_level = level.strip().lower()
    
    if norm_level == 'none':
        return None
        
    noise_presets = {
        'low': {'t1': 100e-6, 't2': 120e-6, 'gate_error': 0.001},
        'medium': {'t1': 50e-6, 't2': 70e-6, 'gate_error': 0.005},
        'high': {'t1': 20e-6, 't2': 30e-6, 'gate_error': 0.02},
    }
    
    if norm_level not in noise_presets:
        raise ValueError(f"Unknown noise level '{level}'. Valid choices are: 'none', 'low', 'medium', 'high'.")
        
    params = noise_presets[norm_level]
    t1 = params['t1']
    t2 = params['t2']
    gate_error = params['gate_error']
    
    # Typical gate times (seconds)
    time_single_qubit = 50e-9   # 50 ns
    time_two_qubit = 300e-9     # 300 ns
    
    noise_model = NoiseModel()
    
    # Single qubit errors
    error_thermal_1q = thermal_relaxation_error(t1, t2, time_single_qubit)
    error_depol_1q = depolarizing_error(gate_error, 1)
    error_1q = error_thermal_1q.compose(error_depol_1q)
    
    # Two qubit errors
    error_thermal_2q = thermal_relaxation_error(t1, t2, time_two_qubit).tensor(
        thermal_relaxation_error(t1, t2, time_two_qubit)
    )
    error_depol_2q = depolarizing_error(gate_error * 2, 2)
    error_2q = error_thermal_2q.compose(error_depol_2q)
    
    single_qubit_gates = ['h', 'rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'x', 'y', 'z']
    two_qubit_gates = ['cx', 'cp', 'cz', 'swap']
    
    noise_model.add_all_qubit_quantum_error(error_1q, single_qubit_gates)
    noise_model.add_all_qubit_quantum_error(error_2q, two_qubit_gates)
    
    return noise_model

def calculate_state_fidelity(ideal_counts: Dict[str, int], noisy_counts: Dict[str, int]) -> float:
    """
    Calculate the classical Hellinger state fidelity percentage between ideal and noisy measurement counts.
    
    Formula:
        Fidelity (%) = ( sum_x sqrt( P_ideal(x) * P_noisy(x) ) )^2 * 100.0
        
    Args:
        ideal_counts (dict): Measurement counts dictionary from ideal simulation.
        noisy_counts (dict): Measurement counts dictionary from noisy simulation.
        
    Returns:
        float: State fidelity percentage between 0.0% and 100.0%.
        
    Raises:
        TypeError: If inputs are not dictionaries.
    """
    if not isinstance(ideal_counts, dict) or not isinstance(noisy_counts, dict):
        raise TypeError("Counts inputs must be dictionaries.")
        
    if not ideal_counts or not noisy_counts:
        return 100.0
        
    total_ideal = sum(ideal_counts.values())
    total_noisy = sum(noisy_counts.values())
    
    if total_ideal == 0 or total_noisy == 0:
        return 0.0
        
    all_keys = set(ideal_counts.keys()).union(set(noisy_counts.keys()))
    
    fidelity_sqrt = 0.0
    for key in all_keys:
        p_ideal = ideal_counts.get(key, 0) / total_ideal
        p_noisy = noisy_counts.get(key, 0) / total_noisy
        fidelity_sqrt += np.sqrt(p_ideal * p_noisy)
        
    fidelity_pct = (fidelity_sqrt ** 2) * 100.0
    return float(np.clip(fidelity_pct, 0.0, 100.0))

def calculate_overhead_ratio(ideal_latency: float, noisy_latency: float) -> float:
    """
    Calculate the percentage increase in computational latency due to noise model calculations.
    
    Formula:
        Overhead Ratio (%) = max(0.0, ((noisy_latency - ideal_latency) / ideal_latency) * 100.0)
        
    Args:
        ideal_latency (float): Execution time for ideal simulation in seconds.
        noisy_latency (float): Execution time for noisy simulation in seconds.
        
    Returns:
        float: Computation overhead percentage ratio.
        
    Raises:
        TypeError: If inputs are not numbers.
        ValueError: If latencies are negative.
    """
    if not isinstance(ideal_latency, (int, float)) or not isinstance(noisy_latency, (int, float)):
        raise TypeError("Latencies must be numbers.")
        
    if ideal_latency < 0 or noisy_latency < 0:
        raise ValueError("Latencies must be non-negative.")
        
    if ideal_latency <= 0:
        return 0.0
        
    overhead = ((noisy_latency - ideal_latency) / ideal_latency) * 100.0
    return float(max(0.0, overhead))
