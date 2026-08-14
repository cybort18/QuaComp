from typing import Dict, Any

def calculate_qsim_score(max_qubits: int, total_gates: int, execution_time: float) -> float:
    """
    Calculate the QuaComp Composite Score (a project-specific heuristic score).
    
    The score combines an exponential state-space capacity metric with gate throughput:
        Capacity Metric (C) = 2^max_qubits
        Throughput Metric (T) = total_gates / mean_execution_time
        QuaComp Composite Score = (C * 10) + T
        
    Note:
        QuaComp Score is a project-specific composite heuristic prioritizing state-space capacity scaling.
        Because statevector memory allocation scales exponentially (2^n), the Capacity Metric (10 * 2^n)
        exponentially dominates the Throughput Metric (T). A system simulating 30 qubits will score higher
        than a system simulating 28 qubits with faster gate throughput, reflecting QuaComp's deliberate
        design choice to prioritize state-space memory scaling over gate execution speed.
        
    Args:
        max_qubits (int): The maximum number of qubits successfully simulated.
        total_gates (int): The total number of gates executed in that simulation.
        execution_time (float): The mean execution time in seconds.
        
    Returns:
        float: The calculated composite heuristic score.
        
    Raises:
        TypeError: If input types are incorrect.
        ValueError: If max_qubits or total_gates is negative.
    """
    breakdown = calculate_scoring_breakdown(max_qubits, total_gates, execution_time)
    return breakdown["composite_score"]

def calculate_scoring_breakdown(max_qubits: int, total_gates: int, execution_time: float) -> Dict[str, float]:
    """
    Calculate detailed breakdown of QuaComp scoring sub-metrics: Capacity Metric (C) and Throughput Metric (T).
    
    Args:
        max_qubits (int): The maximum number of qubits successfully simulated.
        total_gates (int): The total number of gates executed in that simulation.
        execution_time (float): The mean execution time in seconds.
        
    Returns:
        dict: A dictionary containing:
            - "composite_score" (float): Final project-specific heuristic score.
            - "capacity_metric" (float): Qubit state-space capacity (2^n).
            - "throughput_metric" (float): Gates processed per second (gates / time).
    """
    if not isinstance(max_qubits, int) or not isinstance(total_gates, int):
        raise TypeError("max_qubits and total_gates must be integers.")
    if not isinstance(execution_time, (int, float)):
        raise TypeError("execution_time must be a number.")
        
    if max_qubits < 0:
        raise ValueError("max_qubits must be non-negative.")
    if total_gates < 0:
        raise ValueError("total_gates must be non-negative.")
        
    # Prevent division by zero or negative times
    safe_time = max(execution_time, 1e-9)
    
    capacity_metric = float(2 ** max_qubits)
    throughput_metric = float(total_gates / safe_time)
    composite_score = float((capacity_metric * 10.0) + throughput_metric)
    
    return {
        "composite_score": composite_score,
        "capacity_metric": capacity_metric,
        "throughput_metric": throughput_metric
    }

def categorize_score(score: float) -> str:
    """
    Categorize the QuaComp composite score into system performance tiers.
    
    Tiers:
        - Entry-Level: < 100,000 pts (Up to 18-20 Qubits)
        - Mid-Range: 100,000 - 1,000,000 pts (Up to 22-25 Qubits)
        - High-Performance: 1,000,000 - 50,000,000 pts (Up to 26-28 Qubits)
        - Extreme Workstation: > 50,000,000 pts (29+ Qubits)
        
    Args:
        score (float): The calculated benchmark score.
        
    Returns:
        str: The performance tier category name.
    """
    if score < 100000:
        return "Entry-Level"
    elif score < 1000000:
        return "Mid-Range"
    elif score < 50000000:
        return "High-Performance"
    else:
        return "Extreme Workstation"
