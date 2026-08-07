def calculate_qsim_score(max_qubits: int, total_gates: int, execution_time: float) -> float:
    """
    Calculate the QuaComp (QSim) benchmark score.
    
    Formula:
        Score = (2^max_qubits * 10) + (total_gates / execution_time)
        
    Args:
        max_qubits (int): The maximum number of qubits successfully simulated.
        total_gates (int): The total number of gates executed in that simulation.
        execution_time (float): The execution time in seconds.
        
    Returns:
        float: The calculated score.
        
    Raises:
        TypeError: If input types are incorrect.
        ValueError: If max_qubits or total_gates is negative.
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
    
    score = (2 ** max_qubits * 10) + (total_gates / safe_time)
    return float(score)

def categorize_score(score: float) -> str:
    """
    Categorize the QuaComp score into system performance tiers.
    
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
