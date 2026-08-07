import psutil

def estimate_qubit_ram(num_qubits: int) -> int:
    """
    Estimate the RAM required for statevector simulation of a given number of qubits.
    
    Formula:
        RAM (Bytes) = (2^n) * 16 bytes (using complex128)
        
    Args:
        num_qubits (int): Number of qubits.
        
    Returns:
        int: Estimated RAM in bytes.
        
    Raises:
        TypeError: If num_qubits is not an integer.
        ValueError: If num_qubits is negative.
    """
    if not isinstance(num_qubits, int):
        raise TypeError("Number of qubits must be an integer.")
    if num_qubits < 0:
        raise ValueError("Number of qubits must be non-negative.")
    
    return (2 ** num_qubits) * 16

def check_memory_safety(num_qubits: int) -> tuple[bool, str]:
    """
    Check if simulation is safe to run based on the available physical memory.
    
    Safety Threshold:
        If estimated memory exceeds 85% of available RAM, returns False with a warning.
        If estimated memory exceeds 100% of available RAM, returns False with a critical error.
        Otherwise, returns True.
        
    Args:
        num_qubits (int): Number of qubits.
        
    Returns:
        tuple[bool, str]: (is_safe, message)
    """
    try:
        estimated_ram = estimate_qubit_ram(num_qubits)
    except (TypeError, ValueError) as e:
        return False, str(e)
        
    vm = psutil.virtual_memory()
    available_ram = vm.available
    
    estimated_gb = estimated_ram / (1024 ** 3)
    available_gb = available_ram / (1024 ** 3)
    
    if estimated_ram > available_ram:
        return False, (
            f"CRITICAL: Estimated RAM ({estimated_gb:.4f} GB) exceeds "
            f"available physical RAM ({available_gb:.4f} GB). "
            f"Simulation will likely cause Out-Of-Memory (OOM) crash."
        )
        
    safety_limit = int(0.85 * available_ram)
    safety_limit_gb = safety_limit / (1024 ** 3)
    
    if estimated_ram > safety_limit:
        return False, (
            f"WARNING: Estimated RAM ({estimated_gb:.4f} GB) exceeds "
            f"85% safety threshold of available RAM ({safety_limit_gb:.4f} GB / "
            f"total available: {available_gb:.4f} GB). "
            f"High risk of system instability."
        )
        
    return True, (
        f"SAFE: Estimated RAM ({estimated_gb:.4f} GB) is within safe limits. "
        f"Available RAM: {available_gb:.4f} GB (85% threshold: {safety_limit_gb:.4f} GB)."
    )
