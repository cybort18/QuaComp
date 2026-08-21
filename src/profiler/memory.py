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

def check_memory_safety(num_qubits: int, method: str = 'statevector', device: str = 'CPU') -> tuple[bool, str]:
    """
    Check if simulation is safe to run based on available physical memory (system RAM or GPU VRAM).
    
    Args:
        num_qubits (int): Number of qubits.
        method (str): Simulation method ('statevector' or 'mps'/'matrix_product_state').
        device (str): Compute device ('CPU' or 'GPU').
        
    Returns:
        tuple[bool, str]: (is_safe, message)
    """
    if device and str(device).upper() == 'GPU':
        from src.profiler.gpu import check_gpu_vram_safety
        return check_gpu_vram_safety(num_qubits, method)
        
    vm = psutil.virtual_memory()
    available_ram = vm.available
    available_gb = available_ram / (1024 ** 3)
    
    # MPS safety check
    if method in ('mps', 'matrix_product_state'):
        min_ram_needed = 200 * (1024 ** 2)  # 200 MB baseline
        if available_ram < min_ram_needed:
            min_gb = min_ram_needed / (1024 ** 3)
            return False, (
                f"CRITICAL: Available physical RAM ({available_gb:.4f} GB) is lower than "
                f"the baseline threshold required for MPS simulations ({min_gb:.4f} GB)."
            )
        return True, (
            f"SAFE: MPS simulation selected. Exponential Statevector RAM safety checks bypassed. "
            f"Available RAM: {available_gb:.4f} GB."
        )
        
    # Statevector safety check
    try:
        estimated_ram = estimate_qubit_ram(num_qubits)
    except (TypeError, ValueError) as e:
        return False, str(e)
        
    estimated_gb = estimated_ram / (1024 ** 3)
    
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
