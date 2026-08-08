from typing import Dict, Any

def get_mps_configuration(bond_dim: int) -> Dict[str, Any]:
    """
    Get the configuration dictionary for AerSimulator in Matrix Product State (MPS) mode.
    
    Args:
        bond_dim (int): Maximum bond dimension for MPS.
        
    Returns:
        dict: The Aer configuration settings dictionary.
        
    Raises:
        TypeError: If bond_dim is not an integer.
        ValueError: If bond_dim is less than 1.
    """
    if not isinstance(bond_dim, int):
        raise TypeError("bond_dim must be an integer.")
    if bond_dim < 1:
        raise ValueError("bond_dim must be at least 1.")
        
    return {
        "matrix_product_state_max_bond_dimension": bond_dim
    }

def calculate_mps_ram_savings(num_qubits: int, actual_ram_used: float) -> Dict[str, Any]:
    """
    Calculate the RAM savings of MPS compared to theoretical Statevector simulation.
    
    Args:
        num_qubits (int): Number of simulated qubits.
        actual_ram_used (float): Actual physical RAM used in bytes.
        
    Returns:
        dict: A dictionary containing:
            - "statevector_ram_est_bytes" (int): Theoretical RAM allocation for Statevector.
            - "actual_ram_used_bytes" (float): Actual RAM used by simulation.
            - "savings_bytes" (float): Raw savings in bytes (Statevector RAM - Actual RAM).
            - "savings_percent" (float): Savings percentage relative to Statevector.
            
    Raises:
        TypeError: If num_qubits or actual_ram_used types are incorrect.
        ValueError: If num_qubits or actual_ram_used is negative.
    """
    if not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer.")
    if not isinstance(actual_ram_used, (int, float)):
        raise TypeError("actual_ram_used must be a number.")
        
    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative.")
    if actual_ram_used < 0:
        raise ValueError("actual_ram_used must be non-negative.")
        
    # Theoretical statevector memory allocation: 2^n * 16 bytes
    statevector_ram_est = (2 ** num_qubits) * 16
    
    # Calculate savings
    savings_bytes = max(0.0, statevector_ram_est - actual_ram_used)
    
    if statevector_ram_est > 0:
        savings_percent = (savings_bytes / statevector_ram_est) * 100
    else:
        savings_percent = 0.0
        
    return {
        "statevector_ram_est_bytes": statevector_ram_est,
        "actual_ram_used_bytes": float(actual_ram_used),
        "savings_bytes": float(savings_bytes),
        "savings_percent": float(savings_percent)
    }
