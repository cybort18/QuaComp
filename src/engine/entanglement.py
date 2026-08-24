import math
from typing import Dict, Any, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def calculate_bipartite_entropy(
    circuit: QuantumCircuit, 
    subsystem_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculate the Bipartite Von Neumann Entanglement Entropy and Schmidt decomposition metrics
    for a given quantum circuit state.
    
    The quantum system of n qubits is partitioned into two subsystems:
      - Subsystem A: n_A qubits (default: floor(n/2))
      - Subsystem B: n_B = n - n_A qubits
      
    Args:
        circuit (QuantumCircuit): The Qiskit quantum circuit to analyze.
        subsystem_size (Optional[int]): Number of qubits in Subsystem A (default: n // 2).
        
    Returns:
        dict: A dictionary containing entanglement metrics:
            - "num_qubits" (int): Total number of qubits.
            - "subsystem_a_size" (int): Number of qubits in Subsystem A.
            - "subsystem_b_size" (int): Number of qubits in Subsystem B.
            - "von_neumann_entropy" (float): Entanglement entropy S(rho_A) in bits (base 2).
            - "max_possible_entropy" (float): Theoretical maximum entropy log2(dim(A)) = n_A.
            - "entanglement_ratio" (float): Normalized ratio S / S_max in range [0, 1].
            - "schmidt_rank" (int): Number of non-zero Schmidt singular values.
            - "participation_ratio" (float): Effective rank K = 1 / sum(p_i^2).
            - "entanglement_regime" (str): 'Product State', 'Low (Area-law)', 'Moderate', or 'Volume-law (Maximal)'.
            - "mps_hardness" (str): Simulation hardness rating for Matrix Product State methods.
            - "singular_values" (list): Top Schmidt singular values (up to 16).
            
    Raises:
        TypeError: If circuit is not a QuantumCircuit instance.
        ValueError: If subsystem_size is invalid.
    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("Input must be a Qiskit QuantumCircuit instance.")
        
    n = circuit.num_qubits
    if n < 1:
        raise ValueError("Circuit must have at least 1 qubit.")
        
    # Handle single qubit edge case (no bipartite entanglement possible)
    if n == 1:
        return {
            "num_qubits": 1,
            "subsystem_a_size": 1,
            "subsystem_b_size": 0,
            "von_neumann_entropy": 0.0,
            "max_possible_entropy": 0.0,
            "entanglement_ratio": 0.0,
            "schmidt_rank": 1,
            "participation_ratio": 1.0,
            "entanglement_regime": "Product State",
            "mps_hardness": "Trivial (chi=1)",
            "singular_values": [1.0]
        }
        
    # Determine subsystem partition size
    if subsystem_size is None:
        n_a = n // 2
    else:
        if not isinstance(subsystem_size, int):
            raise TypeError("subsystem_size must be an integer.")
        if subsystem_size < 1 or subsystem_size >= n:
            raise ValueError(f"subsystem_size must be between 1 and {n - 1}.")
        n_a = subsystem_size
        
    n_b = n - n_a
    s_max = float(min(n_a, n_b))
    
    # Remove any measurement operations to obtain pure statevector
    circ_clean = circuit.copy()
    circ_clean.remove_final_measurements(inplace=True)
    
    # Obtain exact statevector
    sv = Statevector.from_instruction(circ_clean)
    sv_data = np.asarray(sv.data)
    
    # Reshape statevector array into 2D bipartite matrix (2^n_a x 2^n_b)
    dim_a = 2 ** n_a
    dim_b = 2 ** n_b
    bipartite_matrix = sv_data.reshape((dim_a, dim_b))
    
    # Compute Schmidt singular values via Singular Value Decomposition (SVD)
    singular_values = np.linalg.svd(bipartite_matrix, compute_uv=False)
    
    # Calculate Schmidt probabilities p_i = lambda_i^2
    probs = singular_values ** 2
    # Numerical tolerance filter for non-zero eigenvalues
    non_zero_probs = probs[probs > 1e-14]
    
    if len(non_zero_probs) == 0:
        s_vn = 0.0
        schmidt_rank = 1
        participation_ratio = 1.0
    else:
        # Normalize probabilities to sum exactly to 1.0
        non_zero_probs = non_zero_probs / np.sum(non_zero_probs)
        # Von Neumann Entropy: S = - sum(p * log2(p))
        s_vn = float(-np.sum(non_zero_probs * np.log2(non_zero_probs)))
        schmidt_rank = int(len(non_zero_probs))
        # Participation ratio: K = 1 / sum(p_i^2)
        participation_ratio = float(1.0 / np.sum(non_zero_probs ** 2))
        
    # Clean precision floating point artifacts
    if s_vn < 1e-10:
        s_vn = 0.0
        
    entanglement_ratio = (s_vn / s_max) if s_max > 0 else 0.0
    
    # Classify Entanglement Regime
    if s_vn < 1e-6:
        regime = "Product State"
        hardness = "Trivial (chi=1)"
    elif s_vn <= 1.0:
        regime = "Low (Area-law)"
        hardness = "Efficient (Low chi <= 16)"
    elif s_vn < (0.7 * s_max):
        regime = "Moderate Entanglement"
        hardness = "Challenging (Moderate chi <= 64)"
    else:
        regime = "Volume-law (Maximal)"
        hardness = "Exponentially Hard (Volume-law)"
        
    return {
        "num_qubits": n,
        "subsystem_a_size": n_a,
        "subsystem_b_size": n_b,
        "von_neumann_entropy": round(s_vn, 4),
        "max_possible_entropy": round(s_max, 4),
        "entanglement_ratio": round(entanglement_ratio, 4),
        "schmidt_rank": schmidt_rank,
        "participation_ratio": round(participation_ratio, 4),
        "entanglement_regime": regime,
        "mps_hardness": hardness,
        "singular_values": [round(float(v), 5) for v in singular_values[:16]]
    }
