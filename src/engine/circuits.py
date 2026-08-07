import random
import numpy as np
from qiskit import QuantumCircuit

def generate_shallow_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Generate a shallow quantum circuit.
    
    Workload:
        Hadamard (H) gate on all qubits + CNOT (CX) gate chain.
        
    Args:
        num_qubits (int): Number of qubits.
        
    Returns:
        QuantumCircuit: The generated Qiskit circuit.
        
    Raises:
        TypeError: If num_qubits is not an integer.
        ValueError: If num_qubits is less than 1.
    """
    if not isinstance(num_qubits, int):
        raise TypeError("Number of qubits must be an integer.")
    if num_qubits < 1:
        raise ValueError("Number of qubits must be at least 1.")
        
    qc = QuantumCircuit(num_qubits)
    
    # Hadamard on all qubits
    for i in range(num_qubits):
        qc.h(i)
        
    # CNOT chain
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
        
    return qc

def generate_deep_circuit(num_qubits: int, depth: int, seed: int = 42) -> QuantumCircuit:
    """
    Generate a deep random quantum circuit.
    
    Workload:
        Alternating layers of random rotation gates (Rx, Ry, Rz) and entangling CNOT chains.
        
    Args:
        num_qubits (int): Number of qubits.
        depth (int): Number of alternating layers.
        seed (int): Random seed for reproducibility.
        
    Returns:
        QuantumCircuit: The generated Qiskit circuit.
        
    Raises:
        TypeError: If num_qubits or depth are not integers.
        ValueError: If num_qubits is less than 1 or depth is negative.
    """
    if not isinstance(num_qubits, int) or not isinstance(depth, int):
        raise TypeError("Number of qubits and depth must be integers.")
    if num_qubits < 1:
        raise ValueError("Number of qubits must be at least 1.")
    if depth < 0:
        raise ValueError("Depth must be non-negative.")
        
    qc = QuantumCircuit(num_qubits)
    rng = random.Random(seed)
    
    for _ in range(depth):
        # 1. Apply single-qubit random rotation layer
        for i in range(num_qubits):
            gate_type = rng.choice(['rx', 'ry', 'rz'])
            angle = rng.uniform(0, 2 * np.pi)
            if gate_type == 'rx':
                qc.rx(angle, i)
            elif gate_type == 'ry':
                qc.ry(angle, i)
            else:
                qc.rz(angle, i)
                
        # 2. Apply entangling CNOT chain layer
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
            
    return qc

def generate_qft_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Generate a standard Quantum Fourier Transform (QFT) circuit.
    
    Args:
        num_qubits (int): Number of qubits.
        
    Returns:
        QuantumCircuit: The generated Qiskit circuit.
        
    Raises:
        TypeError: If num_qubits is not an integer.
        ValueError: If num_qubits is less than 1.
    """
    if not isinstance(num_qubits, int):
        raise TypeError("Number of qubits must be an integer.")
    if num_qubits < 1:
        raise ValueError("Number of qubits must be at least 1.")
        
    qc = QuantumCircuit(num_qubits)
    
    for i in range(num_qubits):
        qc.h(i)
        for j in range(i + 1, num_qubits):
            angle = np.pi / (2 ** (j - i))
            qc.cp(angle, j, i)
            
    # Swap qubits to reverse order
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - 1 - i)
        
    return qc
