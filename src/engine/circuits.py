import random
from typing import Dict, List, Optional, Tuple, Any
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


def generate_vqe_circuit(num_qubits: int, depth: int = 2, entanglement: str = "linear") -> QuantumCircuit:
    """
    Generate a parameterized Variational Quantum Eigensolver (VQE) ansatz circuit.
    
    Architecture:
        Alternating layers of parameterized Ry rotations and entangling CX gates (TwoLocal / RealAmplitudes style).
        
    Args:
        num_qubits (int): Number of qubits in the ansatz.
        depth (int): Number of variational layers (default: 2).
        entanglement (str): Entanglement pattern ('linear' or 'full').
        
    Returns:
        QuantumCircuit: A parameterized Qiskit circuit with symbolic ParameterVector.
        
    Raises:
        TypeError: If num_qubits or depth are not integers.
        ValueError: If num_qubits < 1, depth < 1, or invalid entanglement strategy.
    """
    if not isinstance(num_qubits, int) or not isinstance(depth, int):
        raise TypeError("num_qubits and depth must be integers.")
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1.")
    if depth < 1:
        raise ValueError("depth must be at least 1.")
    if entanglement not in ("linear", "full"):
        raise ValueError("entanglement must be 'linear' or 'full'.")

    from qiskit.circuit import ParameterVector
    num_params = num_qubits * (depth + 1)
    theta = ParameterVector("theta", num_params)
    qc = QuantumCircuit(num_qubits, name="VQE_Ansatz")
    
    param_idx = 0
    # Initial rotation layer
    for q in range(num_qubits):
        qc.ry(theta[param_idx], q)
        param_idx += 1
        
    for _ in range(depth):
        # Entanglement layer
        if num_qubits > 1:
            if entanglement == "linear":
                for q in range(num_qubits - 1):
                    qc.cx(q, q + 1)
            elif entanglement == "full":
                for q1 in range(num_qubits):
                    for q2 in range(q1 + 1, num_qubits):
                        qc.cx(q1, q2)
                        
        # Variational rotation layer
        for q in range(num_qubits):
            qc.ry(theta[param_idx], q)
            param_idx += 1
            
    return qc


def profile_parameter_binding(
    circuit: QuantumCircuit, 
    parameter_values: Optional[List[float]] = None, 
    runs: int = 10
) -> Dict[str, Any]:
    """
    Benchmark parameter binding latency and throughput for variational circuits.
    
    Args:
        circuit (QuantumCircuit): Parameterized quantum circuit.
        parameter_values (Optional[List[float]]): Numerical values to bind.
        runs (int): Number of repetition trials.
        
    Returns:
        dict: Binding statistics (mean_latency_ms, std_latency_ms, throughput_bindings_per_sec).
    """
    import time
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("circuit must be a QuantumCircuit instance.")
    if not isinstance(runs, int) or runs < 1:
        raise ValueError("runs must be an integer >= 1.")
        
    param_count = circuit.num_parameters
    if param_count == 0:
        return {
            "num_parameters": 0,
            "mean_binding_ms": 0.0,
            "std_binding_ms": 0.0,
            "throughput_bindings_per_sec": 0.0,
            "runs": runs
        }
        
    if parameter_values is None:
        values = list(np.random.uniform(0.0, 2.0 * np.pi, size=param_count))
    else:
        if len(parameter_values) != param_count:
            raise ValueError(f"Expected {param_count} parameters, got {len(parameter_values)}.")
        values = list(parameter_values)
        
    latencies = []
    # Warmup
    circuit.assign_parameters(values)
    
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = circuit.assign_parameters(values)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms
        
    arr = np.array(latencies)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr))
    throughput = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0
    
    return {
        "num_parameters": param_count,
        "mean_binding_ms": round(mean_ms, 4),
        "std_binding_ms": round(std_ms, 4),
        "min_binding_ms": round(float(np.min(arr)), 4),
        "max_binding_ms": round(float(np.max(arr)), 4),
        "throughput_bindings_per_sec": round(throughput, 2),
        "runs": runs
    }


def generate_qaoa_circuit(
    num_qubits: int, 
    p_steps: int = 1, 
    cost_graph: Optional[List[Tuple[int, int]]] = None
) -> QuantumCircuit:
    """
    Generate a Quantum Approximate Optimization Algorithm (QAOA) ansatz for Max-Cut problem.
    
    Args:
        num_qubits (int): Number of nodes/qubits.
        p_steps (int): Number of QAOA alternating layers (p).
        cost_graph (Optional[List[Tuple[int, int]]]): List of edges (u, v). If None, generates linear chain graph.
        
    Returns:
        QuantumCircuit: A parameterized QAOA ansatz with gamma and beta parameter vectors.
    """
    if not isinstance(num_qubits, int) or not isinstance(p_steps, int):
        raise TypeError("num_qubits and p_steps must be integers.")
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1.")
    if p_steps < 1:
        raise ValueError("p_steps must be at least 1.")
        
    if cost_graph is None:
        cost_graph = [(i, (i + 1) % num_qubits) for i in range(num_qubits - 1)] if num_qubits > 1 else []
        
    from qiskit.circuit import ParameterVector
    gamma = ParameterVector("gamma", p_steps)
    beta = ParameterVector("beta", p_steps)
    
    qc = QuantumCircuit(num_qubits, name="QAOA_Ansatz")
    
    # Initial equal superposition state
    for q in range(num_qubits):
        qc.h(q)
        
    for p in range(p_steps):
        # 1. Cost Hamiltonian layer: exp(-i * gamma * C)
        for u, v in cost_graph:
            if u < 0 or u >= num_qubits or v < 0 or v >= num_qubits:
                raise ValueError(f"Invalid edge ({u}, {v}) for {num_qubits} qubits.")
            qc.cx(u, v)
            qc.rz(2 * gamma[p], v)
            qc.cx(u, v)
            
        # 2. Mixer Hamiltonian layer: exp(-i * beta * B)
        for q in range(num_qubits):
            qc.rx(2 * beta[p], q)
            
    return qc


def generate_quantum_volume_circuit(
    num_qubits: int, 
    depth: Optional[int] = None, 
    seed: int = 42
) -> QuantumCircuit:
    """
    Generate a square Quantum Volume (QV) model circuit according to Cross et al. (2019).
    
    Composed of d = depth (default: num_qubits) layers of random pairwise SU(4) Haar unitaries
    applied to random permutations of the active qubits.
    
    Args:
        num_qubits (int): Number of active qubits in the circuit.
        depth (Optional[int]): Number of layers (default: num_qubits for standard square QV circuit).
        seed (int): Random seed for reproducibility.
        
    Returns:
        QuantumCircuit: Quantum Volume model circuit.
    """
    if not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer.")
    if num_qubits < 2:
        raise ValueError("Quantum Volume requires at least 2 qubits.")
        
    d = depth if depth is not None else num_qubits
    if not isinstance(d, int) or d < 1:
        raise ValueError("depth must be an integer >= 1.")
        
    from qiskit.quantum_info import random_unitary
    
    rng = random.Random(seed)
    qc = QuantumCircuit(num_qubits, name=f"QV_{num_qubits}x{d}")
    
    for layer in range(d):
        perm = list(range(num_qubits))
        rng.shuffle(perm)
        
        # Pair adjacent qubits in the random permutation
        for i in range(0, num_qubits - 1, 2):
            q_a = perm[i]
            q_b = perm[i + 1]
            u_seed = seed + (layer * 1000) + i
            su4_unitary = random_unitary(4, seed=u_seed)
            qc.append(su4_unitary, [q_a, q_b])
            
    return qc


def calculate_heavy_output_probability(
    ideal_probs: Dict[str, float], 
    sampled_counts: Dict[str, int]
) -> Dict[str, Any]:
    """
    Compute heavy output generation probability h_prob to certify Quantum Volume (Cross et al. 2019).
    
    Outputs with ideal probability greater than the median ideal probability are classified
    as 'heavy'. A quantum processor successfully demonstrates capability at depth d if
    h_prob > 2/3 with > 97.7% confidence (h_prob - 2 * sigma > 2/3).
    
    Args:
        ideal_probs (Dict[str, float]): Theoretical probability distribution.
        sampled_counts (Dict[str, int]): Empirical measurement counts.
        
    Returns:
        dict: Analysis including h_prob, standard error, and quantum volume certification result.
    """
    if not ideal_probs or not sampled_counts:
        raise ValueError("ideal_probs and sampled_counts must not be empty.")
        
    prob_values = list(ideal_probs.values())
    p_med = float(np.median(prob_values))
    
    heavy_bitstrings = {bs for bs, p in ideal_probs.items() if p > p_med}
    
    total_shots = sum(sampled_counts.values())
    if total_shots == 0:
        raise ValueError("Total sampled shots must be greater than 0.")
        
    heavy_shots = sum(cnt for bs, cnt in sampled_counts.items() if bs in heavy_bitstrings)
    h_prob = float(heavy_shots / total_shots)
    
    # Binomial standard error
    sigma = float(np.sqrt(max(0.0, h_prob * (1.0 - h_prob) / total_shots)))
    confidence_bound = float(h_prob - 2 * sigma)
    qv_success = bool(confidence_bound > (2.0 / 3.0))
    
    return {
        "median_ideal_probability": round(p_med, 6),
        "heavy_bitstrings_count": len(heavy_bitstrings),
        "total_heavy_shots": heavy_shots,
        "total_shots": total_shots,
        "heavy_output_probability": round(h_prob, 4),
        "standard_error": round(sigma, 4),
        "lower_confidence_bound_2sigma": round(confidence_bound, 4),
        "qv_threshold": round(2.0 / 3.0, 4),
        "qv_certified": qv_success
    }
