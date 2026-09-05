import pytest
import numpy as np
from qiskit import QuantumCircuit
from src.engine.circuits import (
    generate_vqe_circuit,
    profile_parameter_binding,
    generate_qaoa_circuit,
    generate_quantum_volume_circuit,
    calculate_heavy_output_probability
)

def test_generate_vqe_circuit():
    """Verify VQE ansatz structure, parameter counts, and entanglement modes."""
    # 4 qubits, depth 2 -> 4 * (2 + 1) = 12 parameters
    qc_linear = generate_vqe_circuit(4, depth=2, entanglement="linear")
    assert qc_linear.num_qubits == 4
    assert qc_linear.num_parameters == 12
    assert qc_linear.name == "VQE_Ansatz"
    
    # Full entanglement mode
    qc_full = generate_vqe_circuit(4, depth=1, entanglement="full")
    assert qc_full.num_qubits == 4
    assert qc_full.num_parameters == 8
    
    # Argument validation
    with pytest.raises(ValueError):
        generate_vqe_circuit(0)
    with pytest.raises(ValueError):
        generate_vqe_circuit(4, depth=0)
    with pytest.raises(ValueError):
        generate_vqe_circuit(4, entanglement="invalid")

def test_profile_parameter_binding():
    """Verify parameter binding latency and throughput profiling."""
    qc = generate_vqe_circuit(3, depth=1)  # 3 * 2 = 6 parameters
    res = profile_parameter_binding(qc, runs=5)
    
    assert res["num_parameters"] == 6
    assert res["mean_binding_ms"] >= 0.0
    assert res["throughput_bindings_per_sec"] > 0.0
    assert res["runs"] == 5
    
    # Test zero parameters circuit edge case
    qc_empty = QuantumCircuit(2)
    empty_res = profile_parameter_binding(qc_empty)
    assert empty_res["num_parameters"] == 0
    assert empty_res["throughput_bindings_per_sec"] == 0.0

def test_generate_qaoa_circuit():
    """Verify QAOA ansatz creation with gamma and beta parameter vectors."""
    # 4 qubits, 2 steps -> 2 gammas + 2 betas = 4 parameters
    qc = generate_qaoa_circuit(4, p_steps=2)
    assert qc.num_qubits == 4
    assert qc.num_parameters == 4
    assert qc.name == "QAOA_Ansatz"
    
    # Custom graph
    custom_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    qc_custom = generate_qaoa_circuit(4, p_steps=1, cost_graph=custom_edges)
    assert qc_custom.num_parameters == 2
    
    with pytest.raises(ValueError):
        generate_qaoa_circuit(4, p_steps=1, cost_graph=[(0, 10)])

def test_generate_quantum_volume_circuit():
    """Verify Quantum Volume circuit generation with Haar random SU(4) gates."""
    qc_qv = generate_quantum_volume_circuit(4, depth=4, seed=123)
    assert qc_qv.num_qubits == 4
    assert qc_qv.name == "QV_4x4"
    
    # Must reject num_qubits < 2
    with pytest.raises(ValueError):
        generate_quantum_volume_circuit(1)
        
    with pytest.raises(ValueError):
        generate_quantum_volume_circuit(4, depth=0)

def test_calculate_heavy_output_probability():
    """Verify calculation of heavy output probability and QV threshold certification."""
    # Toy ideal probability distribution: 4 outcomes
    ideal_probs = {
        "00": 0.50,  # Heavy (above median)
        "01": 0.30,  # Heavy (above median)
        "10": 0.15,
        "11": 0.05
    }
    
    # High fidelity sampling where mostly heavy outputs are measured
    sampled_counts_success = {
        "00": 800,
        "01": 150,
        "10": 30,
        "11": 20
    }
    # Heavy shots = 800 + 150 = 950 / 1000 = 0.95 > 2/3
    res_success = calculate_heavy_output_probability(ideal_probs, sampled_counts_success)
    assert res_success["total_shots"] == 1000
    assert res_success["heavy_output_probability"] == 0.95
    assert res_success["qv_certified"] is True
    assert res_success["qv_threshold"] == pytest.approx(0.6667, 0.001)
    
    # Uniform / low fidelity sampling
    sampled_counts_fail = {
        "00": 250,
        "01": 250,
        "10": 250,
        "11": 250
    }
    res_fail = calculate_heavy_output_probability(ideal_probs, sampled_counts_fail)
    assert res_fail["heavy_output_probability"] == 0.5
    assert res_fail["qv_certified"] is False
