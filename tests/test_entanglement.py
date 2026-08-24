import pytest
from qiskit import QuantumCircuit
from src.engine.entanglement import calculate_bipartite_entropy
from src.engine.circuits import generate_shallow_circuit, generate_deep_circuit, generate_qft_circuit

def test_calculate_bipartite_entropy_product_state():
    """Verify an unentangled product state has zero Von Neumann entropy."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.h(1)
    qc.x(2)
    # No entangling two-qubit gates
    metrics = calculate_bipartite_entropy(qc)
    assert metrics["num_qubits"] == 4
    assert metrics["subsystem_a_size"] == 2
    assert metrics["von_neumann_entropy"] == 0.0
    assert metrics["schmidt_rank"] == 1
    assert metrics["entanglement_regime"] == "Product State"
    assert metrics["mps_hardness"] == "Trivial (chi=1)"

def test_calculate_bipartite_entropy_bell_state():
    """Verify maximally entangled 2-qubit Bell state has S_vN = 1.0 and Schmidt rank = 2."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    metrics = calculate_bipartite_entropy(qc)
    assert metrics["num_qubits"] == 2
    assert metrics["subsystem_a_size"] == 1
    assert pytest.approx(metrics["von_neumann_entropy"], 0.001) == 1.0
    assert metrics["schmidt_rank"] == 2
    assert metrics["entanglement_ratio"] == 1.0
    assert metrics["entanglement_regime"] == "Low (Area-law)"

def test_calculate_bipartite_entropy_ghz_state():
    """Verify 4-qubit GHZ state has S_vN = 1.0 regardless of equal bipartition size."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)
    
    metrics = calculate_bipartite_entropy(qc, subsystem_size=2)
    assert metrics["num_qubits"] == 4
    assert metrics["subsystem_a_size"] == 2
    assert pytest.approx(metrics["von_neumann_entropy"], 0.001) == 1.0
    assert metrics["schmidt_rank"] == 2

def test_calculate_bipartite_entropy_deep_circuit():
    """Verify Deep random circuits generate high volume-law entanglement entropy."""
    qc = generate_deep_circuit(6, depth=6)
    metrics = calculate_bipartite_entropy(qc)
    assert metrics["num_qubits"] == 6
    assert metrics["subsystem_a_size"] == 3
    assert metrics["von_neumann_entropy"] > 1.5
    assert metrics["schmidt_rank"] > 2
    assert metrics["entanglement_regime"] in ("Moderate Entanglement", "Volume-law (Maximal)")

def test_calculate_bipartite_entropy_qft():
    """Verify QFT on all-zero state generates zero bipartite entanglement."""
    qc = generate_qft_circuit(6)
    metrics = calculate_bipartite_entropy(qc)
    assert metrics["num_qubits"] == 6
    assert metrics["von_neumann_entropy"] == 0.0
    assert metrics["schmidt_rank"] == 1
    assert metrics["entanglement_regime"] == "Product State"

def test_calculate_bipartite_entropy_single_qubit():
    """Verify single qubit edge case is handled cleanly without errors."""
    qc = QuantumCircuit(1)
    qc.h(0)
    metrics = calculate_bipartite_entropy(qc)
    assert metrics["num_qubits"] == 1
    assert metrics["von_neumann_entropy"] == 0.0
    assert metrics["schmidt_rank"] == 1

def test_calculate_bipartite_entropy_input_validation():
    """Verify invalid argument types and subsystem sizes raise appropriate exceptions."""
    with pytest.raises(TypeError):
        calculate_bipartite_entropy("not_a_circuit")
        
    qc = QuantumCircuit(4)
    with pytest.raises(TypeError):
        calculate_bipartite_entropy(qc, subsystem_size="2")
        
    with pytest.raises(ValueError):
        calculate_bipartite_entropy(qc, subsystem_size=4)
        
    with pytest.raises(ValueError):
        calculate_bipartite_entropy(qc, subsystem_size=0)
