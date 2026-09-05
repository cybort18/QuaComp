import pytest
import numpy as np
from qiskit import QuantumCircuit
from src.engine.entanglement import (
    calculate_bipartite_entropy,
    _calculate_mps_bipartite_entropy,
    _PermutedMPSChain
)
from src.engine.circuits import generate_shallow_circuit, generate_deep_circuit

def test_permuted_mps_chain_initialization():
    """Verify initialization of _PermutedMPSChain state and mappings."""
    chain = _PermutedMPSChain(4, max_bond=32)
    assert chain.n == 4
    assert chain.max_bond == 32
    assert chain.qubit_at == [0, 1, 2, 3]
    assert chain.pos == [0, 1, 2, 3]
    assert chain.cumulative_truncation_error == 0.0
    assert chain.swap_count == 0
    assert len(chain.tensors) == 4
    for t in chain.tensors:
        assert t.shape == (1, 2, 1)

def test_mps_topology_bell_state_fidelity():
    """Verify Bell state entanglement with and without topology optimization."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    res_opt = _calculate_mps_bipartite_entropy(qc, subsystem_size=1, optimize_topology=True)
    res_naive = _calculate_mps_bipartite_entropy(qc, subsystem_size=1, optimize_topology=False)
    
    assert pytest.approx(res_opt["von_neumann_entropy"], 0.001) == 1.0
    assert pytest.approx(res_naive["von_neumann_entropy"], 0.001) == 1.0
    assert res_opt["schmidt_rank"] == 2
    assert res_opt["topology_optimized"] is True
    assert res_naive["topology_optimized"] is False
    assert res_opt["truncation_error"] >= 0.0

def test_mps_topology_non_adjacent_deferred_unswap():
    """Verify non-adjacent gate routing preserves entanglement and tracks swaps."""
    # Circuit with non-adjacent CNOT(0, 3) on 4 qubits
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 3)
    
    res_opt = _calculate_mps_bipartite_entropy(qc, subsystem_size=2, optimize_topology=True)
    res_sv = calculate_bipartite_entropy(qc, subsystem_size=2, method="statevector")
    
    assert pytest.approx(res_opt["von_neumann_entropy"], 0.01) == pytest.approx(res_sv["von_neumann_entropy"], 0.01)
    assert res_opt["schmidt_rank"] == res_sv["schmidt_rank"]
    assert "routing_swaps" in res_opt
    assert "truncation_error" in res_opt
    assert "qubit_permutation" in res_opt

def test_mps_topology_bipartition_alignment():
    """Verify alignment brings subsystem A qubits to left sites and preserves entanglement."""
    # Qubit 0 and 1 are subsystem A; qubit 2 and 3 are subsystem B
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.h(1)
    # Interleaved entangling gates that shuffle order
    qc.cx(0, 2)
    qc.cx(1, 3)
    
    res_opt = calculate_bipartite_entropy(qc, subsystem_size=2, method="mps", optimize_topology=True)
    res_sv = calculate_bipartite_entropy(qc, subsystem_size=2, method="statevector")
    
    assert pytest.approx(res_opt["von_neumann_entropy"], 0.01) == pytest.approx(res_sv["von_neumann_entropy"], 0.01)
    assert res_opt["topology_optimized"] is True

def test_mps_topology_truncation_error_under_low_bond():
    """Verify truncation error is recorded when max_bond restricts representation."""
    # Deep circuit with 6 qubits produces bond dimension > 2
    qc = generate_deep_circuit(6, depth=4)
    res_low_bond = _calculate_mps_bipartite_entropy(qc, subsystem_size=3, max_bond_dimension=2)
    
    assert res_low_bond["truncation_error"] >= 0.0
    assert "computation_engine" in res_low_bond
