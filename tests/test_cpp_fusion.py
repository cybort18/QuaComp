import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Barrier
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import HGate, XGate, ZGate, SGate, CXGate, CZGate, RXGate, RZGate

from src.engine.fusion import (
    is_cpp_fusion_available,
    get_fusion_backend,
    py_matmul_2x2,
    py_matmul_4x4,
    py_fuse_matrices,
    fuse_matrix_sequence,
    extract_gate_matrix,
    fuse_circuit_single_pass
)


def test_fusion_backend_status():
    """Verify backend identifier string and boolean availability function."""
    avail = is_cpp_fusion_available()
    backend = get_fusion_backend()
    assert isinstance(avail, bool)
    assert isinstance(backend, str)
    if avail:
        assert backend == "C++ Native Engine"
    else:
        assert backend == "CPython Fallback Engine"


def test_py_matmul_2x2_accuracy():
    """Verify 2x2 complex matrix multiplication against numpy standard."""
    H = Operator(HGate()).data
    X = Operator(XGate()).data
    expected = np.matmul(H, X)
    result = py_matmul_2x2(H, X)
    np.testing.assert_allclose(result, expected, atol=1e-14)


def test_py_matmul_4x4_accuracy():
    """Verify 4x4 complex matrix multiplication against numpy standard."""
    CX = Operator(CXGate()).data
    CZ = Operator(CZGate()).data
    expected = np.matmul(CZ, CX)
    result = py_matmul_4x4(CZ, CX)
    np.testing.assert_allclose(result, expected, atol=1e-14)


def test_fuse_matrices_mathematical_identity():
    """Verify sequential fusion U_fused = U_k * ... * U_1 reproduces H * X * H = Z."""
    H = Operator(HGate()).data
    X = Operator(XGate()).data
    Z = Operator(ZGate()).data
    
    # Chronological application order: H first, then X, then H
    matrices = [H, X, H]
    
    # Python fallback
    fused_py = py_fuse_matrices(matrices)
    np.testing.assert_allclose(fused_py, Z, atol=1e-14)
    
    # General fuse_matrix_sequence
    fused_gen = fuse_matrix_sequence(matrices, force_backend="python")
    np.testing.assert_allclose(fused_gen, Z, atol=1e-14)


def test_fuse_matrix_sequence_dimension_mismatch():
    """Verify error handling on matrix dimension mismatch or empty list."""
    with pytest.raises(ValueError):
        py_fuse_matrices([])
        
    H = Operator(HGate()).data
    CX = Operator(CXGate()).data
    with pytest.raises(ValueError):
        py_fuse_matrices([H, CX])


def test_extract_gate_matrix():
    """Verify unitary extraction for standard gates and non-unitary handling."""
    h_mat = extract_gate_matrix(HGate())
    assert h_mat is not None
    assert h_mat.shape == (2, 2)
    
    cx_mat = extract_gate_matrix(CXGate())
    assert cx_mat is not None
    assert cx_mat.shape == (4, 4)
    
    barrier_mat = extract_gate_matrix(Barrier(1))
    assert barrier_mat is None


def test_fuse_circuit_single_pass_1q():
    """Verify single-pass fusion on consecutive 1-qubit gates and statevector fidelity."""
    qc = QuantumCircuit(2)
    # Consecutive gates on qubit 0: H, X, H (should fuse into Z)
    qc.h(0)
    qc.x(0)
    qc.h(0)
    # Consecutive gates on qubit 1: S, S (should fuse into Z)
    qc.s(1)
    qc.s(1)
    
    fused_qc, metrics = fuse_circuit_single_pass(qc)
    
    assert metrics["original_gate_count"] == 5
    assert metrics["fused_gate_count"] == 2
    assert metrics["reduction_ratio"] == 0.6
    assert metrics["fused_blocks_count"] == 2
    assert metrics["backend"] in ("C++ Native Engine", "CPython Fallback Engine")
    
    # Verify exact statevector equivalence
    sv_orig = Statevector.from_instruction(qc)
    sv_fused = Statevector.from_instruction(fused_qc)
    assert sv_orig.equiv(sv_fused)


def test_fuse_circuit_single_pass_2q():
    """Verify single-pass fusion on consecutive 2-qubit gates on the same pair."""
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cz(0, 1)
    
    fused_qc, metrics = fuse_circuit_single_pass(qc)
    
    assert metrics["original_gate_count"] == 2
    assert metrics["fused_gate_count"] == 1
    assert metrics["reduction_ratio"] == 0.5
    assert metrics["fused_blocks_count"] == 1
    
    # Verify statevector equivalence
    sv_orig = Statevector.from_instruction(qc)
    sv_fused = Statevector.from_instruction(fused_qc)
    assert sv_orig.equiv(sv_fused)


def test_fuse_circuit_with_barriers():
    """Verify barriers act as strict fusion boundaries and are preserved."""
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.x(0)
    qc.barrier(0)
    qc.h(0)
    qc.x(0)
    
    fused_qc, metrics = fuse_circuit_single_pass(qc)
    
    # 2 gates before barrier -> 1 fused gate
    # 1 barrier -> 1 barrier
    # 2 gates after barrier -> 1 fused gate
    # Total = 3 gates
    assert metrics["original_gate_count"] == 5
    assert metrics["fused_gate_count"] == 3
    assert metrics["fused_blocks_count"] == 2
    
    sv_orig = Statevector.from_instruction(qc)
    sv_fused = Statevector.from_instruction(fused_qc)
    assert sv_orig.equiv(sv_fused)


def test_fuse_circuit_type_error():
    """Verify TypeError on invalid non-circuit input."""
    with pytest.raises(TypeError):
        fuse_circuit_single_pass("not a circuit")  # type: ignore


def test_fuse_circuit_single_gate_no_op():
    """Verify single gate passes through without redundant unitary wrapping."""
    qc = QuantumCircuit(1)
    qc.h(0)
    fused_qc, metrics = fuse_circuit_single_pass(qc)
    assert metrics["original_gate_count"] == 1
    assert metrics["fused_gate_count"] == 1
    assert metrics["reduction_ratio"] == 0.0
    assert metrics["fused_blocks_count"] == 0
    assert fused_qc.data[0].operation.name == "h"
