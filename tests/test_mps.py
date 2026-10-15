import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from src.engine.mps import (
    MatrixProductState,
    simulate_mps,
    get_mps_configuration,
    calculate_mps_ram_savings
)
from src.engine.circuits import generate_shallow_circuit, generate_qft_circuit
from src.engine.simulator import run_simulation
from src.profiler.memory import check_memory_safety

def test_get_mps_configuration():
    config = get_mps_configuration(128)
    assert isinstance(config, dict)
    assert config["matrix_product_state_max_bond_dimension"] == 128

    # Test errors
    with pytest.raises(TypeError):
        get_mps_configuration("128")  # type: ignore
    with pytest.raises(ValueError):
        get_mps_configuration(0)
    with pytest.raises(ValueError):
        get_mps_configuration(-10)

def test_calculate_mps_ram_savings():
    # 10 qubits Statevector RAM est: 2^10 * 16 = 16384 bytes
    # actual ram: 4096 bytes
    # savings: 16384 - 4096 = 12288 bytes (75%)
    res = calculate_mps_ram_savings(10, 4096)
    assert res["statevector_ram_est_bytes"] == 16384
    assert res["actual_ram_used_bytes"] == 4096.0
    assert res["savings_bytes"] == 12288.0
    assert res["savings_percent"] == 75.0

    # Test zero qubits case
    res_zero = calculate_mps_ram_savings(0, 10)
    assert res_zero["statevector_ram_est_bytes"] == 16
    assert res_zero["savings_bytes"] == 6.0

    # Test errors
    with pytest.raises(TypeError):
        calculate_mps_ram_savings("10", 4096)  # type: ignore
    with pytest.raises(TypeError):
        calculate_mps_ram_savings(10, "4096")  # type: ignore
    with pytest.raises(ValueError):
        calculate_mps_ram_savings(-5, 4096)
    with pytest.raises(ValueError):
        calculate_mps_ram_savings(10, -100)

def test_check_memory_safety_mps():
    # 40 qubits would require 2^40 * 16 bytes = 17.5 Terabytes for statevector.
    # It must fail safety check in statevector mode.
    is_safe, msg = check_memory_safety(40, method="statevector")
    assert is_safe is False
    assert "CRITICAL" in msg or "WARNING" in msg

    # Under mps mode, it should be marked as safe because it bypasses exponential RAM checks.
    is_safe_mps, msg_mps = check_memory_safety(40, method="mps")
    assert is_safe_mps is True
    assert "SAFE" in msg_mps
    assert "bypassed" in msg_mps.lower()

def test_run_simulation_mps_30_qubits():
    # Generate 30 qubits shallow circuit
    # (Hadamard on all qubits + CNOT chain)
    circuit = generate_shallow_circuit(30)

    # Run simulation with MPS method
    result = run_simulation(circuit, method="mps", bond_dimension=64)

    assert isinstance(result, dict)
    assert result["success"] is True, f"MPS Simulation failed: {result.get('error')}"
    assert result["error"] is None
    assert result["latency"] > 0
    assert "metadata" in result
    assert result["metadata"]["method"] in ("mps", "matrix_product_state")
    assert result["metadata"]["bond_dimension"] == 64
    assert "aer_simulator" in result["metadata"]["backend_name"].lower()


def test_mps_long_range_cnot_exact_probability_distribution():
    """
    Test long-range 2-qubit gate CX(0, 3) on a 4-qubit MPS and verify that the
    SWAP network routing produces an identical probability distribution to the
    exact statevector (max |P_MPS - P_exact| < 1e-6).
    """
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.x(1)
    qc.cx(0, 3)  # Long-range non-adjacent 2-qubit gate (|q0 - q1| = 3 > 1)

    res = simulate_mps(qc, bond_dimension=64, cutoff=1e-12, shots=2048, seed=42)
    assert res["success"] is True
    assert res["qubits"] == 4
    assert res["swap_count"] == 4  # 2 forward SWAPs, 2 reverse SWAPs
    assert res["mps_fidelity"] >= 0.999999
    assert res["cumulative_truncation_error"] <= 1e-6

    # Verify exact statevector distribution
    sv_exact = np.array(Statevector.from_instruction(qc))
    p_exact = np.abs(sv_exact) ** 2
    p_mps = res["probabilities"]

    max_prob_diff = float(np.max(np.abs(p_mps - p_exact)))
    assert max_prob_diff < 1e-6, f"Max probability difference {max_prob_diff} exceeds 1e-6 threshold."

    # Verify reverse direction long-range gate: CX(3, 0)
    qc_rev = QuantumCircuit(4)
    qc_rev.h(3)
    qc_rev.x(2)
    qc_rev.cx(3, 0)
    res_rev = simulate_mps(qc_rev, bond_dimension=64, cutoff=1e-12, shots=1024, seed=42)
    sv_rev_exact = np.array(Statevector.from_instruction(qc_rev))
    p_rev_exact = np.abs(sv_rev_exact) ** 2
    max_diff_rev = float(np.max(np.abs(res_rev["probabilities"] - p_rev_exact)))
    assert max_diff_rev < 1e-6
    assert res_rev["swap_count"] == 4


def test_mps_bond_dimension_one_truncation_on_entangled_states():
    """
    Test that when max_bond_dim = 1 (tensor product without entanglement) is applied
    to Bell and GHZ states, cumulative_truncation_error is strictly > 0 and fidelity < 1.0.
    """
    # Bell State
    bell_qc = QuantumCircuit(2)
    bell_qc.h(0)
    bell_qc.cx(0, 1)

    res_bell = simulate_mps(bell_qc, bond_dimension=1, cutoff=1e-12)
    assert res_bell["cumulative_truncation_error"] > 0.0
    assert res_bell["mps_fidelity"] < 1.0
    # For Bell state (|00> + |11>) / sqrt(2), singular values are (1/sqrt(2), 1/sqrt(2)).
    # Truncating from 2 down to 1 discards 0.5 spectral weight, giving fidelity = 0.5.
    assert pytest.approx(res_bell["mps_fidelity"], abs=0.01) == 0.5
    assert pytest.approx(res_bell["cumulative_truncation_error"], abs=0.01) == 0.5
    assert res_bell["max_active_bond_dim"] == 1

    # GHZ State
    ghz_qc = QuantumCircuit(3)
    ghz_qc.h(0)
    ghz_qc.cx(0, 1)
    ghz_qc.cx(1, 2)

    res_ghz = simulate_mps(ghz_qc, bond_dimension=1, cutoff=1e-12)
    assert res_ghz["cumulative_truncation_error"] > 0.0
    assert res_ghz["mps_fidelity"] < 1.0


def test_mps_qasm_execution_qft_4qubit():
    """
    Test executing the benchmarks/qft_4qubit.qasm OpenQASM benchmark using the MPS engine.
    """
    res = simulate_mps("benchmarks/qft_4qubit.qasm", bond_dimension=64, cutoff=1e-12, shots=1024, seed=123)
    assert res["success"] is True
    assert res["qubits"] == 4
    assert res["method"] == "mps"
    assert res["bond_dimension"] == 64
    assert res["latency"] > 0.0
    assert sum(res["counts"].values()) == 1024
    assert res["mps_fidelity"] > 0.99
    assert res["cumulative_truncation_error"] < 0.01
    assert res["swap_count"] > 0


def test_mps_input_validation():
    """
    Test input validation and error branches for MatrixProductState and simulate_mps.
    """
    with pytest.raises(TypeError):
        MatrixProductState("4")  # type: ignore
    with pytest.raises(ValueError):
        MatrixProductState(0)
    with pytest.raises(TypeError):
        MatrixProductState(4, max_bond_dim="16")  # type: ignore
    with pytest.raises(ValueError):
        MatrixProductState(4, max_bond_dim=0)
    with pytest.raises(TypeError):
        MatrixProductState(4, cutoff="invalid")  # type: ignore
    with pytest.raises(ValueError):
        MatrixProductState(4, cutoff=-0.5)

    mps = MatrixProductState(3)
    with pytest.raises(IndexError):
        mps.apply_1q(3, np.eye(2))
    with pytest.raises(ValueError):
        mps.apply_1q(0, np.eye(3))
    with pytest.raises(ValueError):
        mps.apply_2q(0, 0, np.eye(4))
    with pytest.raises(IndexError):
        mps.apply_2q(0, 4, np.eye(4))
    with pytest.raises(ValueError):
        mps.apply_2q_adjacent(0, 2, np.eye(4))
    with pytest.raises(ValueError):
        mps.apply_2q_adjacent(0, 1, np.eye(2))

    with pytest.raises(TypeError):
        simulate_mps(12345)
