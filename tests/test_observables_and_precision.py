"""
Unit tests for Solver-Aware Roofline Telemetry, Mixed-Precision Emulation (FP32/FP64),
Multi-Paradigm Pauli Expectation Engine, and Scalable Entanglement Entropy.
"""
import math
import os
import numpy as np
import pytest
from qiskit import QuantumCircuit

from src.profiler.telemetry import calculate_effective_memory_bandwidth
from src.scorer.calculator import calculate_scoring_breakdown
from src.engine.mps import MatrixProductState, simulate_mps
from src.engine.stabilizer import StabilizerTableau, simulate_stabilizer
from src.engine.simulator import (
    evolve_statevector,
    evaluate_precision_drift,
    simulate_trajectories,
    run_simulation,
)
from src.engine.observables import (
    expectation_stabilizer,
    expectation_mps,
    expectation_statevector,
    evaluate_pauli_expectation,
    evaluate_hamiltonian,
)
from src.engine.entanglement import (
    calculate_mps_entanglement_entropy,
    calculate_stabilizer_entanglement_entropy,
    calculate_bipartite_entropy,
)
from src.engine.parser import load_qasm


class TestSolverAwareBandwidthAndScoring:
    """Verify solver-aware and precision-aware Roofline memory bandwidth telemetry."""

    def test_100_qubit_stabilizer_bandwidth_no_overflow(self):
        """100-qubit stabilizer simulation must not use 2^100 statevector scaling."""
        res = calculate_effective_memory_bandwidth(
            num_qubits=100,
            total_gates=100,
            duration_sec=0.01,
            sim_method="stabilizer",
            precision="fp64",
        )
        assert res["effective_bandwidth_gbps"] > 0.0
        assert res["effective_bandwidth_gbps"] < 1.0  # ~8 KB in 10 ms is ~0.0008 GB/s, not 10^24 GB/s
        assert res["arithmetic_intensity"] == pytest.approx(0.75)

        score = calculate_scoring_breakdown(
            max_qubits=100,
            total_gates=100,
            execution_time=0.01,
            sim_method="stabilizer",
            precision="fp64",
        )
        assert score["effective_bandwidth_gbps"] < 1.0
        assert score["arithmetic_intensity_flops_per_byte"] == pytest.approx(0.75)

    def test_statevector_and_mps_precision_scaling(self):
        """Verify FP32 halves bytes-per-gate and doubles arithmetic intensity on statevector/MPS."""
        sv_fp64 = calculate_effective_memory_bandwidth(
            num_qubits=20,
            total_gates=50,
            duration_sec=0.1,
            sim_method="statevector",
            precision="fp64",
        )
        sv_fp32 = calculate_effective_memory_bandwidth(
            num_qubits=20,
            total_gates=50,
            duration_sec=0.1,
            sim_method="statevector",
            precision="fp32",
        )
        assert sv_fp64["total_bytes_transferred"] == 2 * sv_fp32["total_bytes_transferred"]
        assert sv_fp64["arithmetic_intensity"] == pytest.approx(0.4375)
        assert sv_fp32["arithmetic_intensity"] == pytest.approx(0.875)

        mps_fp64 = calculate_effective_memory_bandwidth(
            num_qubits=50,
            total_gates=100,
            duration_sec=0.05,
            sim_method="mps",
            precision="fp64",
            active_bond_dim=16,
        )
        mps_fp32 = calculate_effective_memory_bandwidth(
            num_qubits=50,
            total_gates=100,
            duration_sec=0.05,
            sim_method="mps",
            precision="fp32",
            active_bond_dim=16,
        )
        assert mps_fp64["total_bytes_transferred"] == 2 * mps_fp32["total_bytes_transferred"]
        assert mps_fp64["arithmetic_intensity"] == pytest.approx(2.0)
        assert mps_fp32["arithmetic_intensity"] == pytest.approx(4.0)


class TestThreeSolverPauliExpectationParity:
    """Verify exact parity across Statevector, MPS, and Stabilizer Pauli expectation engines."""

    def test_bell_state_three_solver_parity(self):
        """Bell state (|00> + |11>)/sqrt(2) must yield identical expectations on all 3 solvers."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        # 1. Statevector
        sv = evolve_statevector(qc, precision="fp64")

        # 2. MPS
        mps = simulate_mps(qc, max_bond_dim=8, precision="fp64")["mps_state"]

        # 3. Stabilizer Tableau
        tab = StabilizerTableau(2)
        tab.h(0)
        tab.cx(0, 1)

        expected_paulis = {
            "ZZ": 1.0,
            "XX": 1.0,
            "YY": -1.0,
            "ZI": 0.0,
            "IZ": 0.0,
            "XI": 0.0,
            "IX": 0.0,
            "XZ": 0.0,
            "II": 1.0,
        }

        for pauli_str, expected_val in expected_paulis.items():
            val_sv = expectation_statevector(sv, pauli_str)
            val_mps = expectation_mps(mps, pauli_str)
            val_tab = expectation_stabilizer(tab, pauli_str)

            assert val_sv == pytest.approx(expected_val, abs=1e-7), f"SV mismatch for {pauli_str}"
            assert val_mps == pytest.approx(expected_val, abs=1e-7), f"MPS mismatch for {pauli_str}"
            assert val_tab == pytest.approx(expected_val, abs=1e-7), f"Stabilizer mismatch for {pauli_str}"

        # Test unified Hamiltonian evaluator H = 0.5 * ZZ + 0.5 * XX - 0.5 * YY + 0.25 * ZI
        hamiltonian = [
            (0.5, "ZZ"),
            (0.5, "XX"),
            (-0.5, "YY"),
            (0.25, "ZI"),
        ]
        e_sv = evaluate_hamiltonian(sv, hamiltonian)
        e_mps = evaluate_hamiltonian(mps, hamiltonian)
        e_tab = evaluate_hamiltonian(tab, hamiltonian)

        assert e_sv == pytest.approx(1.5, abs=1e-7)
        assert e_mps == pytest.approx(1.5, abs=1e-7)
        assert e_tab == pytest.approx(1.5, abs=1e-7)

    def test_single_qubit_clifford_states_parity(self):
        """Verify +/-1 eigenstates for |0>, |1>, |+>, |->, |+i>, |-i> across all 3 solvers."""
        # Prepare |1> on q0, |-> on q1, |+i> on q2
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.x(1)
        qc.h(1)
        qc.h(2)
        qc.s(2)

        sv = evolve_statevector(qc, precision="fp64")
        mps_res = simulate_mps(qc, max_bond_dim=8, shots=16, seed=42)
        tab_res = simulate_stabilizer(qc, shots=16, seed=42)

        mps = mps_res["mps_state"]
        tab = tab_res["tableau"]

        # Note convention: string index 0 -> q0, index 1 -> q1, index 2 -> q2
        checks = {
            "ZII": -1.0,
            "IXI": -1.0,
            "IIY": 1.0,
            "ZXI": 1.0,
            "ZXY": 1.0,
            "ZZZ": 0.0,
        }
        for p_str, exp_v in checks.items():
            assert evaluate_pauli_expectation(sv, p_str) == pytest.approx(exp_v, abs=1e-7)
            assert evaluate_pauli_expectation(mps, p_str) == pytest.approx(exp_v, abs=1e-7)
            assert evaluate_pauli_expectation(tab, p_str) == pytest.approx(exp_v, abs=1e-7)

    def test_50_qubit_ghz_pauli_expectation_mps_and_stabilizer(self):
        """50-qubit GHZ state expectation evaluated in milliseconds without statevector blowup."""
        n = 50
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)

        mps_res = simulate_mps(qc, max_bond_dim=4, shots=16, seed=42)
        tab_res = simulate_stabilizer(qc, shots=16, seed=42)

        mps = mps_res["mps_state"]
        tab = tab_res["tableau"]

        z_all = "Z" * n
        x_all = "X" * n
        z0_only = "Z" + "I" * (n - 1)
        z_ends = "Z" + "I" * (n - 2) + "Z"

        for engine in (mps, tab):
            assert evaluate_pauli_expectation(engine, z_all) == pytest.approx(1.0, abs=1e-6)
            assert evaluate_pauli_expectation(engine, x_all) == pytest.approx(1.0, abs=1e-6)
            assert evaluate_pauli_expectation(engine, z0_only) == pytest.approx(0.0, abs=1e-6)
            assert evaluate_pauli_expectation(engine, z_ends) == pytest.approx(1.0, abs=1e-6)


class TestScalableEntanglementEntropy:
    """Verify O(chi) MPS and O(n^3) GF(2) Stabilizer bipartite entanglement entropy."""

    def test_100_qubit_stabilizer_entanglement_entropy(self):
        """100-qubit GHZ circuit must have exactly 1.0 ebit bipartite entropy; |0>^100 has 0.0 ebit."""
        qasm_path = os.path.join("benchmarks", "ghz_100qubit.qasm")
        qc_ghz = load_qasm(qasm_path).to_quantum_circuit()
        tab_res = simulate_stabilizer(qc_ghz, shots=16, seed=42)
        tab_ghz = tab_res["tableau"]

        s_ghz = calculate_stabilizer_entanglement_entropy(tab_ghz)
        assert s_ghz == pytest.approx(1.0, abs=1e-7)

        # Also test via calculate_bipartite_entropy dispatcher
        ent_dict = calculate_bipartite_entropy(qc_ghz, method="stabilizer")
        assert ent_dict["von_neumann_entropy"] == pytest.approx(1.0, abs=1e-7)
        assert ent_dict["entanglement_regime"] == "Low (Area-law)"

        # Product state |0>^100 must yield 0.0 ebit
        tab_zero = StabilizerTableau(100)
        assert calculate_stabilizer_entanglement_entropy(tab_zero) == pytest.approx(0.0, abs=1e-7)

    def test_mps_entanglement_entropy_cut_and_product(self):
        """MPS Schmidt singular value entropy matches 1.0 ebit on GHZ and 0.0 ebit on product state."""
        qc_ghz = QuantumCircuit(10)
        qc_ghz.h(0)
        for i in range(9):
            qc_ghz.cx(i, i + 1)

        mps_res = simulate_mps(qc_ghz, max_bond_dim=8, shots=16, seed=42)
        mps_ghz = mps_res["mps_state"]

        s_mps = calculate_mps_entanglement_entropy(mps_ghz)
        assert s_mps == pytest.approx(1.0, abs=1e-6)

        mps_prod = MatrixProductState(10, max_bond_dim=8)
        assert calculate_mps_entanglement_entropy(mps_prod) == pytest.approx(0.0, abs=1e-7)


class TestMixedPrecisionEmulation:
    """Verify FP32 (complex64) vs FP64 (complex128) memory reduction and precision drift."""

    def test_mps_and_statevector_fp32_memory_halving(self):
        """FP32 allocation uses complex64 and exactly 50% of the byte footprint of FP64."""
        qc = QuantumCircuit(8)
        qc.h(0)
        for i in range(7):
            qc.cx(i, i + 1)
            qc.rz(0.314, i + 1)

        sv64 = evolve_statevector(qc, precision="fp64")
        sv32 = evolve_statevector(qc, precision="fp32")
        assert sv64.dtype == np.complex128
        assert sv32.dtype == np.complex64
        assert sv32.nbytes * 2 == sv64.nbytes

        mps64 = simulate_mps(qc, max_bond_dim=8, precision="fp64")["mps_state"]
        mps32 = simulate_mps(qc, max_bond_dim=8, precision="fp32")["mps_state"]
        assert mps64.dtype == np.complex128
        assert mps32.dtype == np.complex64
        assert mps32.memory_bytes * 2 == mps64.memory_bytes

        drift = evaluate_precision_drift(qc)
        assert drift["total_variation_distance"] < 1e-5
        assert drift["l2_state_drift"] < 1e-5
        assert drift["memory_saved_pct"] == pytest.approx(50.0)

    def test_simulate_trajectories_and_run_simulation_fp32(self):
        """Verify trajectory noise simulation and run_simulation execute cleanly in FP32."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)

        traj_res = simulate_trajectories(qc, noise_model="low", shots=64, precision="fp32")
        assert sum(traj_res["counts"].values()) == 64

        sim_res = run_simulation(
            circuit=qc.copy(),
            method="statevector",
            shots=64,
            precision="fp32",
        )
        assert sim_res["success"] is True
        assert sim_res["precision"] == "fp32"
