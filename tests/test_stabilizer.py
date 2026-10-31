import os
import math
import time
import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.engine.stabilizer import (
    StabilizerTableau,
    NonCliffordGateError,
    is_clifford_angle,
    simulate_stabilizer,
    calculate_stabilizer_ram_savings,
)
from src.engine.simulator import run_simulation


class TestCliffordAngleDetection:
    """Verify exact detection of integer multiples of pi/2."""

    def test_exact_multiples_of_pi_over_2(self):
        is_clif, k = is_clifford_angle(0.0)
        assert is_clif is True and k == 0

        is_clif, k = is_clifford_angle(math.pi / 2.0)
        assert is_clif is True and k == 1

        is_clif, k = is_clifford_angle(math.pi)
        assert is_clif is True and k == 2

        is_clif, k = is_clifford_angle(3.0 * math.pi / 2.0)
        assert is_clif is True and k == 3

        is_clif, k = is_clifford_angle(-math.pi / 2.0)
        assert is_clif is True and k == 3

    def test_non_clifford_angles_rejected(self):
        for theta in [math.pi / 4.0, math.pi / 3.0, math.pi / 8.0, 0.12345]:
            is_clif, _ = is_clifford_angle(theta)
            assert is_clif is False


class TestDeterministicQuantumInterference:
    """
    Verify Aaronson-Gottesman rowsum phase exponent g(x1, z1, x2, z2)
    on deterministic quantum phase interference circuits.
    """

    def test_h_z_h_gives_one_deterministically(self):
        """H * Z * H |0> = X |0> = |1> with 100% determinism."""
        tab = StabilizerTableau(1)
        tab.apply_h(0)
        tab.apply_z(0)
        tab.apply_h(0)
        outcome = tab.measure(0, rng=np.random.default_rng(42))
        assert outcome == 1

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.z(0)
        qc.h(0)
        qc.measure(0, 0)
        res = simulate_stabilizer(qc, shots=256, seed=42)
        assert res["counts"] == {"1": 256}

    def test_h_s_s_h_gives_one_deterministically(self):
        """H * S * S * H |0> = H * Z * H |0> = |1> with 100% determinism."""
        tab = StabilizerTableau(1)
        tab.apply_h(0)
        tab.apply_s(0)
        tab.apply_s(0)
        tab.apply_h(0)
        outcome = tab.measure(0, rng=np.random.default_rng(42))
        assert outcome == 1

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.s(0)
        qc.s(0)
        qc.h(0)
        qc.measure(0, 0)
        res = simulate_stabilizer(qc, shots=256, seed=42)
        assert res["counts"] == {"1": 256}

    def test_h_s_sdg_h_gives_zero_deterministically(self):
        """H * S * Sdg * H |0> = H * H |0> = |0> with 100% determinism."""
        tab = StabilizerTableau(1)
        tab.apply_h(0)
        tab.apply_s(0)
        tab.apply_sdg(0)
        tab.apply_h(0)
        outcome = tab.measure(0, rng=np.random.default_rng(42))
        assert outcome == 0

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.s(0)
        qc.sdg(0)
        qc.h(0)
        qc.measure(0, 0)
        res = simulate_stabilizer(qc, shots=256, seed=42)
        assert res["counts"] == {"0": 256}

    def test_multi_qubit_pauli_y_phase_cancellation(self):
        """
        Verify multi-qubit rowsum where Y * Y = +I and X * Z * Z * X = +Y * Y = +I
        produces exact interference in a 2-qubit Bell basis measurement.
        Bell state |Phi-> = (|00> - |11>)/sqrt(2) under H x H becomes
        (|01> + |10>)/sqrt(2), so parity q0 XOR q1 is deterministically 1.
        """
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.z(0)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(1)
        qc.measure([0, 1], [0, 1])

        res = simulate_stabilizer(qc, shots=512, seed=123)
        counts = res["counts"]
        assert set(counts.keys()).issubset({"01", "10"})
        assert sum(counts.values()) == 512


class TestStabilizerVsStatevectorParity:
    """Verify statistical parity between StabilizerTableau and Qiskit Statevector."""

    def test_bell_state_2qubit_parity(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        shots = 4096
        res = simulate_stabilizer(qc, shots=shots, seed=42)
        counts = res["counts"]
        assert set(counts.keys()) == {"00", "11"}
        assert abs(counts["00"] / shots - 0.5) < 0.05
        assert abs(counts["11"] / shots - 0.5) < 0.05

    def test_ghz_state_4qubit_parity(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)

        shots = 4096
        res = simulate_stabilizer(qc, shots=shots, seed=99)
        counts = res["counts"]
        assert set(counts.keys()) == {"0000", "1111"}
        assert abs(counts["0000"] / shots - 0.5) < 0.05
        assert abs(counts["1111"] / shots - 0.5) < 0.05

    def test_quantum_teleportation_coherent_clifford_parity(self):
        """
        Coherent 3-qubit Quantum Teleportation circuit transporting state |-> = H Z |0>
        from q[0] to q[2], followed by H on q[2] so q[2] deterministically measures '1'.
        """
        qc = QuantumCircuit(3)
        # Prepare |-> on q[0]
        qc.h(0)
        qc.z(0)
        # Create Bell pair between q[1] and q[2]
        qc.h(1)
        qc.cx(1, 2)
        # Bell measurement unitary on q[0], q[1]
        qc.cx(0, 1)
        qc.h(0)
        # Coherent Clifford corrections on q[2]
        qc.cx(1, 2)
        qc.cz(0, 2)
        # Undo H on q[2]: since |-> was teleported, H|- > = |1> deterministically
        qc.h(2)

        sv_probs = Statevector.from_instruction(qc).probabilities_dict()
        res = simulate_stabilizer(qc, shots=2048, seed=7)
        for bitstr, count in res["counts"].items():
            # Every sampled bitstring must have q[2] (leftmost bit in Qiskit little-endian) == '1'
            assert bitstr[0] == "1"
            assert sv_probs.get(bitstr, 0.0) > 0.20

    def test_random_clifford_circuit_matches_statevector(self):
        """
        Construct a 5-qubit circuit with diverse Clifford gates (H, S, Sdg, X, Y, Z,
        SX, SXdg, CX, CZ, CY, SWAP, RZ(pi/2)) and verify statistical distribution
        matches exact Statevector probabilities.
        """
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.s(0)
        qc.cx(0, 1)
        qc.sx(2)
        qc.cz(1, 2)
        qc.y(3)
        qc.cy(2, 3)
        qc.swap(3, 4)
        qc.rz(math.pi / 2.0, 4)
        qc.sdg(1)
        qc.h(2)
        qc.cx(4, 0)
        qc.h(0)
        qc.sxdg(3)

        sv_probs = Statevector.from_instruction(qc).probabilities_dict()
        shots = 8192
        res = simulate_stabilizer(qc, shots=shots, seed=2026)
        counts = res["counts"]

        # Every sampled bitstring must exist in the exact Statevector support
        for bitstr, count in counts.items():
            assert sv_probs.get(bitstr, 0.0) > 1e-9, f"Forbidden bitstring {bitstr} sampled!"

        # Total Variation Distance (TVD) between empirical and exact distributions < 0.04
        all_keys = set(sv_probs.keys()) | set(counts.keys())
        tvd = 0.5 * sum(abs(counts.get(k, 0) / shots - sv_probs.get(k, 0.0)) for k in all_keys)
        assert tvd < 0.04


class Test100QubitGHZBenchmark:
    """Verify 100-qubit maximally entangled GHZ simulation in < 0.5s and < 1 MB."""

    def test_100_qubit_ghz_benchmark_execution(self):
        qasm_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "benchmarks",
            "ghz_100qubit.qasm"
        )
        assert os.path.exists(qasm_path), "benchmarks/ghz_100qubit.qasm must exist"
        with open(qasm_path, "r", encoding="utf-8") as f:
            qasm_str = f.read()

        t0 = time.perf_counter()
        res = simulate_stabilizer(qasm_str, shots=200, seed=42)
        elapsed = time.perf_counter() - t0

        assert res["success"] is True
        assert res["num_qubits"] == 100
        assert res["clifford_gate_count"] == 100
        assert elapsed < 0.5, f"100-qubit GHZ took {elapsed:.3f}s (expected < 0.5s)"
        assert res["tableau_memory_bytes"] < 1024 * 1024, "Tableau memory must be < 1 MB"

        counts = res["counts"]
        zero_100 = "0" * 100
        one_100 = "1" * 100
        assert set(counts.keys()) == {zero_100, one_100}
        assert counts[zero_100] + counts[one_100] == 200
        assert 60 <= counts[zero_100] <= 140
        assert 60 <= counts[one_100] <= 140

        savings = calculate_stabilizer_ram_savings(100)
        assert savings["savings_percent"] == 100.0


class TestNonCliffordGateRejection:
    """Verify NonCliffordGateError is raised on non-Clifford operations."""

    def test_t_gate_raises_non_clifford_error(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.t(0)
        with pytest.raises(NonCliffordGateError):
            simulate_stabilizer(qc, shots=10)

    def test_ccx_toffoli_raises_non_clifford_error(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.ccx(0, 1, 2)
        with pytest.raises(NonCliffordGateError):
            simulate_stabilizer(qc, shots=10)

    def test_rx_pi_over_3_raises_non_clifford_error(self):
        qc = QuantumCircuit(1)
        qc.rx(math.pi / 3.0, 0)
        with pytest.raises(NonCliffordGateError):
            simulate_stabilizer(qc, shots=10)

    def test_run_simulation_with_stabilizer_method(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        res = run_simulation(qc, method="stabilizer", shots=128, runs=1)
        assert res["success"] is True
        assert res["metadata"]["method"] == "stabilizer"
        assert set(res["counts"].keys()) == {"0000", "1111"}
