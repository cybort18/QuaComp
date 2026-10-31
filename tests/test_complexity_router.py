import os
import math
import pytest
from qiskit import QuantumCircuit

from src.engine.complexity import (
    analyze_circuit_complexity,
    select_optimal_solver,
)
from src.engine.simulator import run_simulation


BENCHMARKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmarks",
)


def _read_qasm(filename: str) -> str:
    path = os.path.join(BENCHMARKS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class TestCircuitComplexityProfiler:
    """Verify static complexity analysis for Clifford vs Non-Clifford and interaction topology."""

    def test_pure_clifford_qasm_benchmarks(self):
        for fname in ["bell_state.qasm", "ghz_state.qasm", "ghz_100qubit.qasm"]:
            qasm_str = _read_qasm(fname)
            profile = analyze_circuit_complexity(qasm_str)
            assert profile["is_pure_clifford"] is True, f"{fname} should be pure Clifford"
            assert profile["non_clifford_gates"] == 0
            assert profile["magic_density"] == 0.0
            assert profile["clifford_gates"] > 0

    def test_ghz_100qubit_topology_metrics(self):
        qasm_str = _read_qasm("ghz_100qubit.qasm")
        profile = analyze_circuit_complexity(qasm_str)
        assert profile["num_qubits"] == 100
        assert profile["total_gates"] == 100
        assert profile["one_qubit_gates"] == 1
        assert profile["two_qubit_gates"] == 99
        assert profile["max_interaction_distance"] == 1
        assert profile["mean_interaction_distance"] == 1.0
        assert profile["entanglement_topology"] == "1d_nearest_neighbor"

    def test_qft_4qubit_has_quantum_magic(self):
        qasm_str = _read_qasm("qft_4qubit.qasm")
        profile = analyze_circuit_complexity(qasm_str)
        assert profile["is_pure_clifford"] is False
        assert profile["non_clifford_gates"] > 0
        assert profile["magic_density"] > 0.0
        assert profile["max_interaction_distance"] == 3

    def test_pi_over_2_rotations_classified_as_clifford(self):
        qc = QuantumCircuit(3)
        qc.rx(math.pi / 2.0, 0)
        qc.ry(math.pi, 1)
        qc.rz(3.0 * math.pi / 2.0, 2)
        qc.cx(0, 1)
        qc.cz(1, 2)

        profile = analyze_circuit_complexity(qc)
        assert profile["is_pure_clifford"] is True
        assert profile["non_clifford_gates"] == 0
        assert profile["clifford_gates"] == 5
        assert profile["magic_density"] == 0.0


class TestAutoSolverRouter:
    """Verify automatic solver selection across stabilizer, mps, and statevector."""

    def test_auto_routes_ghz_100qubit_to_stabilizer(self):
        qasm_str = _read_qasm("ghz_100qubit.qasm")
        decision = select_optimal_solver(qasm_str, requested_method="auto")
        assert decision["selected_solver"] == "stabilizer"
        assert decision["complexity_profile"]["is_pure_clifford"] is True
        assert "Gottesman-Knill" in decision["routing_reason"]

    def test_auto_routes_30qubit_non_clifford_1d_to_mps(self):
        qc = QuantumCircuit(30)
        for i in range(30):
            qc.h(i)
            qc.t(i)
        for i in range(29):
            qc.cx(i, i + 1)

        decision = select_optimal_solver(qc, requested_method="auto")
        assert decision["selected_solver"] == "mps"
        assert decision["complexity_profile"]["is_pure_clifford"] is False
        assert decision["complexity_profile"]["entanglement_topology"] == "1d_nearest_neighbor"

    def test_auto_routes_8qubit_qft_to_statevector(self):
        from src.engine.circuits import generate_qft_circuit
        qc = generate_qft_circuit(8)

        decision = select_optimal_solver(qc, requested_method="auto")
        assert decision["selected_solver"] == "statevector"
        assert decision["complexity_profile"]["is_pure_clifford"] is False

    def test_explicit_method_override_honored(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        decision = select_optimal_solver(qc, requested_method="mps")
        assert decision["selected_solver"] == "mps"

    def test_run_simulation_auto_executes_stabilizer_on_pure_clifford(self):
        qc = QuantumCircuit(6)
        qc.h(0)
        for i in range(5):
            qc.cx(i, i + 1)

        res = run_simulation(qc, method="auto", shots=256, runs=1)
        assert res["success"] is True
        assert res["metadata"]["method"] == "stabilizer"
        assert res["solver_routing"]["selected_solver"] == "stabilizer"
        assert res["complexity_profile"]["is_pure_clifford"] is True
        assert set(res["counts"].keys()) == {"000000", "111111"}
