"""Regression tests for trajectory engine operation parsing and gate accounting.

The Monte Carlo Wavefunction (MCWF) engine executes 1-qubit and 2-qubit unitary
gates only. Operations whose effect cannot be modeled (mid-circuit measurements,
resets, classical control flow, delays under an active noise model) must fail
explicitly instead of being silently skipped; identity instructions that reach the
engine receive the same noise treatment as any other 1-qubit gate, while Qiskit's
default transpiler removes identity-equivalent operations beforehand; and reported
gate counts must distinguish the original circuit from the decomposed and fused
execution forms.
"""
import numpy as np
import pytest
from unittest.mock import patch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.engine.noise import TrajectoryNoiseModel
from src.engine.parser import ParsedQASMCircuit, QASMInstruction, parse_qasm
from src.engine.simulator import run_simulation, simulate_trajectories


def _zero_error_model() -> TrajectoryNoiseModel:
    """Deterministic trajectory model: no gate errors, no readout error, negligible T1/T2 decay."""
    return TrajectoryNoiseModel(
        t1=1.0,
        t2=2.0,
        gate_error_1q=0.0,
        gate_error_2q=0.0,
        gate_time_1q=1e-12,
        gate_time_2q=1e-12,
        readout_error=0.0,
    )


def test_supported_circuit_deterministic_result():
    """X followed by measurement yields a deterministic bitstring with consistent gate counts."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)

    res = simulate_trajectories(qc, noise_model=_zero_error_model(), shots=64, seed=7, num_workers=1)

    assert res["success"] is True
    assert res["counts"] == {"1": 64}
    assert res["original_gates"] == 1
    assert res["logical_gates"] == 1
    assert res["fused_gates"] == 1
    assert res["metadata"]["original_gates"] == 1


def test_decomposed_toffoli_executes_correctly():
    """A Toffoli gate decompiles into 1-/2-qubit basis gates and executes exactly."""
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.x(1)
    qc.ccx(0, 1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])

    res = simulate_trajectories(qc, noise_model=_zero_error_model(), shots=16, seed=11, num_workers=1)

    assert res["success"] is True
    assert res["counts"] == {"111": 16}
    # The input circuit has 3 logical operations; transpilation expands the Toffoli gate.
    assert res["original_gates"] == 3
    assert res["logical_gates"] > res["original_gates"]


def test_unsupported_reset_fails_explicitly():
    """reset changes circuit semantics and must raise instead of being silently dropped."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.reset(0)
    qc.measure(0, 0)

    with pytest.raises(ValueError, match="reset"):
        simulate_trajectories(qc, noise_model=_zero_error_model(), shots=8, seed=1, num_workers=1)


def test_initialize_decomposing_to_reset_fails_explicitly():
    """initialize decompiles to reset plus unitaries and must not be skipped silently."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.initialize([1, 0], 0)
    qc.measure(0, 0)

    with pytest.raises(ValueError, match="reset"):
        simulate_trajectories(qc, noise_model=_zero_error_model(), shots=8, seed=1, num_workers=1)


def test_classical_control_flow_fails_explicitly():
    """Classical feed-forward cannot be represented as a unitary and must raise."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    with qc.if_test((qc.clbits[0], True)):
        qc.x(0)

    with pytest.raises(ValueError, match="Unsupported operation"):
        simulate_trajectories(qc, noise_model=_zero_error_model(), shots=8, seed=1, num_workers=1)


def test_gate_accounting_and_fusion_metrics():
    """Noiseless runs distinguish original, decomposed, and post-fusion gate counts."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.barrier()
    qc.t(0)
    qc.measure(0, 0)

    shots = 2000
    res = simulate_trajectories(qc, noise_model=None, shots=shots, seed=5, num_workers=1)

    assert res["success"] is True
    # Barriers and measurements are excluded; h and t reach the fusion stage and contract to one unitary.
    assert res["original_gates"] == 2
    assert res["logical_gates"] == 2
    assert res["fused_gates"] == 1
    assert res["fusion_reduction_pct"] == pytest.approx(50.0, abs=0.01)
    assert res["gate_fusion_compression_ratio"] == 2.0
    assert res["fusion_metrics"]["original_gates"] == 2
    assert res["fusion_metrics"]["executed_gate_count"] == 1

    # The fused evolution must match the exact statevector probability distribution.
    exact_circuit = qc.copy()
    exact_circuit.remove_final_measurements(inplace=True)
    p_zero_expected = float(np.abs(Statevector.from_instruction(exact_circuit).data[0]) ** 2)

    assert sum(res["counts"].values()) == shots
    p_zero_observed = res["counts"].get("0", 0) / shots
    mc_tol = 3.0 / (shots ** 0.5)
    assert abs(p_zero_observed - p_zero_expected) < mc_tol


def test_barrier_and_measurement_handling():
    """Barriers are skipped and final measurements are removed without changing statistics."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.barrier()
    qc.cx(0, 1)
    qc.barrier()
    qc.measure([0, 1], [0, 1])

    shots = 400
    res = simulate_trajectories(qc, noise_model=_zero_error_model(), shots=shots, seed=13, num_workers=1)

    assert res["success"] is True
    assert sum(res["counts"].values()) == shots
    assert set(res["counts"].keys()) <= {"00", "11"}
    # Barriers and terminal measurements are excluded from every gate-accounting metric.
    assert res["original_gates"] == 2
    assert res["logical_gates"] == 2
    assert res["fused_gates"] == 2
    assert res["fusion_metrics"]["executed_gate_count"] == 2


def test_mid_circuit_measurement_fails_explicitly():
    """H -> measure -> H must fail: skipping the collapse would preserve coherence and
    return a deterministic '0' instead of the correct 50/50 distribution."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.h(0)

    with pytest.raises(ValueError, match="Mid-circuit measurement"):
        simulate_trajectories(qc, noise_model=None, shots=64, seed=17, num_workers=1)

    # A measurement consumed by classical feed-forward is rejected on the measurement
    # itself before the conditional operation is even parsed.
    cond_qc = QuantumCircuit(1, 1)
    cond_qc.h(0)
    cond_qc.measure(0, 0)
    with cond_qc.if_test((cond_qc.cregs[0], 1)):
        cond_qc.x(0)

    with pytest.raises(ValueError, match="Mid-circuit measurement"):
        simulate_trajectories(cond_qc, noise_model=None, shots=64, seed=17, num_workers=1)


def test_delay_noiseless_is_skipped():
    """Without a noise model an idle delay does not change the ideal statevector and is
    excluded from gate accounting."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.delay(100, 0, unit="ns")
    qc.measure(0, 0)

    res = simulate_trajectories(qc, noise_model=None, shots=16, seed=3, num_workers=1)

    assert res["counts"] == {"1": 16}
    assert res["original_gates"] == 1
    assert res["logical_gates"] == 1
    assert res["fused_gates"] == 1


def test_delay_with_noise_fails_explicitly():
    """With an active trajectory noise model, idle-time decoherence cannot be modeled, so
    the delay must not be dropped as if the delayed evolution were noise-free."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.delay(100, 0, unit="ns")
    qc.measure(0, 0)

    with pytest.raises(ValueError, match="Delay"):
        simulate_trajectories(qc, noise_model=_zero_error_model(), shots=16, seed=3, num_workers=1)


def test_delay_with_noise_fails_through_native_fusion_path():
    """The native-fusion pre-pass must not absorb delays, so run_simulation still rejects
    an unmodelable delay under noise when an explicit backend forces the fusion path."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.delay(100, 0, unit="ns")
    qc.measure(0, 0)

    res = run_simulation(
        qc,
        noise_method="trajectory",
        noise_level="low",
        backend="cpu",
        shots=8,
        runs=1,
    )

    assert res["success"] is False
    assert "Delay" in (res.get("error") or "")


def _identity_passthrough_transpile():
    """Stub Qiskit transpilation so identity instructions reach the trajectory engine.

    Qiskit removes identity-equivalent operations at the default optimization level, so
    this isolates the trajectory engine contract for identity instructions.
    """
    return patch(
        "src.engine.simulator.transpile",
        side_effect=lambda circuit, basis_gates=None, **kwargs: circuit,
    )


def test_identity_instruction_receives_noise_when_it_survives_transpilation():
    """Engine contract: an identity instruction that survives transpilation executes as a
    1-qubit unitary so that the active noise model is applied to it instead of being
    bypassed. Qiskit's default transpiler removes identity-equivalent operations, so
    transpilation is stubbed here; the public path is covered by the test below."""
    p = 0.3
    model = TrajectoryNoiseModel(
        t1=1.0,
        t2=2.0,
        gate_error_1q=p,
        gate_error_2q=0.0,
        gate_time_1q=1e-12,
        gate_time_2q=1e-12,
        readout_error=0.0,
    )
    qc = QuantumCircuit(1, 1)
    qc.id(0)
    qc.measure(0, 0)

    shots = 4000
    with _identity_passthrough_transpile():
        res = simulate_trajectories(qc, noise_model=model, shots=shots, seed=21, num_workers=1)

    assert res["success"] is True
    assert res["original_gates"] == 1
    assert res["logical_gates"] == 1
    assert res["fused_gates"] == 1
    # Depolarizing X/Y jumps flip |0> to |1> with probability p/3 + p/3 = 2p/3.
    p_one = res["counts"].get("1", 0) / shots
    assert abs(p_one - 2.0 * p / 3.0) < 3.0 / (shots ** 0.5)


def test_identity_instruction_noiseless_is_inert_and_consistent():
    """In noiseless runs identity matrices execute and fuse without changing the ideal
    state, keeping all gate-accounting metrics consistent."""
    qc = QuantumCircuit(1, 1)
    qc.id(0)
    qc.barrier()
    qc.id(0)
    qc.measure(0, 0)

    with _identity_passthrough_transpile():
        res = simulate_trajectories(qc, noise_model=None, shots=32, seed=4, num_workers=1)

    assert res["counts"] == {"0": 32}
    assert res["original_gates"] == 2
    assert res["logical_gates"] == 2
    assert res["fused_gates"] == 1
    assert res["fusion_reduction_pct"] == pytest.approx(50.0, abs=0.01)
    assert res["fusion_metrics"]["executed_gate_count"] == 1


def test_explicit_identity_gate_is_removed_by_default_transpilation():
    """Public-path behavior: Qiskit's default transpiler removes identity-equivalent
    operations, so an explicit id in the input circuit does not receive noise. Identity
    noise is applied only to identity instructions that survive transpilation."""
    model = TrajectoryNoiseModel(
        t1=1.0,
        t2=2.0,
        gate_error_1q=0.3,
        gate_error_2q=0.0,
        gate_time_1q=1e-12,
        gate_time_2q=1e-12,
        readout_error=0.0,
    )
    qc = QuantumCircuit(1, 1)
    qc.id(0)
    qc.measure(0, 0)

    res = simulate_trajectories(qc, noise_model=model, shots=64, seed=23, num_workers=1)

    assert res["counts"] == {"0": 64}   # deterministic despite gate_error_1q=0.3
    assert res["original_gates"] == 1   # counted in the input circuit
    assert res["logical_gates"] == 0    # removed by the transpiler before parsing


def test_gate_accounting_excludes_non_gate_directives():
    """Barriers, delays, and terminal measurements must not inflate or skew the counts."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.barrier()
    qc.delay(100, 0, unit="ns")
    qc.measure(0, 0)

    res = simulate_trajectories(qc, noise_model=None, shots=16, seed=6, num_workers=1)

    assert res["counts"] == {"1": 16}
    assert res["original_gates"] == 1
    assert res["logical_gates"] == 1
    assert res["fused_gates"] == 1
    assert res["metadata"]["original_gates"] == 1


def test_gate_accounting_matches_qasm_parser_convention():
    """original_gates follows the shared NON_GATE_INSTRUCTIONS convention for QASM input,
    and delay is treated as a non-gate directive in both representations."""
    qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
barrier q;
measure q -> c;
"""
    parsed = parse_qasm(qasm)
    assert parsed.total_gates == 2

    res = simulate_trajectories(parsed, noise_model=None, shots=16, seed=8, num_workers=1)

    assert res["original_gates"] == parsed.total_gates
    assert res["logical_gates"] == 2
    assert res["fused_gates"] == 2
    assert set(res["counts"].keys()) <= {"00", "11"}

    # OpenQASM 2.0 has no delay statement, so the delay case is exercised on the parser's
    # instruction representation directly and compared with the equivalent Qiskit circuit.
    delay_parsed = ParsedQASMCircuit()
    delay_parsed.qregs["q"] = 1
    delay_parsed.cregs["c"] = 1
    delay_parsed.instructions = [
        QASMInstruction(name="x", qargs=[("q", 0)]),
        QASMInstruction(name="delay", qargs=[("q", 0)]),
        QASMInstruction(name="measure", qargs=[("q", 0)], cargs=[("c", 0)]),
    ]
    assert delay_parsed.total_gates == 1
    assert delay_parsed.one_qubit_gates == 1

    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.delay(100, 0, unit="ns")
    qc.measure(0, 0)
    res_delay = simulate_trajectories(qc, noise_model=None, shots=16, seed=9, num_workers=1)

    assert res_delay["original_gates"] == delay_parsed.total_gates == 1


def test_seeded_reproducibility_for_supported_circuit():
    """A supported Bell circuit remains bit-for-bit reproducible for a fixed seed."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    model = _zero_error_model()
    shots = 200
    res_a = simulate_trajectories(qc, noise_model=model, shots=shots, seed=99, num_workers=1)
    res_b = simulate_trajectories(qc, noise_model=model, shots=shots, seed=99, num_workers=1)

    assert res_a["counts"] == res_b["counts"]
    assert sum(res_a["counts"].values()) == shots
    assert set(res_a["counts"].keys()) <= {"00", "11"}


def test_run_simulation_reports_unsupported_operation():
    """run_simulation surfaces the explicit trajectory error instead of a silent wrong result."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.reset(0)
    qc.measure(0, 0)

    res = run_simulation(qc, noise_method="trajectory", noise_level="low", shots=10, runs=1)

    assert res["success"] is False
    assert res["error"] is not None
    assert "reset" in res["error"]
