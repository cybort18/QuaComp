import os
import math
import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.engine.parser import (
    parse_qasm,
    load_qasm,
    evaluate_parameter_expression,
    QASMParseError,
    ParsedQASMCircuit,
    QASMInstruction,
)
from src.engine.simulator import run_simulation


# ==============================================================================
# 1. Parameter Expression Evaluation Tests
# ==============================================================================

def test_evaluate_parameter_expression_constants_and_scalars():
    assert evaluate_parameter_expression("0") == 0.0
    assert evaluate_parameter_expression("1.5") == 1.5
    assert evaluate_parameter_expression("-3.14") == -3.14
    assert evaluate_parameter_expression("pi") == pytest.approx(math.pi)
    assert evaluate_parameter_expression("PI") == pytest.approx(math.pi)
    assert evaluate_parameter_expression("e") == pytest.approx(math.e)
    assert evaluate_parameter_expression("E") == pytest.approx(math.e)


def test_evaluate_parameter_expression_arithmetic():
    assert evaluate_parameter_expression("pi / 2") == pytest.approx(math.pi / 2.0)
    assert evaluate_parameter_expression("-pi / 4") == pytest.approx(-math.pi / 4.0)
    assert evaluate_parameter_expression("2 * pi") == pytest.approx(2.0 * math.pi)
    assert evaluate_parameter_expression("3 * pi / 4") == pytest.approx(3.0 * math.pi / 4.0)
    assert evaluate_parameter_expression("1 + 2 * 3") == 7.0
    assert evaluate_parameter_expression("2 ^ 3") == 8.0
    assert evaluate_parameter_expression("2 ** 3") == 8.0


def test_evaluate_parameter_expression_functions():
    assert evaluate_parameter_expression("sin(pi / 2)") == pytest.approx(1.0)
    assert evaluate_parameter_expression("cos(0)") == pytest.approx(1.0)
    assert evaluate_parameter_expression("tan(0)") == pytest.approx(0.0)
    assert evaluate_parameter_expression("exp(0)") == pytest.approx(1.0)
    assert evaluate_parameter_expression("ln(e)") == pytest.approx(1.0)
    assert evaluate_parameter_expression("sqrt(16)") == pytest.approx(4.0)


def test_evaluate_parameter_expression_errors():
    with pytest.raises(QASMParseError, match="Empty parameter"):
        evaluate_parameter_expression("")
        
    with pytest.raises(QASMParseError, match="Syntax error"):
        evaluate_parameter_expression("pi /")

    with pytest.raises(QASMParseError, match="Unknown variable"):
        evaluate_parameter_expression("alpha + 1")

    with pytest.raises(QASMParseError, match="Division by zero"):
        evaluate_parameter_expression("pi / 0")

    with pytest.raises(QASMParseError, match="Unsupported function"):
        evaluate_parameter_expression("unknown_func(1)")

    with pytest.raises(QASMParseError, match="requires exactly 1 argument"):
        evaluate_parameter_expression("sin(1, 2)")


# ==============================================================================
# 2. Syntax Parsing & Quantum State Fidelity (Bell State)
# ==============================================================================

def test_parse_bell_state_and_verify_statevector():
    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    
    qreg q[2];
    creg c[2];
    
    h q[0];
    cx q[0], q[1];
    """
    circuit = parse_qasm(qasm_str, name="bell_state")
    
    assert circuit.name == "bell_state"
    assert circuit.num_qubits == 2
    assert circuit.num_clbits == 2
    assert circuit.total_gates == 2
    assert circuit.one_qubit_gates == 1
    assert circuit.two_qubit_gates == 1
    assert circuit.depth == 2
    assert circuit.count_ops() == {"h": 1, "cx": 1}

    # Convert to Qiskit QuantumCircuit
    qc = circuit.to_quantum_circuit()
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 2

    # Verify exact statevector: (|00> + |11>) / sqrt(2)
    sv = Statevector.from_instruction(qc)
    expected_amplitude = 1.0 / math.sqrt(2.0)
    
    # In Qiskit standard indexing: sv[0] is |00>, sv[3] is |11>
    probs = sv.probabilities_dict()
    assert probs.get("00", 0.0) == pytest.approx(0.5, abs=1e-6)
    assert probs.get("11", 0.0) == pytest.approx(0.5, abs=1e-6)
    assert probs.get("01", 0.0) == pytest.approx(0.0, abs=1e-6)
    assert probs.get("10", 0.0) == pytest.approx(0.0, abs=1e-6)

    data = np.asarray(sv.data)
    assert np.abs(data[0]) == pytest.approx(expected_amplitude, abs=1e-6)
    assert np.abs(data[3]) == pytest.approx(expected_amplitude, abs=1e-6)
    assert np.abs(data[1]) == pytest.approx(0.0, abs=1e-6)
    assert np.abs(data[2]) == pytest.approx(0.0, abs=1e-6)


def test_bell_state_simulator_execution():
    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    
    qreg q[2];
    creg c[2];
    
    h q[0];
    cx q[0], q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    """
    circuit = parse_qasm(qasm_str)
    
    # Test execution via QuaComp engine simulator
    res = run_simulation(circuit, runs=2)
    assert res["success"] is True
    assert res["error"] is None
    assert res["latency"] > 0
    assert len(res["latencies"]) == 2


# ==============================================================================
# 3. Parameterized & Multi-Qubit Gates
# ==============================================================================

def test_parse_parameterized_gates():
    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    
    qreg q[2];
    rx(pi / 2) q[0];
    ry(-pi) q[1];
    rz(pi / 4) q[0];
    u1(pi) q[0];
    u2(0, pi) q[1];
    u3(pi / 2, 0, pi) q[0];
    """
    circuit = parse_qasm(qasm_str)
    assert circuit.total_gates == 6
    assert circuit.one_qubit_gates == 6
    assert circuit.two_qubit_gates == 0
    
    ops = circuit.count_ops()
    assert ops["rx"] == 1
    assert ops["ry"] == 1
    assert ops["rz"] == 1
    assert ops["u1"] == 1
    assert ops["u2"] == 1
    assert ops["u3"] == 1

    # Verify parameters are parsed as accurate floats
    rx_inst = [i for i in circuit.instructions if i.name == "rx"][0]
    assert rx_inst.params[0] == pytest.approx(math.pi / 2.0)
    
    rz_inst = [i for i in circuit.instructions if i.name == "rz"][0]
    assert rz_inst.params[0] == pytest.approx(math.pi / 4.0)

    # Ensure to_quantum_circuit generates valid executable Qiskit circuit
    qc = circuit.to_quantum_circuit()
    assert qc.depth() > 0


def test_parse_multi_qubit_gates():
    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    
    qreg q[3];
    cz q[0], q[1];
    cy q[1], q[2];
    ch q[0], q[2];
    swap q[0], q[1];
    iswap q[1], q[2];
    crz(pi/4) q[0], q[1];
    cu1(pi/2) q[1], q[2];
    cu3(pi/2, 0, pi) q[0], q[2];
    rxx(pi/3) q[0], q[1];
    ryy(pi/3) q[1], q[2];
    rzz(pi/3) q[0], q[2];
    ccx q[0], q[1], q[2];
    cswap q[0], q[1], q[2];
    """
    circuit = parse_qasm(qasm_str)
    assert circuit.total_gates == 13
    assert circuit.two_qubit_gates == 11
    assert circuit.multi_qubit_gates == 2

    qc = circuit.to_quantum_circuit()
    assert qc.num_qubits == 3


# ==============================================================================
# 4. Broadcasting & Multi-Registers
# ==============================================================================

def test_register_broadcasting():
    qasm_str = """
    OPENQASM 2.0;
    qreg q[3];
    h q;
    """
    circuit = parse_qasm(qasm_str)
    assert circuit.total_gates == 3
    for i, inst in enumerate(circuit.instructions):
        assert inst.name == "h"
        assert inst.qargs == [("q", i)]


def test_two_register_broadcasting():
    qasm_str = """
    OPENQASM 2.0;
    qreg a[2];
    qreg b[2];
    cx a, b;
    """
    circuit = parse_qasm(qasm_str)
    assert circuit.total_gates == 2
    assert circuit.instructions[0].qargs == [("a", 0), ("b", 0)]
    assert circuit.instructions[1].qargs == [("a", 1), ("b", 1)]


def test_multi_register_flat_mapping():
    qasm_str = """
    OPENQASM 2.0;
    qreg data[3];
    qreg ancilla[2];
    creg c[5];
    """
    circuit = parse_qasm(qasm_str)
    assert circuit.num_qubits == 5
    assert circuit.num_clbits == 5
    
    q_map = circuit.qubit_map
    assert q_map[("data", 0)] == 0
    assert q_map[("data", 1)] == 1
    assert q_map[("data", 2)] == 2
    assert q_map[("ancilla", 0)] == 3
    assert q_map[("ancilla", 1)] == 4


# ==============================================================================
# 5. Custom Gate Macros
# ==============================================================================

def test_custom_gate_macro_expansion():
    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    
    gate my_bell a, b {
        h a;
        cx a, b;
    }
    
    qreg q[2];
    my_bell q[0], q[1];
    """
    circuit = parse_qasm(qasm_str)
    assert "my_bell" in circuit.custom_gates
    assert circuit.total_gates == 2
    assert circuit.instructions[0].name == "h"
    assert circuit.instructions[0].qargs == [("q", 0)]
    assert circuit.instructions[1].name == "cx"
    assert circuit.instructions[1].qargs == [("q", 0), ("q", 1)]


def test_parameterized_custom_gate_macro():
    qasm_str = """
    OPENQASM 2.0;
    gate rot_pair(theta) a, b {
        rz(theta) a;
        rx(theta / 2) b;
    }
    
    qreg q[2];
    rot_pair(pi) q[0], q[1];
    """
    circuit = parse_qasm(qasm_str)
    assert circuit.total_gates == 2
    assert circuit.instructions[0].name == "rz"
    assert circuit.instructions[0].params[0] == pytest.approx(math.pi)
    assert circuit.instructions[1].name == "rx"
    assert circuit.instructions[1].params[0] == pytest.approx(math.pi / 2.0)


# ==============================================================================
# 6. Barrier, Reset, and Measurement
# ==============================================================================

def test_barrier_and_reset_and_measure():
    qasm_str = """
    OPENQASM 2.0;
    qreg q[2];
    creg c[2];
    
    reset q[0];
    barrier q[0], q[1];
    measure q[0] -> c[0];
    barrier;
    """
    circuit = parse_qasm(qasm_str)
    ops = circuit.count_ops()
    assert ops["reset"] == 1
    assert ops["barrier"] == 2
    assert ops["measure"] == 1
    # total_gates should exclude barrier, reset, measure
    assert circuit.total_gates == 0

    qc = circuit.to_quantum_circuit()
    assert isinstance(qc, QuantumCircuit)


# ==============================================================================
# 7. Error Handling & Edge Cases
# ==============================================================================

def test_unsupported_version_error():
    with pytest.raises(QASMParseError, match="Unsupported OPENQASM version"):
        parse_qasm("OPENQASM 3.0;\nqreg q[2];")


def test_malformed_register_error():
    with pytest.raises(QASMParseError, match="Malformed register declaration"):
        parse_qasm("OPENQASM 2.0;\nqreg q;")

    with pytest.raises(QASMParseError, match="must be positive integer"):
        parse_qasm("OPENQASM 2.0;\nqreg q[0];")

    with pytest.raises(QASMParseError, match="must be positive integer"):
        parse_qasm("OPENQASM 2.0;\nqreg q[-2];")


def test_duplicate_register_error():
    with pytest.raises(QASMParseError, match="Duplicate register identifier"):
        parse_qasm("OPENQASM 2.0;\nqreg q[2];\nqreg q[3];")


def test_qubit_out_of_bounds_error():
    with pytest.raises(QASMParseError, match="out of bounds"):
        parse_qasm("OPENQASM 2.0;\nqreg q[2];\nh q[2];")


def test_undeclared_register_error():
    with pytest.raises(QASMParseError, match="Undeclared quantum register"):
        parse_qasm("OPENQASM 2.0;\nqreg q[2];\nh unk[0];")


def test_duplicate_qubit_in_multigate_error():
    with pytest.raises(QASMParseError, match="Duplicate target qubit"):
        parse_qasm("OPENQASM 2.0;\nqreg q[2];\ncx q[0], q[0];")


def test_undefined_gate_error():
    with pytest.raises(QASMParseError, match="Undefined gate"):
        parse_qasm("OPENQASM 2.0;\nqreg q[2];\nsuper_magic q[0];")


def test_parameter_count_mismatch_error():
    with pytest.raises(QASMParseError, match="expects 1 parameter"):
        parse_qasm("OPENQASM 2.0;\nqreg q[2];\nrz(pi, 2) q[0];")

    with pytest.raises(QASMParseError, match="expects 0 parameter"):
        parse_qasm("OPENQASM 2.0;\nqreg q[2];\nh(pi) q[0];")


def test_qubit_count_mismatch_error():
    with pytest.raises(QASMParseError, match="expects 2 qubit"):
        parse_qasm("OPENQASM 2.0;\nqreg q[2];\ncx q[0];")


def test_unclosed_brace_error():
    with pytest.raises(QASMParseError, match="Unclosed curly brace"):
        parse_qasm("OPENQASM 2.0;\ngate test a { h a;")


def test_broadcast_dimension_mismatch_error():
    with pytest.raises(QASMParseError, match="Register dimension mismatch"):
        parse_qasm("OPENQASM 2.0;\nqreg a[2];\nqreg b[3];\ncx a, b;")


def test_measure_size_mismatch_error():
    with pytest.raises(QASMParseError, match="Measurement register size mismatch"):
        parse_qasm("OPENQASM 2.0;\nqreg q[2];\ncreg c[3];\nmeasure q -> c;")


def test_type_error_for_non_string():
    with pytest.raises(TypeError, match="must be a string"):
        parse_qasm(12345)  # type: ignore


# ==============================================================================
# 8. Memory Limit Validation & Summary
# ==============================================================================

def test_memory_limit_validation():
    circuit = ParsedQASMCircuit()
    circuit.qregs["q"] = 10
    assert circuit.validate_memory_limits() is True
    assert circuit.validate_memory_limits(max_allowed_ram_gb=1.0) is True

    # 36 qubits requires 1024 GB, exceeding 35 qubit hard threshold
    circuit_huge = ParsedQASMCircuit()
    circuit_huge.qregs["q"] = 36
    assert circuit_huge.validate_memory_limits() is False

    # 30 qubits = 16 GB, check threshold
    circuit_30 = ParsedQASMCircuit()
    circuit_30.qregs["q"] = 30
    assert circuit_30.validate_memory_limits(max_allowed_ram_gb=8.0) is False
    assert circuit_30.validate_memory_limits(max_allowed_ram_gb=32.0) is True


def test_circuit_summary():
    circuit = ParsedQASMCircuit(name="test_summary")
    circuit.qregs["q"] = 2
    circuit.instructions.append(QASMInstruction(name="h", qargs=[("q", 0)]))
    circuit.instructions.append(QASMInstruction(name="cx", qargs=[("q", 0), ("q", 1)]))
    
    summary = circuit.get_summary()
    assert summary["name"] == "test_summary"
    assert summary["num_qubits"] == 2
    assert summary["total_gates"] == 2
    assert summary["depth"] == 2
    assert summary["one_qubit_gates"] == 1
    assert summary["two_qubit_gates"] == 1


# ==============================================================================
# 9. Qiskit Roundtrip Interop
# ==============================================================================

def test_from_qiskit_interop():
    qc = QuantumCircuit(3, name="interop_qc")
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.5, 2)
    
    parsed = ParsedQASMCircuit.from_qiskit(qc)
    assert parsed.name == "interop_qc"
    assert parsed.num_qubits == 3
    assert parsed.total_gates == 3
    assert parsed.count_ops() == {"h": 1, "cx": 1, "rz": 1}

    # Re-export to Qiskit and verify gate count
    qc_back = parsed.to_quantum_circuit()
    assert qc_back.num_qubits == 3
    assert len(qc_back.data) == 3


# ==============================================================================
# 10. File Loading with Benchmark Fixtures
# ==============================================================================

def test_load_qasm_benchmark_files():
    benchmarks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks")
    
    expected_files = [
        "bell_state.qasm",
        "teleportation.qasm",
        "ghz_state.qasm",
        "qft_4qubit.qasm"
    ]
    
    for filename in expected_files:
        filepath = os.path.join(benchmarks_dir, filename)
        assert os.path.exists(filepath), f"Benchmark fixture missing: {filepath}"
        
        circuit = load_qasm(filepath)
        assert isinstance(circuit, ParsedQASMCircuit)
        assert circuit.num_qubits > 0
        assert circuit.total_gates > 0

        # Verify execution on simulator
        res = run_simulation(circuit, runs=1)
        assert res["success"] is True, f"Failed simulation on {filename}: {res.get('error')}"


def test_load_qasm_file_not_found():
    with pytest.raises(FileNotFoundError, match="OpenQASM file not found"):
        load_qasm("benchmarks/non_existent_circuit.qasm")
