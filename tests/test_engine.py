import pytest
from qiskit import QuantumCircuit
from src.engine.circuits import (
    generate_shallow_circuit,
    generate_deep_circuit,
    generate_qft_circuit
)
from src.engine.simulator import run_simulation

def test_generate_shallow_circuit():
    # Test valid circuits
    qc = generate_shallow_circuit(3)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 3
    
    qc_single = generate_shallow_circuit(1)
    assert qc_single.num_qubits == 1

    # Test invalid inputs
    with pytest.raises(TypeError):
        generate_shallow_circuit("three")  # type: ignore
    with pytest.raises(ValueError):
        generate_shallow_circuit(0)
    with pytest.raises(ValueError):
        generate_shallow_circuit(-5)

def test_generate_deep_circuit():
    # Test valid deep circuit
    qc = generate_deep_circuit(4, 5)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 4
    # Deep circuit should have single-qubit gates and CNOT chain gates.
    # Depth 5 means 5 alternating layers.
    assert qc.depth() > 0

    # Test depth 0 (empty circuit)
    qc_empty = generate_deep_circuit(2, 0)
    assert qc_empty.depth() == 0

    # Test invalid inputs
    with pytest.raises(TypeError):
        generate_deep_circuit("four", 5)  # type: ignore
    with pytest.raises(TypeError):
        generate_deep_circuit(4, "five")  # type: ignore
    with pytest.raises(ValueError):
        generate_deep_circuit(0, 5)
    with pytest.raises(ValueError):
        generate_deep_circuit(4, -1)

def test_generate_qft_circuit():
    # Test valid QFT circuit
    qc = generate_qft_circuit(4)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 4

    # Test invalid inputs
    with pytest.raises(TypeError):
        generate_qft_circuit(4.5)  # type: ignore
    with pytest.raises(ValueError):
        generate_qft_circuit(0)
    with pytest.raises(ValueError):
        generate_qft_circuit(-1)

def test_run_simulation_success():
    qc = generate_shallow_circuit(3)
    res = run_simulation(qc, runs=3)
    
    assert isinstance(res, dict)
    assert res["success"] is True, f"Simulation failed with error: {res.get('error')}"
    assert res["error"] is None
    assert isinstance(res["latency"], float)
    assert res["latency"] > 0
    assert "metadata" in res
    assert res["metadata"]["success"] is True
    
    # Statistical multi-run assertions
    assert len(res["latencies"]) == 3
    assert res["mean_latency"] > 0
    assert res["median_latency"] > 0
    assert res["std_latency"] >= 0
    assert res["runs_count"] == 3

def test_run_simulation_type_error():
    with pytest.raises(TypeError):
        run_simulation("not a circuit")  # type: ignore
    with pytest.raises(ValueError):
        run_simulation(generate_shallow_circuit(2), runs=0)
