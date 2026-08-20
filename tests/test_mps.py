import pytest
from qiskit import QuantumCircuit
from src.engine.mps import get_mps_configuration, calculate_mps_ram_savings
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
    assert result["success"] is True
    assert result["error"] is None
    assert result["latency"] > 0
    assert "metadata" in result
    assert result["metadata"]["method"] in ("mps", "matrix_product_state")
    assert result["metadata"]["bond_dimension"] == 64
    assert "aer_simulator" in result["metadata"]["backend_name"].lower()
