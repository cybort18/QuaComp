import pytest
from unittest.mock import patch, MagicMock
from qiskit import QuantumCircuit
from src.profiler.gpu import (
    get_available_aer_devices,
    is_gpu_available,
    get_gpu_metadata,
    check_gpu_vram_safety
)
from src.profiler.memory import check_memory_safety
from src.engine.simulator import run_simulation

def test_get_available_aer_devices():
    """Verify get_available_aer_devices returns a non-empty list containing CPU."""
    devices = get_available_aer_devices()
    assert isinstance(devices, list)
    assert len(devices) > 0
    assert "CPU" in devices

def test_is_gpu_available_boolean():
    """Verify is_gpu_available returns a boolean."""
    res = is_gpu_available()
    assert isinstance(res, bool)

def test_get_gpu_metadata_schema():
    """Verify get_gpu_metadata returns all mandatory telemetry keys."""
    meta = get_gpu_metadata()
    assert isinstance(meta, dict)
    assert "gpu_name" in meta
    assert "has_gpu" in meta
    assert "aer_gpu_supported" in meta
    assert "total_vram_gb" in meta
    assert "backend_devices" in meta
    assert isinstance(meta["total_vram_gb"], (int, float))

def test_check_gpu_vram_safety_types():
    """Verify check_gpu_vram_safety enforces input argument types."""
    with pytest.raises(TypeError):
        check_gpu_vram_safety("10")
    with pytest.raises(TypeError):
        check_gpu_vram_safety(10, method=123)

def test_check_gpu_vram_safety_mocked_gpu():
    """Verify GPU VRAM safety evaluation under simulated GPU environments."""
    with patch("src.profiler.gpu.is_gpu_available", return_value=True):
        with patch("src.profiler.gpu.get_gpu_metadata", return_value={"total_vram_gb": 8.0}):
            # Small circuit (10 qubits = negligible VRAM)
            safe, msg = check_gpu_vram_safety(10, method="statevector")
            assert safe is True
            assert "SAFE" in msg
            
            # MPS circuit
            safe_mps, msg_mps = check_gpu_vram_safety(50, method="mps")
            assert safe_mps is True
            assert "efficient" in msg_mps

            # Oversized statevector (30 qubits = 16 GB, exceeds 8 GB VRAM)
            safe_big, msg_big = check_gpu_vram_safety(30, method="statevector")
            assert safe_big is False
            assert "CRITICAL" in msg_big

def test_check_memory_safety_with_gpu_device():
    """Verify check_memory_safety delegates to GPU safety when device is GPU."""
    with patch("src.profiler.gpu.is_gpu_available", return_value=False):
        safe, msg = check_memory_safety(10, method="statevector", device="GPU")
        assert safe is False
        assert "Qiskit Aer does not have GPU/CUDA backend support" in msg

def test_run_simulation_device_cpu():
    """Verify run_simulation executes correctly on CPU."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    res = run_simulation(qc, device="CPU", runs=1)
    assert res["success"] is True
    assert res["device"] == "CPU"
    assert res["latency"] > 0

def test_run_simulation_device_gpu_fallback_on_cpu_env():
    """Verify run_simulation gracefully reports lack of GPU backend on CPU-only environment."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    with patch("src.profiler.gpu.get_available_aer_devices", return_value=["CPU"]):
        res = run_simulation(qc, device="GPU", runs=1)
        if not is_gpu_available():
            assert res["success"] is False
            assert "GPU" in res["error"]
        else:
            assert res["device"] == "GPU"

def test_run_simulation_invalid_device():
    """Verify run_simulation raises ValueError on unsupported device name."""
    qc = QuantumCircuit(2)
    with pytest.raises(ValueError, match="Invalid device"):
        run_simulation(qc, device="TPU")
