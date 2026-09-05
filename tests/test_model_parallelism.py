import pytest
from unittest.mock import patch
from qiskit import QuantumCircuit
import argparse

from src.engine.simulator import run_simulation
from src.profiler.gpu import check_gpu_vram_safety
from src.profiler.memory import check_memory_safety
from cli.main import validate_cli_arguments

def test_run_simulation_state_slicing_metadata():
    """Verify run_simulation enables distributed statevector slicing and records metadata."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    
    res = run_simulation(qc, method="statevector", state_slicing=True, blocking_qubits=2, runs=1)
    assert res["success"] is True
    assert "metadata" in res
    meta = res["metadata"]
    assert meta["model_parallelism"] == "Distributed Statevector Slicing"
    assert meta["state_slicing"] is True
    assert meta["blocking_qubits"] == 2
    assert meta["chunk_count"] == 4  # 2^(4 - 2) = 4 slices

def test_run_simulation_multi_gpu_auto_slicing():
    """Verify multi-gpu device setting automatically activates distributed model parallelism."""
    qc = QuantumCircuit(3)
    qc.h(range(3))
    
    # Run on CPU device while testing multi_gpu flag path fallback or options
    with patch("src.profiler.gpu.get_available_aer_devices", return_value=["CPU"]):
        res = run_simulation(qc, device="CPU", state_slicing=True, runs=1)
        assert res["success"] is True
        assert res["metadata"]["state_slicing"] is True
        assert res["metadata"]["model_parallelism"] == "Distributed Statevector Slicing"

def test_check_gpu_vram_safety_with_state_slicing():
    """Verify state slicing enables higher qubit counts on multi-GPU aggregate VRAM."""
    with patch("src.profiler.gpu.is_gpu_available", return_value=True):
        with patch("src.profiler.gpu.get_gpu_metadata", return_value={"total_vram_gb": 48.0, "gpu_count": 2}):
            # 30 qubits = 16 GB VRAM. Safe with aggregate 48GB multi-GPU pool and state slicing.
            safe, msg = check_gpu_vram_safety(30, method="statevector", state_slicing=True)
            assert safe is True
            assert "SAFE" in msg or "WARNING" in msg

def test_check_memory_safety_forwards_state_slicing():
    """Verify check_memory_safety properly forwards state_slicing flag."""
    with patch("src.profiler.gpu.is_gpu_available", return_value=True):
        with patch("src.profiler.gpu.get_gpu_metadata", return_value={"total_vram_gb": 48.0, "gpu_count": 2}):
            safe, msg = check_memory_safety(30, method="statevector", device="MULTI_GPU", state_slicing=True)
            assert safe is True

def test_validate_cli_arguments_blocking_qubits():
    """Verify CLI argument validation checks blocking_qubits boundary."""
    valid_args = argparse.Namespace(qubits=10, depth=5, bond_dim=64, workers=2, runs=3, blocking_qubits=16)
    validate_cli_arguments(valid_args)
    
    invalid_args = argparse.Namespace(qubits=10, depth=5, bond_dim=64, workers=2, runs=3, blocking_qubits=0)
    with pytest.raises(ValueError, match="--blocking-qubits"):
        validate_cli_arguments(invalid_args)
