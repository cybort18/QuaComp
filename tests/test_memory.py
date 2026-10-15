from unittest.mock import patch
import pytest
from src.profiler.memory import estimate_qubit_ram, check_memory_safety

def test_estimate_qubit_ram():
    # 0 qubits = 1 * 16 = 16 bytes
    assert estimate_qubit_ram(0) == 16
    # 4 qubits = 16 * 16 = 256 bytes
    assert estimate_qubit_ram(4) == 256
    # 10 qubits = 1024 * 16 = 16384 bytes
    assert estimate_qubit_ram(10) == 16384
    # 20 qubits = 1048576 * 16 = 16777216 bytes
    assert estimate_qubit_ram(20) == 16777216

    # Test error cases
    with pytest.raises(ValueError):
        estimate_qubit_ram(-1)
    with pytest.raises(TypeError):
        estimate_qubit_ram("not an int") # type: ignore

class MockVirtualMemory:
    def __init__(self, available):
        self.available = available

@patch("psutil.virtual_memory")
def test_check_memory_safety_safe(mock_vm):
    # Set available memory to 1 GB (1024^3 bytes)
    available_ram = 1024 ** 3
    mock_vm.return_value = MockVirtualMemory(available=available_ram)

    # 20 qubits requires 16 MB, which is way below 85% of 1 GB
    is_safe, message = check_memory_safety(20)
    assert is_safe is True
    assert "SAFE" in message
    assert "0.0156 GB" in message

@patch("psutil.virtual_memory")
def test_check_memory_safety_warning(mock_vm):
    # Set available memory to 18 MB (18 * 1024^2 bytes)
    # 85% of 18 MB is 15.3 MB.
    # 20 qubits requires 16 MB, which is between 15.3 MB and 18 MB.
    available_ram = 18 * (1024 ** 2)
    mock_vm.return_value = MockVirtualMemory(available=available_ram)

    is_safe, message = check_memory_safety(20)
    assert is_safe is False
    assert "WARNING" in message
    assert "0.0156 GB" in message
    assert "exceeds 85% safety threshold" in message

@patch("psutil.virtual_memory")
def test_check_memory_safety_critical(mock_vm):
    # Set available memory to 10 MB (10 * 1024^2 bytes)
    # 20 qubits requires 16 MB, which is > 10 MB.
    available_ram = 10 * (1024 ** 2)
    mock_vm.return_value = MockVirtualMemory(available=available_ram)

    is_safe, message = check_memory_safety(20)
    assert is_safe is False
    assert "CRITICAL" in message
    assert "exceeds available physical RAM" in message

def test_check_memory_safety_invalid_input():
    is_safe, message = check_memory_safety(-5)
    assert is_safe is False
    assert "Number of qubits must be non-negative" in message

def test_check_memory_safety_multi_gpu():
    with patch("src.profiler.gpu.is_gpu_available", return_value=True):
        with patch("src.profiler.gpu.get_gpu_metadata", return_value={"total_vram_gb": 32.0, "gpu_count": 2, "multi_gpu_supported": True}):
            is_safe, message = check_memory_safety(25, method="statevector", device="MULTI_GPU")
            assert is_safe is True
            assert "SAFE" in message

@patch("psutil.virtual_memory")
def test_check_memory_safety_mps_low_ram(mock_vm):
    # Available RAM less than 200MB baseline
    mock_vm.return_value = MockVirtualMemory(available=50 * 1024 * 1024)
    is_safe, message = check_memory_safety(40, method="mps")
    assert is_safe is False
    assert "CRITICAL" in message
    assert "lower than the baseline threshold required for MPS" in message


def test_calculate_effective_memory_bandwidth():
    from src.profiler.telemetry import calculate_effective_memory_bandwidth

    # 10 qubits, 100 gates, 0.05 seconds
    # Total bytes = 100 * (2^10) * 32 = 3,276,800 bytes
    # Effective BW = 3,276,800 / (0.05 * 10^9) = 0.065536 GB/s -> 0.0655 GB/s
    res = calculate_effective_memory_bandwidth(10, 100, 0.05)
    assert res["num_qubits"] == 10
    assert res["total_gates"] == 100
    assert res["total_bytes_transferred"] == 3276800.0
    assert res["effective_bandwidth_gbps"] == 0.0655
    assert res["arithmetic_intensity_flops_per_byte"] == 0.4375

    # Safe division clamp for 0 time
    res_zero_time = calculate_effective_memory_bandwidth(4, 10, 0.0)
    assert res_zero_time["execution_time_seconds"] == 1e-9
    assert res_zero_time["effective_bandwidth_gbps"] > 0.0

    # Type and value errors
    with pytest.raises(TypeError):
        calculate_effective_memory_bandwidth("10", 100, 0.05)  # type: ignore
    with pytest.raises(TypeError):
        calculate_effective_memory_bandwidth(10, "100", 0.05)  # type: ignore
    with pytest.raises(TypeError):
        calculate_effective_memory_bandwidth(10, 100, "0.05")  # type: ignore
    with pytest.raises(ValueError):
        calculate_effective_memory_bandwidth(-1, 100, 0.05)
    with pytest.raises(ValueError):
        calculate_effective_memory_bandwidth(10, -5, 0.05)
    with pytest.raises(ValueError):
        calculate_effective_memory_bandwidth(10, 100, -0.01)
