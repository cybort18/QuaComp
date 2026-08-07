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
