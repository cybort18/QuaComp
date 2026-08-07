import pytest
from src.scorer.calculator import calculate_qsim_score, categorize_score

def test_calculate_qsim_score_valid():
    # Formula: (2^max_qubits * 10) + (total_gates / execution_time)
    # Qubits=10: 2^10 * 10 = 10240
    # Gates=100, Time=2.0: 100 / 2.0 = 50.0
    # Expected: 10240 + 50 = 10290.0
    assert calculate_qsim_score(10, 100, 2.0) == 10290.0
    
    # Qubits=0: 2^0 * 10 = 10
    # Expected: 10 + (0 / 1.0) = 10.0
    assert calculate_qsim_score(0, 0, 1.0) == 10.0

def test_calculate_qsim_score_division_by_zero():
    # If time is 0, it should use 1e-9
    # Qubits=4: 2^4 * 10 = 160
    # Gates=5, Time=0.0: 5 / 1e-9 = 5,000,000,000
    # Expected: 160 + 5,000,000,000 = 5000000160.0
    assert calculate_qsim_score(4, 5, 0.0) == 5000000160.0
    
    # Negative time should be treated the same as 0 (using 1e-9)
    assert calculate_qsim_score(4, 5, -10.5) == 5000000160.0

def test_calculate_qsim_score_type_errors():
    with pytest.raises(TypeError):
        calculate_qsim_score("10", 100, 2.0)  # type: ignore
    with pytest.raises(TypeError):
        calculate_qsim_score(10, "100", 2.0)  # type: ignore
    with pytest.raises(TypeError):
        calculate_qsim_score(10, 100, "2.0")  # type: ignore

def test_calculate_qsim_score_value_errors():
    with pytest.raises(ValueError):
        calculate_qsim_score(-1, 100, 2.0)
    with pytest.raises(ValueError):
        calculate_qsim_score(10, -5, 2.0)

def test_categorize_score():
    assert categorize_score(50000.0) == "Entry-Level"
    assert categorize_score(99999.9) == "Entry-Level"
    
    assert categorize_score(100000.0) == "Mid-Range"
    assert categorize_score(500000.0) == "Mid-Range"
    assert categorize_score(999999.9) == "Mid-Range"
    
    assert categorize_score(1000000.0) == "High-Performance"
    assert categorize_score(25000000.0) == "High-Performance"
    assert categorize_score(49999999.9) == "High-Performance"
    
    assert categorize_score(50000000.0) == "Extreme Workstation"
    assert categorize_score(100000000.0) == "Extreme Workstation"
