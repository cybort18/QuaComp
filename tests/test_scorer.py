import pytest
from src.scorer.calculator import (
    calculate_qsim_score, 
    calculate_qsi_score,
    categorize_score, 
    calculate_scoring_breakdown,
    calculate_gate_throughput_kgps,
    calculate_energy_delay_product,
    verify_quantum_volume_status
)

def test_calculate_qsi_score_valid():
    # Formula: n * [w1 + w2 * log10(max(T, 1.0)) + w3 * (fidelity / 100)]
    # Qubits=10, Gates=100, Time=2.0s -> Throughput T = 50 gates/sec
    # log10(50) = 1.69897
    # Bracket: 100 + (50 * 1.69897) + (25 * 1.0) = 209.9485
    # Score: 10 * 209.9485 = 2099.49
    score = calculate_qsim_score(10, 100, 2.0)
    assert score == 2099.49
    
    # Alias calculate_qsi_score should return the identical value
    assert calculate_qsi_score(10, 100, 2.0) == 2099.49
    
    # Qubits=0, Gates=0: Score should be 0.0
    assert calculate_qsim_score(0, 0, 1.0) == 0.0
    
    # Qubits=0, Gates=100, Time=1.0s (T = 100): log10(100) = 2.0 -> 50 * 2 = 100.0
    assert calculate_qsim_score(0, 100, 1.0) == 100.0


def test_qsi_throughput_differentiation_on_same_qubits():
    """Verify that a faster simulation on the same number of qubits achieves higher score."""
    # Base system: 10 qubits, 100 gates in 2.0s (50 gates/s)
    score_slow = calculate_qsim_score(10, 100, 2.0)
    
    # Fast system: 10 qubits, 10000 gates in 2.0s (5000 gates/s)
    score_fast = calculate_qsim_score(10, 10000, 2.0)
    
    # Faster throughput must produce a distinctly higher score (~1000 pts higher)
    assert score_fast > score_slow
    assert score_fast - score_slow == pytest.approx(1000.0, abs=1.0)


def test_qsi_fidelity_penalty():
    """Verify that lower quantum state fidelity reduces the QSI score."""
    score_ideal = calculate_qsim_score(10, 100, 2.0, fidelity=100.0)
    score_noisy = calculate_qsim_score(10, 100, 2.0, fidelity=60.0)
    
    assert score_ideal > score_noisy
    # 40% fidelity drop with w3=25 on 10 qubits = 10 * (25 * 0.4) = 100 pts
    assert score_ideal - score_noisy == pytest.approx(100.0, abs=0.5)


def test_calculate_scoring_breakdown():
    breakdown = calculate_scoring_breakdown(10, 100, 2.0, fidelity=100.0, energy_joules=5.0)
    assert isinstance(breakdown, dict)
    assert breakdown["capacity_metric"] == 1024.0
    assert breakdown["throughput_metric"] == 50.0
    assert breakdown["throughput_kgps"] == 0.05
    assert breakdown["composite_score"] == 2099.49
    assert breakdown["qsi_score"] == 2099.49
    assert breakdown["score_type"] == "QuaComp Synthetic Index (QSI)"
    assert breakdown["fidelity_factor"] == 1.0
    assert breakdown["energy_delay_product"] == pytest.approx(10.0, rel=1e-4)


def test_calculate_qsim_score_safe_division():
    # If time is 0.0, it should safely clamp to 1e-6 without runaway billions
    score_zero_time = calculate_qsim_score(4, 5, 0.0)
    assert score_zero_time > 0.0
    assert score_zero_time < 5000.0  # Well-behaved bounded score
    
    # Negative time should be treated safely (clamped to 1e-6)
    score_neg_time = calculate_qsim_score(4, 5, -10.5)
    assert score_neg_time == score_zero_time


def test_calculate_qsim_score_type_errors():
    with pytest.raises(TypeError):
        calculate_qsim_score("10", 100, 2.0)  # type: ignore
    with pytest.raises(TypeError):
        calculate_qsim_score(10, "100", 2.0)  # type: ignore
    with pytest.raises(TypeError):
        calculate_qsim_score(10, 100, "2.0")  # type: ignore
    with pytest.raises(TypeError):
        calculate_qsim_score(10, 100, 2.0, fidelity="100")  # type: ignore


def test_calculate_qsim_score_value_errors():
    with pytest.raises(ValueError):
        calculate_qsim_score(-1, 100, 2.0)
    with pytest.raises(ValueError):
        calculate_qsim_score(10, -5, 2.0)


def test_academic_metrics_gate_throughput():
    assert calculate_gate_throughput_kgps(1000, 1.0) == 1.0
    assert calculate_gate_throughput_kgps(5000, 2.0) == 2.5
    assert calculate_gate_throughput_kgps(100, 0.5) == 0.2
    
    with pytest.raises(TypeError):
        calculate_gate_throughput_kgps("1000", 1.0)  # type: ignore
    with pytest.raises(ValueError):
        calculate_gate_throughput_kgps(-10, 1.0)


def test_academic_metrics_energy_delay_product():
    assert calculate_energy_delay_product(2.0, 15.0) == 30.0
    assert calculate_energy_delay_product(0.5, 10.0) == 5.0
    assert calculate_energy_delay_product(2.0, None) is None
    assert calculate_energy_delay_product(None, 15.0) is None
    
    with pytest.raises(TypeError):
        calculate_energy_delay_product("2.0", 10.0)  # type: ignore
    with pytest.raises(ValueError):
        calculate_energy_delay_product(-1.0, 10.0)


def test_academic_metrics_quantum_volume_status():
    # Pass case: h_prob > 2/3 (0.6667)
    res_pass = verify_quantum_volume_status(0.85)
    assert res_pass["qv_certified"] is True
    assert res_pass["status"] == "PASSED"
    assert res_pass["margin"] > 0
    
    # Fail case: h_prob <= 2/3
    res_fail = verify_quantum_volume_status(0.55)
    assert res_fail["qv_certified"] is False
    assert res_fail["status"] == "FAILED"
    
    # 2-Sigma confidence bound check: h_prob is 0.70 but 2-sigma lower bound is 0.64 (< 2/3)
    res_bound = verify_quantum_volume_status(0.70, confidence_2sigma=0.64)
    assert res_bound["qv_certified"] is False
    assert res_bound["lower_bound_2sigma"] == 0.64
    
    with pytest.raises(TypeError):
        verify_quantum_volume_status("0.85")  # type: ignore
    with pytest.raises(ValueError):
        verify_quantum_volume_status(1.5)


def test_categorize_score():
    assert categorize_score(1500.0) == "Entry-Level"
    assert categorize_score(2499.9) == "Entry-Level"
    
    assert categorize_score(2500.0) == "Mid-Range"
    assert categorize_score(4500.0) == "Mid-Range"
    assert categorize_score(5999.9) == "Mid-Range"
    
    assert categorize_score(6000.0) == "High-Performance"
    assert categorize_score(8500.0) == "High-Performance"
    assert categorize_score(9999.9) == "High-Performance"
    
    assert categorize_score(10000.0) == "Extreme Workstation"
    assert categorize_score(15000.0) == "Extreme Workstation"
