import pytest
from qiskit_aer.noise import NoiseModel
from src.engine.noise import get_noise_model, calculate_state_fidelity, calculate_overhead_ratio
from src.engine.circuits import generate_shallow_circuit
from src.engine.simulator import run_simulation

def test_get_noise_model_presets():
    assert get_noise_model("none") is None
    assert get_noise_model("NONE") is None
    
    model_low = get_noise_model("low")
    assert isinstance(model_low, NoiseModel)
    
    model_med = get_noise_model("medium")
    assert isinstance(model_med, NoiseModel)
    
    model_high = get_noise_model("high")
    assert isinstance(model_high, NoiseModel)
    
    # Test invalid options
    with pytest.raises(TypeError):
        get_noise_model(123)  # type: ignore
    with pytest.raises(ValueError):
        get_noise_model("ultra_high")

def test_calculate_state_fidelity():
    ideal = {"00": 1000}
    noisy_same = {"00": 1000}
    noisy_orthogonal = {"11": 1000}
    noisy_mixed = {"00": 800, "01": 200}
    
    # Identical states -> 100% fidelity
    fid_same = calculate_state_fidelity(ideal, noisy_same)
    assert abs(fid_same - 100.0) < 1e-5
    
    # Orthogonal states -> 0% fidelity
    fid_orth = calculate_state_fidelity(ideal, noisy_orthogonal)
    assert abs(fid_orth - 0.0) < 1e-5
    
    # Partial overlap -> between 0% and 100%
    fid_mix = calculate_state_fidelity(ideal, noisy_mixed)
    assert 0.0 < fid_mix < 100.0
    assert abs(fid_mix - 80.0) < 1e-5
    
    # Empty counts -> 100% default
    assert calculate_state_fidelity({}, {}) == 100.0
    
    # Invalid types
    with pytest.raises(TypeError):
        calculate_state_fidelity([], {})  # type: ignore

def test_calculate_overhead_ratio():
    assert calculate_overhead_ratio(1.0, 1.5) == 50.0
    assert calculate_overhead_ratio(2.0, 2.0) == 0.0
    assert calculate_overhead_ratio(2.0, 1.0) == 0.0  # No negative overhead ratio
    assert calculate_overhead_ratio(0.0, 1.0) == 0.0
    
    with pytest.raises(TypeError):
        calculate_overhead_ratio("1.0", 1.5)  # type: ignore
    with pytest.raises(ValueError):
        calculate_overhead_ratio(-1.0, 1.5)

def test_simulation_with_noise():
    circuit = generate_shallow_circuit(5)
    
    # Ideal simulation
    res_ideal = run_simulation(circuit, noise_model=None, noise_level="none")
    assert res_ideal["success"] is True
    assert len(res_ideal["counts"]) > 0
    
    # Noisy simulation with medium noise
    noise_model = get_noise_model("medium")
    res_noisy = run_simulation(circuit, noise_model=noise_model, noise_level="medium")
    assert res_noisy["success"] is True
    assert len(res_noisy["counts"]) > 0
    assert res_noisy["metadata"]["noise_level"] == "medium"
    
    # Calculate fidelity between ideal and noisy run
    fidelity = calculate_state_fidelity(res_ideal["counts"], res_noisy["counts"])
    assert 0.0 <= fidelity <= 100.0
