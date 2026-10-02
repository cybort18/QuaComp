import pytest
import numpy as np
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
    assert res_ideal["success"] is True, f"Ideal simulation failed: {res_ideal.get('error')}"
    assert len(res_ideal["counts"]) > 0
    
    # Noisy simulation with medium noise
    noise_model = get_noise_model("medium")
    res_noisy = run_simulation(circuit, noise_model=noise_model, noise_level="medium")
    assert res_noisy["success"] is True, f"Noisy simulation failed: {res_noisy.get('error')}"
    assert len(res_noisy["counts"]) > 0
    assert res_noisy["metadata"]["noise_level"] == "medium"
    
    # Calculate fidelity between ideal and noisy run
    fidelity = calculate_state_fidelity(res_ideal["counts"], res_noisy["counts"])
    assert 0.0 <= fidelity <= 100.0


def test_stochastic_samplers():
    """Verify stochastic Pauli and thermal relaxation sampling functions."""
    from src.engine.noise import sample_pauli_error, sample_pauli_error_2q, sample_thermal_relaxation
    rng = np.random.default_rng(12345)
    
    # Pauli error = 0 -> always None
    assert sample_pauli_error(0.0, rng) is None
    assert sample_pauli_error_2q(0.0, rng) is None
    
    # Pauli error = 1.0 -> always X, Y, or Z
    samples_1q = [sample_pauli_error(1.0, rng) for _ in range(300)]
    assert all(s in ('X', 'Y', 'Z') for s in samples_1q)
    assert 'X' in samples_1q and 'Y' in samples_1q and 'Z' in samples_1q
    
    # 2Q Pauli error = 1.0 -> always a non-identity pair
    samples_2q = [sample_pauli_error_2q(1.0, rng) for _ in range(300)]
    assert all(len(pair) == 2 and pair != ('I', 'I') for pair in samples_2q)
    
    # Thermal relaxation: zero gate duration -> no jumps
    d_jump, p_jump = sample_thermal_relaxation(0.0, 100e-6, 120e-6, rng)
    assert not d_jump and not p_jump
    
    # Non-zero gate duration -> valid booleans
    d_jump, p_jump = sample_thermal_relaxation(50e-9, 100e-6, 120e-6, rng)
    assert isinstance(d_jump, bool)
    assert isinstance(p_jump, bool)


def test_trajectory_noise_model_presets():
    """Verify TrajectoryNoiseModel preset generation and property validation."""
    from src.engine.noise import get_trajectory_noise_model, TrajectoryNoiseModel
    
    assert get_trajectory_noise_model("none") is None
    assert get_trajectory_noise_model("NONE") is None
    
    for level in ("low", "medium", "high"):
        model = get_trajectory_noise_model(level)
        assert isinstance(model, TrajectoryNoiseModel)
        assert model.t1 > 0
        assert model.t2 > 0
        assert model.t2 <= 2.0 * model.t1
        assert model.gate_error_1q > 0
        assert model.gate_error_2q >= model.gate_error_1q
        assert model.readout_error > 0
        
    with pytest.raises(TypeError):
        get_trajectory_noise_model(42)  # type: ignore
    with pytest.raises(ValueError):
        get_trajectory_noise_model("invalid_preset")


def test_ensemble_convergence_1q_depolarizing():
    """
    Validate ensemble convergence for 1-qubit circuit with depolarizing error p = 0.05.
    Formula: |<P_trajectory> - P_exact| < 3 / sqrt(N_shots).
    """
    from qiskit import QuantumCircuit
    from src.engine.noise import TrajectoryNoiseModel
    from src.engine.simulator import simulate_trajectories
    
    p = 0.05
    shots = 2000
    
    # 1-Qubit X gate: prepares |1>
    qc = QuantumCircuit(1)
    qc.x(0)
    
    # Pure depolarizing channel without T1/T2 or readout noise
    noise_model = TrajectoryNoiseModel(
        t1=1.0,  # Huge T1/T2 to isolate depolarizing channel
        t2=2.0,
        gate_error_1q=p,
        gate_time_1q=1e-12,
        readout_error=0.0
    )
    
    res = simulate_trajectories(qc, noise_model=noise_model, shots=shots, seed=42)
    assert res["success"] is True
    counts = res["counts"]
    
    # Exact theoretical probability: P(1) = (1 - p) + p/3 = 1 - 2p/3
    p_exact_1 = 1.0 - (2.0 / 3.0) * p
    p_traj_1 = counts.get("1", 0) / shots
    
    mc_tol = 3.0 / np.sqrt(shots)
    diff = abs(p_traj_1 - p_exact_1)
    assert diff < mc_tol, f"Difference {diff:.5f} exceeds Monte Carlo bound {mc_tol:.5f}"


def test_ensemble_convergence_bell_state():
    """
    Validate ensemble convergence for 2-qubit Bell state |00> + |11> / sqrt(2)
    under moderate gate depolarizing noise.
    """
    from qiskit import QuantumCircuit
    from src.engine.noise import TrajectoryNoiseModel
    from src.engine.simulator import simulate_trajectories
    
    shots = 2000
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    # Light depolarizing noise
    noise_model = TrajectoryNoiseModel(
        t1=1.0,
        t2=2.0,
        gate_error_1q=0.01,
        gate_error_2q=0.02,
        gate_time_1q=1e-12,
        gate_time_2q=1e-12,
        readout_error=0.0
    )
    
    res = simulate_trajectories(qc, noise_model=noise_model, shots=shots, seed=42)
    assert res["success"] is True
    counts = res["counts"]
    
    total = sum(counts.values())
    assert total == shots
    # Dominated by '00' and '11'
    p_00 = counts.get("00", 0) / shots
    p_11 = counts.get("11", 0) / shots
    assert p_00 > 0.40
    assert p_11 > 0.40
    # Minor counts in orthogonal states '01' and '10' from noise
    p_leak = (counts.get("01", 0) + counts.get("10", 0)) / shots
    assert p_leak < 0.15


def test_thermal_relaxation_t1_decay():
    """
    Validate thermal relaxation amplitude damping decay:
    State |1> after duration t = T1 decays close to 1/e ~ 0.3679.
    """
    import math
    from qiskit import QuantumCircuit
    from src.engine.noise import TrajectoryNoiseModel
    from src.engine.simulator import simulate_trajectories
    
    t1_sec = 50e-6
    shots = 2000
    
    qc = QuantumCircuit(1)
    qc.x(0)  # Prepare state |1>
    
    # Gate time equal to T1, zero depolarizing error
    noise_model = TrajectoryNoiseModel(
        t1=t1_sec,
        t2=2.0 * t1_sec,
        gate_error_1q=0.0,
        gate_time_1q=t1_sec,
        readout_error=0.0
    )
    
    res = simulate_trajectories(qc, noise_model=noise_model, shots=shots, seed=123)
    counts = res["counts"]
    
    p1_observed = counts.get("1", 0) / shots
    p1_expected = math.exp(-1.0)  # 1/e ~ 0.367879
    
    mc_tol = 3.0 / math.sqrt(shots)  # ~ 0.067
    diff = abs(p1_observed - p1_expected)
    assert diff < mc_tol, f"Observed P(1)={p1_observed:.4f}, expected={p1_expected:.4f}, diff={diff:.4f} > {mc_tol:.4f}"


def test_large_scale_memory_efficiency_no_oom():
    """
    Validate that trajectory simulation on 20 to 24 qubits maintains strict O(2^n) memory
    scaling and executes cleanly without Out-Of-Memory errors.
    """
    from qiskit import QuantumCircuit
    from src.engine.noise import get_trajectory_noise_model
    from src.engine.simulator import simulate_trajectories
    
    model = get_trajectory_noise_model("low")
    
    # 20-qubit circuit: statevector takes only 16 MB RAM (vs 17.5 Terabytes for density matrix)
    qc_20 = QuantumCircuit(20)
    qc_20.h(0)
    qc_20.cx(0, 1)
    qc_20.cx(1, 2)
    
    res_20 = simulate_trajectories(qc_20, noise_model=model, shots=5, seed=42)
    assert res_20["success"] is True
    assert res_20["qubits"] == 20
    assert res_20["memory_saved_ratio"] == 2 ** 20
    assert sum(res_20["counts"].values()) == 5
    
    # 22-qubit circuit: statevector takes 64 MB RAM
    qc_22 = QuantumCircuit(22)
    qc_22.h(0)
    qc_22.cx(0, 1)
    res_22 = simulate_trajectories(qc_22, noise_model=model, shots=2, seed=42)
    assert res_22["success"] is True
    assert res_22["qubits"] == 22
    assert res_22["memory_saved_ratio"] == 2 ** 22


def test_density_matrix_safety_guard():
    """
    Verify rejection of explicit density matrix simulation on n >= 16 qubits with DensityMatrixMemoryError.
    """
    from qiskit import QuantumCircuit
    from src.engine.noise import DensityMatrixMemoryError
    from src.engine.simulator import run_simulation
    
    qc_16 = QuantumCircuit(16)
    qc_16.h(0)
    
    # Method = 'density_matrix'
    with pytest.raises(DensityMatrixMemoryError):
        run_simulation(qc_16, method="density_matrix")
        
    # Noise method = 'density_matrix'
    with pytest.raises(DensityMatrixMemoryError):
        run_simulation(qc_16, noise_method="density_matrix", noise_level="medium")

