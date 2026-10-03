import os
import pytest
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from src.engine.physical_noise import (
    PhysicalNoiseModel,
    verify_kraus_completeness,
    generate_kraus_amplitude_damping,
    generate_kraus_phase_damping,
    generate_thermal_relaxation_kraus,
    generate_kraus_depolarizing,
    generate_kraus_2q_depolarizing
)


@pytest.fixture
def sample_profile_path():
    path = os.path.join("results", "noise_profiles", "ibm_brisbane_sample.json")
    assert os.path.exists(path), f"Sample calibration profile must exist at {path}"
    return path


def test_kraus_completeness_helper():
    """Verify verify_kraus_completeness accurately evaluates completeness."""
    # Identity is complete
    I = np.eye(2, dtype=np.complex128)
    assert verify_kraus_completeness([I])
    
    # Sub-unitary matrix is incomplete
    incomplete = [0.5 * I]
    assert not verify_kraus_completeness(incomplete)
    assert not verify_kraus_completeness([])


def test_amplitude_damping_kraus():
    """Verify amplitude damping Kraus operators satisfy trace preservation."""
    kraus = generate_kraus_amplitude_damping(t1_us=200.0, gate_time_ns=35.0)
    assert len(kraus) == 2
    assert kraus[0].shape == (2, 2)
    assert kraus[1].shape == (2, 2)
    assert verify_kraus_completeness(kraus)


def test_phase_damping_kraus():
    """Verify phase damping Kraus operators satisfy trace preservation."""
    kraus = generate_kraus_phase_damping(t_phi_us=150.0, gate_time_ns=35.0)
    assert len(kraus) == 2
    assert kraus[0].shape == (2, 2)
    assert kraus[1].shape == (2, 2)
    assert verify_kraus_completeness(kraus)


def test_thermal_relaxation_kraus():
    """Verify combined thermal relaxation Kraus operators satisfy trace preservation and physical bounds."""
    # Standard realistic regime: T2 < 2*T1
    kraus = generate_thermal_relaxation_kraus(t1_us=240.0, t2_us=160.0, gate_time_ns=35.0)
    assert len(kraus) == 4
    assert verify_kraus_completeness(kraus)
    
    # Boundary regime: T2 = 2*T1 (pure dephasing is zero)
    kraus_bound = generate_thermal_relaxation_kraus(t1_us=200.0, t2_us=400.0, gate_time_ns=35.0)
    assert verify_kraus_completeness(kraus_bound)
    
    # Physical violation regime: T2 > 2*T1 (must clamp to 2*T1 without error)
    kraus_clamped = generate_thermal_relaxation_kraus(t1_us=200.0, t2_us=500.0, gate_time_ns=35.0)
    assert verify_kraus_completeness(kraus_clamped)


def test_depolarizing_kraus():
    """Verify single and two-qubit depolarizing Kraus operators satisfy trace preservation."""
    k_1q = generate_kraus_depolarizing(p=0.005)
    assert len(k_1q) == 4
    assert verify_kraus_completeness(k_1q)
    
    k_2q = generate_kraus_2q_depolarizing(p=0.015)
    assert len(k_2q) == 16
    assert k_2q[0].shape == (4, 4)
    assert verify_kraus_completeness(k_2q)


def test_physical_noise_model_parser(sample_profile_path):
    """Verify loading from JSON file and property extraction."""
    model = PhysicalNoiseModel.from_json(sample_profile_path)
    assert model.backend_name == "ibm_brisbane"
    assert model.provider == "IBM Quantum"
    assert model.num_qubits == 127
    assert model.gate_time_1q_ns == 35.5
    assert model.gate_time_2q_ns == 320.0
    
    # Qubit 0 properties
    q0 = model.get_qubit_properties(0)
    assert q0["T1_us"] == 248.5
    assert q0["T2_us"] == 165.2
    assert q0["readout_error"] == 0.0118
    assert q0["single_qubit_gate_error"] == 0.000182
    
    # Readout confusion matrix
    M = model.get_readout_confusion_matrix(0)
    assert M.shape == (2, 2)
    # Column sums must be 1.0 (probabilities of observing 0 or 1 given state 0/1)
    np.testing.assert_allclose(np.sum(M, axis=0), [1.0, 1.0], atol=1e-12)
    
    # Couplings
    err_01 = model.get_coupling_error(0, 1)
    assert err_01 == 0.0072
    
    # Unregistered qubit fallback
    q999 = model.get_qubit_properties(999)
    assert q999["T1_us"] > 0
    assert q999["T2_us"] > 0


def test_physical_noise_model_file_not_found():
    """Verify error on nonexistent calibration file."""
    with pytest.raises(FileNotFoundError):
        PhysicalNoiseModel.from_json("nonexistent_qpu_profile_xyz.json")
        
    with pytest.raises(TypeError):
        PhysicalNoiseModel.from_json(12345)  # type: ignore


def test_to_qiskit_noise_model_and_simulation(sample_profile_path):
    """Verify conversion to Qiskit Aer NoiseModel and execution with AerSimulator."""
    model = PhysicalNoiseModel.from_json(sample_profile_path)
    
    # Build noise model for active qubits 0 and 1
    aer_noise = model.to_qiskit_noise_model(active_qubits=[0, 1])
    assert aer_noise is not None
    
    # Run a simple 2-qubit Bell circuit with the noise model
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    
    sim = AerSimulator(noise_model=aer_noise)
    t_qc = transpile(qc, sim)
    result = sim.run(t_qc, shots=500).result()
    counts = result.get_counts()
    
    assert len(counts) > 0
    assert sum(counts.values()) == 500
    # Because of real QPU noise, minor counts in '01' and '10' may appear
    assert "00" in counts or "11" in counts


def test_to_trajectory_noise_model_and_simulation(sample_profile_path):
    """Verify conversion to TrajectoryNoiseModel and execution with MCWF simulate_trajectories."""
    from src.engine.simulator import simulate_trajectories
    from src.engine.noise import TrajectoryNoiseModel
    
    model = PhysicalNoiseModel.from_json(sample_profile_path)
    traj_model = model.to_trajectory_noise_model(active_qubits=[0, 1])
    assert isinstance(traj_model, TrajectoryNoiseModel)
    assert traj_model.name == "Trajectory:ibm_brisbane"
    assert traj_model.gate_time_1q > 0
    assert traj_model.gate_time_2q > 0
    
    # Run a simple 2-qubit Bell circuit with the trajectory noise model
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    res = simulate_trajectories(qc, noise_model=traj_model, shots=500, seed=42)
    assert res["success"] is True
    counts = res["counts"]
    assert len(counts) > 0
    assert sum(counts.values()) == 500
    assert "00" in counts and "11" in counts


def test_simulate_trajectories_parallelization_and_seed_reproducibility(sample_profile_path):
    """
    Verify trajectory parallelization across multiple worker processes with SeedSequence:
    1. Independent parallel execution with num_workers=2 produces identical results when given the same seed.
    2. Statistical output counts are consistent with expected Bell state distribution ('00' and '11' dominate).
    3. Trajectory returns metadata with workers=2.
    """
    from src.engine.simulator import simulate_trajectories
    
    model = PhysicalNoiseModel.from_json(sample_profile_path)
    traj_model = model.to_trajectory_noise_model(active_qubits=[0, 1])
    
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    shots = 400
    
    # Run 1: Parallel with num_workers=2, seed=42
    res_parallel_1 = simulate_trajectories(qc, noise_model=traj_model, shots=shots, seed=42, num_workers=2)
    assert res_parallel_1["success"] is True
    assert res_parallel_1["workers"] == 2
    assert sum(res_parallel_1["counts"].values()) == shots
    
    # Run 2: Parallel with same seed=42 and num_workers=2 -> must be 100% bit-for-bit identical
    res_parallel_2 = simulate_trajectories(qc, noise_model=traj_model, shots=shots, seed=42, num_workers=2)
    assert res_parallel_1["counts"] == res_parallel_2["counts"]
    
    # Run 3: Serial with num_workers=1, shots=shots
    res_serial = simulate_trajectories(qc, noise_model=traj_model, shots=shots, seed=42, num_workers=1)
    assert res_serial["success"] is True
    assert res_serial["workers"] == 1
    assert sum(res_serial["counts"].values()) == shots
    
    # Both parallel and serial show Bell state physics: dominant '00' and '11'
    p00_par = res_parallel_1["counts"].get("00", 0) / shots
    p11_par = res_parallel_1["counts"].get("11", 0) / shots
    assert p00_par > 0.35 and p11_par > 0.35
    
    p00_ser = res_serial["counts"].get("00", 0) / shots
    p11_ser = res_serial["counts"].get("11", 0) / shots
    assert p00_ser > 0.35 and p11_ser > 0.35


