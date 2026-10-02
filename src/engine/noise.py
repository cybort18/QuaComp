import math
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error


class DensityMatrixMemoryError(MemoryError):
    """Raised when an explicit density_matrix simulation would exceed memory capacity."""
    pass


def sample_pauli_error(p_error: float, rng: Optional[np.random.Generator] = None) -> Optional[str]:
    """
    Sample a stochastic single-qubit Pauli jump from a depolarizing channel with error rate p_error.
    
    With probability (1 - p_error): returns None (Identity / no error).
    With probability p_error: returns 'X', 'Y', or 'Z' uniformly (probability p_error / 3 each).
    
    Args:
        p_error (float): Depolarizing error probability in [0, 1].
        rng (Optional[np.random.Generator]): NumPy random generator.
        
    Returns:
        Optional[str]: 'X', 'Y', 'Z', or None.
    """
    if p_error <= 0.0:
        return None
    if rng is None:
        rng = np.random.default_rng()
    r = float(rng.random())
    if r >= p_error:
        return None
    sub_r = r / p_error
    if sub_r < 1.0 / 3.0:
        return 'X'
    elif sub_r < 2.0 / 3.0:
        return 'Y'
    else:
        return 'Z'


def sample_pauli_error_2q(p_error: float, rng: Optional[np.random.Generator] = None) -> Optional[Tuple[str, str]]:
    """
    Sample a stochastic two-qubit Pauli jump from a two-qubit depolarizing channel with error rate p_error.
    
    With probability (1 - p_error): returns None.
    With probability p_error: returns one of the 15 non-identity 2-qubit Pauli pairs uniformly.
    
    Args:
        p_error (float): Two-qubit depolarizing error probability in [0, 1].
        rng (Optional[np.random.Generator]): NumPy random generator.
        
    Returns:
        Optional[Tuple[str, str]]: (Pauli on q0, Pauli on q1), or None.
    """
    if p_error <= 0.0:
        return None
    if rng is None:
        rng = np.random.default_rng()
    r = float(rng.random())
    if r >= p_error:
        return None
    paulis = ['I', 'X', 'Y', 'Z']
    non_identity_pairs = [(p0, p1) for p0 in paulis for p1 in paulis if not (p0 == 'I' and p1 == 'I')]
    idx = int((r / p_error) * 15)
    idx = min(14, max(0, idx))
    return non_identity_pairs[idx]


def sample_thermal_relaxation(
    t_gate: float, 
    t1: float, 
    t2: float, 
    rng: Optional[np.random.Generator] = None
) -> Tuple[bool, bool]:
    """
    Sample stochastic thermal relaxation jumps (amplitude damping and pure dephasing).
    
    Args:
        t_gate (float): Gate duration in seconds.
        t1 (float): Longitudinal relaxation time T1 in seconds.
        t2 (float): Transverse coherence time T2 in seconds (enforces T2 <= 2*T1).
        rng (Optional[np.random.Generator]): NumPy random generator.
        
    Returns:
        Tuple[bool, bool]: (decay_jump, phase_jump)
            decay_jump (bool): True if an amplitude damping jump occurs (|1> -> |0>).
            phase_jump (bool): True if a pure dephasing jump occurs (phase flip Z).
    """
    if rng is None:
        rng = np.random.default_rng()
    if t_gate <= 0.0:
        return False, False
        
    # Amplitude damping probability: gamma = 1 - exp(-t / T1)
    if t1 <= 0.0:
        p_decay = 1.0
    else:
        p_decay = 1.0 - math.exp(-t_gate / t1)
    p_decay = min(1.0, max(0.0, p_decay))
    decay_jump = bool(rng.random() < p_decay)
    
    # Pure dephasing probability: 1/T_phi = max(0, 1/T2 - 1/(2*T1))
    effective_t2 = min(t2, 2.0 * t1)
    if effective_t2 >= 2.0 * t1 or effective_t2 <= 0.0:
        p_phase = 0.0
    else:
        rate_phi = (1.0 / effective_t2) - (1.0 / (2.0 * t1))
        t_phi = 1.0 / rate_phi if rate_phi > 0.0 else float("inf")
        p_phase = 0.5 * (1.0 - math.exp(-t_gate / t_phi))
    p_phase = min(0.5, max(0.0, p_phase))
    phase_jump = bool(rng.random() < p_phase)
    
    return decay_jump, phase_jump


def apply_gate_1q_to_statevector(
    state: np.ndarray, 
    matrix: np.ndarray, 
    target: int, 
    num_qubits: int
) -> np.ndarray:
    """
    Apply a 1-qubit unitary matrix in-place to an n-qubit statevector O(2^n).
    """
    axis = num_qubits - 1 - target
    s = state.reshape([2] * num_qubits)
    s_moved = np.moveaxis(s, axis, 0)
    v0 = s_moved[0].copy()
    v1 = s_moved[1].copy()
    s_moved[0] = matrix[0, 0] * v0 + matrix[0, 1] * v1
    s_moved[1] = matrix[1, 0] * v0 + matrix[1, 1] * v1
    return state


def apply_gate_2q_to_statevector(
    state: np.ndarray, 
    matrix: np.ndarray, 
    q0: int, 
    q1: int, 
    num_qubits: int
) -> np.ndarray:
    """
    Apply a 2-qubit unitary matrix in-place to an n-qubit statevector O(2^n).
    Convention: q0 is bit 0 of 2-qubit operator, q1 is bit 1.
    """
    axis0 = num_qubits - 1 - q0
    axis1 = num_qubits - 1 - q1
    s = state.reshape([2] * num_qubits)
    s_moved = np.moveaxis(s, [axis0, axis1], [0, 1])
    v00 = s_moved[0, 0].copy()
    v01 = s_moved[1, 0].copy()
    v10 = s_moved[0, 1].copy()
    v11 = s_moved[1, 1].copy()
    s_moved[0, 0] = matrix[0, 0] * v00 + matrix[0, 1] * v01 + matrix[0, 2] * v10 + matrix[0, 3] * v11
    s_moved[1, 0] = matrix[1, 0] * v00 + matrix[1, 1] * v01 + matrix[1, 2] * v10 + matrix[1, 3] * v11
    s_moved[0, 1] = matrix[2, 0] * v00 + matrix[2, 1] * v01 + matrix[2, 2] * v10 + matrix[2, 3] * v11
    s_moved[1, 1] = matrix[3, 0] * v00 + matrix[3, 1] * v01 + matrix[3, 2] * v10 + matrix[3, 3] * v11
    return state


def apply_pauli_to_statevector(
    state: np.ndarray, 
    pauli: str, 
    target: int, 
    num_qubits: int
) -> np.ndarray:
    """
    Apply a Pauli operator ('X', 'Y', 'Z') in-place to an n-qubit statevector O(2^n).
    """
    if pauli == 'I' or not pauli:
        return state
    axis = num_qubits - 1 - target
    s = state.reshape([2] * num_qubits)
    s_moved = np.moveaxis(s, axis, 0)
    
    if pauli == 'X':
        temp = s_moved[0].copy()
        s_moved[0] = s_moved[1]
        s_moved[1] = temp
    elif pauli == 'Y':
        temp = s_moved[0].copy()
        s_moved[0] = -1.0j * s_moved[1]
        s_moved[1] = 1.0j * temp
    elif pauli == 'Z':
        s_moved[1] *= -1.0
    return state


def apply_amplitude_damping_mcwf(
    state: np.ndarray, 
    target: int, 
    num_qubits: int, 
    gamma: float, 
    rng: np.random.Generator
) -> np.ndarray:
    """
    Apply stochastic amplitude damping jump (or conditioned no-jump evolution) in-place to statevector O(2^n).
    """
    if gamma <= 0.0:
        return state
    axis = num_qubits - 1 - target
    s = state.reshape([2] * num_qubits)
    s_moved = np.moveaxis(s, axis, 0)
    
    # Population of state |1> on target qubit
    p_one = float(np.sum(np.abs(s_moved[1]) ** 2))
    p_jump = min(1.0, max(0.0, gamma * p_one))
    
    r = float(rng.random())
    if r < p_jump and p_one > 1e-15:
        # Jump occurred: |1> -> |0>
        s_moved[0] = s_moved[1] / math.sqrt(p_one)
        s_moved[1] = 0.0
    else:
        # No jump occurred: continuous E0 evolution (non-unitary decay followed by renormalization)
        s_moved[1] *= math.sqrt(max(0.0, 1.0 - gamma))
        norm = float(np.linalg.norm(state))
        if norm > 0.0:
            state /= norm
    return state


class TrajectoryNoiseModel:
    """
    Monte Carlo Wavefunction (MCWF) / Quantum Trajectories Noise Model.
    
    Simulates physical decoherence (depolarizing, amplitude damping T1, pure dephasing T2,
    and readout bit-flip errors) using stochastic quantum jumps directly on statevectors.
    Preserves strict O(2^n) memory scaling and eliminates the O(4^n) density matrix memory wall.
    """
    def __init__(
        self,
        t1: float = 100e-6,
        t2: float = 120e-6,
        gate_error_1q: float = 0.001,
        gate_error_2q: float = 0.005,
        gate_time_1q: float = 50e-9,
        gate_time_2q: float = 300e-9,
        readout_error: float = 0.01,
        prob_meas0_prep1: Optional[float] = None,
        prob_meas1_prep0: Optional[float] = None,
        qubit_properties: Optional[Dict[int, Dict[str, Any]]] = None,
        coupling_errors: Optional[Dict[Tuple[int, int], float]] = None,
        name: str = "TrajectoryNoiseModel"
    ):
        self.t1 = float(t1)
        self.t2 = float(min(t2, 2.0 * t1))
        self.gate_error_1q = float(gate_error_1q)
        self.gate_error_2q = float(gate_error_2q)
        self.gate_time_1q = float(gate_time_1q)
        self.gate_time_2q = float(gate_time_2q)
        self.readout_error = float(readout_error)
        self.prob_meas0_prep1 = float(prob_meas0_prep1 if prob_meas0_prep1 is not None else readout_error)
        self.prob_meas1_prep0 = float(prob_meas1_prep0 if prob_meas1_prep0 is not None else readout_error)
        self.qubit_properties = qubit_properties or {}
        self.coupling_errors = coupling_errors or {}
        self.name = str(name)
        
    def get_qubit_properties(self, qubit_idx: int) -> Dict[str, Any]:
        """Fetch physical noise properties for a specific qubit."""
        if qubit_idx in self.qubit_properties:
            return self.qubit_properties[qubit_idx]
        return {
            "T1_s": self.t1,
            "T2_s": self.t2,
            "single_qubit_gate_error": self.gate_error_1q,
            "readout_error": self.readout_error,
            "prob_meas0_prep1": self.prob_meas0_prep1,
            "prob_meas1_prep0": self.prob_meas1_prep0
        }
        
    def get_two_qubit_gate_error(self, q0: int, q1: int) -> float:
        """Fetch two-qubit gate error rate for a specific pair of qubits."""
        if (q0, q1) in self.coupling_errors:
            return self.coupling_errors[(q0, q1)]
        if (q1, q0) in self.coupling_errors:
            return self.coupling_errors[(q1, q0)]
        return self.gate_error_2q

    def apply_stochastic_gate_noise(
        self,
        statevector: np.ndarray,
        target_qubits: List[int],
        gate_name: str,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Apply stochastic quantum jumps in-place to statevector O(2^n) following gate execution.
        
        Args:
            statevector (np.ndarray): Statevector array of size 2^n.
            target_qubits (List[int]): Indices of target qubits.
            gate_name (str): Standard gate name (e.g. 'h', 'cx', 'x').
            rng (Optional[np.random.Generator]): NumPy random number generator.
            
        Returns:
            np.ndarray: Updated and renormalized statevector.
        """
        if rng is None:
            rng = np.random.default_rng()
            
        num_qubits = int(round(math.log2(len(statevector))))
        
        if len(target_qubits) == 1:
            q = target_qubits[0]
            props = self.get_qubit_properties(q)
            t1 = float(props.get("T1_s", self.t1))
            t2 = float(props.get("T2_s", self.t2))
            p_1q = float(props.get("single_qubit_gate_error", self.gate_error_1q))
            
            # 1. Thermal Relaxation: Amplitude damping jump
            gamma = 1.0 - math.exp(-self.gate_time_1q / t1) if t1 > 0 else 1.0
            apply_amplitude_damping_mcwf(statevector, q, num_qubits, gamma, rng)
            
            # 2. Pure Dephasing Jump
            effective_t2 = min(t2, 2.0 * t1)
            if effective_t2 < 2.0 * t1 and effective_t2 > 0:
                rate_phi = (1.0 / effective_t2) - (1.0 / (2.0 * t1))
                t_phi = 1.0 / rate_phi if rate_phi > 0 else float("inf")
                p_phase = 0.5 * (1.0 - math.exp(-self.gate_time_1q / t_phi))
                if rng.random() < p_phase:
                    apply_pauli_to_statevector(statevector, 'Z', q, num_qubits)
                    
            # 3. Single-Qubit Depolarizing Jump
            pauli_jump = sample_pauli_error(p_1q, rng)
            if pauli_jump:
                apply_pauli_to_statevector(statevector, pauli_jump, q, num_qubits)
                
        elif len(target_qubits) == 2:
            q0, q1 = target_qubits[0], target_qubits[1]
            p_2q = self.get_two_qubit_gate_error(q0, q1)
            
            # 1. Thermal relaxation on both qubits
            for q in (q0, q1):
                props = self.get_qubit_properties(q)
                t1 = float(props.get("T1_s", self.t1))
                t2 = float(props.get("T2_s", self.t2))
                gamma = 1.0 - math.exp(-self.gate_time_2q / t1) if t1 > 0 else 1.0
                apply_amplitude_damping_mcwf(statevector, q, num_qubits, gamma, rng)
                
                effective_t2 = min(t2, 2.0 * t1)
                if effective_t2 < 2.0 * t1 and effective_t2 > 0:
                    rate_phi = (1.0 / effective_t2) - (1.0 / (2.0 * t1))
                    t_phi = 1.0 / rate_phi if rate_phi > 0 else float("inf")
                    p_phase = 0.5 * (1.0 - math.exp(-self.gate_time_2q / t_phi))
                    if rng.random() < p_phase:
                        apply_pauli_to_statevector(statevector, 'Z', q, num_qubits)
                        
            # 2. Two-Qubit Depolarizing Jump
            p_pair = sample_pauli_error_2q(p_2q, rng)
            if p_pair:
                if p_pair[0] != 'I':
                    apply_pauli_to_statevector(statevector, p_pair[0], q0, num_qubits)
                if p_pair[1] != 'I':
                    apply_pauli_to_statevector(statevector, p_pair[1], q1, num_qubits)
                    
        return statevector

    def apply_readout_error(
        self,
        bitstring: str,
        rng: Optional[np.random.Generator] = None
    ) -> str:
        """
        Apply stochastic classical readout bit-flip errors according to P(0|1) and P(1|0).
        
        Args:
            bitstring (str): Ideal measurement bitstring (e.g. '0101', where bit string index 0 is MSB).
            rng (Optional[np.random.Generator]): NumPy random generator.
            
        Returns:
            str: Noisy bitstring.
        """
        if rng is None:
            rng = np.random.default_rng()
            
        num_qubits = len(bitstring)
        noisy_chars = list(bitstring)
        
        for bit_idx, char in enumerate(bitstring):
            # In Qiskit bitstring convention, bitstring[0] is qubit (num_qubits - 1), bitstring[-1] is qubit 0
            q_idx = num_qubits - 1 - bit_idx
            props = self.get_qubit_properties(q_idx)
            p01 = float(props.get("prob_meas0_prep1", self.prob_meas0_prep1))
            p10 = float(props.get("prob_meas1_prep0", self.prob_meas1_prep0))
            
            if char == '0':
                if rng.random() < p10:
                    noisy_chars[bit_idx] = '1'
            elif char == '1':
                if rng.random() < p01:
                    noisy_chars[bit_idx] = '0'
                    
        return "".join(noisy_chars)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration summary."""
        return {
            "name": self.name,
            "method": "Monte Carlo Wavefunction (Trajectory)",
            "T1_s": self.t1,
            "T2_s": self.t2,
            "gate_error_1q": self.gate_error_1q,
            "gate_error_2q": self.gate_error_2q,
            "readout_error": self.readout_error,
            "prob_meas0_prep1": self.prob_meas0_prep1,
            "prob_meas1_prep0": self.prob_meas1_prep0,
            "gate_time_1q_s": self.gate_time_1q,
            "gate_time_2q_s": self.gate_time_2q
        }


def get_trajectory_noise_model(level: str = 'none') -> Optional[TrajectoryNoiseModel]:
    """
    Generate a TrajectoryNoiseModel preset level ('none', 'low', 'medium', 'high').
    
    Presets:
        - 'none': None (ideal simulation).
        - 'low': T1=100µs, T2=120µs, gate error rate 0.001, readout error 0.005.
        - 'medium': T1=50µs, T2=70µs, gate error rate 0.005, readout error 0.015.
        - 'high': T1=20µs, T2=30µs, gate error rate 0.02, readout error 0.03.
    """
    if not isinstance(level, str):
        raise TypeError("Noise level must be a string.")
        
    norm_level = level.strip().lower()
    if norm_level == 'none':
        return None
        
    presets = {
        'low': {'t1': 100e-6, 't2': 120e-6, 'gate_error': 0.001, 'readout_error': 0.005},
        'medium': {'t1': 50e-6, 't2': 70e-6, 'gate_error': 0.005, 'readout_error': 0.015},
        'high': {'t1': 20e-6, 't2': 30e-6, 'gate_error': 0.02, 'readout_error': 0.03},
    }
    
    if norm_level not in presets:
        raise ValueError(f"Unknown noise level '{level}'. Valid choices are: 'none', 'low', 'medium', 'high'.")
        
    params = presets[norm_level]
    return TrajectoryNoiseModel(
        t1=params['t1'],
        t2=params['t2'],
        gate_error_1q=params['gate_error'],
        gate_error_2q=params['gate_error'] * 2,
        readout_error=params['readout_error'],
        name=f"TrajectoryPreset:{norm_level.capitalize()}"
    )


def get_noise_model(level: str = 'none') -> Optional[NoiseModel]:
    """
    Generate a Qiskit Aer NoiseModel based on the specified noise preset level.
    
    Presets:
        - 'none': Returns None (ideal simulation).
        - 'low': T1=100µs, T2=120µs, gate error rate 0.001 (0.1%).
        - 'medium': T1=50µs, T2=70µs, gate error rate 0.005 (0.5%).
        - 'high': T1=20µs, T2=30µs, gate error rate 0.02 (2.0%).
        
    Args:
        level (str): Noise level ('none', 'low', 'medium', 'high').
        
    Returns:
        Optional[NoiseModel]: Built Qiskit Aer NoiseModel or None if level is 'none'.
        
    Raises:
        TypeError: If level is not a string.
        ValueError: If level is not one of the valid options.
    """
    if not isinstance(level, str):
        raise TypeError("Noise level must be a string.")
        
    norm_level = level.strip().lower()
    
    if norm_level == 'none':
        return None
        
    noise_presets = {
        'low': {'t1': 100e-6, 't2': 120e-6, 'gate_error': 0.001},
        'medium': {'t1': 50e-6, 't2': 70e-6, 'gate_error': 0.005},
        'high': {'t1': 20e-6, 't2': 30e-6, 'gate_error': 0.02},
    }
    
    if norm_level not in noise_presets:
        raise ValueError(f"Unknown noise level '{level}'. Valid choices are: 'none', 'low', 'medium', 'high'.")
        
    params = noise_presets[norm_level]
    t1 = params['t1']
    t2 = params['t2']
    gate_error = params['gate_error']
    
    # Typical gate times (seconds)
    time_single_qubit = 50e-9   # 50 ns
    time_two_qubit = 300e-9     # 300 ns
    
    noise_model = NoiseModel()
    
    # Single qubit errors
    error_thermal_1q = thermal_relaxation_error(t1, t2, time_single_qubit)
    error_depol_1q = depolarizing_error(gate_error, 1)
    error_1q = error_thermal_1q.compose(error_depol_1q)
    
    # Two qubit errors
    error_thermal_2q = thermal_relaxation_error(t1, t2, time_two_qubit).tensor(
        thermal_relaxation_error(t1, t2, time_two_qubit)
    )
    error_depol_2q = depolarizing_error(gate_error * 2, 2)
    error_2q = error_thermal_2q.compose(error_depol_2q)
    
    single_qubit_gates = ['h', 'rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'x', 'y', 'z']
    two_qubit_gates = ['cx', 'cp', 'cz', 'swap']
    
    noise_model.add_all_qubit_quantum_error(error_1q, single_qubit_gates)
    noise_model.add_all_qubit_quantum_error(error_2q, two_qubit_gates)
    
    return noise_model

def calculate_state_fidelity(ideal_counts: Dict[str, int], noisy_counts: Dict[str, int]) -> float:
    """
    Calculate the classical Hellinger state fidelity percentage between ideal and noisy measurement counts.
    
    Formula:
        Fidelity (%) = ( sum_x sqrt( P_ideal(x) * P_noisy(x) ) )^2 * 100.0
        
    Args:
        ideal_counts (dict): Measurement counts dictionary from ideal simulation.
        noisy_counts (dict): Measurement counts dictionary from noisy simulation.
        
    Returns:
        float: State fidelity percentage between 0.0% and 100.0%.
        
    Raises:
        TypeError: If inputs are not dictionaries.
    """
    if not isinstance(ideal_counts, dict) or not isinstance(noisy_counts, dict):
        raise TypeError("Counts inputs must be dictionaries.")
        
    if not ideal_counts or not noisy_counts:
        return 100.0
        
    total_ideal = sum(ideal_counts.values())
    total_noisy = sum(noisy_counts.values())
    
    if total_ideal == 0 or total_noisy == 0:
        return 0.0
        
    all_keys = set(ideal_counts.keys()).union(set(noisy_counts.keys()))
    
    fidelity_sqrt = 0.0
    for key in all_keys:
        p_ideal = ideal_counts.get(key, 0) / total_ideal
        p_noisy = noisy_counts.get(key, 0) / total_noisy
        fidelity_sqrt += np.sqrt(p_ideal * p_noisy)
        
    fidelity_pct = (fidelity_sqrt ** 2) * 100.0
    return float(np.clip(fidelity_pct, 0.0, 100.0))

def calculate_overhead_ratio(ideal_latency: float, noisy_latency: float) -> float:
    """
    Calculate the percentage increase in computational latency due to noise model calculations.
    
    Formula:
        Overhead Ratio (%) = max(0.0, ((noisy_latency - ideal_latency) / ideal_latency) * 100.0)
        
    Args:
        ideal_latency (float): Execution time for ideal simulation in seconds.
        noisy_latency (float): Execution time for noisy simulation in seconds.
        
    Returns:
        float: Computation overhead percentage ratio.
        
    Raises:
        TypeError: If inputs are not numbers.
        ValueError: If latencies are negative.
    """
    if not isinstance(ideal_latency, (int, float)) or not isinstance(noisy_latency, (int, float)):
        raise TypeError("Latencies must be numbers.")
        
    if ideal_latency < 0 or noisy_latency < 0:
        raise ValueError("Latencies must be non-negative.")
        
    if ideal_latency <= 0:
        return 0.0
        
    overhead = ((noisy_latency - ideal_latency) / ideal_latency) * 100.0
    return float(max(0.0, overhead))
