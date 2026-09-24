import os
import json
import math
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

# Pauli matrices
I2 = np.eye(2, dtype=np.complex128)
X_MAT = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y_MAT = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z_MAT = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
PAULIS_1Q = [I2, X_MAT, Y_MAT, Z_MAT]


def verify_kraus_completeness(kraus_ops: List[np.ndarray], atol: float = 1e-10) -> bool:
    """
    Verify that a set of Kraus operators satisfies the trace-preserving completeness condition:
    sum_k E_k^dagger * E_k = I.
    
    Args:
        kraus_ops (List[np.ndarray]): List of Kraus operator matrices.
        atol (float): Absolute numerical tolerance.
        
    Returns:
        bool: True if complete, False otherwise.
    """
    if not kraus_ops:
        return False
    dim = kraus_ops[0].shape[0]
    total = np.zeros((dim, dim), dtype=np.complex128)
    for E in kraus_ops:
        total += np.matmul(E.conj().T, E)
    expected = np.eye(dim, dtype=np.complex128)
    return bool(np.allclose(total, expected, atol=atol))


def generate_kraus_amplitude_damping(t1_us: float, gate_time_ns: float) -> List[np.ndarray]:
    """
    Generate Kraus operators for amplitude damping (thermal relaxation to ground state |0>):
    gamma = 1 - exp(-t / T1).
    
    E0 = [[1, 0], [0, sqrt(1 - gamma)]]
    E1 = [[0, sqrt(gamma)], [0, 0]]
    """
    if t1_us <= 0.0:
        gamma = 1.0
    else:
        dt_us = max(0.0, gate_time_ns * 1e-3)
        gamma = 1.0 - math.exp(-dt_us / t1_us)
    gamma = min(1.0, max(0.0, gamma))
    
    E0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]], dtype=np.complex128)
    E1 = np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128)
    return [E0, E1]


def generate_kraus_phase_damping(t_phi_us: float, gate_time_ns: float) -> List[np.ndarray]:
    """
    Generate Kraus operators for pure dephasing channel (loss of phase coherence):
    lambda = 1 - exp(-t / T_phi).
    
    E0 = [[1, 0], [0, sqrt(1 - lambda)]]
    E1 = [[0, 0], [0, sqrt(lambda)]]
    """
    if t_phi_us <= 0.0:
        lam = 1.0
    elif math.isinf(t_phi_us):
        lam = 0.0
    else:
        dt_us = max(0.0, gate_time_ns * 1e-3)
        lam = 1.0 - math.exp(-dt_us / t_phi_us)
    lam = min(1.0, max(0.0, lam))
    
    E0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - lam)]], dtype=np.complex128)
    E1 = np.array([[0.0, 0.0], [0.0, math.sqrt(lam)]], dtype=np.complex128)
    return [E0, E1]


def generate_thermal_relaxation_kraus(t1_us: float, t2_us: float, gate_time_ns: float) -> List[np.ndarray]:
    """
    Generate combined thermal relaxation Kraus operators combining longitudinal (T1)
    and transverse (T2) relaxation.
    
    Enforces the fundamental quantum physical bound: T2 <= 2 * T1.
    Pure dephasing rate: 1/T_phi = max(0.0, 1/T2 - 1/(2*T1)).
    """
    # Enforce physical bound T2 <= 2 * T1
    effective_t2 = min(t2_us, 2.0 * t1_us)
    
    # Calculate pure dephasing time T_phi
    if effective_t2 >= 2.0 * t1_us:
        t_phi_us = float("inf")
    else:
        rate_phi = (1.0 / effective_t2) - (1.0 / (2.0 * t1_us))
        t_phi_us = 1.0 / rate_phi if rate_phi > 0.0 else float("inf")
        
    E_amp = generate_kraus_amplitude_damping(t1_us, gate_time_ns)
    E_phase = generate_kraus_phase_damping(t_phi_us, gate_time_ns)
    
    # Combined channel via composition E_ij = E_phase_i * E_amp_j
    combined_kraus = []
    for Ep in E_phase:
        for Ea in E_amp:
            combined_kraus.append(np.matmul(Ep, Ea))
            
    return combined_kraus


def generate_kraus_depolarizing(p: float) -> List[np.ndarray]:
    """
    Generate Kraus operators for single-qubit depolarizing error with parameter p:
    E0 = sqrt(1 - 3p/4) * I
    E1 = sqrt(p/4) * X
    E2 = sqrt(p/4) * Y
    E3 = sqrt(p/4) * Z
    """
    p = min(1.0, max(0.0, p))
    w0 = math.sqrt(max(0.0, 1.0 - 0.75 * p))
    wp = math.sqrt(max(0.0, 0.25 * p))
    
    return [
        w0 * I2,
        wp * X_MAT,
        wp * Y_MAT,
        wp * Z_MAT
    ]


def generate_kraus_2q_depolarizing(p: float) -> List[np.ndarray]:
    """
    Generate Kraus operators for two-qubit depolarizing error with parameter p (16 Kraus ops).
    """
    p = min(1.0, max(0.0, p))
    w0 = math.sqrt(max(0.0, 1.0 - (15.0 / 16.0) * p))
    wp = math.sqrt(max(0.0, p / 16.0))
    
    kraus = []
    for a in range(4):
        for b in range(4):
            mat = np.kron(PAULIS_1Q[a], PAULIS_1Q[b])
            if a == 0 and b == 0:
                kraus.append(w0 * mat)
            else:
                kraus.append(wp * mat)
    return kraus


class PhysicalNoiseModel:
    """
    Parser and mathematical Kraus operator generator for real physical QPU calibration data
    (IBM Quantum, Rigetti, or generic superconducting architectures).
    """
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.backend_name = str(data.get("backend_name", "generic_qpu"))
        self.backend_version = str(data.get("backend_version", "1.0.0"))
        self.provider = str(data.get("provider", "Generic"))
        self.architecture = str(data.get("architecture", "Superconducting"))
        self.num_qubits = int(data.get("num_qubits", len(data.get("qubits", {}))))
        
        gen_params = data.get("general_parameters", {})
        self.gate_time_1q_ns = float(gen_params.get("gate_time_1q_ns", 35.0))
        self.gate_time_2q_ns = float(gen_params.get("gate_time_2q_ns", 300.0))
        
        self.qubits_data = data.get("qubits", {})
        self.couplings_data = data.get("couplings", [])
        
    @classmethod
    def from_json(cls, source: Union[str, Dict[str, Any]]) -> "PhysicalNoiseModel":
        """
        Load calibration profile from a JSON file path, preset name, or dictionary.
        """
        if isinstance(source, dict):
            return cls(source)
            
        if not isinstance(source, str):
            raise TypeError("Source must be a file path string or dictionary.")
            
        candidate_paths = [
            source,
            os.path.join("results", "noise_profiles", source),
            os.path.join("results", "noise_profiles", f"{source}.json"),
            os.path.abspath(source),
        ]
        
        target_path = None
        for p in candidate_paths:
            if os.path.exists(p) and os.path.isfile(p):
                target_path = p
                break
                
        if not target_path:
            raise FileNotFoundError(f"Physical QPU noise calibration file not found: '{source}'")
            
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        return cls(data)

    def get_qubit_properties(self, qubit_idx: int) -> Dict[str, Any]:
        """
        Extract physical noise parameters for an individual qubit.
        """
        str_idx = str(qubit_idx)
        if str_idx in self.qubits_data:
            q = self.qubits_data[str_idx]
            t1 = float(q.get("T1_us", 200.0))
            t2 = float(q.get("T2_us", 150.0))
            # Physical consistency check: T2 <= 2*T1
            t2 = min(t2, 2.0 * t1)
            return {
                "qubit": qubit_idx,
                "T1_us": t1,
                "T2_us": t2,
                "frequency_GHz": float(q.get("frequency_GHz", 5.0)),
                "readout_error": float(q.get("readout_error", 0.015)),
                "prob_meas0_prep1": float(q.get("prob_meas0_prep1", 0.018)),
                "prob_meas1_prep0": float(q.get("prob_meas1_prep0", 0.008)),
                "single_qubit_gate_error": float(q.get("single_qubit_gate_error", 0.0002))
            }
        else:
            # Fallback to balanced representative default
            return {
                "qubit": qubit_idx,
                "T1_us": 220.0,
                "T2_us": 150.0,
                "frequency_GHz": 5.0,
                "readout_error": 0.015,
                "prob_meas0_prep1": 0.018,
                "prob_meas1_prep0": 0.008,
                "single_qubit_gate_error": 0.0002
            }

    def get_coupling_error(self, q1: int, q2: int) -> float:
        """
        Get two-qubit gate error rate for a specific coupled pair.
        """
        for link in self.couplings_data:
            pair = link.get("qubits", [])
            if (pair == [q1, q2]) or (pair == [q2, q1]):
                return float(link.get("two_qubit_gate_error", 0.008))
        return 0.008

    def get_readout_confusion_matrix(self, qubit_idx: int) -> np.ndarray:
        """
        Compute the classical 2x2 readout confusion matrix:
        M = [[P(meas 0 | prep 0), P(meas 0 | prep 1)],
             [P(meas 1 | prep 0), P(meas 1 | prep 1)]]
        """
        props = self.get_qubit_properties(qubit_idx)
        p10 = props["prob_meas1_prep0"]
        p01 = props["prob_meas0_prep1"]
        p00 = 1.0 - p10
        p11 = 1.0 - p01
        return np.array([[p00, p01], [p10, p11]], dtype=np.float64)

    def get_qubit_kraus_operators(self, qubit_idx: int) -> Dict[str, List[np.ndarray]]:
        """
        Generate complete analytical Kraus operator sets for an individual qubit:
        - 'thermal_relaxation'
        - 'single_qubit_depolarizing'
        """
        props = self.get_qubit_properties(qubit_idx)
        thermal_kraus = generate_thermal_relaxation_kraus(
            props["T1_us"], props["T2_us"], self.gate_time_1q_ns
        )
        depol_kraus = generate_kraus_depolarizing(props["single_qubit_gate_error"])
        
        return {
            "thermal_relaxation": thermal_kraus,
            "single_qubit_depolarizing": depol_kraus
        }

    def to_qiskit_noise_model(self, active_qubits: Optional[List[int]] = None) -> Any:
        """
        Translate physical calibration parameters into an active Qiskit Aer NoiseModel.
        
        Args:
            active_qubits (Optional[List[int]]): Subset of qubits to model, or None for all.
            
        Returns:
            qiskit_aer.noise.NoiseModel: Ready-to-simulate noise model.
        """
        from qiskit_aer.noise import NoiseModel, ReadoutError, thermal_relaxation_error, depolarizing_error
        
        noise_model = NoiseModel()
        qubit_indices = active_qubits if active_qubits is not None else list(range(max(8, self.num_qubits)))
        
        # 1. Add Single-Qubit Errors & Readout Errors per Qubit
        for q_idx in qubit_indices:
            props = self.get_qubit_properties(q_idx)
            
            # Readout error
            p10 = props["prob_meas1_prep0"]
            p01 = props["prob_meas0_prep1"]
            probabilities = [[1.0 - p10, p10], [p01, 1.0 - p01]]
            ro_error = ReadoutError(probabilities)
            noise_model.add_readout_error(ro_error, [q_idx])
            
            # 1-Qubit Thermal Relaxation error
            t1_ns = props["T1_us"] * 1e3
            t2_ns = props["T2_us"] * 1e3
            t1_error = thermal_relaxation_error(t1_ns, t2_ns, self.gate_time_1q_ns)
            
            # 1-Qubit Depolarizing error
            depol_error = depolarizing_error(props["single_qubit_gate_error"], 1)
            composed_1q = t1_error.compose(depol_error)
            
            # Add to common 1-qubit basis gates
            for gname in ("sx", "x", "rz", "h", "u", "s"):
                noise_model.add_quantum_error(composed_1q, gname, [q_idx])
                
        # 2. Add Two-Qubit Errors for Coupled Pairs
        for link in self.couplings_data:
            q_pair = link.get("qubits", [])
            if len(q_pair) == 2:
                q0, q1 = q_pair
                if (active_qubits is None) or (q0 in active_qubits and q1 in active_qubits):
                    p2 = float(link.get("two_qubit_gate_error", 0.008))
                    err2 = depolarizing_error(p2, 2)
                    for gname in ("ecr", "cx", "cz"):
                        noise_model.add_quantum_error(err2, gname, [q0, q1])
                        noise_model.add_quantum_error(err2, gname, [q1, q0])
                        
        return noise_model
