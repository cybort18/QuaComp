import time
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
from qiskit import QuantumCircuit, transpile


class MatrixProductState:
    """
    Matrix Product State (MPS) representation with open boundary conditions,
    supporting nearest-neighbor contraction, SWAP network routing for non-adjacent
    2-qubit gates, and adaptive SVD singular value truncation error tracking.
    """

    def __init__(self, num_qubits: int, max_bond_dim: int = 64, cutoff: float = 1e-12) -> None:
        """
        Initialize an MPS for an n-qubit system in state |0...0>.

        Args:
            num_qubits (int): Total number of qubits.
            max_bond_dim (int): Maximum allowable bond dimension (chi_max).
            cutoff (float): Singular value truncation threshold.

        Raises:
            TypeError: If types of arguments are invalid.
            ValueError: If parameters are outside acceptable operational bounds.
        """
        if not isinstance(num_qubits, int):
            raise TypeError("num_qubits must be an integer.")
        if num_qubits < 1:
            raise ValueError("num_qubits must be at least 1.")
        if not isinstance(max_bond_dim, int):
            raise TypeError("max_bond_dim must be an integer.")
        if max_bond_dim < 1:
            raise ValueError("max_bond_dim must be at least 1.")
        if not isinstance(cutoff, (int, float)):
            raise TypeError("cutoff must be a real number.")
        if cutoff < 0.0:
            raise ValueError("cutoff must be non-negative.")

        self.num_qubits: int = num_qubits
        self.max_bond_dim: int = max_bond_dim
        self.cutoff: float = float(cutoff)

        # Tensor chain: each tensor has shape (d_left, 2, d_right)
        # Initialized to product state |0...0>
        self.tensors: List[np.ndarray] = []
        for _ in range(num_qubits):
            site_t = np.zeros((1, 2, 1), dtype=np.complex128)
            site_t[0, 0, 0] = 1.0 + 0.0j
            self.tensors.append(site_t)

        self.mps_fidelity: float = 1.0
        self.cumulative_truncation_error: float = 0.0
        self.max_active_bond_dim: int = 1
        self.swap_count: int = 0

    def apply_1q(self, qubit: int, u_matrix: np.ndarray) -> None:
        """
        Apply a single-qubit unitary gate to a specified site.

        Args:
            qubit (int): Global index of target qubit (0 to n - 1).
            u_matrix (np.ndarray): 2x2 complex unitary matrix.

        Raises:
            IndexError: If qubit index is out of bounds.
            ValueError: If u_matrix is not of shape (2, 2).
        """
        if not (0 <= qubit < self.num_qubits):
            raise IndexError(f"Qubit index {qubit} is out of bounds for {self.num_qubits}-qubit MPS.")
        if u_matrix.shape != (2, 2):
            raise ValueError(f"u_matrix must have shape (2, 2), got {u_matrix.shape}.")

        # Contraction: lir = sum_s (u_is * lsr)
        self.tensors[qubit] = np.einsum('is,lsr->lir', u_matrix, self.tensors[qubit])

    def apply_2q_adjacent(
        self,
        site_l: int,
        site_r: int,
        u_4x4: np.ndarray,
        reverse: bool = False
    ) -> None:
        """
        Apply a 2-qubit unitary gate to two adjacent sites (site_l, site_l + 1)
        using SVD truncation and fidelity tracking.

        Args:
            site_l (int): Left site index.
            site_r (int): Right site index (must equal site_l + 1).
            u_4x4 (np.ndarray): 4x4 unitary matrix in Qiskit ordering |q1 q0>.
            reverse (bool): If True, target qubit ordering is (q1, q0) rather than (q0, q1).

        Raises:
            ValueError: If sites are not strictly adjacent or matrix is not 4x4.
        """
        if site_r != site_l + 1:
            raise ValueError(f"Sites must be adjacent: site_l={site_l}, site_r={site_r}.")
        if u_4x4.shape != (4, 4):
            raise ValueError(f"u_4x4 must have shape (4, 4), got {u_4x4.shape}.")

        # Qiskit basis ordering: |q1 q0>
        # Reshaped to (q1_out, q0_out, q1_in, q0_in)
        u_reshaped = u_4x4.reshape(2, 2, 2, 2)

        tensor_l = self.tensors[site_l]  # shape (l, 2, m)
        tensor_r = self.tensors[site_r]  # shape (m, 2, r)

        # Two-site contraction over bond m: lcdr = sum_m (lcm * mdr)
        theta = np.einsum('lcm,mdr->lcdr', tensor_l, tensor_r)

        if not reverse:
            # q0 is at site_l (index c), q1 is at site_r (index d)
            # new q0 is at site_l (index a), new q1 is at site_r (index b)
            theta_prime = np.einsum('badc,lcdr->labr', u_reshaped, theta)
        else:
            # q1 is at site_l (index c), q0 is at site_r (index d)
            # new q1 is at site_l (index b), new q0 is at site_r (index a)
            theta_prime = np.einsum('badc,ldcr->lbar', u_reshaped, theta)

        l_dim = theta_prime.shape[0]
        r_dim = theta_prime.shape[3]
        m_matrix = theta_prime.reshape(l_dim * 2, 2 * r_dim)

        u_svd, s_svd, vh_svd = np.linalg.svd(m_matrix, full_matrices=False)

        # Spectral truncation and fidelity calculation
        s_sq = s_svd ** 2
        s_sum = float(np.sum(s_sq))
        if s_sum > 0.0:
            sigma_sq = s_sq / s_sum
        else:
            sigma_sq = s_sq

        # Determine retained bond dimension
        cutoff_threshold = self.cutoff ** 2
        keep_mask = sigma_sq >= cutoff_threshold
        num_above_cutoff = int(np.sum(keep_mask))
        chi = min(self.max_bond_dim, max(1, num_above_cutoff))

        # Discarded spectral weight for this step
        if len(sigma_sq) > chi:
            eps_step = float(np.sum(sigma_sq[chi:]))
        else:
            eps_step = 0.0

        eps_step = max(0.0, min(1.0, eps_step))

        # Update cumulative fidelity and truncation error
        self.mps_fidelity = float(self.mps_fidelity * (1.0 - eps_step))
        self.mps_fidelity = max(0.0, min(1.0, self.mps_fidelity))
        self.cumulative_truncation_error = float(1.0 - self.mps_fidelity)
        self.max_active_bond_dim = max(self.max_active_bond_dim, chi)

        # Truncate singular components
        u_trunc = u_svd[:, :chi]
        s_trunc = s_svd[:chi]
        vh_trunc = vh_svd[:chi, :]

        # Normalize truncated state
        s_trunc_norm = np.linalg.norm(s_trunc)
        if s_trunc_norm > 0.0:
            s_trunc = s_trunc / s_trunc_norm

        # Reassign tensors to MPS chain
        self.tensors[site_l] = u_trunc.reshape(l_dim, 2, chi)
        self.tensors[site_r] = (np.diag(s_trunc) @ vh_trunc).reshape(chi, 2, r_dim)

    def apply_2q(self, q0: int, q1: int, u_4x4: np.ndarray) -> None:
        """
        Apply a 2-qubit unitary gate across arbitrary pairs of qubits.
        If |q0 - q1| > 1, applies a nearest-neighbor SWAP network to route
        q0 adjacent to q1, executes the gate, and applies the inverse SWAP
        chain to restore original topological indexing.

        Args:
            q0 (int): Control / first qubit index.
            q1 (int): Target / second qubit index.
            u_4x4 (np.ndarray): 4x4 unitary matrix.

        Raises:
            IndexError: If qubit indices are out of bounds.
            ValueError: If q0 == q1.
        """
        if not (0 <= q0 < self.num_qubits) or not (0 <= q1 < self.num_qubits):
            raise IndexError(f"Qubit indices ({q0}, {q1}) out of bounds for {self.num_qubits}-qubit MPS.")
        if q0 == q1:
            raise ValueError(f"Cannot apply 2-qubit gate on identical qubit index {q0}.")

        swap_gate = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.complex128)

        diff = abs(q0 - q1)
        if diff == 1:
            if q0 < q1:
                self.apply_2q_adjacent(q0, q1, u_4x4, reverse=False)
            else:
                self.apply_2q_adjacent(q1, q0, u_4x4, reverse=True)
            return

        # Non-adjacent SWAP routing
        if q0 < q1:
            # Forward routing: bring q0 to site q1 - 1
            for s in range(q0, q1 - 1):
                self.apply_2q_adjacent(s, s + 1, swap_gate, reverse=False)
                self.swap_count += 1

            # Contract target gate at adjacent sites
            self.apply_2q_adjacent(q1 - 1, q1, u_4x4, reverse=False)

            # Inverse routing: return q0 to original site
            for s in range(q1 - 2, q0 - 1, -1):
                self.apply_2q_adjacent(s, s + 1, swap_gate, reverse=False)
                self.swap_count += 1
        else:
            # Forward routing: bring q0 down to site q1 + 1
            for s in range(q0 - 1, q1, -1):
                self.apply_2q_adjacent(s, s + 1, swap_gate, reverse=False)
                self.swap_count += 1

            # Contract target gate at adjacent sites
            self.apply_2q_adjacent(q1, q1 + 1, u_4x4, reverse=True)

            # Inverse routing: return q0 to original site
            for s in range(q1 + 1, q0):
                self.apply_2q_adjacent(s, s + 1, swap_gate, reverse=False)
                self.swap_count += 1

    def to_statevector(self) -> np.ndarray:
        """
        Contract all MPS tensors into an exact full statevector.

        Returns:
            np.ndarray: Complex128 1D statevector conforming to standard Qiskit basis ordering.

        Raises:
            MemoryError: If num_qubits exceeds safe statevector limits (n > 24).
        """
        if self.num_qubits > 24:
            raise MemoryError(
                f"Direct statevector contraction for {self.num_qubits} qubits requires "
                f"excessive memory. Use tensor network contraction or sampling methods."
            )

        curr = self.tensors[0]  # shape (1, 2, r1)
        for k in range(1, self.num_qubits):
            next_t = self.tensors[k]  # shape (rk, 2, rk+1)
            curr = np.tensordot(curr, next_t, axes=(-1, 0))

        # Resulting shape is (1, 2, 2, ..., 2, 1)
        curr = np.squeeze(curr, axis=(0, -1))
        if self.num_qubits == 1:
            return curr.flatten()

        # Transpose from physical site ordering (q0, q1, ..., qn-1) to
        # Qiskit little-endian basis ordering (qn-1, ..., q1, q0)
        curr_transposed = np.transpose(curr, tuple(range(self.num_qubits - 1, -1, -1)))
        return curr_transposed.flatten()

    def get_probabilities(self) -> np.ndarray:
        """
        Compute full probability distribution vector from MPS statevector.

        Returns:
            np.ndarray: Normalized probability array of length 2^n.
        """
        sv = self.to_statevector()
        probs = np.abs(sv) ** 2
        p_sum = float(np.sum(probs))
        if p_sum > 0.0:
            probs = probs / p_sum
        return probs

    def sample_counts(self, shots: int = 1024, seed: Optional[int] = None) -> Dict[str, int]:
        """
        Sample projective measurement outcomes from the MPS quantum state.

        Args:
            shots (int): Number of measurement shots.
            seed (Optional[int]): Random seed for sampling.

        Returns:
            Dict[str, int]: Measurement bitstrings and observed counts.
        """
        probs = self.get_probabilities()
        rng = np.random.default_rng(seed)
        samples = rng.choice(len(probs), size=shots, p=probs)
        counts: Dict[str, int] = {}
        for s in samples:
            bs = f"{int(s):0{self.num_qubits}b}"
            counts[bs] = counts.get(bs, 0) + 1
        return counts


def simulate_mps(
    circuit: Any,
    bond_dimension: int = 64,
    cutoff: float = 1e-12,
    shots: int = 1024,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Execute quantum circuit simulation using the native Matrix Product State (MPS) engine,
    supporting SWAP network routing and SVD truncation error tracking.

    Args:
        circuit: Qiskit QuantumCircuit, ParsedQASMCircuit, or path string to OpenQASM file.
        bond_dimension (int): Maximum bond dimension for SVD truncation.
        cutoff (float): SVD truncation cutoff threshold.
        shots (int): Number of projective measurement shots.
        seed (Optional[int]): Random seed for sampling.

    Returns:
        Dict[str, Any]: Simulation results containing counts, fidelity, bond dimension, and latency.
    """
    # Resolve circuit input
    qc: Optional[QuantumCircuit] = None
    if isinstance(circuit, str):
        from src.engine.parser import load_qasm
        parsed = load_qasm(circuit)
        qc = parsed.to_quantum_circuit()
    elif hasattr(circuit, "to_quantum_circuit"):
        qc = circuit.to_quantum_circuit()
    elif isinstance(circuit, QuantumCircuit):
        qc = circuit
    else:
        raise TypeError("circuit must be a QuantumCircuit, ParsedQASMCircuit, or QASM file path.")

    num_qubits = qc.num_qubits
    if num_qubits < 1:
        raise ValueError("Circuit must have at least 1 qubit.")

    # Remove measurement gates for pure MPS propagation
    circ_clean = qc.copy()
    if len(circ_clean.clbits) > 0:
        circ_clean.remove_final_measurements(inplace=True)

    basis_gates = ['h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz', 'cx', 'cz', 'swap', 'u', 'u1', 'u2', 'u3', 'p', 'sx', 'id']
    tqc = transpile(circ_clean, basis_gates=basis_gates)

    mps = MatrixProductState(num_qubits, max_bond_dim=bond_dimension, cutoff=cutoff)

    t0 = time.perf_counter()

    for inst in tqc.data:
        op = inst.operation
        if op.name in ('barrier', 'measure', 'id'):
            continue
        q_indices = [tqc.find_bit(q).index for q in inst.qubits]
        try:
            mat = op.to_matrix()
        except Exception:
            continue

        if len(q_indices) == 1:
            mps.apply_1q(q_indices[0], mat)
        elif len(q_indices) == 2:
            mps.apply_2q(q_indices[0], q_indices[1], mat)

    latency = time.perf_counter() - t0

    probs = mps.get_probabilities() if num_qubits <= 20 else None
    counts = mps.sample_counts(shots=shots, seed=seed)
    sv = mps.to_statevector() if num_qubits <= 16 else None

    return {
        "success": True,
        "qubits": num_qubits,
        "latency": latency,
        "mean_latency": latency,
        "median_latency": latency,
        "std_latency": 0.0,
        "runs_count": 1,
        "counts": counts,
        "probabilities": probs,
        "statevector": sv,
        "mps_fidelity": mps.mps_fidelity,
        "cumulative_truncation_error": mps.cumulative_truncation_error,
        "max_active_bond_dim": mps.max_active_bond_dim,
        "swap_count": mps.swap_count,
        "method": "mps",
        "bond_dimension": bond_dimension,
        "cutoff": cutoff,
        "shots": shots,
        "error": None
    }


def get_mps_configuration(bond_dim: int) -> Dict[str, Any]:
    """
    Get the configuration dictionary for AerSimulator in Matrix Product State (MPS) mode.

    Args:
        bond_dim (int): Maximum bond dimension for MPS.

    Returns:
        dict: The Aer configuration settings dictionary.

    Raises:
        TypeError: If bond_dim is not an integer.
        ValueError: If bond_dim is less than 1.
    """
    if not isinstance(bond_dim, int):
        raise TypeError("bond_dim must be an integer.")
    if bond_dim < 1:
        raise ValueError("bond_dim must be at least 1.")

    return {
        "matrix_product_state_max_bond_dimension": bond_dim
    }


def calculate_mps_ram_savings(num_qubits: int, actual_ram_used: float) -> Dict[str, Any]:
    """
    Calculate the RAM savings of MPS compared to theoretical Statevector simulation.

    Args:
        num_qubits (int): Number of simulated qubits.
        actual_ram_used (float): Actual physical RAM used in bytes.

    Returns:
        dict: A dictionary containing:
            - "statevector_ram_est_bytes" (int): Theoretical RAM allocation for Statevector.
            - "actual_ram_used_bytes" (float): Actual RAM used by simulation.
            - "savings_bytes" (float): Raw savings in bytes (Statevector RAM - Actual RAM).
            - "savings_percent" (float): Savings percentage relative to Statevector.

    Raises:
        TypeError: If num_qubits or actual_ram_used types are incorrect.
        ValueError: If num_qubits or actual_ram_used is negative.
    """
    if not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer.")
    if not isinstance(actual_ram_used, (int, float)):
        raise TypeError("actual_ram_used must be a number.")

    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative.")
    if actual_ram_used < 0:
        raise ValueError("actual_ram_used must be non-negative.")

    # Theoretical statevector memory allocation: 2^n * 16 bytes
    statevector_ram_est = (2 ** num_qubits) * 16

    # Calculate savings
    savings_bytes = max(0.0, statevector_ram_est - actual_ram_used)

    if statevector_ram_est > 0:
        savings_percent = (savings_bytes / statevector_ram_est) * 100
    else:
        savings_percent = 0.0

    return {
        "statevector_ram_est_bytes": statevector_ram_est,
        "actual_ram_used_bytes": float(actual_ram_used),
        "savings_bytes": float(savings_bytes),
        "savings_percent": float(savings_percent)
    }
