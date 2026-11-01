"""
Stabilizer Tableau Engine based on the Aaronson-Gottesman CHP Algorithm.

Reference:
    S. Aaronson and D. Gottesman, "Improved Simulation of Stabilizer Circuits",
    Physical Review A 70, 052328 (2004). https://arxiv.org/abs/quant-ph/0406196

Represents an n-qubit stabilizer state using a binary symplectic tableau of
dimensions (2n + 1) x (2n + 1) over GF(2):
    - Rows 0 .. n-1: Destabilizer generators (R_0 .. R_{n-1})
    - Rows n .. 2n-1: Stabilizer generators (R_n .. R_{2n-1})
    - Row 2n: Scratchpad row for Pauli group multiplication (rowsum)
"""

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit


class NonCliffordGateError(ValueError):
    """Exception raised when a non-Clifford operation is encountered by the Stabilizer engine."""
    pass


def is_clifford_angle(theta: Any, tol: float = 1e-7) -> Tuple[bool, int]:
    """
    Determine whether a rotation angle theta is an integer multiple of pi/2.

    Args:
        theta (Any): Rotation angle in radians.
        tol (float): Numerical tolerance for multiple comparison.

    Returns:
        Tuple[bool, int]: (is_clifford, k_mod_4) where theta = k * (pi/2).
    """
    try:
        val = float(theta)
    except (TypeError, ValueError):
        return False, 0

    half_pi = math.pi / 2.0
    k = round(val / half_pi)
    if abs(val - k * half_pi) <= tol:
        return True, int(k) % 4
    return False, 0


class StabilizerTableau:
    """
    Symplectic binary tableau simulator for n-qubit Clifford circuits.

    Memory complexity: O(n^2) bytes.
    Unitary Clifford gate complexity: O(n) per gate via vectorized column updates.
    Computational-basis measurement complexity: O(n^2).
    """

    def __init__(self, num_qubits: int, _init_identity: bool = True) -> None:
        if not isinstance(num_qubits, int):
            raise TypeError("num_qubits must be an integer.")
        if num_qubits < 1:
            raise ValueError("num_qubits must be at least 1.")

        self.num_qubits: int = num_qubits
        n = num_qubits

        # Rows 0..n-1: Destabilizers; Rows n..2n-1: Stabilizers; Row 2n: Scratchpad
        self.x: np.ndarray = np.zeros((2 * n + 1, n), dtype=np.uint8)
        self.z: np.ndarray = np.zeros((2 * n + 1, n), dtype=np.uint8)
        self.r: np.ndarray = np.zeros(2 * n + 1, dtype=np.uint8)
        self.clifford_gate_count: int = 0

        if _init_identity:
            idx = np.arange(n)
            self.x[idx, idx] = 1
            self.z[idx + n, idx] = 1

    def copy(self) -> "StabilizerTableau":
        """Create an independent copy of the stabilizer tableau."""
        clone = StabilizerTableau(self.num_qubits, _init_identity=False)
        clone.x = self.x.copy()
        clone.z = self.z.copy()
        clone.r = self.r.copy()
        clone.clifford_gate_count = self.clifford_gate_count
        return clone

    @property
    def tableau_memory_bytes(self) -> int:
        """Total memory footprint of the binary symplectic tableau arrays in bytes."""
        return int(self.x.nbytes + self.z.nbytes + self.r.nbytes)

    def _validate_qubit(self, a: int) -> None:
        if not isinstance(a, (int, np.integer)):
            raise TypeError(f"Qubit index must be an integer, got {type(a).__name__}.")
        if a < 0 or a >= self.num_qubits:
            raise IndexError(f"Qubit index {a} out of range for {self.num_qubits}-qubit tableau.")

    # =========================================================================
    # Unitary Clifford Gate Transformations (O(n) per gate)
    # =========================================================================

    def apply_id(self, a: int) -> None:
        """Identity gate I(a)."""
        self._validate_qubit(a)
        self.clifford_gate_count += 1

    id = apply_id

    def apply_h(self, a: int) -> None:
        """Hadamard gate H(a): r_i ^= x_{i,a} & z_{i,a}; swap(x_{i,a}, z_{i,a})."""
        self._validate_qubit(a)
        n2 = 2 * self.num_qubits
        xa = self.x[:n2, a]
        za = self.z[:n2, a]
        self.r[:n2] ^= (xa & za)
        tmp = xa.copy()
        self.x[:n2, a] = za
        self.z[:n2, a] = tmp
        self.clifford_gate_count += 1

    h = apply_h

    def apply_s(self, a: int) -> None:
        """Phase gate S(a): r_i ^= x_{i,a} & z_{i,a}; z_{i,a} ^= x_{i,a}."""
        self._validate_qubit(a)
        n2 = 2 * self.num_qubits
        xa = self.x[:n2, a]
        za = self.z[:n2, a]
        self.r[:n2] ^= (xa & za)
        self.z[:n2, a] ^= xa
        self.clifford_gate_count += 1

    s = apply_s

    def apply_sdg(self, a: int) -> None:
        """Inverse Phase gate S^dagger(a): r_i ^= x_{i,a} & (1 ^ z_{i,a}); z_{i,a} ^= x_{i,a}."""
        self._validate_qubit(a)
        n2 = 2 * self.num_qubits
        xa = self.x[:n2, a]
        za = self.z[:n2, a]
        self.r[:n2] ^= (xa & (1 ^ za))
        self.z[:n2, a] ^= xa
        self.clifford_gate_count += 1

    sdg = apply_sdg

    def apply_x(self, a: int) -> None:
        """Pauli-X gate X(a): r_i ^= z_{i,a}."""
        self._validate_qubit(a)
        n2 = 2 * self.num_qubits
        self.r[:n2] ^= self.z[:n2, a]
        self.clifford_gate_count += 1

    def apply_y(self, a: int) -> None:
        """Pauli-Y gate Y(a): r_i ^= x_{i,a} ^ z_{i,a}."""
        self._validate_qubit(a)
        n2 = 2 * self.num_qubits
        self.r[:n2] ^= (self.x[:n2, a] ^ self.z[:n2, a])
        self.clifford_gate_count += 1

    y = apply_y

    def apply_z(self, a: int) -> None:
        """Pauli-Z gate Z(a): r_i ^= x_{i,a}."""
        self._validate_qubit(a)
        n2 = 2 * self.num_qubits
        self.r[:n2] ^= self.x[:n2, a]
        self.clifford_gate_count += 1

    def apply_sx(self, a: int) -> None:
        """Sqrt(X) gate SX(a) = H(a) S(a) H(a)."""
        self.apply_h(a)
        self.apply_s(a)
        self.apply_h(a)
        self.clifford_gate_count -= 2

    sx = apply_sx

    def apply_sxdg(self, a: int) -> None:
        """Inverse Sqrt(X) gate SX^dagger(a) = H(a) S^dagger(a) H(a)."""
        self.apply_h(a)
        self.apply_sdg(a)
        self.apply_h(a)
        self.clifford_gate_count -= 2

    sxdg = apply_sxdg

    def apply_cx(self, a: int, b: int) -> None:
        """
        Controlled-NOT gate CX(a, b) with control a and target b:
            r_i ^= x_{i,a} & z_{i,b} & (x_{i,b} ^ z_{i,a} ^ 1)
            x_{i,b} ^= x_{i,a}
            z_{i,a} ^= z_{i,b}
        """
        self._validate_qubit(a)
        self._validate_qubit(b)
        if a == b:
            raise ValueError("Control and target qubits for CX must be distinct.")
        n2 = 2 * self.num_qubits
        xa = self.x[:n2, a]
        za = self.z[:n2, a]
        xb = self.x[:n2, b]
        zb = self.z[:n2, b]

        self.r[:n2] ^= (xa & zb & (xb ^ za ^ 1))
        self.x[:n2, b] ^= xa
        self.z[:n2, a] ^= zb
        self.clifford_gate_count += 1

    cx = apply_cx

    def apply_cz(self, a: int, b: int) -> None:
        """Controlled-Z gate CZ(a, b) = H(b) CX(a, b) H(b)."""
        self._validate_qubit(a)
        self._validate_qubit(b)
        if a == b:
            raise ValueError("Control and target qubits for CZ must be distinct.")
        self.apply_h(b)
        self.apply_cx(a, b)
        self.apply_h(b)
        self.clifford_gate_count -= 2

    cz = apply_cz

    def apply_cy(self, a: int, b: int) -> None:
        """Controlled-Y gate CY(a, b) = S^dagger(b) CX(a, b) S(b)."""
        self._validate_qubit(a)
        self._validate_qubit(b)
        if a == b:
            raise ValueError("Control and target qubits for CY must be distinct.")
        self.apply_sdg(b)
        self.apply_cx(a, b)
        self.apply_s(b)
        self.clifford_gate_count -= 2

    cy = apply_cy

    def apply_swap(self, a: int, b: int) -> None:
        """SWAP gate SWAP(a, b): swap columns a and b in both X and Z."""
        self._validate_qubit(a)
        self._validate_qubit(b)
        if a == b:
            self.clifford_gate_count += 1
            return
        n2 = 2 * self.num_qubits
        x_tmp = self.x[:n2, a].copy()
        self.x[:n2, a] = self.x[:n2, b]
        self.x[:n2, b] = x_tmp

        z_tmp = self.z[:n2, a].copy()
        self.z[:n2, a] = self.z[:n2, b]
        self.z[:n2, b] = z_tmp
        self.clifford_gate_count += 1

    swap = apply_swap

    def apply_iswap(self, a: int, b: int) -> None:
        """iSWAP gate decomposition: S(a) S(b) H(a) CX(a, b) CX(b, a) H(b)."""
        self.apply_s(a)
        self.apply_s(b)
        self.apply_h(a)
        self.apply_cx(a, b)
        self.apply_cx(b, a)
        self.apply_h(b)
        self.clifford_gate_count -= 5

    iswap = apply_iswap

    def apply_dcx(self, a: int, b: int) -> None:
        """Double-CNOT gate DCX(a, b) = CX(a, b) CX(b, a)."""
        self.apply_cx(a, b)
        self.apply_cx(b, a)
        self.clifford_gate_count -= 1

    dcx = apply_dcx

    def apply_ecr(self, a: int, b: int) -> None:
        """Echoed Cross-Resonance gate ECR(a, b)."""
        self.apply_s(a)
        self.apply_sx(b)
        self.apply_cx(a, b)
        self.apply_x(a)
        self.clifford_gate_count -= 3

    ecr = apply_ecr

    def apply_rz(self, a: int, theta: float) -> None:
        """Rotation around Z by multiple of pi/2."""
        self._validate_qubit(a)
        ok, k = is_clifford_angle(theta)
        if not ok:
            raise NonCliffordGateError(
                f"Non-Clifford rotation rz({theta}) on qubit {a}: angle is not an integer multiple of pi/2."
            )
        if k == 0:
            self.apply_id(a)
        elif k == 1:
            self.apply_s(a)
        elif k == 2:
            self.apply_z(a)
        elif k == 3:
            self.apply_sdg(a)

    rz = apply_rz

    def apply_rx(self, a: int, theta: float) -> None:
        """Rotation around X by multiple of pi/2."""
        self._validate_qubit(a)
        ok, k = is_clifford_angle(theta)
        if not ok:
            raise NonCliffordGateError(
                f"Non-Clifford rotation rx({theta}) on qubit {a}: angle is not an integer multiple of pi/2."
            )
        if k == 0:
            self.apply_id(a)
        elif k == 1:
            self.apply_sx(a)
        elif k == 2:
            self.apply_x(a)
        elif k == 3:
            self.apply_sxdg(a)

    rx = apply_rx

    def apply_ry(self, a: int, theta: float) -> None:
        """Rotation around Y by multiple of pi/2."""
        self._validate_qubit(a)
        ok, k = is_clifford_angle(theta)
        if not ok:
            raise NonCliffordGateError(
                f"Non-Clifford rotation ry({theta}) on qubit {a}: angle is not an integer multiple of pi/2."
            )
        if k == 0:
            self.apply_id(a)
        elif k == 1:
            # Ry(pi/2) = S Rx(pi/2) S^dagger = H Z (in circuit order: Z then H)
            self.apply_z(a)
            self.apply_h(a)
            self.clifford_gate_count -= 1
        elif k == 2:
            self.apply_y(a)
        elif k == 3:
            # Ry(-pi/2) = H X (in circuit order: X then H)
            self.apply_x(a)
            self.apply_h(a)
            self.clifford_gate_count -= 1

    ry = apply_ry

    # =========================================================================
    # Pauli Row Multiplication (rowsum) & Projective Measurement (O(n^2))
    # =========================================================================

    @staticmethod
    def _phase_exponent_g(
        x1: np.ndarray, z1: np.ndarray, x2: np.ndarray, z2: np.ndarray
    ) -> np.ndarray:
        """
        Compute the Aaronson-Gottesman phase exponent function g(x1, z1, x2, z2):
            - x1=0, z1=0: 0
            - x1=1, z1=1: z2 - x2
            - x1=1, z1=0: z2 * (2*x2 - 1)
            - x1=0, z1=1: x2 * (1 - 2*z2)
        """
        x1_i = x1.astype(np.int32)
        z1_i = z1.astype(np.int32)
        x2_i = x2.astype(np.int32)
        z2_i = z2.astype(np.int32)

        return (
            (x1_i & z1_i) * (z2_i - x2_i)
            + (x1_i & (1 - z1_i)) * z2_i * (2 * x2_i - 1)
            + ((1 - x1_i) & z1_i) * x2_i * (1 - 2 * z2_i)
        )

    def rowsum(self, h: int, i: int) -> None:
        """
        Multiply Pauli row h on the left by Pauli row i (R_h <- R_i * R_h),
        updating the phase bit r_h in accordance with Aaronson-Gottesman (2004):
            2*r_h + 2*r_i + sum_j g(x_{i,j}, z_{i,j}, x_{h,j}, z_{h,j}) (mod 4).
        """
        g = self._phase_exponent_g(self.x[i], self.z[i], self.x[h], self.z[h])
        total_phase = (2 * int(self.r[h]) + 2 * int(self.r[i]) + int(np.sum(g))) % 4
        self.r[h] = 1 if total_phase == 2 else 0
        self.x[h] ^= self.x[i]
        self.z[h] ^= self.z[i]

    def _rowsum_batch(self, rows_h: np.ndarray, i: int) -> None:
        """Vectorized rowsum applying row i to multiple target rows rows_h simultaneously."""
        if len(rows_h) == 0:
            return
        g = self._phase_exponent_g(
            self.x[i][None, :], self.z[i][None, :], self.x[rows_h], self.z[rows_h]
        )
        total_phase = (
            2 * self.r[rows_h].astype(np.int32)
            + 2 * int(self.r[i])
            + np.sum(g, axis=1)
        ) % 4
        self.r[rows_h] = (total_phase == 2).astype(np.uint8)
        self.x[rows_h] ^= self.x[i][None, :]
        self.z[rows_h] ^= self.z[i][None, :]

    def _deterministic_outcome(self, a: int) -> int:
        """
        Compute the deterministic measurement outcome for qubit a using the
        scratchpad row 2n and vectorized prefix-XOR accumulation of rowsum.
        """
        n = self.num_qubits
        active_destab = np.nonzero(self.x[:n, a])[0]
        if len(active_destab) == 0:
            self.x[2 * n].fill(0)
            self.z[2 * n].fill(0)
            self.r[2 * n] = 0
            return 0

        stab_rows = active_destab + n
        if len(stab_rows) == 1:
            row_idx = int(stab_rows[0])
            self.x[2 * n] = self.x[row_idx]
            self.z[2 * n] = self.z[row_idx]
            self.r[2 * n] = self.r[row_idx]
            return int(self.r[2 * n])

        # Vectorized prefix-XOR across stab_rows to evaluate all sequential rowsums at once
        x_seq = self.x[stab_rows]
        z_seq = self.z[stab_rows]
        x_cum = np.cumsum(x_seq, axis=0, dtype=np.int32) & 1
        z_cum = np.cumsum(z_seq, axis=0, dtype=np.int32) & 1

        # At step m >= 1, the scratchpad before multiplying stab_rows[m] holds x_cum[m-1], z_cum[m-1]
        g_matrix = self._phase_exponent_g(
            x_seq[1:], z_seq[1:], x_cum[:-1], z_cum[:-1]
        )
        total_phase = (
            2 * int(np.sum(self.r[stab_rows])) + int(np.sum(g_matrix))
        ) % 4
        outcome = 1 if total_phase == 2 else 0

        self.x[2 * n] = x_cum[-1].astype(np.uint8)
        self.z[2 * n] = z_cum[-1].astype(np.uint8)
        self.r[2 * n] = outcome
        return outcome

    def measure(self, a: int, rng: Optional[np.random.Generator] = None) -> int:
        """
        Measure qubit a in the computational (Z) basis.

        Args:
            a (int): Index of qubit to measure (0 <= a < n).
            rng (Optional[np.random.Generator]): Random generator for indeterministic collapse.

        Returns:
            int: Measurement result bit (0 or 1).
        """
        self._validate_qubit(a)
        if rng is None:
            rng = np.random.default_rng()

        n = self.num_qubits
        p_candidates = np.nonzero(self.x[n:2 * n, a])[0]

        if len(p_candidates) > 0:
            # Case I: Indeterministic measurement outcome
            p = int(p_candidates[0]) + n
            all_rows = np.nonzero(self.x[:2 * n, a])[0]
            other_rows = all_rows[all_rows != p]
            if len(other_rows) > 0:
                self._rowsum_batch(other_rows, p)

            self.x[p - n] = self.x[p]
            self.z[p - n] = self.z[p]
            self.r[p - n] = self.r[p]

            self.x[p].fill(0)
            self.z[p].fill(0)
            self.z[p, a] = 1
            outcome = int(rng.integers(0, 2))
            self.r[p] = outcome
            return outcome
        else:
            # Case II: Deterministic measurement outcome
            return self._deterministic_outcome(a)

    def reset(self, a: int, rng: Optional[np.random.Generator] = None) -> None:
        """Projectively reset qubit a to |0>."""
        outcome = self.measure(a, rng=rng)
        if outcome == 1:
            self.apply_x(a)
            self.clifford_gate_count -= 1

    def sample_measurements(
        self,
        shots: int = 1024,
        seed: Optional[int] = None,
        measured_Map: Optional[List[Tuple[int, int]]] = None,
        num_clbits: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Perform multi-shot computational-basis sampling from the current stabilizer state.

        Exploits the symplectic invariance of (X, Z) under computational-basis measurement:
        the (X, Z) tableau trajectory and Pauli phase increments g(x1, z1, x2, z2) are
        computed once, while the binary phase vectors R of shape (2n, shots) are propagated
        in parallel across all shots simultaneously.

        Args:
            shots (int): Number of measurement shots.
            seed (Optional[int]): Random seed for reproducible sampling.
            measured_Map (Optional[List[Tuple[int, int]]]): Optional list of (qubit_idx, clbit_idx) pairs.
            num_clbits (Optional[int]): Number of classical bits if measured_Map is provided.

        Returns:
            Dict[str, int]: Dictionary mapping bitstrings (little-endian) to observed counts.
        """
        if shots < 1:
            raise ValueError("shots must be >= 1.")

        rng = np.random.default_rng(seed)
        n = self.num_qubits
        work = self.copy()

        if measured_Map is not None and len(measured_Map) > 0:
            # Deduplicate while preserving measurement order
            seen_q = {}
            for q_idx, c_idx in measured_Map:
                seen_q[q_idx] = c_idx
            measure_qubits = list(seen_q.keys())
            c_len = num_clbits if (num_clbits is not None and num_clbits > 0) else n
        else:
            measure_qubits = list(range(n))
            seen_q = {q: q for q in range(n)}
            c_len = n

        # Phase matrix R of shape (2*n, shots) for parallel shot evolution
        R = np.repeat(work.r[:2 * n, None], shots, axis=1)
        qubit_outcomes: Dict[int, np.ndarray] = {}

        for a in measure_qubits:
            p_candidates = np.nonzero(work.x[n:2 * n, a])[0]
            if len(p_candidates) > 0:
                p = int(p_candidates[0]) + n
                all_rows = np.nonzero(work.x[:2 * n, a])[0]
                other_rows = all_rows[(all_rows != p) & (all_rows != (p - n))]

                if len(other_rows) > 0:
                    g = work._phase_exponent_g(
                        work.x[p][None, :],
                        work.z[p][None, :],
                        work.x[other_rows],
                        work.z[other_rows]
                    )
                    g_bit = ((np.sum(g, axis=1) // 2) & 1).astype(np.uint8)
                    R[other_rows, :] ^= R[p, None, :] ^ g_bit[:, None]
                    work.x[other_rows] ^= work.x[p][None, :]
                    work.z[other_rows] ^= work.z[p][None, :]

                work.x[p - n] = work.x[p]
                work.z[p - n] = work.z[p]
                R[p - n, :] = R[p, :]

                work.x[p].fill(0)
                work.z[p].fill(0)
                work.z[p, a] = 1

                outcomes_a = rng.integers(0, 2, size=shots, dtype=np.uint8)
                R[p, :] = outcomes_a
                qubit_outcomes[a] = outcomes_a
            else:
                active_destab = np.nonzero(work.x[:n, a])[0]
                if len(active_destab) == 0:
                    qubit_outcomes[a] = np.zeros(shots, dtype=np.uint8)
                else:
                    stab_rows = active_destab + n
                    if len(stab_rows) == 1:
                        qubit_outcomes[a] = R[stab_rows[0], :].copy()
                    else:
                        x_seq = work.x[stab_rows]
                        z_seq = work.z[stab_rows]
                        x_cum = np.cumsum(x_seq, axis=0, dtype=np.int32) & 1
                        z_cum = np.cumsum(z_seq, axis=0, dtype=np.int32) & 1
                        g_matrix = work._phase_exponent_g(
                            x_seq[1:], z_seq[1:], x_cum[:-1], z_cum[:-1]
                        )
                        g_det_bit = int((np.sum(g_matrix) // 2) & 1)
                        r_xor = np.bitwise_xor.reduce(R[stab_rows, :], axis=0)
                        qubit_outcomes[a] = r_xor ^ g_det_bit

        # Assemble classical register bitstrings in Qiskit little-endian order (c_{c_len-1} ... c_1 c_0)
        clbit_matrix = np.zeros((c_len, shots), dtype=np.uint8)
        for q_idx, c_idx in seen_q.items():
            if 0 <= c_idx < c_len and q_idx in qubit_outcomes:
                clbit_matrix[c_len - 1 - c_idx, :] = qubit_outcomes[q_idx]

        # Convert columns of clbit_matrix into bitstrings and count unique occurrences
        counts: Dict[str, int] = {}
        # Transpose to (shots, c_len) for fast byte/string conversion
        shots_bits = clbit_matrix.T
        unique_rows, row_counts = np.unique(shots_bits, axis=0, return_counts=True)
        for row, cnt in zip(unique_rows, row_counts):
            bs = "".join("1" if b else "0" for b in row)
            counts[bs] = int(cnt)

        return counts

    def apply_gate(
        self,
        name: str,
        qubits: List[int],
        params: Optional[List[float]] = None,
        rng: Optional[np.random.Generator] = None
    ) -> Optional[int]:
        """
        Dispatch a single gate instruction by name onto the tableau.

        Raises:
            NonCliffordGateError: If the operation is outside the Clifford group.
        """
        op_name = name.lower().strip()
        p = params or []

        if op_name in ("barrier", "delay"):
            return None
        if op_name in ("id", "i"):
            self.apply_id(qubits[0])
            return None
        if op_name == "h":
            self.apply_h(qubits[0])
            return None
        if op_name == "s":
            self.apply_s(qubits[0])
            return None
        if op_name == "sdg":
            self.apply_sdg(qubits[0])
            return None
        if op_name == "x":
            self.apply_x(qubits[0])
            return None
        if op_name == "y":
            self.apply_y(qubits[0])
            return None
        if op_name == "z":
            self.apply_z(qubits[0])
            return None
        if op_name == "sx":
            self.apply_sx(qubits[0])
            return None
        if op_name == "sxdg":
            self.apply_sxdg(qubits[0])
            return None
        if op_name in ("cx", "cnot"):
            self.apply_cx(qubits[0], qubits[1])
            return None
        if op_name == "cz":
            self.apply_cz(qubits[0], qubits[1])
            return None
        if op_name == "cy":
            self.apply_cy(qubits[0], qubits[1])
            return None
        if op_name == "swap":
            self.apply_swap(qubits[0], qubits[1])
            return None
        if op_name == "iswap":
            self.apply_iswap(qubits[0], qubits[1])
            return None
        if op_name == "dcx":
            self.apply_dcx(qubits[0], qubits[1])
            return None
        if op_name == "ecr":
            self.apply_ecr(qubits[0], qubits[1])
            return None
        if op_name in ("rz", "p", "u1"):
            if len(p) < 1:
                raise ValueError(f"Gate '{op_name}' requires 1 angle parameter.")
            self.apply_rz(qubits[0], float(p[0]))
            return None
        if op_name == "rx":
            if len(p) < 1:
                raise ValueError("Gate 'rx' requires 1 angle parameter.")
            self.apply_rx(qubits[0], float(p[0]))
            return None
        if op_name == "ry":
            if len(p) < 1:
                raise ValueError("Gate 'ry' requires 1 angle parameter.")
            self.apply_ry(qubits[0], float(p[0]))
            return None
        if op_name == "u2":
            if len(p) < 2:
                raise ValueError("Gate 'u2' requires 2 angle parameters (phi, lam).")
            ok1, _ = is_clifford_angle(p[0])
            ok2, _ = is_clifford_angle(p[1])
            if not (ok1 and ok2):
                raise NonCliffordGateError(
                    f"Non-Clifford u2({p[0]}, {p[1]}) gate on qubit {qubits[0]}."
                )
            self.apply_rz(qubits[0], float(p[1]))
            self.apply_ry(qubits[0], math.pi / 2.0)
            self.apply_rz(qubits[0], float(p[0]))
            self.clifford_gate_count -= 2
            return None
        if op_name in ("u", "u3"):
            if len(p) < 3:
                raise ValueError(f"Gate '{op_name}' requires 3 angle parameters (theta, phi, lam).")
            ok1, _ = is_clifford_angle(p[0])
            ok2, _ = is_clifford_angle(p[1])
            ok3, _ = is_clifford_angle(p[2])
            if not (ok1 and ok2 and ok3):
                raise NonCliffordGateError(
                    f"Non-Clifford {op_name}({p[0]}, {p[1]}, {p[2]}) gate on qubit {qubits[0]}."
                )
            self.apply_rz(qubits[0], float(p[2]))
            self.apply_ry(qubits[0], float(p[0]))
            self.apply_rz(qubits[0], float(p[1]))
            self.clifford_gate_count -= 2
            return None
        if op_name in ("cp", "cu1", "crz"):
            if len(p) < 1:
                raise ValueError(f"Gate '{op_name}' requires 1 angle parameter.")
            ok, k = is_clifford_angle(p[0])
            # Controlled-phase is Clifford only when angle is an integer multiple of pi (k = 0 or 2 mod 4)
            if not ok or (k % 2 != 0):
                raise NonCliffordGateError(
                    f"Non-Clifford controlled rotation {op_name}({p[0]}) on qubits {qubits}."
                )
            if k == 2:
                self.apply_cz(qubits[0], qubits[1])
            else:
                self.clifford_gate_count += 1
            return None
        if op_name == "reset":
            self.reset(qubits[0], rng=rng)
            return None
        if op_name == "measure":
            return self.measure(qubits[0], rng=rng)

        raise NonCliffordGateError(
            f"Unsupported non-Clifford gate '{op_name}' on qubit(s) {qubits}. "
            f"The Aaronson-Gottesman Stabilizer Tableau engine only supports Clifford operations."
        )


def calculate_stabilizer_ram_savings(num_qubits: int, tableau_memory_bytes: Optional[int] = None) -> Dict[str, Any]:
    """
    Calculate the RAM savings of the O(n^2) Stabilizer Tableau compared to O(2^n) Statevector.

    Args:
        num_qubits (int): Number of qubits simulated.
        tableau_memory_bytes (Optional[int]): Actual memory footprint of the stabilizer tableau in bytes.

    Returns:
        Dict[str, Any]: RAM comparison metrics.
    """
    if tableau_memory_bytes is None:
        tableau_memory_bytes = (2 * num_qubits + 1) * (2 * num_qubits + 1)
    statevector_ram_est_int = (1 << num_qubits) * 16
    # Prevent IEEE 754 float64 overflow for extreme qubit counts (n >= 1020)
    if num_qubits < 1000:
        sv_float = float(statevector_ram_est_int)
        savings_bytes = max(0.0, sv_float - float(tableau_memory_bytes))
        savings_pct = (savings_bytes / sv_float) * 100.0 if sv_float > 0 else 0.0
    else:
        sv_float = float("inf")
        savings_bytes = float("inf")
        savings_pct = 100.0

    return {
        "statevector_ram_est_bytes": statevector_ram_est_int,
        "actual_ram_used_bytes": float(tableau_memory_bytes),
        "tableau_memory_bytes": int(tableau_memory_bytes),
        "savings_bytes": savings_bytes,
        "savings_percent": float(savings_pct)
    }


def _extract_instructions_from_circuit(
    circuit: Any
) -> Tuple[int, int, List[Tuple[str, List[int], List[float], List[int], Optional[Tuple[int, int]]]]]:
    """
    Extract a normalized instruction sequence from a QuantumCircuit, ParsedQASMCircuit, or QASM file/string.

    Returns:
        Tuple of:
            - num_qubits (int)
            - num_clbits (int)
            - instructions: List of (name, q_indices, params, c_indices, condition)
    """
    from src.engine.parser import ParsedQASMCircuit, load_qasm, parse_qasm

    if isinstance(circuit, str):
        if os.path.exists(circuit) or circuit.lower().endswith(".qasm"):
            parsed = load_qasm(circuit)
        else:
            parsed = parse_qasm(circuit)
        return _extract_instructions_from_circuit(parsed)

    if isinstance(circuit, ParsedQASMCircuit):
        qmap = circuit.qubit_map
        cmap = circuit.clbit_map
        inst_list = []
        for inst in circuit.instructions:
            q_idx = [qmap[qa] for qa in inst.qargs]
            c_idx = [cmap[ca] for ca in inst.cargs] if inst.cargs else []
            inst_list.append((inst.name.lower(), q_idx, list(inst.params), c_idx, None))
        return circuit.num_qubits, circuit.num_clbits, inst_list

    if isinstance(circuit, QuantumCircuit):
        num_qubits = circuit.num_qubits
        num_clbits = circuit.num_clbits
        inst_list = []

        def _flatten_qc(qc_obj: QuantumCircuit, q_remap: List[int], c_remap: List[int]) -> None:
            for inst in qc_obj.data:
                op = inst.operation
                name = op.name.lower()
                q_idx = [q_remap[qc_obj.find_bit(q).index] for q in inst.qubits]
                c_idx = [c_remap[qc_obj.find_bit(c).index] for c in inst.clbits]
                params = []
                for p in getattr(op, "params", []):
                    try:
                        params.append(float(p))
                    except (TypeError, ValueError) as exc:
                        raise NonCliffordGateError(
                            f"Unbound or non-numeric parameter '{p}' in gate '{name}'."
                        ) from exc

                # Check classical condition if present
                cond_tuple: Optional[Tuple[int, int]] = None
                condition = getattr(op, "condition", None)
                if condition is not None:
                    target, val = condition
                    if hasattr(target, "index"):
                        cond_tuple = (qc_obj.find_bit(target).index, int(val))
                    else:
                        cond_tuple = (0, int(val))

                # Handle composite custom gates that are not native Clifford/non-Clifford primitives
                known_primitives = {
                    "id", "i", "h", "s", "sdg", "x", "y", "z", "sx", "sxdg",
                    "cx", "cnot", "cz", "cy", "swap", "iswap", "dcx", "ecr",
                    "rx", "ry", "rz", "p", "u1", "u2", "u", "u3",
                    "cp", "cu1", "crz", "t", "tdg", "ccx", "toffoli", "cswap", "fredkin",
                    "ch", "csx", "ccz", "crx", "cry", "cu", "cu3",
                    "barrier", "delay", "measure", "reset"
                }
                if name not in known_primitives and getattr(op, "definition", None) is not None:
                    sub_qc = op.definition
                    sub_c_remap = c_idx if len(c_idx) == sub_qc.num_clbits else list(range(sub_qc.num_clbits))
                    _flatten_qc(sub_qc, q_idx, sub_c_remap)
                else:
                    inst_list.append((name, q_idx, params, c_idx, cond_tuple))

        _flatten_qc(circuit, list(range(num_qubits)), list(range(num_clbits)))
        return num_qubits, num_clbits, inst_list

    raise TypeError(
        "circuit must be a Qiskit QuantumCircuit, ParsedQASMCircuit, or QASM file path/string."
    )


def simulate_stabilizer(
    circuit: Any,
    shots: int = 1024,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Execute an n-qubit Clifford circuit using the Aaronson-Gottesman Stabilizer Tableau engine.

    Optimization:
        If the circuit is a pure unitary Clifford circuit (with optional terminal measurements
        and no mid-circuit resets or mid-circuit measurements), the tableau is evolved once
        and sampled across all shots in parallel via vectorized phase propagation.

    Args:
        circuit (Any): QuantumCircuit, ParsedQASMCircuit, or OpenQASM 2.0 path/string.
        shots (int): Number of measurement shots (default 1024).
        seed (Optional[int]): Random seed for reproducible sampling.

    Returns:
        Dict[str, Any]: Structured simulation telemetry including counts, probabilities,
            num_qubits, tableau_memory_bytes, ram_savings_vs_statevector, and clifford_gate_count.

    Raises:
        NonCliffordGateError: If the circuit contains any non-Clifford gate.
    """
    if not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be a positive integer >= 1.")

    num_qubits, num_clbits, instructions = _extract_instructions_from_circuit(circuit)
    if num_qubits < 1:
        raise ValueError("Circuit must contain at least 1 qubit.")

    # Separate unitary prefix from terminal measurements, or detect dynamic mid-circuit ops
    has_mid_circuit_nonunitary = False
    seen_measurement_on_qubit = set()
    unitary_ops: List[Tuple[str, List[int], List[float]]] = []
    terminal_measurements: List[Tuple[int, int]] = []

    for name, q_idx, params, c_idx, cond in instructions:
        if name in ("barrier", "delay"):
            continue
        if cond is not None or name == "reset":
            has_mid_circuit_nonunitary = True
            break
        if name == "measure":
            seen_measurement_on_qubit.update(q_idx)
            c_target = c_idx[0] if c_idx else q_idx[0]
            terminal_measurements.append((q_idx[0], c_target))
        else:
            if any(q in seen_measurement_on_qubit for q in q_idx):
                has_mid_circuit_nonunitary = True
                break
            unitary_ops.append((name, q_idx, params))

    t0 = time.perf_counter()
    active_tableau = None

    if not has_mid_circuit_nonunitary:
        # Fast-Path: Evolve unitary Clifford tableau once, then batch-sample all shots
        tableau = StabilizerTableau(num_qubits)
        for name, q_idx, params in unitary_ops:
            tableau.apply_gate(name, q_idx, params)

        active_tableau = tableau
        counts = tableau.sample_measurements(
            shots=shots,
            seed=seed,
            measured_Map=terminal_measurements if terminal_measurements else None,
            num_clbits=num_clbits if terminal_measurements else num_qubits
        )
        mem_bytes = tableau.tableau_memory_bytes
        clifford_count = tableau.clifford_gate_count
    else:
        # Dynamic Mid-Circuit Path (resets, mid-circuit measurements, classical feed-forward)
        rng = np.random.default_rng(seed)
        counts = {}
        c_len = num_clbits if num_clbits > 0 else num_qubits
        mem_bytes = 0
        clifford_count = 0

        for shot_idx in range(shots):
            tab = StabilizerTableau(num_qubits)
            clbits = [0] * c_len
            measured_any = False

            for name, q_idx, params, c_idx, cond in instructions:
                if cond is not None:
                    c_bit_idx, expected_val = cond
                    if clbits[c_bit_idx] != expected_val:
                        continue
                if name == "measure":
                    measured_any = True
                    m_val = tab.measure(q_idx[0], rng=rng)
                    c_target = c_idx[0] if c_idx else q_idx[0]
                    if 0 <= c_target < c_len:
                        clbits[c_target] = m_val
                else:
                    tab.apply_gate(name, q_idx, params, rng=rng)

            if not measured_any:
                for q in range(num_qubits):
                    m_val = tab.measure(q, rng=rng)
                    if q < c_len:
                        clbits[q] = m_val

            if shot_idx == 0:
                active_tableau = tab
                mem_bytes = tab.tableau_memory_bytes
                clifford_count = tab.clifford_gate_count

            bs = "".join(str(clbits[i]) for i in range(c_len - 1, -1, -1))
            counts[bs] = counts.get(bs, 0) + 1

    latency = time.perf_counter() - t0
    probabilities = {k: v / float(shots) for k, v in counts.items()}
    ram_savings = calculate_stabilizer_ram_savings(num_qubits, mem_bytes)

    return {
        "success": True,
        "qubits": num_qubits,
        "num_qubits": num_qubits,
        "latency": latency,
        "mean_latency": latency,
        "median_latency": latency,
        "std_latency": 0.0,
        "runs_count": 1,
        "counts": counts,
        "probabilities": probabilities,
        "tableau": active_tableau,
        "tableau_memory_bytes": mem_bytes,
        "ram_savings_vs_statevector": ram_savings,
        "clifford_gate_count": clifford_count,
        "method": "stabilizer",
        "shots": shots,
        "error": None
    }
