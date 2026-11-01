import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

from src.engine.mps import MatrixProductState
from src.engine.stabilizer import StabilizerTableau
from src.engine.noise import apply_pauli_to_statevector


PAULI_MATRICES_FP64: Dict[str, np.ndarray] = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


def parse_pauli_string(
    pauli_str: str,
    expected_qubits: Optional[int] = None
) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Validate and convert an n-qubit Pauli string over {'I', 'X', 'Y', 'Z'} into
    binary symplectic vectors (x_p, z_p) over GF(2).

    Convention:
        Character index k (0 <= k < n) acts on qubit k:
            'I' -> (x=0, z=0)
            'X' -> (x=1, z=0)
            'Y' -> (x=1, z=1)
            'Z' -> (x=0, z=1)

    Args:
        pauli_str (str): Pauli string of length n (e.g., 'ZZII', 'XX', 'YY').
        expected_qubits (Optional[int]): Expected number of qubits n.

    Returns:
        Tuple[str, np.ndarray, np.ndarray]: (normalized_pauli_str, x_p, z_p)
    """
    if not isinstance(pauli_str, str):
        raise TypeError("pauli_str must be a string over {'I', 'X', 'Y', 'Z'}.")
    norm = pauli_str.strip().upper()
    if len(norm) == 0:
        raise ValueError("pauli_str cannot be empty.")
    if expected_qubits is not None and len(norm) != expected_qubits:
        raise ValueError(
            f"Pauli string length ({len(norm)}) does not match system qubit count ({expected_qubits})."
        )

    n = len(norm)
    x_p = np.zeros(n, dtype=np.uint8)
    z_p = np.zeros(n, dtype=np.uint8)

    for k, ch in enumerate(norm):
        if ch == "I":
            continue
        elif ch == "X":
            x_p[k] = 1
        elif ch == "Y":
            x_p[k] = 1
            z_p[k] = 1
        elif ch == "Z":
            z_p[k] = 1
        else:
            raise ValueError(
                f"Invalid Pauli character '{ch}' at position {k} in '{pauli_str}'. "
                f"Allowed characters: 'I', 'X', 'Y', 'Z'."
            )

    return norm, x_p, z_p


def expectation_stabilizer(tableau: StabilizerTableau, pauli_str: str) -> float:
    """
    Compute the exact expectation value <psi| P |psi> in {-1.0, 0.0, +1.0} for an n-qubit
    Pauli string P on a StabilizerTableau in O(n^2) time without statevector expansion.

    Algorithm:
        1. Encode P into binary symplectic vectors (x_P, z_P) in GF(2)^{2n}.
        2. Check symplectic inner product with all active stabilizer generators S_k (rows n..2n-1):
               omega(P, S_k) = sum_j (x_{P,j} z_{S_k,j} XOR z_{P,j} x_{S_k,j}) mod 2.
           If P anticommutes with any stabilizer generator S_k, then <psi| P |psi> = 0.0.
        3. Otherwise, +/-P belongs to the stabilizer group. Find all destabilizers D_i (rows 0..n-1)
           that anticommute with P:
               omega(P, D_i) == 1.
           Multiply the corresponding stabilizer generators S_i (rows i + n) into the scratchpad
           using the Aaronson-Gottesman rowsum phase rule. The resulting phase bit r in {0, 1}
           satisfies (-1)^r P |psi> = +|psi>, yielding <psi| P |psi> = +1.0 if r == 0 else -1.0.

    Args:
        tableau (StabilizerTableau): Active n-qubit stabilizer tableau.
        pauli_str (str): Pauli string of length n over {'I', 'X', 'Y', 'Z'}.

    Returns:
        float: Exact expectation value (+1.0, -1.0, or 0.0).
    """
    if not isinstance(tableau, StabilizerTableau):
        raise TypeError("tableau must be a StabilizerTableau instance.")

    n = tableau.num_qubits
    norm, x_p, z_p = parse_pauli_string(pauli_str, expected_qubits=n)

    # Identity operator I^{\otimes n} always has expectation +1.0
    if not np.any(x_p) and not np.any(z_p):
        return 1.0

    # Step 1: Check commutation with all stabilizer generators S_k (rows n .. 2n-1)
    stab_x = tableau.x[n:2 * n, :n]
    stab_z = tableau.z[n:2 * n, :n]
    anticomm_stab = np.sum((x_p[None, :] & stab_z) ^ (z_p[None, :] & stab_x), axis=1) & 1
    if np.any(anticomm_stab):
        return 0.0

    # Step 2: P commutes with all stabilizers, so +/- P is in the stabilizer group.
    # Identify destabilizer generators D_i (rows 0 .. n-1) that anticommute with P.
    destab_x = tableau.x[:n, :n]
    destab_z = tableau.z[:n, :n]
    anticomm_destab = np.sum((x_p[None, :] & destab_z) ^ (z_p[None, :] & destab_x), axis=1) & 1
    active_indices = np.nonzero(anticomm_destab)[0]

    if len(active_indices) == 0:
        return 1.0

    stab_rows = active_indices + n
    if len(stab_rows) == 1:
        r_bit = int(tableau.r[stab_rows[0]])
        return -1.0 if r_bit == 1 else 1.0

    # Accumulate product of stab_rows via Aaronson-Gottesman rowsum phase exponent g
    x_seq = tableau.x[stab_rows, :n]
    z_seq = tableau.z[stab_rows, :n]
    x_cum = np.cumsum(x_seq, axis=0, dtype=np.int32) & 1
    z_cum = np.cumsum(z_seq, axis=0, dtype=np.int32) & 1
    g_matrix = tableau._phase_exponent_g(
        x_seq[1:], z_seq[1:], x_cum[:-1], z_cum[:-1]
    )
    total_phase = (
        2 * int(np.sum(tableau.r[stab_rows])) + int(np.sum(g_matrix))
    ) % 4
    r_bit = 1 if total_phase == 2 else 0
    return -1.0 if r_bit == 1 else 1.0


def expectation_mps(mps: MatrixProductState, pauli_str: str) -> float:
    """
    Compute the expectation value <psi| P |psi> for an n-qubit Pauli string P on a
    MatrixProductState in O(n * chi^3) time via 1D transfer-matrix contraction,
    without ever expanding the MPS to a 2^n statevector.

    Contraction at site k = 0 .. n-1:
        E_{alpha', beta'} = sum_{alpha, beta, s, s'}
            E_{alpha, beta} * conj(A^{[k]}_{alpha, s, alpha'}) * O_{k, s, s'} * A^{[k]}_{beta, s', beta'}

    Args:
        mps (MatrixProductState): Active n-qubit MatrixProductState instance.
        pauli_str (str): Pauli string of length n over {'I', 'X', 'Y', 'Z'}.

    Returns:
        float: Real expectation value in [-1.0, +1.0].
    """
    if not isinstance(mps, MatrixProductState):
        raise TypeError("mps must be a MatrixProductState instance.")

    n = mps.num_qubits
    norm, _, _ = parse_pauli_string(pauli_str, expected_qubits=n)

    env = np.ones((1, 1), dtype=np.complex128)
    for k, ch in enumerate(norm):
        ak = mps.tensors[k].astype(np.complex128, copy=False)  # shape (d_l, 2, d_r)
        ak_conj = np.conj(ak)
        op_k = PAULI_MATRICES_FP64[ch]  # shape (2, 2)

        # Step 1: contract environment E_{a, b} with A^{[k]}_{b, s', b'} -> T1_{a, s', b'}
        t1 = np.einsum("ab,bsR->asR", env, ak)
        # Step 2: contract local Pauli operator O_{s, s'} with T1_{a, s', b'} -> T2_{a, s, b'}
        t2 = np.einsum("st,atR->asR", op_k, t1)
        # Step 3: contract conj(A^{[k]})_{a, s, a'} with T2_{a, s, b'} -> E_{a', b'}
        env = np.einsum("asL,asR->LR", ak_conj, t2)

    val = float(np.real(env[0, 0]))
    if abs(val) < 1e-12:
        return 0.0
    return float(np.clip(val, -1.0, 1.0))


def expectation_statevector(statevector: np.ndarray, pauli_str: str) -> float:
    """
    Compute the expectation value Re(<psi| P |psi>) for an n-qubit Pauli string P
    in-place on a statevector of length 2^n in O(n * 2^n) bitwise tensor operations
    without constructing a 2^n x 2^n operator matrix.

    Args:
        statevector (np.ndarray): 1D complex statevector array of length 2^n.
        pauli_str (str): Pauli string of length n over {'I', 'X', 'Y', 'Z'}.

    Returns:
        float: Real expectation value in [-1.0, +1.0].
    """
    sv = np.asarray(statevector)
    if sv.ndim != 1 or len(sv) == 0 or (len(sv) & (len(sv) - 1)) != 0:
        raise ValueError("statevector must be a 1D array whose length is a power of 2.")

    n = int(round(math.log2(len(sv))))
    norm, _, _ = parse_pauli_string(pauli_str, expected_qubits=n)

    psi_p = sv.copy()
    for q, ch in enumerate(norm):
        if ch != "I":
            apply_pauli_to_statevector(psi_p, ch, q, n)

    val = float(np.real(np.vdot(sv, psi_p)))
    if abs(val) < 1e-12:
        return 0.0
    return float(np.clip(val, -1.0, 1.0))


def evaluate_pauli_expectation(state_or_engine: Any, pauli_str: str) -> float:
    """
    Unified dispatcher computing <psi| P |psi> across StabilizerTableau,
    MatrixProductState, or Statevector (np.ndarray / Qiskit Statevector).
    """
    if isinstance(state_or_engine, StabilizerTableau):
        return expectation_stabilizer(state_or_engine, pauli_str)
    if isinstance(state_or_engine, MatrixProductState):
        return expectation_mps(state_or_engine, pauli_str)
    if hasattr(state_or_engine, "data") and not isinstance(state_or_engine, np.ndarray):
        return expectation_statevector(np.asarray(state_or_engine.data), pauli_str)
    if isinstance(state_or_engine, np.ndarray):
        return expectation_statevector(state_or_engine, pauli_str)
    raise TypeError(
        f"Unsupported state_or_engine type '{type(state_or_engine).__name__}'. "
        f"Expected StabilizerTableau, MatrixProductState, or np.ndarray."
    )


def evaluate_hamiltonian(
    state_or_engine: Any,
    hamiltonian_terms: Union[Sequence[Tuple[str, float]], Dict[str, float]]
) -> float:
    """
    Compute the total Hamiltonian expectation value:
        <psi| H |psi> = sum_k c_k <psi| P_k |psi>
    across StabilizerTableau, MatrixProductState, or Statevector without exponential
    operator matrix construction.

    Args:
        state_or_engine (Any): StabilizerTableau, MatrixProductState, or 1D statevector np.ndarray.
        hamiltonian_terms: Sequence of (pauli_str, coefficient) tuples or dict {pauli_str: coeff}.

    Returns:
        float: Real energy / Hamiltonian expectation value.
    """
    if isinstance(hamiltonian_terms, dict):
        items: Iterable[Tuple[str, float]] = hamiltonian_terms.items()
    else:
        items = hamiltonian_terms

    total_energy = 0.0
    for term_a, term_b in items:
        if isinstance(term_a, str):
            pauli_str, coeff = term_a, term_b
        else:
            coeff, pauli_str = term_a, term_b
        c = float(coeff)
        if abs(c) < 1e-15:
            continue
        exp_val = evaluate_pauli_expectation(state_or_engine, pauli_str)
        total_energy += c * exp_val

    return float(total_energy)
