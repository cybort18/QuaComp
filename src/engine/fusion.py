import time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Barrier, Measure, Reset
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

# Attempt to import high-performance native C++ extension
try:
    import quacomp_cpp
    HAS_CPP_FUSION = hasattr(quacomp_cpp, "fuse_matrices")
except (ImportError, AttributeError):
    quacomp_cpp = None
    HAS_CPP_FUSION = False


def is_cpp_fusion_available() -> bool:
    """Check whether native C++ gate fusion extension is loaded and active."""
    return HAS_CPP_FUSION


def get_fusion_backend() -> str:
    """Return human-readable identifier of the active gate fusion engine."""
    return "C++ Native Engine" if HAS_CPP_FUSION else "CPython Fallback Engine"


def py_matmul_2x2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Explicit 2x2 complex matrix multiplication.
    
    Args:
        A (np.ndarray): 2x2 complex matrix.
        B (np.ndarray): 2x2 complex matrix.
        
    Returns:
        np.ndarray: Result matrix C = A * B.
    """
    C = np.empty((2, 2), dtype=np.complex128)
    C[0, 0] = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]
    C[0, 1] = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]
    C[1, 0] = A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]
    C[1, 1] = A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]
    return C


def py_matmul_4x4(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    4x4 complex matrix multiplication.
    
    Args:
        A (np.ndarray): 4x4 complex matrix.
        B (np.ndarray): 4x4 complex matrix.
        
    Returns:
        np.ndarray: Result matrix C = A * B.
    """
    return np.matmul(A, B, dtype=np.complex128)


def py_fuse_matrices(matrices: List[np.ndarray]) -> np.ndarray:
    """
    Python fallback implementation of Single-Pass Sequential Gate Fusion:
    U_fused = U_k * ... * U_2 * U_1.
    
    Args:
        matrices (List[np.ndarray]): Sequence of unitary matrices applied in chronological order.
        
    Returns:
        np.ndarray: Fused unitary matrix.
    """
    if not matrices:
        raise ValueError("Cannot fuse an empty list of matrices.")
        
    dim = matrices[0].shape[0]
    fused = np.eye(dim, dtype=np.complex128)
    
    for mat in matrices:
        if mat.shape != (dim, dim):
            raise ValueError(f"Matrix shape mismatch: expected ({dim}, {dim}), got {mat.shape}.")
        if dim == 2:
            fused = py_matmul_2x2(mat, fused)
        elif dim == 4:
            fused = py_matmul_4x4(mat, fused)
        else:
            fused = np.matmul(mat, fused, dtype=np.complex128)
            
    return fused


def fuse_matrix_sequence(matrices: List[np.ndarray], force_backend: Optional[str] = None) -> np.ndarray:
    """
    Fuse a sequence of unitary matrices U_k * ... * U_1 using C++ extension if available,
    otherwise falling back automatically to Python / NumPy.
    
    Args:
        matrices (List[np.ndarray]): Unitaries in chronological circuit application order.
        force_backend (Optional[str]): Force 'cpp' or 'python' execution for benchmarking.
        
    Returns:
        np.ndarray: Fused unitary matrix.
    """
    if not matrices:
        raise ValueError("Cannot fuse an empty sequence of matrices.")
        
    use_cpp = (force_backend != "python") and HAS_CPP_FUSION
    
    if use_cpp and quacomp_cpp is not None:
        try:
            # Convert NumPy arrays to list of nested lists of complex numbers for pybind11
            mat_lists = [m.tolist() for m in matrices]
            res = quacomp_cpp.fuse_matrices(mat_lists)
            return np.array(res, dtype=np.complex128)
        except Exception:
            # Safe graceful fallback on any C++ invocation failure
            pass
            
    return py_fuse_matrices(matrices)


def extract_gate_matrix(operation: Any) -> Optional[np.ndarray]:
    """
    Safely extract unitary matrix representation from a Qiskit gate operation.
    
    Returns:
        np.ndarray or None if the operation is non-unitary (barrier, measure, reset).
    """
    if isinstance(operation, (Barrier, Measure, Reset)):
        return None
        
    name = getattr(operation, "name", "").lower()
    if name in ("barrier", "measure", "reset"):
        return None
        
    try:
        if hasattr(operation, "to_matrix"):
            mat = operation.to_matrix()
            return np.asarray(mat, dtype=np.complex128)
    except Exception:
        pass
        
    try:
        op = Operator(operation)
        return np.asarray(op.data, dtype=np.complex128)
    except Exception:
        return None


def fuse_circuit_single_pass(
    circuit: QuantumCircuit, 
    max_block_size: int = 8,
    force_backend: Optional[str] = None
) -> Tuple[QuantumCircuit, Dict[str, Any]]:
    """
    Single-Pass Gate Fusion Algorithm.
    
    Compresses consecutive 1-qubit operations on the same qubit and consecutive 2-qubit 
    operations on identical adjacent qubit pairs into unified unitary gates U_fused = U_k * ... * U_1.
    Preserves all barriers, measurements, resets, and classical registers.
    
    Args:
        circuit (QuantumCircuit): Original Qiskit circuit.
        max_block_size (int): Maximum consecutive gates to fuse into a single unitary (default: 8).
        force_backend (Optional[str]): Force 'cpp' or 'python' engine for testing.
        
    Returns:
        Tuple[QuantumCircuit, Dict[str, Any]]:
            - fused_circuit: Equivalent optimized QuantumCircuit with reduced gate count.
            - metrics: Fusion performance dictionary (gate reduction, elapsed time, backend).
    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("Input must be a Qiskit QuantumCircuit.")
        
    start_time = time.perf_counter()
    num_qubits = circuit.num_qubits
    fused_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=f"{circuit.name}_fused")
    
    # Track pending 1-qubit gate buffers per qubit: qubit_idx -> list of (matrix, operation)
    pending_1q: Dict[int, List[Tuple[np.ndarray, Any]]] = {q: [] for q in range(num_qubits)}
    
    # Track pending 2-qubit gate: (q_tuple, list of (matrix, operation))
    pending_2q: Optional[Tuple[Tuple[int, int], List[Tuple[np.ndarray, Any]]]] = None
    
    fused_blocks_count = 0
    
    def flush_1q(q_idx: int):
        nonlocal fused_blocks_count
        buf = pending_1q[q_idx]
        if not buf:
            return
        if len(buf) == 1:
            # Single gate: emit original gate
            _, orig_op = buf[0]
            fused_circuit.append(orig_op, [circuit.qubits[q_idx]])
        else:
            # Multiple gates: fuse into single UnitaryGate
            matrices = [m for m, _ in buf]
            fused_u = fuse_matrix_sequence(matrices, force_backend=force_backend)
            ugate = UnitaryGate(fused_u, label=f"U1q_{len(buf)}")
            fused_circuit.append(ugate, [circuit.qubits[q_idx]])
            fused_blocks_count += 1
        pending_1q[q_idx] = []

    def flush_2q():
        nonlocal pending_2q, fused_blocks_count
        if pending_2q is None:
            return
        (qa, qb), buf = pending_2q
        if len(buf) == 1:
            _, orig_op = buf[0]
            fused_circuit.append(orig_op, [circuit.qubits[qa], circuit.qubits[qb]])
        else:
            matrices = [m for m, _ in buf]
            fused_u = fuse_matrix_sequence(matrices, force_backend=force_backend)
            ugate = UnitaryGate(fused_u, label=f"U2q_{len(buf)}")
            fused_circuit.append(ugate, [circuit.qubits[qa], circuit.qubits[qb]])
            fused_blocks_count += 1
        pending_2q = None

    for inst in circuit.data:
        op = inst.operation
        q_indices = tuple(circuit.find_bit(q).index for q in inst.qubits)
        clbits = inst.clbits
        
        # Check if operation is a non-unitary barrier / measure / reset
        mat = extract_gate_matrix(op)
        if mat is None:
            # Flush affected qubits
            for q_idx in q_indices:
                flush_1q(q_idx)
            flush_2q()
            fused_circuit.append(op, inst.qubits, clbits)
            continue
            
        if len(q_indices) == 1:
            q0 = q_indices[0]
            # If a pending 2-qubit gate touches this qubit, flush it first
            if pending_2q and (q0 in pending_2q[0]):
                flush_2q()
                
            pending_1q[q0].append((mat, op))
            if len(pending_1q[q0]) >= max_block_size:
                flush_1q(q0)
                
        elif len(q_indices) == 2:
            q0, q1 = q_indices
            # Flush pending 1-qubit gates on both qubits
            flush_1q(q0)
            flush_1q(q1)
            
            # Check if this 2-qubit gate matches the currently pending 2-qubit pair
            pair = (q0, q1)
            if pending_2q is not None and pending_2q[0] == pair:
                pending_2q[1].append((mat, op))
                if len(pending_2q[1]) >= max_block_size:
                    flush_2q()
            else:
                flush_2q()
                pending_2q = (pair, [(mat, op)])
        else:
            # 3+ qubit gate: flush everything and append verbatim
            for q_idx in range(num_qubits):
                flush_1q(q_idx)
            flush_2q()
            fused_circuit.append(op, inst.qubits, clbits)
            
    # Final flush at end of circuit
    for q_idx in range(num_qubits):
        flush_1q(q_idx)
    flush_2q()
    
    elapsed = time.perf_counter() - start_time
    orig_gates = len(circuit.data)
    fused_gates = len(fused_circuit.data)
    reduction = (orig_gates - fused_gates) / orig_gates if orig_gates > 0 else 0.0
    
    active_backend = "C++ Native Engine" if ((force_backend != "python") and HAS_CPP_FUSION) else "CPython Fallback Engine"
    
    metrics = {
        "original_gate_count": orig_gates,
        "fused_gate_count": fused_gates,
        "reduction_ratio": round(reduction, 4),
        "fused_blocks_count": fused_blocks_count,
        "elapsed_seconds": round(elapsed, 6),
        "backend": active_backend
    }
    
    return fused_circuit, metrics
