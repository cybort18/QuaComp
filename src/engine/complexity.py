"""
Circuit Complexity Profiler & Auto-Solver Router.

Analyzes the structural anatomy of a quantum circuit before execution:
    1. Entanglement Topology (1D nearest-neighbor, short-range, or all-to-all long-range).
    2. Quantum Magic / Non-Clifford Gate Density (Clifford vs T / arbitrary-angle gates).

Routes execution automatically across QuaComp's three simulation paradigms:
    - 'stabilizer': O(n^2) Aaronson-Gottesman CHP Tableau (for pure Clifford circuits)
    - 'mps': O(n * chi^3) Matrix Product State Tensor Network (for high-n / area-law circuits)
    - 'statevector': O(2^n) Fused Statevector Engine (for dense non-Clifford circuits within RAM)
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import psutil
from qiskit import QuantumCircuit

from src.engine.parser import NON_GATE_INSTRUCTIONS, ParsedQASMCircuit, load_qasm, parse_qasm
from src.engine.stabilizer import NonCliffordGateError, is_clifford_angle


PURE_CLIFFORD_1Q_GATES = {
    "id", "i", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg"
}

PURE_CLIFFORD_2Q_GATES = {
    "cx", "cnot", "cz", "cy", "swap", "iswap", "dcx", "ecr"
}

KNOWN_NON_CLIFFORD_GATES = {
    "t", "tdg", "ccx", "toffoli", "cswap", "fredkin", "ch", "csx", "ccz"
}

SINGLE_ANGLE_ROTATION_GATES = {
    "rx", "ry", "rz", "p", "u1", "rxx", "ryy", "rzz"
}

CONTROLLED_PHASE_GATES = {
    "cp", "cu1", "crz", "crx", "cry"
}


def _is_gate_clifford(name: str, params: List[Any]) -> bool:
    """
    Determine whether a single quantum gate operation belongs to the Clifford group.

    Includes parameter-free Clifford gates and parameterized rotations whose angles
    are integer multiples of pi/2 (or integer multiples of pi for controlled rotations).
    """
    op_name = name.lower().strip()

    if op_name in PURE_CLIFFORD_1Q_GATES or op_name in PURE_CLIFFORD_2Q_GATES:
        return True

    if op_name in KNOWN_NON_CLIFFORD_GATES:
        return False

    if op_name in SINGLE_ANGLE_ROTATION_GATES:
        if len(params) < 1:
            return False
        ok, _ = is_clifford_angle(params[0])
        return ok

    if op_name == "u2":
        if len(params) < 2:
            return False
        ok1, _ = is_clifford_angle(params[0])
        ok2, _ = is_clifford_angle(params[1])
        return ok1 and ok2

    if op_name in ("u", "u3"):
        if len(params) < 3:
            return False
        ok1, _ = is_clifford_angle(params[0])
        ok2, _ = is_clifford_angle(params[1])
        ok3, _ = is_clifford_angle(params[2])
        return ok1 and ok2 and ok3

    if op_name in CONTROLLED_PHASE_GATES:
        if len(params) < 1:
            return False
        ok, k = is_clifford_angle(params[0])
        # Controlled rotation by k*(pi/2) is Clifford only when k is even (multiple of pi)
        if not ok:
            return False
        if op_name in ("cp", "cu1", "crz"):
            return (k % 2) == 0
        # For crx, cry: controlled-X/Y rotation is Clifford only at multiples of pi (k % 2 == 0)
        return (k % 2) == 0

    return False


def _normalize_circuit_ops(
    circuit: Any
) -> Tuple[int, List[Tuple[str, List[int], List[Any]]]]:
    """
    Convert supported circuit representations into (num_qubits, [(gate_name, q_indices, params), ...]).
    """
    if isinstance(circuit, str):
        if os.path.exists(circuit) or circuit.lower().endswith(".qasm"):
            parsed = load_qasm(circuit)
        else:
            parsed = parse_qasm(circuit)
        return _normalize_circuit_ops(parsed)

    if isinstance(circuit, ParsedQASMCircuit):
        qmap = circuit.qubit_map
        ops: List[Tuple[str, List[int], List[Any]]] = []
        for inst in circuit.instructions:
            q_idx = [qmap[qa] for qa in inst.qargs]
            ops.append((inst.name.lower(), q_idx, list(inst.params)))
        return circuit.num_qubits, ops

    if isinstance(circuit, QuantumCircuit):
        num_qubits = circuit.num_qubits
        ops = []

        def _extract_qc(qc_obj: QuantumCircuit, q_remap: List[int]) -> None:
            for inst in qc_obj.data:
                op = inst.operation
                name = op.name.lower()
                q_idx = [q_remap[qc_obj.find_bit(q).index] for q in inst.qubits]
                params = list(getattr(op, "params", []))

                known_all = (
                    PURE_CLIFFORD_1Q_GATES
                    | PURE_CLIFFORD_2Q_GATES
                    | KNOWN_NON_CLIFFORD_GATES
                    | SINGLE_ANGLE_ROTATION_GATES
                    | CONTROLLED_PHASE_GATES
                    | {"u2", "u", "u3"}
                    | set(NON_GATE_INSTRUCTIONS)
                )
                if name not in known_all and getattr(op, "definition", None) is not None:
                    _extract_qc(op.definition, q_idx)
                else:
                    ops.append((name, q_idx, params))

        _extract_qc(circuit, list(range(num_qubits)))
        return num_qubits, ops

    if isinstance(circuit, (list, tuple)):
        ops = []
        max_q = -1
        for item in circuit:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                raise ValueError(
                    "Instruction tuple must have at least (gate_name, qubit_indices)."
                )
            name = str(item[0]).lower()
            q_idx = [int(q) for q in item[1]]
            params = list(item[2]) if len(item) >= 3 and item[2] is not None else []
            if q_idx:
                max_q = max(max_q, max(q_idx))
            ops.append((name, q_idx, params))
        return max(1, max_q + 1), ops

    raise TypeError(
        "circuit must be a QuantumCircuit, ParsedQASMCircuit, QASM path/string, or list of gate tuples."
    )


def analyze_circuit_complexity(circuit: Any) -> Dict[str, Any]:
    """
    Perform static structural analysis of a quantum circuit to quantify entanglement
    interaction range and non-Clifford quantum magic density.

    Args:
        circuit (Any): QuantumCircuit, ParsedQASMCircuit, .qasm path/string, or list of gate tuples.

    Returns:
        Dict[str, Any]: Complexity profile dictionary containing:
            - num_qubits (int)
            - total_gates (int)
            - one_qubit_gates (int)
            - two_qubit_gates (int)
            - three_qubit_gates (int)
            - clifford_gates (int)
            - non_clifford_gates (int)
            - magic_density (float)
            - is_pure_clifford (bool)
            - max_interaction_distance (int)
            - mean_interaction_distance (float)
            - entanglement_topology (str)
    """
    num_qubits, ops = _normalize_circuit_ops(circuit)

    total_gates = 0
    one_qubit_gates = 0
    two_qubit_gates = 0
    three_qubit_gates = 0
    clifford_gates = 0
    non_clifford_gates = 0
    interaction_distances: List[int] = []

    for name, q_idx, params in ops:
        if name in NON_GATE_INSTRUCTIONS:
            continue

        total_gates += 1
        arity = len(q_idx)
        if arity == 1:
            one_qubit_gates += 1
        elif arity == 2:
            two_qubit_gates += 1
            dist = abs(q_idx[0] - q_idx[1])
            interaction_distances.append(dist)
        elif arity >= 3:
            three_qubit_gates += 1
            dist = max(q_idx) - min(q_idx)
            interaction_distances.append(dist)

        if _is_gate_clifford(name, params):
            clifford_gates += 1
        else:
            non_clifford_gates += 1

    magic_density = float(non_clifford_gates) / float(max(1, total_gates))
    is_pure_clifford = (non_clifford_gates == 0)

    if interaction_distances:
        max_interaction_distance = int(max(interaction_distances))
        mean_interaction_distance = float(sum(interaction_distances)) / float(len(interaction_distances))
    else:
        max_interaction_distance = 0
        mean_interaction_distance = 0.0

    if max_interaction_distance == 0:
        entanglement_topology = "unentangled"
    elif max_interaction_distance == 1:
        entanglement_topology = "1d_nearest_neighbor"
    elif max_interaction_distance <= 3:
        entanglement_topology = "short_range"
    else:
        entanglement_topology = "all_to_all_long_range"

    return {
        "num_qubits": num_qubits,
        "total_gates": total_gates,
        "one_qubit_gates": one_qubit_gates,
        "two_qubit_gates": two_qubit_gates,
        "three_qubit_gates": three_qubit_gates,
        "clifford_gates": clifford_gates,
        "non_clifford_gates": non_clifford_gates,
        "magic_density": round(magic_density, 6),
        "is_pure_clifford": is_pure_clifford,
        "max_interaction_distance": max_interaction_distance,
        "mean_interaction_distance": round(mean_interaction_distance, 4),
        "entanglement_topology": entanglement_topology
    }


def select_optimal_solver(
    circuit: Any,
    requested_method: str = "auto",
    available_ram_bytes: Optional[int] = None
) -> Dict[str, Any]:
    """
    Select the optimal simulation paradigm ('stabilizer', 'mps', or 'statevector')
    based on static circuit complexity analysis and hardware memory constraints.

    Args:
        circuit (Any): Circuit to analyze (or precomputed complexity profile dict).
        requested_method (str): 'auto', 'statevector', 'mps', or 'stabilizer'.
        available_ram_bytes (Optional[int]): Available physical RAM in bytes.

    Returns:
        Dict[str, Any]: Routing decision containing:
            - selected_solver (str): 'stabilizer', 'mps', or 'statevector'
            - requested_method (str)
            - reason (str)
            - warning (Optional[str])
            - complexity_profile (Dict[str, Any])

    Raises:
        NonCliffordGateError: If requested_method is 'stabilizer' on a circuit with non-Clifford gates.
        ValueError: If requested_method is unrecognized.
    """
    if isinstance(circuit, dict) and "is_pure_clifford" in circuit and "num_qubits" in circuit:
        profile = circuit
    else:
        profile = analyze_circuit_complexity(circuit)

    if available_ram_bytes is None:
        available_ram_bytes = int(psutil.virtual_memory().available)

    req = str(requested_method).lower().strip()
    if req == "matrix_product_state":
        req = "mps"

    if req not in ("auto", "statevector", "mps", "stabilizer"):
        raise ValueError(
            f"Invalid simulation method '{requested_method}'. Must be one of: "
            f"'auto', 'statevector', 'mps', 'stabilizer'."
        )

    n = profile["num_qubits"]
    is_pure_clifford = profile["is_pure_clifford"]
    two_q = profile["two_qubit_gates"]
    max_dist = profile["max_interaction_distance"]
    sv_ram_bytes = (1 << n) * 16 if n < 1000 else float("inf")

    # Explicit user solver selection
    if req != "auto":
        warning_msg: Optional[str] = None
        if req == "stabilizer" and not is_pure_clifford:
            raise NonCliffordGateError(
                f"Cannot execute circuit with 'stabilizer' solver: detected "
                f"{profile['non_clifford_gates']} non-Clifford gate(s) "
                f"(magic_density={profile['magic_density']:.4f}). "
                f"Use 'statevector', 'mps', or 'auto'."
            )
        if req == "statevector" and sv_ram_bytes > 0.8 * available_ram_bytes:
            warning_msg = (
                f"Statevector simulation for {n} qubits requires ~{sv_ram_bytes / (1024**3):.2f} GB, "
                f"which exceeds 80% of available RAM ({available_ram_bytes / (1024**3):.2f} GB)."
            )
        elif req == "mps" and is_pure_clifford and (n > 20 or two_q > 0):
            warning_msg = (
                "Circuit is pure Clifford; 'stabilizer' tableau engine would provide exact O(n^2) execution."
            )

        return {
            "selected_solver": req,
            "requested_method": requested_method,
            "reason": f"Explicitly requested '{req}' simulation engine.",
            "routing_reason": f"Explicitly requested '{req}' simulation engine.",
            "warning": warning_msg,
            "complexity_profile": profile
        }

    # Auto-Solver Routing Rules
    # Rule 1: Gottesman-Knill Fast-Path
    if is_pure_clifford and (n > 20 or two_q > 0 or profile["total_gates"] > 0):
        reason_str = (
            "Pure Clifford circuit detected (0 non-Clifford gates); "
            "routing to O(n^2) Aaronson-Gottesman (Gottesman-Knill) Stabilizer Tableau."
        )
        return {
            "selected_solver": "stabilizer",
            "requested_method": "auto",
            "reason": reason_str,
            "routing_reason": reason_str,
            "warning": None,
            "complexity_profile": profile
        }

    # Rule 2: Area-Law Tensor Network Path
    if not is_pure_clifford and (
        n > 26
        or sv_ram_bytes > 0.8 * available_ram_bytes
        or (n >= 18 and max_dist <= 3)
    ):
        reason_str = (
            "High qubit count or 1D/short-range entanglement topology; "
            "routing to O(n*chi^3) Matrix Product State."
        )
        return {
            "selected_solver": "mps",
            "requested_method": "auto",
            "reason": reason_str,
            "routing_reason": reason_str,
            "warning": None,
            "complexity_profile": profile
        }

    # Rule 3: Dense Statevector Path
    reason_str = (
        "Dense non-Clifford circuit within classical RAM limits; "
        "routing to O(2^n) Fused Statevector Engine."
    )
    return {
        "selected_solver": "statevector",
        "requested_method": "auto",
        "reason": reason_str,
        "routing_reason": reason_str,
        "warning": None,
        "complexity_profile": profile
    }
