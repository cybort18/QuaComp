import os
import re
import ast
import math
import operator
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


class QASMParseError(Exception):
    """Exception raised for syntax, semantic, or boundary errors in OpenQASM parsing."""
    def __init__(self, message: str, line_num: Optional[int] = None, statement: Optional[str] = None):
        self.message = message
        self.line_num = line_num
        self.statement = statement
        prefix = f"Line {line_num}: " if line_num is not None else ""
        stmt_suffix = f" in '{statement}'" if statement else ""
        super().__init__(f"{prefix}{message}{stmt_suffix}")


# Safe mathematical AST evaluator for parameter expressions (e.g. pi/2, -pi/4, 2*pi, 1.57)
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_CONSTANTS = {
    "pi": math.pi,
    "PI": math.pi,
    "e": math.e,
    "E": math.e,
}

SAFE_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "ln": math.log,
    "log": math.log,
    "sqrt": math.sqrt,
}


def evaluate_parameter_expression(expr_str: str) -> float:
    """
    Safely evaluate a mathematical parameter expression for a quantum gate.
    
    Supports numbers, basic arithmetic (+, -, *, /, **), constants (pi, e),
    and safe standard functions (sin, cos, tan, exp, ln, sqrt).
    
    Args:
        expr_str (str): Expression string to evaluate.
        
    Returns:
        float: Evaluated numerical value.
        
    Raises:
        QASMParseError: If expression contains syntax errors or unsafe operations.
    """
    expr_clean = expr_str.strip()
    if not expr_clean:
        raise QASMParseError("Empty parameter expression encountered.")
        
    # Replace common caret power operator if used
    expr_clean = expr_clean.replace("^", "**")
    
    try:
        parsed_ast = ast.parse(expr_clean, mode="eval")
    except SyntaxError as se:
        raise QASMParseError(f"Syntax error in parameter expression '{expr_str}': {se.msg}")
        
    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise QASMParseError(f"Unsupported constant type '{type(node.value)}' in parameter expression.")
        elif isinstance(node, ast.Name):
            if node.id in SAFE_CONSTANTS:
                return float(SAFE_CONSTANTS[node.id])
            raise QASMParseError(f"Unknown variable or constant '{node.id}' in parameter expression.")
        elif isinstance(node, ast.UnaryOp):
            op_func = SAFE_OPERATORS.get(type(node.op))
            if op_func is None:
                raise QASMParseError(f"Unsupported unary operator '{type(node.op)}' in parameter expression.")
            return float(op_func(_eval_node(node.operand)))
        elif isinstance(node, ast.BinOp):
            op_func = SAFE_OPERATORS.get(type(node.op))
            if op_func is None:
                raise QASMParseError(f"Unsupported binary operator '{type(node.op)}' in parameter expression.")
            left_val = _eval_node(node.left)
            right_val = _eval_node(node.right)
            try:
                return float(op_func(left_val, right_val))
            except ZeroDivisionError:
                raise QASMParseError(f"Division by zero in parameter expression '{expr_str}'.")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
                func = SAFE_FUNCTIONS[node.func.id]
                args = [_eval_node(arg) for arg in node.args]
                if len(args) != 1:
                    raise QASMParseError(f"Function '{node.func.id}' requires exactly 1 argument, got {len(args)}.")
                return float(func(args[0]))
            raise QASMParseError(f"Unsupported function call '{ast.dump(node.func)}' in parameter expression.")
        else:
            raise QASMParseError(f"Unsupported AST node '{type(node)}' in parameter expression '{expr_str}'.")

    return _eval_node(parsed_ast.body)


@dataclass
class QASMInstruction:
    """Represents a single parsed OpenQASM 2.0 quantum gate or operation."""
    name: str
    params: List[float] = field(default_factory=list)
    qargs: List[Tuple[str, int]] = field(default_factory=list)
    cargs: List[Tuple[str, int]] = field(default_factory=list)
    line_num: Optional[int] = None
    raw_statement: str = ""


# Catalog of standard OpenQASM 2.0 gates with expected (param_count, qubit_count)
# None means variable or custom.
STANDARD_GATES_SPECS: Dict[str, Tuple[Optional[int], int]] = {
    # 1-qubit non-parameterized
    "x": (0, 1),
    "y": (0, 1),
    "z": (0, 1),
    "h": (0, 1),
    "s": (0, 1),
    "sdg": (0, 1),
    "t": (0, 1),
    "tdg": (0, 1),
    "id": (0, 1),
    "iden": (0, 1),
    "sx": (0, 1),
    "sxdg": (0, 1),
    # 1-qubit parameterized
    "rx": (1, 1),
    "ry": (1, 1),
    "rz": (1, 1),
    "u1": (1, 1),
    "p": (1, 1),
    "u0": (1, 1),
    "u2": (2, 1),
    "u3": (3, 1),
    "u": (3, 1),
    # 2-qubit non-parameterized
    "cx": (0, 2),
    "cnot": (0, 2),
    "cy": (0, 2),
    "cz": (0, 2),
    "ch": (0, 2),
    "swap": (0, 2),
    "iswap": (0, 2),
    # 2-qubit parameterized
    "crz": (1, 2),
    "cu1": (1, 2),
    "cp": (1, 2),
    "cu3": (3, 2),
    "rxx": (1, 2),
    "ryy": (1, 2),
    "rzz": (1, 2),
    # 3-qubit
    "ccx": (0, 3),
    "toffoli": (0, 3),
    "cswap": (0, 3),
    "fredkin": (0, 3),
}


class ParsedQASMCircuit:
    """
    Internal circuit data structure representing a parsed OpenQASM 2.0 program.
    
    Provides queryable gate metrics, AST instructions, and conversion to Qiskit QuantumCircuit.
    """
    def __init__(self, name: str = "qasm_circuit"):
        self.name = name
        self.qregs: Dict[str, int] = {}
        self.cregs: Dict[str, int] = {}
        self.instructions: List[QASMInstruction] = []
        self.custom_gates: Dict[str, Dict[str, Any]] = {}
        
    @property
    def num_qubits(self) -> int:
        """Total number of quantum bits across all declared quantum registers."""
        return sum(self.qregs.values())

    @property
    def num_clbits(self) -> int:
        """Total number of classical bits across all declared classical registers."""
        return sum(self.cregs.values())

    @property
    def qubit_map(self) -> Dict[Tuple[str, int], int]:
        """Maps (register_name, reg_index) to contiguous global qubit index (0-indexed)."""
        mapping: Dict[Tuple[str, int], int] = {}
        idx = 0
        for reg_name, size in self.qregs.items():
            for i in range(size):
                mapping[(reg_name, i)] = idx
                idx += 1
        return mapping

    @property
    def clbit_map(self) -> Dict[Tuple[str, int], int]:
        """Maps (register_name, reg_index) to contiguous global classical bit index (0-indexed)."""
        mapping: Dict[Tuple[str, int], int] = {}
        idx = 0
        for reg_name, size in self.cregs.items():
            for i in range(size):
                mapping[(reg_name, i)] = idx
                idx += 1
        return mapping

    @property
    def total_gates(self) -> int:
        """Total count of quantum gates (excluding barrier, reset, and measurement)."""
        return sum(1 for inst in self.instructions if inst.name not in ("barrier", "measure", "reset"))

    @property
    def one_qubit_gates(self) -> int:
        """Count of single-qubit quantum gates."""
        return sum(1 for inst in self.instructions if inst.name not in ("barrier", "measure", "reset") and len(inst.qargs) == 1)

    @property
    def two_qubit_gates(self) -> int:
        """Count of two-qubit quantum gates."""
        return sum(1 for inst in self.instructions if inst.name not in ("barrier", "measure", "reset") and len(inst.qargs) == 2)

    @property
    def multi_qubit_gates(self) -> int:
        """Count of multi-qubit (>= 3 qubits) quantum gates."""
        return sum(1 for inst in self.instructions if inst.name not in ("barrier", "measure", "reset") and len(inst.qargs) >= 3)

    @property
    def depth(self) -> int:
        """Compute circuit critical path depth."""
        if not self.instructions or self.num_qubits == 0:
            return 0
        q_depths: Dict[Tuple[str, int], int] = {k: 0 for k in self.qubit_map.keys()}
        for inst in self.instructions:
            if inst.name == "barrier":
                continue
            if inst.qargs:
                max_d = max(q_depths.get(qa, 0) for qa in inst.qargs)
                new_d = max_d + 1
                for qa in inst.qargs:
                    q_depths[qa] = new_d
        return max(q_depths.values()) if q_depths else 0

    def count_ops(self) -> Dict[str, int]:
        """Return a dictionary of gate counts per operation name."""
        counts: Dict[str, int] = {}
        for inst in self.instructions:
            counts[inst.name] = counts.get(inst.name, 0) + 1
        return counts

    def validate_memory_limits(self, max_allowed_ram_gb: Optional[float] = None) -> bool:
        """
        Validate whether this circuit can be simulated within memory limits.
        
        Formula: Statevector RAM = 2^n * 16 bytes.
        """
        if self.num_qubits > 35:
            return False
        req_bytes = (2 ** self.num_qubits) * 16.0
        req_gb = req_bytes / (1024.0 ** 3)
        if max_allowed_ram_gb is not None:
            return req_gb <= max_allowed_ram_gb
        return True

    def to_quantum_circuit(self) -> QuantumCircuit:
        """
        Convert this parsed QASM representation to a Qiskit QuantumCircuit instance.
        
        Returns:
            QuantumCircuit: Equivalent executable Qiskit QuantumCircuit.
        """
        qreg_objs = {name: QuantumRegister(size, name) for name, size in self.qregs.items()}
        creg_objs = {name: ClassicalRegister(size, name) for name, size in self.cregs.items()}
        
        qc = QuantumCircuit(*qreg_objs.values(), *creg_objs.values(), name=self.name)
        
        for inst in self.instructions:
            name = inst.name
            params = inst.params
            q_targets = [qreg_objs[r][i] for r, i in inst.qargs]
            c_targets = [creg_objs[r][i] for r, i in inst.cargs]
            
            if name in ("id", "iden"):
                qc.id(q_targets[0])
            elif name == "x":
                qc.x(q_targets[0])
            elif name == "y":
                qc.y(q_targets[0])
            elif name == "z":
                qc.z(q_targets[0])
            elif name == "h":
                qc.h(q_targets[0])
            elif name == "s":
                qc.s(q_targets[0])
            elif name == "sdg":
                qc.sdg(q_targets[0])
            elif name == "t":
                qc.t(q_targets[0])
            elif name == "tdg":
                qc.tdg(q_targets[0])
            elif name == "sx":
                qc.sx(q_targets[0])
            elif name == "sxdg":
                qc.sxdg(q_targets[0])
            elif name == "rx":
                qc.rx(params[0], q_targets[0])
            elif name == "ry":
                qc.ry(params[0], q_targets[0])
            elif name == "rz":
                qc.rz(params[0], q_targets[0])
            elif name in ("u1", "p"):
                qc.p(params[0], q_targets[0])
            elif name == "u2":
                qc.u(math.pi / 2.0, params[0], params[1], q_targets[0])
            elif name in ("u3", "u"):
                qc.u(params[0], params[1], params[2], q_targets[0])
            elif name == "u0":
                qc.id(q_targets[0])
            elif name in ("cx", "cnot"):
                qc.cx(q_targets[0], q_targets[1])
            elif name == "cz":
                qc.cz(q_targets[0], q_targets[1])
            elif name == "cy":
                qc.cy(q_targets[0], q_targets[1])
            elif name == "ch":
                qc.ch(q_targets[0], q_targets[1])
            elif name == "swap":
                qc.swap(q_targets[0], q_targets[1])
            elif name == "iswap":
                qc.iswap(q_targets[0], q_targets[1])
            elif name == "crz":
                qc.crz(params[0], q_targets[0], q_targets[1])
            elif name in ("cu1", "cp"):
                qc.cp(params[0], q_targets[0], q_targets[1])
            elif name == "cu3":
                qc.cu(params[0], params[1], params[2], 0.0, q_targets[0], q_targets[1])
            elif name == "rxx":
                qc.rxx(params[0], q_targets[0], q_targets[1])
            elif name == "ryy":
                qc.ryy(params[0], q_targets[0], q_targets[1])
            elif name == "rzz":
                qc.rzz(params[0], q_targets[0], q_targets[1])
            elif name in ("ccx", "toffoli"):
                qc.ccx(q_targets[0], q_targets[1], q_targets[2])
            elif name in ("cswap", "fredkin"):
                qc.cswap(q_targets[0], q_targets[1], q_targets[2])
            elif name == "measure":
                qc.measure(q_targets[0], c_targets[0])
            elif name == "reset":
                qc.reset(q_targets[0])
            elif name == "barrier":
                qc.barrier(*q_targets)
            else:
                # If custom gate is registered or mapped
                raise QASMParseError(f"Unsupported gate mapping to Qiskit: '{name}'")
                
        return qc

    @classmethod
    def from_qiskit(cls, circuit: QuantumCircuit) -> "ParsedQASMCircuit":
        """
        Construct a ParsedQASMCircuit from an existing Qiskit QuantumCircuit.
        
        Args:
            circuit (QuantumCircuit): Source Qiskit QuantumCircuit.
            
        Returns:
            ParsedQASMCircuit: Populated parsed circuit representation.
        """
        parsed = cls(name=circuit.name or "qiskit_circuit")
        
        for qreg in circuit.qregs:
            parsed.qregs[qreg.name] = qreg.size
        for creg in circuit.cregs:
            parsed.cregs[creg.name] = creg.size
            
        # If no explicit registers exist, create default
        if not parsed.qregs and circuit.num_qubits > 0:
            parsed.qregs["q"] = circuit.num_qubits
        if not parsed.cregs and circuit.num_clbits > 0:
            parsed.cregs["c"] = circuit.num_clbits
            
        # Map instructions
        for item in circuit.data:
            op = item.operation
            name = op.name
            params = [float(p) for p in op.params if isinstance(p, (int, float))]
            
            qargs: List[Tuple[str, int]] = []
            for qb in item.qubits:
                # Resolve register and index
                qreg_name = "q"
                q_idx = circuit.find_bit(qb).index
                # Search matching register
                curr_idx = q_idx
                for qr_name, qr_size in parsed.qregs.items():
                    if curr_idx < qr_size:
                        qreg_name = qr_name
                        break
                    curr_idx -= qr_size
                qargs.append((qreg_name, curr_idx))
                
            cargs: List[Tuple[str, int]] = []
            for cb in item.clbits:
                creg_name = "c"
                c_idx = circuit.find_bit(cb).index
                curr_idx = c_idx
                for cr_name, cr_size in parsed.cregs.items():
                    if curr_idx < cr_size:
                        creg_name = cr_name
                        break
                    curr_idx -= cr_size
                cargs.append((creg_name, curr_idx))
                
            parsed.instructions.append(
                QASMInstruction(name=name, params=params, qargs=qargs, cargs=cargs)
            )
            
        return parsed

    def get_summary(self) -> Dict[str, Any]:
        """Return a structured summary of the circuit properties."""
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "num_clbits": self.num_clbits,
            "total_gates": self.total_gates,
            "one_qubit_gates": self.one_qubit_gates,
            "two_qubit_gates": self.two_qubit_gates,
            "multi_qubit_gates": self.multi_qubit_gates,
            "depth": self.depth,
            "ops": self.count_ops(),
            "qregs": dict(self.qregs),
            "cregs": dict(self.cregs)
        }


# ==============================================================================
# OpenQASM 2.0 Lexer & Parser Implementation
# ==============================================================================

def _split_into_statements(qasm_text: str) -> List[Tuple[str, int]]:
    """
    Extract individual semicolon-delimited statements and compound gate blocks
    while stripping comments and tracking line numbers.
    """
    statements: List[Tuple[str, int]] = []
    current_tokens: List[str] = []
    start_line = 1
    current_line = 1
    brace_depth = 0
    in_string = False
    
    i = 0
    n = len(qasm_text)
    
    while i < n:
        ch = qasm_text[i]
        
        if ch == "\n":
            current_line += 1
            if not current_tokens:
                start_line = current_line
            current_tokens.append(ch)
            i += 1
            continue
            
        # Check single-line comment //
        if ch == "/" and i + 1 < n and qasm_text[i + 1] == "/":
            # Skip until end of line
            while i < n and qasm_text[i] != "\n":
                i += 1
            continue
            
        if ch == '"':
            in_string = not in_string
            current_tokens.append(ch)
            i += 1
            continue
            
        if in_string:
            current_tokens.append(ch)
            i += 1
            continue
            
        if ch == "{":
            brace_depth += 1
            current_tokens.append(ch)
            i += 1
            continue
            
        if ch == "}":
            brace_depth -= 1
            current_tokens.append(ch)
            i += 1
            if brace_depth == 0:
                stmt = "".join(current_tokens).strip()
                if stmt:
                    statements.append((stmt, start_line))
                current_tokens = []
                start_line = current_line
            continue
            
        if ch == ";" and brace_depth == 0:
            stmt = "".join(current_tokens).strip()
            if stmt:
                statements.append((stmt, start_line))
            current_tokens = []
            start_line = current_line
            i += 1
            continue
            
        if not current_tokens and not ch.isspace():
            start_line = current_line
            
        current_tokens.append(ch)
        i += 1
        
    rem = "".join(current_tokens).strip()
    if rem:
        if brace_depth > 0:
            raise QASMParseError("Unclosed curly brace '{' encountered at end of file.", line_num=start_line)
        statements.append((rem, start_line))
        
    return statements


def parse_qasm(qasm_text: str, name: Optional[str] = None) -> ParsedQASMCircuit:
    """
    Parse an OpenQASM 2.0 program string into a structured ParsedQASMCircuit object.
    
    Args:
        qasm_text (str): OpenQASM 2.0 source code string.
        name (Optional[str]): Optional circuit name (defaults to 'qasm_circuit').
        
    Returns:
        ParsedQASMCircuit: Parsed and validated circuit instance.
        
    Raises:
        QASMParseError: If syntax, type, boundary, or undefined gate errors are found.
    """
    if not isinstance(qasm_text, str):
        raise TypeError("qasm_text must be a string.")
        
    circuit = ParsedQASMCircuit(name=name or "qasm_circuit")
    raw_statements = _split_into_statements(qasm_text)
    
    # Custom gate macro registry: name -> {"params": [...], "args": [...], "body": [...]}
    custom_gate_macros: Dict[str, Dict[str, Any]] = {}
    
    for stmt_str, line_num in raw_statements:
        stmt = " ".join(stmt_str.split()) # normalize whitespace
        if not stmt:
            continue
            
        # 1. Ignore headers & includes
        if stmt.startswith("OPENQASM"):
            m = re.match(r"^OPENQASM\s+([0-9.]+)\s*$", stmt)
            if not m:
                raise QASMParseError("Malformed OPENQASM version declaration.", line_num, stmt)
            ver = m.group(1)
            if not ver.startswith("2."):
                raise QASMParseError(f"Unsupported OPENQASM version '{ver}'. Only 2.0 is supported.", line_num, stmt)
            continue
            
        if stmt.startswith("include"):
            # Check include "filename"
            m = re.match(r'^include\s+"([^"]+)"\s*$', stmt)
            if not m:
                raise QASMParseError("Malformed include statement.", line_num, stmt)
            continue
            
        if stmt.startswith("opaque"):
            # Skip opaque declarations
            continue
            
        # 2. Register declarations: qreg and creg
        if stmt.startswith("qreg ") or stmt.startswith("creg "):
            m = re.match(r"^(qreg|creg)\s+([a-zA-Z_][a-zA-Z0-9_]*)\[\s*(-?\d+)\s*\]$", stmt)
            if not m:
                raise QASMParseError("Malformed register declaration. Format: (qreg|creg) <name>[<size>];", line_num, stmt)
            reg_type, reg_name, size_str = m.group(1), m.group(2), m.group(3)
            size = int(size_str)
            if size <= 0:
                raise QASMParseError(f"Register size must be positive integer >= 1, got {size}.", line_num, stmt)
            if reg_name in circuit.qregs or reg_name in circuit.cregs:
                raise QASMParseError(f"Duplicate register identifier '{reg_name}' already declared.", line_num, stmt)
            if reg_type == "qreg":
                circuit.qregs[reg_name] = size
            else:
                circuit.cregs[reg_name] = size
            continue
            
        # 3. Custom gate macro definition: gate name(params) args { body }
        if stmt.startswith("gate ") or re.match(r"^gate\s+", stmt):
            m = re.match(r"^gate\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*\(([^)]*)\))?\s+([^\{]+)\{\s*(.*?)\s*\}$", stmt, re.DOTALL)
            if not m:
                raise QASMParseError("Malformed custom gate definition.", line_num, stmt)
            g_name, g_params_str, g_args_str, g_body_str = m.group(1), m.group(2), m.group(3), m.group(4)
            macro_params = [p.strip() for p in g_params_str.split(",") if p.strip()] if g_params_str else []
            macro_args = [a.strip() for a in g_args_str.split(",") if a.strip()]
            
            # Split body statements
            body_statements = [s.strip() for s in g_body_str.split(";") if s.strip()]
            custom_gate_macros[g_name] = {
                "params": macro_params,
                "args": macro_args,
                "body": body_statements,
                "line": line_num
            }
            circuit.custom_gates[g_name] = custom_gate_macros[g_name]
            continue
            
        # 4. Measurement: measure q -> c;
        if stmt.startswith("measure ") or re.match(r"^measure\s+", stmt):
            m = re.match(r"^measure\s+([a-zA-Z0-9_\[\]]+)\s*->\s*([a-zA-Z0-9_\[\]]+)$", stmt)
            if not m:
                raise QASMParseError("Malformed measure statement. Format: measure <qarg> -> <carg>;", line_num, stmt)
            q_token, c_token = m.group(1).strip(), m.group(2).strip()
            
            q_pairs = _resolve_qubit_or_clbit_arg(q_token, circuit.qregs, "quantum", line_num, stmt)
            c_pairs = _resolve_qubit_or_clbit_arg(c_token, circuit.cregs, "classical", line_num, stmt)
            
            if len(q_pairs) != len(c_pairs):
                raise QASMParseError(f"Measurement register size mismatch: {len(q_pairs)} qubits vs {len(c_pairs)} clbits.", line_num, stmt)
                
            for (qr, qi), (cr, ci) in zip(q_pairs, c_pairs):
                circuit.instructions.append(
                    QASMInstruction(name="measure", params=[], qargs=[(qr, qi)], cargs=[(cr, ci)], line_num=line_num, raw_statement=stmt)
                )
            continue
            
        # 5. Reset statement: reset q[i];
        if stmt.startswith("reset ") or re.match(r"^reset\s+", stmt):
            m = re.match(r"^reset\s+([a-zA-Z0-9_\[\]]+)$", stmt)
            if not m:
                raise QASMParseError("Malformed reset statement. Format: reset <qarg>;", line_num, stmt)
            q_token = m.group(1).strip()
            q_pairs = _resolve_qubit_or_clbit_arg(q_token, circuit.qregs, "quantum", line_num, stmt)
            for qr, qi in q_pairs:
                circuit.instructions.append(
                    QASMInstruction(name="reset", params=[], qargs=[(qr, qi)], line_num=line_num, raw_statement=stmt)
                )
            continue
            
        # 6. Barrier statement: barrier q[0], q[1];
        if stmt.startswith("barrier") or re.match(r"^barrier(?:\s+|$)", stmt):
            args_part = re.sub(r"^barrier\s*", "", stmt).strip()
            barrier_qargs: List[Tuple[str, int]] = []
            if args_part:
                for arg_item in args_part.split(","):
                    arg_item = arg_item.strip()
                    if arg_item:
                        barrier_qargs.extend(_resolve_qubit_or_clbit_arg(arg_item, circuit.qregs, "quantum", line_num, stmt))
            else:
                # Barrier all active qubits
                for qr_name, qr_size in circuit.qregs.items():
                    for qi in range(qr_size):
                        barrier_qargs.append((qr_name, qi))
            circuit.instructions.append(
                QASMInstruction(name="barrier", params=[], qargs=barrier_qargs, line_num=line_num, raw_statement=stmt)
            )
            continue
            
        # 7. Quantum Gate statement: gate_name(params) qargs;
        _parse_gate_statement(stmt, line_num, circuit, custom_gate_macros)
        
    return circuit


def _resolve_qubit_or_clbit_arg(
    token: str, 
    registers: Dict[str, int], 
    reg_kind: str, 
    line_num: int, 
    stmt: str
) -> List[Tuple[str, int]]:
    """Resolve argument into explicit list of (reg_name, index) tuples."""
    token = token.strip()
    m_indexed = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\[\s*(-?\d+)\s*\]$", token)
    if m_indexed:
        reg_name, idx_str = m_indexed.group(1), m_indexed.group(2)
        idx = int(idx_str)
        if reg_name not in registers:
            raise QASMParseError(f"Undeclared {reg_kind} register '{reg_name}'.", line_num, stmt)
        max_size = registers[reg_name]
        if idx < 0 or idx >= max_size:
            raise QASMParseError(f"{reg_kind.capitalize()} index {idx} out of bounds for register '{reg_name}' of size {max_size}.", line_num, stmt)
        return [(reg_name, idx)]
    else:
        # Whole register broadcasting
        reg_name = token
        if reg_name not in registers:
            raise QASMParseError(f"Undeclared {reg_kind} register '{reg_name}'.", line_num, stmt)
        return [(reg_name, i) for i in range(registers[reg_name])]


def _parse_gate_statement(
    stmt: str, 
    line_num: int, 
    circuit: ParsedQASMCircuit, 
    custom_gate_macros: Dict[str, Dict[str, Any]]
) -> None:
    """Parse a single gate application statement into instructions, expanding macros if needed."""
    m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*\(([^)]*)\))?\s+(.+)$", stmt)
    if not m:
        raise QASMParseError(f"Invalid statement or unrecognized syntax: '{stmt}'.", line_num, stmt)
        
    gate_name = m.group(1)
    params_str = m.group(2)
    qargs_str = m.group(3)
    
    # 1. Parse parameters
    eval_params: List[float] = []
    if params_str is not None:
        raw_p_list = [p.strip() for p in params_str.split(",") if p.strip()]
        for p_expr in raw_p_list:
            val = evaluate_parameter_expression(p_expr)
            eval_params.append(val)
            
    # 2. Parse qubit arguments
    raw_qargs = [q.strip() for q in qargs_str.split(",") if q.strip()]
    if not raw_qargs:
        raise QASMParseError(f"Gate '{gate_name}' missing target qubit arguments.", line_num, stmt)
        
    resolved_arg_lists = [
        _resolve_qubit_or_clbit_arg(arg, circuit.qregs, "quantum", line_num, stmt)
        for arg in raw_qargs
    ]
    
    # Check broadcasting consistency: lengths must either be 1 or identical K
    max_len = max(len(lst) for lst in resolved_arg_lists)
    for lst in resolved_arg_lists:
        if len(lst) not in (1, max_len):
            raise QASMParseError(f"Register dimension mismatch in gate '{gate_name}': cannot broadcast {len(lst)} to {max_len}.", line_num, stmt)
            
    # Expand broadcast into individual gate operations
    for i in range(max_len):
        step_qargs: List[Tuple[str, int]] = []
        for lst in resolved_arg_lists:
            item = lst[i] if len(lst) == max_len else lst[0]
            step_qargs.append(item)
            
        # Verify no duplicate qubits in the same multi-qubit gate
        if len(step_qargs) != len(set(step_qargs)):
            raise QASMParseError(f"Duplicate target qubit in gate '{gate_name}': {step_qargs}.", line_num, stmt)
            
        # Check custom gate macro expansion
        if gate_name in custom_gate_macros:
            _expand_custom_gate_macro(gate_name, eval_params, step_qargs, circuit, custom_gate_macros, line_num)
            continue
            
        # Verify against standard gate catalog
        if gate_name not in STANDARD_GATES_SPECS:
            raise QASMParseError(f"Undefined gate: '{gate_name}'.", line_num, stmt)
            
        exp_params, exp_qubits = STANDARD_GATES_SPECS[gate_name]
        if exp_params is not None and len(eval_params) != exp_params:
            raise QASMParseError(f"Gate '{gate_name}' expects {exp_params} parameter(s), got {len(eval_params)}.", line_num, stmt)
        if len(step_qargs) != exp_qubits:
            raise QASMParseError(f"Gate '{gate_name}' expects {exp_qubits} qubit(s), got {len(step_qargs)}.", line_num, stmt)
            
        circuit.instructions.append(
            QASMInstruction(
                name=gate_name,
                params=eval_params,
                qargs=step_qargs,
                line_num=line_num,
                raw_statement=stmt
            )
        )


def _expand_custom_gate_macro(
    macro_name: str,
    params: List[float],
    qargs: List[Tuple[str, int]],
    circuit: ParsedQASMCircuit,
    custom_gate_macros: Dict[str, Dict[str, Any]],
    line_num: int
) -> None:
    """Expand a custom user-defined gate macro by substituting parameters and qubit arguments."""
    macro = custom_gate_macros[macro_name]
    expected_params = macro["params"]
    expected_args = macro["args"]
    
    if len(params) != len(expected_params):
        raise QASMParseError(f"Macro gate '{macro_name}' expects {len(expected_params)} parameter(s), got {len(params)}.", line_num)
    if len(qargs) != len(expected_args):
        raise QASMParseError(f"Macro gate '{macro_name}' expects {len(expected_args)} qubit(s), got {len(qargs)}.", line_num)
        
    param_map = dict(zip(expected_params, params))
    arg_map = dict(zip(expected_args, qargs))
    
    for sub_stmt in macro["body"]:
        sub_stmt = sub_stmt.strip()
        if not sub_stmt:
            continue
            
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*\(([^)]*)\))?\s+(.+)$", sub_stmt)
        if not m:
            raise QASMParseError(f"Malformed sub-statement in macro '{macro_name}': '{sub_stmt}'.", line_num)
            
        sub_gate = m.group(1)
        sub_params_str = m.group(2)
        sub_args_str = m.group(3)
        
        # Substitute parameter symbols
        sub_eval_params: List[float] = []
        if sub_params_str:
            for p_expr in sub_params_str.split(","):
                p_expr = p_expr.strip()
                for p_sym, p_val in param_map.items():
                    # Substitute symbol in expression
                    p_expr = re.sub(rf"\b{p_sym}\b", str(p_val), p_expr)
                sub_eval_params.append(evaluate_parameter_expression(p_expr))
                
        # Substitute qubit args
        sub_qargs: List[Tuple[str, int]] = []
        for a_token in sub_args_str.split(","):
            a_token = a_token.strip()
            if a_token in arg_map:
                sub_qargs.append(arg_map[a_token])
            else:
                # Direct register index
                resolved = _resolve_qubit_or_clbit_arg(a_token, circuit.qregs, "quantum", line_num, sub_stmt)
                sub_qargs.extend(resolved)
                
        if sub_gate in custom_gate_macros:
            _expand_custom_gate_macro(sub_gate, sub_eval_params, sub_qargs, circuit, custom_gate_macros, line_num)
        elif sub_gate in STANDARD_GATES_SPECS:
            circuit.instructions.append(
                QASMInstruction(name=sub_gate, params=sub_eval_params, qargs=sub_qargs, line_num=line_num, raw_statement=sub_stmt)
            )
        else:
            raise QASMParseError(f"Undefined sub-gate '{sub_gate}' in macro '{macro_name}'.", line_num)


def load_qasm(filepath: str) -> ParsedQASMCircuit:
    """
    Load and parse an OpenQASM 2.0 file from disk.
    
    Args:
        filepath (str): Path to .qasm file.
        
    Returns:
        ParsedQASMCircuit: Parsed and validated circuit instance.
        
    Raises:
        FileNotFoundError: If filepath does not exist.
        QASMParseError: If parsing or syntax validation fails.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"OpenQASM file not found: {filepath}")
        
    circuit_name = os.path.splitext(os.path.basename(filepath))[0]
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    return parse_qasm(content, name=circuit_name)
