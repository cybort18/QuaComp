# QuaComp 🌌

> **Quantum Computer Simulation Benchmark** — A modular Python utility designed to measure, stress-test, and benchmark quantum computer simulation limits on local hardware environments.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests Status](https://img.shields.io/badge/tests-passing-green.svg)](#running-tests)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Getting Started](#-getting-started)
- [Usage Examples](#-usage-examples)
- [Running Tests](#-running-tests)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## Overview

**QuaComp** is an open-source tool and benchmarking suite developed to profile and identify the computational boundaries of local machines when executing quantum statevector simulations. By utilizing $2^n$ dimensional simulation workloads, QuaComp helps researchers, students, and hardware enthusiasts evaluate execution latencies, CPU/memory performance, and calculate standard reference metrics for comparison across platforms.

---

## Key Features

### Pre-flight Memory Safety (Phase 1)
- Computes estimated memory requirements prior to simulation runs using:
  $$\text{RAM Bytes} = 2^n \times 16 \text{ bytes (for complex128 representation)}$$
- Integrates with `psutil` to dynamically inspect physical system memory.
- Blocks and warns simulations exceeding 85% of available RAM to prevent OS crashes and Out-Of-Memory (OOM) situations.

### Quantum Workload Generators (Phase 1)
- **Shallow Workloads**: Initial state allocations using Hadamard gates coupled with 1D entanglement (CNOT chains).
- **Deep Workloads**: Intensive random rotation matrices ($R_x, R_y, R_z$) and multi-layered entanglement chains designed to stress memory-bandwidth.
- **Quantum Fourier Transform (QFT)**: Standard implementation representing realistic quantum algorithms.

### Aer Simulator Engine (Phase 1)
- Seamless execution wrapper around Qiskit Aer's `AerSimulator(method='statevector')` backend.
- Accurate telemetry tracking for simulation execution latency.

---

## Project Architecture

```text
QuaComp/
├── src/
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── circuits.py     # Circuit generators (Shallow, Deep, QFT)
│   │   └── simulator.py    # Aer Simulator wrapper & latency profiler
│   └── profiler/
│       ├── __init__.py
│       └── memory.py       # Pre-flight memory estimator & safety check
├── tests/
│   ├── test_engine.py      # Engine and circuit generation tests
│   └── test_memory.py      # Memory estimation & mock virtual memory tests
├── requirements.txt        # Package dependencies
├── PRD.md                  # Product Requirement Document
├── README.md               # Project documentation
└── .gitignore              # Git ignore file
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/QuaComp.git
cd QuaComp
```

### 2. Install dependencies
Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```

---

## Usage Examples

### Memory Estimation and Pre-flight Verification
```python
from src.profiler.memory import estimate_qubit_ram, check_memory_safety

# Estimate RAM needed for 26 qubits (in bytes)
num_qubits = 26
bytes_needed = estimate_qubit_ram(num_qubits)
print(f"RAM needed for {num_qubits} qubits: {bytes_needed / (1024**3):.2f} GB")

# Check if it is safe to run on your local machine
is_safe, status_msg = check_memory_safety(num_qubits)
print(status_msg)
```

### Creating and Running Workloads
```python
from src.engine.circuits import generate_qft_circuit
from src.engine.simulator import run_simulation

# Generate an 8-qubit QFT circuit
circuit = generate_qft_circuit(8)

# Run the simulation on the statevector backend
result = run_simulation(circuit)

if result["success"]:
    print(f"Simulation completed successfully in {result['latency']:.4f} seconds!")
    print(f"Execution Metadata: {result['metadata']}")
else:
    print(f"Simulation failed: {result['error']}")
```

---

## Running Tests

Automated unit tests are written with `pytest`. They mock hardware details to ensure compatibility and correctness across all environments.

To execute the test suite, run:
```bash
python -m pytest
```

Output:
```text
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.1.1, pluggy-1.6.0
collected 10 items

tests\test_engine.py .....                                               [ 50%]
tests\test_memory.py .....                                               [100%]

============================= 10 passed in 4.38s ==============================
```

---

## Roadmap

- [x] **Phase 1: Core Simulation & Safety**
  - Implement memory safety checks.
  - Implement circuit workload generators (Shallow, Deep, QFT).
  - Integrate Aer simulator execution & time tracking.
  - Build out full unit test coverage.
- [ ] **Phase 2: Scoring & CLI Interface**
  - Implement benchmark scoring algorithms ("QuaComp Score").
  - Create interactive terminal GUI using the `rich` library.
- [ ] **Phase 3: Exporters & Reports**
  - Add JSON / Markdown export features.
  - Publish documentation.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
