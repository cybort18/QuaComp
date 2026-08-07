# QuaComp

> **Quantum Computer Simulation Benchmark** — A modular Python utility designed to measure, stress-test, and benchmark quantum computer simulation limits on local hardware environments.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests Status](https://img.shields.io/badge/tests-passing-green.svg)](#running-tests)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Getting Started](#getting-started)
- [Usage Examples](#usage-examples)
- [Running Tests](#running-tests)
- [Roadmap](#roadmap)
- [Scoring Categories](#scoring-categories)
- [Contribution Guide](#contribution-guide)
- [License](#license)

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

### System Telemetry & Benchmark Scorer (Phase 2)
- Inspects real-time multi-core CPU usage, processor models, RAM, OS environment details.
- Computes overall **QuaComp Score** via the formula:
  $$\text{Score} = (2^{\text{max\_qubits}} \times 10) + \left(\frac{\text{total\_gates}}{\text{execution\_time}}\right)$$

### JSON & Markdown Exporters (Phase 3)
- Automatically serializes run telemetry to `results/benchmark_<timestamp>.json`.
- Exports readable summary reports to `results/report.md` for copy-pasting to GitHub issues or discussions.

---

## Project Architecture

```text
QuaComp/
├── cli/
│   ├── __init__.py
│   └── main.py             # Rich terminal GUI CLI entry point
├── src/
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── circuits.py     # Circuit generators (Shallow, Deep, QFT)
│   │   └── simulator.py    # Aer Simulator wrapper & latency profiler
│   ├── profiler/
│   │   ├── __init__.py
│   │   ├── memory.py       # Pre-flight memory estimator & safety check
│   │   └── telemetry.py    # CPU and hardware profiler
│   ├── scorer/
│   │   ├── __init__.py
│   │   └── calculator.py   # Benchmark scorer engine
│   └── reporter/
│       ├── __init__.py
│       ├── json_exporter.py# Save results in JSON format
│       └── md_exporter.py  # Save reports in Markdown format
├── tests/
│   ├── test_engine.py      # Circuit and simulation execution tests
│   ├── test_memory.py      # Memory limits and checker tests
│   ├── test_scorer.py      # Score calculations & categorization tests
│   └── test_reporter.py    # Exporters files creation tests
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

### Running the CLI Benchmark Suite

You can execute the benchmark program via the terminal. Specify `PYTHONPATH` to ensure Python resolves the codebase packages correctly:

```bash
# Run a quick benchmark on qubits 10, 15, and 20
$env:PYTHONPATH="." ; python cli/main.py --quick

# Run a full incremental stress test starting from 10 qubits until memory safety limits
$env:PYTHONPATH="." ; python cli/main.py --full

# Run a custom 12 qubits deep workload simulation of depth 15
$env:PYTHONPATH="." ; python cli/main.py --custom --qubits 12 --type deep --depth 15
```

### Export Options
By default, completing a benchmark run automatically creates exports of both JSON and Markdown reports. You can control the output using the `--export` flag:
```bash
# Export only Markdown reports
$env:PYTHONPATH="." ; python cli/main.py --quick --export md

# Export only JSON reports
$env:PYTHONPATH="." ; python cli/main.py --quick --export json
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
rootdir: C:\Users\HP\Documents\PROJECT\QuaComp
collected 19 items

tests\test_engine.py .....                                               [ 26%]
tests\test_memory.py .....                                               [ 52%]
tests\test_reporter.py ....                                              [ 73%]
tests\test_scorer.py .....                                               [100%]

============================= 19 passed in 2.76s ==============================
```

---

## Scoring Categories

QuaComp Score maps directly into performance tiers, reflecting the computing capabilities of local environments:

| Tier Category | Score Range (Points) | Max Qubits Simulation Range |
| :--- | :--- | :--- |
| **Entry-Level** | $< 100,000$ | Up to 18-20 Qubits |
| **Mid-Range** | $100,000$ to $1,000,000$ | Up to 22-25 Qubits |
| **High-Performance** | $1,000,000$ to $50,000,000$ | Up to 26-28 Qubits |
| **Extreme Workstation** | $> 50,000,000$ | $29+$ Qubits |

---

## Roadmap

- [x] **Phase 1: Core Simulation & Safety**
  - Implement memory safety checks.
  - Implement circuit workload generators (Shallow, Deep, QFT).
  - Integrate Aer simulator execution & time tracking.
  - Build out full unit test coverage.
- [x] **Phase 2: Scoring & CLI Interface**
  - Implement benchmark scoring algorithms ("QuaComp Score").
  - Create interactive terminal GUI using the `rich` library.
- [x] **Phase 3: Exporters & Reports**
  - Add JSON / Markdown export features.
  - Publish documentation.

---

## Contribution Guide

Contributions are welcome! Please follow these steps to contribute:
1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

Make sure to run the `pytest` test suite before submitting pull requests to verify all system features remain functional.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
