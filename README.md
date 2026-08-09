# QuaComp

> **Quantum Computer Simulation Benchmark** — A modular Python utility designed to measure, stress-test, and profile quantum computer simulation limits on local hardware environments.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests Status](https://img.shields.io/badge/tests-27%20passed-green.svg)](#running-tests)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Getting Started](#getting-started)
- [Usage Examples](#usage-examples)
- [Running Tests](#running-tests)
- [Scoring Categories](#scoring-categories)
- [Roadmap](#roadmap)
- [Contribution Guide](#contribution-guide)
- [License](#license)

---

## Overview

**QuaComp** is an open-source tool and benchmarking suite developed to profile local machine performance during quantum circuit simulation. Supporting Statevector, Matrix Product State (MPS), and Noisy Intermediate-Scale Quantum (NISQ) noise engines, QuaComp evaluates execution latencies, CPU/memory performance, state fidelity loss, and calculates consistent metrics defined by QuaComp for comparative profiling across local environments.

---

## Key Features

### Pre-flight Memory Safety (Phase 1)
- Computes estimated memory requirements prior to statevector simulation runs using:
  $$\text{RAM Bytes} = 2^n \times 16 \text{ bytes (for complex128 representation)}$$
- Integrates with `psutil` to dynamically inspect physical system memory.
- Blocks and warns simulations exceeding 85% of available RAM to prevent OS crashes and Out-Of-Memory (OOM) situations.

### Quantum Workload Generators (Phase 1)
- **Shallow Workloads**: Initial state allocations using Hadamard gates coupled with 1D entanglement (CNOT chains).
- **Deep Workloads**: Intensive random rotation matrices ($R_x, R_y, R_z$) and multi-layered entanglement chains designed to stress memory bandwidth.
- **Quantum Fourier Transform (QFT)**: Standard implementation representing realistic quantum algorithms.

### Multi-Engine Simulator Core (Phase 1 & Phase 4)
- **Statevector Simulation Engine**: Exact statevector simulation method (`AerSimulator(method='statevector')`).
- **Matrix Product State (MPS) Engine**: Tensor network simulation engine (`AerSimulator(method='matrix_product_state')`) enabling high-qubit simulation ($30\text{--}100+$ qubits) specifically for circuits with low-to-moderate entanglement using custom bond dimensions (`--bond-dim`, default 64).
- **RAM Efficiency Profiling**: Calculates exact memory savings achieved by MPS compared to theoretical statevector memory footprint ($2^n \times 16$ bytes).

### NISQ Noise & Fidelity Profiler (Phase 5)
- **Synthetic Parameterized Noise Channels**: Incorporates Thermal Relaxation ($T_1, T_2$) and Depolarizing Errors using `qiskit_aer.noise`.
- **Preset Noise Profiles**: Configurable noise presets via `--noise-level [none|low|medium|high]`:
  - `none`: Ideal noise-free simulation.
  - `low`: Mild decoherence ($T_1=100\,\mu\text{s}, T_2=120\,\mu\text{s}$, gate error $0.1\%$).
  - `medium`: Synthetic representative noise profile ($T_1=50\,\mu\text{s}, T_2=70\,\mu\text{s}$, gate error $0.5\%$).
  - `high`: Heavy noise profile for extreme stress testing ($T_1=20\,\mu\text{s}, T_2=30\,\mu\text{s}$, gate error $2.0\%$).
- **Fidelity & Overhead Metrics**: Computes classical Hellinger Quantum State Fidelity (%) and CPU Computation Overhead ratio (%).

### Multi-Run Benchmarking & Telemetry (Methodology Revision)
- **Statistical Repeatability**: Executes `--runs INT` (default 3) benchmark iterations per circuit to compute Mean ($\mu$), Median, and Standard Deviation ($\sigma$) of execution latency, mitigating CPU governor and background task noise.
- **Composite Heuristic Scoring**: Computes the **QuaComp Composite Score** (a project-specific heuristic score) that separates state-space capacity from gate throughput:
  $$\text{Score} = (C \times 10) + T = (2^{\text{max qubits}} \times 10) + \left(\frac{\text{Total Gates}}{\mu_{\text{latency}}}\right)$$
  - **Capacity Metric ($C = 2^{\text{max qubits}}$)**: Qubit state-space capacity metric.
  - **Throughput Metric ($T = \frac{\text{Total Gates}}{\mu_{\text{latency}}}$)**: Gate processing throughput metric (gates/second).
  *Note: QuaComp Score is a project-specific composite heuristic prioritizing state-space capacity scaling.*

### JSON & Markdown Exporters (Phase 3)
- Automatically serializes run telemetry and statistical summaries to `results/benchmark_<timestamp>.json`.
- Exports readable summary reports to `results/report.md` formatted for GitHub issues or discussions.

---

## Project Architecture

```text
QuaComp/
├── cli/
│   ├── __init__.py
│   └── main.py             # Rich terminal GUI CLI entry point (supports --method, --bond-dim, --noise-level, --runs)
├── src/
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── circuits.py     # Circuit generators (Shallow, Deep, QFT)
│   │   ├── mps.py          # MPS configuration & RAM savings profiler
│   │   ├── noise.py        # NISQ noise presets & state fidelity calculator
│   │   └── simulator.py    # Aer Simulator wrapper (Multi-run statistics, Statevector, MPS, Noise support)
│   ├── profiler/
│   │   ├── __init__.py
│   │   ├── memory.py       # Pre-flight memory estimator & safety check
│   │   └── telemetry.py    # CPU and hardware profiler
│   ├── scorer/
│   │   ├── __init__.py
│   │   └── calculator.py   # Benchmark scorer engine & breakdown calculator
│   └── reporter/
│       ├── __init__.py
│       ├── json_exporter.py# Save results & statistics in JSON format
│       └── md_exporter.py  # Save reports in Markdown format
├── tests/
│   ├── test_engine.py      # Circuit and simulation execution tests
│   ├── test_memory.py      # Memory limits and checker tests
│   ├── test_scorer.py      # Score calculations & breakdown tests
│   ├── test_reporter.py    # Exporters files creation tests
│   ├── test_mps.py         # Matrix Product State (MPS) logic tests
│   └── test_noise.py       # NISQ noise models and state fidelity tests
├── requirements.txt        # Package dependencies
├── PRD.md                  # Product Requirement Document
├── README.md               # Project documentation
└── .gitignore              # Git ignore file
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/cybort18/QuaComp.git
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
# Run a quick benchmark on qubits 10, 15, and 20 with 3 runs per circuit
$env:PYTHONPATH="." ; python cli/main.py --quick

# Run a full incremental stress test starting from 10 qubits with 5 statistical runs
$env:PYTHONPATH="." ; python cli/main.py --full --runs 5

# Run a custom 30 qubits simulation using Matrix Product State (MPS) engine for low-entanglement circuits
$env:PYTHONPATH="." ; python cli/main.py --custom --qubits 30 --method mps --bond-dim 64

# Run a custom simulation under a synthetic representative noise profile (medium)
$env:PYTHONPATH="." ; python cli/main.py --custom --qubits 10 --noise-level medium --runs 5
```

### Command Flags Reference

| Flag | Options / Default | Description |
| :--- | :--- | :--- |
| `--quick` | N/A | Runs benchmark suite on 10, 15, and 20 qubits. |
| `--full` | N/A | Incremental stress test starting from 10 qubits. |
| `--custom` | N/A | Custom simulation mode with specific qubit parameters. |
| `--qubits` | `INT` (default: `10`) | Qubit count for custom simulation run. |
| `--type` | `shallow`, `deep`, `qft` (default: `qft`) | Quantum circuit workload type. |
| `--depth` | `INT` (default: `10`) | Depth parameter for deep random circuit workloads. |
| `--method` | `statevector`, `mps` (default: `statevector`) | Simulation engine method. |
| `--bond-dim` | `INT` (default: `64`) | Maximum bond dimension for MPS tensor network engine. |
| `--noise-level` | `none`, `low`, `medium`, `high` (default: `none`) | NISQ synthetic noise preset level. |
| `--runs` | `INT` (default: `3`) | Number of benchmark iterations per circuit for statistical mean/std calculation. |
| `--export` | `json`, `md`, `all` (default: `all`) | Benchmark report output format. |

---

## Running Tests

Automated unit tests are written with `pytest`. They cover statevector simulation, multi-run latency statistics, MPS tensor compression, NISQ synthetic noise models, scoring breakdown, and report exporters.

To execute the full test suite, run:
```bash
python -m pytest
```

Output:
```text
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\HP\Documents\PROJECT\QuaComp
collected 27 items

tests\test_engine.py .....                                               [ 18%]
tests\test_memory.py .....                                               [ 37%]
tests\test_mps.py ....                                                   [ 51%]
tests\test_noise.py ....                                                 [ 66%]
tests\test_reporter.py ....                                              [ 81%]
tests\test_scorer.py .....                                               [100%]

============================= 27 passed in 3.68s ==============================
```

---

## Scoring Categories

QuaComp Composite Score maps directly into performance tiers, reflecting the computing capabilities of local environments:

| Tier Category | Score Range (Points) | Max Qubits Simulation Range |
| :--- | :--- | :--- |
| **Entry-Level** | $< 100,000$ | Up to 18-20 Qubits |
| **Mid-Range** | $100,000$ to $1,000,000$ | Up to 22-25 Qubits |
| **High-Performance** | $1,000,000$ to $50,000,000$ | Up to 26-28 Qubits |
| **Extreme Workstation** | $> 50,000,000$ | $29+$ Qubits |

*Note: QuaComp Score is a project-specific heuristic score combining capacity scaling ($2^n$) and gate throughput.*

---

## Roadmap

- [x] **Phase 1: Core Simulation & Safety**
  - Implement memory safety checks.
  - Implement circuit workload generators (Shallow, Deep, QFT).
  - Integrate Aer simulator execution & time tracking.
  - Build out unit test coverage.
- [x] **Phase 2: Scoring & CLI Interface**
  - Implement benchmark scoring algorithms ("QuaComp Score").
  - Create interactive terminal GUI using the `rich` library.
- [x] **Phase 3: Exporters & Reports**
  - Add JSON / Markdown export features.
  - Publish documentation.
- [x] **Phase 4: Matrix Product State (MPS) Engine**
  - High-qubit simulation capabilities ($30\text{--}100+$ qubits for low-to-moderate entanglement).
  - Parameterizable bond dimension (`--bond-dim`).
  - Memory efficiency savings profiler.
- [x] **Phase 5: NISQ Noise & Fidelity Benchmarking**
  - Qiskit Aer synthetic noise channel integration ($T_1/T_2$ relaxation & depolarizing error).
  - Customizable noise presets (`--noise-level [none|low|medium|high]`).
  - Quantum State Fidelity (%) & CPU Computation Overhead (%) tracking.
- [x] **Methodological Revision Phase**
  - Multi-run statistical benchmarking (`--runs INT`, Mean, Median, Std Dev).
  - Scoring breakdown (Capacity Metric $C$ & Throughput Metric $T$).
  - Softened academic terminology across documentation.

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
