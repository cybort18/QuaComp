# Product Requirement Document (PRD)
# Project Name: QuaComp (Quantum Computer Simulation Benchmark)

**Version:** 1.1.0  
**Status:** Approved / Completed  
**Target Environment:** Cross-platform (Windows, macOS, Linux)  
**Primary Tech Stack:** Python 3.10+, Qiskit / Aer, psutil, Rich, Pytest  

---

## 1. Executive Summary & Vision

### 1.1 Overview
`QuaComp` is an open-source library and CLI-based benchmarking tool designed to test the limits of local hardware (PC / Laptop / Workstation) when simulating quantum computers. 

By leveraging state-vector simulation of dimension $2^n$, `QuaComp-Bench` measures memory allocation (RAM), CPU usage, and execution latency of quantum gate operations across different qubit sizes and circuit depths. In addition to the state-vector method, QuaComp supports Matrix Product State (MPS) simulations to efficiently handle large-scale quantum circuits (30 to 100+ qubits) by compressing the state-space tensor representation on memory-constrained systems. Furthermore, QuaComp supports Noisy Intermediate-Scale Quantum (NISQ) simulation benchmarking using built-in noise models (Thermal Relaxation $T_1/T_2$ & Depolarizing Error) to evaluate CPU computational overhead and quantum state fidelity loss compared to ideal circuit execution.

### 1.2 Core Value Proposition
- **Pre-flight Safety:** Prevents system crashes and Out-Of-Memory (OOM) errors by calculating theoretical $2^n$ RAM requirements before running simulation runs.
- **Realistic Benchmarks:** Evaluates hardware against diverse quantum circuit workloads (Shallow, Deep, and Algorithmic QFT).
- **Comprehensive Profiling:** Tracks execution time, latency, peak RAM consumption, and multi-core CPU utilization.
- **Standardized Scoring:** Computes a standardized quantum score ("QuaComp Score") allowing users to compare performance across different machine configurations.
- **Scalable MPS Simulation:** Supports high-qubit simulation (up to 100+ qubits) using tensor network compression (Matrix Product State), overcoming conventional state-vector memory limits for low-to-moderate entanglement circuits.
- **NISQ Noise & Fidelity Profiling:** Evaluates hardware computational overhead under realistic quantum noise channels (thermal relaxation and depolarizing errors) while measuring quantum state fidelity loss.

---

## 2. Target Audience & Primary Use Cases

1. **Quantum Researchers & Students:** Discovering the maximum number of qubits that can be simulated locally before deploying circuits to IBM Quantum cloud platforms.
2. **Hardware Enthusiasts & Benchmarkers:** Stress-testing CPU performance, RAM speed, and memory bandwidth using large-scale quantum matrix calculations.
3. **Developers & AI Agents:** Utilizing this repository as a reference for structured, modular Python engineering ready for future extensions.

---

## 3. Functional Requirements (FR)

### FR-1: Pre-flight Memory Safety System
- **Description:** Before allocating a quantum state-vector of dimension $2^n$, the system must check available physical memory.
- **Memory Estimation Formula:** 
  $$\text{RAM Bytes} = 2^n \times 16 \text{ bytes (for complex128 representation)}$$
- **Behavior:** If the estimated memory exceeds $85\%$ of the remaining available physical RAM, the system issues a warning or aborts the execution to prevent OOM crashes.

### FR-2: Incremental Qubit Stress Testing Engine
- **Description:** Gradually benchmark qubits starting from $n = 10$ up to the system's maximum limit ($n = 28\text{--}32+$ depending on RAM).
- **Execution Modes:**
  1. *Quick Benchmark:* Runs standard tests on $n \in \{10, 15, 20\}$ qubits.
  2. *Full Stress Test:* Incrementally adds $+1$ qubit until the RAM safety limit is hit.
  3. *Custom Test:* Allows the user to specify custom qubits, workloads, and circuit depths.

### FR-3: Benchmark Workload Suite
Three types of quantum workloads are supported:
1. **Shallow Workload (Bell / Hadamard):**
   - Tests initial overhead and state initialization.
   - Circuit structure: Hadamard ($H$) gates on all qubits followed by a chain of CNOT ($CX$) gates.
2. **Deep Workload (Random Circuit):**
   - Tests dense matrix multiplication.
   - Circuit structure: Layered random rotation gates ($R_x, R_y, R_z$) and multi-level $CNOT$ gates with depths of $d \in \{10, 50, 100\}$.
3. **Algorithmic Workload (Quantum Fourier Transform - QFT):**
   - Evaluates performance under standard quantum algorithms.
   - Circuit structure: Quantum Fourier Transform (QFT) logic applied to $n$ qubits.

### FR-4: Hardware Profiler & Telemetry Module
- **Metrics Tracked:**
  - **Baseline Memory:** RAM footprint before simulation execution.
  - **Peak Memory Usage:** Maximum RAM utilized during simulation matrix allocation.
  - **Execution Latency:** Execution latency in seconds per circuit.
  - **CPU Core Utilization:** CPU core utilization percentage.
  - **System Metadata:** CPU model name, total physical RAM, operating system, and Python/Qiskit version details.

### FR-5: Scoring Formula Engine
The final score (**QuaComp Score**) is calculated by combining the *Maximum Qubits* simulated with the *Execution Throughput*:
$$\text{QuaComp Score} = \left( 2^{n_{\text{max}}} \times 10 \right) + \left( \frac{\text{Total Gates Executed}}{\text{Total Time (seconds)}} \right)$$
- **Scoring Tiers:**
  - *Entry-Level:* < 100,000 pts (Max 18-20 Qubits)
  - *Mid-Range:* 100,000 - 1,000,000 pts (Max 22-25 Qubits)
  - *High-Performance:* 1,000,000 - 50,000,000 pts (Max 26-28 Qubits)
  - *Extreme Workstation:* > 50,000,000 pts (>= 29 Qubits)

### FR-6: Report & Export Module
- Benchmark results can be exported as:
  1. **JSON Output:** For programmatic parsing (`results/benchmark_<timestamp>.json`).
  2. **Markdown Summary:** Clean reports ready for GitHub Issues/Discussions (`results/report.md`).
  3. **Terminal Dashboard:** Structured tables and interactive progress spinners inside the CLI.

### FR-7: Matrix Product State (MPS) Simulation Engine
- **Description:** The system must provide options to execute circuits using the Matrix Product State (MPS) tensor network method to simulate higher qubit counts with minimal RAM usage.
- **Specifications & Behavior:**
  - Integrates with the `qiskit_aer.AerSimulator(method='matrix_product_state')` backend.
  - Allows configuring the maximum bond dimension ($\chi$) via CLI `--bond-dim` (default $\chi = 64$).
  - Supports high-qubit stress tests in the range of $n = 30\text{--}100$ qubits for circuits with low-to-moderate entanglement.
  - Tracks specific MPS metrics:
    - RAM footprint comparison between the MPS method and theoretical Statevector memory requirements ($2^n \times 16$ bytes).
    - Maximum active bond dimension used during simulation.

### FR-8: Noisy Quantum Simulation Engine (NISQ)
- **Description:** The system must provide options to run simulations under realistic quantum noise channels using Qiskit Aer's noise module to measure computational overhead and state fidelity degradation.
- **Specifications & Behavior:**
  - Integration with `qiskit_aer.noise` (`NoiseModel`, `ThermalRelaxationError`, `depolarizing_error`).
  - Customizable noise level presets configurable via CLI `--noise-level [none|low|medium|high]`:
    - `none`: Ideal simulation (no noise applied).
    - `low`: Mild decoherence ($T_1 = 100\,\mu\text{s}, T_2 = 120\,\mu\text{s}$, gate error rate $0.1\%$).
    - `medium`: Standard physical hardware profile ($T_1 = 50\,\mu\text{s}, T_2 = 70\,\mu\text{s}$, gate error rate $0.5\%$).
    - `high`: Heavy noise profile for extreme stress testing ($T_1 = 20\,\mu\text{s}, T_2 = 30\,\mu\text{s}$, gate error rate $2.0\%$).
  - Simulation support using `density_matrix` or `statevector` simulation backends with inserted noise channels.
  - Tracking specific NISQ metrics:
    - **Quantum State Fidelity / Overlap Loss:** Quantitative comparison between ideal statevector and noisy simulation outcome.
    - **CPU Computation Overhead Ratio:** Percentage increase in CPU execution latency resulting from noise channel matrix operations.

---

## 4. Non-Functional Requirements (NFR)

- **Performance:** Profiler telemetry overhead must not exceed $1\%$ of the primary simulation execution time or memory.
- **Portability:** Cross-platform compatibility on Windows 10/11, macOS (Intel & Apple Silicon M1/M2/M3), and Ubuntu/Linux.
- **User Experience (CLI):** Modern CLI interface using progress bars and styled tables (via the `rich` library).
- **Code Quality:** PEP8 compliant formatting, full type annotation, and unit test coverage > 80%.

---

## 5. Technical Architecture & File Structure

```text
QuaComp-bench/
├── cli/
│   ├── __init__.py
│   └── main.py             # CLI Entry point (Rich UI - supports --method, --bond-dim, --noise-level)
├── src/
│   ├── __init__.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── circuits.py     # Quantum circuit generator (Shallow, Deep, QFT)
│   │   ├── mps.py          # MPS configuration and tensor compression metrics handler
│   │   ├── noise.py        # Preset noise model generator & state fidelity calculator
│   │   └── simulator.py    # Qiskit Aer simulator wrapper (supports 'statevector', 'matrix_product_state', & noise models)
│   ├── profiler/
│   │   ├── __init__.py
│   │   ├── memory.py       # Pre-flight safety checks & RAM estimation
│   │   └── telemetry.py    # CPU and system metadata profiler
│   ├── scorer/
│   │   ├── __init__.py
│   │   └── calculator.py   # QuaComp score calculator and tier categorization
│   └── reporter/
│       ├── __init__.py
│       ├── json_exporter.py
│       └── md_exporter.py
├── tests/
│   ├── test_engine.py      # Circuit generator and simulator tests
│   ├── test_memory.py      # RAM safety estimator tests
│   ├── test_scorer.py      # Score formula and tier classification tests
│   ├── test_mps.py         # Matrix Product State (MPS) logic tests
│   └── test_noise.py       # NISQ noise models and state fidelity tests
├── requirements.txt
├── PRD.md
├── README.md
└── LICENSE
```

---

## 6. Implementation Roadmap for AI Agent

### Phase 1: Core Simulation & Safety
- [x] Create `profiler/memory.py` for RAM estimation and physical RAM checks using `psutil`.
- [x] Create `engine/circuits.py` to generate Shallow, Deep, and QFT circuits.
- [x] Create `engine/simulator.py` to run simulations using `AerSimulator(method='statevector')`.

### Phase 2: Scoring & CLI Interface
- [x] Implement score calculation and tier classification in `scorer/calculator.py`.
- [x] Create the CLI benchmarking console in `cli/main.py` using `rich`.

### Phase 3: Exporters, Tests & Documentation
- [x] Add JSON and Markdown report exporters in `reporter/`.
- [x] Complete the pytest suite in the `tests/` directory.
- [x] Write comprehensive, professional documentation in `README.md`.

### Phase 4: MPS Integration & High-Qubit Benchmarking
- [x] Implement `engine/mps.py` and update `engine/simulator.py` to support the MPS backend.
- [x] Integrate `--method` and `--bond-dim` command flags in `cli/main.py`.
- [x] Implement specialized MPS tests in `tests/test_mps.py`.
- [x] Update JSON and Markdown exporters to report MPS metrics and RAM savings.

### Phase 5: NISQ Simulation & Fidelity Benchmarking (Week 5)
- [ ] Create `src/engine/noise.py` to generate preset noise models and calculate quantum state fidelity.
- [ ] Update `src/engine/simulator.py` to accept a `noise_model` parameter.
- [ ] Integrate `--noise-level` flag in `cli/main.py` and display fidelity/overhead metrics in the Rich CLI output.
- [ ] Add NISQ unit tests in `tests/test_noise.py`.
- [ ] Update reporter modules (JSON/MD) to record noise parameters and fidelity metrics.
