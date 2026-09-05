# Product Requirement Document (PRD)
# Project Name: QuaComp (Quantum Computer Simulation Benchmark)

**Version:** 1.0.0  
**Status:** Approved / Completed  
**Target Environment:** Cross-platform (Windows, macOS, Linux)  
**Primary Tech Stack:** Python 3.10+, Qiskit / Aer, psutil, Rich, Matplotlib, Seaborn, Setuptools, Pytest, GitHub Actions  

---

## 1. Executive Summary & Vision

### 1.1 Overview
`QuaComp` is an open-source library and CLI-based benchmarking tool designed to profile local hardware performance (PC / Laptop / Workstation) when simulating quantum computers. 

By leveraging state-vector simulation of dimension $2^n$, `QuaComp` measures memory allocation (RAM), CPU usage, and execution latency of quantum gate operations across different qubit sizes and circuit depths. In addition to the state-vector method, QuaComp supports Matrix Product State (MPS) simulations to efficiently handle large-scale quantum circuits (30 to 100+ qubits for low-to-moderate entanglement workloads) by compressing the state-space tensor representation on memory-constrained systems. Furthermore, QuaComp supports Noisy Intermediate-Scale Quantum (NISQ) simulation benchmarking using synthetic parameterized noise channels (Thermal Relaxation $T_1/T_2$ & Depolarizing Error) to evaluate CPU computational overhead and quantum state fidelity loss compared to ideal circuit execution. QuaComp also features an automated Visualization Engine (`--chart`) that generates high-resolution telemetry plots for latency, memory safety thresholds, MPS savings, and NISQ noise fidelity impacts, alongside the **Relative Benchmark Comparison Engine (`--compare`)**, **GPU Acceleration Support (`--device gpu` / `--gpu`)**, and **Entanglement Entropy Metrics (`--entropy`)**.

### 1.2 Core Value Proposition
- **Pre-flight Safety:** Prevents system crashes and Out-Of-Memory (OOM) errors by calculating theoretical $2^n$ RAM requirements before running simulation runs.
- **Representative Workloads:** Evaluates hardware against diverse quantum circuit workloads (Shallow, Deep, and Algorithmic QFT).
- **Statistical Multi-Run Profiling:** Executes multiple benchmark iterations per circuit (`--runs INT`, default 3) to compute Mean (μ), Median, and Standard Deviation (σ) of execution latencies, mitigating CPU governor and background process noise.
- **Project-Specific Composite Scoring:** Computes a project-specific composite heuristic score ("QuaComp Composite Score") that decouples state-space capacity $C = 2^n$ and gate throughput $T = \text{gates}/\mu_{\text{latency}}$.
- **Scalable MPS Simulation:** Supports high-qubit simulation (up to 100+ qubits) specifically for low-to-moderate entanglement circuits using tensor network compression (Matrix Product State), overcoming conventional state-vector memory limits.
- **NISQ Synthetic Noise & Fidelity Profiling:** Evaluates hardware computational overhead under synthetic parameterized noise channels (thermal relaxation and depolarizing errors) while measuring classical Hellinger state fidelity loss.
- **Automated Telemetry Visualizations:** Generates modern high-DPI chart graphics (`--chart`) illustrating qubit scalability, memory safety boundaries, simulation method comparisons, and noise fidelity degradation.

---

## 2. Target Audience & Primary Use Cases

1. **Quantum Researchers & Students:** Discovering the maximum number of qubits that can be simulated locally before deploying circuits to cloud platforms.
2. **Hardware Enthusiasts & Benchmarkers:** Stress-testing CPU performance, RAM speed, and memory bandwidth using large-scale quantum matrix calculations.
3. **Developers & AI Agents:** Utilizing this repository as a reference for structured, modular Python engineering ready for future extensions.

---

## 3. Functional Requirements (FR)

### FR-1: Pre-flight Memory Safety System
- **Description:** Before allocating a quantum state-vector of dimension $2^n$, the system must check available physical memory.
- **Memory Estimation Formula:** 
  $$\text{RAM Bytes} = 2^n \times 16 \text{ bytes (for complex128 representation)}$$
- **Behavior:** If the estimated memory exceeds $85\%$ of the remaining available physical RAM, the system issues a warning or aborts execution to prevent OOM crashes.

### FR-2: Incremental Qubit Stress Testing Engine
- **Description:** Gradually benchmark qubits starting from $n = 10$ up to the system's maximum limit (28 to 32+ qubits depending on RAM).
- **Execution Modes:**
  1. *Quick Benchmark:* Runs standard tests on $n \in \{10, 15, 20\}$ qubits.
  2. *Full Stress Test:* Incrementally adds $+1$ qubit until the RAM safety limit is hit.
  3. *Custom Test:* Allows the user to specify custom qubits, workloads, circuit depths, and run iteration counts.

### FR-3: Benchmark Workload Suite
Three types of quantum workloads are supported:
1. **Shallow Workload (Bell / Hadamard):**
   - Tests initial overhead and state initialization.
   - Circuit structure: Hadamard $H$ gates on all qubits followed by a chain of CNOT $CX$ gates.
2. **Deep Workload (Random Circuit):**
   - Tests dense matrix multiplication.
   - Circuit structure: Layered random rotation gates $R_x, R_y, R_z$ and multi-level CNOT gates with depths of $d \in \{10, 50, 100\}$.
3. **Algorithmic Workload (Quantum Fourier Transform - QFT):**
   - Evaluates performance under standard quantum algorithms.
   - Circuit structure: Quantum Fourier Transform (QFT) logic applied to $n$ qubits.

### FR-4: Hardware Profiler & Multi-Run Telemetry Module
- **Statistical Multi-Run Support:** Supports configurable `--runs INT` (default 3) iterations per benchmark test.
- **Metrics Tracked:**
  - **Baseline & Peak Memory:** Memory footprint before and during simulation matrix allocation.
  - **Execution Latency Statistics:** Sample Mean $\mu_{\text{latency}}$, Median, and Standard Deviation $\sigma_{\text{latency}}$ in seconds per circuit.
  - **CPU Core Utilization:** Multi-core CPU utilization percentage.
  - **System Metadata:** CPU model name, total physical RAM, operating system, and Python/Qiskit version details.

### FR-5: Composite Heuristic Scoring Engine
The composite benchmark score (**QuaComp Composite Score**) is a project-specific heuristic score that decouples state-space capacity and gate throughput:
$$\text{QuaComp Composite Score} = (C \times 10) + T = (2^{n_{\text{max}}} \times 10) + \left( \frac{\text{Total Gates Executed}}{\mu_{\text{latency}}} \right)$$
- **Capacity Metric**: $C = 2^{n_{\text{max}}}$ (prioritizes memory capacity scaling).
- **Throughput Metric**: $T = \frac{\text{Total Gates Executed}}{\mu_{\text{latency}}}$ (gates processed per second).
- **Scoring Tiers:**
  - *Entry-Level:* < 100,000 pts (Max 18-20 Qubits)
  - *Mid-Range:* 100,000 - 1,000,000 pts (Max 22-25 Qubits)
  - *High-Performance:* 1,000,000 - 50,000,000 pts (Max 26-28 Qubits)
  - *Extreme Workstation:* > 50,000,000 pts (>= 29 Qubits)
- **Methodology Note on Capacity Dominance:** Because state-vector memory allocation scales exponentially with $2^n$, the Capacity Metric of $10 \times 2^n$ exponentially dominates the Throughput Metric $T$. A machine simulating 30 qubits will score higher than a machine simulating 28 qubits with faster gate throughput, reflecting QuaComp's deliberate design choice to prioritize memory capacity scaling over execution speed.

### FR-6: Report & Export Module
- Benchmark results can be exported as:
  1. **JSON Output:** For programmatic parsing including statistical summaries (`results/benchmark_<timestamp>.json`).
  2. **Markdown Summary:** Clean reports formatted for GitHub Issues/Discussions (`results/report.md`).
  3. **Terminal Dashboard:** Structured Rich CLI tables displaying `Mean Latency ± Std Dev`.

### FR-7: Matrix Product State (MPS) Simulation Engine
- **Description:** The system must provide options to execute circuits using the Matrix Product State (MPS) tensor network method to simulate higher qubit counts with minimal RAM usage.
- **Specifications & Behavior:**
  - Integrates with the `qiskit_aer.AerSimulator(method='matrix_product_state')` backend.
  - Allows configuring the maximum bond dimension $\chi$ via CLI `--bond-dim` (default $\chi = 64$).
  - Supports high-qubit stress tests in the range of $n = 30$ to $100$ qubits specifically for circuits with low-to-moderate entanglement.
  - Tracks specific MPS metrics:
    - RAM footprint comparison between the MPS method and theoretical Statevector memory requirements of $2^n \times 16$ bytes.
    - Maximum active bond dimension used during simulation.

### FR-8: Noisy Quantum Simulation Engine (NISQ)
- **Description:** The system must provide options to run simulations under synthetic parameterized quantum noise channels using Qiskit Aer's noise module to measure computational overhead and state fidelity degradation.
- **Specifications & Behavior:**
  - Integration with `qiskit_aer.noise` (`NoiseModel`, `thermal_relaxation_error`, `depolarizing_error`).
  - Customizable noise level presets configurable via CLI `--noise-level [none|low|medium|high]`:
    - `none`: Ideal noise-free simulation.
    - `low`: $T_1 = 100\,\mu\text{s}, T_2 = 120\,\mu\text{s}$, gate error rate 0.1%.
    - `medium`: Representative synthetic noise: $T_1 = 50\,\mu\text{s}, T_2 = 70\,\mu\text{s}$, gate error rate 0.5%.
    - `high`: Heavy noise profile: $T_1 = 20\,\mu\text{s}, T_2 = 30\,\mu\text{s}$, gate error rate 2.0%.
  - Calculates classical Hellinger state fidelity percentage (%) and CPU computation overhead ratio (%).

### FR-9: Visualization Engine & Chart Generator
- **Description:** The system must provide automated generation of clean, modern visualization chart images (PNG format) from simulation telemetry via CLI flag `--chart`.
- **Specifications & Behavior:**
  - Integrates `matplotlib` (non-interactive `Agg` backend) and `seaborn` plotting libraries.
  - Automatically produces 4 core chart artifacts in `results/`:
    1. `qubit_vs_latency.png`: Qubit Count vs Mean Latency (seconds) with standard deviation error shading.
    2. `qubit_vs_ram.png`: Qubit Count vs Memory Allocation (GB) with physical RAM safety threshold line (85% limit).
    3. `method_comparison.png`: Latency comparison between Statevector and Matrix Product State (MPS) engines.
    4. `noise_fidelity_impact.png`: NISQ noise profile impact on Quantum State Fidelity (%) and CPU Computation Overhead (%).
  - Automatically embeds generated chart image links into `results/report.md`.

### FR-10: Python Packaging & CI/CD Pipeline
- **Description:** The system must adhere to modern Python PEP 517/621 packaging standards, providing direct terminal CLI binaries and automated multi-platform continuous integration.
- **Specifications & Behavior:**
  - Standard `pyproject.toml` build system with `quacomp = "cli.main:main"` console script entry point.
  - Single-command installation support via `pip install -e .` without manual `PYTHONPATH` exports.
  - Multi-platform GitHub Actions CI workflow (`.github/workflows/ci.yml`) testing on Ubuntu, Windows, and macOS across Python 3.10, 3.11, 3.12, and 3.13.

### FR-11: Relative Benchmark Comparison Engine (`--compare`)
- **Description:** The system must provide automated mathematical and visual side-by-side comparison between two benchmark result JSON runs or between a live benchmark run and a target reference hardware baseline.
- **Specifications & Behavior:**
  - **Mathematical Differencing:**
    - Calculates Composite Score Ratio: $Score_{\text{target}} / Score_{\text{base}}$ and Score Delta $\Delta Score\%$.
    - Computes Throughput Speedup Factor: $T_{\text{target}} / T_{\text{base}}$ and Throughput Gain Percentage.
    - Evaluates Qubit Simulation Capacity Gap: $\Delta n = n_{\text{target}} - n_{\text{base}}$ and State-space Scaling of $2^{\Delta n}\times$.
    - Performs per-qubit latency matching, computing speedup multipliers: $t_{\text{base}} / t_{\text{target}}$ and execution time savings.
  - **CLI Interface & Presets:**
    - `quacomp --compare <file1.json> <file2.json>`: Standalone two-file comparison.
    - `quacomp --compare results/report.json --target <alias>`: Compares against built-in preset aliases (`apple_m3`, `ryzen3_5300u`, `ryzen7_5800h`).
    - `quacomp --quick --compare --target apple_m3`: Executes live benchmark and performs instantaneous comparison against the baseline target.
  - **Rich Output & Reports:**
    - Renders color-coded Rich comparison tables and an academic summary verdict in the terminal.
    - Automatically exports `results/comparison.json` and `results/comparison_report.md`.
    - Generates grouped bar chart plots (`qubit_latency_comparison.png` and `throughput_comparison.png`) when `--chart` is provided.

### FR-12: GPU, Multi-GPU & Distributed Parallel Acceleration (`--device`, `--gpu`, `--multi-gpu`, `--workers`)
- **Description:** The system must support hardware-accelerated quantum simulation using single-GPU, multi-GPU compute devices (NVIDIA CUDA, Apple GPU, AMD) via Qiskit Aer, and multi-process distributed worker parallelism.
- **Specifications & Behavior:**
  - **Hardware Discovery & Telemetry:**
    - Detects GPU device presence, count, and individual/aggregate VRAM across Windows (CIM/WMI), Linux (`lspci` / `nvidia-smi`), and macOS (`system_profiler`).
    - Queries Qiskit Aer supported devices via `AerSimulator().available_devices()`.
    - Measures aggregate multi-GPU VRAM in GB.
  - **Memory Safety Pre-flight Check:**
    - Evaluates theoretical statevector VRAM allocation of $2^n \times 16$ bytes against available GPU / Multi-GPU VRAM.
    - Emits warnings if statevector allocation exceeds 70% of VRAM, and halts if exceeding 85%.
  - **Execution & Parallel Workers:**
    - Configures `AerSimulator(method=method, device='GPU', batched_shots_gpu=True, blocking_enable=True)` when `--multi-gpu` or `--device multi_gpu` is specified.
    - Supports distributed multi-worker parallel batch simulation (`--workers <INT>`).
    - Provides graceful diagnostic feedback on CPU-only environments without throwing unhandled exceptions.
  - **CLI Integration:**
    - `--device [cpu|gpu|multi_gpu]` (default: `cpu`).
    - `--gpu`: Shorthand flag for `--device gpu`.
    - `--multi-gpu`: Enables multi-GPU distributed simulation.
    - `--workers`: Configures parallel distributed worker threads/processes.

### FR-13: Entanglement Entropy Metrics & Simulation Hardness Profiler (`--entropy`)
- **Description:** The system must calculate bipartite Von Neumann entanglement entropy, Schmidt decomposition rank, and simulation complexity classifications across both small and large qubit regimes (up to 100+ qubits) without memory exhaustion.
- **Specifications & Behavior:**
  - **Dual Entanglement Engines (Statevector SVD & Native MPS Tensor SVD):**
    - For small circuits with $n \le 22$: Computes exact SVD on reshaped statevector array of dimension $2^{n_A} \times 2^{n_B}$.
    - For large circuits with $n > 22$ or MPS method: Employs a native 1D Matrix Product State (MPS) tensor network contractor that performs local unitary contractions, canonical QR sweeps, and central bond SVD ($O(\chi^3)$ complexity, under 2 MB RAM overhead for 100 qubits).
    - Calculates Von Neumann Entanglement Entropy:
      $$S(\rho_A) = -\sum_i \lambda_i^2 \log_2(\lambda_i^2)$$
    - Evaluates Schmidt Rank $r = \text{count}(\lambda_i > 10^{-14})$ and Participation Ratio:
      $$K = \frac{1}{\sum_i \lambda_i^4}$$
  - **Simulation Complexity Classification:**
    - Categorizes entanglement regime: `Product State` with $S=0$, `Low (Area-law)` with $S \le 1.0$, `Moderate Entanglement` with $1.0 < S < 0.7 \times S_{\max}$, and `Volume-law (Maximal)` with $S \ge 0.7 \times S_{\max}$.
    - Evaluates Matrix Product State (MPS) simulation hardness based on bond dimension scaling $\chi \sim 2^{S(\rho_A)}$: `Trivial (chi=1)`, `Efficient (Low chi <= 16)`, `Challenging (Moderate chi <= 64)`, `Exponentially Hard (Volume-law)`.
  - **Reporting & Visualizations:**
    - Adds `--entropy` CLI flag to evaluate and display Entanglement & Hardness Rich tables in terminal output.
    - Generates `results/entanglement_entropy.png` plotting entropy scaling curves against theoretical maximum bipartite bound $S_{\max}$.
    - Records entanglement metrics in exported JSON and Markdown reports.



