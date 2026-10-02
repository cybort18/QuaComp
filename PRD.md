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
- **Dual-Mode Scoring & Academic Suite:** Balances rapid engineering comparison via the logarithmic QuaComp Synthetic Index (QSI) while providing formal scientific benchmarks (kGates/s throughput, Quantum Volume verification, and Energy-Delay Product).
- **Scalable MPS Simulation:** Supports high-qubit simulation (up to 100+ qubits) specifically for low-to-moderate entanglement circuits using tensor network compression (Matrix Product State), overcoming conventional state-vector memory limits.
- **NISQ Synthetic Noise & Fidelity Profiling:** Evaluates hardware computational overhead under synthetic parameterized noise channels (thermal relaxation and depolarizing errors) while measuring classical Hellinger state fidelity loss.
- **Standard OpenQASM 2.0 Ingestion:** Consumes industry-standard OpenQASM 2.0 quantum assembly code without third-party compilation dependencies, enabling direct benchmarking of production algorithms from QASMBench, IBM Quantum, and academic literature.
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

### FR-5: Dual-Mode Scoring & Academic Verification Engine

QuaComp employs a **Dual-Mode Scoring Architecture** to serve both rapid hardware evaluation and peer-reviewed scientific benchmarking:

#### 1. QuaComp Synthetic Index (QSI) — Engineering Heuristic
For rapid CLI comparison and hardware tier classification, QuaComp calculates the **QuaComp Synthetic Index (QSI)**:

$$\text{QSI} = \text{round}\left( n_{\text{qubits}} \cdot \left[ w_1 + w_2 \cdot \log_{10}(\max(T, 1.0)) + w_3 \cdot \left(\frac{\text{Fidelity}}{100}\right) \right], 2 \right)$$

where:
- $n_{\text{qubits}}$ = Maximum successfully simulated qubits.
- $T = \frac{\text{Total Gates}}{\max(\mu_{\text{latency}}, 10^{-6})}$ = Gate throughput in gates/second.
- $w_1 = 100.0$ (State-space capacity base weight).
- $w_2 = 50.0$ (Throughput dynamic sensitivity factor).
- $w_3 = 25.0$ (Quantum state fidelity factor).

**QSI Performance Tiers:**
| Performance Tier | Score Range (Points) | Typical Hardware Capability |
| :--- | :--- | :--- |
| **Entry-Level** | < 2,500 | Up to ~10-12 Qubits |
| **Mid-Range** | 2,500 to 6,000 | ~14-22 Qubits |
| **High-Performance** | 6,000 to 10,000 | ~24-30 Qubits (Fast SIMD / GPU) |
| **Extreme Workstation** | ≥ 10,000 | 30+ Qubits / Advanced Tensor Networks |

#### 2. Formal Academic Verification Suite (Quantum HPC Standards)
> [!IMPORTANT]
> **Methodological Transparency Notice:**  
> The **QuaComp Synthetic Index (QSI)** is an engineering heuristic index designed for rapid hardware comparison. For formal peer-reviewed academic research, system architects must evaluate and cite the **Academic Verification Suite** metrics exported by QuaComp:
> 1. **Normalized Gate Throughput (kGates/s):** Thousands of quantum gates executed per second ($T / 1000$).
> 2. **Quantum Volume Verification ($h_{\text{prob}} > 2/3$):** Heavy Output Probability certification under Cross et al. (2019) with 2-sigma lower confidence bound testing.
> 3. **Energy-Delay Product (EDP):** Measured in Joule-seconds ($J \cdot s$) via direct Linux RAPL hardware counters or calibrated dynamic TDP modeling ($\text{Latency} \times \text{Energy}$).

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

### FR-14: Hardware Power & Energy Telemetry (EQO) & Sensor Transparency
- **Description:** The system must profile CPU/SoC energy consumption and calculate Energy per Quantum Operation (EQO) with full sensor origin transparency.
- **Specifications & Behavior:**
  - **Linux Hardware RAPL Counter:** Reads `/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj` with microjoule precision when read permissions are available.
  - **Dynamic TDP Integration Model:** On Windows, macOS, or unprivileged Linux environments, continuously samples CPU utilization $U(t)$ and integrates instantaneous power:
    $$P(t) = P_{\text{idle}} + \left(\frac{U(t)}{100}\right) \times (P_{\text{TDP}} - P_{\text{idle}})$$
  - **Sensor Source Transparency:** Tags all energy output metrics with explicit origin badges (`[Sensor: RAPL]` for hardware counters vs `[Sensor: TDP Estimate]` for dynamic mathematical models) across CLI displays and Markdown reports.
  - **Energy per Quantum Operation (EQO):** Computes $EQO = \text{Total Energy (Joules)} / \text{Gates Executed}$ in µJ/gate.

### FR-15: Native Kernel Acceleration & C++ Gate Fusion Engine (`--use-native-kernels`)
- **Description:** The system must support single-pass gate fusion and hardware acceleration kernels across C++ SIMD, Apple Metal, and NVIDIA CUDA with multi-tier graceful fallback.
- **Specifications & Behavior:**
  - **Single-Pass Gate Fusion:** Sequences of quantum operations targeting identical or adjacent qubits are mathematically consolidated into a single unitary matrix $U_{\text{fused}} = U_k \times \dots \times U_1$, reducing quantum circuit depth and simulation dispatch overhead.
  - **C++ SIMD Extension (`quacomp_cpp`):** Implemented in modern C++17 via Pybind11 with unrolled $2\times 2$ and $4\times 4$ complex matrix multiplications and Kronecker products.
  - **Apple Metal Shaders:** GPU compute pipeline with Metal Shading Language (`fusion.metal`) for acceleration on Apple Silicon.
  - **CUDA JIT Stub:** Extensible interface for NVIDIA GPU CUDA execution.
  - **Multi-Tier Graceful Fallback:** If GPU shaders or compiled C++ binaries are absent, execution seamlessly falls back to optimized NumPy/CPython routines without interruption or exceptions.

### FR-16: Real Physical QPU Noise Calibration & Analytical Kraus Generator (`--noise-profile`)
- **Description:** The system must ingest real hardware calibration profiles from production superconducting QPUs (such as IBM Quantum or Rigetti) and construct trace-preserving Kraus representation channels.
- **Specifications & Behavior:**
  - **Calibration Parser:** Ingests JSON calibration snapshots (e.g. `ibm_brisbane_sample.json`), extracting qubit thermal relaxation $T_1$, dephasing $T_2$, readout assignment error confusion matrices $P(\text{read}|\text{true})$, and gate error rates.
  - **Analytical Kraus Operators:** Generates exact trace-preserving Kraus operators $\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger$ ($\sum_k E_k^\dagger E_k = I$):
    - Amplitude damping with rate $\gamma = 1 - e^{-t/T_1}$.
    - Phase damping with rate $\lambda = 1 - e^{-2t/T_\phi}$.
    - Physical consistency bounds enforcing $T_2 \le 2T_1$ to avoid unphysical negative rates.
  - **Noise Model Conversion:** Converts calibration parameters into Qiskit Aer `NoiseModel` objects for live circuit execution and fidelity analysis.
 
### FR-17: OpenQASM 2.0 Parser & Standard Circuit Interoperability (`--qasm`)
- **Description:** The system must provide a standalone, zero-overhead OpenQASM 2.0 parser and circuit interoperability module to ingest industry-standard quantum assembly files without manual Python construction.
- **Specifications & Behavior:**
  - **Native Tokenizer & AST Grammar Engine:**
    - Parses register declarations: `qreg <name>[<size>];` and `creg <name>[<size>];`.
    - Skips headers (`OPENQASM 2.0;`), standard includes (`include "qelib1.inc";`), single-line comments (`//`), and barrier statements.
    - Full support for 1-qubit gates (`h`, `x`, `y`, `z`, `s`, `sdg`, `t`, `tdg`, `rx`, `ry`, `rz`, `u1`, `u2`, `u3`), 2-qubit gates (`cx`/`cnot`, `cz`, `cy`, `ch`, `swap`, `iswap`, `crz`, `cu1`, `cu3`, `rxx`, `ryy`, `rzz`), and 3-qubit gates (`ccx`/`toffoli`, `cswap`/`fredkin`).
    - Measurement routing: `measure q[i] -> c[j];` and state resets: `reset q[i];`.
    - Custom gate macro expansion: `gate <name>(<params>) <args> { <body> }`.
    - Whole-register broadcasting: automatically expands statements such as `h q;` into element-wise operations across register length.
  - **Safe Parameter Expression Evaluation:**
    - Safely evaluates mathematical angle expressions via AST parsing supporting numerical constants (`pi`, `e`), basic arithmetic operators, and standard functions (`sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`).
  - **Internal Circuit Data Structure (`ParsedQASMCircuit`):**
    - Tracks global linear qubit indices mapping `(reg_name, reg_idx)` to tensor basis statevector positions.
    - Pre-flight memory safety verification: checks that total declared qubits satisfy theoretical RAM bounds prior to simulation dispatch.
    - Bidirectional Qiskit interop: converts to `qiskit.QuantumCircuit` via `.to_quantum_circuit()` and imports existing circuits via `ParsedQASMCircuit.from_qiskit()`.
  - **CLI Integration & Telemetry:**
    - `--qasm <filepath>`: Executes custom simulation directly from an external `.qasm` file.
    - Displays dedicated Rich telemetry table detailing circuit name, depth, total gates, 1-qubit / 2-qubit breakdown, and parse latency.

### FR-18: NVIDIA CUDA Acceleration Backend (--backend cuda)
- **Description:** The system must provide a native NVIDIA CUDA acceleration backend (`CUDABackend`) with parity to Apple Metal, enabling high-performance double-precision complex statevector manipulation (`cuDoubleComplex` / `complex128`) and fused unitary gate execution.
- **Specifications & Behavior:**
  - **Native CUDA Kernels (`src/engine/cuda/fusion.cu`, `fusion.cuh`):**
    - 128-bit complex floating-point representation using `cuDoubleComplex` for 64-bit IEEE 754 precision.
    - 1-Qubit Gate Kernel (`apply_gate_1q_cuda`): Thread mapping over index pairs with coalesced global memory access and bit-twiddling basis indexing.
    - 2-Qubit Fused Gate Kernel (`apply_gate_2q_cuda`): Arbitrary 2-qubit unitary transformations over index quadruplets with bit-mask insertion at control and target qubit positions.
    - Optimized thread configuration: 256 threads per block with warp-level divergence mitigation.
  - **Modular Runtime Integration (`CUDABackend`):**
    - Dynamic device availability detection via `is_cuda_available()`.
    - Multi-tier CUDA execution priority:
      1. CuPy `RawKernel` JIT compilation and memory management.
      2. PyTorch CUDA stream tensors (`torch.cuda`).
      3. Vectorized NumPy CPU fallback.
    - Direct VRAM allocation (`allocate_statevector`) and zero-overhead host copy (`copy_to_host`).
  - **Hardware Priority Hierarchy (`get_best_backend`):**
    - Automatic engine selection order:
      CUDA GPU → Metal GPU → C++ Native SIMD Engine → Pure CPython / NumPy Fallback
  - **Telemetry & Monitoring (`get_cuda_telemetry`):**
    - Queries active CUDA GPU device name, compute capability, VRAM allocated and peak usage, and utilization via NVML / PyTorch / CuPy / `nvidia-smi`.
  - **CLI Integration & Diagnostics:**
    - Adds `--backend {auto,cuda,metal,cpp,numpy}` flag across `cli/main.py` and `cli/runner.py`.
    - Rich UI terminal displays active CUDA device metadata, compute capability, and initial memory allocation.
    - Non-blocking warning and graceful fallback when `--backend cuda` is requested on systems without NVIDIA hardware.

---

## 4. Technical Limitations & Architecture Transparency

### 4.1 Multi-GPU Model Parallelism (Python-Level Statevector Slicing)
- **Mechanism:** Multi-GPU statevector slicing in QuaComp is orchestrated at the Python process level via Qiskit Aer's chunk-based memory distribution backend (`batched_shots_gpu=True`, `blocking_enable=True`, `blocking_qubits`).
- **Capabilities:** Successfully aggregates VRAM across multiple GPUs (e.g., $2\times 16\text{ GB} = 32\text{ GB}$), enabling simulation of circuits with $n \ge 29$ qubits that exceed the VRAM of any single GPU.
- **Limitations:**
  1. *Inter-Device Overhead:* Gate operations applied across chunk boundaries require inter-GPU memory swaps across the PCIe or NVLink bus, introducing non-negligible data transfer latency.
  2. *No Bare-Metal Kernel Fusion:* It does not implement custom CUDA/C++ kernel fusion or low-level MPI cluster synchronization across distributed multi-node server clusters.
  3. *Host-to-Device Memory Staging:* Overall execution speed is bounded by host-to-device interconnect bandwidth during chunk redistribution.

### 4.2 Linux RAPL Hardware Interface Access Permissions & Fallback
- **Mechanism:** High-precision hardware power telemetry on Linux utilizes the Running Average Power Limit (RAPL) driver via the sysfs interface (`/sys/class/powercap/intel-rapl`).
- **Permission Requirements:** Linux security hardening restricts reading `/sys/class/powercap` to privileged users (root/sudo) or users granted read access (`chmod +r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj` or specialized capability flags).
- **Graceful Fallback:** If read permission is denied or the host is running Windows/macOS, QuaComp automatically and silently falls back to the dynamic TDP mathematical model without interrupting benchmark execution.
- **Full Transparency:** CLI terminal output and Markdown reports prominently display `[Sensor: TDP Estimate]` to inform users that power values are model-derived rather than direct physical hardware counter readings.

### 4.3 Hardware Parity Matrix & Multi-Tier Graceful Fallback Strategy

| Hardware Platform | Accelerator Engine | Runtime Technology | Supported OS | Fallback Hierarchy |
| :--- | :--- | :--- | :--- | :--- |
| Apple Silicon (M1/M2/M3/M4) | Metal GPU Compute | Apple Metal Shaders (fusion.metal) | macOS | C++ SIMD → CPython / NumPy |
| NVIDIA GPU (RTX / A100 / H100) | CUDA GPU Acceleration | CUDA Kernels (fusion.cu) & CuPy/PyTorch | Linux, Windows | C++ SIMD → CPython / NumPy |
| x86_64 / ARM64 CPU (Modern) | C++ Native SIMD | Pybind11 Unrolled Matrix Fusion | Linux, macOS, Windows | CPython / NumPy |
| Any Generic CPU | Pure CPython / NumPy | Vectorized NumPy Fallback Engine | All Platforms | Built-in Base Level |

- **Tier 1 (GPU Hardware Compute):** NVIDIA CUDA Kernels (Linux/Windows) and Apple Metal Shaders (macOS).
- **Tier 2 (C++ SIMD Native Extension):** Unrolled complex matrix arithmetic compiled via Pybind11 with OpenMP/AVX.
- **Tier 3 (Pure CPython / NumPy Fallback):** Vectorized Python fallback ensuring 100% execution guarantees on any host without compilation tools.
- **Optional Build Strategy:** Setuptools `BuildExtOptional` ensures `pip install -e .` never fails even on bare systems lacking MSVC/GCC/Clang compilers.


