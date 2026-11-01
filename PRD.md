# Product Requirement Document (PRD)
# Project Name: QuaComp (Quantum Computer Simulation Benchmark)

**Version:** 1.0.0  
**Status:** Approved / Completed  
**Target Environment:** Cross-platform (Windows, macOS, Linux)  
**Primary Tech Stack:** Python 3.10+, Qiskit / Aer, psutil, Rich, Matplotlib, Seaborn, Setuptools, Pytest, GitHub Actions  

---

## 1. Executive Summary & Vision

### 1.1 Overview
`QuaComp` is an open-source **Multi-Paradigm Quantum-Classical Emulation Engine & Hardware Roofline Profiler** designed to evaluate local and workstation hardware performance (CPU, NVIDIA CUDA, Apple Metal, and C++ SIMD) when simulating quantum computation across three distinct mathematical paradigms:

1. **Dense Statevector Engine ($\mathcal{O}(2^n)$):** Exact full-amplitude simulation in double precision (`fp64` / `complex128`) or single precision (`fp32` / `complex64`, 50% memory reduction) up to physical memory limits ($\sim 28\text{--}30$ qubits).
2. **Matrix Product State / MPS Tensor Network Engine ($\mathcal{O}(n \cdot \chi^3)$):** Factorized 1D tensor train with automated SWAP network routing for non-local 2-qubit gates and cumulative SVD truncation fidelity tracking for circuits beyond 30 qubits.
3. **Stabilizer Tableau Engine ($\mathcal{O}(n^2)$):** Binary symplectic implementation of the Aaronson-Gottesman CHP algorithm capable of simulating **100 to 1,000+ maximally entangled qubits** (Clifford circuits and noisy Pauli frame Monte Carlo trajectories) in milliseconds.

In addition, QuaComp incorporates a static **Circuit Complexity Profiler & Auto-Solver Router (`--sim-method auto`)** that analyzes entanglement topology versus non-Clifford magic density before execution, an **In-Place Multi-Paradigm Pauli Expectation & Bipartite Entanglement Engine**, and a **Solver-Aware Roofline Memory Bandwidth Profiler** (`GB/s` and `FLOPs/Byte`).

### 1.2 Core Value Proposition
- **Solver-Aware Pre-flight Safety:** Prevents Out-Of-Memory (OOM) crashes by enforcing an 85% RAM/VRAM safety budget for $\mathcal{O}(2^n)$ statevectors while allowing 100 to 1,000+ qubit circuits to execute smoothly on $\mathcal{O}(n^2)$ Stabilizer and $\mathcal{O}(n \cdot \chi^2)$ MPS engines.
- **Automated Complexity Routing:** Inspects quantum circuits prior to execution—measuring Clifford vs. non-Clifford gate ratios (Quantum Magic) and interaction distances—to automatically select `stabilizer`, `mps`, or `statevector`.
- **Multi-Paradigm Physical Observables:** Evaluates 1D transverse-field Ising energy expectations ($\langle \hat{H}_{\text{Ising}} \rangle$), end-to-end spin correlations ($\langle Z_0 Z_{n-1} \rangle$), and bipartite Von Neumann entanglement entropy ($S(\rho_A)$ in *ebits*) in-place across all three solvers.
- **Solver-Aware Roofline & Mixed-Precision Telemetry:** Measures effective memory bandwidth (`GB/s`), arithmetic intensity (`FLOPs/Byte`), and numerical drift between `fp64` and `fp32` representations.
- **Stochastic Noise Beyond the Density Matrix Wall:** Combines $\mathcal{O}(2^n)$ Monte Carlo Wavefunction (MCWF) jump trajectories for general circuits with $\mathcal{O}(n^2)$ Pauli frame propagation for noisy 100+ qubit Clifford circuits.
- **Dual-Mode Scoring & Academic Suite:** Balances rapid engineering comparison via the logarithmic QuaComp Synthetic Index (QSI) with peer-reviewed HPC metrics (kGates/s, Effective Memory Bandwidth, Quantum Volume, and Energy-Delay Product).

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

### FR-19: Monte Carlo Wavefunction (Quantum Trajectories) Noise Simulation (`--noise-method trajectory`)
- **Description:** The system must provide a Monte Carlo Wavefunction (MCWF) / Quantum Trajectories noise simulation algorithm maintaining pure statevector $\mathcal{O}(2^n)$ memory scaling, enabling physical dephasing and relaxation noise simulations up to 28 qubits on standard 16 GB memory machines and eliminating the $\mathcal{O}(4^n)$ density matrix memory wall.
- **Specifications & Behavior:**
  - **Stochastic Quantum Jump Sampling:**
    - Replaces exponential density matrix operations by stochastically sampling quantum jump trajectories directly on the statevector.
    - Single-Qubit Depolarizing Channel: With probability $1 - p$, gate proceeds error-free ($I$); with probability $p/3$, applies a stochastic Pauli jump ($X, Y, Z$).
    - Two-Qubit Depolarizing Channel: With probability $1 - p$, proceeds error-free ($II$); with probability $p/15$, applies one of the 15 non-identity 2-qubit Pauli operator pairs.
    - Thermal Relaxation ($T_1, T_2$):
      - Amplitude damping (energy relaxation): Evaluates spontaneous emission jump probability $p_{\text{jump}} = \gamma \cdot p_1$ where $\gamma = 1 - e^{-t_g / T_1}$ and $p_1 = \sum_{\text{target}=1} |\psi|^2$.
        - Jump branch: Applies $K_1 = \sqrt{\gamma}|0\rangle\langle 1|$ (collapses $|1\rangle \to |0\rangle$) followed by statevector renormalization.
        - No-jump drift branch: Applies non-unitary Kraus operator $K_0 = \text{diag}(1, \sqrt{1 - \gamma})$ followed by statevector renormalization, preserving exact open quantum system coherence and $T_2$ dynamics.
      - Pure dephasing: Applies stochastic phase flip ($Z$) with probability $p_{\text{phase}} = \frac{1 - e^{-t_g / T_\phi}}{2}$ where $1 / T_\phi = \max(0, 1/T_2 - 1/(2T_1))$.
  - **Parallelized Trajectory Execution:**
    - Automatically distributes trajectory shots across worker pools using `ProcessPoolExecutor` when $N_{\text{shots}} \ge 100$ and `num_workers > 1`, utilizing orthogonal `SeedSequence.spawn()` random streams to prevent inter-trajectory correlation. Single-process execution is maintained on CUDA GPU backends to prevent multi-processing driver context collisions.
  - **Stochastic Classical Readout Bit-Flip:**
    - Injects classical measurement bit-flips according to calibrated hardware readout confusion probabilities $P(0\vert{}1)$ and $P(1\vert{}0)$.
  - **Memory Scaling & Safety Guard:**
    - Guarantees strict $\mathcal{O}(2^n)$ memory allocation at all compute stages.
    - Pre-flight Density Matrix Safety Guard: If explicit `--noise-method density_matrix` is requested, automatically inspects qubit count and rejects execution for $n \ge 16$ qubits with `DensityMatrixMemoryError` to prevent operating system freezes.
    - Pre-flight GPU VRAM 90% Safety Headroom: Enforces a 90% safety threshold on available VRAM to safeguard against driver memory fragmentation, with defensive exception handling automatically catching CUDA OOM runtime exceptions and falling back smoothly to CPU/NumPy.
  - **CLI Integration & Telemetry:**
    - Adds `--noise-method {trajectory,density_matrix}` (default: `trajectory`).
    - Adds `--shots <int>` (default: `1000`).
    - Runner displays noise simulation mode, shot count, and memory savings ratio:

      $$\text{Memory Saved Ratio} \approx 2^n \times$$

### FR-20: MPS SWAP Network Routing & SVD Truncation Fidelity (`--mps`, `--max-bond-dim`, `--svd-cutoff`)
- **Description:** The native Matrix Product State (`MatrixProductState`) engine in `src/engine/mps.py` must support arbitrary non-adjacent 2-qubit gates ($|i - j| > 1$) from OpenQASM 2.0 and algorithmic circuits while tracking cumulative singular value truncation error and overall tensor-network fidelity.
- **Specifications & Behavior:**
  - **Automated SWAP Network Routing:** Decomposes long-range 2-qubit gates acting on sites $(i, j)$ with $|i - j| > 1$ into a bridal chain of nearest-neighbor SWAP contractions to bring the qubits adjacent, applies the target 2-qubit unitary, and reverses the SWAP chain to restore canonical site ordering while incrementing `swap_count`.
  - **SVD Truncation Fidelity Tracking:** During each 2-qubit bond SVD split capped at `max_bond_dim` ($\chi_{\max}$) and `svd_cutoff`, calculates the discarded spectral weight $\epsilon_{\text{step}} = \sum_{k > \chi_{\max}} \lambda_k^2 / \sum_{\text{all}} \lambda_k^2$ and updates multiplicative fidelity:

    $$F_{\text{MPS}} = \prod_{\text{steps}} (1 - \epsilon_{\text{step}})$$

  - **Pre-Execution 1Q Gate Contraction:** Consecutive 1-qubit gates acting on the same wire in `simulate_trajectories` are fused into a single $2 \times 2$ unitary matrix prior to trajectory sampling, tracking `logical_gate_count`, `executed_gate_count`, and `gate_fusion_compression_ratio`.

### FR-21: Solver-Aware Roofline Memory Bandwidth & Mixed-Precision Emulation (`--precision {fp64,fp32}`)
- **Description:** The telemetry profiler (`src/profiler/telemetry.py`) and scoring engine (`src/scorer/calculator.py`) must compute solver-aware and precision-aware Effective Memory Bandwidth (`GB/s`) and Arithmetic Intensity (`FLOPs/Byte`), while supporting mixed-precision (`fp32`/`complex64` vs `fp64`/`complex128`) statevector and MPS execution.
- **Specifications & Behavior:**
  - **Solver-Aware Byte Traffic Formulas:**
    - `statevector`: $\text{Bytes} = G \times 2^n \times (2 \times B_{\text{elem}})$, where $B_{\text{elem}} = 16\text{ B}$ (`fp64`) or $8\text{ B}$ (`fp32`), with Arithmetic Intensity $0.4375\text{ FLOPs/Byte}$ (`fp64`) or $0.875\text{ FLOPs/Byte}$ (`fp32`).
    - `mps`: $\text{Bytes} = (G + 3 \cdot N_{\text{swaps}}) \times 4 \cdot \chi_{\text{active}}^2 \times 2 \cdot B_{\text{elem}}$, with Arithmetic Intensity $2.0\text{ FLOPs/Byte}$.
    - `stabilizer`: $\text{Bytes} = G \times 2n \times \lceil (2n + 1) / 8 \rceil \times 2$, with Arithmetic Intensity $0.25\text{ FLOPs/Byte}$, preventing exponential $\mathcal{O}(2^n)$ overflow on 100+ qubit Clifford circuits.
  - **Mixed-Precision Drift Analysis:** Provides `measure_precision_drift()` to quantify the L2 norm divergence $\|\psi_{\text{fp64}} - \psi_{\text{fp32}}\|_2$ and fidelity loss between double-precision and single-precision emulation.

### FR-22: Aaronson-Gottesman Stabilizer Tableau Engine ($\mathcal{O}(n^2)$) & Noisy Pauli Frame Trajectories (`--stabilizer`)
- **Description:** The system must provide a polynomial-memory $\mathcal{O}(n^2)$ Stabilizer Tableau engine (`src/engine/stabilizer.py`) based on the Gottesman-Knill theorem and the Aaronson-Gottesman (2004) CHP algorithm, supporting both ideal Clifford simulation and multi-shot noisy Pauli frame trajectories for 100 to 1,000+ qubits.
- **Specifications & Behavior:**
  - **Binary Symplectic Tableau (`StabilizerTableau`):** Maintains a $2n \times (2n + 1)$ boolean matrix $(X \mid Z \mid r)$ tracking $n$ destabilizer generators and $n$ stabilizer generators in $\mathcal{O}(n^2)$ memory ($< 10\text{ KB}$ for 100 qubits).
  - **Exact Clifford Gate Set:** Supports $\mathcal{O}(n)$ column updates for $H, S, S^\dagger, X, Y, Z, I, SX, \text{CNOT}(c, t), \text{CZ}(c, t), \text{CY}(c, t), \text{SWAP}(a, b)$ and $\mathcal{O}(n^2)$ projective computational-basis measurements via symplectic `rowsum`.
  - **Non-Clifford Rejection Guard:** Raises `NonCliffordGateError` with diagnostic guidance when non-Clifford gates ($T, T^\dagger, \text{CCX}$, or non-multiples of $\pi/2$ rotations) are encountered.
  - **Noisy Stabilizer Trajectories (`simulate_noisy_stabilizer_trajectories`):** Injects stochastic 1-qubit and 2-qubit depolarizing Pauli errors ($X, Y, Z$) after each Clifford gate and readout bit-flips across $N_{\text{shots}}$ trajectories, computing ensemble fidelity and entanglement retention at 100+ qubits in $\mathcal{O}(N_{\text{shots}} \cdot G \cdot n)$ time.

### FR-23: Circuit Complexity Profiler & Auto-Solver Router (`--sim-method auto`)
- **Description:** The system must provide a static pre-execution circuit complexity analyzer (`src/engine/complexity.py`) that profiles entanglement topology against non-Clifford magic density and automatically routes execution to the optimal simulation engine.
- **Specifications & Behavior:**
  - **Two-Axis Complexity Classification (`analyze_circuit_complexity`):**
    - Axis 1 (Quantum Magic): Counts `clifford_gates`, `non_clifford_gates`, and `magic_density`. Rotations $R_x(\theta), R_y(\theta), R_z(\theta), U_1(\theta)$ whose angles are integer multiples of $\pi/2$ are recognized as Clifford operations.
    - Axis 2 (Entanglement Topology): Computes `two_qubit_gate_density`, `max_interaction_distance` ($\max |i - j|$), and classifies topology as `unentangled`, `1D_local`, `moderate_entangled`, or `volume_law`.
  - **Deterministic Routing Rules (`select_optimal_solver`):**
    1. If `non_clifford_gates == 0`, routes to `stabilizer` ($\mathcal{O}(n^2)$ exact).
    2. If `num_qubits > 24` or estimated statevector RAM exceeds 80% of available physical RAM, or if `num_qubits >= 16` with `1D_local`/`unentangled` topology, routes to `mps`.
    3. Otherwise (`num_qubits <= 24` with non-Clifford gates), routes to `statevector`.

### FR-24: Multi-Paradigm Pauli Expectation ($\langle \hat{H} \rangle$) & Scalable Bipartite Entanglement Entropy
- **Description:** The system must compute in-place expectation values for arbitrary Pauli strings $P \in \{I, X, Y, Z\}^{\otimes n}$, linear Hamiltonians $\hat{H} = \sum_k c_k P_k$ (`src/engine/observables.py`), and bipartite Von Neumann entanglement entropy (`src/engine/entanglement.py`) across all three solvers without converting compressed representations into $2^n$ statevectors.
- **Specifications & Behavior:**
  - **In-Place Pauli & Hamiltonian Evaluation:**
    - `statevector`: Bitwise index permutation and parity popcount in $\mathcal{O}(2^n)$ time with zero matrix allocation.
    - `mps`: Local rank-2 transfer-matrix contraction from site $0$ to $n-1$ in $\mathcal{O}(n \cdot \chi^3)$ time.
    - `stabilizer`: Symplectic anticommutation check against stabilizer generators and destabilizer row reduction in $\mathcal{O}(n^2)$ time, returning exact expectations in $\{-1.0, 0.0, +1.0\}$.
  - **Scalable Bipartite Entanglement Entropy ($S(\rho_A)$):**
    - Computed via central bond singular values $\lambda_i$ in `mps` ($\mathcal{O}(\chi^3)$) and via $\text{GF}(2)$ binary Gaussian elimination rank of the cross-partition stabilizer commutator matrix in `stabilizer` ($\mathcal{O}(n^3)$), enabling exact entanglement verification on 100+ qubit states.

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

### 4.4 Noise Simulation Memory Scaling: Monte Carlo Wavefunction vs. Density Matrix

Simulating open quantum systems with physical noise via density matrices ($\rho$) scales quadratically in memory as $\mathcal{O}(4^n) = \mathcal{O}(2^{2n})$. In contrast, QuaComp's Monte Carlo Wavefunction (Quantum Trajectories) preserves pure statevector scaling of $\mathcal{O}(2^n)$, delivering exponential memory savings:

| Qubits (n) | Density Matrix Memory O(2^(2n)) | MCWF Trajectory Memory O(2^n) | Memory Savings Ratio | 16 GB RAM Feasibility |
| :--- | :--- | :--- | :--- | :--- |
| 10 | 16 MB | 16 KB | 1,024× | Both Supported |
| 14 | 4 GB | 256 KB | 16,384× | Density Matrix Limit (16 GB RAM) |
| 16 | 64 GB | 1 MB | 65,536× | Trajectory Only (DM Out of Memory) |
| 20 | 16 TB (16,384 GB) | 16 MB | 1,048,576× | Trajectory Only |
| 24 | 4.3 PB | 256 MB | 16,777,216× | Trajectory Only |
| 28 | 1.1 EB (Exabytes) | 4.0 GB | 268,435,456× | Trajectory Safe on 16 GB RAM |
| 29 | 4.4 EB | 8.0 GB | 536,870,912× | Trajectory Safe on 16 GB RAM |

- **Industrial Significance:** On a standard workstation with 16 GB RAM, density matrix simulations hit an insurmountable memory barrier at 14 qubits. QuaComp's MCWF trajectory engine breaks this barrier, allowing noisy circuit evaluation up to 28 qubits on the same machine.



