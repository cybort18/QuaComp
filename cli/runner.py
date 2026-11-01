from typing import List, Dict, Any, Optional
import time
import psutil
import numpy as np
from rich.console import Console
from rich.status import Status

from src.profiler.memory import check_memory_safety
from src.profiler.telemetry import get_cpu_utilization
from src.engine.circuits import (
    generate_shallow_circuit,
    generate_deep_circuit,
    generate_qft_circuit,
    generate_vqe_circuit,
    profile_parameter_binding,
    generate_qaoa_circuit,
    generate_quantum_volume_circuit,
    calculate_heavy_output_probability
)
from src.engine.simulator import run_simulation
from src.engine.mps import calculate_mps_ram_savings
from src.engine.noise import get_noise_model, calculate_state_fidelity, calculate_overhead_ratio
from src.engine.entanglement import calculate_bipartite_entropy
from src.engine.accelerator import get_best_backend, InsufficientVRAMError

default_console = Console()

def run_single_simulation(
    qubits: int,
    workload_type: str,
    depth: int,
    method: str = 'statevector',
    bond_dimension: int = 64,
    device: str = 'cpu',
    noise_level: str = 'none',
    runs: int = 3,
    workers: int = 1,
    compute_entropy: bool = False,
    state_slicing: bool = False,
    blocking_qubits: Optional[int] = None,
    use_native_kernels: bool = False,
    noise_profile: Optional[str] = None,
    qasm_circuit: Optional[Any] = None,
    qasm_metrics: Optional[Dict[str, Any]] = None,
    backend: str = 'auto',
    shots: int = 1000,
    noise_method: str = 'trajectory',
    svd_cutoff: float = 1e-9,
    precision: str = 'fp64'
) -> Dict[str, Any]:

    """
    Execute a single quantum simulation workload with profiling.

    Returns:
        dict: Simulation results and execution metrics.
    """
    # 1. Pre-flight memory safety check (deferred if method == 'auto' until solver is selected)
    requested_method = method
    is_safe = True
    msg = "SAFE"
    if method != "auto":
        if qasm_circuit is not None and hasattr(qasm_circuit, "num_qubits"):
            qubits = qasm_circuit.num_qubits
        is_safe, msg = check_memory_safety(
            qubits,
            method=method,
            bond_dimension=bond_dimension,
            device=device,
            state_slicing=state_slicing
        )
        if not is_safe:
            return {
                "qubits": qubits,
                "method": method,
                "bond_dimension": bond_dimension if method in ('mps', 'matrix_product_state') else None,
                "device": device.upper(),
                "workers": workers,
                "noise_level": noise_level,
                "model_parallelism": "Aggregated VRAM Slicing" if state_slicing else "None",
                "chunk_count": 1,
                "fidelity": 0.0,
                "overhead_ratio": 0.0,
                "success": False,
                "latency": 0.0,
                "mean_latency": 0.0,
                "median_latency": 0.0,
                "std_latency": 0.0,
                "latencies": [],
                "runs_count": 0,
                "gates": 0,
                "cpu_usage": 0.0,
                "ram_status": "UNSAFE",
                "error": msg,
                "ram_savings": {},
                "entanglement_metrics": {},
                "parameter_binding_metrics": {},
                "qv_metrics": {},
                "energy_metrics": {},
                "complexity_profile": {},
                "solver_routing": {}
            }

    # 2. Circuit generation
    circuit = None
    param_binding_metrics = {}
    qv_metrics = {}

    if qasm_circuit is not None:
        circuit = qasm_circuit.to_quantum_circuit() if hasattr(qasm_circuit, 'to_quantum_circuit') else qasm_circuit
        qubits = circuit.num_qubits
    elif workload_type == "shallow":
        circuit = generate_shallow_circuit(qubits)
    elif workload_type == "deep":
        circuit = generate_deep_circuit(qubits, depth)
    elif workload_type == "qft":
        circuit = generate_qft_circuit(qubits)
    elif workload_type == "vqe":
        vqe_depth = max(1, depth // 5) if depth > 5 else depth
        raw_vqe = generate_vqe_circuit(qubits, depth=vqe_depth)
        param_binding_metrics = profile_parameter_binding(raw_vqe)
        rng_vals = [float(v) for v in np.linspace(0.1, 3.14, raw_vqe.num_parameters)]
        circuit = raw_vqe.assign_parameters(rng_vals)
    elif workload_type == "qaoa":
        qaoa_steps = max(1, depth // 10) if depth > 10 else 1
        raw_qaoa = generate_qaoa_circuit(qubits, p_steps=qaoa_steps)
        param_binding_metrics = profile_parameter_binding(raw_qaoa)
        rng_vals = [float(v) for v in np.linspace(0.1, 1.57, raw_qaoa.num_parameters)]
        circuit = raw_qaoa.assign_parameters(rng_vals)
    elif workload_type == "qv":
        qv_qubits = max(2, qubits)
        circuit = generate_quantum_volume_circuit(qv_qubits, depth=depth if depth is not None else qv_qubits)
    else:
        circuit = generate_qft_circuit(qubits)

    num_gates = sum(circuit.count_ops().values())

    # 2b. Static Circuit Complexity Analysis & Auto-Solver Routing
    from src.engine.complexity import analyze_circuit_complexity, select_optimal_solver
    target_for_complexity = qasm_circuit if qasm_circuit is not None else circuit
    complexity_profile = analyze_circuit_complexity(target_for_complexity)
    solver_routing = select_optimal_solver(complexity_profile, requested_method=requested_method)
    if requested_method == "auto":
        method = solver_routing["selected_solver"]
        is_safe, msg = check_memory_safety(
            qubits,
            method=method,
            bond_dimension=bond_dimension,
            device=device,
            state_slicing=state_slicing
        )
        if not is_safe:
            return {
                "qubits": qubits,
                "method": method,
                "bond_dimension": bond_dimension if method in ('mps', 'matrix_product_state') else None,
                "device": device.upper(),
                "workers": workers,
                "noise_level": noise_level,
                "model_parallelism": "Aggregated VRAM Slicing" if state_slicing else "None",
                "chunk_count": 1,
                "fidelity": 0.0,
                "overhead_ratio": 0.0,
                "success": False,
                "latency": 0.0,
                "mean_latency": 0.0,
                "median_latency": 0.0,
                "std_latency": 0.0,
                "latencies": [],
                "runs_count": 0,
                "gates": 0,
                "cpu_usage": 0.0,
                "ram_status": "UNSAFE",
                "error": msg,
                "ram_savings": {},
                "entanglement_metrics": {},
                "parameter_binding_metrics": {},
                "qv_metrics": {},
                "energy_metrics": {},
                "complexity_profile": complexity_profile,
                "solver_routing": solver_routing
            }

    # 3. Energy and CPU measurement start
    get_cpu_utilization()

    # 4. Simulation run with Energy Profiler, noise, multi-GPU/device, workers, and method support (multiple runs)
    from src.profiler.energy import EnergyProfiler
    noise_model = None
    if noise_profile:
        from src.engine.physical_noise import PhysicalNoiseModel
        pnm = PhysicalNoiseModel.from_json(noise_profile)
        noise_model = pnm.to_qiskit_noise_model(active_qubits=list(range(circuit.num_qubits)))
        if noise_level == 'none':
            noise_level = f"physical:{pnm.backend_name}"
    else:
        noise_model = get_noise_model(noise_level)

    # 4. Device synchronization and simulation execution with pre-flight VRAM safety
    active_be = get_best_backend(backend)
    active_be.device_synchronize()

    if method in ('mps', 'matrix_product_state'):
        from src.engine.mps import simulate_mps
        mps_latencies = []
        last_mps_res = None
        with EnergyProfiler() as energy_prof:
            for _ in range(runs):
                mps_res = simulate_mps(
                    circuit,
                    bond_dimension=bond_dimension,
                    cutoff=svd_cutoff,
                    shots=shots,
                    precision=precision
                )
                mps_latencies.append(mps_res["latency"])
                last_mps_res = mps_res

        mean_lat = float(np.mean(mps_latencies))
        med_lat = float(np.median(mps_latencies))
        std_lat = float(np.std(mps_latencies, ddof=1)) if len(mps_latencies) > 1 else 0.0

        sim_result = {
            "success": True,
            "latency": mean_lat,
            "mean_latency": mean_lat,
            "median_latency": med_lat,
            "std_latency": std_lat,
            "latencies": mps_latencies,
            "runs_count": len(mps_latencies),
            "counts": last_mps_res["counts"],
            "device": device.upper(),
            "workers": workers,
            "error": None,
            "metadata": {
                "backend_name": "QuaComp MPS Tensor Engine",
                "backend_version": "1.0.0",
                "method": "mps",
                "bond_dimension": bond_dimension,
                "cutoff": svd_cutoff,
                "shots": shots,
                "counts": last_mps_res["counts"]
            },
            "native_kernel_used": False,
            "accelerator_backend": "CPU (MPS Tensor Network)",
            "accelerator_badge": None,
            "fusion_metrics": None,
            "physical_noise_profile": None,
            "mps_fidelity": last_mps_res["mps_fidelity"],
            "cumulative_truncation_error": last_mps_res["cumulative_truncation_error"],
            "max_active_bond_dim": last_mps_res["max_active_bond_dim"],
            "swap_count": last_mps_res["swap_count"],
            "bipartite_entropy_ebits": last_mps_res.get("bipartite_entropy_ebits", 0.0),
            "bipartite_entanglement_entropy_ebits": last_mps_res.get("bipartite_entropy_ebits", 0.0),
            "end_to_end_zz_expectation": last_mps_res.get("end_to_end_zz_expectation", 0.0),
            "zz_correlation_expectation": last_mps_res.get("end_to_end_zz_expectation", 0.0),
            "ising_energy_expectation": last_mps_res.get("ising_energy_expectation", 0.0)
        }
    elif method == 'stabilizer':
        from src.engine.stabilizer import simulate_stabilizer
        stab_latencies = []
        last_stab_res = None
        stab_input = qasm_circuit if qasm_circuit is not None else circuit
        with EnergyProfiler() as energy_prof:
            for _ in range(runs):
                stab_res = simulate_stabilizer(
                    stab_input,
                    shots=shots,
                    noise_level=noise_level,
                    noise_model=noise_model
                )
                stab_latencies.append(stab_res["latency"])
                last_stab_res = stab_res

        mean_lat = float(np.mean(stab_latencies))
        med_lat = float(np.median(stab_latencies))
        std_lat = float(np.std(stab_latencies, ddof=1)) if len(stab_latencies) > 1 else 0.0

        sim_result = {
            "success": True,
            "latency": mean_lat,
            "mean_latency": mean_lat,
            "median_latency": med_lat,
            "std_latency": std_lat,
            "latencies": stab_latencies,
            "runs_count": len(stab_latencies),
            "counts": last_stab_res["counts"],
            "device": "CPU",
            "workers": workers,
            "error": None,
            "metadata": {
                "backend_name": "QuaComp Stabilizer Tableau Engine (Aaronson-Gottesman CHP)",
                "backend_version": "1.0.0",
                "method": "stabilizer",
                "shots": shots,
                "counts": last_stab_res["counts"],
                "tableau_memory_bytes": last_stab_res["tableau_memory_bytes"],
                "clifford_gate_count": last_stab_res["clifford_gate_count"]
            },
            "native_kernel_used": False,
            "accelerator_backend": "CPU (Aaronson-Gottesman CHP Tableau)",
            "accelerator_badge": "[CHP O(n^2)]",
            "fusion_metrics": None,
            "physical_noise_profile": None,
            "tableau_memory_bytes": last_stab_res["tableau_memory_bytes"],
            "clifford_gate_count": last_stab_res["clifford_gate_count"],
            "ram_savings_vs_statevector": last_stab_res["ram_savings_vs_statevector"],
            "bipartite_entropy_ebits": last_stab_res.get("bipartite_entropy_ebits", 0.0),
            "bipartite_entanglement_entropy_ebits": last_stab_res.get("bipartite_entropy_ebits", 0.0),
            "end_to_end_zz_expectation": last_stab_res.get("end_to_end_zz_expectation", 0.0),
            "zz_correlation_expectation": last_stab_res.get("end_to_end_zz_expectation", 0.0),
            "ising_energy_expectation": last_stab_res.get("ising_energy_expectation", 0.0),
            "noisy_fidelity_estimate": last_stab_res.get("noisy_fidelity_estimate", 1.0),
            "pauli_jump_count": last_stab_res.get("pauli_jump_count", 0)
        }
    else:
        try:
            with EnergyProfiler() as energy_prof:
                sim_result = run_simulation(
                    circuit,
                    method=method,
                    bond_dimension=bond_dimension,
                    device=device,
                    noise_model=noise_model,
                    noise_level=noise_level,
                    shots=shots,
                    runs=runs,
                    workers=workers,
                    state_slicing=state_slicing,
                    blocking_qubits=blocking_qubits,
                    use_native_kernels=use_native_kernels,
                    physical_noise_profile=noise_profile,
                    backend=backend,
                    noise_method=noise_method,
                    precision=precision
                )
        except (InsufficientVRAMError, MemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "cuda out of memory" not in str(e).lower() and "out of memory" not in str(e).lower():
                raise
            default_console.print(
                f"[bold yellow][WARNING] VRAM Limit Exceeded / CUDA OOM:[/] {e}\n"
                f"[bold cyan][INFO] Automatically falling back to CPU / NumPy execution...[/]"
            )
            with EnergyProfiler() as energy_prof:
                sim_result = run_simulation(
                    circuit,
                    method=method,
                    bond_dimension=bond_dimension,
                    device="cpu",
                    noise_model=noise_model,
                    noise_level=noise_level,
                    shots=shots,
                    runs=runs,
                    workers=workers,
                    state_slicing=False,
                    blocking_qubits=None,
                    use_native_kernels=False,
                    physical_noise_profile=noise_profile,
                    backend="numpy",
                    noise_method=noise_method,
                    precision=precision
                )
        finally:
            active_be.device_synchronize()

    energy_metrics = energy_prof.get_metrics(num_gates=num_gates)

    # 5. CPU measurement end
    cpu_metric = get_cpu_utilization()
    cpu_usage = float(cpu_metric.get("overall_percent", 0.0)) if isinstance(cpu_metric, dict) else float(cpu_metric)

    # 6. Calculate MPS / Stabilizer RAM savings if applicable
    ram_savings = {}
    if method in ('mps', 'matrix_product_state') and sim_result["success"]:
        process = psutil.Process()
        actual_ram_bytes = process.memory_info().rss
        ram_savings = calculate_mps_ram_savings(qubits, actual_ram_bytes)
    elif method == 'stabilizer' and sim_result["success"]:
        ram_savings = sim_result.get("ram_savings_vs_statevector", {})

    # 7. Calculate NISQ metrics (Fidelity and CPU Overhead) if noisy
    fidelity = 100.0
    overhead_ratio = 0.0
    if method in ('mps', 'matrix_product_state'):
        fidelity = round(sim_result.get("mps_fidelity", 1.0) * 100.0, 4)
    elif method == 'stabilizer':
        fidelity = round(sim_result.get("noisy_fidelity_estimate", 1.0) * 100.0, 4)
    elif noise_level != "none" and sim_result["success"]:
        active_be.device_synchronize()
        try:
            ideal_sim_result = run_simulation(
                circuit,
                method=method,
                bond_dimension=bond_dimension,
                device=device,
                noise_model=None,
                noise_level="none",
                shots=shots,
                runs=1,
                workers=workers,
                state_slicing=state_slicing,
                blocking_qubits=blocking_qubits,
                backend=backend,
                noise_method="trajectory"
            )
        except (InsufficientVRAMError, MemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "cuda out of memory" not in str(e).lower() and "out of memory" not in str(e).lower():
                raise
            ideal_sim_result = run_simulation(
                circuit,
                method=method,
                bond_dimension=bond_dimension,
                device="cpu",
                noise_model=None,
                noise_level="none",
                shots=shots,
                runs=1,
                workers=workers,
                state_slicing=False,
                blocking_qubits=None,
                backend="numpy",
                noise_method="trajectory"
            )
        active_be.device_synchronize()

        if ideal_sim_result["success"]:
            fidelity = calculate_state_fidelity(ideal_sim_result["counts"], sim_result["counts"])
            overhead_ratio = calculate_overhead_ratio(ideal_sim_result["latency"], sim_result["mean_latency"])

    # 8. Calculate Entanglement Entropy Metrics if requested (Supports Native MPS & Stabilizer for large qubits)
    entanglement_metrics = {}
    if compute_entropy and sim_result["success"] and circuit is not None:
        try:
            if method == "stabilizer" or sim_result.get("method_used") == "stabilizer_tableau":
                ent_method = "stabilizer"
            elif method in ('mps', 'matrix_product_state') or qubits > 20:
                ent_method = "mps"
            else:
                ent_method = "auto"
            entanglement_metrics = calculate_bipartite_entropy(
                circuit,
                method=ent_method,
                max_bond_dimension=bond_dimension
            )
        except Exception as ee:
            entanglement_metrics = {"error": str(ee)}

    # 8b. Ensure scalable bipartite entanglement entropy & Pauli expectations are populated across all solvers
    bipartite_entropy_ebits = sim_result.get("bipartite_entropy_ebits")
    end_to_end_zz_expectation = sim_result.get("end_to_end_zz_expectation")
    ising_energy_expectation = sim_result.get("ising_energy_expectation")

    if sim_result["success"] and circuit is not None and (
        bipartite_entropy_ebits is None or end_to_end_zz_expectation is None or ising_energy_expectation is None
    ):
        try:
            if entanglement_metrics and "von_neumann_entropy" in entanglement_metrics:
                bipartite_entropy_ebits = float(entanglement_metrics["von_neumann_entropy"])
            elif qubits <= 16:
                from src.engine.simulator import evolve_statevector
                from src.engine.observables import expectation_statevector, evaluate_hamiltonian
                sv_obs = evolve_statevector(circuit, precision=precision)
                if bipartite_entropy_ebits is None:
                    ent_res = calculate_bipartite_entropy(circuit, method="auto", max_bond_dimension=bond_dimension)
                    bipartite_entropy_ebits = float(ent_res.get("von_neumann_entropy", 0.0))
                if qubits == 1:
                    end_to_end_zz_expectation = float(expectation_statevector(sv_obs, "Z"))
                    ising_energy_expectation = -end_to_end_zz_expectation
                else:
                    zz_end_str = "Z" + ("I" * (qubits - 2)) + "Z"
                    end_to_end_zz_expectation = float(expectation_statevector(sv_obs, zz_end_str))
                    weight = -1.0 / float(qubits - 1)
                    ising_terms = [
                        (("I" * i) + "ZZ" + ("I" * (qubits - 2 - i)), weight)
                        for i in range(qubits - 1)
                    ]
                    ising_energy_expectation = float(evaluate_hamiltonian(sv_obs, ising_terms))
        except Exception:
            pass

    if bipartite_entropy_ebits is None:
        bipartite_entropy_ebits = 0.0
    if end_to_end_zz_expectation is None:
        end_to_end_zz_expectation = 0.0
    if ising_energy_expectation is None:
        ising_energy_expectation = 0.0

    # 9. Calculate Quantum Volume Heavy Output Metrics if applicable
    if workload_type == "qv" and sim_result["success"] and sim_result.get("counts") and qubits <= 16:
        try:
            from qiskit.quantum_info import Statevector
            circ_clean = circuit.copy()
            if len(circ_clean.clbits) > 0:
                circ_clean.remove_final_measurements(inplace=True)
            ideal_sv = Statevector.from_instruction(circ_clean)
            ideal_probs = ideal_sv.probabilities_dict()
            qv_metrics = calculate_heavy_output_probability(ideal_probs, sim_result["counts"])
            if noise_profile or noise_level != "none":
                qv_metrics["noise_degradation_active"] = True
                qv_metrics["fidelity_percent"] = fidelity
        except Exception:
            pass

    mean_lat = sim_result["mean_latency"]
    med_lat = sim_result["median_latency"]
    std_lat = sim_result["std_latency"]
    runs_cnt = sim_result["runs_count"]
    sim_meta = sim_result.get("metadata", {})

    from src.profiler.telemetry import calculate_effective_memory_bandwidth
    active_bond = sim_result.get("max_active_bond_dim") or bond_dimension or 16
    bw_telemetry = calculate_effective_memory_bandwidth(
        num_qubits=qubits,
        total_gates=num_gates,
        execution_time=mean_lat,
        sim_method=method,
        precision=precision,
        active_bond_dim=active_bond
    ) if sim_result["success"] else {
        "effective_bandwidth_gbps": 0.0,
        "arithmetic_intensity_flops_per_byte": 0.4375,
        "total_bytes_transferred": 0.0
    }

    return {
        "qubits": qubits,
        "method": method,
        "precision": precision,
        "effective_bandwidth_gbps": bw_telemetry["effective_bandwidth_gbps"],
        "arithmetic_intensity_flops_per_byte": bw_telemetry["arithmetic_intensity_flops_per_byte"],
        "bond_dimension": bond_dimension if method in ('mps', 'matrix_product_state') else None,
        "device": device.upper(),
        "workers": workers,
        "noise_level": noise_level,
        "model_parallelism": sim_meta.get("model_parallelism", "None"),
        "chunk_count": sim_meta.get("chunk_count", 1),
        "fidelity": fidelity,
        "overhead_ratio": overhead_ratio,
        "success": sim_result["success"],
        "latency": mean_lat,
        "mean_latency": mean_lat,
        "median_latency": med_lat,
        "std_latency": std_lat,
        "latencies": sim_result.get("latencies", []),
        "runs_count": runs_cnt,
        "gates": num_gates,
        "cpu_usage": cpu_usage,
        "ram_status": "SAFE" if is_safe else "UNSAFE",
        "error": sim_result.get("error"),
        "native_kernel_used": sim_result.get("native_kernel_used", False),
        "accelerator_backend": sim_result.get("accelerator_backend"),
        "accelerator_badge": sim_result.get("accelerator_badge"),
        "fusion_metrics": sim_result.get("fusion_metrics"),
        "physical_noise_profile": sim_result.get("physical_noise_profile"),
        "backend": backend,
        "cuda_telemetry": sim_result.get("cuda_telemetry"),
        "ram_savings": ram_savings,
        "mps_ram_savings": ram_savings,
        "entanglement_metrics": entanglement_metrics,
        "parameter_binding_metrics": param_binding_metrics,
        "qv_metrics": qv_metrics,
        "noise_method": sim_meta.get("noise_method", noise_method),
        "shots": shots,
        "memory_saved_ratio": sim_meta.get("memory_saved_ratio", (1 << qubits) if qubits < 1000 else float("inf")),
        "energy_metrics": energy_metrics,
        "qasm_metrics": qasm_metrics,
        "complexity_profile": complexity_profile,
        "solver_routing": solver_routing,
        "bipartite_entropy_ebits": float(bipartite_entropy_ebits),
        "bipartite_entanglement_entropy_ebits": float(bipartite_entropy_ebits),
        "end_to_end_zz_expectation": float(end_to_end_zz_expectation),
        "zz_correlation_expectation": float(end_to_end_zz_expectation),
        "ising_energy_expectation": float(ising_energy_expectation),
        "noisy_fidelity_estimate": sim_result.get("noisy_fidelity_estimate") if method == "stabilizer" else None,
        "tableau_memory_bytes": sim_result.get("tableau_memory_bytes") if method == "stabilizer" else None,
        "clifford_gate_count": sim_result.get("clifford_gate_count") if method == "stabilizer" else None,
        "mps_fidelity": sim_result.get("mps_fidelity") if method in ('mps', 'matrix_product_state') else None,
        "cumulative_truncation_error": sim_result.get("cumulative_truncation_error") if method in ('mps', 'matrix_product_state') else None,
        "max_active_bond_dim": sim_result.get("max_active_bond_dim") if method in ('mps', 'matrix_product_state') else None,
        "swap_count": sim_result.get("swap_count") if method in ('mps', 'matrix_product_state') else None,
        "max_bond_dim": bond_dimension if method in ('mps', 'matrix_product_state') else None,
        "svd_cutoff": svd_cutoff if method in ('mps', 'matrix_product_state') else None
    }

def run_quick_benchmark(
    args: Any,
    effective_device: str,
    console: Optional[Console] = None
) -> List[Dict[str, Any]]:
    """Execute Quick Benchmark Suite (Qubits: 10, 15, 20)."""
    c = console or default_console
    workers = getattr(args, 'workers', 1)
    backend = getattr(args, 'backend', 'auto')
    shots = getattr(args, 'shots', 1000)
    noise_method = getattr(args, 'noise_method', 'trajectory')
    state_slicing = getattr(args, 'state_slicing', False) or (effective_device == 'multi_gpu')
    blocking_qubits = getattr(args, 'blocking_qubits', None)
    use_native_kernels = getattr(args, 'use_native_kernels', False)
    noise_profile = getattr(args, 'noise_profile', None)
    slicing_label = " [Distributed State Slicing]" if state_slicing else ""
    native_label = " [Native Kernel Fusion]" if use_native_kernels else ""
    backend_label = f" [Backend: {backend.upper()}]" if backend != "auto" else ""

    if backend == "cuda" or (backend == "auto" and effective_device in ("gpu", "multi_gpu")):
        from src.profiler.gpu import get_cuda_telemetry
        cuda_info = get_cuda_telemetry()
        if cuda_info.get("available"):
            c.print(f"[bold cyan]NVIDIA CUDA Hardware Active:[/bold cyan] {cuda_info['device_name']} (Compute: {cuda_info['compute_capability']}, VRAM: {cuda_info['vram_total_mb']:.1f} MB)")

    if args.noise_level != 'none' or noise_profile is not None:
        noise_desc = "Monte Carlo Wavefunction (Trajectory)" if noise_method == "trajectory" else "Density Matrix"
        c.print(f"[bold cyan]Noise Simulation Mode:[/] [bold magenta]{noise_desc}[/] | Shots: [bold green]{shots:,}[/] | Preserving O(2^n) Scaling (Memory Saved: [bold yellow]~2^n x[/bold yellow])")

    c.print(f"[bold yellow]Executing Quick Benchmark Suite (Qubits: 10, 15, 20) [Method: {args.method.upper()}, Device: {effective_device.upper()}{slicing_label}{native_label}{backend_label}, Workers: {workers}, Noise: {args.noise_level.upper()} ({noise_method.upper()}, {shots} shots), Entropy: {args.entropy}, Runs: {args.runs}]...[/bold yellow]\n")
    qubits_list = [10, 15, 20]
    results = []

    workload = getattr(args, 'type', 'qft')
    depth = getattr(args, 'depth', 10)
    effective_bond_dim = getattr(args, 'max_bond_dim', None) or getattr(args, 'bond_dim', 64)
    svd_cutoff = getattr(args, 'svd_cutoff', 1e-9)
    precision = getattr(args, 'precision', 'fp64')
    for q in qubits_list:
        with Status(f"Running simulation for {q} qubits on {effective_device.upper()} ({args.runs} runs, {workers} workers)...", console=c):
            res = run_single_simulation(
                q, workload, depth, args.method, effective_bond_dim, effective_device,
                args.noise_level, args.runs, workers=workers, compute_entropy=args.entropy,
                state_slicing=state_slicing, blocking_qubits=blocking_qubits,
                use_native_kernels=use_native_kernels, noise_profile=noise_profile,
                backend=backend, shots=shots, noise_method=noise_method,
                svd_cutoff=svd_cutoff, precision=precision
            )
            res["workload_label"] = workload.upper()
            results.append(res)
            if not res["success"]:
                c.print(f"[bold red]Skipping remaining runs due to error/limit at {q} qubits:[/bold red] {res['error']}")
                break
    return results

def run_full_stress_test(
    args: Any,
    effective_device: str,
    console: Optional[Console] = None
) -> List[Dict[str, Any]]:
    """Execute Full Incremental Stress Test starting from 10 qubits."""
    c = console or default_console
    workers = getattr(args, 'workers', 1)
    backend = getattr(args, 'backend', 'auto')
    shots = getattr(args, 'shots', 1000)
    noise_method = getattr(args, 'noise_method', 'trajectory')
    state_slicing = getattr(args, 'state_slicing', False) or (effective_device == 'multi_gpu')
    blocking_qubits = getattr(args, 'blocking_qubits', None)
    use_native_kernels = getattr(args, 'use_native_kernels', False)
    noise_profile = getattr(args, 'noise_profile', None)
    slicing_label = " [Distributed State Slicing]" if state_slicing else ""
    native_label = " [Native Kernel Fusion]" if use_native_kernels else ""
    backend_label = f" [Backend: {backend.upper()}]" if backend != "auto" else ""

    if backend == "cuda" or (backend == "auto" and effective_device in ("gpu", "multi_gpu")):
        from src.profiler.gpu import get_cuda_telemetry
        cuda_info = get_cuda_telemetry()
        if cuda_info.get("available"):
            c.print(f"[bold cyan]NVIDIA CUDA Hardware Active:[/bold cyan] {cuda_info['device_name']} (Compute: {cuda_info['compute_capability']}, VRAM: {cuda_info['vram_total_mb']:.1f} MB)")

    if args.noise_level != 'none' or noise_profile is not None:
        noise_desc = "Monte Carlo Wavefunction (Trajectory)" if noise_method == "trajectory" else "Density Matrix"
        c.print(f"[bold cyan]Noise Simulation Mode:[/] [bold magenta]{noise_desc}[/] | Shots: [bold green]{shots:,}[/] | Preserving O(2^n) Scaling (Memory Saved: [bold yellow]~2^n x[/bold yellow])")

    c.print(f"[bold yellow]Executing Full Incremental Stress Test (starting from 10 qubits) [Method: {args.method.upper()}, Device: {effective_device.upper()}{slicing_label}{native_label}{backend_label}, Workers: {workers}, Noise: {args.noise_level.upper()} ({noise_method.upper()}, {shots} shots), Entropy: {args.entropy}, Runs: {args.runs}]...[/bold yellow]\n")
    q = 10
    max_limit = 50 if args.method == 'mps' else 100
    results = []

    effective_bond_dim = getattr(args, 'max_bond_dim', None) or getattr(args, 'bond_dim', 64)
    svd_cutoff = getattr(args, 'svd_cutoff', 1e-9)
    precision = getattr(args, 'precision', 'fp64')
    while q <= max_limit:
        with Status(f"Running simulation for {q} qubits on {effective_device.upper()} ({args.runs} runs, {workers} workers)...", console=c):
            res = run_single_simulation(
                q, "qft", 0, args.method, effective_bond_dim, effective_device,
                args.noise_level, args.runs, workers=workers, compute_entropy=args.entropy,
                state_slicing=state_slicing, blocking_qubits=blocking_qubits,
                use_native_kernels=use_native_kernels, noise_profile=noise_profile,
                backend=backend, shots=shots, noise_method=noise_method,
                svd_cutoff=svd_cutoff, precision=precision
            )
            res["workload_label"] = "QFT"
            results.append(res)
            if not res["success"]:
                c.print(f"[bold red]Stress test stopped at {q} qubits:[/bold red] {res['error']}")
                break
            q += 1
    return results

def run_custom_simulation(
    args: Any,
    effective_device: str,
    console: Optional[Console] = None
) -> List[Dict[str, Any]]:
    """Execute Custom Simulation configuration, supporting procedural workloads or loaded OpenQASM 2.0 circuits."""
    c = console or default_console
    workers = getattr(args, 'workers', 1)
    backend = getattr(args, 'backend', 'auto')
    shots = getattr(args, 'shots', 1000)
    noise_method = getattr(args, 'noise_method', 'trajectory')
    state_slicing = getattr(args, 'state_slicing', False) or (effective_device == 'multi_gpu')
    blocking_qubits = getattr(args, 'blocking_qubits', None)
    use_native_kernels = getattr(args, 'use_native_kernels', False)
    noise_profile = getattr(args, 'noise_profile', None)
    slicing_label = " [Distributed State Slicing]" if state_slicing else ""
    native_label = " [Native Kernel Fusion]" if use_native_kernels else ""
    backend_label = f" [Backend: {backend.upper()}]" if backend != "auto" else ""

    if backend == "cuda" or (backend == "auto" and effective_device in ("gpu", "multi_gpu")):
        from src.profiler.gpu import get_cuda_telemetry
        cuda_info = get_cuda_telemetry()
        if cuda_info.get("available"):
            c.print(f"[bold cyan]NVIDIA CUDA Hardware Active:[/bold cyan] {cuda_info['device_name']} (Compute: {cuda_info['compute_capability']}, VRAM: {cuda_info['vram_total_mb']:.1f} MB)")

    parsed_qasm = None
    qasm_metrics = None
    qasm_path = getattr(args, 'qasm', None)

    if qasm_path:
        from src.engine.parser import load_qasm
        t0_parse = time.perf_counter()
        try:
            parsed_qasm = load_qasm(qasm_path)
        except Exception as pe:
            c.print(f"[bold red]QASM Parse Error:[/bold red] Failed to parse '{qasm_path}': {pe}")
            return []
        parse_lat = time.perf_counter() - t0_parse
        summary = parsed_qasm.get_summary()
        qasm_metrics = {
            "name": summary["name"],
            "file_path": qasm_path,
            "depth": summary["depth"],
            "one_qubit_gates": summary["one_qubit_gates"],
            "two_qubit_gates": summary["two_qubit_gates"],
            "multi_qubit_gates": summary["multi_qubit_gates"],
            "total_gates": summary["total_gates"],
            "parse_latency": parse_lat
        }
        effective_qubits = parsed_qasm.num_qubits
        c.print(f"[bold green]OpenQASM 2.0 Circuit Loaded:[/bold green] [bold yellow]{summary['name']}[/bold yellow] ({qasm_path})")
        c.print(f"  - Qubits: [bold cyan]{summary['num_qubits']}[/bold cyan] | Classical Bits: [cyan]{summary['num_clbits']}[/cyan] | Depth: [bold magenta]{summary['depth']}[/bold magenta]")
        c.print(f"  - Total Gates: [bold green]{summary['total_gates']}[/bold green] (1-Qubit: [cyan]{summary['one_qubit_gates']}[/cyan], 2-Qubit: [magenta]{summary['two_qubit_gates']}[/magenta], Multi-Qubit: [yellow]{summary['multi_qubit_gates']}[/yellow])")
        c.print(f"  - Parse Latency: [dim cyan]{parse_lat * 1000.0:.2f} ms[/dim cyan]")
        c.print()
        workload_desc = f"QASM: {summary['name']}"
    else:
        effective_qubits = args.qubits
        workload_desc = args.type.upper()

    if args.noise_level != 'none' or noise_profile is not None:
        if args.method == "stabilizer":
            noise_desc = "Noisy Stabilizer Trajectories (Monte Carlo Pauli Jumps)"
            scale_desc = "Preserving O(n^2) Stabilizer Tableau Scaling"
        else:
            noise_desc = "Monte Carlo Wavefunction (Trajectory)" if noise_method == "trajectory" else "Density Matrix"
            scale_desc = (
                f"Preserving O(2^n) Scaling (Memory Saved: [bold yellow]~2^{effective_qubits} = ~{1 << effective_qubits:,} x[/bold yellow])"
                if effective_qubits <= 30
                else f"Preserving O(2^n) Scaling (Memory Saved: [bold yellow]~2^{effective_qubits} x[/bold yellow])"
            )
        c.print(f"[bold cyan]Noise Simulation Mode:[/] [bold magenta]{noise_desc}[/] | Shots: [bold green]{shots:,}[/] | {scale_desc}")

    c.print(f"[bold yellow]Executing Custom Simulation (Qubits: {effective_qubits}, Workload: {workload_desc}, Method: {args.method.upper()}, Device: {effective_device.upper()}{slicing_label}{native_label}{backend_label}, Workers: {workers}, Noise: {args.noise_level.upper()} ({noise_method.upper()}, {shots} shots), Entropy: {args.entropy}, Runs: {args.runs})...[/bold yellow]\n")
    results = []

    effective_bond_dim = getattr(args, 'max_bond_dim', None) or getattr(args, 'bond_dim', 64)
    svd_cutoff = getattr(args, 'svd_cutoff', 1e-9)
    precision = getattr(args, 'precision', 'fp64')
    with Status(f"Running simulation for {effective_qubits} qubits on {effective_device.upper()} ({args.runs} runs, {workers} workers)...", console=c):
        res = run_single_simulation(
            effective_qubits, args.type, args.depth, args.method, effective_bond_dim, effective_device,
            args.noise_level, args.runs, workers=workers, compute_entropy=args.entropy,
            state_slicing=state_slicing, blocking_qubits=blocking_qubits,
            use_native_kernels=use_native_kernels, noise_profile=noise_profile,
            qasm_circuit=parsed_qasm, qasm_metrics=qasm_metrics,
            backend=backend, shots=shots, noise_method=noise_method,
            svd_cutoff=svd_cutoff, precision=precision
        )
        if parsed_qasm:
            res["workload_label"] = f"QASM: {parsed_qasm.name}"
        else:
            res["workload_label"] = args.type.upper()
            if args.type == "deep":
                res["workload_label"] += f" (d={args.depth})"
        results.append(res)
        if not res["success"]:
            c.print(f"[bold red]Simulation aborted:[/bold red] {res['error']}")
    return results
