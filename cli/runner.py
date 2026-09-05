from typing import List, Dict, Any, Optional
import psutil
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
    blocking_qubits: Optional[int] = None
) -> Dict[str, Any]:
    """
    Execute a single quantum simulation workload with profiling.
    
    Returns:
        dict: Simulation results and execution metrics.
    """
    # 1. Pre-flight memory safety check
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
            "energy_metrics": {}
        }
        
    # 2. Circuit generation
    circuit = None
    param_binding_metrics = {}
    qv_metrics = {}
    
    if workload_type == "shallow":
        circuit = generate_shallow_circuit(qubits)
    elif workload_type == "deep":
        circuit = generate_deep_circuit(qubits, depth)
    elif workload_type == "qft":
        circuit = generate_qft_circuit(qubits)
    elif workload_type == "vqe":
        vqe_depth = max(1, depth // 5) if depth > 5 else depth
        raw_vqe = generate_vqe_circuit(qubits, depth=vqe_depth)
        param_binding_metrics = profile_parameter_binding(raw_vqe)
        import numpy as np
        rng_vals = [float(v) for v in np.linspace(0.1, 3.14, raw_vqe.num_parameters)]
        circuit = raw_vqe.assign_parameters(rng_vals)
    elif workload_type == "qaoa":
        qaoa_steps = max(1, depth // 10) if depth > 10 else 1
        raw_qaoa = generate_qaoa_circuit(qubits, p_steps=qaoa_steps)
        param_binding_metrics = profile_parameter_binding(raw_qaoa)
        import numpy as np
        rng_vals = [float(v) for v in np.linspace(0.1, 1.57, raw_qaoa.num_parameters)]
        circuit = raw_qaoa.assign_parameters(rng_vals)
    elif workload_type == "qv":
        qv_qubits = max(2, qubits)
        circuit = generate_quantum_volume_circuit(qv_qubits, depth=depth if depth is not None else qv_qubits)
    else:
        circuit = generate_qft_circuit(qubits)
        
    num_gates = sum(circuit.count_ops().values())
    
    # 3. Energy and CPU measurement start
    get_cpu_utilization()
    
    # 4. Simulation run with Energy Profiler, noise, multi-GPU/device, workers, and method support (multiple runs)
    from src.profiler.energy import EnergyProfiler
    noise_model = get_noise_model(noise_level)
    with EnergyProfiler() as energy_prof:
        sim_result = run_simulation(
            circuit, 
            method=method, 
            bond_dimension=bond_dimension, 
            device=device,
            noise_model=noise_model,
            noise_level=noise_level,
            runs=runs,
            workers=workers,
            state_slicing=state_slicing,
            blocking_qubits=blocking_qubits
        )
    energy_metrics = energy_prof.get_metrics(num_gates=num_gates)
    
    # 5. CPU measurement end
    cpu_metric = get_cpu_utilization()
    cpu_usage = float(cpu_metric.get("overall_percent", 0.0)) if isinstance(cpu_metric, dict) else float(cpu_metric)
    
    # 6. Calculate MPS RAM savings if applicable
    ram_savings = {}
    if method in ('mps', 'matrix_product_state') and sim_result["success"]:
        process = psutil.Process()
        actual_ram_bytes = process.memory_info().rss
        ram_savings = calculate_mps_ram_savings(qubits, actual_ram_bytes)
        
    # 7. Calculate NISQ metrics (Fidelity and CPU Overhead) if noisy
    fidelity = 100.0
    overhead_ratio = 0.0
    if noise_level != "none" and sim_result["success"]:
        ideal_sim_result = run_simulation(
            circuit, 
            method=method, 
            bond_dimension=bond_dimension,
            device=device,
            noise_model=None, 
            noise_level="none",
            runs=1,
            workers=workers,
            state_slicing=state_slicing,
            blocking_qubits=blocking_qubits
        )
        if ideal_sim_result["success"]:
            fidelity = calculate_state_fidelity(ideal_sim_result["counts"], sim_result["counts"])
            overhead_ratio = calculate_overhead_ratio(ideal_sim_result["latency"], sim_result["mean_latency"])
            
    # 8. Calculate Entanglement Entropy Metrics if requested (Supports Native MPS Tensor for large qubits!)
    entanglement_metrics = {}
    if compute_entropy and sim_result["success"] and circuit is not None:
        try:
            ent_method = "mps" if method in ('mps', 'matrix_product_state') else "auto"
            entanglement_metrics = calculate_bipartite_entropy(
                circuit, 
                method=ent_method, 
                max_bond_dimension=bond_dimension
            )
        except Exception as ee:
            entanglement_metrics = {"error": str(ee)}
    
    # 9. Calculate Quantum Volume Heavy Output Metrics if applicable
    if workload_type == "qv" and sim_result["success"] and sim_result.get("counts") and qubits <= 16:
        try:
            from qiskit.quantum_info import Statevector
            circ_clean = circuit.copy()
            circ_clean.remove_final_measurements(inplace=True)
            ideal_sv = Statevector.from_instruction(circ_clean)
            ideal_probs = ideal_sv.probabilities_dict()
            qv_metrics = calculate_heavy_output_probability(ideal_probs, sim_result["counts"])
        except Exception:
            pass

    mean_lat = sim_result["mean_latency"]
    med_lat = sim_result["median_latency"]
    std_lat = sim_result["std_latency"]
    runs_cnt = sim_result["runs_count"]
    sim_meta = sim_result.get("metadata", {})
    
    return {
        "qubits": qubits,
        "method": method,
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
        "ram_savings": ram_savings,
        "entanglement_metrics": entanglement_metrics,
        "parameter_binding_metrics": param_binding_metrics,
        "qv_metrics": qv_metrics,
        "energy_metrics": energy_metrics
    }

def run_quick_benchmark(
    args: Any, 
    effective_device: str, 
    console: Optional[Console] = None
) -> List[Dict[str, Any]]:
    """Execute Quick Benchmark Suite (Qubits: 10, 15, 20)."""
    c = console or default_console
    workers = getattr(args, 'workers', 1)
    state_slicing = getattr(args, 'state_slicing', False) or (effective_device == 'multi_gpu')
    blocking_qubits = getattr(args, 'blocking_qubits', None)
    slicing_label = " [Distributed State Slicing]" if state_slicing else ""
    c.print(f"[bold yellow]Executing Quick Benchmark Suite (Qubits: 10, 15, 20) [Method: {args.method.upper()}, Device: {effective_device.upper()}{slicing_label}, Workers: {workers}, Noise: {args.noise_level.upper()}, Entropy: {args.entropy}, Runs: {args.runs}]...[/bold yellow]\n")
    qubits_list = [10, 15, 20]
    results = []
    
    workload = getattr(args, 'type', 'qft')
    depth = getattr(args, 'depth', 10)
    for q in qubits_list:
        with Status(f"Running simulation for {q} qubits on {effective_device.upper()} ({args.runs} runs, {workers} workers)...", console=c):
            res = run_single_simulation(
                q, workload, depth, args.method, args.bond_dim, effective_device, 
                args.noise_level, args.runs, workers=workers, compute_entropy=args.entropy,
                state_slicing=state_slicing, blocking_qubits=blocking_qubits
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
    state_slicing = getattr(args, 'state_slicing', False) or (effective_device == 'multi_gpu')
    blocking_qubits = getattr(args, 'blocking_qubits', None)
    slicing_label = " [Distributed State Slicing]" if state_slicing else ""
    c.print(f"[bold yellow]Executing Full Incremental Stress Test (starting from 10 qubits) [Method: {args.method.upper()}, Device: {effective_device.upper()}{slicing_label}, Workers: {workers}, Noise: {args.noise_level.upper()}, Entropy: {args.entropy}, Runs: {args.runs}]...[/bold yellow]\n")
    q = 10
    max_limit = 50 if args.method == 'mps' else 100
    results = []
    
    while q <= max_limit:
        with Status(f"Running simulation for {q} qubits on {effective_device.upper()} ({args.runs} runs, {workers} workers)...", console=c):
            res = run_single_simulation(
                q, "qft", 0, args.method, args.bond_dim, effective_device, 
                args.noise_level, args.runs, workers=workers, compute_entropy=args.entropy,
                state_slicing=state_slicing, blocking_qubits=blocking_qubits
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
    """Execute Custom Simulation configuration."""
    c = console or default_console
    workers = getattr(args, 'workers', 1)
    state_slicing = getattr(args, 'state_slicing', False) or (effective_device == 'multi_gpu')
    blocking_qubits = getattr(args, 'blocking_qubits', None)
    slicing_label = " [Distributed State Slicing]" if state_slicing else ""
    c.print(f"[bold yellow]Executing Custom Simulation (Qubits: {args.qubits}, Workload: {args.type.upper()}, Method: {args.method.upper()}, Device: {effective_device.upper()}{slicing_label}, Workers: {workers}, Noise: {args.noise_level.upper()}, Entropy: {args.entropy}, Runs: {args.runs})...[/bold yellow]\n")
    results = []
    
    with Status(f"Running simulation for {args.qubits} qubits on {effective_device.upper()} ({args.runs} runs, {workers} workers)...", console=c):
        res = run_single_simulation(
            args.qubits, args.type, args.depth, args.method, args.bond_dim, effective_device, 
            args.noise_level, args.runs, workers=workers, compute_entropy=args.entropy,
            state_slicing=state_slicing, blocking_qubits=blocking_qubits
        )
        res["workload_label"] = args.type.upper()
        if args.type == "deep":
            res["workload_label"] += f" (d={args.depth})"
        results.append(res)
        if not res["success"]:
            c.print(f"[bold red]Simulation aborted:[/bold red] {res['error']}")
    return results
