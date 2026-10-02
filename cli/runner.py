from typing import List, Dict, Any, Optional
import time
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
    backend: str = 'auto'
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

    try:
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
                blocking_qubits=blocking_qubits,
                use_native_kernels=use_native_kernels,
                physical_noise_profile=noise_profile,
                backend=backend
            )
    except InsufficientVRAMError as vram_err:
        default_console.print(
            f"[bold yellow]⚠️  VRAM Limit Exceeded:[/] {vram_err}\n"
            f"[bold cyan]ℹ️  Automatically falling back to CPU / NumPy execution...[/]"
        )
        with EnergyProfiler() as energy_prof:
            sim_result = run_simulation(
                circuit, 
                method=method, 
                bond_dimension=bond_dimension, 
                device="cpu",
                noise_model=noise_model,
                noise_level=noise_level,
                runs=runs,
                workers=workers,
                state_slicing=False,
                blocking_qubits=None,
                use_native_kernels=False,
                physical_noise_profile=noise_profile,
                backend="numpy"
            )
    finally:
        active_be.device_synchronize()

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
        active_be.device_synchronize()
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
            blocking_qubits=blocking_qubits,
            backend=backend
        )
        active_be.device_synchronize()
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
        "native_kernel_used": sim_result.get("native_kernel_used", False),
        "accelerator_backend": sim_result.get("accelerator_backend"),
        "accelerator_badge": sim_result.get("accelerator_badge"),
        "fusion_metrics": sim_result.get("fusion_metrics"),
        "physical_noise_profile": sim_result.get("physical_noise_profile"),
        "backend": backend,
        "cuda_telemetry": sim_result.get("cuda_telemetry"),
        "ram_savings": ram_savings,
        "entanglement_metrics": entanglement_metrics,
        "parameter_binding_metrics": param_binding_metrics,
        "qv_metrics": qv_metrics,
        "energy_metrics": energy_metrics,
        "qasm_metrics": qasm_metrics
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
            
    c.print(f"[bold yellow]Executing Quick Benchmark Suite (Qubits: 10, 15, 20) [Method: {args.method.upper()}, Device: {effective_device.upper()}{slicing_label}{native_label}{backend_label}, Workers: {workers}, Noise: {args.noise_level.upper()}, Entropy: {args.entropy}, Runs: {args.runs}]...[/bold yellow]\n")
    qubits_list = [10, 15, 20]
    results = []
    
    workload = getattr(args, 'type', 'qft')
    depth = getattr(args, 'depth', 10)
    for q in qubits_list:
        with Status(f"Running simulation for {q} qubits on {effective_device.upper()} ({args.runs} runs, {workers} workers)...", console=c):
            res = run_single_simulation(
                q, workload, depth, args.method, args.bond_dim, effective_device, 
                args.noise_level, args.runs, workers=workers, compute_entropy=args.entropy,
                state_slicing=state_slicing, blocking_qubits=blocking_qubits,
                use_native_kernels=use_native_kernels, noise_profile=noise_profile,
                backend=backend
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

    c.print(f"[bold yellow]Executing Full Incremental Stress Test (starting from 10 qubits) [Method: {args.method.upper()}, Device: {effective_device.upper()}{slicing_label}{native_label}{backend_label}, Workers: {workers}, Noise: {args.noise_level.upper()}, Entropy: {args.entropy}, Runs: {args.runs}]...[/bold yellow]\n")
    q = 10
    max_limit = 50 if args.method == 'mps' else 100
    results = []
    
    while q <= max_limit:
        with Status(f"Running simulation for {q} qubits on {effective_device.upper()} ({args.runs} runs, {workers} workers)...", console=c):
            res = run_single_simulation(
                q, "qft", 0, args.method, args.bond_dim, effective_device, 
                args.noise_level, args.runs, workers=workers, compute_entropy=args.entropy,
                state_slicing=state_slicing, blocking_qubits=blocking_qubits,
                use_native_kernels=use_native_kernels, noise_profile=noise_profile,
                backend=backend
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
        
    c.print(f"[bold yellow]Executing Custom Simulation (Qubits: {effective_qubits}, Workload: {workload_desc}, Method: {args.method.upper()}, Device: {effective_device.upper()}{slicing_label}{native_label}{backend_label}, Workers: {workers}, Noise: {args.noise_level.upper()}, Entropy: {args.entropy}, Runs: {args.runs})...[/bold yellow]\n")
    results = []
    
    with Status(f"Running simulation for {effective_qubits} qubits on {effective_device.upper()} ({args.runs} runs, {workers} workers)...", console=c):
        res = run_single_simulation(
            effective_qubits, args.type, args.depth, args.method, args.bond_dim, effective_device, 
            args.noise_level, args.runs, workers=workers, compute_entropy=args.entropy,
            state_slicing=state_slicing, blocking_qubits=blocking_qubits,
            use_native_kernels=use_native_kernels, noise_profile=noise_profile,
            qasm_circuit=parsed_qasm, qasm_metrics=qasm_metrics,
            backend=backend
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
