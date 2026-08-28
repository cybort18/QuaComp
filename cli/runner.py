from typing import List, Dict, Any, Optional
import psutil
from rich.console import Console
from rich.status import Status

from src.profiler.memory import check_memory_safety
from src.profiler.telemetry import get_cpu_utilization
from src.engine.circuits import generate_shallow_circuit, generate_deep_circuit, generate_qft_circuit
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
    compute_entropy: bool = False
) -> Dict[str, Any]:
    """Execute a single simulation step with safety checks, multi-run noise options, device selection, and telemetry collection."""
    # 1. Memory safety check (supports RAM and GPU VRAM check)
    is_safe, msg = check_memory_safety(qubits, method, device=device)
    if not is_safe:
        return {
            "qubits": qubits,
            "method": method,
            "bond_dimension": bond_dimension if method in ('mps', 'matrix_product_state') else None,
            "device": device.upper(),
            "noise_level": noise_level,
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
            "entanglement_metrics": {}
        }
        
    # 2. Circuit generation
    circuit = None
    if workload_type == "shallow":
        circuit = generate_shallow_circuit(qubits)
    elif workload_type == "deep":
        circuit = generate_deep_circuit(qubits, depth)
    elif workload_type == "qft":
        circuit = generate_qft_circuit(qubits)
        
    num_gates = sum(circuit.count_ops().values())
    
    # 3. CPU measurement start
    get_cpu_utilization()
    
    # 4. Simulation run with noise, device, and method support (multiple runs)
    noise_model = get_noise_model(noise_level)
    sim_result = run_simulation(
        circuit, 
        method=method, 
        bond_dimension=bond_dimension,
        device=device,
        noise_model=noise_model,
        noise_level=noise_level,
        runs=runs
    )
    
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
            runs=1
        )
        if ideal_sim_result["success"]:
            fidelity = calculate_state_fidelity(ideal_sim_result["counts"], sim_result["counts"])
            overhead_ratio = calculate_overhead_ratio(ideal_sim_result["latency"], sim_result["mean_latency"])
            
    # 8. Calculate Entanglement Entropy Metrics if requested
    entanglement_metrics = {}
    if compute_entropy and sim_result["success"] and circuit is not None:
        try:
            entanglement_metrics = calculate_bipartite_entropy(circuit)
        except Exception as ee:
            entanglement_metrics = {"error": str(ee)}
    
    mean_lat = sim_result["mean_latency"]
    med_lat = sim_result["median_latency"]
    std_lat = sim_result["std_latency"]
    runs_cnt = sim_result["runs_count"]
    
    return {
        "qubits": qubits,
        "method": method,
        "bond_dimension": bond_dimension if method in ('mps', 'matrix_product_state') else None,
        "device": device.upper(),
        "noise_level": noise_level,
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
        "entanglement_metrics": entanglement_metrics
    }

def run_quick_benchmark(
    args: Any, 
    effective_device: str, 
    console: Optional[Console] = None
) -> List[Dict[str, Any]]:
    """Execute Quick Benchmark Suite (Qubits: 10, 15, 20)."""
    c = console or default_console
    c.print(f"[bold yellow]Executing Quick Benchmark Suite (Qubits: 10, 15, 20) [Method: {args.method.upper()}, Device: {effective_device.upper()}, Noise: {args.noise_level.upper()}, Entropy: {args.entropy}, Runs: {args.runs}]...[/bold yellow]\n")
    qubits_list = [10, 15, 20]
    results = []
    
    for q in qubits_list:
        with Status(f"Running simulation for {q} qubits on {effective_device.upper()} ({args.runs} runs)...", console=c):
            res = run_single_simulation(q, "qft", 0, args.method, args.bond_dim, effective_device, args.noise_level, args.runs, compute_entropy=args.entropy)
            res["workload_label"] = "QFT"
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
    c.print(f"[bold yellow]Executing Full Incremental Stress Test (starting from 10 qubits) [Method: {args.method.upper()}, Device: {effective_device.upper()}, Noise: {args.noise_level.upper()}, Entropy: {args.entropy}, Runs: {args.runs}]...[/bold yellow]\n")
    q = 10
    max_limit = 35 if args.method == 'mps' else 100
    results = []
    
    while q <= max_limit:
        with Status(f"Running simulation for {q} qubits on {effective_device.upper()} ({args.runs} runs)...", console=c):
            res = run_single_simulation(q, "qft", 0, args.method, args.bond_dim, effective_device, args.noise_level, args.runs, compute_entropy=args.entropy)
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
    c.print(f"[bold yellow]Executing Custom Simulation (Qubits: {args.qubits}, Workload: {args.type.upper()}, Method: {args.method.upper()}, Device: {effective_device.upper()}, Noise: {args.noise_level.upper()}, Entropy: {args.entropy}, Runs: {args.runs})...[/bold yellow]\n")
    results = []
    
    with Status(f"Running simulation for {args.qubits} qubits on {effective_device.upper()} ({args.runs} runs)...", console=c):
        res = run_single_simulation(args.qubits, args.type, args.depth, args.method, args.bond_dim, effective_device, args.noise_level, args.runs, compute_entropy=args.entropy)
        res["workload_label"] = args.type.upper()
        if args.type == "deep":
            res["workload_label"] += f" (d={args.depth})"
        results.append(res)
        if not res["success"]:
            c.print(f"[bold red]Simulation aborted:[/bold red] {res['error']}")
    return results
