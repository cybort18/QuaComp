import argparse
import sys
import time
from typing import List, Dict, Any
import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.status import Status

from src.profiler.memory import check_memory_safety, estimate_qubit_ram
from src.profiler.telemetry import get_system_metadata, get_cpu_utilization
from src.engine.circuits import generate_shallow_circuit, generate_deep_circuit, generate_qft_circuit
from src.engine.simulator import run_simulation
from src.scorer.calculator import calculate_qsim_score, categorize_score, calculate_scoring_breakdown
from src.reporter.json_exporter import export_to_json
from src.reporter.md_exporter import export_to_markdown
from src.reporter.charts import generate_benchmark_charts
from src.engine.mps import calculate_mps_ram_savings
from src.engine.noise import get_noise_model, calculate_state_fidelity, calculate_overhead_ratio

console = Console()

BANNER = r"""
 [bold cyan]  ____             ____                     [/bold cyan]
 [bold cyan] / __ \__  ______ / ___| ___  _ __ ___  _ __ [/bold cyan]
 [bold cyan]/ / / / / / / __ `/ /   / _ \| '_ ` _ \| '_ \[/bold cyan]
 [bold cyan]/ /_/ / /_/ / /_/ / |__| (_) | | | | | | |_) |[/bold cyan]
 [bold cyan]\___\_\__,_/\__,_/\____/\___/|_| |_| |_| .__/ [/bold cyan]
 [bold cyan]                                       |_|    [/bold cyan]
 [bold yellow]======= Quantum Computer Simulation Benchmark =======[/bold yellow]
"""

def print_system_info():
    """Print system hardware and environment metadata in a beautiful table."""
    metadata = get_system_metadata()
    table = Table(title="System Metadata & Telemetry", show_header=True, header_style="bold magenta", expand=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("System Value", style="green")
    
    table.add_row("CPU Name", metadata["cpu_name"])
    table.add_row("Total Physical RAM", f"{metadata['total_ram_gb']:.2f} GB")
    table.add_row("Operating System", f"{metadata['os_name']} ({metadata['os_release']})")
    table.add_row("Python Version", metadata["python_version"])
    
    console.print(table)
    console.print()

def run_single_simulation(
    qubits: int, 
    workload_type: str, 
    depth: int, 
    method: str = 'statevector', 
    bond_dimension: int = 64,
    noise_level: str = 'none',
    runs: int = 3
) -> Dict[str, Any]:
    """Execute a single simulation step with safety checks, multi-run noise options, and telemetry collection."""
    # 1. Memory safety check
    is_safe, msg = check_memory_safety(qubits, method)
    if not is_safe:
        return {
            "qubits": qubits,
            "method": method,
            "bond_dimension": bond_dimension if method in ('mps', 'matrix_product_state') else None,
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
            "ram_savings": {}
        }
        
    # 2. Get baseline CPU usage and process memory
    cpu_before = get_cpu_utilization()["overall_percent"]
    process = psutil.Process()
    ram_before = process.memory_info().rss
    
    # 3. Generate circuit
    try:
        if workload_type == "shallow":
            circuit = generate_shallow_circuit(qubits)
        elif workload_type == "deep":
            circuit = generate_deep_circuit(qubits, depth)
        else:
            circuit = generate_qft_circuit(qubits)
    except Exception as e:
        return {
            "qubits": qubits,
            "method": method,
            "bond_dimension": bond_dimension if method in ('mps', 'matrix_product_state') else None,
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
            "ram_status": "FAIL_GEN",
            "error": f"Circuit generation error: {e}",
            "ram_savings": {}
        }
        
    gate_count = circuit.size()
    
    # 4. Execute simulation (ideal baseline + noisy if requested) with runs iterations
    fidelity = 100.0
    overhead_ratio = 0.0
    
    if noise_level != 'none':
        ideal_res = run_simulation(circuit, method, bond_dimension, noise_model=None, noise_level='none', runs=runs)
        noise_model = get_noise_model(noise_level)
        sim_result = run_simulation(circuit, method, bond_dimension, noise_model=noise_model, noise_level=noise_level, runs=runs)
        
        if ideal_res["success"] and sim_result["success"]:
            fidelity = calculate_state_fidelity(ideal_res["counts"], sim_result["counts"])
            overhead_ratio = calculate_overhead_ratio(ideal_res["mean_latency"], sim_result["mean_latency"])
    else:
        sim_result = run_simulation(circuit, method, bond_dimension, noise_model=None, noise_level='none', runs=runs)
    
    # 5. Get CPU usage and memory footprint after
    cpu_after = get_cpu_utilization()["overall_percent"]
    avg_cpu = (cpu_before + cpu_after) / 2
    ram_after = process.memory_info().rss
    actual_ram_used = max(0, ram_after - ram_before)
    if actual_ram_used == 0:
        actual_ram_used = process.memory_info().rss
        
    ram_savings = {}
    if sim_result["success"] and method in ('mps', 'matrix_product_state'):
        ram_savings = calculate_mps_ram_savings(qubits, actual_ram_used)
    
    if sim_result["success"]:
        return {
            "qubits": qubits,
            "method": method,
            "bond_dimension": bond_dimension if method in ('mps', 'matrix_product_state') else None,
            "noise_level": noise_level,
            "fidelity": fidelity,
            "overhead_ratio": overhead_ratio,
            "success": True,
            "latency": sim_result["mean_latency"],
            "mean_latency": sim_result["mean_latency"],
            "median_latency": sim_result["median_latency"],
            "std_latency": sim_result["std_latency"],
            "latencies": sim_result["latencies"],
            "runs_count": sim_result["runs_count"],
            "gates": gate_count,
            "cpu_usage": avg_cpu,
            "ram_status": "SAFE",
            "error": None,
            "ram_savings": ram_savings
        }
    else:
        return {
            "qubits": qubits,
            "method": method,
            "bond_dimension": bond_dimension if method in ('mps', 'matrix_product_state') else None,
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
            "gates": gate_count,
            "cpu_usage": avg_cpu,
            "ram_status": "FAIL_EXEC",
            "error": sim_result["error"],
            "ram_savings": {}
        }

def display_results(results: List[Dict[str, Any]]):
    """Present simulation results in a beautiful table and calculate overall score."""
    table = Table(title="QuaComp Benchmark Results", show_header=True, header_style="bold blue")
    table.add_column("Qubits", justify="right", style="cyan")
    table.add_column("Method", style="magenta")
    table.add_column("Noise Profile", style="yellow")
    table.add_column("Workload", style="yellow")
    table.add_column("Total Gates", justify="right", style="green")
    table.add_column("Latency (Mean ± Std Dev)", justify="right", style="green")
    table.add_column("Fidelity %", justify="right", style="bold cyan")
    table.add_column("Avg CPU %", justify="right", style="magenta")
    table.add_column("RAM Status", style="blue")
    table.add_column("Status", style="bold")
    
    successful_runs = []
    
    for r in results:
        status_text = "[green]SUCCESS[/green]" if r["success"] else "[red]FAILED[/red]"
        ram_color = "green" if r["ram_status"] == "SAFE" else "red"
        
        method_str = r.get("method", "statevector")
        if method_str in ('mps', 'matrix_product_state') and r.get("bond_dimension"):
            method_str = f"mps (chi={r['bond_dimension']})"
            
        noise_str = r.get("noise_level", "none")
        fidelity_val = r.get("fidelity", 100.0)
        fidelity_str = f"{fidelity_val:.2f}%" if r["success"] else "-"
        
        mean_lat = r.get("mean_latency", r.get("latency", 0.0))
        std_lat = r.get("std_latency", 0.0)
        latency_str = f"{mean_lat:.4f} ± {std_lat:.4f}s" if r["success"] else "-"
        
        table.add_row(
            str(r["qubits"]),
            method_str,
            noise_str,
            r.get("workload_label", "QFT"),
            str(r["gates"]),
            latency_str,
            fidelity_str,
            f"{r['cpu_usage']:.1f}%" if r["success"] else "-",
            f"[{ram_color}]{r['ram_status']}[/{ram_color}]",
            status_text
        )
        
        if r["success"]:
            successful_runs.append(r)
            
    console.print(table)
    console.print()
    
    if not successful_runs:
        console.print("[bold red]All benchmark simulations failed or were skipped due to memory safety limits.[/bold red]")
        return
        
    # Calculate overall score based on the highest successful run
    best_run = max(successful_runs, key=lambda x: x["qubits"])
    max_qubits = best_run["qubits"]
    gates = best_run["gates"]
    mean_latency = best_run.get("mean_latency", best_run.get("latency", 0.0))
    
    breakdown = calculate_scoring_breakdown(max_qubits, gates, mean_latency)
    score = breakdown["composite_score"]
    category = categorize_score(score)
    
    # Tier descriptions and styling
    tier_colors = {
        "Entry-Level": "green",
        "Mid-Range": "cyan",
        "High-Performance": "magenta",
        "Extreme Workstation": "bold yellow"
    }
    color = tier_colors.get(category, "white")
    
    panel_content = Text()
    panel_content.append("QuaComp Composite Score: ", style="bold")
    panel_content.append(f"{score:,.2f}\n", style=f"bold {color}")
    panel_content.append("  (Project-Specific Heuristic Score)\n", style="dim italic white")
    panel_content.append(f"  - Capacity Metric (2^n): {breakdown['capacity_metric']:,.0f}\n", style="dim cyan")
    panel_content.append(f"  - Throughput Metric: {breakdown['throughput_metric']:,.2f} gates/sec\n", style="dim cyan")
    panel_content.append("Performance Category: ", style="bold")
    panel_content.append(f"{category}\n", style=f"bold {color}")
    
    method_used = best_run.get("method", "statevector")
    if method_used in ('mps', 'matrix_product_state') and best_run.get("bond_dimension"):
        method_used = f"MPS (max_bond_dimension={best_run['bond_dimension']})"
    panel_content.append(f"Simulation Method: {method_used}\n", style="cyan")
    
    noise_used = best_run.get("noise_level", "none")
    if noise_used != "none":
        panel_content.append(f"NISQ Noise Profile: {noise_used} (synthetic representative)\n", style="bold yellow")
        panel_content.append(f"Quantum State Fidelity: {best_run['fidelity']:.2f}%\n", style="bold cyan")
        panel_content.append(f"CPU Computation Overhead: +{best_run['overhead_ratio']:.2f}%\n", style="magenta")
        
    ram_savings = best_run.get("ram_savings", {})
    if ram_savings:
        savings_gb = ram_savings["savings_bytes"] / (1024 ** 3)
        panel_content.append(f"MPS RAM Efficiency: {ram_savings['savings_percent']:.2f}% savings (Saved ~{savings_gb:.4f} GB vs Statevector)\n", style="bold green")
        
    runs_cnt = best_run.get("runs_count", 1)
    std_lat = best_run.get("std_latency", 0.0)
    panel_content.append(f"Statistical Repeatability: {runs_cnt} runs (Mean: {mean_latency:.4f}s, Std Dev: {std_lat:.4f}s)\n", style="dim green")
    panel_content.append(f"Max Qubits Simulated: {max_qubits} qubits (using {gates} gates)", style="italic")
    
    console.print(Panel(panel_content, title="[bold gold3]Final Benchmark Report[/bold gold3]", border_style=color, expand=False))

def main():
    parser = argparse.ArgumentParser(description="QuaComp Quantum Simulator Benchmark CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true", help="Quick benchmark on 10, 15, and 20 qubits.")
    group.add_argument("--full", action="store_true", help="Incremental stress test starting from 10 qubits until memory threshold.")
    group.add_argument("--custom", action="store_true", help="Custom simulation configuration.")
    
    parser.add_argument("--qubits", type=int, default=10, help="Number of qubits for custom run (default 10).")
    parser.add_argument("--type", choices=["shallow", "deep", "qft"], default="qft", help="Workload type (default qft).")
    parser.add_argument("--depth", type=int, default=10, help="Depth for deep workload (default 10).")
    parser.add_argument("--method", choices=["statevector", "mps"], default="statevector", help="Simulation method (default statevector).")
    parser.add_argument("--bond-dim", type=int, default=64, help="Max bond dimension for MPS simulation (default 64).")
    parser.add_argument("--noise-level", choices=["none", "low", "medium", "high"], default="none", help="NISQ noise model preset level (default none).")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark iterations per circuit for statistical reproducibility (default 3).")
    parser.add_argument("--chart", action="store_true", help="Generate visualization chart PNG images in results directory.")
    parser.add_argument("--export", choices=["json", "md", "all"], default="all", help="Export results format (default all).")
    
    args = parser.parse_args()
    
    console.print(BANNER)
    print_system_info()
    
    results = []
    
    if args.quick:
        console.print(f"[bold yellow]Executing Quick Benchmark Suite (Qubits: 10, 15, 20) [Method: {args.method.upper()}, Noise: {args.noise_level.upper()}, Runs: {args.runs}]...[/bold yellow]\n")
        qubits_list = [10, 15, 20]
        
        for q in qubits_list:
            with Status(f"Running simulation for {q} qubits ({args.runs} runs)...", console=console) as status:
                res = run_single_simulation(q, "qft", 0, args.method, args.bond_dim, args.noise_level, args.runs)
                res["workload_label"] = "QFT"
                results.append(res)
                if not res["success"]:
                    console.print(f"[bold red]Skipping remaining runs due to limit/warning at {q} qubits:[/bold red] {res['error']}")
                    break
                    
    elif args.full:
        console.print(f"[bold yellow]Executing Full Incremental Stress Test (starting from 10 qubits) [Method: {args.method.upper()}, Noise: {args.noise_level.upper()}, Runs: {args.runs}]...[/bold yellow]\n")
        q = 10
        # If method is MPS, let's limit full benchmark to 35 qubits to prevent excessive CPU runtime
        max_limit = 35 if args.method == 'mps' else 100
        while q <= max_limit:
            with Status(f"Running simulation for {q} qubits ({args.runs} runs)...", console=console) as status:
                res = run_single_simulation(q, "qft", 0, args.method, args.bond_dim, args.noise_level, args.runs)
                res["workload_label"] = "QFT"
                results.append(res)
                if not res["success"]:
                    console.print(f"[bold red]Stress test stopped at {q} qubits:[/bold red] {res['error']}")
                    break
                q += 1
                
    elif args.custom:
        console.print(f"[bold yellow]Executing Custom Simulation (Qubits: {args.qubits}, Workload: {args.type.upper()}, Method: {args.method.upper()}, Noise: {args.noise_level.upper()}, Runs: {args.runs})...[/bold yellow]\n")
        with Status(f"Running simulation for {args.qubits} qubits ({args.runs} runs)...", console=console) as status:
            res = run_single_simulation(args.qubits, args.type, args.depth, args.method, args.bond_dim, args.noise_level, args.runs)
            res["workload_label"] = args.type.upper()
            if args.type == "deep":
                res["workload_label"] += f" (d={args.depth})"
            results.append(res)
            if not res["success"]:
                console.print(f"[bold red]Simulation aborted:[/bold red] {res['error']}")
                
    display_results(results)
    
    # Export results and charts if successful runs or if results exist
    if results:
        system_metadata = get_system_metadata()
        generated_charts = []
        if args.chart:
            generated_charts = generate_benchmark_charts(results, system_metadata)
            if generated_charts:
                console.print(f"[bold green]Benchmark charts successfully generated in results/ directory:[/bold green]")
                for cpath in generated_charts:
                    console.print(f"  - {cpath}")
                    
        if args.export in ("json", "all"):
            json_path = export_to_json(results, system_metadata)
            console.print(f"[bold green]JSON report exported to:[/bold green] {json_path}")
        if args.export in ("md", "all"):
            md_path = export_to_markdown(results, system_metadata, generated_charts=generated_charts)
            console.print(f"[bold green]Markdown report exported to:[/bold green] {md_path}")

if __name__ == "__main__":
    main()
