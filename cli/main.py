import argparse
import sys
import time
import os
from typing import List, Dict, Any, Optional
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
from src.reporter.charts import generate_benchmark_charts, generate_comparison_charts
from src.engine.mps import calculate_mps_ram_savings
from src.engine.noise import get_noise_model, calculate_state_fidelity, calculate_overhead_ratio
from src.comparator.differ import load_benchmark_json, compare_benchmarks, resolve_target_profile
from src.comparator.reporter import (
    render_comparison_terminal,
    export_comparison_to_json,
    export_comparison_to_markdown
)

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
    
    # 4. Simulation run with noise and method support (multiple runs)
    noise_model = get_noise_model(noise_level)
    sim_result = run_simulation(
        circuit, 
        method=method, 
        bond_dimension=bond_dimension,
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
        # Measure actual process resident memory usage in bytes
        process = psutil.Process()
        actual_ram_bytes = process.memory_info().rss
        ram_savings = calculate_mps_ram_savings(qubits, actual_ram_bytes)
        
    # 7. Calculate NISQ metrics (Fidelity and CPU Overhead) if noisy
    fidelity = 100.0
    overhead_ratio = 0.0
    if noise_level != "none" and sim_result["success"]:
        # Run ideal simulation to calculate baseline counts and latency
        ideal_sim_result = run_simulation(
            circuit, 
            method=method, 
            bond_dimension=bond_dimension, 
            noise_model=None, 
            noise_level="none",
            runs=1
        )
        if ideal_sim_result["success"]:
            fidelity = calculate_state_fidelity(ideal_sim_result["counts"], sim_result["counts"])
            overhead_ratio = calculate_overhead_ratio(ideal_sim_result["latency"], sim_result["mean_latency"])
    
    mean_lat = sim_result["mean_latency"]
    med_lat = sim_result["median_latency"]
    std_lat = sim_result["std_latency"]
    runs_cnt = sim_result["runs_count"]
    
    return {
        "qubits": qubits,
        "method": method,
        "bond_dimension": bond_dimension if method in ('mps', 'matrix_product_state') else None,
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
        "ram_savings": ram_savings
    }

def display_results(results: List[Dict[str, Any]]) -> None:
    """Display benchmark execution results in a Rich table and summary score panel."""
    table = Table(title="Simulation Benchmark Results", show_header=True, header_style="bold magenta", expand=False)
    table.add_column("Qubits", style="cyan", justify="center")
    table.add_column("Workload", style="white", justify="center")
    table.add_column("Method", style="magenta", justify="center")
    table.add_column("Noise Profile", style="yellow", justify="center")
    table.add_column("Gates", style="white", justify="right")
    table.add_column("Latency (Mean ± Std Dev)", style="yellow", justify="right")
    table.add_column("CPU Usage", style="cyan", justify="right")
    table.add_column("Fidelity", style="green", justify="right")
    table.add_column("RAM Status", style="bold", justify="center")
    table.add_column("Outcome", style="bold", justify="center")
    
    successful_runs = [r for r in results if r["success"]]
    
    for r in results:
        ram_style = "green" if r["ram_status"] == "SAFE" else "red"
        outcome_str = "[bold green]PASS[/bold green]" if r["success"] else "[bold red]FAIL[/bold red]"
        
        method_label = r.get("method", "statevector").upper()
        if method_label == "MATRIX_PRODUCT_STATE":
            method_label = "MPS"
        if r.get("bond_dimension"):
            method_label += f" (χ={r['bond_dimension']})"
            
        noise_label = r.get("noise_level", "none").upper()
        fidelity_str = f"{r['fidelity']:.1f}%" if noise_label != "NONE" else "100.0%"
        
        runs_count = r.get("runs_count", 1)
        mean_lat = r.get("mean_latency", r.get("latency", 0.0))
        std_lat = r.get("std_latency", 0.0)
        
        if runs_count > 1 and r["success"]:
            latency_str = f"{mean_lat:.4f}s [dim]±{std_lat:.3f}s[/dim]"
        elif r["success"]:
            latency_str = f"{mean_lat:.4f}s"
        else:
            latency_str = "—"
            
        table.add_row(
            str(r["qubits"]),
            r["workload_label"],
            method_label,
            noise_label,
            str(r["gates"]) if r["success"] else "—",
            latency_str,
            f"{r['cpu_usage']:.1f}%" if r["success"] else "—",
            fidelity_str if r["success"] else "—",
            f"[{ram_style}]{r['ram_status']}[/{ram_style}]",
            outcome_str
        )
        
    console.print(table)
    console.print()
    
    if not successful_runs:
        console.print(Panel("[bold red]No simulations completed successfully. Unable to calculate benchmark score.[/bold red]", title="Score Summary", border_style="red", expand=False))
        return

    # Calculate final composite heuristic score using best successful run
    best_run = max(successful_runs, key=lambda x: x["qubits"])
    max_qubits = best_run["qubits"]
    gates = best_run["gates"]
    mean_latency = best_run.get("mean_latency", best_run.get("latency", 0.0))
    
    score = calculate_qsim_score(max_qubits, gates, mean_latency)
    category = categorize_score(score)
    breakdown = calculate_scoring_breakdown(max_qubits, gates, mean_latency)
    
    tier_colors = {
        "Entry-Level": "blue",
        "Mid-Range": "green",
        "High-Performance": "yellow",
        "Extreme Workstation": "magenta"
    }
    color = tier_colors.get(category, "white")
    
    panel_content = Text()
    panel_content.append("Final Composite Heuristic Score: ", style="bold")
    panel_content.append(f"{score:,.2f}\n", style=f"bold {color}")
    panel_content.append("Score Formulation: (2^Qubits * 10) + (Gates / Latency)\n", style="dim white")
    panel_content.append("Scoring Metric Breakdown:\n", style="bold white")
    panel_content.append(f"  - Capacity Metric:  {breakdown['capacity_metric']:,.0f} (2^{max_qubits})\n", style="dim cyan")
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

def handle_comparison_mode(args, current_run_results: Optional[List[Dict[str, Any]]] = None) -> None:
    """
    Handle relative comparison logic between two JSON files or current live benchmark and target JSON.
    """
    compare_args = args.compare if isinstance(args.compare, list) else []
    
    base_data = None
    target_data = None
    base_file_path = None
    target_file_path = None
    
    if current_run_results:
        # Case A: Live benchmark just ran -> use it as base!
        system_metadata = get_system_metadata()
        best_run = max([r for r in current_run_results if r["success"]], key=lambda x: x["qubits"], default=None)
        if best_run:
            score = calculate_qsim_score(best_run["qubits"], best_run["gates"], best_run["latency"])
            cat = categorize_score(score)
            breakdown = calculate_scoring_breakdown(best_run["qubits"], best_run["gates"], best_run["latency"])
        else:
            score = 0.0
            cat = "Unknown"
            breakdown = {"capacity_metric": 0.0, "throughput_metric": 0.0}
            
        base_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "final_score": score,
            "final_composite_score": score,
            "scoring_breakdown": breakdown,
            "performance_category": cat,
            "max_qubits_simulated": best_run["qubits"] if best_run else 0,
            "system_metadata": system_metadata,
            "results": current_run_results
        }
        
        # Determine target from --compare [target] or --target [target]
        target_spec = None
        if len(compare_args) > 0:
            target_spec = compare_args[0]
        elif args.target:
            target_spec = args.target
        else:
            # Default target if none specified: apple_m3 reference
            target_spec = "apple_m3"
            
        target_file_path = resolve_target_profile(target_spec)
        target_data = load_benchmark_json(target_file_path)
        
    else:
        # Case B: Standalone comparison mode (no live benchmark)
        if len(compare_args) >= 2:
            base_file_path = resolve_target_profile(compare_args[0])
            target_file_path = resolve_target_profile(compare_args[1])
        elif len(compare_args) == 1:
            if args.target:
                base_file_path = resolve_target_profile(compare_args[0])
                target_file_path = resolve_target_profile(args.target)
            else:
                # Default base: results/report.json if exists, else compare against apple_m3
                if os.path.exists("results/report.json") and os.path.abspath(compare_args[0]) != os.path.abspath("results/report.json"):
                    base_file_path = os.path.abspath("results/report.json")
                    target_file_path = resolve_target_profile(compare_args[0])
                else:
                    base_file_path = resolve_target_profile(compare_args[0])
                    target_file_path = resolve_target_profile("apple_m3")
        else:
            # No files passed to --compare, check --target and results/report.json
            if not os.path.exists("results/report.json"):
                console.print("[bold red]Error:[/bold red] No local 'results/report.json' found. Please provide two benchmark files to compare:")
                console.print("  [cyan]quacomp --compare path/to/base.json path/to/target.json[/cyan]")
                sys.exit(1)
            base_file_path = os.path.abspath("results/report.json")
            target_spec = args.target if args.target else "apple_m3"
            target_file_path = resolve_target_profile(target_spec)
            
        base_data = load_benchmark_json(base_file_path)
        target_data = load_benchmark_json(target_file_path)
        
    # Perform mathematical comparison
    diff_data = compare_benchmarks(base_data, target_data)
    
    # Render rich comparison table in terminal
    render_comparison_terminal(diff_data, console)
    
    # Generate comparison charts if requested
    gen_comp_charts = []
    if args.chart:
        gen_comp_charts = generate_comparison_charts(diff_data)
        if gen_comp_charts:
            console.print(f"[bold green]Relative comparison charts generated in results/ directory:[/bold green]")
            for cp in gen_comp_charts:
                console.print(f"  - {cp}")
                
    # Export comparison reports
    if args.export in ("json", "all"):
        json_path = export_comparison_to_json(diff_data)
        console.print(f"[bold green]Comparison JSON exported to:[/bold green] {json_path}")
    if args.export in ("md", "all"):
        md_path = export_comparison_to_markdown(diff_data, generated_charts=gen_comp_charts)
        console.print(f"[bold green]Comparison Markdown report exported to:[/bold green] {md_path}")

def main():
    parser = argparse.ArgumentParser(description="QuaComp Quantum Simulator Benchmark & Comparison CLI")
    
    # Benchmark execution modes
    mode_group = parser.add_argument_group("Benchmark Modes")
    mode_group.add_argument("--quick", action="store_true", help="Quick benchmark on 10, 15, and 20 qubits.")
    mode_group.add_argument("--full", action="store_true", help="Incremental stress test starting from 10 qubits until memory threshold.")
    mode_group.add_argument("--custom", action="store_true", help="Custom simulation configuration.")
    
    # Benchmark parameters
    param_group = parser.add_argument_group("Simulation Parameters")
    param_group.add_argument("--qubits", type=int, default=10, help="Number of qubits for custom run (default 10).")
    param_group.add_argument("--type", choices=["shallow", "deep", "qft"], default="qft", help="Workload type (default qft).")
    param_group.add_argument("--depth", type=int, default=10, help="Depth for deep workload (default 10).")
    param_group.add_argument("--method", choices=["statevector", "mps"], default="statevector", help="Simulation method (default statevector).")
    param_group.add_argument("--bond-dim", type=int, default=64, help="Max bond dimension for MPS simulation (default 64).")
    param_group.add_argument("--noise-level", choices=["none", "low", "medium", "high"], default="none", help="NISQ noise model preset level (default none).")
    param_group.add_argument("--runs", type=int, default=3, help="Number of benchmark iterations per circuit (default 3).")
    
    # Comparison Options
    comp_group = parser.add_argument_group("Comparison Options")
    comp_group.add_argument("--compare", nargs="*", metavar="FILE", help="Compare two benchmark JSON result files, or compare live run with a target JSON.")
    comp_group.add_argument("--target", type=str, default=None, help="Target reference preset alias (apple_m3, ryzen3_5300u, ryzen7_5800h) or path.")
    
    # Reporting options
    report_group = parser.add_argument_group("Reporting Options")
    report_group.add_argument("--chart", action="store_true", help="Generate visualization chart PNG images in results directory.")
    report_group.add_argument("--export", choices=["json", "md", "all"], default="all", help="Export results format (default all).")
    
    args = parser.parse_args()
    
    # Validate arguments: Must specify at least one benchmark mode OR --compare
    if not (args.quick or args.full or args.custom or args.compare is not None):
        console.print(BANNER)
        console.print("[bold yellow]Please select a benchmark mode or comparison mode:[/bold yellow]")
        console.print("  [cyan]quacomp --quick[/cyan]                                                (Quick 10, 15, 20 qubits benchmark)")
        console.print("  [cyan]quacomp --full[/cyan]                                                 (Incremental stress test)")
        console.print("  [cyan]quacomp --compare <file1.json> <file2.json>[/cyan]                     (Compare two benchmark results)")
        console.print("  [cyan]quacomp --compare results/report.json --target apple_m3[/cyan]         (Compare with reference profile)")
        console.print("  [cyan]quacomp --quick --compare results/samples/example_apple_m3.json[/cyan] (Run benchmark & compare)")
        console.print("\nRun [bold green]quacomp --help[/bold green] for full options.\n")
        return
        
    console.print(BANNER)
    
    results = []
    
    if args.quick or args.full or args.custom:
        print_system_info()
        
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
        
        # Export individual benchmark results and charts
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
                
    # If comparison was requested (either standalone or with live benchmark)
    if args.compare is not None:
        try:
            handle_comparison_mode(args, current_run_results=results if results else None)
        except Exception as e:
            console.print(f"[bold red]Comparison Error:[/bold red] {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main()
