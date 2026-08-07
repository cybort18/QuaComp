import argparse
import sys
import time
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.status import Status

from src.profiler.memory import check_memory_safety, estimate_qubit_ram
from src.profiler.telemetry import get_system_metadata, get_cpu_utilization
from src.engine.circuits import generate_shallow_circuit, generate_deep_circuit, generate_qft_circuit
from src.engine.simulator import run_simulation
from src.scorer.calculator import calculate_qsim_score, categorize_score

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

def run_single_simulation(qubits: int, workload_type: str, depth: int) -> Dict[str, Any]:
    """Execute a single simulation step with safety checks and telemetry collection."""
    # 1. Memory safety check
    is_safe, msg = check_memory_safety(qubits)
    if not is_safe:
        return {
            "qubits": qubits,
            "success": False,
            "latency": 0.0,
            "gates": 0,
            "cpu_usage": 0.0,
            "ram_status": "UNSAFE",
            "error": msg
        }
        
    # 2. Get baseline CPU usage
    cpu_before = get_cpu_utilization()["overall_percent"]
    
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
            "success": False,
            "latency": 0.0,
            "gates": 0,
            "cpu_usage": 0.0,
            "ram_status": "FAIL_GEN",
            "error": f"Circuit generation error: {e}"
        }
        
    gate_count = circuit.size()
    
    # 4. Execute simulation
    sim_result = run_simulation(circuit)
    
    # 5. Get CPU usage after
    cpu_after = get_cpu_utilization()["overall_percent"]
    avg_cpu = (cpu_before + cpu_after) / 2
    
    if sim_result["success"]:
        return {
            "qubits": qubits,
            "success": True,
            "latency": sim_result["latency"],
            "gates": gate_count,
            "cpu_usage": avg_cpu,
            "ram_status": "SAFE",
            "error": None
        }
    else:
        return {
            "qubits": qubits,
            "success": False,
            "latency": sim_result["latency"],
            "gates": gate_count,
            "cpu_usage": avg_cpu,
            "ram_status": "FAIL_EXEC",
            "error": sim_result["error"]
        }

def display_results(results: List[Dict[str, Any]]):
    """Present simulation results in a beautiful table and calculate overall score."""
    table = Table(title="QuaComp Benchmark Results", show_header=True, header_style="bold blue")
    table.add_column("Qubits", justify="right", style="cyan")
    table.add_column("Workload", style="yellow")
    table.add_column("Total Gates", justify="right", style="green")
    table.add_column("Latency (s)", justify="right", style="green")
    table.add_column("Avg CPU %", justify="right", style="magenta")
    table.add_column("RAM Status", style="blue")
    table.add_column("Status", style="bold")
    
    successful_runs = []
    
    for r in results:
        status_text = "[green]SUCCESS[/green]" if r["success"] else "[red]FAILED[/red]"
        ram_color = "green" if r["ram_status"] == "SAFE" else "red"
        
        table.add_row(
            str(r["qubits"]),
            r.get("workload_label", "QFT"),
            str(r["gates"]),
            f"{r['latency']:.4f}" if r["success"] else "-",
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
    latency = best_run["latency"]
    
    score = calculate_qsim_score(max_qubits, gates, latency)
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
    panel_content.append("QuaComp Score: ", style="bold")
    panel_content.append(f"{score:,.2f}\n", style=f"bold {color}")
    panel_content.append("Performance Category: ", style="bold")
    panel_content.append(f"{category}\n", style=f"bold {color}")
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
    
    args = parser.parse_args()
    
    console.print(BANNER)
    print_system_info()
    
    results = []
    
    if args.quick:
        console.print("[bold yellow]Executing Quick Benchmark Suite (Qubits: 10, 15, 20)...[/bold yellow]\n")
        qubits_list = [10, 15, 20]
        
        for q in qubits_list:
            with Status(f"Running simulation for {q} qubits...", console=console) as status:
                res = run_single_simulation(q, "qft", 0)
                res["workload_label"] = "QFT"
                results.append(res)
                if not res["success"]:
                    console.print(f"[bold red]Skipping remaining runs due to OOM warning/error at {q} qubits:[/bold red] {res['error']}")
                    break
                    
    elif args.full:
        console.print("[bold yellow]Executing Full Incremental Stress Test (starting from 10 qubits)...[/bold yellow]\n")
        q = 10
        while True:
            with Status(f"Running simulation for {q} qubits...", console=console) as status:
                res = run_single_simulation(q, "qft", 0)
                res["workload_label"] = "QFT"
                results.append(res)
                if not res["success"]:
                    console.print(f"[bold red]Stress test stopped at {q} qubits:[/bold red] {res['error']}")
                    break
                q += 1
                
    elif args.custom:
        console.print(f"[bold yellow]Executing Custom Simulation (Qubits: {args.qubits}, Workload: {args.type.upper()})...[/bold yellow]\n")
        with Status(f"Running simulation for {args.qubits} qubits...", console=console) as status:
            res = run_single_simulation(args.qubits, args.type, args.depth)
            res["workload_label"] = args.type.upper()
            if args.type == "deep":
                res["workload_label"] += f" (d={args.depth})"
            results.append(res)
            if not res["success"]:
                console.print(f"[bold red]Simulation aborted:[/bold red] {res['error']}")
                
    display_results(results)

if __name__ == "__main__":
    main()
