import argparse
import sys
from typing import List, Dict, Any

from rich.console import Console

from src.profiler.telemetry import get_system_metadata
from src.reporter.json_exporter import export_to_json
from src.reporter.md_exporter import export_to_markdown
from src.reporter.charts import generate_benchmark_charts

from cli.ui import BANNER, print_system_info, display_results, display_help_notice
from cli.runner import (
    run_quick_benchmark,
    run_full_stress_test,
    run_custom_simulation
)
from cli.comparison import handle_comparison_mode

console = Console()

def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the command line argument parser for QuaComp."""
    parser = argparse.ArgumentParser(description="QuaComp Quantum Simulator Benchmark & Comparison CLI")
    
    # Benchmark execution modes
    mode_group = parser.add_argument_group("Benchmark Modes")
    mode_group.add_argument("--quick", action="store_true", help="Quick benchmark on 10, 15, and 20 qubits.")
    mode_group.add_argument("--full", action="store_true", help="Incremental stress test starting from 10 qubits until memory threshold.")
    mode_group.add_argument("--custom", action="store_true", help="Custom simulation configuration.")
    
    # Benchmark parameters
    param_group = parser.add_argument_group("Simulation Parameters")
    param_group.add_argument("--qubits", type=int, default=10, help="Number of qubits for custom run (default 10).")
    param_group.add_argument("--type", choices=["shallow", "deep", "qft", "vqe", "qaoa", "qv"], default="qft", help="Workload type (default qft).")
    param_group.add_argument("--vqe", action="store_true", help="Shorthand for VQE variational ansatz workload (--type vqe).")
    param_group.add_argument("--qaoa", action="store_true", help="Shorthand for QAOA Max-Cut variational workload (--type qaoa).")
    param_group.add_argument("--qv", action="store_true", help="Shorthand for Quantum Volume square model benchmark (--type qv).")
    param_group.add_argument("--depth", type=int, default=10, help="Depth for deep workload (default 10).")
    param_group.add_argument("--method", choices=["statevector", "mps"], default="statevector", help="Simulation method (default statevector).")
    param_group.add_argument("--bond-dim", type=int, default=64, help="Max bond dimension for MPS simulation (default 64).")
    param_group.add_argument("--device", choices=["cpu", "gpu", "multi_gpu"], default="cpu", help="Simulation compute device backend (default cpu).")
    param_group.add_argument("--gpu", action="store_true", help="Shorthand flag to enable GPU acceleration (--device gpu).")
    param_group.add_argument("--multi-gpu", action="store_true", help="Enable multi-GPU distributed simulation backend.")
    param_group.add_argument("--state-slicing", action="store_true", help="Enable distributed statevector slicing model parallelism across VRAM.")
    param_group.add_argument("--blocking-qubits", type=int, default=None, help="Statevector chunk slice size in qubits for distributed model parallelism.")
    param_group.add_argument("--workers", type=int, default=1, help="Parallel distributed worker threads/processes for batch execution (default 1).")
    param_group.add_argument("--entropy", action="store_true", help="Calculate bipartite Von Neumann entanglement entropy and simulation complexity.")
    param_group.add_argument("--noise-level", choices=["none", "low", "medium", "high"], default="none", help="NISQ noise model preset level (default none).")
    param_group.add_argument("--runs", type=int, default=3, help="Number of benchmark iterations per circuit (default 3).")
    
    # Comparison Options
    comp_group = parser.add_argument_group("Comparison Options")
    comp_group.add_argument("--compare", nargs="*", metavar="FILE", help="Compare two benchmark JSON result files, or compare live run with a target JSON.")
    comp_group.add_argument("--target", type=str, default=None, help="Target reference preset alias (apple_m3, ryzen3_5300u, ryzen7_5800h) or path.")
    comp_group.add_argument("--fetch-baselines", action="store_true", help="Synchronize enterprise comparison baselines from QuaComp remote registry.")
    
    # Reporting options
    report_group = parser.add_argument_group("Reporting Options")
    report_group.add_argument("--chart", action="store_true", help="Generate visualization chart PNG images in results directory.")
    report_group.add_argument("--export", choices=["json", "md", "all"], default="all", help="Export results format (default all).")
    
    return parser

def validate_cli_arguments(args: argparse.Namespace) -> None:
    """Validate numerical boundary conditions for CLI simulation arguments."""
    if getattr(args, 'qubits', None) is not None and args.qubits < 1:
        raise ValueError("Argument --qubits must be a positive integer >= 1.")
    if getattr(args, 'depth', None) is not None and args.depth < 0:
        raise ValueError("Argument --depth must be a non-negative integer >= 0.")
    if getattr(args, 'bond_dim', None) is not None and args.bond_dim < 1:
        raise ValueError("Argument --bond-dim must be a positive integer >= 1.")
    if getattr(args, 'workers', None) is not None and args.workers < 1:
        raise ValueError("Argument --workers must be a positive integer >= 1.")
    if getattr(args, 'runs', None) is not None and args.runs < 1:
        raise ValueError("Argument --runs must be a positive integer >= 1.")
    if getattr(args, 'blocking_qubits', None) is not None and args.blocking_qubits < 1:
        raise ValueError("Argument --blocking-qubits must be a positive integer >= 1.")

def main():
    """Main CLI entrypoint for QuaComp."""
    parser = build_argument_parser()
    args = parser.parse_args()
    
    # Resolve baseline synchronization
    if getattr(args, 'fetch_baselines', False):
        from src.comparator.registry import fetch_remote_baselines, list_available_baselines
        from rich.table import Table
        console.print(BANNER)
        console.print("[bold yellow]Synchronizing QuaComp Enterprise Baseline Registry...[/bold yellow]\n")
        sync_report = fetch_remote_baselines()
        baselines = list_available_baselines()
        
        reg_table = Table(title="QuaComp Enterprise Baseline Registry", show_header=True, header_style="bold cyan")
        reg_table.add_column("Alias", style="bold yellow", justify="left")
        reg_table.add_column("Hardware Platform", style="white", justify="left")
        reg_table.add_column("Device", style="cyan", justify="center")
        reg_table.add_column("Category", style="magenta", justify="center")
        reg_table.add_column("Status", style="green", justify="center")
        
        for b in baselines:
            status_str = "[bold green]Available (Ready)[/bold green]" if b["is_available"] else "[dim]Remote Only[/dim]"
            reg_table.add_row(b["alias"], b["description"], b["device"], b["category"], status_str)
            
        console.print(reg_table)
        mode_note = "Offline Fallback Cache" if sync_report["offline_mode"] else "Remote Cloud Synchronized"
        console.print(f"\n[bold green]Synchronization Complete:[/bold green] {sync_report['total_available']} baseline profiles active ({mode_note}).\n")
        return

    # Resolve workload shorthands
    if args.vqe:
        args.type = "vqe"
    elif args.qaoa:
        args.type = "qaoa"
    elif args.qv:
        args.type = "qv"
        
    if (args.vqe or args.qaoa or args.qv) and not (args.quick or args.full):
        args.custom = True
        
    # Validate arguments: Must specify at least one benchmark mode OR --compare
    if not (args.quick or args.full or args.custom or args.compare is not None or args.fetch_baselines):
        display_help_notice(console=console)
        return
        
    try:
        validate_cli_arguments(args)
    except ValueError as ve:
        console.print(f"[bold red]Argument Validation Error:[/bold red] {str(ve)}")
        sys.exit(1)
        
    console.print(BANNER)
    
    results: List[Dict[str, Any]] = []
    
    if args.multi_gpu or args.device == "multi_gpu":
        effective_device = "multi_gpu"
    elif args.gpu or args.device == "gpu":
        effective_device = "gpu"
    else:
        effective_device = args.device
    
    if args.quick or args.full or args.custom:
        print_system_info(console=console)
        
        if args.quick:
            results = run_quick_benchmark(args, effective_device, console=console)
        elif args.full:
            results = run_full_stress_test(args, effective_device, console=console)
        elif args.custom:
            results = run_custom_simulation(args, effective_device, console=console)
                    
        display_results(results, console=console)
        
        # Export individual benchmark results and charts
        if results:
            system_metadata = get_system_metadata()
            generated_charts = []
            if args.chart:
                generated_charts = generate_benchmark_charts(results, system_metadata)
                if generated_charts:
                    console.print("[bold green]Benchmark charts successfully generated in results/ directory:[/bold green]")
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
            handle_comparison_mode(args, current_run_results=results if results else None, console=console)
        except Exception as e:
            console.print(f"[bold red]Comparison Error:[/bold red] {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main()
