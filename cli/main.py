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
    run_single_simulation,
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
    param_group.add_argument("--type", choices=["shallow", "deep", "qft"], default="qft", help="Workload type (default qft).")
    param_group.add_argument("--depth", type=int, default=10, help="Depth for deep workload (default 10).")
    param_group.add_argument("--method", choices=["statevector", "mps"], default="statevector", help="Simulation method (default statevector).")
    param_group.add_argument("--bond-dim", type=int, default=64, help="Max bond dimension for MPS simulation (default 64).")
    param_group.add_argument("--device", choices=["cpu", "gpu", "multi_gpu"], default="cpu", help="Simulation compute device backend (default cpu).")
    param_group.add_argument("--gpu", action="store_true", help="Shorthand flag to enable GPU acceleration (--device gpu).")
    param_group.add_argument("--multi-gpu", action="store_true", help="Enable multi-GPU distributed simulation backend.")
    param_group.add_argument("--workers", type=int, default=1, help="Parallel distributed worker threads/processes for batch execution (default 1).")
    param_group.add_argument("--entropy", action="store_true", help="Calculate bipartite Von Neumann entanglement entropy and simulation complexity.")
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
    
    return parser

def main():
    """Main CLI entrypoint for QuaComp."""
    parser = build_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments: Must specify at least one benchmark mode OR --compare
    if not (args.quick or args.full or args.custom or args.compare is not None):
        display_help_notice(console=console)
        return
        
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
