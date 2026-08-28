import sys
from typing import List, Dict, Any, Optional
from rich.console import Console

from src.reporter.charts import generate_comparison_charts
from src.comparator.differ import load_benchmark_json, compare_benchmarks, resolve_target_profile
from src.comparator.reporter import (
    render_comparison_terminal,
    export_comparison_to_json,
    export_comparison_to_markdown
)

default_console = Console()

def handle_comparison_mode(
    args: Any, 
    current_run_results: Optional[List[Dict[str, Any]]] = None,
    console: Optional[Console] = None
) -> None:
    """
    Handle relative comparison logic between two JSON files or current live benchmark and target JSON.
    """
    c = console or default_console
    compare_args = args.compare if isinstance(args.compare, list) else []
    
    base_data = None
    target_data = None
    base_file_path = None
    target_file_path = None
    
    if len(compare_args) == 2:
        base_file_path = compare_args[0]
        target_file_path = compare_args[1]
        base_data = load_benchmark_json(base_file_path)
        target_data = load_benchmark_json(target_file_path)
    elif len(compare_args) == 1:
        if current_run_results:
            target_file_path = resolve_target_profile(compare_args[0])
            target_data = load_benchmark_json(target_file_path)
        else:
            base_file_path = compare_args[0]
            base_data = load_benchmark_json(base_file_path)
            if not args.target:
                c.print("[bold red]Error:[/bold red] When providing only 1 JSON file to --compare, you must also specify --target <preset_or_path>.")
                sys.exit(1)
            target_file_path = resolve_target_profile(args.target)
            target_data = load_benchmark_json(target_file_path)
    elif len(compare_args) == 0:
        if current_run_results:
            if not args.target:
                c.print("[bold red]Error:[/bold red] --compare in live mode requires --target <preset_alias_or_path>.")
                sys.exit(1)
            target_file_path = resolve_target_profile(args.target)
            target_data = load_benchmark_json(target_file_path)
        else:
            c.print("[bold red]Error:[/bold red] --compare requires either 2 files, 1 file + --target, or a live benchmark execution.")
            sys.exit(1)
            
    # If comparing live benchmark results against a target JSON
    if current_run_results and base_data is None:
        from src.profiler.telemetry import get_system_metadata
        base_data = {
            "results": current_run_results,
            "system_metadata": get_system_metadata()
        }
        
    # Perform mathematical diffing
    diff_data = compare_benchmarks(
        base=base_data,
        target=target_data
    )
    
    # Render terminal presentation
    render_comparison_terminal(diff_data, console=c)
    
    # Generate optional comparison charts
    gen_comp_charts = []
    if getattr(args, 'chart', False):
        gen_comp_charts = generate_comparison_charts(diff_data)
        if gen_comp_charts:
            c.print("[bold green]Comparison charts generated:[/bold green]")
            for cp in gen_comp_charts:
                c.print(f"  - {cp}")
                
    # Export reports
    if args.export in ("json", "all"):
        json_path = export_comparison_to_json(diff_data)
        c.print(f"[bold green]Comparison JSON report exported to:[/bold green] {json_path}")
    if args.export in ("md", "all"):
        md_path = export_comparison_to_markdown(diff_data, generated_charts=gen_comp_charts)
        c.print(f"[bold green]Comparison Markdown report exported to:[/bold green] {md_path}")
