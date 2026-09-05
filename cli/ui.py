from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from src.profiler.telemetry import get_system_metadata
from src.profiler.gpu import get_gpu_metadata
from src.scorer.calculator import calculate_qsim_score, categorize_score, calculate_scoring_breakdown

default_console = Console()

BANNER = r"""
 [bold cyan]  ____             ____                     [/bold cyan]
 [bold cyan] / __ \__  ______ / ___| ___  _ __ ___  _ __ [/bold cyan]
 [bold cyan]/ / / / / / / __ `/ /   / _ \| '_ ` _ \| '_ \ [/bold cyan]
 [bold cyan]/ /_/ / /_/ / /_/ / |__| (_) | | | | | | |_) |[/bold cyan]
 [bold cyan]\___\_\__,_/\__,_/\____/\___/|_| |_| |_| .__/ [/bold cyan]
 [bold cyan]                                       |_|    [/bold cyan]
 [bold yellow]======= Quantum Computer Simulation Benchmark =======[/bold yellow]
"""

def print_system_info(console: Optional[Console] = None) -> None:
    """Print system hardware and environment metadata in a clean Rich table."""
    c = console or default_console
    metadata = get_system_metadata()
    gpu_meta = get_gpu_metadata()
    table = Table(title="System Metadata & Telemetry", show_header=True, header_style="bold magenta", expand=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("System Value", style="green")
    
    table.add_row("CPU Name", metadata["cpu_name"])
    table.add_row("CPU Physical / Logical Cores", f"{metadata['cpu_count_physical']} cores / {metadata['cpu_count_logical']} threads")
    table.add_row("Total Physical RAM", f"{metadata['total_ram_gb']:.2f} GB")
    
    if gpu_meta["has_gpu"]:
        aer_status = "Available (CUDA)" if gpu_meta["aer_gpu_supported"] else "CPU-only Backend"
        gpu_label = gpu_meta["gpu_name"]
        if gpu_meta.get("gpu_count", 1) > 1:
            gpu_label = f"{gpu_meta['gpu_count']}x GPUs ({gpu_meta['gpu_name']})"
        table.add_row("GPU Hardware", f"{gpu_label} [dim](Aer: {aer_status})[/dim]")
        if gpu_meta["total_vram_gb"] > 0:
            table.add_row("Total GPU VRAM", f"{gpu_meta['total_vram_gb']:.2f} GB" + (" (Aggregated)" if gpu_meta.get("multi_gpu_supported") else ""))
            
    table.add_row("Operating System", f"{metadata['os_name']} ({metadata['os_release']})")
    table.add_row("Python Version", metadata["python_version"])
    
    c.print(table)
    c.print()

def display_results(results: List[Dict[str, Any]], console: Optional[Console] = None) -> None:
    """Display benchmark execution results, entanglement analysis, and final score summary."""
    c = console or default_console
    table = Table(title="Simulation Benchmark Results", show_header=True, header_style="bold magenta", expand=False)
    table.add_column("Qubits", style="cyan", justify="center")
    table.add_column("Workload", style="white", justify="center")
    table.add_column("Method", style="magenta", justify="center")
    table.add_column("Device", style="bold cyan", justify="center")
    table.add_column("Noise", style="yellow", justify="center")
    table.add_column("Gates", style="white", justify="right")
    table.add_column("Latency (Mean +/- Std)", style="yellow", justify="right")
    table.add_column("CPU", style="cyan", justify="right")
    table.add_column("Fidelity", style="green", justify="right")
    table.add_column("RAM", style="bold", justify="center")
    table.add_column("Outcome", style="bold", justify="center")
    
    successful_runs = [r for r in results if r["success"]]
    
    for r in results:
        ram_style = "green" if r["ram_status"] == "SAFE" else "red"
        outcome_str = "[bold green]PASS[/bold green]" if r["success"] else "[bold red]FAIL[/bold red]"
        
        method_label = r.get("method", "statevector").upper()
        if method_label == "MATRIX_PRODUCT_STATE":
            method_label = "MPS"
        if r.get("bond_dimension"):
            method_label += f" (chi={r['bond_dimension']})"
            
        noise_label = r.get("noise_level", "none").upper()
        fidelity_str = f"{r['fidelity']:.1f}%" if noise_label != "NONE" else "100.0%"
        
        runs_count = r.get("runs_count", 1)
        mean_lat = r.get("mean_latency", r.get("latency", 0.0))
        std_lat = r.get("std_latency", 0.0)
        
        if runs_count > 1 and r["success"]:
            latency_str = f"{mean_lat:.4f}s [dim]+/-{std_lat:.3f}s[/dim]"
        elif r["success"]:
            latency_str = f"{mean_lat:.4f}s"
        else:
            latency_str = "-"
            
        device_label = r.get("device", "CPU").upper()
        
        table.add_row(
            str(r["qubits"]),
            r["workload_label"],
            method_label,
            f"[bold cyan]{device_label}[/bold cyan]" if device_label == "GPU" else "[dim]CPU[/dim]",
            noise_label,
            str(r["gates"]) if r["success"] else "-",
            latency_str,
            f"{r['cpu_usage']:.1f}%" if r["success"] else "-",
            fidelity_str if r["success"] else "-",
            f"[{ram_style}]{r['ram_status']}[/{ram_style}]",
            outcome_str
        )
        
    c.print(table)
    c.print()
    
    if not successful_runs:
        c.print(Panel("[bold red]No simulations completed successfully. Unable to calculate benchmark score.[/bold red]", title="Score Summary", border_style="red", expand=False))
        return

    # Render Entanglement & Simulation Hardness Table if metrics are present
    entropy_runs = [r for r in successful_runs if r.get("entanglement_metrics") and "von_neumann_entropy" in r["entanglement_metrics"]]
    if entropy_runs:
        ent_table = Table(title="Entanglement Entropy & Simulation Hardness Analysis", show_header=True, header_style="bold magenta", expand=False)
        ent_table.add_column("Qubits", style="cyan", justify="center")
        ent_table.add_column("Workload", style="white", justify="center")
        ent_table.add_column("Von Neumann Entropy (SvN)", style="bold yellow", justify="right")
        ent_table.add_column("Max Bipartite Bound", style="dim white", justify="right")
        ent_table.add_column("Schmidt Rank", style="cyan", justify="center")
        ent_table.add_column("Entanglement Regime", style="green", justify="center")
        ent_table.add_column("MPS Complexity Tier", style="bold magenta", justify="center")
        
        for er in entropy_runs:
            em = er["entanglement_metrics"]
            ent_table.add_row(
                str(er["qubits"]),
                er["workload_label"],
                f"{em['von_neumann_entropy']:.4f} bits",
                f"{em['max_possible_entropy']:.1f}",
                str(em["schmidt_rank"]),
                em["entanglement_regime"],
                em["mps_hardness"]
            )
        c.print(ent_table)
        c.print()

    # Render Hardware Energy & Power Telemetry Table if present
    energy_runs = [r for r in successful_runs if r.get("energy_metrics") and "total_energy_joules" in r["energy_metrics"]]
    if energy_runs:
        en_table = Table(title="Hardware Power & Energy Telemetry (EQO)", show_header=True, header_style="bold green", expand=False)
        en_table.add_column("Qubits", style="cyan", justify="center")
        en_table.add_column("Workload", style="white", justify="center")
        en_table.add_column("Avg Power", style="yellow", justify="right")
        en_table.add_column("Total Energy", style="bold yellow", justify="right")
        en_table.add_column("EQO (uJ/Gate)", style="bold green", justify="right")
        en_table.add_column("Telemetry Backend", style="dim cyan", justify="center")
        
        for enr in energy_runs:
            enm = enr["energy_metrics"]
            en_table.add_row(
                str(enr["qubits"]),
                enr["workload_label"],
                f"{enm.get('average_power_watts', 0.0):.2f} W",
                f"{enm.get('total_energy_joules', 0.0):.4f} J",
                f"{enm.get('eqo_microjoules', 0.0):.2f} uJ",
                str(enm.get("energy_backend", "Generic Model"))
            )
        c.print(en_table)
        c.print()

    # Render Quantum Volume Certification Table if present
    qv_runs = [r for r in successful_runs if r.get("qv_metrics") and "heavy_output_probability" in r["qv_metrics"]]
    if qv_runs:
        qv_table = Table(title="Quantum Volume (QV) Verification", show_header=True, header_style="bold blue", expand=False)
        qv_table.add_column("Qubits", style="cyan", justify="center")
        qv_table.add_column("Heavy Prob (h_prob)", style="bold yellow", justify="right")
        qv_table.add_column("2-Sigma Lower Bound", style="white", justify="right")
        qv_table.add_column("Threshold", style="dim white", justify="center")
        qv_table.add_column("Status", style="bold green", justify="center")
        for qvr in qv_runs:
            qvm = qvr["qv_metrics"]
            status_str = "[bold green]CERTIFIED (PASSED)[/bold green]" if qvm["qv_certified"] else "[bold red]FAILED[/bold red]"
            qv_table.add_row(
                str(qvr["qubits"]),
                f"{qvm['heavy_output_probability']:.4f}",
                f"{qvm['lower_confidence_bound_2sigma']:.4f}",
                "> 0.6667",
                status_str
            )
        c.print(qv_table)
        c.print()

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
        
    if best_run.get("entanglement_metrics") and "von_neumann_entropy" in best_run["entanglement_metrics"]:
        em = best_run["entanglement_metrics"]
        panel_content.append(f"Entanglement Entropy: S_vN = {em['von_neumann_entropy']:.4f} bits (Schmidt Rank: {em['schmidt_rank']} | {em['entanglement_regime']})\n", style="bold red")
        panel_content.append(f"MPS Simulation Complexity: {em['mps_hardness']}\n", style="bold magenta")
        
    runs_cnt = best_run.get("runs_count", 1)
    std_lat = best_run.get("std_latency", 0.0)
    panel_content.append(f"Statistical Repeatability: {runs_cnt} runs (Mean: {mean_latency:.4f}s, Std Dev: {std_lat:.4f}s)\n", style="dim green")
    panel_content.append(f"Max Qubits Simulated: {max_qubits} qubits (using {gates} gates)", style="italic")
    
    c.print(Panel(panel_content, title="[bold gold3]Final Benchmark Report[/bold gold3]", border_style=color, expand=False))

def display_help_notice(console: Optional[Console] = None) -> None:
    """Display quick guidance notice when no execution flags are passed."""
    c = console or default_console
    c.print(BANNER)
    c.print("[bold yellow]Please select a benchmark mode or comparison mode:[/bold yellow]")
    c.print("  [cyan]quacomp --quick[/cyan]                                                (Quick 10, 15, 20 qubits benchmark)")
    c.print("  [cyan]quacomp --quick --gpu[/cyan]                                          (Quick benchmark with GPU acceleration)")
    c.print("  [cyan]quacomp --quick --entropy[/cyan]                                      (Quick benchmark with Entanglement Entropy)")
    c.print("  [cyan]quacomp --full[/cyan]                                                 (Incremental stress test)")
    c.print("  [cyan]quacomp --compare <file1.json> <file2.json>[/cyan]                     (Compare two benchmark results)")
    c.print("  [cyan]quacomp --compare results/report.json --target apple_m3[/cyan]         (Compare with reference profile)")
    c.print("  [cyan]quacomp --quick --compare results/samples/example_apple_m3.json[/cyan] (Run benchmark & compare)")
    c.print("\nRun [bold green]quacomp --help[/bold green] for full options.\n")
