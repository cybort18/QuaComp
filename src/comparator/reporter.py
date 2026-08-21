import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

def render_comparison_terminal(diff_data: Dict[str, Any], console: Console) -> None:
    """
    Render beautiful side-by-side benchmark comparison tables in the terminal using Rich.
    
    Args:
        diff_data: Comparison dictionary produced by compare_benchmarks().
        console: Rich Console instance.
    """
    lbl_base = diff_data["base_label"]
    lbl_target = diff_data["target_label"]
    meta_base = diff_data.get("base_metadata", {})
    meta_target = diff_data.get("target_metadata", {})
    summary = diff_data["score_summary"]
    matched = diff_data.get("matched_qubits", [])
    
    # 1. Header Banner
    console.print()
    console.print(Panel(
        f"[bold cyan]Base System:[/bold cyan] {lbl_base}\n"
        f"[bold green]Target System:[/bold green] {lbl_target}",
        title="[bold yellow]QuaComp Relative Benchmark Comparison[/bold yellow]",
        border_style="yellow",
        expand=False
    ))
    console.print()
    
    # 2. Hardware Specification Comparison Table
    spec_table = Table(title="Hardware & Environment Comparison", show_header=True, header_style="bold magenta", expand=False)
    spec_table.add_column("Parameter", style="cyan", no_wrap=True)
    spec_table.add_column(f"Base: {lbl_base}", style="white")
    spec_table.add_column(f"Target: {lbl_target}", style="white")
    
    spec_table.add_row("Processor / CPU", meta_base.get("cpu_name", "N/A"), meta_target.get("cpu_name", "N/A"))
    if meta_base.get("gpu_name") or meta_target.get("gpu_name"):
        spec_table.add_row("GPU Hardware", meta_base.get("gpu_name", "None detected"), meta_target.get("gpu_name", "None detected"))
    spec_table.add_row("Physical RAM", f"{meta_base.get('total_ram_gb', 0.0):.1f} GB", f"{meta_target.get('total_ram_gb', 0.0):.1f} GB")
    spec_table.add_row("Operating System", f"{meta_base.get('os_name', 'N/A')} ({meta_base.get('os_release', '')})", f"{meta_target.get('os_name', 'N/A')} ({meta_target.get('os_release', '')})")
    spec_table.add_row("Python Environment", meta_base.get("python_version", "N/A"), meta_target.get("python_version", "N/A"))
    console.print(spec_table)
    console.print()
    
    # 3. High-Level Performance & Scoring Summary Table
    score_table = Table(title="Score & Metric Breakdown Comparison", show_header=True, header_style="bold magenta", expand=False)
    score_table.add_column("Benchmark Metric", style="cyan", no_wrap=True)
    score_table.add_column("Base Value", style="white", justify="right")
    score_table.add_column("Target Value", style="white", justify="right")
    score_table.add_column("Relative Delta / Speedup", style="bold", justify="center")
    
    # Composite Score
    score_base = summary["score_base"]
    score_target = summary["score_target"]
    s_delta = summary["score_delta_pct"]
    s_delta_str = f"+{s_delta:.1f}%" if s_delta >= 0 else f"{s_delta:.1f}%"
    s_badge = f"[green]{s_delta_str} ({summary['score_ratio']:.2f}x)[/green]" if s_delta >= 0 else f"[red]{s_delta_str} ({summary['score_ratio']:.2f}x)[/red]"
    score_table.add_row("Composite Heuristic Score", f"{score_base:,.1f}", f"{score_target:,.1f}", s_badge)
    
    # Performance Tier
    score_table.add_row("Performance Category", summary["category_base"], summary["category_target"], "-")
    
    # Max Qubits
    q_base = summary["max_qubits_base"]
    q_target = summary["max_qubits_target"]
    q_gap = summary["qubit_gap"]
    q_gap_str = f"+{q_gap} Qubits ({2**q_gap:.0f}x space)" if q_gap > 0 else (f"{q_gap} Qubits" if q_gap < 0 else "Par (Equal)")
    q_badge = f"[green]{q_gap_str}[/green]" if q_gap > 0 else (f"[red]{q_gap_str}[/red]" if q_gap < 0 else "[yellow]Par[/yellow]")
    score_table.add_row("Max Qubits Simulated", f"{q_base} Qubits", f"{q_target} Qubits", q_badge)
    
    # Capacity Metric
    c_base = summary["capacity_metric_base"]
    c_target = summary["capacity_metric_target"]
    c_ratio = summary["capacity_ratio"]
    c_badge = f"[green]{c_ratio:.2f}x Capacity[/green]" if c_ratio >= 1.0 else f"[red]{c_ratio:.2f}x Capacity[/red]"
    score_table.add_row("Capacity Metric (C = 2^n)", f"{c_base:,.0f}", f"{c_target:,.0f}", c_badge)
    
    # Throughput Metric
    t_base = summary["throughput_metric_base"]
    t_target = summary["throughput_metric_target"]
    t_speedup = summary["throughput_speedup"]
    t_delta = summary["throughput_delta_pct"]
    t_delta_str = f"+{t_delta:.1f}%" if t_delta >= 0 else f"{t_delta:.1f}%"
    t_badge = f"[green]{t_speedup:.2f}x Faster ({t_delta_str})[/green]" if t_speedup >= 1.0 else f"[red]{1.0/t_speedup:.2f}x Slower ({t_delta_str})[/red]"
    score_table.add_row("Throughput Metric (T = G/t)", f"{t_base:,.2f} g/s", f"{t_target:,.2f} g/s", t_badge)
    
    console.print(score_table)
    console.print()
    
    # 4. Per-Qubit Latency Comparison Table
    if matched:
        lat_table = Table(title="Per-Qubit Latency & Execution Breakdown", show_header=True, header_style="bold magenta", expand=False)
        lat_table.add_column("Qubits", style="cyan", justify="center")
        lat_table.add_column("Workload / Method", style="white", justify="center")
        lat_table.add_column(f"Base Latency ({lbl_base[:12]})", style="yellow", justify="right")
        lat_table.add_column(f"Target Latency ({lbl_target[:12]})", style="yellow", justify="right")
        lat_table.add_column("Latency Delta (%)", justify="center")
        lat_table.add_column("Speedup Factor", style="bold", justify="center")
        lat_table.add_column("CPU Usage (Base vs Target)", style="white", justify="center")
        
        for m in matched:
            q = m["qubits"]
            workload = f"{m['workload_label']} ({m['method']})"
            lat_b = f"{m['latency_base']:.3f}s"
            lat_t = f"{m['latency_target']:.3f}s"
            
            delta_pct = m["latency_delta_pct"]
            speedup = m["speedup_factor"]
            
            if delta_pct < 0:
                # Target is faster
                delta_str = f"[green]{delta_pct:.1f}%[/green]"
                speedup_str = f"[green]{speedup:.2f}x Faster[/green]"
            elif delta_pct > 0:
                delta_str = f"[red]+{delta_pct:.1f}%[/red]"
                speedup_str = f"[red]{1.0/speedup:.2f}x Slower[/red]" if speedup > 0 else "[red]N/A[/red]"
            else:
                delta_str = "[yellow]Par (0.0%)[/yellow]"
                speedup_str = "[yellow]1.00x Par[/yellow]"
                
            cpu_comp = f"{m['cpu_base']:.1f}% vs {m['cpu_target']:.1f}%"
            lat_table.add_row(str(q), workload, lat_b, lat_t, delta_str, speedup_str, cpu_comp)
            
        console.print(lat_table)
        console.print()
        
    # 5. Academic Verdict Panel
    console.print(Panel(
        f"[bold white]{diff_data['verdict']}[/bold white]",
        title="[bold green]Comparative Benchmark Verdict[/bold green]",
        border_style="green",
        expand=False
    ))
    console.print()

def export_comparison_to_json(diff_data: Dict[str, Any], output_dir: str = "results") -> str:
    """
    Export relative comparison results to a structured JSON file.
    
    Args:
        diff_data: Comparison result dictionary.
        output_dir: Directory where comparison.json will be saved.
        
    Returns:
        Absolute path to the created JSON file.
    """
    if not isinstance(diff_data, dict):
        raise TypeError("diff_data must be a dictionary")
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    out_payload = {
        "timestamp": datetime.now().isoformat(),
        "report_type": "QuaComp Relative Benchmark Comparison",
        "base_label": diff_data["base_label"],
        "target_label": diff_data["target_label"],
        "base_metadata": diff_data.get("base_metadata", {}),
        "target_metadata": diff_data.get("target_metadata", {}),
        "score_summary": diff_data["score_summary"],
        "matched_qubits": diff_data.get("matched_qubits", []),
        "verdict": diff_data["verdict"]
    }
    
    file_path = os.path.abspath(os.path.join(output_dir, "comparison.json"))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
        
    return file_path

def export_comparison_to_markdown(
    diff_data: Dict[str, Any], 
    output_dir: str = "results", 
    generated_charts: Optional[List[str]] = None
) -> str:
    """
    Export relative comparison results to a GitHub-flavored Markdown report.
    
    Args:
        diff_data: Comparison result dictionary.
        output_dir: Directory where comparison_report.md will be saved.
        generated_charts: List of chart image file paths to embed.
        
    Returns:
        Absolute path to the created Markdown file.
    """
    if not isinstance(diff_data, dict):
        raise TypeError("diff_data must be a dictionary")
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    lbl_base = diff_data["base_label"]
    lbl_target = diff_data["target_label"]
    meta_base = diff_data.get("base_metadata", {})
    meta_target = diff_data.get("target_metadata", {})
    summary = diff_data["score_summary"]
    matched = diff_data.get("matched_qubits", [])
    
    score_base = summary["score_base"]
    score_target = summary["score_target"]
    s_delta = summary["score_delta_pct"]
    s_delta_str = f"+{s_delta:.1f}%" if s_delta >= 0 else f"{s_delta:.1f}%"
    
    t_base = summary["throughput_metric_base"]
    t_target = summary["throughput_metric_target"]
    t_speedup = summary["throughput_speedup"]
    t_delta = summary["throughput_delta_pct"]
    t_delta_str = f"+{t_delta:.1f}%" if t_delta >= 0 else f"{t_delta:.1f}%"
    
    q_gap = summary["qubit_gap"]
    
    md_lines = [
        "# QuaComp Relative Benchmark Comparison Report",
        "",
        f"> **Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> **Base Configuration**: `{lbl_base}`  ",
        f"> **Target Configuration**: `{lbl_target}`",
        "",
        "## 1. Executive Summary & Verdict",
        "",
        f"**Verdict**: {diff_data['verdict']}",
        "",
        "| Metric | Base System (`" + lbl_base[:18] + "`) | Target System (`" + lbl_target[:18] + "`) | Relative Comparison / Speedup |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Composite Score** | {score_base:,.1f} | {score_target:,.1f} | **{s_delta_str}** ({summary['score_ratio']:.2f}x) |",
        f"| **Performance Tier** | {summary['category_base']} | {summary['category_target']} | — |",
        f"| **Max Qubit Limit** | {summary['max_qubits_base']} Qubits | {summary['max_qubits_target']} Qubits | **{'+' if q_gap > 0 else ''}{q_gap} Qubits** ({2**q_gap:.0f}x space) |",
        f"| **Capacity Metric ($C=2^n$)** | {summary['capacity_metric_base']:,.0f} | {summary['capacity_metric_target']:,.0f} | **{summary['capacity_ratio']:.2f}x Capacity** |",
        f"| **Throughput ($T=G/t$)** | {t_base:,.2f} g/s | {t_target:,.2f} g/s | **{t_speedup:.2f}x Speedup** ({t_delta_str}) |",
        "",
        "---",
        "",
        "## 2. Hardware & Environment Specifications",
        "",
        "| Specification | Base System | Target System |",
        "| :--- | :--- | :--- |",
        f"| **CPU / Processor** | {meta_base.get('cpu_name', 'N/A')} | {meta_target.get('cpu_name', 'N/A')} |",
        f"| **GPU Hardware** | {meta_base.get('gpu_name', 'None detected')} | {meta_target.get('gpu_name', 'None detected')} |",
        f"| **Physical RAM** | {meta_base.get('total_ram_gb', 0.0):.1f} GB | {meta_target.get('total_ram_gb', 0.0):.1f} GB |",
        f"| **Operating System** | {meta_base.get('os_name', 'N/A')} ({meta_base.get('os_release', '')}) | {meta_target.get('os_name', 'N/A')} ({meta_target.get('os_release', '')}) |",
        f"| **Python Version** | {meta_base.get('python_version', 'N/A')} | {meta_target.get('python_version', 'N/A')} |",
        "",
        "---",
        "",
        "## 3. Per-Qubit Latency & Execution Breakdown",
        ""
    ]
    
    if matched:
        md_lines.extend([
            "| Qubits | Workload / Method | Base Latency | Target Latency | Latency Delta (%) | Speedup Factor | CPU Usage (Base vs Target) |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ])
        for m in matched:
            q = m["qubits"]
            workload = f"{m['workload_label']} ({m['method']})"
            lat_b = f"{m['latency_base']:.4f}s"
            lat_t = f"{m['latency_target']:.4f}s"
            d_pct = m["latency_delta_pct"]
            d_str = f"{d_pct:+.1f}%"
            sp_str = f"{m['speedup_factor']:.2f}x {'Faster' if m['is_faster'] else 'Slower'}"
            cpu_comp = f"{m['cpu_base']:.1f}% vs {m['cpu_target']:.1f}%"
            md_lines.append(f"| {q} | {workload} | {lat_b} | {lat_t} | **{d_str}** | **{sp_str}** | {cpu_comp} |")
        md_lines.append("")
        
    if generated_charts:
        md_lines.extend([
            "---",
            "",
            "## 4. Visual Comparison Plots",
            ""
        ])
        for chart_path in generated_charts:
            chart_filename = os.path.basename(chart_path)
            md_lines.append(f"![{chart_filename}]({chart_filename})\n")
            
    md_lines.append("*(Report automatically generated by QuaComp Relative Benchmark Comparator v1.6.0)*\n")
    
    file_path = os.path.abspath(os.path.join(output_dir, "comparison_report.md"))
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    return file_path
