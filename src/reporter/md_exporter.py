import os
import time
from typing import List, Dict, Any

def export_to_markdown(results: List[Dict[str, Any]], system_metadata: Dict[str, Any], output_path: str = "results/report.md") -> str:
    """
    Export benchmark results, scoring breakdown, statistical metrics, and system telemetry to a Markdown file.
    
    Args:
        results (list): List of simulation run dictionaries.
        system_metadata (dict): Collected system metadata.
        output_path (str): File path where the Markdown report should be saved.
        
    Returns:
        str: Absolute path to the saved Markdown file.
        
    Raises:
        TypeError: If results or system_metadata types are incorrect.
    """
    if not isinstance(results, list):
        raise TypeError("results must be a list of dictionaries.")
    if not isinstance(system_metadata, dict):
        raise TypeError("system_metadata must be a dictionary.")
        
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    abs_path = os.path.abspath(output_path)
    
    # Calculate overall best score from successful runs
    successful_runs = [r for r in results if r.get("success", False)]
    best_qubits = 0
    gates = 0
    score = 0.0
    category = "N/A"
    
    best_method = "statevector"
    best_bond_dim = None
    ram_savings = {}
    best_noise_level = "none"
    best_fidelity = 100.0
    best_overhead_ratio = 0.0
    
    best_mean_latency = 0.0
    best_std_latency = 0.0
    best_runs_count = 0
    capacity_metric = 0.0
    throughput_metric = 0.0
    
    if successful_runs:
        from src.scorer.calculator import calculate_scoring_breakdown, categorize_score
        best_run = max(successful_runs, key=lambda x: x["qubits"])
        best_qubits = best_run["qubits"]
        gates = best_run["gates"]
        best_mean_latency = best_run.get("mean_latency", best_run.get("latency", 0.0))
        best_std_latency = best_run.get("std_latency", 0.0)
        best_runs_count = best_run.get("runs_count", 1)
        
        breakdown = calculate_scoring_breakdown(best_qubits, gates, best_mean_latency)
        score = breakdown["composite_score"]
        capacity_metric = breakdown["capacity_metric"]
        throughput_metric = breakdown["throughput_metric"]
        category = categorize_score(score)
        
        best_method = best_run.get("method", "statevector")
        best_bond_dim = best_run.get("bond_dimension")
        ram_savings = best_run.get("ram_savings", {})
        best_noise_level = best_run.get("noise_level", "none")
        best_fidelity = best_run.get("fidelity", 100.0)
        best_overhead_ratio = best_run.get("overhead_ratio", 0.0)
        
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    md_content = []
    md_content.append("# QuaComp Benchmark Report")
    md_content.append(f"Generated on: `{current_time}`")
    md_content.append("\n---\n")
    
    # Final score summary
    md_content.append("## Benchmark Summary")
    md_content.append(f"> **QuaComp Composite Score:** `{score:,.2f}` *(Project-Specific Heuristic Score)*")
    md_content.append(f"> - **Capacity Metric (2^n):** `{capacity_metric:,.0f}`")
    md_content.append(f"> - **Throughput Metric:** `{throughput_metric:,.2f} gates/sec`")
    md_content.append(f"> **Performance Category:** `{category}`")
    
    method_name = "Statevector" if best_method == "statevector" else f"MPS (max_bond_dimension={best_bond_dim})"
    md_content.append(f"> **Simulation Method:** `{method_name}`")
    
    if best_noise_level != "none":
        md_content.append(f"> **NISQ Noise Profile:** `{best_noise_level}` *(synthetic representative)*")
        md_content.append(f"> **Quantum State Fidelity:** `{best_fidelity:.2f}%`")
        md_content.append(f"> **CPU Computation Overhead:** `+{best_overhead_ratio:.2f}%`")
    
    if ram_savings:
        savings_gb = ram_savings["savings_bytes"] / (1024 ** 3)
        md_content.append(f"> **MPS RAM Efficiency:** `{ram_savings['savings_percent']:.2f}%` savings (Saved ~`{savings_gb:.4f} GB` vs Statevector)")
        
    md_content.append(f"> **Statistical Repeatability:** `{best_runs_count} runs` (Mean Latency: `{best_mean_latency:.4f}s`, Std Dev: `{best_std_latency:.4f}s`)")
    md_content.append(f"> **Max Qubits Simulated:** `{best_qubits} qubits` (using `{gates:,}` gates)")
    md_content.append("\n---\n")
    
    # System metadata
    md_content.append("## System Metadata & Telemetry")
    md_content.append("| Parameter | System Value |")
    md_content.append("| :--- | :--- |")
    md_content.append(f"| **CPU Name** | {system_metadata.get('cpu_name', 'Unknown')} |")
    md_content.append(f"| **Total Physical RAM** | {system_metadata.get('total_ram_gb', 0.0):.2f} GB |")
    md_content.append(f"| **Operating System** | {system_metadata.get('os_name', 'Unknown')} ({system_metadata.get('os_release', '')}) |")
    md_content.append(f"| **Python Version** | {system_metadata.get('python_version', 'Unknown')} |")
    md_content.append("\n---\n")
    
    # Runs summary
    md_content.append("## Detailed Simulation Runs")
    md_content.append("| Qubits | Method | Noise Profile | Workload | Total Gates | Latency (Mean ± Std Dev) | Fidelity % | Avg CPU % | RAM Status | Success |")
    md_content.append("| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for r in results:
        success_str = "SUCCESS" if r["success"] else "FAILED"
        mean_lat = r.get("mean_latency", r.get("latency", 0.0))
        std_lat = r.get("std_latency", 0.0)
        latency_str = f"{mean_lat:.4f} ± {std_lat:.4f}s" if r["success"] else "-"
        cpu_str = f"{r['cpu_usage']:.1f}%" if r["success"] else "-"
        workload = r.get("workload_label", "QFT")
        
        method_str = r.get("method", "statevector")
        if method_str in ('mps', 'matrix_product_state') and r.get("bond_dimension"):
            method_str = f"mps (chi={r['bond_dimension']})"
            
        noise_str = r.get("noise_level", "none")
        fidelity_val = r.get("fidelity", 100.0)
        fidelity_str = f"{fidelity_val:.2f}%" if r["success"] else "-"
            
        md_content.append(
            f"| {r['qubits']} | {method_str} | {noise_str} | {workload} | {r['gates']} | {latency_str} | {fidelity_str} | {cpu_str} | {r['ram_status']} | {success_str} |"
        )
        
    md_content.append("\n---\n")
    md_content.append("## GitHub Ready")
    md_content.append("This report is formatted and ready to be posted directly into GitHub Issues, pull request reviews, or Discussions.")
    
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
        
    return abs_path
