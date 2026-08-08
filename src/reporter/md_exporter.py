import os
import time
from typing import List, Dict, Any

def export_to_markdown(results: List[Dict[str, Any]], system_metadata: Dict[str, Any], output_path: str = "results/report.md") -> str:
    """
    Export benchmark results and system telemetry to a clean Markdown file.
    
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
        
    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    abs_path = os.path.abspath(output_path)
    
    # Calculate overall best score from successful runs
    successful_runs = [r for r in results if r.get("success", False)]
    best_qubits = 0
    gates = 0
    latency = 0.0
    score = 0.0
    category = "N/A"
    
    best_method = "statevector"
    best_bond_dim = None
    ram_savings = {}
    
    if successful_runs:
        from src.scorer.calculator import calculate_qsim_score, categorize_score
        best_run = max(successful_runs, key=lambda x: x["qubits"])
        best_qubits = best_run["qubits"]
        gates = best_run["gates"]
        latency = best_run["latency"]
        score = calculate_qsim_score(best_qubits, gates, latency)
        category = categorize_score(score)
        best_method = best_run.get("method", "statevector")
        best_bond_dim = best_run.get("bond_dimension")
        ram_savings = best_run.get("ram_savings", {})
        
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    md_content = []
    md_content.append("# QuaComp Benchmark Report")
    md_content.append(f"Generated on: `{current_time}`")
    md_content.append("\n---\n")
    
    # Final score summary
    md_content.append("## Benchmark Summary")
    md_content.append(f"> **QuaComp Score:** `{score:,.2f}`")
    md_content.append(f"> **Performance Category:** `{category}`")
    
    method_name = "Statevector" if best_method == "statevector" else f"MPS (max_bond_dimension={best_bond_dim})"
    md_content.append(f"> **Simulation Method:** `{method_name}`")
    
    if ram_savings:
        savings_gb = ram_savings["savings_bytes"] / (1024 ** 3)
        md_content.append(f"> **MPS RAM Efficiency:** `{ram_savings['savings_percent']:.2f}%` savings (Saved ~`{savings_gb:.4f} GB` vs Statevector)")
        
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
    md_content.append("| Qubits | Method | Workload | Total Gates | Latency (s) | Avg CPU % | RAM Status | Success |")
    md_content.append("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    
    for r in results:
        success_str = "SUCCESS" if r["success"] else "FAILED"
        latency_str = f"{r['latency']:.4f}" if r["success"] else "-"
        cpu_str = f"{r['cpu_usage']:.1f}%" if r["success"] else "-"
        workload = r.get("workload_label", "QFT")
        
        method_str = r.get("method", "statevector")
        if method_str in ('mps', 'matrix_product_state') and r.get("bond_dimension"):
            method_str = f"mps (chi={r['bond_dimension']})"
            
        md_content.append(
            f"| {r['qubits']} | {method_str} | {workload} | {r['gates']} | {latency_str} | {cpu_str} | {r['ram_status']} | {success_str} |"
        )
        
    md_content.append("\n---\n")
    md_content.append("## GitHub Ready")
    md_content.append("This report is formatted and ready to be posted directly into GitHub Issues, pull request reviews, or Discussions.")
    
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
        
    return abs_path
