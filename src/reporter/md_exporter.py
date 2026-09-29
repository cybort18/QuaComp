import os
import time
from typing import List, Dict, Any

def export_to_markdown(
    results: List[Dict[str, Any]], 
    system_metadata: Dict[str, Any], 
    output_path: str = "results/report.md",
    generated_charts: List[str] = None
) -> str:
    """
    Export benchmark results, scoring breakdown, statistical metrics, academic benchmarks,
    system telemetry, and chart images to a Markdown file.
    
    Args:
        results (list): List of simulation run dictionaries.
        system_metadata (dict): Collected system metadata.
        output_path (str): File path where the Markdown report should be saved.
        generated_charts (list): Optional list of absolute chart file paths.
        
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
    throughput_kgps = 0.0
    best_entanglement = {}
    best_energy = {}
    best_native_kernel = False
    best_accelerator_badge = ""
    best_accelerator_backend = ""
    best_physical_noise = None
    edp_val = None
    
    qv_hprob_str = "N/A"
    qv_cert_str = "Not Evaluated"
    
    if successful_runs:
        from src.scorer.calculator import (
            calculate_scoring_breakdown, 
            categorize_score
        )
        best_run = max(successful_runs, key=lambda x: x["qubits"])
        best_qubits = best_run["qubits"]
        gates = best_run["gates"]
        best_mean_latency = best_run.get("mean_latency", best_run.get("latency", 0.0))
        best_std_latency = best_run.get("std_latency", 0.0)
        best_runs_count = best_run.get("runs_count", 1)
        best_fidelity = best_run.get("fidelity", 100.0)
        best_overhead_ratio = best_run.get("overhead_ratio", 0.0)
        
        best_energy = best_run.get("energy_metrics", {})
        total_energy_j = best_energy.get("total_energy_joules") if best_energy else None
        
        breakdown = calculate_scoring_breakdown(
            best_qubits, gates, best_mean_latency, fidelity=best_fidelity, energy_joules=total_energy_j
        )
        score = breakdown["composite_score"]
        capacity_metric = breakdown["capacity_metric"]
        throughput_metric = breakdown["throughput_metric"]
        throughput_kgps = breakdown["throughput_kgps"]
        edp_val = breakdown["energy_delay_product"]
        category = categorize_score(score)
        
        best_method = best_run.get("method", "statevector")
        best_bond_dim = best_run.get("bond_dimension")
        ram_savings = best_run.get("ram_savings", {})
        best_noise_level = best_run.get("noise_level", "none")
        best_entanglement = best_run.get("entanglement_metrics", {})
        best_native_kernel = best_run.get("native_kernel_used", False)
        best_accelerator_badge = best_run.get("accelerator_badge", "")
        best_accelerator_backend = best_run.get("accelerator_backend", "")
        best_physical_noise = best_run.get("physical_noise_profile", None)
        
        # Check for Quantum Volume evaluations across runs
        qv_runs = [r for r in successful_runs if r.get("qv_metrics")]
        if qv_runs:
            best_qv_run = max(qv_runs, key=lambda x: x["qubits"])
            qvm = best_qv_run["qv_metrics"]
            hprob = qvm.get("heavy_output_probability", 0.0)
            qv_hprob_str = f"{hprob:.4f}"
            if qvm.get("qv_certified"):
                qv_cert_str = "CERTIFIED (h_prob > 2/3)"
            else:
                qv_cert_str = "FAILED (h_prob <= 2/3)"
                
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    md_content = []
    md_content.append("# QuaComp Benchmark Report")
    md_content.append(f"Generated on: `{current_time}`")
    md_content.append("\n---\n")
    
    # Dual-Mode Benchmark Summary
    md_content.append("## Benchmark Summary")
    
    # 1. Block 1: Hardware Benchmark Overview
    method_name = "Statevector" if best_method == "statevector" else f"MPS (chi={best_bond_dim})"
    accel_str = f"`{best_accelerator_badge}` ({best_accelerator_backend})" if best_accelerator_badge else "Standard CPU"
    ram_status_str = "SAFE" if successful_runs else "UNSAFE"
    memory_str = f"{system_metadata.get('total_ram_gb', 0.0):.1f} GB RAM"
    
    md_content.append("### Hardware Benchmark Overview (QuaComp Synthetic Index)")
    md_content.append("| Benchmark Metric | Measured Value | Performance Tier / Status |")
    md_content.append("| :--- | :---: | :---: |")
    md_content.append(f"| **QuaComp Synthetic Index (QSI)** | `{score:,.2f}` pts | **{category}** |")
    md_content.append(f"| **Capacity Metric (2^n)** | `{capacity_metric:,.0f}` | Max `{best_qubits}` qubits simulated |")
    md_content.append(f"| **Throughput Metric** | `{throughput_metric:,.2f} gates/s` | Dynamic sensitivity scale |")
    md_content.append(f"| **Execution Latency (Mean)** | `{best_mean_latency:.4f}s` | ± `{best_std_latency:.4f}s` ({best_runs_count} runs) |")
    md_content.append(f"| **Simulation Method** | `{method_name}` | Acceleration: {accel_str} |")
    md_content.append("")
    
    # 2. Block 2: Formal Academic Metrics
    edp_str = f"{edp_val:.6f} J · s" if edp_val is not None else "N/A (Energy profiling disabled)"
    fidelity_str = f"{best_fidelity:.2f}%" if successful_runs else "N/A"
    
    md_content.append("### Formal Academic Metrics (Quantum HPC Standards)")
    md_content.append("| Academic Metric | Measured Value | Standard / Significance |")
    md_content.append("| :--- | :---: | :--- |")
    md_content.append(f"| **Normalized Gate Throughput** | `{throughput_kgps:.4f} kGates/s` | `{gates:,}` total gates processed |")
    md_content.append(f"| **Energy-Delay Product (EDP)** | `{edp_str}` | Energy-delay efficiency (Lower indicates superior efficiency) |")
    md_content.append(f"| **Quantum Volume Certification** | `{qv_cert_str}` | Heavy Output Probability: `{qv_hprob_str}` (Threshold: > 0.6667) |")
    md_content.append(f"| **Quantum State Fidelity** | `{fidelity_str}` | Statevector fidelity against ideal analytical target |")
    md_content.append("\n---\n")
    
    # Hardware Telemetry Details (Annotations)
    if successful_runs and best_native_kernel and best_accelerator_badge:
        md_content.append(f"> **Native Acceleration Engine:** `{best_accelerator_badge}` *({best_accelerator_backend})*")
        
    if successful_runs and best_physical_noise:
        md_content.append(f"> **Physical QPU Noise Calibration:** `{best_physical_noise}` *(Realistic Kraus Operator Decomposition)*")
        md_content.append(f"> **Quantum State Fidelity:** `{best_fidelity:.2f}%`")
    elif best_noise_level != "none":
        md_content.append(f"> **NISQ Noise Profile:** `{best_noise_level}` *(synthetic representative)*")
        md_content.append(f"> **Quantum State Fidelity:** `{best_fidelity:.2f}%`")
        md_content.append(f"> **CPU Computation Overhead:** `+{best_overhead_ratio:.2f}%`")
    
    if ram_savings:
        savings_gb = ram_savings["savings_bytes"] / (1024 ** 3)
        md_content.append(f"> **MPS RAM Efficiency:** `{ram_savings['savings_percent']:.2f}%` savings (Saved ~`{savings_gb:.4f} GB` vs Statevector)")
        
    if best_entanglement and "von_neumann_entropy" in best_entanglement:
        md_content.append(f"> **Entanglement Entropy:** `S_vN = {best_entanglement['von_neumann_entropy']:.4f} bits` (Schmidt Rank: `{best_entanglement['schmidt_rank']}` | `{best_entanglement['entanglement_regime']}`)")
        md_content.append(f"> **MPS Simulation Complexity:** `{best_entanglement['mps_hardness']}`")
        
    if best_energy and "total_energy_joules" in best_energy:
        backend_raw = str(best_energy.get("energy_backend", "Generic Model"))
        sensor_tag = "[Sensor: RAPL]" if "rapl" in backend_raw.lower() else "[Sensor: TDP Estimate]"
        md_content.append(f"> **Hardware Energy Telemetry:** `{sensor_tag}` `{best_energy.get('total_energy_joules', 0.0):.4f} J` (Avg Power: `{best_energy.get('average_power_watts', 0.0):.2f} W`, EQO: `{best_energy.get('eqo_microjoules', 0.0):.2f} µJ/gate | `{backend_raw}`)")
        
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
    
    # Detailed simulation runs
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
        if r.get("accelerator_badge"):
            method_str = f"{method_str} `{r['accelerator_badge']}`"
            
        if r.get("physical_noise_profile"):
            noise_str = f"QPU: {r['physical_noise_profile']}"
        else:
            noise_str = r.get("noise_level", "none")
            
        fidelity_val = r.get("fidelity", 100.0)
        fidelity_str = f"{fidelity_val:.2f}%" if r["success"] else "-"
            
        md_content.append(
            f"| {r['qubits']} | {method_str} | {noise_str} | {workload} | {r['gates']} | {latency_str} | {fidelity_str} | {cpu_str} | {r['ram_status']} | {success_str} |"
        )
        
    md_content.append("\n---\n")
    
    # Entanglement Entropy Table (if available)
    entropy_runs = [r for r in results if r.get("success", False) and r.get("entanglement_metrics") and "von_neumann_entropy" in r["entanglement_metrics"]]
    if entropy_runs:
        md_content.append("## Entanglement Entropy & Simulation Hardness Analysis")
        md_content.append("| Qubits | Workload | Von Neumann Entropy (S_vN) | Max Bound | Schmidt Rank | Entanglement Regime | MPS Complexity Tier |")
        md_content.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for er in entropy_runs:
            em = er["entanglement_metrics"]
            md_content.append(
                f"| {er['qubits']} | {er.get('workload_label', 'Circuit')} | {em['von_neumann_entropy']:.4f} bits | {em['max_possible_entropy']:.1f} | {em['schmidt_rank']} | {em['entanglement_regime']} | {em['mps_hardness']} |"
            )
        md_content.append("\n---\n")
        
    # Hardware Power & Energy Telemetry Table (if available)
    energy_runs = [r for r in results if r.get("success", False) and r.get("energy_metrics") and "total_energy_joules" in r["energy_metrics"]]
    if energy_runs:
        md_content.append("## Hardware Power & Energy Telemetry (EQO)")
        md_content.append("| Qubits | Workload | Avg Power (W) | Total Energy (J) | EQO (µJ/Gate) | Telemetry Backend & Source |")
        md_content.append("| :---: | :---: | :---: | :---: | :---: | :--- |")
        for enr in energy_runs:
            enm = enr["energy_metrics"]
            backend_raw = str(enm.get("energy_backend", "Generic Model"))
            sensor_tag = "[Sensor: RAPL]" if "rapl" in backend_raw.lower() else "[Sensor: TDP Estimate]"
            md_content.append(
                f"| {enr['qubits']} | {enr.get('workload_label', 'Circuit')} | {enm.get('average_power_watts', 0.0):.2f} W | {enm.get('total_energy_joules', 0.0):.4f} J | {enm.get('eqo_microjoules', 0.0):.2f} µJ | `{sensor_tag}` {backend_raw} |"
            )
        md_content.append("\n---\n")
        
    # Telemetry Visualizations (if charts exist)
    chart_files = []
    if generated_charts:
        chart_files = generated_charts
    else:
        # Check default files in output_dir
        target_dir = output_dir if output_dir else "results"
        potential_charts = [
            "qubit_vs_latency.png",
            "qubit_vs_ram.png",
            "method_comparison.png",
            "noise_fidelity_impact.png",
            "entanglement_entropy.png"
        ]
        for cfile in potential_charts:
            cpath = os.path.join(target_dir, cfile)
            if os.path.exists(cpath):
                chart_files.append(cpath)
                
    if chart_files:
        md_content.append("## Telemetry Visualizations")
        for cpath in chart_files:
            fname = os.path.basename(cpath)
            title = fname.replace(".png", "").replace("_", " ").title()
            md_content.append(f"![{title}]({fname})\n")
        md_content.append("\n---\n")
        
    md_content.append("## GitHub Ready")
    md_content.append("This report is formatted and ready to be posted directly into GitHub Issues, pull request reviews, or Discussions.")
    
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
        
    return abs_path
