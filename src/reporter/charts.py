import os
from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.profiler.memory import estimate_qubit_ram

def generate_benchmark_charts(
    results: List[Dict[str, Any]], 
    system_metadata: Dict[str, Any], 
    output_dir: str = 'results'
) -> List[str]:
    """
    Generate clean, modern benchmark visualization charts from simulation telemetry.
    
    Charts generated:
        1. qubit_vs_latency.png: Qubit Count vs Mean Execution Latency (seconds).
        2. qubit_vs_ram.png: Qubit Count vs Memory Allocation (GB) with RAM safety threshold line.
        3. method_comparison.png (if multi-method data present): Latency comparison between Statevector & MPS.
        4. noise_fidelity_impact.png (if noise profile data present): Bar chart of Noise Level vs Fidelity (%) & CPU Overhead (%).
        
    Args:
        results (list): Benchmark simulation run result dictionaries.
        system_metadata (dict): System metadata details.
        output_dir (str): Output directory where chart images will be saved.
        
    Returns:
        list: List of absolute file paths to generated PNG chart images.
        
    Raises:
        TypeError: If results or system_metadata types are invalid.
    """
    if not isinstance(results, list):
        raise TypeError("results must be a list of dictionaries.")
    if not isinstance(system_metadata, dict):
        raise TypeError("system_metadata must be a dictionary.")
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Configure global seaborn plot style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        "font.family": "sans-serif",
        "figure.dpi": 300,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })
    
    generated_chart_paths: List[str] = []
    successful_runs = [r for r in results if r.get("success", False)]
    
    if not successful_runs:
        return generated_chart_paths
        
    # Sort runs by qubits
    sorted_runs = sorted(successful_runs, key=lambda x: x["qubits"])
    qubits = [r["qubits"] for r in sorted_runs]
    mean_latencies = [r.get("mean_latency", r.get("latency", 0.0)) for r in sorted_runs]
    std_latencies = [r.get("std_latency", 0.0) for r in sorted_runs]
    
    # ---------------------------------------------------------
    # Chart 1: Qubit vs Mean Latency (Line Plot)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(qubits, mean_latencies, marker='o', color='#1f77b4', linewidth=2.5, markersize=7, label='Mean Latency (s)')
    
    # Add error bounds if std_latency > 0
    if any(s > 0 for s in std_latencies):
        lower_bound = [max(0.0, m - s) for m, s in zip(mean_latencies, std_latencies)]
        upper_bound = [m + s for m, s in zip(mean_latencies, std_latencies)]
        ax.fill_between(qubits, lower_bound, upper_bound, color='#1f77b4', alpha=0.2, label='Std Dev (Jitter)')
        
    ax.set_title("QuaComp Benchmark: Qubit Count vs Mean Execution Latency")
    ax.set_xlabel("Qubit Count (n)")
    ax.set_ylabel("Execution Latency (seconds)")
    ax.set_xticks(qubits)
    ax.legend(loc="upper left")
    
    # Annotate points
    for q, m in zip(qubits, mean_latencies):
        ax.annotate(f"{m:.3f}s", (q, m), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    path_lat = os.path.abspath(os.path.join(output_dir, "qubit_vs_latency.png"))
    plt.savefig(path_lat)
    plt.close(fig)
    generated_chart_paths.append(path_lat)
    
    # ---------------------------------------------------------
    # Chart 2: Qubit vs Memory Allocation (GB)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Calculate estimated statevector RAM in GB for each qubit size
    ram_gb_estimated = [estimate_qubit_ram(q) / (1024 ** 3) for q in qubits]
    total_ram_gb = system_metadata.get("total_ram_gb", 16.0)
    safety_limit_gb = total_ram_gb * 0.85
    
    ax.plot(qubits, ram_gb_estimated, marker='s', color='#d62728', linewidth=2.5, markersize=7, label='Theoretical Statevector RAM (GB)')
    ax.axhline(y=safety_limit_gb, color='#ff7f0e', linestyle='--', linewidth=2, label=f'85% RAM Safety Limit ({safety_limit_gb:.1f} GB)')
    ax.axhline(y=total_ram_gb, color='#7f7f7f', linestyle=':', linewidth=1.5, label=f'Total System RAM ({total_ram_gb:.1f} GB)')
    
    ax.set_title("QuaComp Benchmark: Memory Footprint vs Physical Threshold")
    ax.set_xlabel("Qubit Count (n)")
    ax.set_ylabel("Memory Footprint (GB)")
    ax.set_xticks(qubits)
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    path_ram = os.path.abspath(os.path.join(output_dir, "qubit_vs_ram.png"))
    plt.savefig(path_ram)
    plt.close(fig)
    generated_chart_paths.append(path_ram)
    
    # ---------------------------------------------------------
    # Chart 3: Method Comparison (Statevector vs MPS) - Optional
    # ---------------------------------------------------------
    methods_present = set(r.get("method", "statevector") for r in sorted_runs)
    if len(methods_present) > 1 or any(r.get("method") in ('mps', 'matrix_product_state') for r in sorted_runs):
        fig, ax = plt.subplots(figsize=(8, 5))
        
        methods = [r.get("method", "statevector") for r in sorted_runs]
        latencies = [r.get("mean_latency", r.get("latency", 0.0)) for r in sorted_runs]
        labels = [f"q={r['qubits']}\n({r.get('method', 'sv')})" for r in sorted_runs]
        
        colors = ['#2ca02c' if m in ('mps', 'matrix_product_state') else '#1f77b4' for m in methods]
        bars = ax.bar(labels, latencies, color=colors, width=0.5)
        
        ax.set_title("Simulation Engine Method Performance Comparison")
        ax.set_xlabel("Circuit Setup")
        ax.set_ylabel("Execution Latency (seconds)")
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}s", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
            
        plt.tight_layout()
        path_method = os.path.abspath(os.path.join(output_dir, "method_comparison.png"))
        plt.savefig(path_method)
        plt.close(fig)
        generated_chart_paths.append(path_method)
        
    # ---------------------------------------------------------
    # Chart 4: NISQ Noise & State Fidelity Impact - Optional
    # ---------------------------------------------------------
    noise_runs = [r for r in sorted_runs if r.get("noise_level", "none") != "none"]
    if noise_runs:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        noise_labels = [f"q={r['qubits']} ({r['noise_level']})" for r in noise_runs]
        fidelities = [r.get("fidelity", 100.0) for r in noise_runs]
        overheads = [r.get("overhead_ratio", 0.0) for r in noise_runs]
        
        x = np.arange(len(noise_labels))
        width = 0.35
        
        rects1 = ax1.bar(x - width/2, fidelities, width, label='Quantum State Fidelity (%)', color='#2ca02c')
        ax1.set_ylabel('Fidelity (%)', color='#2ca02c')
        ax1.tick_params(axis='y', labelcolor='#2ca02c')
        ax1.set_ylim(0, 110)
        
        ax2 = ax1.twinx()
        rects2 = ax2.bar(x + width/2, overheads, width, label='CPU Overhead (%)', color='#ff7f0e')
        ax2.set_ylabel('CPU Latency Overhead (%)', color='#ff7f0e')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(noise_labels)
        ax1.set_title("NISQ Synthetic Noise Profile Impact on Fidelity & Overhead")
        
        plt.tight_layout()
        path_noise = os.path.abspath(os.path.join(output_dir, "noise_fidelity_impact.png"))
        plt.savefig(path_noise)
        plt.close(fig)
        generated_chart_paths.append(path_noise)
        
    return generated_chart_paths

def generate_comparison_charts(
    diff_data: Dict[str, Any], 
    output_dir: str = 'results'
) -> List[str]:
    """
    Generate side-by-side relative comparison charts for two benchmark runs.
    
    Charts generated:
        1. qubit_latency_comparison.png: Grouped bar chart of Mean Latency for Base vs Target.
        2. throughput_comparison.png: Bar chart of Throughput Metric (gates/sec) for Base vs Target.
        
    Args:
        diff_data: Benchmark comparison dictionary produced by compare_benchmarks().
        output_dir: Output directory where chart images will be saved.
        
    Returns:
        List of absolute file paths to generated PNG chart images.
    """
    if not isinstance(diff_data, dict):
        raise TypeError("diff_data must be a dictionary.")
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        "font.family": "sans-serif",
        "figure.dpi": 300,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })
    
    chart_paths: List[str] = []
    matched = diff_data.get("matched_qubits", [])
    lbl_base = diff_data.get("base_label", "Base")[:16]
    lbl_target = diff_data.get("target_label", "Target")[:16]
    summary = diff_data.get("score_summary", {})
    
    # Chart 1: Grouped Latency Comparison per Qubit
    if matched:
        fig, ax = plt.subplots(figsize=(8, 5))
        qubit_labels = [f"{m['qubits']} Qubits\n({m['workload_label']})" for m in matched]
        lat_base = [m["latency_base"] for m in matched]
        lat_target = [m["latency_target"] for m in matched]
        
        x = np.arange(len(qubit_labels))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, lat_base, width, label=f"Base: {lbl_base}", color='#1f77b4')
        rects2 = ax.bar(x + width/2, lat_target, width, label=f"Target: {lbl_target}", color='#2ca02c')
        
        ax.set_ylabel("Execution Latency (seconds)")
        ax.set_title("QuaComp Comparison: Qubit Simulation Latency (Lower is Better)")
        ax.set_xticks(x)
        ax.set_xticklabels(qubit_labels)
        ax.legend(loc="upper left")
        
        # Annotate bars
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}s", xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                        
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}s", xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
                        
        plt.tight_layout()
        path_lat_comp = os.path.abspath(os.path.join(output_dir, "qubit_latency_comparison.png"))
        plt.savefig(path_lat_comp)
        plt.close(fig)
        chart_paths.append(path_lat_comp)
        
    # Chart 2: Throughput Metric Comparison
    t_base = summary.get("throughput_metric_base", 0.0)
    t_target = summary.get("throughput_metric_target", 0.0)
    if t_base > 0 or t_target > 0:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar([f"Base:\n{lbl_base}", f"Target:\n{lbl_target}"], [t_base, t_target], color=['#1f77b4', '#2ca02c'], width=0.45)
        
        ax.set_ylabel("Simulation Throughput (gates / second)")
        ax.set_title("Quantum Gate Simulation Throughput Comparison (Higher is Better)")
        
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:,.1f} g/s", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
                        
        plt.tight_layout()
        path_tput = os.path.abspath(os.path.join(output_dir, "throughput_comparison.png"))
        plt.savefig(path_tput)
        plt.close(fig)
        chart_paths.append(path_tput)
        
    return chart_paths

