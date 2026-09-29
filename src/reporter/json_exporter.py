import json
import os
import time
from typing import List, Dict, Any

def export_to_json(results: List[Dict[str, Any]], system_metadata: Dict[str, Any], output_dir: str = "results") -> str:
    """
    Export benchmark results, scoring breakdown, statistical metrics, academic benchmarks,
    and system telemetry to a JSON file.
    
    Args:
        results (list): List of simulation run dictionaries.
        system_metadata (dict): Collected system metadata.
        output_dir (str): Directory where JSON files should be saved.
        
    Returns:
        str: Absolute path to the created JSON file.
        
    Raises:
        TypeError: If results or system_metadata types are incorrect.
    """
    if not isinstance(results, list):
        raise TypeError("results must be a list of dictionaries.")
    if not isinstance(system_metadata, dict):
        raise TypeError("system_metadata must be a dictionary.")
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{timestamp}.json"
    file_path = os.path.abspath(os.path.join(output_dir, filename))
    
    # Calculate overall best score from successful runs
    successful_runs = [r for r in results if r.get("success", False)]
    best_qubits = 0
    gates = 0
    score = 0.0
    category = "N/A"
    
    best_method = "statevector"
    best_bond_dim = None
    best_ram_savings = {}
    best_noise_level = "none"
    best_fidelity = 100.0
    best_overhead_ratio = 0.0
    
    best_mean_latency = 0.0
    best_median_latency = 0.0
    best_std_latency = 0.0
    best_runs_count = 0
    capacity_metric = 0.0
    throughput_metric = 0.0
    throughput_kgps = 0.0
    fidelity_factor = 1.0
    edp_val = None
    best_entanglement_metrics = {}
    best_energy_metrics = {}
    
    qv_hprob = None
    qv_certified = None
    
    if successful_runs:
        from src.scorer.calculator import (
            calculate_scoring_breakdown, 
            categorize_score
        )
        best_run = max(successful_runs, key=lambda x: x["qubits"])
        best_qubits = best_run["qubits"]
        gates = best_run["gates"]
        best_mean_latency = best_run.get("mean_latency", best_run.get("latency", 0.0))
        best_median_latency = best_run.get("median_latency", best_mean_latency)
        best_std_latency = best_run.get("std_latency", 0.0)
        best_runs_count = best_run.get("runs_count", 1)
        best_fidelity = best_run.get("fidelity", 100.0)
        best_overhead_ratio = best_run.get("overhead_ratio", 0.0)
        
        best_energy_metrics = best_run.get("energy_metrics", {})
        total_energy_j = best_energy_metrics.get("total_energy_joules") if best_energy_metrics else None
        
        breakdown = calculate_scoring_breakdown(
            best_qubits, gates, best_mean_latency, fidelity=best_fidelity, energy_joules=total_energy_j
        )
        score = breakdown["composite_score"]
        capacity_metric = breakdown["capacity_metric"]
        throughput_metric = breakdown["throughput_metric"]
        throughput_kgps = breakdown["throughput_kgps"]
        fidelity_factor = breakdown["fidelity_factor"]
        edp_val = breakdown["energy_delay_product"]
        category = categorize_score(score)
        
        best_method = best_run.get("method", "statevector")
        best_bond_dim = best_run.get("bond_dimension")
        best_ram_savings = best_run.get("ram_savings", {})
        best_noise_level = best_run.get("noise_level", "none")
        best_native_kernel = best_run.get("native_kernel_used", False)
        best_accelerator_backend = best_run.get("accelerator_backend", "none")
        best_accelerator_badge = best_run.get("accelerator_badge", "")
        best_physical_noise = best_run.get("physical_noise_profile", None)
        best_entanglement_metrics = best_run.get("entanglement_metrics", {})
        
        # Check if Quantum Volume metrics exist across successful runs
        qv_runs = [r for r in successful_runs if r.get("qv_metrics")]
        if qv_runs:
            best_qv_run = max(qv_runs, key=lambda x: x["qubits"])
            qvm = best_qv_run["qv_metrics"]
            qv_hprob = qvm.get("heavy_output_probability")
            qv_certified = qvm.get("qv_certified", False)
            
    academic_benchmarks = {
        "gate_throughput_kgps": throughput_kgps,
        "energy_delay_product_js": edp_val,
        "qv_heavy_output_probability": qv_hprob,
        "qv_certified": qv_certified
    }
        
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "final_score": score,
        "final_composite_score": score,
        "score_type": "QuaComp Synthetic Index (QSI)",
        "scoring_breakdown": {
            "capacity_metric": capacity_metric,
            "throughput_metric": throughput_metric,
            "throughput_kgps": throughput_kgps,
            "fidelity_factor": fidelity_factor
        },
        "academic_benchmarks": academic_benchmarks,
        "performance_category": category,
        "max_qubits_simulated": best_qubits,
        "simulation_method": best_method,
        "bond_dimension": best_bond_dim,
        "native_kernel_used": best_native_kernel if successful_runs else False,
        "accelerator_backend": best_accelerator_backend if successful_runs else "none",
        "accelerator_badge": best_accelerator_badge if successful_runs else "",
        "physical_noise_profile": best_physical_noise if successful_runs else None,
        "noise_level": best_noise_level,
        "quantum_state_fidelity": best_fidelity,
        "cpu_overhead_ratio": best_overhead_ratio,
        "ram_savings": best_ram_savings,
        "entanglement_metrics": best_entanglement_metrics,
        "energy_metrics": best_energy_metrics if successful_runs else {},
        "statistical_summary": {
            "runs_count": best_runs_count,
            "mean_latency_seconds": best_mean_latency,
            "median_latency_seconds": best_median_latency,
            "std_latency_seconds": best_std_latency
        },
        "system_metadata": system_metadata,
        "results": results
    }
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    return file_path
