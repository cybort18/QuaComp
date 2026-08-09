import json
import os
import time
from typing import List, Dict, Any

def export_to_json(results: List[Dict[str, Any]], system_metadata: Dict[str, Any], output_dir: str = "results") -> str:
    """
    Export benchmark results, scoring breakdown, statistical metrics, and system telemetry to a JSON file.
    
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
    
    if successful_runs:
        from src.scorer.calculator import calculate_scoring_breakdown, categorize_score
        best_run = max(successful_runs, key=lambda x: x["qubits"])
        best_qubits = best_run["qubits"]
        gates = best_run["gates"]
        best_mean_latency = best_run.get("mean_latency", best_run.get("latency", 0.0))
        best_median_latency = best_run.get("median_latency", best_mean_latency)
        best_std_latency = best_run.get("std_latency", 0.0)
        best_runs_count = best_run.get("runs_count", 1)
        
        breakdown = calculate_scoring_breakdown(best_qubits, gates, best_mean_latency)
        score = breakdown["composite_score"]
        capacity_metric = breakdown["capacity_metric"]
        throughput_metric = breakdown["throughput_metric"]
        category = categorize_score(score)
        
        best_method = best_run.get("method", "statevector")
        best_bond_dim = best_run.get("bond_dimension")
        best_ram_savings = best_run.get("ram_savings", {})
        best_noise_level = best_run.get("noise_level", "none")
        best_fidelity = best_run.get("fidelity", 100.0)
        best_overhead_ratio = best_run.get("overhead_ratio", 0.0)
        
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "final_score": score,
        "final_composite_score": score,
        "score_type": "project-specific composite heuristic score",
        "scoring_breakdown": {
            "capacity_metric": capacity_metric,
            "throughput_metric": throughput_metric
        },
        "performance_category": category,
        "max_qubits_simulated": best_qubits,
        "simulation_method": best_method,
        "bond_dimension": best_bond_dim,
        "noise_level": best_noise_level,
        "quantum_state_fidelity": best_fidelity,
        "cpu_overhead_ratio": best_overhead_ratio,
        "ram_savings": best_ram_savings,
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
