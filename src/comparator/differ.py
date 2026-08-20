import os
import json
from typing import Dict, Any, List, Optional, Tuple

SAMPLE_PROFILES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "samples"))

KNOWN_ALIASES = {
    "apple_m3": "example_apple_m3.json",
    "m3": "example_apple_m3.json",
    "ryzen3_5300u": "example_ryzen3_5300u.json",
    "5300u": "example_ryzen3_5300u.json",
    "ryzen7_5800h": "example_ryzen7_5800h.json",
    "5800h": "example_ryzen7_5800h.json"
}

def resolve_target_profile(target_alias_or_path: str) -> str:
    """
    Resolve a target file path or built-in sample reference profile alias.
    
    Args:
        target_alias_or_path: Path to a JSON benchmark result or preset alias name.
        
    Returns:
        Absolute file path to the resolved JSON benchmark file.
    """
    if not isinstance(target_alias_or_path, str):
        raise TypeError(f"Target profile or path must be a string, got {type(target_alias_or_path).__name__}")
        
    # Check if target is an exact existing file path
    if os.path.exists(target_alias_or_path) and os.path.isfile(target_alias_or_path):
        return os.path.abspath(target_alias_or_path)
        
    # Check if target matches built-in sample aliases
    clean_alias = target_alias_or_path.strip().lower()
    if clean_alias in KNOWN_ALIASES:
        resolved = os.path.join(SAMPLE_PROFILES_DIR, KNOWN_ALIASES[clean_alias])
        if os.path.exists(resolved):
            return os.path.abspath(resolved)
            
    # Check if file exists inside results/ or results/samples/
    candidates = [
        os.path.join(SAMPLE_PROFILES_DIR, f"{clean_alias}.json"),
        os.path.join(SAMPLE_PROFILES_DIR, f"example_{clean_alias}.json"),
        os.path.join("results", f"{clean_alias}.json"),
        os.path.join("results", f"{target_alias_or_path}")
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return os.path.abspath(c)
            
    raise FileNotFoundError(
        f"Could not resolve benchmark comparison target '{target_alias_or_path}'. "
        f"Available preset aliases: {', '.join(sorted(KNOWN_ALIASES.keys()))} or provide a valid JSON path."
    )

def load_benchmark_json(file_path: str) -> Dict[str, Any]:
    """
    Load and validate a QuaComp benchmark JSON report file.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        Dictionary containing the benchmark data.
    """
    if not isinstance(file_path, str):
        raise TypeError(f"File path must be a string, got {type(file_path).__name__}")
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON benchmark file '{file_path}': {str(e)}")
        
    if not isinstance(data, dict):
        raise ValueError(f"Benchmark JSON in '{file_path}' must be a dictionary object.")
        
    # Validate required keys
    required_keys = ["results", "system_metadata"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Invalid QuaComp benchmark file '{file_path}': Missing required key '{key}'.")
            
    return data

def compare_benchmarks(
    base: Dict[str, Any], 
    target: Dict[str, Any], 
    base_label: Optional[str] = None, 
    target_label: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform relative mathematical comparison between Base and Target benchmark runs.
    
    Args:
        base: Dictionary of base benchmark data.
        target: Dictionary of target benchmark data.
        base_label: Custom label for base system.
        target_label: Custom label for target system.
        
    Returns:
        Structured dictionary containing global deltas, matched qubit comparisons, and summary verdict.
    """
    if not isinstance(base, dict) or not isinstance(target, dict):
        raise TypeError("Both base and target must be benchmark dictionary objects.")
        
    # Determine labels from hardware metadata if not explicitly provided
    base_meta = base.get("system_metadata", {})
    target_meta = target.get("system_metadata", {})
    
    lbl_base = base_label or base_meta.get("cpu_name", "Base System")
    lbl_target = target_label or target_meta.get("cpu_name", "Target System")
    
    # 1. Global Score & Metrics
    score_base = float(base.get("final_composite_score", base.get("final_score", 0.0)))
    score_target = float(target.get("final_composite_score", target.get("final_score", 0.0)))
    
    score_ratio = (score_target / score_base) if score_base > 0 else 1.0
    score_delta_pct = ((score_target - score_base) / score_base * 100.0) if score_base > 0 else 0.0
    
    breakdown_base = base.get("scoring_breakdown", {})
    breakdown_target = target.get("scoring_breakdown", {})
    
    cap_base = float(breakdown_base.get("capacity_metric", 0.0))
    cap_target = float(breakdown_target.get("capacity_metric", 0.0))
    cap_ratio = (cap_target / cap_base) if cap_base > 0 else 1.0
    
    tput_base = float(breakdown_base.get("throughput_metric", 0.0))
    tput_target = float(breakdown_target.get("throughput_metric", 0.0))
    tput_speedup = (tput_target / tput_base) if tput_base > 0 else 1.0
    tput_delta_pct = ((tput_target - tput_base) / tput_base * 100.0) if tput_base > 0 else 0.0
    
    max_q_base = int(base.get("max_qubits_simulated", 0))
    max_q_target = int(target.get("max_qubits_simulated", 0))
    qubit_gap = max_q_target - max_q_base
    
    # 2. Match Qubit Simulation Steps
    results_base = [r for r in base.get("results", []) if r.get("success", False)]
    results_target = [r for r in target.get("results", []) if r.get("success", False)]
    
    # Index target runs by (qubits, method)
    target_map = {
        (r["qubits"], r.get("method", "statevector")): r for r in results_target
    }
    
    matched_qubit_diffs: List[Dict[str, Any]] = []
    
    for rb in results_base:
        q = rb["qubits"]
        method = rb.get("method", "statevector")
        key = (q, method)
        
        if key in target_map:
            rt = target_map[key]
            
            lat_base = float(rb.get("mean_latency", rb.get("latency", 0.0)))
            lat_target = float(rt.get("mean_latency", rt.get("latency", 0.0)))
            
            # Latency delta: (Target - Base) / Base * 100
            # Negative means target is faster!
            lat_delta_pct = ((lat_target - lat_base) / lat_base * 100.0) if lat_base > 0 else 0.0
            
            # Speedup: Base / Target (e.g., 2.0s / 0.5s = 4.0x speedup)
            speedup = (lat_base / lat_target) if lat_target > 0 else 1.0
            
            matched_qubit_diffs.append({
                "qubits": q,
                "method": method,
                "workload_label": rb.get("workload_label", "QFT"),
                "gates": rb.get("gates", 0),
                "latency_base": lat_base,
                "latency_target": lat_target,
                "latency_delta_pct": lat_delta_pct,
                "speedup_factor": speedup,
                "cpu_base": float(rb.get("cpu_usage", 0.0)),
                "cpu_target": float(rt.get("cpu_usage", 0.0)),
                "fidelity_base": float(rb.get("fidelity", 100.0)),
                "fidelity_target": float(rt.get("fidelity", 100.0)),
                "is_faster": lat_target < lat_base
            })
            
    # Sort matched runs by qubit count
    matched_qubit_diffs.sort(key=lambda x: x["qubits"])
    
    # 3. Formulate Summary Verdict
    if tput_speedup >= 1.05:
        speedup_str = f"{tput_speedup:.2f}x faster simulation throughput (+{tput_delta_pct:.1f}%)"
    elif tput_speedup <= 0.95:
        speedup_str = f"{1.0/tput_speedup:.2f}x slower simulation throughput ({tput_delta_pct:.1f}%)"
    else:
        speedup_str = "virtually identical simulation throughput (~par)"
        
    if qubit_gap > 0:
        cap_str = f"with a +{qubit_gap} qubit capacity advantage (2^{qubit_gap}x statevector space)"
    elif qubit_gap < 0:
        cap_str = f"with a {qubit_gap} qubit capacity limit"
    else:
        cap_str = "with identical maximum qubit capacity"
        
    verdict = f"{lbl_target} demonstrates {speedup_str} {cap_str} compared to {lbl_base}."
    
    return {
        "base_label": lbl_base,
        "target_label": lbl_target,
        "base_metadata": base_meta,
        "target_metadata": target_meta,
        "score_summary": {
            "score_base": score_base,
            "score_target": score_target,
            "score_ratio": score_ratio,
            "score_delta_pct": score_delta_pct,
            "category_base": base.get("performance_category", "Unknown"),
            "category_target": target.get("performance_category", "Unknown"),
            "capacity_metric_base": cap_base,
            "capacity_metric_target": cap_target,
            "capacity_ratio": cap_ratio,
            "throughput_metric_base": tput_base,
            "throughput_metric_target": tput_target,
            "throughput_speedup": tput_speedup,
            "throughput_delta_pct": tput_delta_pct,
            "max_qubits_base": max_q_base,
            "max_qubits_target": max_q_target,
            "qubit_gap": qubit_gap
        },
        "matched_qubits": matched_qubit_diffs,
        "verdict": verdict
    }
