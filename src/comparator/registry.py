import os
import json
import time
import socket
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional

DEFAULT_REGISTRY_CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "registry"))
DEFAULT_TTL_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".quacomp_cache"))
DEFAULT_CACHE_TTL_SECONDS = 86400  # 24 hours TTL
CACHE_METADATA_FILE = "cache_metadata.json"

REMOTE_REGISTRY_BASE_URL = "https://raw.githubusercontent.com/cybort18/QuaComp/main/results/registry/"

ENTERPRISE_BASELINES: Dict[str, Dict[str, Any]] = {
    "apple_m3": {
        "filename": "example_apple_m3.json",
        "description": "Apple M3 (8-core CPU, 10-core GPU, 24GB Unified Memory)",
        "device": "CPU / Metal",
        "category": "High-End Laptop",
        "default_data": {
            "timestamp": "2026-08-12T10:15:00",
            "final_score": 335544830.0,
            "final_composite_score": 335544830.0,
            "score_type": "project-specific composite heuristic score",
            "scoring_breakdown": {"capacity_metric": 33554432.0, "throughput_metric": 510.0},
            "performance_category": "Extreme Workstation",
            "max_qubits_simulated": 25,
            "system_metadata": {
                "cpu_name": "Apple M3 (8-core CPU)",
                "total_ram_gb": 24.0,
                "os_name": "Darwin",
                "os_release": "23.4.0",
                "python_version": "3.12.2"
            },
            "results": [
                {"qubits": 10, "success": True, "latency": 0.082, "mean_latency": 0.082, "median_latency": 0.081, "std_latency": 0.008, "runs_count": 3, "gates": 60, "cpu_usage": 92.0, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0},
                {"qubits": 20, "success": True, "latency": 0.245, "mean_latency": 0.245, "median_latency": 0.243, "std_latency": 0.012, "runs_count": 3, "gates": 220, "cpu_usage": 89.4, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0},
                {"qubits": 25, "success": True, "latency": 0.637, "mean_latency": 0.637, "median_latency": 0.635, "std_latency": 0.018, "runs_count": 3, "gates": 330, "cpu_usage": 94.0, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0}
            ]
        }
    },
    "ryzen3_5300u": {
        "filename": "example_ryzen3_5300u.json",
        "description": "AMD Ryzen 3 5300U (4 cores / 8 threads, 12GB DDR4)",
        "device": "CPU",
        "category": "Entry-Level Laptop",
        "default_data": {
            "timestamp": "2026-08-10T21:02:55",
            "final_score": 10486120.47,
            "final_composite_score": 10486120.47,
            "score_type": "project-specific composite heuristic score",
            "scoring_breakdown": {"capacity_metric": 1048576.0, "throughput_metric": 360.47},
            "performance_category": "High-Performance",
            "max_qubits_simulated": 20,
            "system_metadata": {
                "cpu_name": "AMD Ryzen 3 5300U with Radeon Graphics",
                "total_ram_gb": 11.33,
                "os_name": "Windows",
                "os_release": "11",
                "python_version": "3.13.3"
            },
            "results": [
                {"qubits": 10, "success": True, "latency": 0.312, "mean_latency": 0.312, "median_latency": 0.305, "std_latency": 0.081, "runs_count": 3, "gates": 60, "cpu_usage": 82.8, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0},
                {"qubits": 15, "success": True, "latency": 0.315, "mean_latency": 0.315, "median_latency": 0.310, "std_latency": 0.082, "runs_count": 3, "gates": 127, "cpu_usage": 70.7, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0},
                {"qubits": 20, "success": True, "latency": 0.6103, "mean_latency": 0.6103, "median_latency": 0.605, "std_latency": 0.1028, "runs_count": 3, "gates": 220, "cpu_usage": 88.0, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0}
            ]
        }
    },
    "ryzen7_5800h": {
        "filename": "example_ryzen7_5800h.json",
        "description": "AMD Ryzen 7 5800H (8 cores / 16 threads, 16GB DDR4)",
        "device": "CPU",
        "category": "Performance Workstation",
        "default_data": {
            "timestamp": "2026-08-11T14:30:00",
            "final_score": 167772480.0,
            "final_composite_score": 167772480.0,
            "score_type": "project-specific composite heuristic score",
            "scoring_breakdown": {"capacity_metric": 16777216.0, "throughput_metric": 320.0},
            "performance_category": "Extreme Workstation",
            "max_qubits_simulated": 24,
            "system_metadata": {
                "cpu_name": "AMD Ryzen 7 5800H with Radeon Graphics",
                "total_ram_gb": 16.0,
                "os_name": "Linux",
                "os_release": "6.5.0-ubuntu",
                "python_version": "3.11.8"
            },
            "results": [
                {"qubits": 10, "success": True, "latency": 0.125, "mean_latency": 0.125, "median_latency": 0.122, "std_latency": 0.015, "runs_count": 3, "gates": 60, "cpu_usage": 88.5, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0},
                {"qubits": 20, "success": True, "latency": 0.385, "mean_latency": 0.385, "median_latency": 0.381, "std_latency": 0.022, "runs_count": 3, "gates": 220, "cpu_usage": 82.0, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0},
                {"qubits": 24, "success": True, "latency": 0.9625, "mean_latency": 0.9625, "median_latency": 0.958, "std_latency": 0.045, "runs_count": 3, "gates": 300, "cpu_usage": 91.0, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0}
            ]
        }
    },
    "apple_m4_max": {
        "filename": "example_apple_m4_max.json",
        "description": "Apple M4 Max (16-core CPU, 40-core GPU, 128GB Unified Memory)",
        "device": "CPU / Metal",
        "category": "Extreme Workstation",
        "default_data": {
            "timestamp": "2026-08-15T12:00:00",
            "final_score": 671089200.0,
            "final_composite_score": 671089200.0,
            "score_type": "project-specific composite heuristic score",
            "scoring_breakdown": {"capacity_metric": 67108864.0, "throughput_metric": 1200.0},
            "performance_category": "Extreme Workstation",
            "max_qubits_simulated": 26,
            "system_metadata": {
                "cpu_name": "Apple M4 Max (16-core CPU)",
                "total_ram_gb": 128.0,
                "os_name": "Darwin",
                "os_release": "24.0.0",
                "python_version": "3.13.0"
            },
            "results": [
                {"qubits": 10, "success": True, "latency": 0.042, "mean_latency": 0.042, "median_latency": 0.041, "std_latency": 0.003, "runs_count": 3, "gates": 60, "cpu_usage": 85.0, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0},
                {"qubits": 20, "success": True, "latency": 0.125, "mean_latency": 0.125, "median_latency": 0.124, "std_latency": 0.005, "runs_count": 3, "gates": 220, "cpu_usage": 91.0, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0},
                {"qubits": 26, "success": True, "latency": 0.890, "mean_latency": 0.890, "median_latency": 0.885, "std_latency": 0.015, "runs_count": 3, "gates": 360, "cpu_usage": 96.0, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0}
            ]
        }
    },
    "nvidia_h100": {
        "filename": "example_nvidia_h100.json",
        "description": "NVIDIA H100 SXM5 80GB (Qiskit Aer GPU Backend)",
        "device": "GPU",
        "category": "Data Center Accelerator",
        "default_data": {
            "timestamp": "2026-08-16T14:00:00",
            "final_score": 1342178000.0,
            "final_composite_score": 1342178000.0,
            "score_type": "project-specific composite heuristic score",
            "scoring_breakdown": {"capacity_metric": 134217728.0, "throughput_metric": 3800.0},
            "performance_category": "Data Center GPU",
            "max_qubits_simulated": 27,
            "system_metadata": {
                "cpu_name": "Intel Xeon Platinum 8480+",
                "gpu_name": "NVIDIA H100 80GB HBM3",
                "total_vram_gb": 80.0,
                "total_ram_gb": 512.0,
                "os_name": "Linux",
                "os_release": "6.5.0-generic",
                "python_version": "3.12.3"
            },
            "results": [
                {"qubits": 10, "success": True, "latency": 0.015, "mean_latency": 0.015, "median_latency": 0.014, "std_latency": 0.001, "runs_count": 3, "gates": 60, "cpu_usage": 30.0, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0},
                {"qubits": 20, "success": True, "latency": 0.038, "mean_latency": 0.038, "median_latency": 0.037, "std_latency": 0.002, "runs_count": 3, "gates": 220, "cpu_usage": 45.0, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0},
                {"qubits": 27, "success": True, "latency": 0.450, "mean_latency": 0.450, "median_latency": 0.448, "std_latency": 0.008, "runs_count": 3, "gates": 390, "cpu_usage": 60.0, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0}
            ]
        }
    },
    "aws_graviton4": {
        "filename": "example_aws_graviton4.json",
        "description": "AWS Graviton4 (96 Neoverse V4 cores, 768GB DDR5)",
        "device": "CPU",
        "category": "Cloud HPC",
        "default_data": {
            "timestamp": "2026-08-18T08:00:00",
            "final_score": 536872000.0,
            "final_composite_score": 536872000.0,
            "score_type": "project-specific composite heuristic score",
            "scoring_breakdown": {"capacity_metric": 53687091.0, "throughput_metric": 1850.0},
            "performance_category": "Cloud Server",
            "max_qubits_simulated": 26,
            "system_metadata": {
                "cpu_name": "AWS Graviton4",
                "total_ram_gb": 768.0,
                "os_name": "Linux",
                "os_release": "6.1.0-amazonlinux",
                "python_version": "3.12.4"
            },
            "results": [
                {"qubits": 10, "success": True, "latency": 0.035, "mean_latency": 0.035, "median_latency": 0.034, "std_latency": 0.002, "runs_count": 3, "gates": 60, "cpu_usage": 70.0, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0},
                {"qubits": 20, "success": True, "latency": 0.095, "mean_latency": 0.095, "median_latency": 0.094, "std_latency": 0.004, "runs_count": 3, "gates": 220, "cpu_usage": 80.0, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0},
                {"qubits": 26, "success": True, "latency": 0.720, "mean_latency": 0.720, "median_latency": 0.715, "std_latency": 0.012, "runs_count": 3, "gates": 360, "cpu_usage": 92.0, "ram_status": "SAFE", "workload_label": "QFT", "method": "statevector", "noise_level": "none", "fidelity": 100.0, "overhead_ratio": 0.0}
            ]
        }
    }
}

def get_cache_metadata_path(cache_root: str) -> str:
    """Return path to cache metadata JSON file within specified cache directory."""
    return os.path.join(cache_root, CACHE_METADATA_FILE)

def load_cache_metadata(cache_root: str) -> Dict[str, Any]:
    """Load metadata dictionary from cache directory."""
    meta_path = get_cache_metadata_path(cache_root)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache_metadata(cache_root: str, meta: Dict[str, Any]) -> None:
    """Persist metadata dictionary to cache directory."""
    os.makedirs(cache_root, exist_ok=True)
    meta_path = get_cache_metadata_path(cache_root)
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

def is_cache_valid(
    alias: str,
    cache_root: Optional[str] = None,
    ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS
) -> bool:
    """
    Check if a cached baseline in the local TTL cache exists and has not expired.
    
    Args:
        alias: Enterprise baseline identifier alias.
        cache_root: TTL cache root folder (defaults to .quacomp_cache).
        ttl_seconds: Time-To-Live in seconds (defaults to 86400 / 24 hours).
        
    Returns:
        bool: True if cache is present, valid, and within TTL; False otherwise.
    """
    root = os.path.abspath(cache_root or DEFAULT_TTL_CACHE_DIR)
    if alias not in ENTERPRISE_BASELINES:
        return False
        
    filename = ENTERPRISE_BASELINES[alias]["filename"]
    cache_file = os.path.join(root, filename)
    if not os.path.exists(cache_file):
        return False
        
    meta = load_cache_metadata(root)
    cached_info = meta.get(alias)
    if cached_info and "cached_at" in cached_info:
        cached_at = cached_info["cached_at"]
        return (time.time() - cached_at) < ttl_seconds
        
    # Fallback to filesystem mtime check
    try:
        mtime = os.path.getmtime(cache_file)
        return (time.time() - mtime) < ttl_seconds
    except OSError:
        return False

def invalidate_cache(
    alias: Optional[str] = None,
    cache_root: Optional[str] = None
) -> None:
    """
    Invalidate an entry or all entries in the local TTL cache.
    
    Args:
        alias: Specific alias to invalidate, or None to clear entire cache.
        cache_root: TTL cache directory (defaults to .quacomp_cache).
    """
    root = os.path.abspath(cache_root or DEFAULT_TTL_CACHE_DIR)
    if not os.path.exists(root):
        return
        
    meta = load_cache_metadata(root)
    if alias:
        if alias in ENTERPRISE_BASELINES:
            fname = ENTERPRISE_BASELINES[alias]["filename"]
            fpath = os.path.join(root, fname)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass
        if alias in meta:
            del meta[alias]
            save_cache_metadata(root, meta)
    else:
        for a, entry in ENTERPRISE_BASELINES.items():
            fpath = os.path.join(root, entry["filename"])
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass
        save_cache_metadata(root, {})

def fetch_remote_baselines(
    target_alias: Optional[str] = None,
    cache_dir: Optional[str] = None,
    ttl_cache_dir: Optional[str] = None,
    timeout: float = 5.0,
    ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Synchronize remote enterprise benchmark baselines with local cache and TTL store.
    
    Checks local TTL cache (.quacomp_cache/) before initiating network requests.
    If network is unavailable, offline, or GitHub API rate-limited, gracefully
    populates from authoritative local offline fallback catalog into cache without error.
    
    Args:
        target_alias (Optional[str]): Specific alias to synchronize, or None for all.
        cache_dir (Optional[str]): Destination cache directory (defaults to results/registry).
        ttl_cache_dir (Optional[str]): Local TTL cache directory (defaults to .quacomp_cache).
        timeout (float): Network timeout in seconds (default: 5.0s).
        ttl_seconds (float): Cache Time-To-Live in seconds (default: 86400 / 24 hours).
        force_refresh (bool): Re-download even if already cached.
        
    Returns:
        dict: Detailed sync summary report including offline status, rate-limit telemetry,
              and informative error diagnostics.
    """
    target_dir = os.path.abspath(cache_dir or DEFAULT_REGISTRY_CACHE)
    if ttl_cache_dir:
        ttl_dir = os.path.abspath(ttl_cache_dir)
    elif cache_dir:
        ttl_dir = os.path.abspath(os.path.join(cache_dir, ".quacomp_cache"))
    else:
        ttl_dir = os.path.abspath(DEFAULT_TTL_CACHE_DIR)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(ttl_dir, exist_ok=True)
    
    aliases_to_sync = [target_alias.lower()] if target_alias else list(ENTERPRISE_BASELINES.keys())
    
    synced = []
    cached = []
    fallback = []
    errors: Dict[str, str] = {}
    offline = False
    rate_limited = False
    
    for alias in aliases_to_sync:
        if alias not in ENTERPRISE_BASELINES:
            continue
            
        entry = ENTERPRISE_BASELINES[alias]
        filename = entry["filename"]
        local_dest = os.path.join(target_dir, filename)
        ttl_file = os.path.join(ttl_dir, filename)
        
        # 1. Check TTL cache (.quacomp_cache/) before making any network request
        if not force_refresh and is_cache_valid(alias, cache_root=ttl_dir, ttl_seconds=ttl_seconds):
            if not os.path.exists(local_dest) or local_dest != ttl_file:
                with open(ttl_file, "r", encoding="utf-8") as sf:
                    data = sf.read()
                with open(local_dest, "w", encoding="utf-8") as df:
                    df.write(data)
            cached.append(alias)
            continue
            
        # 2. Check if local_dest in target_dir is already cached and not force_refresh
        if os.path.exists(local_dest) and not force_refresh and target_dir == DEFAULT_REGISTRY_CACHE:
            if not os.path.exists(ttl_file):
                with open(local_dest, "r", encoding="utf-8") as sf:
                    data = sf.read()
                with open(ttl_file, "w", encoding="utf-8") as df:
                    df.write(data)
                meta = load_cache_metadata(ttl_dir)
                meta[alias] = {"filename": filename, "cached_at": time.time(), "source": "registry_preloaded"}
                save_cache_metadata(ttl_dir, meta)
            cached.append(alias)
            continue
            
        # Attempt remote HTTP download with explicit 5-second timeout and robust header inspection
        downloaded = False
        url = REMOTE_REGISTRY_BASE_URL + filename
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "QuaComp-Registry/1.0 (Quantum Benchmark Suite)",
                    "Accept": "application/json, text/plain"
                }
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    raw_content = resp.read().decode("utf-8")
                    # Validate JSON structure before committing to disk
                    json_data = json.loads(raw_content)
                    with open(local_dest, "w", encoding="utf-8") as f:
                        json.dump(json_data, f, indent=2)
                    with open(ttl_file, "w", encoding="utf-8") as tf:
                        json.dump(json_data, tf, indent=2)
                    meta = load_cache_metadata(ttl_dir)
                    meta[alias] = {
                        "filename": filename,
                        "cached_at": time.time(),
                        "source": "remote_download",
                        "ttl_seconds": ttl_seconds
                    }
                    save_cache_metadata(ttl_dir, meta)
                    synced.append(alias)
                    downloaded = True
        except urllib.error.HTTPError as e:
            offline = True
            downloaded = False
            if e.code in (403, 429) or "rate limit" in str(e).lower():
                rate_limited = True
                errors[alias] = f"GitHub API rate limit exceeded (HTTP {e.code}): {e.reason}"
            elif e.code == 404:
                errors[alias] = f"Remote baseline profile not found on registry (HTTP 404): {url}"
            else:
                errors[alias] = f"HTTP error {e.code}: {e.reason}"
        except urllib.error.URLError as e:
            offline = True
            downloaded = False
            errors[alias] = f"Network offline or connection refused: {e.reason}"
        except (TimeoutError, socket.timeout):
            offline = True
            downloaded = False
            errors[alias] = f"Network request timed out (exceeded {timeout}s threshold)"
        except Exception as e:
            offline = True
            downloaded = False
            errors[alias] = f"Remote fetch exception: {str(e)}"
            
        # If remote download was not possible, use canonical registry profile or built-in authoritative fallback
        if not downloaded:
            canonical_file = os.path.join(DEFAULT_REGISTRY_CACHE, filename)
            data_to_write = None
            if target_dir != DEFAULT_REGISTRY_CACHE and os.path.exists(canonical_file):
                with open(canonical_file, "r", encoding="utf-8") as sf:
                    data_to_write = sf.read()
            elif "default_data" in entry:
                data_to_write = json.dumps(entry["default_data"], indent=2)
            elif os.path.exists(local_dest):
                with open(local_dest, "r", encoding="utf-8") as lf:
                    data_to_write = lf.read()
                    
            if data_to_write is not None:
                with open(local_dest, "w", encoding="utf-8") as df:
                    df.write(data_to_write)
                with open(ttl_file, "w", encoding="utf-8") as tf:
                    tf.write(data_to_write)
                meta = load_cache_metadata(ttl_dir)
                meta[alias] = {
                    "filename": filename,
                    "cached_at": time.time(),
                    "source": "offline_fallback",
                    "ttl_seconds": ttl_seconds
                }
                save_cache_metadata(ttl_dir, meta)
                fallback.append(alias)
                
    # Generate informative summary message
    if synced and not offline:
        status_msg = f"Synchronized {len(synced)} baseline(s) successfully from remote registry."
    elif cached and not offline and not synced:
        status_msg = f"Using {len(cached)} existing cached baseline(s)."
    elif rate_limited:
        status_msg = f"GitHub API rate limit exceeded. Gracefully fell back to {len(fallback)} authoritative local/offline profile(s)."
    elif offline:
        status_msg = f"Remote registry unreachable (offline mode). Gracefully populated {len(fallback)} baseline(s) from authoritative local catalog."
    else:
        status_msg = f"Total {len(synced) + len(cached) + len(fallback)} baseline(s) available in registry."
                
    return {
        "target_directory": target_dir,
        "ttl_cache_directory": ttl_dir,
        "synced_remote": synced,
        "existing_cached": cached,
        "offline_fallback": fallback,
        "offline_mode": offline,
        "rate_limited": rate_limited,
        "errors": errors,
        "status_message": status_msg,
        "total_available": len(synced) + len(cached) + len(fallback)
    }

def list_available_baselines(cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all available baseline profiles in the local registry cache.
    """
    target_dir = os.path.abspath(cache_dir or DEFAULT_REGISTRY_CACHE)
    baselines = []
    
    for alias, meta in ENTERPRISE_BASELINES.items():
        fname = meta["filename"]
        cache_path = os.path.join(target_dir, fname)
        is_cached = os.path.exists(cache_path)
        resolved_path = cache_path if is_cached else None
        
        baselines.append({
            "alias": alias,
            "filename": fname,
            "description": meta["description"],
            "device": meta["device"],
            "category": meta["category"],
            "is_available": (resolved_path is not None),
            "file_path": resolved_path
        })
        
    return baselines

def get_baseline_path(alias_or_path: str, cache_dir: Optional[str] = None) -> Optional[str]:
    """
    Find the file path for a baseline profile from registry.
    """
    if os.path.exists(alias_or_path) and os.path.isfile(alias_or_path):
        return os.path.abspath(alias_or_path)
        
    clean_alias = alias_or_path.strip().lower()
    target_dir = os.path.abspath(cache_dir or DEFAULT_REGISTRY_CACHE)
    
    # Direct alias lookup
    if clean_alias in ENTERPRISE_BASELINES:
        fname = ENTERPRISE_BASELINES[clean_alias]["filename"]
        c_path = os.path.join(target_dir, fname)
        if os.path.exists(c_path):
            return c_path
            
    # Check directory matches
    cand = os.path.join(target_dir, f"{clean_alias}.json")
    if os.path.exists(cand):
        return cand
    cand_ex = os.path.join(target_dir, f"example_{clean_alias}.json")
    if os.path.exists(cand_ex):
        return cand_ex
            
    return None
