import time
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def run_simulation(
    circuit: QuantumCircuit, 
    method: str = 'statevector', 
    bond_dimension: int = 64,
    device: str = 'CPU',
    noise_model: Any = None,
    noise_level: str = 'none',
    shots: int = 1024,
    runs: int = 3,
    workers: int = 1
) -> Dict[str, Any]:
    """
    Execute a Qiskit quantum circuit using AerSimulator across multiple benchmark runs for statistical repeatability,
    with multi-GPU and parallel distributed worker pool support.
    
    Measures execution latency across `runs` iterations and calculates Mean, Median, and Standard Deviation.
    
    Args:
        circuit (QuantumCircuit): The Qiskit quantum circuit to execute.
        method (str): The simulation method ('statevector' or 'mps'/'matrix_product_state').
        bond_dimension (int): Max bond dimension for MPS.
        device (str): Compute device ('CPU', 'GPU', 'MULTI_GPU', or 'GPU:ALL').
        noise_model (Optional[NoiseModel]): Optional Qiskit Aer NoiseModel instance.
        noise_level (str): Label of the noise level ('none', 'low', 'medium', 'high').
        shots (int): Number of measurement shots (default 1024).
        runs (int): Number of benchmark iterations (default 3).
        workers (int): Number of parallel distributed worker processes/threads (default 1).
        
    Returns:
        dict: A dictionary containing execution telemetry and statistical metrics:
            - "success" (bool): True if simulation succeeded, False otherwise.
            - "latency" (float): Mean execution time in seconds.
            - "latencies" (list): List of execution latencies for each run.
            - "mean_latency" (float): Sample mean latency.
            - "median_latency" (float): Sample median latency.
            - "std_latency" (float): Sample standard deviation of latency.
            - "runs_count" (int): Number of successful runs completed.
            - "device" (str): Device used ('CPU', 'GPU', or 'MULTI_GPU').
            - "workers" (int): Worker pool parallelism level used.
            - "error" (str or None): Error message if failed, None if succeeded.
            - "counts" (dict): Measurement counts dictionary from final run.
            - "metadata" (dict): Simulator execution metadata if succeeded, empty dict otherwise.
            
    Raises:
        TypeError: If the input circuit is not a Qiskit QuantumCircuit.
        ValueError: If runs is less than 1 or device is invalid.
    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("Input must be a Qiskit QuantumCircuit instance.")
        
    if not isinstance(runs, int) or runs < 1:
        raise ValueError("Number of runs must be an integer >= 1.")
        
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("Number of workers must be an integer >= 1.")
        
    dev_clean = str(device).upper()
    is_multi_gpu = dev_clean in ("MULTI_GPU", "GPU:ALL", "MULTI-GPU")
    is_gpu_req = dev_clean.startswith("GPU") or is_multi_gpu
    
    valid_devices = ("CPU", "GPU", "MULTI_GPU", "GPU:ALL", "MULTI-GPU")
    if dev_clean not in valid_devices:
        raise ValueError(f"Invalid device '{device}'. Must be one of {valid_devices}.")
        
    try:
        # Check available devices in AerSimulator
        temp_sim = AerSimulator()
        avail_devices = [str(d).upper() for d in temp_sim.available_devices()]
        
        if is_gpu_req and "GPU" not in avail_devices:
            return {
                "success": False,
                "latency": 0.0,
                "latencies": [],
                "mean_latency": 0.0,
                "median_latency": 0.0,
                "std_latency": 0.0,
                "runs_count": 0,
                "counts": {},
                "device": "MULTI_GPU" if is_multi_gpu else "GPU",
                "workers": workers,
                "error": f"GPU device requested, but Qiskit Aer on this system does not have GPU/CUDA backend support enabled. Available devices: {avail_devices}",
                "metadata": {}
            }
            
        # Prepare circuit with measurements for count extraction if needed
        circ_to_run = circuit.copy()
        if len(circ_to_run.cregs) == 0:
            circ_to_run.measure_all()
            
        # Initialize AerSimulator based on simulation method, device, multi-GPU options & noise model
        aer_device_target = "GPU" if is_gpu_req else "CPU"
        sim_kwargs: Dict[str, Any] = {"device": aer_device_target}
        
        # Configure multi-device / parallel cluster options
        if is_multi_gpu:
            sim_kwargs["batched_shots_gpu"] = True
            sim_kwargs["blocking_enable"] = True
            
        if method in ('mps', 'matrix_product_state'):
            sim_kwargs['method'] = 'matrix_product_state'
            sim_kwargs['matrix_product_state_max_bond_dimension'] = bond_dimension
        else:
            sim_kwargs['method'] = 'statevector'
            
        if noise_model is not None:
            sim_kwargs['noise_model'] = noise_model
            
        simulator = AerSimulator(**sim_kwargs)
        transpiled_circuit = transpile(circ_to_run, simulator)
        
        latencies: List[float] = []
        last_counts: Dict[str, int] = {}
        last_result = None
        
        def _execute_single_run(_):
            t0 = time.perf_counter()
            job = simulator.run(transpiled_circuit, shots=shots)
            res = job.result()
            lat = time.perf_counter() - t0
            return lat, res
            
        # Multi-worker parallel execution vs serial execution
        if workers > 1 and runs > 1:
            with ThreadPoolExecutor(max_workers=min(workers, runs)) as executor:
                futures = [executor.submit(_execute_single_run, i) for i in range(runs)]
                for fut in as_completed(futures):
                    run_lat, res = fut.result()
                    latencies.append(run_lat)
                    last_result = res
        else:
            for _ in range(runs):
                run_lat, res = _execute_single_run(_)
                latencies.append(run_lat)
                last_result = res
            
        if last_result is not None:
            last_counts = last_result.get_counts()
            
        mean_latency = float(np.mean(latencies))
        median_latency = float(np.median(latencies))
        std_latency = float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0
        
        device_label = "MULTI_GPU" if is_multi_gpu else aer_device_target
        
        # Extract metadata from result
        metadata = {
            "backend_name": last_result.backend_name if last_result else "AerSimulator",
            "backend_version": last_result.backend_version if last_result else "Unknown",
            "job_id": last_result.job_id if last_result else None,
            "success": last_result.success if last_result else True,
            "method": method,
            "bond_dimension": bond_dimension if method in ('mps', 'matrix_product_state') else None,
            "device": device_label,
            "workers": workers,
            "noise_level": noise_level,
            "runs_count": runs,
            "counts": last_counts
        }
        
        return {
            "success": True,
            "latency": mean_latency,
            "latencies": latencies,
            "mean_latency": mean_latency,
            "median_latency": median_latency,
            "std_latency": std_latency,
            "runs_count": len(latencies),
            "counts": last_counts,
            "device": device_label,
            "workers": workers,
            "error": None,
            "metadata": metadata
        }
        
    except Exception as e:
        return {
            "success": False,
            "latency": 0.0,
            "latencies": [],
            "mean_latency": 0.0,
            "median_latency": 0.0,
            "std_latency": 0.0,
            "runs_count": 0,
            "counts": {},
            "device": "MULTI_GPU" if is_multi_gpu else dev_clean,
            "workers": workers,
            "error": str(e),
            "metadata": {}
        }
