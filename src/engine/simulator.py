import time
from typing import Any, Dict, List
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
    runs: int = 3
) -> Dict[str, Any]:
    """
    Execute a Qiskit quantum circuit using AerSimulator across multiple benchmark runs for statistical repeatability.
    
    Measures execution latency across `runs` iterations and calculates Mean, Median, and Standard Deviation.
    
    Args:
        circuit (QuantumCircuit): The Qiskit quantum circuit to execute.
        method (str): The simulation method ('statevector' or 'mps'/'matrix_product_state').
        bond_dimension (int): Max bond dimension for MPS.
        device (str): Compute device ('CPU' or 'GPU').
        noise_model (Optional[NoiseModel]): Optional Qiskit Aer NoiseModel instance.
        noise_level (str): Label of the noise level ('none', 'low', 'medium', 'high').
        shots (int): Number of measurement shots (default 1024).
        runs (int): Number of benchmark iterations (default 3).
        
    Returns:
        dict: A dictionary containing execution telemetry and statistical metrics:
            - "success" (bool): True if simulation succeeded, False otherwise.
            - "latency" (float): Mean execution time in seconds.
            - "latencies" (list): List of execution latencies for each run.
            - "mean_latency" (float): Sample mean latency.
            - "median_latency" (float): Sample median latency.
            - "std_latency" (float): Sample standard deviation of latency.
            - "runs_count" (int): Number of successful runs completed.
            - "device" (str): Device used ('CPU' or 'GPU').
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
        
    dev_clean = str(device).upper()
    if dev_clean not in ("CPU", "GPU"):
        raise ValueError(f"Invalid device '{device}'. Must be 'CPU' or 'GPU'.")
        
    try:
        # Check available devices in AerSimulator
        temp_sim = AerSimulator()
        avail_devices = [str(d).upper() for d in temp_sim.available_devices()]
        if dev_clean == "GPU" and "GPU" not in avail_devices:
            return {
                "success": False,
                "latency": 0.0,
                "latencies": [],
                "mean_latency": 0.0,
                "median_latency": 0.0,
                "std_latency": 0.0,
                "runs_count": 0,
                "counts": {},
                "device": dev_clean,
                "error": f"GPU device requested, but Qiskit Aer on this system does not have GPU/CUDA backend support enabled. Available devices: {avail_devices}",
                "metadata": {}
            }
            
        # Prepare circuit with measurements for count extraction if needed
        circ_to_run = circuit.copy()
        if len(circ_to_run.cregs) == 0:
            circ_to_run.measure_all()
            
        # Initialize AerSimulator based on simulation method, device & noise model
        sim_kwargs: Dict[str, Any] = {"device": dev_clean}
        if method in ('mps', 'matrix_product_state'):
            sim_kwargs['method'] = 'matrix_product_state'
            sim_kwargs['matrix_product_state_max_bond_dimension'] = bond_dimension
        else:
            sim_kwargs['method'] = 'statevector'
            
        if noise_model is not None:
            sim_kwargs['noise_model'] = noise_model
            
        simulator = AerSimulator(**sim_kwargs)
        
        latencies: List[float] = []
        last_counts: Dict[str, int] = {}
        last_result = None
        
        # Execute multi-run benchmark loop for statistical reproducibility
        for _ in range(runs):
            run_start = time.perf_counter()
            transpiled_circuit = transpile(circ_to_run, simulator)
            job = simulator.run(transpiled_circuit, shots=shots)
            last_result = job.result()
            run_lat = time.perf_counter() - run_start
            latencies.append(run_lat)
            
        if last_result is not None:
            last_counts = last_result.get_counts()
            
        mean_latency = float(np.mean(latencies))
        median_latency = float(np.median(latencies))
        std_latency = float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0
        
        # Extract metadata from result
        metadata = {
            "backend_name": last_result.backend_name if last_result else "AerSimulator",
            "backend_version": last_result.backend_version if last_result else "Unknown",
            "job_id": last_result.job_id if last_result else None,
            "success": last_result.success if last_result else True,
            "method": method,
            "bond_dimension": bond_dimension if method in ('mps', 'matrix_product_state') else None,
            "device": dev_clean,
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
            "device": dev_clean,
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
            "error": str(e),
            "metadata": {}
        }
