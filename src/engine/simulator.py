import os
import time
import math
from typing import Any, Dict, List, Optional, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from src.engine.noise import (
    DensityMatrixMemoryError,
    TrajectoryNoiseModel,
    get_trajectory_noise_model,
    apply_gate_1q_to_statevector,
    apply_gate_2q_to_statevector
)


def _trajectory_worker_batch(
    parsed_ops: List[Tuple[str, List[int], np.ndarray]],
    num_qubits: int,
    batch_shots: int,
    traj_model: Optional[TrajectoryNoiseModel],
    seed_seq: Any
) -> Dict[str, int]:
    """
    Execute a batch of stochastic quantum trajectories independently on a single worker process or thread.
    """
    rng = np.random.default_rng(seed_seq)
    worker_counts: Dict[str, int] = {}
    
    if traj_model is None:
        sv = np.zeros(1 << num_qubits, dtype=np.complex128)
        sv[0] = 1.0 + 0.0j
        for name, q_idx, mat in parsed_ops:
            if len(q_idx) == 1:
                apply_gate_1q_to_statevector(sv, mat, q_idx[0], num_qubits)
            elif len(q_idx) == 2:
                apply_gate_2q_to_statevector(sv, mat, q_idx[0], q_idx[1], num_qubits)
                
        probs = np.abs(sv) ** 2
        p_sum = float(np.sum(probs))
        if p_sum > 0:
            probs /= p_sum
        sampled_indices = rng.choice(1 << num_qubits, size=batch_shots, p=probs)
        for idx in sampled_indices:
            bs = f"{int(idx):0{num_qubits}b}"
            worker_counts[bs] = worker_counts.get(bs, 0) + 1
    else:
        for _ in range(batch_shots):
            sv = np.zeros(1 << num_qubits, dtype=np.complex128)
            sv[0] = 1.0 + 0.0j
            for name, q_idx, mat in parsed_ops:
                if len(q_idx) == 1:
                    apply_gate_1q_to_statevector(sv, mat, q_idx[0], num_qubits)
                elif len(q_idx) == 2:
                    apply_gate_2q_to_statevector(sv, mat, q_idx[0], q_idx[1], num_qubits)
                traj_model.apply_stochastic_gate_noise(sv, q_idx, name, rng)
                
            probs = np.abs(sv) ** 2
            p_sum = float(np.sum(probs))
            if p_sum > 0:
                probs /= p_sum
                
            sample_idx = int(rng.choice(1 << num_qubits, p=probs))
            raw_bs = f"{sample_idx:0{num_qubits}b}"
            noisy_bs = traj_model.apply_readout_error(raw_bs, rng)
            worker_counts[noisy_bs] = worker_counts.get(noisy_bs, 0) + 1
            
    return worker_counts


def simulate_trajectories(
    circuit: QuantumCircuit,
    noise_model: Any = None,
    shots: int = 1024,
    backend: str = 'auto',
    seed: Optional[int] = None,
    num_workers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Execute quantum circuit simulation using the Monte Carlo Wavefunction (MCWF) /
    Quantum Trajectories method, strictly scaling memory at O(2^n).
    
    Args:
        circuit (QuantumCircuit): The Qiskit quantum circuit to execute.
        noise_model (Any): TrajectoryNoiseModel, PhysicalNoiseModel, preset string ('low', 'medium', 'high'), or None.
        shots (int): Number of trajectory paths to sample (default 1024).
        backend (str): Compute accelerator backend ('auto', 'cuda', 'metal', 'cpp', 'numpy').
        seed (Optional[int]): Random seed for stochastic jump reproducibility.
        num_workers (Optional[int]): Number of parallel worker threads/processes for batch trajectory sampling.
        
    Returns:
        Dict[str, Any]: Execution metrics and measurement counts.
    """
    if hasattr(circuit, "to_quantum_circuit"):
        circuit = circuit.to_quantum_circuit()
    elif not isinstance(circuit, QuantumCircuit):
        raise TypeError("Input must be a Qiskit QuantumCircuit or ParsedQASMCircuit instance.")
        
    num_qubits = circuit.num_qubits
    if num_qubits < 1:
        raise ValueError("Circuit must have at least 1 qubit.")
        
    from src.engine.accelerator import get_best_backend
    active_be = get_best_backend(backend)
    active_be.device_synchronize()
    
    # Resolve TrajectoryNoiseModel
    traj_model: Optional[TrajectoryNoiseModel] = None
    if isinstance(noise_model, TrajectoryNoiseModel):
        traj_model = noise_model
    elif isinstance(noise_model, str):
        traj_model = get_trajectory_noise_model(noise_model)
    elif hasattr(noise_model, "to_trajectory_noise_model"):
        traj_model = noise_model.to_trajectory_noise_model(active_qubits=list(range(num_qubits)))
        
    t0 = time.perf_counter()
    
    # Clean circuit for execution (remove measurements for pure statevector propagation)
    circ_clean = circuit.copy()
    if len(circ_clean.clbits) > 0:
        circ_clean.remove_final_measurements(inplace=True)
        
    basis_gates = ['h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz', 'cx', 'cz', 'swap', 'u', 'u1', 'u2', 'u3', 'p', 'sx', 'id']
    tqc = transpile(circ_clean, basis_gates=basis_gates)
    
    # Pre-parse instructions once
    parsed_ops = []
    for inst in tqc.data:
        op = inst.operation
        if op.name in ('barrier', 'measure', 'id'):
            continue
        q_indices = [tqc.find_bit(q).index for q in inst.qubits]
        try:
            mat = op.to_matrix()
            parsed_ops.append((op.name, q_indices, mat))
        except Exception:
            pass
            
    # Determine parallel worker allocation:
    # On CUDA GPU, force single-process to avoid CUDA context multi-processing collisions
    if backend.lower() == 'cuda' or getattr(active_be, "name", "").lower() == "cuda":
        effective_workers = 1
    elif num_workers is not None:
        effective_workers = max(1, int(num_workers))
    else:
        effective_workers = min(os.cpu_count() or 1, 8)
        
    counts: Dict[str, int] = {}
    
    if shots < 100 or effective_workers <= 1:
        # Serial execution in main thread
        counts = _trajectory_worker_batch(parsed_ops, num_qubits, shots, traj_model, seed)
    else:
        # Parallel execution across worker pool with orthogonal SeedSequence streams
        base_shots = shots // effective_workers
        rem = shots % effective_workers
        batch_sizes = [base_shots + (1 if i < rem else 0) for i in range(effective_workers)]
        batch_sizes = [b for b in batch_sizes if b > 0]
        actual_workers = len(batch_sizes)
        
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(actual_workers)
        
        batch_results = []
        try:
            with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                futures = [
                    executor.submit(_trajectory_worker_batch, parsed_ops, num_qubits, batch_sizes[i], traj_model, child_seeds[i])
                    for i in range(actual_workers)
                ]
                for fut in as_completed(futures):
                    batch_results.append(fut.result())
        except Exception:
            batch_results = []
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                futures = [
                    executor.submit(_trajectory_worker_batch, parsed_ops, num_qubits, batch_sizes[i], traj_model, child_seeds[i])
                    for i in range(actual_workers)
                ]
                for fut in as_completed(futures):
                    batch_results.append(fut.result())
                    
        for b_counts in batch_results:
            for bs, cnt in b_counts.items():
                counts[bs] = counts.get(bs, 0) + cnt
            
    active_be.device_synchronize()
    latency = time.perf_counter() - t0
    saved_ratio = 1 << num_qubits
    
    metadata = {
        "backend_name": "QuaComp Trajectory Engine (MCWF)",
        "backend_version": "1.0.0",
        "method": "trajectory",
        "noise_method": "trajectory",
        "shots": shots,
        "workers": effective_workers,
        "qubits": num_qubits,
        "memory_saved_ratio": saved_ratio,
        "noise_model": traj_model.name if traj_model else "None",
        "counts": counts
    }
    
    return {
        "success": True,
        "qubits": num_qubits,
        "latency": latency,
        "latencies": [latency],
        "mean_latency": latency,
        "median_latency": latency,
        "std_latency": 0.0,
        "runs_count": 1,
        "counts": counts,
        "device": backend.upper(),
        "workers": effective_workers,
        "native_kernel_used": False,
        "accelerator_backend": active_be.tier_label,
        "accelerator_badge": None,
        "fusion_metrics": None,
        "physical_noise_profile": traj_model.name if traj_model else None,
        "noise_method": "trajectory",
        "memory_saved_ratio": saved_ratio,
        "error": None,
        "metadata": metadata
    }



def run_simulation(
    circuit: QuantumCircuit, 
    method: str = 'statevector', 
    bond_dimension: int = 64,
    device: str = 'CPU',
    noise_model: Any = None,
    noise_level: str = 'none',
    shots: int = 1024,
    runs: int = 3,
    workers: int = 1,
    state_slicing: bool = False,
    blocking_qubits: Optional[int] = None,
    use_native_kernels: bool = False,
    physical_noise_profile: Optional[str] = None,
    backend: str = 'auto',
    noise_method: str = 'trajectory'
) -> Dict[str, Any]:

    """
    Execute a Qiskit quantum circuit using AerSimulator across multiple benchmark runs for statistical repeatability,
    with multi-GPU, native kernel gate fusion (C++/Metal/CUDA), physical QPU noise calibration, and parallel distributed worker pool support.
    
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
    if hasattr(circuit, "to_quantum_circuit"):
        circuit = circuit.to_quantum_circuit()
    elif not isinstance(circuit, QuantumCircuit):
        raise TypeError("Input must be a Qiskit QuantumCircuit or ParsedQASMCircuit instance.")
        
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

    # Pre-flight Safety Guard for Density Matrix simulation (O(4^n) exponential blowup)
    num_q_check = circuit.num_qubits if hasattr(circuit, 'num_qubits') else getattr(circuit, 'num_qubits', 10)
    is_dm_check = (method in ('density_matrix', 'density-matrix') or noise_method == 'density_matrix')
    if is_dm_check and num_q_check >= 16:
        req_dm_bytes = (1 << (2 * num_q_check)) * 16
        req_gb = req_dm_bytes / (1024 ** 3)
        raise DensityMatrixMemoryError(
            f"Density matrix simulation requires O(4^n) = O(2^(2*{num_q_check})) memory "
            f"(~{req_gb:,.1f} GB), which exceeds the physical memory safety threshold for n >= 16 qubits. "
            f"Please use '--noise-method trajectory' (Monte Carlo Wavefunction) to maintain O(2^n) memory scaling."
        )
        
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
            
        # Resolve Physical QPU Noise Model if requested
        active_noise_profile = None
        if physical_noise_profile:
            from src.engine.physical_noise import PhysicalNoiseModel
            pnm = PhysicalNoiseModel.from_json(physical_noise_profile)
            noise_model = pnm.to_qiskit_noise_model(active_qubits=list(range(circuit.num_qubits)))
            active_noise_profile = pnm.backend_name
            if noise_level == 'none':
                noise_level = f"physical:{pnm.backend_name}"
                
        # Resolve Native Kernel Acceleration (Single-Pass Gate Fusion: C++/Metal/CUDA)
        circ_to_run = circuit.copy()
        fusion_metrics = None
        accelerator_backend = None
        accelerator_badge = None
        
        from src.engine.accelerator import get_best_backend, get_accelerator_badge, InsufficientVRAMError
        is_backend_req = bool(backend and str(backend).lower() != "auto")
        active_backend_obj = get_best_backend(backend if is_backend_req else ("cuda" if is_gpu_req else None))

        if use_native_kernels or is_backend_req:
            from src.engine.fusion import fuse_circuit_single_pass
            circ_to_run, fusion_metrics = fuse_circuit_single_pass(circ_to_run)
            accelerator_backend = active_backend_obj.tier_label
            accelerator_badge = get_accelerator_badge(accelerator_backend)
            
        # Prepare circuit with measurements for count extraction if needed
        if len(circ_to_run.cregs) == 0:
            circ_to_run.measure_all()
            
        # Initialize AerSimulator based on simulation method, device, multi-GPU options & noise model
        aer_device_target = "GPU" if is_gpu_req else "CPU"
        sim_kwargs: Dict[str, Any] = {"device": aer_device_target}
        
        # Configure multi-device / parallel cluster options & distributed statevector slicing
        num_q = circ_to_run.num_qubits
        use_slicing = is_multi_gpu or state_slicing
        effective_blocking = None
        chunk_count = 1
        
        if use_slicing:
            sim_kwargs["batched_shots_gpu"] = True
            sim_kwargs["blocking_enable"] = True
            if blocking_qubits is not None:
                effective_blocking = max(1, min(blocking_qubits, num_q - 1))
            else:
                # Optimal statevector chunk size: chunk into slices if num_q > 12
                effective_blocking = max(10, min(num_q - 2, 20)) if num_q > 12 else max(1, num_q - 1)
            sim_kwargs["blocking_qubits"] = effective_blocking
            if num_q > effective_blocking:
                chunk_count = 2 ** (num_q - effective_blocking)

        is_dm = is_dm_check
            
        # Trajectory Simulation Path (Monte Carlo Wavefunction)
        if noise_method == 'trajectory' and method not in ('mps', 'matrix_product_state'):
            traj_model = None
            if physical_noise_profile:
                from src.engine.physical_noise import PhysicalNoiseModel
                pnm = PhysicalNoiseModel.from_json(physical_noise_profile)
                traj_model = pnm.to_trajectory_noise_model(active_qubits=list(range(num_q)))
            elif isinstance(noise_model, TrajectoryNoiseModel):
                traj_model = noise_model
            elif noise_level != 'none':
                traj_model = get_trajectory_noise_model(noise_level)
                
            if traj_model is not None:
                latencies: List[float] = []
                last_counts: Dict[str, int] = {}
                for _ in range(runs):
                    t_res = simulate_trajectories(
                        circ_to_run,
                        noise_model=traj_model,
                        shots=shots,
                        backend=backend,
                        seed=None,
                        num_workers=workers
                    )
                    latencies.append(t_res["latency"])
                    last_counts = t_res["counts"]
                    
                mean_latency = float(np.mean(latencies))
                median_latency = float(np.median(latencies))
                std_latency = float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0
                device_label = "MULTI_GPU" if is_multi_gpu else aer_device_target
                saved_ratio = 1 << num_q
                
                metadata = {
                    "backend_name": "QuaComp Trajectory Engine (MCWF)",
                    "backend_version": "1.0.0",
                    "job_id": None,
                    "success": True,
                    "method": "trajectory",
                    "noise_method": "trajectory",
                    "bond_dimension": None,
                    "device": device_label,
                    "workers": workers,
                    "noise_level": noise_level,
                    "physical_noise_profile": active_noise_profile,
                    "native_kernel_used": use_native_kernels,
                    "accelerator_backend": accelerator_backend,
                    "accelerator_badge": accelerator_badge,
                    "fusion_metrics": fusion_metrics,
                    "runs_count": runs,
                    "model_parallelism": "Distributed Statevector Slicing" if use_slicing else "None",
                    "blocking_qubits": effective_blocking,
                    "chunk_count": chunk_count,
                    "state_slicing": use_slicing,
                    "memory_saved_ratio": saved_ratio,
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
                    "native_kernel_used": use_native_kernels or is_backend_req,
                    "accelerator_backend": accelerator_backend,
                    "accelerator_badge": accelerator_badge,
                    "fusion_metrics": fusion_metrics,
                    "physical_noise_profile": active_noise_profile,
                    "noise_method": "trajectory",
                    "memory_saved_ratio": saved_ratio,
                    "error": None,
                    "metadata": metadata
                }

        if method in ('mps', 'matrix_product_state'):
            sim_kwargs['method'] = 'matrix_product_state'
            sim_kwargs['matrix_product_state_max_bond_dimension'] = bond_dimension
        elif is_dm:
            sim_kwargs['method'] = 'density_matrix'
        else:
            sim_kwargs['method'] = 'statevector'
            
        if noise_model is not None and not isinstance(noise_model, TrajectoryNoiseModel):
            sim_kwargs['noise_model'] = noise_model
            
        simulator = AerSimulator(**sim_kwargs)
        transpiled_circuit = transpile(circ_to_run, simulator)

        
        latencies: List[float] = []
        last_counts: Dict[str, int] = {}
        last_result = None
        
        def _execute_single_run(_):
            if active_backend_obj is not None:
                active_backend_obj.device_synchronize()
            t0 = time.perf_counter()
            job = simulator.run(transpiled_circuit, shots=shots)
            res = job.result()
            if active_backend_obj is not None:
                active_backend_obj.device_synchronize()
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
            "physical_noise_profile": active_noise_profile,
            "native_kernel_used": use_native_kernels,
            "accelerator_backend": accelerator_backend,
            "accelerator_badge": accelerator_badge,
            "fusion_metrics": fusion_metrics,
            "runs_count": runs,
            "model_parallelism": "Distributed Statevector Slicing" if use_slicing else "None",
            "blocking_qubits": effective_blocking,
            "chunk_count": chunk_count,
            "state_slicing": use_slicing,
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
            "native_kernel_used": use_native_kernels or is_backend_req,
            "accelerator_backend": accelerator_backend,
            "accelerator_badge": accelerator_badge,
            "fusion_metrics": fusion_metrics,
            "physical_noise_profile": active_noise_profile,
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
            "native_kernel_used": use_native_kernels or is_backend_req,
            "accelerator_backend": None,
            "accelerator_badge": None,
            "fusion_metrics": None,
            "physical_noise_profile": physical_noise_profile,
            "error": str(e),
            "metadata": {}
        }
