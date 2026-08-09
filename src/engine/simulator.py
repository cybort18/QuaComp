import time
from typing import Any, Dict
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def run_simulation(
    circuit: QuantumCircuit, 
    method: str = 'statevector', 
    bond_dimension: int = 64,
    noise_model: Any = None,
    noise_level: str = 'none',
    shots: int = 1024
) -> Dict[str, Any]:
    """
    Execute a Qiskit quantum circuit using AerSimulator with selected method (statevector or mps) and noise model.
    
    Measures the execution latency including transpilation and simulation run.
    
    Args:
        circuit (QuantumCircuit): The Qiskit quantum circuit to execute.
        method (str): The simulation method ('statevector' or 'mps'/'matrix_product_state').
        bond_dimension (int): Max bond dimension for MPS.
        noise_model (Optional[NoiseModel]): Optional Qiskit Aer NoiseModel instance.
        noise_level (str): Label of the noise level ('none', 'low', 'medium', 'high').
        shots (int): Number of measurement shots (default 1024).
        
    Returns:
        dict: A dictionary containing execution telemetry:
            - "success" (bool): True if simulation succeeded, False otherwise.
            - "latency" (float): Total execution time in seconds.
            - "error" (str or None): Error message if failed, None if succeeded.
            - "counts" (dict): Measurement counts dictionary.
            - "metadata" (dict): Simulator execution metadata if succeeded, empty dict otherwise.
            
    Raises:
        TypeError: If the input circuit is not a Qiskit QuantumCircuit.
    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("Input must be a Qiskit QuantumCircuit instance.")
        
    start_time = time.perf_counter()
    
    try:
        # Prepare circuit with measurements for count extraction if needed
        circ_to_run = circuit.copy()
        if len(circ_to_run.cregs) == 0:
            circ_to_run.measure_all()
            
        # Initialize AerSimulator based on simulation method & noise model
        sim_kwargs: Dict[str, Any] = {}
        if method in ('mps', 'matrix_product_state'):
            sim_kwargs['method'] = 'matrix_product_state'
            sim_kwargs['matrix_product_state_max_bond_dimension'] = bond_dimension
        else:
            sim_kwargs['method'] = 'statevector'
            
        if noise_model is not None:
            sim_kwargs['noise_model'] = noise_model
            
        simulator = AerSimulator(**sim_kwargs)
            
        # Transpile circuit for the simulator backend
        transpiled_circuit = transpile(circ_to_run, simulator)
        
        # Run simulation and fetch result
        job = simulator.run(transpiled_circuit, shots=shots)
        result = job.result()
        
        counts = result.get_counts()
        latency = time.perf_counter() - start_time
        
        # Extract metadata from result
        metadata = {
            "backend_name": result.backend_name,
            "backend_version": result.backend_version,
            "job_id": result.job_id,
            "success": result.success,
            "method": method,
            "bond_dimension": bond_dimension if method in ('mps', 'matrix_product_state') else None,
            "noise_level": noise_level,
            "counts": counts
        }
        
        return {
            "success": True,
            "latency": latency,
            "counts": counts,
            "error": None,
            "metadata": metadata
        }
        
    except Exception as e:
        latency = time.perf_counter() - start_time
        return {
            "success": False,
            "latency": latency,
            "counts": {},
            "error": str(e),
            "metadata": {}
        }
