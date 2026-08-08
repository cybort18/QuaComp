import time
from typing import Any, Dict
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def run_simulation(circuit: QuantumCircuit, method: str = 'statevector', bond_dimension: int = 64) -> Dict[str, Any]:
    """
    Execute a Qiskit quantum circuit using AerSimulator with selected method (statevector or mps).
    
    Measures the execution latency including transpilation and simulation run.
    
    Args:
        circuit (QuantumCircuit): The Qiskit quantum circuit to execute.
        method (str): The simulation method ('statevector' or 'mps'/'matrix_product_state').
        bond_dimension (int): Max bond dimension for MPS.
        
    Returns:
        dict: A dictionary containing execution telemetry:
            - "success" (bool): True if simulation succeeded, False otherwise.
            - "latency" (float): Total execution time in seconds.
            - "error" (str or None): Error message if failed, None if succeeded.
            - "metadata" (dict): Simulator execution metadata if succeeded, empty dict otherwise.
            
    Raises:
        TypeError: If the input circuit is not a Qiskit QuantumCircuit.
    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("Input must be a Qiskit QuantumCircuit instance.")
        
    start_time = time.perf_counter()
    
    try:
        # Initialize AerSimulator based on simulation method
        if method in ('mps', 'matrix_product_state'):
            simulator = AerSimulator(
                method='matrix_product_state',
                matrix_product_state_max_bond_dimension=bond_dimension
            )
        else:
            simulator = AerSimulator(method='statevector')
            
        # Transpile circuit for the simulator backend
        transpiled_circuit = transpile(circuit, simulator)
        
        # Run simulation and fetch result
        job = simulator.run(transpiled_circuit)
        result = job.result()
        
        latency = time.perf_counter() - start_time
        
        # Extract metadata from result
        metadata = {
            "backend_name": result.backend_name,
            "backend_version": result.backend_version,
            "job_id": result.job_id,
            "success": result.success,
            "method": method,
            "bond_dimension": bond_dimension if method in ('mps', 'matrix_product_state') else None
        }
        
        return {
            "success": True,
            "latency": latency,
            "error": None,
            "metadata": metadata
        }
        
    except Exception as e:
        latency = time.perf_counter() - start_time
        return {
            "success": False,
            "latency": latency,
            "error": str(e),
            "metadata": {}
        }
