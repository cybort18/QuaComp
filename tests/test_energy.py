import time
import pytest
from src.profiler.energy import estimate_cpu_tdp, EnergyProfiler

def test_estimate_cpu_tdp():
    """Verify TDP estimation heuristics for distinct CPU architectures."""
    assert estimate_cpu_tdp("AMD Ryzen 3 5300U with Radeon Graphics") == 15.0
    assert estimate_cpu_tdp("AMD Ryzen 7 5800H") == 45.0
    assert estimate_cpu_tdp("Apple M3 Max") == 30.0
    assert estimate_cpu_tdp("Intel Core i9-13900K") == 105.0
    assert estimate_cpu_tdp("AMD Ryzen Threadripper 3990X") == 180.0
    assert estimate_cpu_tdp("Unknown Generic CPU") == 35.0

def test_energy_profiler_context():
    """Verify EnergyProfiler context manager records duration, power, and energy."""
    with EnergyProfiler(tdp_watts=25.0, sample_interval=0.01) as ep:
        # Simulate small compute workload
        _ = sum(i * i for i in range(100_000))
        time.sleep(0.05)
        
    metrics = ep.get_metrics(num_gates=50)
    
    assert "energy_backend" in metrics
    assert metrics["duration_seconds"] >= 0.04
    assert metrics["estimated_tdp_watts"] == 25.0
    assert metrics["average_power_watts"] > 0.0
    assert metrics["total_energy_joules"] > 0.0
    assert metrics["energy_per_quantum_op_joules"] > 0.0
    assert metrics["eqo_microjoules"] > 0.0
    
    # Verify EQO is energy divided by gates
    expected_eqo = metrics["total_energy_joules"] / 50
    assert pytest.approx(metrics["energy_per_quantum_op_joules"], rel=1e-3) == expected_eqo

def test_energy_profiler_zero_gates_safe():
    """Verify handling of 0 gates edge case without ZeroDivisionError."""
    with EnergyProfiler(tdp_watts=15.0) as ep:
        time.sleep(0.01)
    metrics = ep.get_metrics(num_gates=0)
    assert metrics["energy_per_quantum_op_joules"] >= 0.0
