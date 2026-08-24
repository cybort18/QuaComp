import os
import pytest
from src.reporter.charts import generate_benchmark_charts, generate_entanglement_charts

@pytest.fixture
def mock_results_data():
    results = [
        {
            "qubits": 10,
            "method": "statevector",
            "noise_level": "none",
            "workload_label": "QFT",
            "gates": 60,
            "latency": 0.25,
            "mean_latency": 0.25,
            "median_latency": 0.25,
            "std_latency": 0.01,
            "fidelity": 100.0,
            "overhead_ratio": 0.0,
            "cpu_usage": 25.0,
            "ram_status": "SAFE",
            "success": True,
            "ram_savings": {},
            "entanglement_metrics": {
                "num_qubits": 10,
                "von_neumann_entropy": 0.0,
                "max_possible_entropy": 5.0,
                "schmidt_rank": 1,
                "entanglement_regime": "Product State",
                "mps_hardness": "Trivial (chi=1)"
            }
        },
        {
            "qubits": 15,
            "method": "statevector",
            "noise_level": "medium",
            "workload_label": "DEEP",
            "gates": 120,
            "latency": 0.85,
            "mean_latency": 0.85,
            "median_latency": 0.84,
            "std_latency": 0.02,
            "fidelity": 45.0,
            "overhead_ratio": 50.0,
            "cpu_usage": 40.0,
            "ram_status": "SAFE",
            "success": True,
            "ram_savings": {},
            "entanglement_metrics": {
                "num_qubits": 15,
                "von_neumann_entropy": 4.85,
                "max_possible_entropy": 7.0,
                "schmidt_rank": 128,
                "entanglement_regime": "Moderate Entanglement",
                "mps_hardness": "Challenging (Moderate chi <= 64)"
            }
        },
        {
            "qubits": 30,
            "method": "mps",
            "bond_dimension": 64,
            "noise_level": "none",
            "workload_label": "QFT",
            "gates": 480,
            "latency": 0.35,
            "mean_latency": 0.35,
            "median_latency": 0.35,
            "std_latency": 0.01,
            "fidelity": 100.0,
            "overhead_ratio": 0.0,
            "cpu_usage": 30.0,
            "ram_status": "SAFE",
            "success": True,
            "ram_savings": {"savings_bytes": 1000000, "savings_percent": 99.9}
        }
    ]
    metadata = {
        "cpu_name": "Test CPU",
        "total_ram_gb": 16.0,
        "os_name": "TestOS",
        "os_release": "1.0",
        "python_version": "3.10"
    }
    return results, metadata

def test_generate_benchmark_charts_success(tmp_path, mock_results_data):
    results, metadata = mock_results_data
    out_dir = tmp_path / "results"
    
    chart_paths = generate_benchmark_charts(results, metadata, output_dir=str(out_dir))
    
    assert isinstance(chart_paths, list)
    assert len(chart_paths) >= 2
    
    for path in chart_paths:
        assert os.path.exists(path)
        assert path.endswith(".png")
        assert os.path.getsize(path) > 0

def test_generate_entanglement_charts_success(tmp_path, mock_results_data):
    results, _ = mock_results_data
    out_dir = tmp_path / "results"
    
    chart_paths = generate_entanglement_charts(results, output_dir=str(out_dir))
    assert len(chart_paths) == 1
    assert chart_paths[0].endswith("entanglement_entropy.png")
    assert os.path.exists(chart_paths[0])
    assert os.path.getsize(chart_paths[0]) > 0

def test_generate_benchmark_charts_empty_results(tmp_path, mock_results_data):
    _, metadata = mock_results_data
    out_dir = tmp_path / "results"
    
    chart_paths = generate_benchmark_charts([], metadata, output_dir=str(out_dir))
    assert chart_paths == []

def test_generate_benchmark_charts_type_errors(mock_results_data):
    results, metadata = mock_results_data
    
    with pytest.raises(TypeError):
        generate_benchmark_charts("invalid", metadata)  # type: ignore
    with pytest.raises(TypeError):
        generate_benchmark_charts(results, "invalid")  # type: ignore
    with pytest.raises(TypeError):
        generate_entanglement_charts("invalid")  # type: ignore

