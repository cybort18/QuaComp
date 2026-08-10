import os
import pytest
from src.reporter.charts import generate_benchmark_charts

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
            "ram_savings": {}
        },
        {
            "qubits": 15,
            "method": "statevector",
            "noise_level": "medium",
            "workload_label": "QFT",
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
            "ram_savings": {}
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
