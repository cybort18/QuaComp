import os
import json
import pytest
from src.reporter.json_exporter import export_to_json
from src.reporter.md_exporter import export_to_markdown

@pytest.fixture
def sample_data():
    results = [
        {
            "qubits": 10,
            "success": True,
            "latency": 0.25,
            "mean_latency": 0.25,
            "median_latency": 0.25,
            "std_latency": 0.01,
            "runs_count": 3,
            "latencies": [0.24, 0.25, 0.26],
            "gates": 60,
            "cpu_usage": 30.0,
            "ram_status": "SAFE",
            "error": None,
            "workload_label": "QFT"
        },
        {
            "qubits": 15,
            "success": False,
            "latency": 0.0,
            "mean_latency": 0.0,
            "median_latency": 0.0,
            "std_latency": 0.0,
            "runs_count": 0,
            "latencies": [],
            "gates": 127,
            "cpu_usage": 0.0,
            "ram_status": "UNSAFE",
            "error": "Memory limit exceeded"
        }
    ]
    system_metadata = {
        "cpu_name": "Test CPU",
        "total_ram_gb": 16.0,
        "os_name": "TestOS",
        "os_release": "1.0",
        "python_version": "3.10"
    }
    return results, system_metadata

def test_export_to_json(tmp_path, sample_data):
    results, system_metadata = sample_data
    output_dir = tmp_path / "results"
    
    # Run exporter
    file_path = export_to_json(results, system_metadata, output_dir=str(output_dir))
    
    assert os.path.exists(file_path)
    assert file_path.endswith(".json")
    
    # Read and verify content
    with open(file_path, "r") as f:
        data = json.load(f)
        
    assert "timestamp" in data
    assert data["final_score"] > 0.0
    assert data["final_composite_score"] > 0.0
    assert "scoring_breakdown" in data
    assert "capacity_metric" in data["scoring_breakdown"]
    assert "throughput_metric" in data["scoring_breakdown"]
    assert "statistical_summary" in data
    assert data["performance_category"] == "Entry-Level"
    assert data["max_qubits_simulated"] == 10
    assert data["system_metadata"]["cpu_name"] == "Test CPU"
    assert len(data["results"]) == 2
    assert data["results"][0]["qubits"] == 10
    assert data["results"][1]["error"] == "Memory limit exceeded"

def test_export_to_json_type_errors():
    with pytest.raises(TypeError):
        export_to_json("not a list", {})  # type: ignore
    with pytest.raises(TypeError):
        export_to_json([], "not a dict")  # type: ignore

def test_export_to_markdown(tmp_path, sample_data):
    results, system_metadata = sample_data
    output_path = tmp_path / "results" / "report.md"
    
    # Run exporter
    file_path = export_to_markdown(results, system_metadata, output_path=str(output_path))
    
    assert os.path.exists(file_path)
    assert file_path.endswith("report.md")
    
    # Read and verify content
    with open(file_path, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    assert "# QuaComp Benchmark Report" in md_text
    assert "Test CPU" in md_text
    assert "16.00 GB" in md_text
    assert "UNSAFE" in md_text
    assert "Entry-Level" in md_text
    assert "Capacity Metric" in md_text
    assert "Throughput Metric" in md_text

def test_export_to_markdown_type_errors():
    with pytest.raises(TypeError):
        export_to_markdown("not a list", {})  # type: ignore
    with pytest.raises(TypeError):
        export_to_markdown([], "not a dict")  # type: ignore
