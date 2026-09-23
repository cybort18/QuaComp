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

def test_export_to_json_all_failed_runs(tmp_path):
    failed_results = [
        {
            "qubits": 35,
            "success": False,
            "latency": 0.0,
            "mean_latency": 0.0,
            "median_latency": 0.0,
            "std_latency": 0.0,
            "runs_count": 0,
            "latencies": [],
            "gates": 0,
            "cpu_usage": 0.0,
            "ram_status": "UNSAFE",
            "error": "CRITICAL: Out of Memory"
        }
    ]
    metadata = {"cpu_name": "Test CPU", "total_ram_gb": 8.0}
    out_dir = tmp_path / "results"
    
    file_path = export_to_json(failed_results, metadata, output_dir=str(out_dir))
    assert os.path.exists(file_path)
    with open(file_path, "r") as f:
        data = json.load(f)
    assert data["final_score"] == 0.0
    assert data["entanglement_metrics"] == {}
    assert data["max_qubits_simulated"] == 0

def test_export_to_markdown_all_failed_runs(tmp_path):
    failed_results = [
        {
            "qubits": 35,
            "success": False,
            "latency": 0.0,
            "mean_latency": 0.0,
            "std_latency": 0.0,
            "runs_count": 0,
            "gates": 0,
            "cpu_usage": 0.0,
            "ram_status": "UNSAFE",
            "error": "CRITICAL: Out of Memory"
        }
    ]
    metadata = {"cpu_name": "Test CPU", "total_ram_gb": 8.0}
    out_path = tmp_path / "results" / "report.md"
    
    file_path = export_to_markdown(failed_results, metadata, output_path=str(out_path))
    assert os.path.exists(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "# QuaComp Benchmark Report" in content
    assert "UNSAFE" in content
    assert "FAILED" in content


def test_export_to_markdown_energy_telemetry_rapl(tmp_path):
    """Verify Markdown report correctly displays [Sensor: RAPL] for RAPL backend."""
    results = [
        {
            "qubits": 12,
            "success": True,
            "latency": 0.42,
            "mean_latency": 0.42,
            "std_latency": 0.02,
            "runs_count": 2,
            "gates": 100,
            "cpu_usage": 55.0,
            "ram_status": "SAFE",
            "workload_label": "QFT",
            "energy_metrics": {
                "energy_backend": "Linux RAPL Hardware Interface",
                "sensor_source": "[Sensor: RAPL]",
                "duration_seconds": 0.42,
                "average_power_watts": 28.5,
                "total_energy_joules": 11.97,
                "eqo_microjoules": 119.7
            }
        }
    ]
    metadata = {"cpu_name": "Intel Xeon", "total_ram_gb": 64.0}
    out_path = tmp_path / "report_rapl.md"
    file_path = export_to_markdown(results, metadata, output_path=str(out_path))
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "Hardware Power & Energy Telemetry (EQO)" in content
    assert "[Sensor: RAPL]" in content
    assert "Linux RAPL Hardware Interface" in content
    assert "11.9700 J" in content
    assert "28.50 W" in content
    assert "119.70 µJ" in content


def test_export_to_markdown_energy_telemetry_tdp_estimate(tmp_path):
    """Verify Markdown report correctly displays [Sensor: TDP Estimate] for TDP models."""
    results = [
        {
            "qubits": 14,
            "success": True,
            "latency": 0.85,
            "mean_latency": 0.85,
            "std_latency": 0.03,
            "runs_count": 2,
            "gates": 250,
            "cpu_usage": 70.0,
            "ram_status": "SAFE",
            "workload_label": "GHZ",
            "energy_metrics": {
                "energy_backend": "Windows Dynamic TDP Model",
                "duration_seconds": 0.85,
                "average_power_watts": 35.0,
                "total_energy_joules": 29.75,
                "eqo_microjoules": 119.0
            }
        }
    ]
    metadata = {"cpu_name": "AMD Ryzen 7", "total_ram_gb": 32.0}
    out_path = tmp_path / "report_tdp.md"
    file_path = export_to_markdown(results, metadata, output_path=str(out_path))
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "Hardware Power & Energy Telemetry (EQO)" in content
    assert "[Sensor: TDP Estimate]" in content
    assert "Windows Dynamic TDP Model" in content
    assert "29.7500 J" in content
    assert "35.00 W" in content


def test_export_to_json_energy_telemetry(tmp_path):
    """Verify JSON export contains top-level energy_metrics."""
    results = [
        {
            "qubits": 10,
            "success": True,
            "latency": 0.1,
            "mean_latency": 0.1,
            "gates": 50,
            "cpu_usage": 20.0,
            "ram_status": "SAFE",
            "energy_metrics": {
                "energy_backend": "Linux RAPL Hardware Interface",
                "sensor_source": "[Sensor: RAPL]",
                "total_energy_joules": 2.5
            }
        }
    ]
    metadata = {"cpu_name": "Test CPU", "total_ram_gb": 16.0}
    out_dir = tmp_path / "json_out"
    file_path = export_to_json(results, metadata, output_dir=str(out_dir))
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "energy_metrics" in data
    assert data["energy_metrics"]["sensor_source"] == "[Sensor: RAPL]"
    assert data["energy_metrics"]["total_energy_joules"] == 2.5

