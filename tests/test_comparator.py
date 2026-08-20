import os
import json
import pytest
from src.comparator.differ import resolve_target_profile, load_benchmark_json, compare_benchmarks
from src.comparator.reporter import export_comparison_to_json, export_comparison_to_markdown
from src.reporter.charts import generate_comparison_charts

@pytest.fixture
def mock_benchmark_pair():
    base = {
        "timestamp": "2026-08-10T12:00:00",
        "final_composite_score": 10485800.0,
        "performance_category": "High-Performance",
        "max_qubits_simulated": 20,
        "scoring_breakdown": {
            "capacity_metric": 1048576.0,
            "throughput_metric": 240.0
        },
        "system_metadata": {
            "cpu_name": "Test Base CPU (4-core)",
            "total_ram_gb": 16.0,
            "os_name": "Windows",
            "os_release": "11",
            "python_version": "3.11.0"
        },
        "results": [
            {
                "qubits": 10,
                "workload_label": "QFT",
                "method": "statevector",
                "gates": 60,
                "latency": 0.20,
                "mean_latency": 0.20,
                "cpu_usage": 35.0,
                "fidelity": 100.0,
                "success": True
            },
            {
                "qubits": 20,
                "workload_label": "QFT",
                "method": "statevector",
                "gates": 220,
                "latency": 0.90,
                "mean_latency": 0.90,
                "cpu_usage": 50.0,
                "fidelity": 100.0,
                "success": True
            }
        ]
    }
    
    target = {
        "timestamp": "2026-08-12T10:15:00",
        "final_composite_score": 335544830.0,
        "performance_category": "Extreme Workstation",
        "max_qubits_simulated": 25,
        "scoring_breakdown": {
            "capacity_metric": 33554432.0,
            "throughput_metric": 510.0
        },
        "system_metadata": {
            "cpu_name": "Test Target CPU (8-core)",
            "total_ram_gb": 32.0,
            "os_name": "Darwin",
            "os_release": "23.4.0",
            "python_version": "3.12.0"
        },
        "results": [
            {
                "qubits": 10,
                "workload_label": "QFT",
                "method": "statevector",
                "gates": 60,
                "latency": 0.08,
                "mean_latency": 0.08,
                "cpu_usage": 80.0,
                "fidelity": 100.0,
                "success": True
            },
            {
                "qubits": 20,
                "workload_label": "QFT",
                "method": "statevector",
                "gates": 220,
                "latency": 0.24,
                "mean_latency": 0.24,
                "cpu_usage": 85.0,
                "fidelity": 100.0,
                "success": True
            },
            {
                "qubits": 25,
                "workload_label": "QFT",
                "method": "statevector",
                "gates": 325,
                "latency": 0.64,
                "mean_latency": 0.64,
                "cpu_usage": 90.0,
                "fidelity": 100.0,
                "success": True
            }
        ]
    }
    return base, target

def test_resolve_target_profile():
    # Test preset aliases
    path_m3 = resolve_target_profile("apple_m3")
    assert os.path.exists(path_m3)
    assert path_m3.endswith("example_apple_m3.json")
    
    path_5300u = resolve_target_profile("ryzen3_5300u")
    assert os.path.exists(path_5300u)
    assert path_5300u.endswith("example_ryzen3_5300u.json")
    
    path_5800h = resolve_target_profile("5800h")
    assert os.path.exists(path_5800h)
    assert path_5800h.endswith("example_ryzen7_5800h.json")
    
    # Test errors
    with pytest.raises(FileNotFoundError):
        resolve_target_profile("non_existent_processor_xyz")
    with pytest.raises(TypeError):
        resolve_target_profile(12345)  # type: ignore

def test_load_benchmark_json():
    # Load actual sample file
    path = resolve_target_profile("apple_m3")
    data = load_benchmark_json(path)
    assert isinstance(data, dict)
    assert "final_score" in data or "final_composite_score" in data
    assert "results" in data
    assert "system_metadata" in data
    
    # Test errors
    with pytest.raises(FileNotFoundError):
        load_benchmark_json("results/non_existent_file.json")
    with pytest.raises(TypeError):
        load_benchmark_json(None)  # type: ignore

def test_load_benchmark_json_invalid_schema(tmp_path):
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, "w") as f:
        json.dump({"foo": "bar"}, f)
        
    with pytest.raises(ValueError) as exc:
        load_benchmark_json(str(invalid_file))
    assert "Missing required key" in str(exc.value)

def test_compare_benchmarks(mock_benchmark_pair):
    base, target = mock_benchmark_pair
    diff = compare_benchmarks(base, target, base_label="Base Box", target_label="Target Box")
    
    assert diff["base_label"] == "Base Box"
    assert diff["target_label"] == "Target Box"
    
    summary = diff["score_summary"]
    # Score ratio: 335544830.0 / 10485800.0 = ~32.0x
    assert summary["score_ratio"] > 30.0
    assert summary["score_delta_pct"] > 0
    
    # Capacity gap: 25 - 20 = 5 qubits (2^5 = 32x state space)
    assert summary["qubit_gap"] == 5
    assert summary["capacity_ratio"] == 32.0
    
    # Throughput speedup: 510.0 / 240.0 = ~2.125x
    assert abs(summary["throughput_speedup"] - 2.125) < 1e-3
    assert summary["throughput_delta_pct"] > 100.0
    
    # Matched qubits (10 and 20 qubits)
    matched = diff["matched_qubits"]
    assert len(matched) == 2
    
    # 10 qubits: Base 0.20s vs Target 0.08s -> 2.5x speedup
    q10 = matched[0]
    assert q10["qubits"] == 10
    assert abs(q10["speedup_factor"] - 2.5) < 1e-3
    assert q10["latency_delta_pct"] == -60.0  # -60% latency
    assert q10["is_faster"] is True
    
    # 20 qubits: Base 0.90s vs Target 0.24s -> 3.75x speedup
    q20 = matched[1]
    assert q20["qubits"] == 20
    assert abs(q20["speedup_factor"] - 3.75) < 1e-3
    assert q20["is_faster"] is True
    
    # Academic verdict text
    assert "faster simulation throughput" in diff["verdict"]
    assert "+5 qubit capacity advantage" in diff["verdict"]

def test_compare_benchmarks_type_error():
    with pytest.raises(TypeError):
        compare_benchmarks("not a dict", {})  # type: ignore
    with pytest.raises(TypeError):
        compare_benchmarks({}, "not a dict")  # type: ignore

def test_export_comparison_to_json_and_markdown(tmp_path, mock_benchmark_pair):
    base, target = mock_benchmark_pair
    diff = compare_benchmarks(base, target)
    out_dir = tmp_path / "results"
    
    # 1. Export JSON
    json_path = export_comparison_to_json(diff, output_dir=str(out_dir))
    assert os.path.exists(json_path)
    assert json_path.endswith(".json")
    
    with open(json_path, "r") as f:
        loaded = json.load(f)
    assert loaded["report_type"] == "QuaComp Relative Benchmark Comparison"
    assert loaded["score_summary"]["qubit_gap"] == 5
    
    # 2. Export Markdown
    md_path = export_comparison_to_markdown(diff, output_dir=str(out_dir))
    assert os.path.exists(md_path)
    assert md_path.endswith(".md")
    
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    assert "QuaComp Relative Benchmark Comparison Report" in md_text
    assert "Executive Summary & Verdict" in md_text
    assert "Per-Qubit Latency & Execution Breakdown" in md_text

def test_generate_comparison_charts(tmp_path, mock_benchmark_pair):
    base, target = mock_benchmark_pair
    diff = compare_benchmarks(base, target)
    out_dir = tmp_path / "results"
    
    charts = generate_comparison_charts(diff, output_dir=str(out_dir))
    assert isinstance(charts, list)
    assert len(charts) == 2
    
    for c in charts:
        assert os.path.exists(c)
        assert c.endswith(".png")
        assert os.path.getsize(c) > 0
        
    # Test type error
    with pytest.raises(TypeError):
        generate_comparison_charts("invalid")  # type: ignore
