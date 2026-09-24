import os
import json
import pytest
from qiskit import QuantumCircuit

from cli.main import build_argument_parser
from src.engine.simulator import run_simulation
from src.reporter.json_exporter import export_to_json
from src.reporter.md_exporter import export_to_markdown


def test_cli_parser_native_and_noise_flags():
    """Verify that CLI argument parser correctly parses --use-native-kernels and --noise-profile."""
    parser = build_argument_parser()
    
    # Defaults
    args = parser.parse_args(["--quick"])
    assert args.use_native_kernels is False
    assert args.noise_profile is None
    
    # Explicit flags
    args = parser.parse_args(["--quick", "--use-native-kernels", "--noise-profile", "ibm_brisbane_sample"])
    assert args.use_native_kernels is True
    assert args.noise_profile == "ibm_brisbane_sample"


def test_simulation_with_native_kernels_flag():
    """Verify run_simulation executes with use_native_kernels=True and sets accelerator metadata."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.x(1)
    qc.h(2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    
    res = run_simulation(qc, method="statevector", use_native_kernels=True, runs=1)
    assert res["success"] is True
    assert res["native_kernel_used"] is True
    assert "accelerator_backend" in res
    assert "accelerator_badge" in res
    assert "[Engine:" in res["accelerator_badge"]


def test_simulation_with_physical_noise_profile():
    """Verify run_simulation executes with physical_noise_profile and tracks fidelity."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    
    res = run_simulation(
        qc,
        method="statevector",
        physical_noise_profile="ibm_brisbane_sample",
        runs=1
    )
    assert res["success"] is True
    assert "ibm_brisbane" in res["physical_noise_profile"]
    assert "counts" in res
    assert len(res["counts"]) > 0


def test_simulation_combined_native_and_physical_noise():
    """Verify simultaneous activation of native kernels and physical noise model."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    res = run_simulation(
        qc,
        method="statevector",
        use_native_kernels=True,
        physical_noise_profile="ibm_brisbane_sample",
        runs=2
    )
    assert res["success"] is True
    assert res["native_kernel_used"] is True
    assert "ibm_brisbane" in res["physical_noise_profile"]
    assert res["runs_count"] == 2


def test_json_and_markdown_export_includes_native_and_noise_info(tmp_path):
    """Verify that export_to_json and export_to_markdown output accelerator badges and noise profiles."""
    from cli.runner import run_single_simulation
    
    # Run full runner simulation pipeline to populate complete benchmark result schema
    bench_res = run_single_simulation(
        qubits=2,
        workload_type="shallow",
        depth=2,
        method="statevector",
        device="cpu",
        use_native_kernels=True,
        noise_profile="ibm_brisbane_sample",
        runs=1
    )
    bench_res["workload_label"] = "Shallow Circuit"
    
    dummy_system_metadata = {
        "cpu_name": "Test CPU",
        "cpu_count_physical": 4,
        "cpu_count_logical": 8,
        "total_ram_gb": 16.0,
        "os_name": "TestOS",
        "os_release": "1.0",
        "python_version": "3.13"
    }
    
    # Test JSON export
    json_path = export_to_json([bench_res], dummy_system_metadata, output_dir=str(tmp_path))
    assert os.path.exists(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["native_kernel_used"] is True
    assert "accelerator_backend" in data
    assert "accelerator_badge" in data
    assert "ibm_brisbane" in data["physical_noise_profile"]
    
    # Test Markdown export
    md_file = tmp_path / "report.md"
    md_path = export_to_markdown([bench_res], dummy_system_metadata, output_path=str(md_file))
    assert os.path.exists(md_path)
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    assert "Native Acceleration Engine:" in md_text
    assert "Physical QPU Noise Calibration:" in md_text
    assert "ibm_brisbane" in md_text
