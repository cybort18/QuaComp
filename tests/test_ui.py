from rich.console import Console
from cli.ui import print_system_info, display_results, display_help_notice

def test_display_results_with_energy_telemetry():
    """Verify display_results renders correctly with RAPL and TDP energy telemetry."""
    console = Console(record=True, width=120)
    
    results = [
        {
            "qubits": 10,
            "success": True,
            "latency": 0.25,
            "mean_latency": 0.25,
            "std_latency": 0.01,
            "runs_count": 2,
            "latencies": [0.24, 0.26],
            "gates": 80,
            "cpu_usage": 45.0,
            "fidelity": 99.8,
            "ram_status": "SAFE",
            "workload_label": "QFT",
            "method": "statevector",
            "device": "CPU",
            "noise_level": "none",
            "energy_metrics": {
                "energy_backend": "Linux RAPL Hardware Interface",
                "average_power_watts": 25.0,
                "total_energy_joules": 6.25,
                "eqo_microjoules": 78.125
            }
        },
        {
            "qubits": 12,
            "success": True,
            "latency": 0.50,
            "mean_latency": 0.50,
            "std_latency": 0.02,
            "runs_count": 2,
            "latencies": [0.48, 0.52],
            "gates": 120,
            "cpu_usage": 60.0,
            "fidelity": 98.5,
            "ram_status": "SAFE",
            "workload_label": "GHZ",
            "method": "mps",
            "bond_dimension": 32,
            "device": "CPU",
            "noise_level": "none",
            "energy_metrics": {
                "energy_backend": "Windows Dynamic TDP Model",
                "average_power_watts": 35.0,
                "total_energy_joules": 17.5,
                "eqo_microjoules": 145.83
            }
        }
    ]
    
    display_results(results, console=console)
    output = console.export_text()
    
    # Verify tables rendered
    assert "Simulation Benchmark Results" in output
    assert "Hardware Power & Energy Telemetry (EQO)" in output
    assert "[Sensor: RAPL]" in output
    assert "[Sensor: TDP Estimate]" in output
    assert "Linux RAPL Hardware Interface" in output
    assert "Windows Dynamic TDP Model" in output
    assert "Final Benchmark Report" in output

def test_display_results_all_failed():
    """Verify display_results handles case where all runs failed."""
    console = Console(record=True, width=120)
    results = [
        {
            "qubits": 32,
            "success": False,
            "latency": 0.0,
            "gates": 0,
            "cpu_usage": 0.0,
            "fidelity": 0.0,
            "ram_status": "UNSAFE",
            "workload_label": "QFT",
            "method": "statevector"
        }
    ]
    display_results(results, console=console)
    output = console.export_text()
    assert "No simulations completed successfully" in output

def test_print_system_info_and_help():
    """Verify print_system_info and display_help_notice render cleanly."""
    console = Console(record=True, width=120)
    print_system_info(console=console)
    output_info = console.export_text()
    assert "System Metadata & Telemetry" in output_info
    
    console_help = Console(record=True, width=120)
    display_help_notice(console=console_help)
    output_help = console_help.export_text()
    assert "Quantum Computer Simulation Benchmark" in output_help
    assert "quacomp --quick" in output_help
