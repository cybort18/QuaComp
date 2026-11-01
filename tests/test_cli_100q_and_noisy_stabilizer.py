"""
Unit and end-to-end integration tests for:
1. 100-Qubit CLI execution (--stabilizer, --sim-method auto, and --noise-level low) with Exit Code 0.
2. Noisy Stabilizer Trajectories (Monte Carlo Pauli jumps and readout bit-flip errors on 50-100+ qubits).
3. Automatic in-place Observables & Entanglement reporting in simulate_stabilizer, simulate_mps, and cli/runner.py.
"""
import glob
import os
import subprocess
import sys
import pytest
from qiskit import QuantumCircuit

from src.engine.stabilizer import simulate_stabilizer
from src.engine.mps import simulate_mps
from src.engine.parser import load_qasm
from cli.runner import run_single_simulation


@pytest.fixture
def preserve_results_dir():
    """Preserve existing files in results/ around subprocess CLI invocations."""
    results_dir = os.path.abspath("results")
    before_files = set(glob.glob(os.path.join(results_dir, "benchmark_*.json")))
    report_path = os.path.join(results_dir, "report.md")
    report_backup = None
    if os.path.exists(report_path):
        with open(report_path, "rb") as f:
            report_backup = f.read()

    yield

    after_files = set(glob.glob(os.path.join(results_dir, "benchmark_*.json")))
    for new_file in after_files - before_files:
        try:
            os.remove(new_file)
        except OSError:
            pass
    if report_backup is not None:
        with open(report_path, "wb") as f:
            f.write(report_backup)


class TestCLI100QubitSubprocess:
    """Verify 100-qubit OpenQASM CLI execution completes cleanly with Exit Code 0."""

    def test_cli_100q_stabilizer_exit_code_zero(self, preserve_results_dir):
        """Running --qasm benchmarks/ghz_100qubit.qasm --stabilizer --runs 1 must return exit code 0."""
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "cli.main",
                "--qasm",
                "benchmarks/ghz_100qubit.qasm",
                "--stabilizer",
                "--runs",
                "1",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert res.returncode == 0, f"CLI failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
        assert "PASS" in res.stdout
        assert "100" in res.stdout

    def test_cli_100q_auto_solver_exit_code_zero(self, preserve_results_dir):
        """Running --qasm benchmarks/ghz_100qubit.qasm --sim-method auto --runs 1 must return exit code 0."""
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "cli.main",
                "--qasm",
                "benchmarks/ghz_100qubit.qasm",
                "--sim-method",
                "auto",
                "--runs",
                "1",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert res.returncode == 0, f"CLI failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
        assert "PASS" in res.stdout

    def test_cli_100q_noisy_stabilizer_exit_code_zero(self, tmp_path):
        """Running 100-qubit noisy stabilizer via CLI (--noise-level low) must return exit code 0."""
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "cli.main",
                "--qasm",
                "benchmarks/ghz_100qubit.qasm",
                "--stabilizer",
                "--noise-level",
                "low",
                "--shots",
                "256",
                "--runs",
                "1",
                "--output-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert res.returncode == 0, f"Noisy 100Q CLI failed:\n{res.stderr}\n{res.stdout}"
        assert "PASS" in res.stdout


class TestNoisyStabilizerTrajectories:
    """Verify Monte Carlo Pauli jump trajectories on 50-100+ qubit stabilizer circuits."""

    def test_50_qubit_ghz_noiseless_vs_medium_noise(self):
        """Compare 50-qubit GHZ on noise_level='none' vs noise_level='medium'."""
        n = 50
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)

        shots = 512
        ideal_res = simulate_stabilizer(qc, shots=shots, seed=42, noise_level="none")
        noisy_res = simulate_stabilizer(qc, shots=shots, seed=42, noise_level="medium")

        all_zeros = "0" * n
        all_ones = "1" * n

        # Ideal GHZ state has 100% population in |0...0> and |1...1>
        ideal_target_pop = ideal_res["counts"].get(all_zeros, 0) + ideal_res["counts"].get(all_ones, 0)
        assert ideal_target_pop == shots
        assert ideal_res["noisy_fidelity_estimate"] == pytest.approx(1.0)
        assert ideal_res["pauli_jump_count"] == 0

        # Medium noise introduces stochastic Pauli jumps and readout bit-flips
        noisy_target_pop = noisy_res["counts"].get(all_zeros, 0) + noisy_res["counts"].get(all_ones, 0)
        assert noisy_target_pop < shots
        assert len(noisy_res["counts"]) > 2
        assert noisy_res["pauli_jump_count"] > 0
        assert 0.0 < noisy_res["noisy_fidelity_estimate"] < 1.0
        assert noisy_res["tableau_memory_bytes"] == (2 * n + 1) * (2 * n + 1)


class TestObservablesAndEntanglementIntegration:
    """Verify simulate_stabilizer, simulate_mps, and run_single_simulation report exact observables."""

    def test_ghz_observables_in_simulate_stabilizer_and_mps(self):
        """Bell/GHZ circuits must report S(rho_A)=1.0, <Z_0 Z_{n-1}>=1.0, <H_Ising>=-1.0."""
        n = 12
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)

        stab_res = simulate_stabilizer(qc, shots=128, seed=123)
        mps_res = simulate_mps(qc, bond_dimension=8, shots=128, seed=123)

        for res in (stab_res, mps_res):
            assert res["bipartite_entropy_ebits"] == pytest.approx(1.0, abs=1e-5)
            assert res["bipartite_entanglement_entropy_ebits"] == pytest.approx(1.0, abs=1e-5)
            assert res["end_to_end_zz_expectation"] == pytest.approx(1.0, abs=1e-5)
            assert res["zz_correlation_expectation"] == pytest.approx(1.0, abs=1e-5)
            assert res["ising_energy_expectation"] == pytest.approx(-1.0, abs=1e-5)

    def test_runner_100_qubit_ghz_observables_and_entropy(self):
        """run_single_simulation on 100-qubit GHZ reports exact entanglement and Ising observables."""
        parsed = load_qasm("benchmarks/ghz_100qubit.qasm")
        run_res = run_single_simulation(
            qubits=100,
            workload_type="shallow",
            depth=1,
            method="auto",
            runs=1,
            shots=128,
            qasm_circuit=parsed,
        )
        assert run_res["success"] is True
        assert run_res["method"] == "stabilizer"
        assert run_res["bipartite_entropy_ebits"] == pytest.approx(1.0, abs=1e-6)
        assert run_res["bipartite_entanglement_entropy_ebits"] == pytest.approx(1.0, abs=1e-6)
        assert run_res["end_to_end_zz_expectation"] == pytest.approx(1.0, abs=1e-6)
        assert run_res["zz_correlation_expectation"] == pytest.approx(1.0, abs=1e-6)
        assert run_res["ising_energy_expectation"] == pytest.approx(-1.0, abs=1e-6)

