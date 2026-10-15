# QuaComp Benchmark Report
Generated on: `2026-10-15 17:35:38`

---

## Benchmark Summary
### Hardware Benchmark Overview (QuaComp Synthetic Index)
| Benchmark Metric | Measured Value | Performance Tier / Status |
| :--- | :---: | :---: |
| **QuaComp Synthetic Index (QSI)** | `5,685.80` pts | **Mid-Range** |
| **Capacity Metric (2^n)** | `1,048,576` | Max `20` qubits simulated |
| **Throughput Metric** | `1,533.91 gates/s` | Dynamic sensitivity scale |
| **Execution Latency (Mean)** | `0.1434s` | ± `0.0032s` (3 runs) |
| **Simulation Method** | `Statevector` | Acceleration: Standard CPU |

### Formal Academic Metrics (Quantum HPC Standards)
| Academic Metric | Measured Value | Standard / Significance |
| :--- | :---: | :--- |
| **Normalized Gate Throughput** | `1.5339 kGates/s` | `220` total gates processed |
| **Effective Memory Bandwidth** | `51.4694 GB/s` | Arithmetic Intensity: `0.4375 FLOPs/Byte` (Memory-bound) |
| **Energy-Delay Product (EDP)** | `1.859458 J · s` | Energy-delay efficiency (Lower indicates superior efficiency) |
| **Quantum Volume Certification** | `Not Evaluated` | Heavy Output Probability: `N/A` (Threshold: > 0.6667) |
| **Quantum State Fidelity** | `100.00%` | Statevector fidelity against ideal analytical target |

---

> **Hardware Energy Telemetry:** `[Sensor: TDP Estimate]` `12.9647 J` (Avg Power: `6.48 W`, EQO: `58930.54 µJ/gate | `Windows Dynamic TDP Model`)

---

## System Metadata & Telemetry
| Parameter | System Value |
| :--- | :--- |
| **CPU Name** | AMD Ryzen 3 5300U with Radeon Graphics |
| **Total Physical RAM** | 11.33 GB |
| **Operating System** | Windows (11) |
| **Python Version** | 3.13.3 |

---

## Detailed Simulation Runs
| Qubits | Method | Noise Profile | Workload | Total Gates | Latency (Mean ± Std Dev) | Fidelity % | Avg CPU % | RAM Status | Success |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 10 | statevector | none | QFT | 60 | 0.0123 ± 0.0091s | 100.00% | 70.0% | SAFE | SUCCESS |
| 15 | statevector | none | QFT | 127 | 0.0170 ± 0.0020s | 100.00% | 64.7% | SAFE | SUCCESS |
| 20 | statevector | none | QFT | 220 | 0.1434 ± 0.0032s | 100.00% | 68.6% | SAFE | SUCCESS |

---

## Hardware Power & Energy Telemetry (EQO)
| Qubits | Workload | Avg Power (W) | Total Energy (J) | EQO (µJ/Gate) | Telemetry Backend & Source |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 10 | QFT | 8.31 W | 15.8416 J | 264027.34 µJ | `[Sensor: TDP Estimate]` Windows Dynamic TDP Model |
| 15 | QFT | 6.20 W | 9.8209 J | 77330.10 µJ | `[Sensor: TDP Estimate]` Windows Dynamic TDP Model |
| 20 | QFT | 6.48 W | 12.9647 J | 58930.54 µJ | `[Sensor: TDP Estimate]` Windows Dynamic TDP Model |

---

## Telemetry Visualizations
![Qubit Vs Latency](qubit_vs_latency.png)

![Qubit Vs Ram](qubit_vs_ram.png)


---

## GitHub Ready
This report is formatted and ready to be posted directly into GitHub Issues, pull request reviews, or Discussions.