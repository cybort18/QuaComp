# QuaComp Benchmark Report
Generated on: `2026-08-10 21:02:55`

---

## Benchmark Summary
> **QuaComp Composite Score:** `10,486,120.47` *(Project-Specific Heuristic Score)*
> - **Capacity Metric (2^n):** `1,048,576`
> - **Throughput Metric:** `360.47 gates/sec`
> **Performance Category:** `High-Performance`
> **Simulation Method:** `Statevector`
> **Statistical Repeatability:** `3 runs` (Mean Latency: `0.6103s`, Std Dev: `0.1028s`)
> **Max Qubits Simulated:** `20 qubits` (using `220` gates)

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
| 10 | statevector | none | QFT | 60 | 0.3171 ± 0.0853s | 100.00% | 82.8% | SAFE | SUCCESS |
| 15 | statevector | none | QFT | 127 | 0.3171 ± 0.0814s | 100.00% | 70.7% | SAFE | SUCCESS |
| 20 | statevector | none | QFT | 220 | 0.6103 ± 0.1028s | 100.00% | 43.9% | SAFE | SUCCESS |

---

## Telemetry Visualizations
![Qubit Vs Latency](qubit_vs_latency.png)

![Qubit Vs Ram](qubit_vs_ram.png)


---

## GitHub Ready
This report is formatted and ready to be posted directly into GitHub Issues, pull request reviews, or Discussions.