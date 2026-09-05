# QuaComp Benchmark Report
Generated on: `2026-09-05 14:53:08`

---

## Benchmark Summary
> **QuaComp Composite Score:** `10,487,219.28` *(Project-Specific Heuristic Score)*
> - **Capacity Metric (2^n):** `1,048,576`
> - **Throughput Metric:** `1,459.28 gates/sec`
> **Performance Category:** `High-Performance`
> **Simulation Method:** `Statevector`
> **Statistical Repeatability:** `1 runs` (Mean Latency: `0.1508s`, Std Dev: `0.0000s`)
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
| 10 | statevector | none | QFT | 60 | 0.0211 ± 0.0000s | 100.00% | 47.9% | SAFE | SUCCESS |
| 15 | statevector | none | QFT | 127 | 0.0184 ± 0.0000s | 100.00% | 80.0% | SAFE | SUCCESS |
| 20 | statevector | none | QFT | 220 | 0.1508 ± 0.0000s | 100.00% | 45.8% | SAFE | SUCCESS |

---

## Telemetry Visualizations
![Qubit Vs Latency](qubit_vs_latency.png)

![Qubit Vs Ram](qubit_vs_ram.png)

![Entanglement Entropy](entanglement_entropy.png)


---

## GitHub Ready
This report is formatted and ready to be posted directly into GitHub Issues, pull request reviews, or Discussions.