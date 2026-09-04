# QuaComp Benchmark Report
Generated on: `2026-09-04 22:30:28`

---

## Benchmark Summary
> **QuaComp Composite Score:** `10,486,714.96` *(Project-Specific Heuristic Score)*
> - **Capacity Metric (2^n):** `1,048,576`
> - **Throughput Metric:** `954.96 gates/sec`
> **Performance Category:** `High-Performance`
> **Simulation Method:** `Statevector`
> **Entanglement Entropy:** `S_vN = 0.0000 bits` (Schmidt Rank: `1` | `Product State`)
> **MPS Simulation Complexity:** `Trivial (chi=1)`
> **Statistical Repeatability:** `3 runs` (Mean Latency: `0.2304s`, Std Dev: `0.0252s`)
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
| 10 | statevector | none | QFT | 60 | 0.0175 ± 0.0101s | 100.00% | 83.3% | SAFE | SUCCESS |
| 15 | statevector | none | QFT | 127 | 0.0210 ± 0.0017s | 100.00% | 88.0% | SAFE | SUCCESS |
| 20 | statevector | none | QFT | 220 | 0.2304 ± 0.0252s | 100.00% | 87.5% | SAFE | SUCCESS |

---

## Entanglement Entropy & Simulation Hardness Analysis
| Qubits | Workload | Von Neumann Entropy (S_vN) | Max Bound | Schmidt Rank | Entanglement Regime | MPS Complexity Tier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 10 | QFT | 0.0000 bits | 5.0 | 1 | Product State | Trivial (chi=1) |
| 15 | QFT | 0.0000 bits | 7.0 | 1 | Product State | Trivial (chi=1) |
| 20 | QFT | 0.0000 bits | 10.0 | 1 | Product State | Trivial (chi=1) |

---

## Telemetry Visualizations
![Qubit Vs Latency](qubit_vs_latency.png)

![Qubit Vs Ram](qubit_vs_ram.png)

![Entanglement Entropy](entanglement_entropy.png)


---

## GitHub Ready
This report is formatted and ready to be posted directly into GitHub Issues, pull request reviews, or Discussions.