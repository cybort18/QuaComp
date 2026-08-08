# Product Requirement Document (PRD)
# Project Name: QuaComp (Quantum Computer Simulation Benchmark)

**Version:** 1.0.0  
**Status:** Approved for Development  
**Target Environment:** Cross-platform (Windows, macOS, Linux)  
**Primary Tech Stack:** Python 3.10+, Qiskit / Aer, psutil, Rich, Pytest  

---

## 1. Executive Summary & Vision

### 1.1 Overview
`QuaComp` adalah pustaka *open-source* dan alat penguji berbasis CLI (*Command Line Interface*) yang dirancang untuk menguji batas kemampuan komputasi lokal (PC / Laptop / Workstation) dalam melakukan simulasi komputer kuantum. 

Dengan memanfaatkan simulasi ruang keadaan (*state-vector simulation*) berdimensi $2^n$, `QuaComp-Bench` mengukur alokasi memori (RAM), tingkat penggunaan CPU/GPU, serta kecepatan eksekusi matriks gerbang kuantum (*quantum gates*) pada berbagai tingkat *qubit* dan kedalaman sirkuit (*circuit depth*). Selain metode state-vector, QuaComp juga mendukung simulasi Matrix Product State (MPS) untuk menangani sirkuit kuantum skala besar (30 hingga 100+ qubit) secara efisien dengan melakukan kompresi tensor ruang keadaan pada kapasitas RAM yang terbatas.

### 1.2 Core Value Proposition
- **Pre-flight Safety:** Mencegah terjadinya *Crash / Out-Of-Memory (OOM)* dengan menghitung estimasi alokasi $2^n$ sebelum eksekusi.
- **Realistic Benchmarks:** Menguji berbagai jenis beban sirkuit kuantum (*Shallow*, *Deep*, dan *Algorithmic QFT*).
- **Comprehensive Profiling:** Melacak waktu eksekusi, *latency*, alokasi RAM puncak (*peak RAM*), serta penggunaan CPU *multi-core*.
- **Standardized Scoring:** Menghasilkan skor acuan kuantum ("QuaComp Score") agar pengguna dapat membandingkan performa antar perangkat.
- **Scalable MPS Simulation:** Mendukung simulasi sirkuit kuantum dengan qubit tinggi (hingga 100+ qubit) menggunakan tensor compression (Matrix Product State), melampaui batasan memori state-vector konvensional pada sirkuit ber-entanglement rendah-sedang.

---

## 2. Target Audience & Primary Use Cases

1. **Peneliti & Mahasiswa Kuantum:** Mengetahui batas maksimum *qubit* yang dapat disimulasikan secara lokal sebelum mendeploy sirkuit ke IBM Quantum / Cloud.
2. **Hardware Enthusiasts & Benchmarkers:** Menguji kekuatan CPU, kecepatan RAM, dan *bandwidth* memori menggunakan kalkulasi matriks kuantum skala besar.
3. **Developer / AI Agent:** Menjadikan repository ini sebagai referensi proyek modular yang siap dikembangkan lebih lanjut.

---

## 3. Functional Requirements (FR)

### FR-1: Pre-flight Memory Safety System
- **Deskripsi:** Sebelum mengalokasikan vektor keadaan kuantum $2^n$, sistem wajib memeriksa ketersediaan RAM fisik sistem.
- **Formula Estimasi RAM:** 
  $$	ext{RAM Bytes} = 2^n 	imes 16 	ext{ bytes (untuk complex128)}$$
- **Perilaku:** Jika estimasi memori melebihi $85\%$ dari sisa RAM fisik yang tersedia, sistem memberikan peringatan (*warning*) atau secara otomatis menghentikan pengujian untuk mencegah OOM Crash.

### FR-2: Incremental Qubit Stress Testing Engine
- **Deskripsi:** Pengujian *qubit* secara bertahap mulai dari $n = 10$ hingga batas maksimum sistem ($n = 28	ext{--}32+$ tergantung RAM).
- **Mode Eksekusi:**
  1. *Quick Benchmark:* Tes cepat $n \in \{10, 15, 20, 24, 28\}$.
  2. *Full Stress Test:* Incremental $+1$ qubit hingga menyentuh batas RAM/OOM Threshold.
  3. *Custom Test:* Pengguna menentukan jumlah qubit dan kedalaman sirkuit sendiri.

### FR-3: Benchmark Workload Suite
Terdapat 3 tipe sirkuit kuantum yang diuji:
1. **Shallow Workload (Bell / Hadamard):**
   - Menguji *overhead* awal dan inisialisasi keadaan.
   - Sirkuit: Gerbang $H$ pada semua qubit + pengikatan $CX$.
2. **Deep Workload (Random Circuit):**
   - Menguji eksekusi perkalian matriks secara intensif.
   - Sirkuit: Gerbang rotasi acak ($R_x, R_y, R_z$) dan gerbang $CNOT$ bertingkat dengan depth $d \in \{10, 50, 100\}$.
3. **Algorithmic Workload (Quantum Fourier Transform - QFT):**
   - Menguji performa pada algoritma kuantum nyata.
   - Sirkuit: *QFT* bertingkat pada $n$ qubit.

### FR-4: Hardware Profiler & Telemetry Module
- **Indikator yang Dilacak:**
  - **Baseline Memory:** RAM sebelum simulasi berjalan.
  - **Peak Memory Usage:** RAM tertinggi saat matriks kuantum dialokasikan.
  - **Execution Latency:** Waktu kuis (ms/s) per sirkuit.
  - **CPU Core Utilization:** Persentase penggunaan core CPU.
  - **System Metadata:** Spesifikasi CPU, Total RAM Fisik, OS, Versi Python/Qiskit.

### FR-5: Scoring Formula Engine
Skor akhir (**QuaComp Score**) dihitung berdasarkan kombinasi *Qubit Maksimum* dan *Throughput Eksekusi*:
$$\text{QuaComp Score} = \left( 2^{n_{\text{max}}} \times 10 
\right) + \left( \frac{\text{Total Gates Executed}}{\text{Total Time (seconds)}} 
\right)$$
- **Skor Kategori:**
  - *Entry-Level:* < 100,000 pts (Maks 18-20 Qubit)
  - *Mid-Range:* 100,000 - 1,000,000 pts (Maks 22-25 Qubit)
  - *High-Performance:* 1,000,000 - 50,000,000 pts (Maks 26-28 Qubit)
  - *Extreme Workstation:* > 50,000,000 pts (> 29 Qubit)

### FR-6: Report & Export Module
- Hasil pengujian dapat diekspor ke:
  1. **JSON Output:** Untuk konsumsi programatis (`results/benchmark_<timestamp>.json`).
  2. **Markdown Summary:** Laporan rapi yang siap di-paste ke GitHub Issue/Discussion (`results/report.md`).
  3. **Terminal Dashboard:** Tampilan tabel & progress bar interaktif di CLI.

### FR-7: Matrix Product State (MPS) Simulation Engine
- **Deskripsi:** Sistem harus menyediakan opsi untuk mengeksekusi sirkuit menggunakan metode network tensor Matrix Product State (MPS) untuk mensimulasikan qubit dalam jumlah sangat besar dengan konsumsi memori RAM yang minimal.
- **Spesifikasi & Perilaku:**
  - Integrasi dengan backend `qiskit_aer.AerSimulator(method='matrix_product_state')`.
  - Konfigurasi batas Bond Dimension ($\chi$) yang dapat disesuaikan via CLI `--bond-dim` (nilai default $\chi = 64$).
  - Kemampuan simulasi high-qubit stress test pada rentang $n = 30\text{--}100$ qubits untuk tipe sirkuit dengan tingkat entanglement rendah hingga sedang.
  - Pelacakan metrik MPS khusus:
    - Perbandingan efisiensi RAM antara metode MPS dengan Statevector teoritis ($2^n \times 16$ bytes).
    - Ukuran puncak Bond Dimension yang aktif digunakan selama simulasi berjalan.

---

## 4. Non-Functional Requirements (NFR)

- **Performance:** Module profiler tidak boleh menambah *overhead* memori/CPU lebih dari $1\%$ dari simulasi utama.
- **Portability:** Berjalan di Windows 10/11, macOS (Intel & Apple Silicon M1/M2/M3), dan Ubuntu/Linux.
- **User Experience (CLI):** Menggunakan antarmuka terminal modern dengan visualisasi progress bar dan tabel berwarna (via library `rich`).
- **Code Quality:** Penulisan kode PEP8 compliant, Type Hints penuh, dan Unit Test coverage > 80%.

---

## 5. Technical Architecture & File Structure

```text
QuaComp-bench/
├── cli/
│   ├── __init__.py
│   └── main.py             # Entry point CLI (Rich UI - update: Opsi argumen --method dan --bond-dim)
├── src/
│   ├── __init__.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── circuits.py     # Pembuat sirkuit kuantum (Shallow, Deep, QFT)
│   │   ├── mps.py          # Handler konfigurasi MPS & kalkulasi kompresi tensor
│   │   └── simulator.py    # Wrapper eksekusi Qiskit Aer (mendukung 'statevector' & 'matrix_product_state')
│   ├── profiler/
│   │   ├── __init__.py
│   │   ├── memory.py       # Pre-flight safety check & RAM profiler
│   │   └── telemetry.py    # CPU & execution timer
│   ├── scorer/
│   │   ├── __init__.py
│   │   └── calculator.py   # Kalkulasi skor QuaComp
│   └── reporter/
│       ├── __init__.py
│       ├── json_exporter.py
│       └── md_exporter.py
├── tests/
│   ├── test_engine.py      # Pengujian sirkuit dan simulator
│   ├── test_memory.py      # Pengujian estimator memori
│   ├── test_scorer.py      # Pengujian kalkulasi skor
│   └── test_mps.py         # Pengujian modul MPS baru
├── requirements.txt
├── PRD.md
├── README.md
└── LICENSE
```

---

## 6. Implementation Roadmap for AI Agent

### Phase 1: Core Simulation & Safety (Week 1)
- [ ] Buat modul `profiler/memory.py` untuk estimasi alokasi memori $2^n$ dan pengecekan RAM fisik via `psutil`.
- [ ] Buat modul `engine/circuits.py` untuk membangkitkan sirkuit Bell, Random, dan QFT.
- [ ] Buat modul `engine/simulator.py` untuk mengeksekusi sirkuit menggunakan Qiskit `AerSimulator(method='statevector')`.

### Phase 2: Scoring & CLI Interface (Week 2)
- [ ] Implementasikan formula kalkulasi skor di `scorer/calculator.py`.
- [ ] Buat CLI interaktif di `cli/main.py` menggunakan `rich` (Progress bar, Live table, Banner ASCII).

### Phase 3: Exporters, Tests & Documentation (Week 3)
- [ ] Tambahkan eksportir ke JSON dan Markdown summary di `reporter/`.
- [ ] Buat unit test lengkap di folder `tests/`.
- [ ] Tulis dokumentasi `README.md` open-source dengan contoh penggunaan & petunjuk kontribusi.

### Phase 4: MPS Integration & High-Qubit Benchmarking (Week 4)
- [ ] Buat modul `src/engine/mps.py` dan perbarui `src/engine/simulator.py` untuk mendukung backend MPS.
- [ ] Integrasikan flag `--method` dan `--bond-dim` pada CLI (`cli/main.py`).
- [ ] Tambahkan unit test khusus MPS di `tests/test_mps.py`.
- [ ] Perbarui modul reporter (JSON/MD) untuk menampilkan metrik statistik MPS.
