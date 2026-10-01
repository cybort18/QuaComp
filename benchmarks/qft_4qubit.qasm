// OpenQASM 2.0 Benchmark: 4-Qubit Quantum Fourier Transform (QFT)
OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];
creg c[4];

// Qubit 0
h q[0];
cu1(pi/2) q[1], q[0];
cu1(pi/4) q[2], q[0];
cu1(pi/8) q[3], q[0];

// Qubit 1
h q[1];
cu1(pi/2) q[2], q[1];
cu1(pi/4) q[3], q[1];

// Qubit 2
h q[2];
cu1(pi/2) q[3], q[2];

// Qubit 3
h q[3];

// Swap network for bit reversal
swap q[0], q[3];
swap q[1], q[2];

barrier q;
measure q -> c;
