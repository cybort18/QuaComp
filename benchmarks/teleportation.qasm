// OpenQASM 2.0 Benchmark: Quantum Teleportation Protocol
OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c0[1];
creg c1[1];

// Prepare state to teleport on q[0]
rx(pi/4) q[0];
rz(pi/2) q[0];

// Create Bell pair between q[1] and q[2]
h q[1];
cx q[1], q[2];

// Bell measurement on q[0] and q[1]
cx q[0], q[1];
h q[0];

measure q[0] -> c0[0];
measure q[1] -> c1[0];

// Corrections applied to q[2]
cx q[1], q[2];
cz q[0], q[2];
