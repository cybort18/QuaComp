// OpenQASM 2.0 Benchmark: 4-Qubit GHZ Entangled State
OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];
creg c[4];

h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];

barrier q;
measure q -> c;
