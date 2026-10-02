#include <cuda_runtime.h>
#include <cuComplex.h>

#include "fusion.cuh"

extern "C" {

// -----------------------------------------------------------------------------
// 1. Matrix Multiplication Kernels for Gate Fusion
// -----------------------------------------------------------------------------

/**
 * 2x2 Complex Unitary Matrix Multiplication Kernel on NVIDIA CUDA.
 * Computes C = A * B where A, B, and C are 2x2 complex matrices.
 */
__global__ void matmul_2x2_cuda(const cuDoubleComplex* A, const cuDoubleComplex* B, cuDoubleComplex* C) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < 4) {
        int row = id / 2;
        int col = id % 2;
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
        for (int k = 0; k < 2; ++k) {
            sum = c_add(sum, c_mul(A[row * 2 + k], B[k * 2 + col]));
        }
        C[id] = sum;
    }
}

/**
 * 4x4 Complex Unitary Matrix Multiplication Kernel on NVIDIA CUDA.
 * Computes C = A * B where A, B, and C are 4x4 complex matrices (2-qubit fusion).
 */
__global__ void matmul_4x4_cuda(const cuDoubleComplex* A, const cuDoubleComplex* B, cuDoubleComplex* C) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < 16) {
        int row = id / 4;
        int col = id % 4;
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
        for (int k = 0; k < 4; ++k) {
            sum = c_add(sum, c_mul(A[row * 4 + k], B[k * 4 + col]));
        }
        C[id] = sum;
    }
}

// -----------------------------------------------------------------------------
// 2. Direct Statevector Application Kernels
// -----------------------------------------------------------------------------

/**
 * Initialize quantum statevector to |00...0> on GPU VRAM.
 */
__global__ void init_statevector_cuda(cuDoubleComplex* state, unsigned long long total_states) {
    unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_states) {
        state[idx] = (idx == 0ULL) ? make_cuDoubleComplex(1.0, 0.0) : make_cuDoubleComplex(0.0, 0.0);
    }
}

/**
 * 1-Qubit Unitary Transformation Kernel on Statevector in GPU VRAM.
 * 
 * Thread Mapping:
 *   Each thread processes a basis state pair (i0, i1) differing only at bit `target`.
 *   Bit-twiddling inserts 0 at position `target` to form i0, and 1 to form i1.
 */
__global__ void apply_gate_1q_cuda(
    cuDoubleComplex* state, 
    const cuDoubleComplex* U, 
    unsigned int target, 
    unsigned long long total_pairs
) {
    unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_pairs) return;

    unsigned long long low = idx & ((1ULL << target) - 1ULL);
    unsigned long long high = (idx >> target) << (target + 1ULL);
    unsigned long long i0 = high | low;
    unsigned long long i1 = i0 | (1ULL << target);

    cuDoubleComplex v0 = state[i0];
    cuDoubleComplex v1 = state[i1];

    // [psi[i0]'] = [ U[0] U[1] ] [ v0 ]
    // [psi[i1]']   [ U[2] U[3] ] [ v1 ]
    state[i0] = c_add(c_mul(U[0], v0), c_mul(U[1], v1));
    state[i1] = c_add(c_mul(U[2], v0), c_mul(U[3], v1));
}

/**
 * 2-Qubit Fused Unitary Transformation Kernel on Statevector in GPU VRAM.
 * 
 * Thread Mapping:
 *   Each thread processes a 4-tuple of basis states (i00, i01, i10, i11) differing 
 *   at bit positions `q0` (target) and `q1` (control).
 */
__global__ void apply_gate_2q_cuda(
    cuDoubleComplex* state, 
    const cuDoubleComplex* U, 
    unsigned int q0, 
    unsigned int q1, 
    unsigned long long total_quads
) {
    unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_quads) return;

    unsigned int q_min = (q0 < q1) ? q0 : q1;
    unsigned int q_max = (q0 < q1) ? q1 : q0;

    unsigned long long low = idx & ((1ULL << q_min) - 1ULL);
    unsigned long long mid = (idx >> q_min) & ((1ULL << (q_max - q_min - 1ULL)) - 1ULL);
    unsigned long long high = idx >> (q_max - 1ULL);
    unsigned long long base = low | (mid << (q_min + 1ULL)) | (high << (q_max + 1ULL));

    unsigned long long i00 = base;
    unsigned long long i01 = base | (1ULL << q0);
    unsigned long long i10 = base | (1ULL << q1);
    unsigned long long i11 = base | (1ULL << q0) | (1ULL << q1);

    cuDoubleComplex v0 = state[i00];
    cuDoubleComplex v1 = state[i01];
    cuDoubleComplex v2 = state[i10];
    cuDoubleComplex v3 = state[i11];

    // Transformation: V_out = U * V_in
    state[i00] = c_add(c_add(c_mul(U[0], v0), c_mul(U[1], v1)), c_add(c_mul(U[2], v2), c_mul(U[3], v3)));
    state[i01] = c_add(c_add(c_mul(U[4], v0), c_mul(U[5], v1)), c_add(c_mul(U[6], v2), c_mul(U[7], v3)));
    state[i10] = c_add(c_add(c_mul(U[8], v0), c_mul(U[9], v1)), c_add(c_mul(U[10], v2), c_mul(U[11], v3)));
    state[i11] = c_add(c_add(c_mul(U[12], v0), c_mul(U[13], v1)), c_add(c_mul(U[14], v2), c_mul(U[15], v3)));
}

} // extern "C"
