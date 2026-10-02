#ifndef QUACOMP_CUDA_FUSION_CUH
#define QUACOMP_CUDA_FUSION_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif

// 128-bit double-precision complex number representation
typedef cuDoubleComplex cdouble;

// Complex arithmetic inline device helpers
__device__ __forceinline__ cdouble c_mul(cdouble a, cdouble b) {
    return make_cuDoubleComplex(
        cuCreal(a) * cuCreal(b) - cuCimag(a) * cuCimag(b),
        cuCreal(a) * cuCimag(b) + cuCimag(a) * cuCreal(b)
    );
}

__device__ __forceinline__ cdouble c_add(cdouble a, cdouble b) {
    return make_cuDoubleComplex(
        cuCreal(a) + cuCreal(b), 
        cuCimag(a) + cuCimag(b)
    );
}

__device__ __forceinline__ cdouble c_sub(cdouble a, cdouble b) {
    return make_cuDoubleComplex(
        cuCreal(a) - cuCreal(b), 
        cuCimag(a) - cuCimag(b)
    );
}

// Kernel signatures
__global__ void matmul_2x2_cuda(const cdouble* A, const cdouble* B, cdouble* C);

__global__ void matmul_4x4_cuda(const cdouble* A, const cdouble* B, cdouble* C);

__global__ void apply_gate_1q_cuda(
    cdouble* state, 
    const cdouble* U, 
    unsigned int target, 
    unsigned long long total_pairs
);

__global__ void apply_gate_2q_cuda(
    cdouble* state, 
    const cdouble* U, 
    unsigned int q0, 
    unsigned int q1, 
    unsigned long long total_quads
);

__global__ void init_statevector_cuda(
    cdouble* state, 
    unsigned long long total_states
);

#ifdef __cplusplus
}
#endif

#endif // QUACOMP_CUDA_FUSION_CUH
