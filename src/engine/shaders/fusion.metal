#include <metal_stdlib>
using namespace metal;

// 2D Complex number representation in Metal: x = real, y = imag
typedef float2 cfloat;

inline cfloat c_mul(cfloat a, cfloat b) {
    return cfloat(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline cfloat c_add(cfloat a, cfloat b) {
    return cfloat(a.x + b.x, a.y + b.y);
}

// 2x2 Complex Unitary Matrix Multiplication Kernel on Apple Metal
// C = A * B
kernel void matmul_2x2_metal(
    device const cfloat* A [[buffer(0)]],
    device const cfloat* B [[buffer(1)]],
    device cfloat* C [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < 4) {
        uint row = id / 2;
        uint col = id % 2;
        cfloat sum = cfloat(0.0f, 0.0f);
        for (uint k = 0; k < 2; ++k) {
            sum = c_add(sum, c_mul(A[row * 2 + k], B[k * 2 + col]));
        }
        C[id] = sum;
    }
}

// 4x4 Complex Unitary Matrix Multiplication Kernel on Apple Metal
// C = A * B (used for 2-qubit gate fusion)
kernel void matmul_4x4_metal(
    device const cfloat* A [[buffer(0)]],
    device const cfloat* B [[buffer(1)]],
    device cfloat* C [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < 16) {
        uint row = id / 4;
        uint col = id % 4;
        cfloat sum = cfloat(0.0f, 0.0f);
        for (uint k = 0; k < 4; ++k) {
            sum = c_add(sum, c_mul(A[row * 4 + k], B[k * 4 + col]));
        }
        C[id] = sum;
    }
}
