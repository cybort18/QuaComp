#ifndef QUACOMP_CPP_FUSION_HPP
#define QUACOMP_CPP_FUSION_HPP

#include <vector>
#include <complex>
#include <stdexcept>
#include <string>

namespace quacomp {

using Complex = std::complex<double>;
using Matrix = std::vector<std::vector<Complex>>;

// Matrix operations
Matrix create_identity(size_t dim);
Matrix matmul(const Matrix& A, const Matrix& B);
Matrix matmul_2x2(const Matrix& A, const Matrix& B);
Matrix matmul_4x4(const Matrix& A, const Matrix& B);
Matrix kronecker_product(const Matrix& A, const Matrix& B);

// Single-Pass Gate Fusion
// Given list of unitaries [U_1, U_2, ..., U_k] applied sequentially,
// the combined unitary acting on state |psi> is U_fused = U_k * ... * U_2 * U_1.
Matrix fuse_matrices(const std::vector<Matrix>& matrices);

} // namespace quacomp

#endif // QUACOMP_CPP_FUSION_HPP
