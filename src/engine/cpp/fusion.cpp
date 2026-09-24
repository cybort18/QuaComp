#include "fusion.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

namespace py = pybind11;

namespace quacomp {

Matrix create_identity(size_t dim) {
    Matrix I(dim, std::vector<Complex>(dim, Complex(0.0, 0.0)));
    for (size_t i = 0; i < dim; ++i) {
        I[i][i] = Complex(1.0, 0.0);
    }
    return I;
}

Matrix matmul(const Matrix& A, const Matrix& B) {
    size_t rowsA = A.size();
    if (rowsA == 0) return Matrix();
    size_t colsA = A[0].size();
    size_t rowsB = B.size();
    if (rowsB == 0) return Matrix();
    size_t colsB = B[0].size();

    if (colsA != rowsB) {
        throw std::invalid_argument("Matrix dimension mismatch: cols(A) must equal rows(B).");
    }

    Matrix C(rowsA, std::vector<Complex>(colsB, Complex(0.0, 0.0)));
    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t k = 0; k < colsA; ++k) {
            Complex a_ik = A[i][k];
            for (size_t j = 0; j < colsB; ++j) {
                C[i][j] += a_ik * B[k][j];
            }
        }
    }
    return C;
}

Matrix matmul_2x2(const Matrix& A, const Matrix& B) {
    if (A.size() != 2 || A[0].size() != 2 || B.size() != 2 || B[0].size() != 2) {
        throw std::invalid_argument("Input matrices must be 2x2.");
    }
    Matrix C(2, std::vector<Complex>(2));
    C[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0];
    C[0][1] = A[0][0] * B[0][1] + A[0][1] * B[1][1];
    C[1][0] = A[1][0] * B[0][0] + A[1][1] * B[1][0];
    C[1][1] = A[1][0] * B[0][1] + A[1][1] * B[1][1];
    return C;
}

Matrix matmul_4x4(const Matrix& A, const Matrix& B) {
    if (A.size() != 4 || A[0].size() != 4 || B.size() != 4 || B[0].size() != 4) {
        throw std::invalid_argument("Input matrices must be 4x4.");
    }
    Matrix C(4, std::vector<Complex>(4, Complex(0.0, 0.0)));
    for (size_t i = 0; i < 4; ++i) {
        for (size_t k = 0; k < 4; ++k) {
            Complex a_ik = A[i][k];
            for (size_t j = 0; j < 4; ++j) {
                C[i][j] += a_ik * B[k][j];
            }
        }
    }
    return C;
}

Matrix kronecker_product(const Matrix& A, const Matrix& B) {
    size_t rA = A.size(), cA = A[0].size();
    size_t rB = B.size(), cB = B[0].size();
    Matrix C(rA * rB, std::vector<Complex>(cA * cB, Complex(0.0, 0.0)));
    for (size_t i = 0; i < rA; ++i) {
        for (size_t j = 0; j < cA; ++j) {
            for (size_t k = 0; k < rB; ++k) {
                for (size_t l = 0; l < cB; ++l) {
                    C[i * rB + k][j * cB + l] = A[i][j] * B[k][l];
                }
            }
        }
    }
    return C;
}

Matrix fuse_matrices(const std::vector<Matrix>& matrices) {
    if (matrices.empty()) {
        return Matrix();
    }
    size_t dim = matrices[0].size();
    if (dim == 0 || matrices[0][0].size() != dim) {
        throw std::invalid_argument("Matrices must be square.");
    }

    Matrix fused = create_identity(dim);
    // Sequential execution: U_k * ... * U_2 * U_1
    // In statevector evolution: |psi'> = U_k * ... * U_1 |psi>
    for (const auto& mat : matrices) {
        if (mat.size() != dim || mat[0].size() != dim) {
            throw std::invalid_argument("All matrices in fusion chain must have identical square dimensions.");
        }
        if (dim == 2) {
            fused = matmul_2x2(mat, fused);
        } else if (dim == 4) {
            fused = matmul_4x4(mat, fused);
        } else {
            fused = matmul(mat, fused);
        }
    }
    return fused;
}

} // namespace quacomp

PYBIND11_MODULE(quacomp_cpp, m) {
    m.doc() = "QuaComp High-Performance C++ Native Gate Fusion Extension";

    m.def("is_cpp_extension_available", []() { return true; }, "Check if C++ extension is loaded.");
    m.def("version", []() { return "1.0.0"; }, "C++ extension version string.");

    m.def("matmul", &quacomp::matmul, "General complex matrix multiplication C = A * B");
    m.def("matmul_2x2", &quacomp::matmul_2x2, "Fast 2x2 complex matrix multiplication");
    m.def("matmul_4x4", &quacomp::matmul_4x4, "Fast 4x4 complex matrix multiplication");
    m.def("kronecker_product", &quacomp::kronecker_product, "Kronecker tensor product C = A (x) B");
    m.def("fuse_matrices", &quacomp::fuse_matrices, "Single-pass sequential gate fusion U_fused = U_k * ... * U_1");
}
