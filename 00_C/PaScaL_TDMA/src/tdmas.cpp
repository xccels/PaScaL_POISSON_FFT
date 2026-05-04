//======================================================================================================================
//> @file        tdmas.f90
//> @brief       Tridiagonal matrix (TDM) solvers using the Thomas algorithm.
//> @details     A single TDM solver and many TDM solver with non-cyclic and cyclic conditions.
//>
//> @author      
//>              - Ki-Ha Kim (k-kiha@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
//>              - Ji-Hoon Kang (jhkang@kisti.re.kr), Korea Institute of Science and Technology Information
//>              - Jung-Il Choi (jic@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
//>
//> @date        May 2023
//> @version     2.0
//> @par         Copyright
//>              Copyright (c) 2019-2023 Ki-Ha Kim and Jung-Il choi, Yonsei University and 
//>              Ji-Hoon Kang, Korea Institute of Science and Technology Information, All rights reserved.
//> @par         License     
//>              This project is release under the terms of the MIT License (see LICENSE file).
//======================================================================================================================

//>
//> @brief       Solve a single tridiagonal system of equations using the Thomas algorithm.
//> @param       a       Coefficients in lower diagonal elements
//> @param       b       Coefficients in diagonal elements
//> @param       c       Coefficients in upper diagonal elements
//> @param       d       Coefficients in the right-hand side terms
//> @param       n1      Number of rows in each process, dimension of tridiagonal matrix N divided by nprocs
//>

#include "pascal_tdma.hpp"
#include <vector>

// Solve a single tridiagonal system using Thomas algorithm
extern "C" void tdma_single(double* a, double* b, double* c, double* d, int n1) {
    d[0] /= b[0];
    c[0] /= b[0];
    for (int i = 1; i < n1; ++i) {
        double r = 1.0 / (b[i] - a[i] * c[i - 1]);
        d[i] = r * (d[i] - a[i] * d[i - 1]);
        c[i] = r * c[i];
    }
    for (int i = n1 - 2; i >= 0; --i) {
        d[i] -= c[i] * d[i + 1];
    }
}

// Solve a single cyclic tridiagonal system
extern "C" void tdma_cycl_single(double* a, double* b, double* c, double* d, int n1) {
    std::vector<double> e(n1, 0.0);
    e[1] = -a[1];
    e[n1 - 1] = -c[n1 - 1];

    d[1] /= b[1];
    e[1] /= b[1];
    c[1] /= b[1];
    for (int i = 2; i < n1; ++i) {
        double rr = 1.0 / (b[i] - a[i] * c[i - 1]);
        d[i] = rr * (d[i] - a[i] * d[i - 1]);
        e[i] = rr * (e[i] - a[i] * e[i - 1]);
        c[i] = rr * c[i];
    }
    for (int i = n1 - 2; i >= 1; --i) {
        d[i] -= c[i] * d[i + 1];
        e[i] -= c[i] * e[i + 1];
    }
    d[0] = (d[0] - a[0] * d[n1 - 1] - c[0] * d[1]) /
           (b[0] + a[0] * e[n1 - 1] + c[0] * e[1]);
    for (int i = 1; i < n1; ++i) {
        d[i] += d[0] * e[i];
    }
}

namespace {
inline std::size_t idx(std::size_t i, std::size_t j, std::size_t ncols) {
    return j * ncols + i;
}
}

// Solve many tridiagonal systems
extern "C" void tdma_many(double* __restrict__ a, double* __restrict__ b, double* __restrict__ c, double* __restrict__ d,
                           int n1, int n2) {
    std::vector<double> r(n1);
    for (int i = 0; i < n1; ++i) {
        std::size_t id = idx(i, 0, n1);
        d[id] /= b[id];
        c[id] /= b[id];
    }
    for (int j = 1; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
            std::size_t id = idx(i, j, n1);
            std::size_t idm1 = idx(i, j - 1, n1);
            r[i] = 1.0 / (b[id] - a[id] * c[idm1]);
            d[id] = r[i] * (d[id] - a[id] * d[idm1]);
            c[id] = r[i] * c[id];
        }
    }
    for (int j = n2 - 2; j >= 0; --j) {
        for (int i = 0; i < n1; ++i) {
            std::size_t id = idx(i, j, n1);
            std::size_t idp1 = idx(i, j + 1, n1);
            d[id] -= c[id] * d[idp1];
        }
    }
}

// extern "C" void tdma_many(double* __restrict__ a,
//                           double* __restrict__ b,
//                           double* __restrict__ c,
//                           double* __restrict__ d,
//                           int n1, int n2)
// {
//     // r는 열 방향 보조 버퍼
//     std::vector<double> r(n1);

//     #pragma omp parallel
//     // 첫 행
//     {
//         const std::size_t row = 0; // j=0
//         #pragma omp for simd schedule(static) aligned(a,b,c,d,r:64)
//         for (int i = 0; i < n1; ++i) {
//             const std::size_t id = row + i;
//             const double inv = 1.0 / b[id];
//             d[id] *= inv;
//             c[id] *= inv;
//         }
//     }

//     // forward sweep: j = 1..n2-1
//     for (int j = 1; j < n2; ++j) {
//         const std::size_t row   = (std::size_t)j * n1;
//         const std::size_t rowm1 = (std::size_t)(j - 1) * n1;

//         #pragma omp for simd schedule(static) aligned(a,b,c,d,r:64)
//         for (int i = 0; i < n1; ++i) {
//             const std::size_t id   = row   + i;
//             const std::size_t idm1 = rowm1 + i;

//             const double denom = b[id] - a[id] * c[idm1];
//             const double inv   = 1.0 / denom;

//             // r[i]  = inv;
//             d[id] = inv * (d[id] - a[id] * d[idm1]);
//             c[id] = inv * c[id];
//         }
//     }

//     // backward sweep: j = n2-2..0
//     for (int j = n2 - 2; j >= 0; --j) {
//         const std::size_t row   = (std::size_t)j * n1;
//         const std::size_t rowp1 = (std::size_t)(j + 1) * n1;

//         #pragma omp for simd schedule(static) aligned(a,b,c,d:64)
//         for (int i = 0; i < n1; ++i) {
//             const std::size_t id   = row   + i;
//             const std::size_t idp1 = rowp1 + i;
//             d[id] -= c[id] * d[idp1];
//         }
//     }
// }


// Solve many cyclic tridiagonal systems
extern "C" void tdma_cycl_many(double* a, double* b, double* c, double* d,
                                int n1, int n2) {
    std::vector<double> e(n1 * n2, 0.0);
    std::vector<double> rr(n1);
    for (int i = 0; i < n1; ++i) {
        e[idx(i, 1, n2)] = -a[idx(i, 1, n2)];
        e[idx(i, n2 - 1, n2)] = -c[idx(i, n2 - 1, n2)];
    }
    for (int i = 0; i < n1; ++i) {
        std::size_t id = idx(i, 1, n2);
        d[id] /= b[id];
        e[id] /= b[id];
        c[id] /= b[id];
    }
    for (int j = 2; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
            std::size_t id = idx(i, j, n2);
            std::size_t idm1 = idx(i, j - 1, n2);
            rr[i] = 1.0 / (b[id] - a[id] * c[idm1]);
            d[id] = rr[i] * (d[id] - a[id] * d[idm1]);
            e[id] = rr[i] * (e[id] - a[id] * e[idm1]);
            c[id] = rr[i] * c[id];
        }
    }
    for (int j = n2 - 2; j >= 1; --j) {
        for (int i = 0; i < n1; ++i) {
            std::size_t id = idx(i, j, n2);
            std::size_t idp1 = idx(i, j + 1, n2);
            d[id] -= c[id] * d[idp1];
            e[id] -= c[id] * e[idp1];
        }
    }
    for (int i = 0; i < n1; ++i) {
        std::size_t id0 = idx(i, 0, n2);
        std::size_t id1 = idx(i, 1, n2);
        std::size_t idn = idx(i, n2 - 1, n2);
        d[id0] = (d[id0] - a[id0] * d[idn] - c[id0] * d[id1]) /
                 (b[id0] + a[id0] * e[idn] + c[id0] * e[id1]);
    }
    for (int j = 1; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
            std::size_t id = idx(i, j, n2);
            std::size_t id0 = idx(i, 0, n2);
            d[id] += d[id0] * e[id];
        }
    }
}