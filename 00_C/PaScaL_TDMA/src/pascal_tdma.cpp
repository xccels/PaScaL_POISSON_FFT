//======================================================================================================================
//> @file        pascal_tdma.cpp
//> @brief       PaScaL_TDMA - Parallel and Scalable Library for TriDiagonal Matrix Algorithm
//> @details     PaScaL_TDMA provides an efficient and scalable computational procedure 
//>              to solve many tridiagonal systems in multi-dimensional partial differential equations. 
//>              The modified Thomas algorithm proposed by Laszlo et al.(2016) and the newly designed communication 
//>              scheme have been used to reduce the communication overhead in solving many tridiagonal systems.
//>              This library is for both single and many tridiagonal systems of equations. 
//>              The main algorithm for a tridiagonal matrix consists of the following five steps: 
//>
//>              (1) Transform the partitioned submatrices in the tridiagonal systems into modified submatrices:
//>                  Each computing core transforms the partitioned submatrices in the tridiagonal systems 
//>                  of equations into modified forms by applying the modified Thomas algorithm.
//>              (2) Construct reduced tridiagonal systems from the modified submatrices:
//>                  The reduced tridiagonal systems are constructed by collecting the first and last rows 
//>                  of the modified submatrices from each core using MPI_Ialltoallw.
//>              (3) Solve the reduced tridiagonal systems:
//>                  The reduced tridiagonal systems constructed in Step 2 are solved by applying the Thomas algorithm.
//>              (4) Distribute the solutions of the reduced tridiagonal systems:
//>                  The solutions of the reduced tridiagonal systems in Step 3 are distributed to each core 
//>                  using MPI_Ialltoallw. This communication is an exact inverse of the communication in Step 2.
//>              (5) Update the other unknowns in the modified tridiagonal systems:
//>                  The remaining unknowns in the modified submatrices in Step 1 are solved in each computing core 
//>                  using the solutions obtained in Step 3 and Step 4.
//>
//>              Step 1 and Step 5 are similar to the method proposed by Laszlo et al.(2016)
//>              which uses parallel cyclic reduction (PCR) algorithm to build and solve the reduced tridiagonal systems.
//>              Instead of using the PCR, we develop an all-to-all communication scheme using the MPI_Ialltoall
//>              function after the modified Thomas algorithm is executed. The number of coefficients for
//>              the reduced tridiagonal systems are greatly reduced, so we can avoid the communication 
//>              bandwidth problem, which is a main bottleneck for all-to-all communications.
//>              Our algorithm is also distinguished from the work of Mattor et al. (1995) which
//>              assembles the undetermined coefficients of the temporary solutions in a single processor 
//>              using MPI_Gather, where load imbalances are serious.
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
//> @brief       Module for PaScaL-TDMA library.
//> @details     It contains plans for tridiagonal systems of equations and subroutines for solving them 
//>              using the defined plans. The operation of the library includes the following three phases:
//>              (1) Create a data structure called a plan, which has the information for communication and reduced systems.
//>              (2) Solve the tridiagonal systems of equations executing from Step 1 to Step 5
//>              (3) Destroy the created plan

#include "pascal_tdma.hpp"

#include <algorithm>
#include <numeric>
#include <omp.h>

namespace timer {
void timer_stamp0(int);
void timer_stamp(int, int);
}

extern "C" {

// tdma solvers provided elsewhere
void tdma_single(double* a, double* b, double* c, double* d, int n1);
void tdma_cycl_single(double* a, double* b, double* c, double* d, int n1);
void tdma_many(double* a, double* b, double* c, double* d, int n1, int n2);
void tdma_cycl_many(double* a, double* b, double* c, double* d, int n1, int n2);

}

namespace {
inline std::size_t idx(std::size_t i, std::size_t j, std::size_t ncols) {
    return j * ncols + i;
}

void para_range(int n1, int n2, int nprocs, int myrank, int& ista, int& iend) {
    int iwork1 = (n2 - n1 + 1) / nprocs;
    int iwork2 = (n2 - n1 + 1) % nprocs;
    ista = myrank * iwork1 + n1 + std::min(myrank, iwork2);
    iend = ista + iwork1 - 1;
    if (iwork2 > myrank) ++iend;
}
} // unnamed namespace

extern "C" {

void PaScaL_TDMA_plan_single_create(ptdma_plan_single* plan, int myrank,
                                    int nprocs, MPI_Comm mpi_world,
                                    int gather_rank) {
    const int nr_rd = 2;
    const int nr_rt = nr_rd * nprocs;

    plan->myrank = myrank;
    plan->nprocs = nprocs;
    plan->gather_rank = gather_rank;
    plan->ptdma_world = mpi_world;
    plan->n_row_rt = nr_rt;

    plan->A_rd.assign(nr_rd, 0.0);
    plan->B_rd.assign(nr_rd, 0.0);
    plan->C_rd.assign(nr_rd, 0.0);
    plan->D_rd.assign(nr_rd, 0.0);

    plan->A_rt.assign(nr_rt, 0.0);
    plan->B_rt.assign(nr_rt, 0.0);
    plan->C_rt.assign(nr_rt, 0.0);
    plan->D_rt.assign(nr_rt, 0.0);
}

void PaScaL_TDMA_plan_single_destroy(ptdma_plan_single* plan) {
    plan->A_rd.clear(); plan->B_rd.clear(); plan->C_rd.clear(); plan->D_rd.clear();
    plan->A_rt.clear(); plan->B_rt.clear(); plan->C_rt.clear(); plan->D_rt.clear();
}

void PaScaL_TDMA_plan_many_create(ptdma_plan_many* plan, int n_sys,
                                  int myrank, int nprocs, MPI_Comm mpi_world) {
    const int ns_rd = n_sys;
    const int nr_rd = 2;
    int ista, iend;
    para_range(1, ns_rd, nprocs, myrank, ista, iend);
    int ns_rt = iend - ista + 1;

    std::vector<int> ns_rt_array(nprocs);
    MPI_Allgather(&ns_rt, 1, MPI_INT, ns_rt_array.data(), 1, MPI_INT, mpi_world);
    int nr_rt = nr_rd * nprocs;

    plan->nprocs = nprocs;
    plan->n_sys_rt = ns_rt;
    plan->n_row_rt = nr_rt;
    plan->ptdma_world = mpi_world;

    plan->A_rd.assign(ns_rd * nr_rd, 0.0);
    plan->B_rd.assign(ns_rd * nr_rd, 0.0);
    plan->C_rd.assign(ns_rd * nr_rd, 0.0);
    plan->D_rd.assign(ns_rd * nr_rd, 0.0);
    plan->A_rt.assign(ns_rt * nr_rt, 0.0);
    plan->B_rt.assign(ns_rt * nr_rt, 0.0);
    plan->C_rt.assign(ns_rt * nr_rt, 0.0);
    plan->D_rt.assign(ns_rt * nr_rt, 0.0);

    plan->ddtype_FS.resize(nprocs);
    plan->ddtype_BS.resize(nprocs);
    plan->count_send.assign(nprocs, 1);
    plan->displ_send.assign(nprocs, 0);
    plan->count_recv.assign(nprocs, 1);
    plan->displ_recv.assign(nprocs, 0);

    int bigsize[2], subsize[2], start[2];
    int offset = 0;
    for (int i = 0; i < nprocs; ++i) {
        // 얘가 반대가 돼야함
        // bigsize[0] = ns_rd;
        // bigsize[1] = nr_rd;
        // subsize[0] = ns_rt_array[i];
        // subsize[1] = nr_rd;
        // start[0] = offset; start[1] = 0;
        bigsize[1] = ns_rd;
        bigsize[0] = nr_rd;
        subsize[1] = ns_rt_array[i];
        subsize[0] = nr_rd;

        start[1] = offset; start[0] = 0;


        MPI_Type_create_subarray(2, bigsize, subsize, start,
                                 MPI_ORDER_C, MPI_DOUBLE,
                                 &plan->ddtype_FS[i]);
        MPI_Type_commit(&plan->ddtype_FS[i]);

        // 얘네도 반대임
        // bigsize[0] = ns_rt;
        // bigsize[1] = nr_rt;
        // subsize[0] = ns_rt;
        // subsize[1] = nr_rd;
        // start[0] = 0;
        // start[1] = nr_rd * i;
        bigsize[1] = ns_rt;
        bigsize[0] = nr_rt;
        subsize[1] = ns_rt;
        subsize[0] = nr_rd;
        start[1] = 0;
        start[0] = nr_rd * i;

        MPI_Type_create_subarray(2, bigsize, subsize, start,
                                 MPI_ORDER_C, MPI_DOUBLE,
                                 &plan->ddtype_BS[i]);
        MPI_Type_commit(&plan->ddtype_BS[i]);
        offset += ns_rt_array[i];
    }
}

void PaScaL_TDMA_plan_many_destroy(ptdma_plan_many* plan, int nprocs) {
    for (int i = 0; i < nprocs; ++i) {
        MPI_Type_free(&plan->ddtype_FS[i]);
        MPI_Type_free(&plan->ddtype_BS[i]);
    }
    plan->ddtype_FS.clear(); 
    plan->ddtype_BS.clear();
    plan->count_send.clear(); 
    plan->displ_send.clear();
    plan->count_recv.clear(); 
    plan->displ_recv.clear();
    plan->A_rd.clear(); 
    plan->B_rd.clear(); 
    plan->C_rd.clear(); 
    plan->D_rd.clear();
    plan->A_rt.clear(); 
    plan->B_rt.clear(); 
    plan->C_rt.clear(); 
    plan->D_rt.clear();
}

void PaScaL_TDMA_single_solve(ptdma_plan_single* plan, double* A, double* B,
                              double* C, double* D, int n_row) {
    if (plan->nprocs == 1) {
        tdma_single(A, B, C, D, n_row);
        return;
    }

    A[0] /= B[0];
    D[0] /= B[0];
    C[0] /= B[0];
    A[1] /= B[1];
    D[1] /= B[1];
    C[1] /= B[1];

    for (int i = 2; i < n_row; ++i) {
        double r = 1.0 / (B[i] - A[i] * C[i - 1]);
        D[i] = r * (D[i] - A[i] * D[i - 1]);
        C[i] = r * C[i];
        A[i] = -r * A[i] * A[i - 1];
    }

    for (int i = n_row - 3; i >= 1; --i) {
        D[i] = D[i] - C[i] * D[i + 1];
        A[i] = A[i] - C[i] * A[i + 1];
        C[i] = -C[i] * C[i + 1];
    }

    double r = 1.0 / (1.0 - A[1] * C[0]);
    D[0] = r * (D[0] - C[0] * D[1]);
    A[0] = r * A[0];
    C[0] = -r * C[0] * C[1];

    plan->A_rd[0] = A[0];
    plan->A_rd[1] = A[n_row - 1];
    plan->B_rd[0] = 1.0; plan->B_rd[1] = 1.0;
    plan->C_rd[0] = C[0];
    plan->C_rd[1] = C[n_row - 1];
    plan->D_rd[0] = D[0];
    plan->D_rd[1] = D[n_row - 1];

    MPI_Request request[4];
    MPI_Igather(plan->A_rd.data(), 2, MPI_DOUBLE,
                plan->A_rt.data(), 2, MPI_DOUBLE,
                plan->gather_rank, plan->ptdma_world, &request[0]);
    MPI_Igather(plan->B_rd.data(), 2, MPI_DOUBLE,
                plan->B_rt.data(), 2, MPI_DOUBLE,
                plan->gather_rank, plan->ptdma_world, &request[1]);
    MPI_Igather(plan->C_rd.data(), 2, MPI_DOUBLE,
                plan->C_rt.data(), 2, MPI_DOUBLE,
                plan->gather_rank, plan->ptdma_world, &request[2]);
    MPI_Igather(plan->D_rd.data(), 2, MPI_DOUBLE,
                plan->D_rt.data(), 2, MPI_DOUBLE,
                plan->gather_rank, plan->ptdma_world, &request[3]);
    MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

    if (plan->myrank == plan->gather_rank) {
        tdma_single(plan->A_rt.data(), plan->B_rt.data(),
                    plan->C_rt.data(), plan->D_rt.data(), plan->n_row_rt);
    }

    MPI_Iscatter(plan->D_rt.data(), 2, MPI_DOUBLE,
                 plan->D_rd.data(), 2, MPI_DOUBLE,
                 plan->gather_rank, plan->ptdma_world, &request[0]);
    MPI_Waitall(1, request, MPI_STATUSES_IGNORE);

    D[0] = plan->D_rd[0];
    D[n_row - 1] = plan->D_rd[1];
    for (int i = 1; i < n_row - 1; ++i) {
        D[i] = D[i] - A[i] * D[0] - C[i] * D[n_row - 1];
    }
}

void PaScaL_TDMA_single_solve_cycle(ptdma_plan_single* plan, double* A,
                                    double* B, double* C, double* D,
                                    int n_row) {
    if (plan->nprocs == 1) {
        tdma_cycl_single(A, B, C, D, n_row);
        return;
    }

    A[0] /= B[0];
    D[0] /= B[0];
    C[0] /= B[0];
    A[1] /= B[1];
    D[1] /= B[1];
    C[1] /= B[1];

    for (int i = 2; i < n_row; ++i) {
        double rr = 1.0 / (B[i] - A[i] * C[i - 1]);
        D[i] = rr * (D[i] - A[i] * D[i - 1]);
        C[i] = rr * C[i];
        A[i] = -rr * A[i] * A[i - 1];
    }

    for (int i = n_row - 3; i >= 1; --i) {
        D[i] = D[i] - C[i] * D[i + 1];
        A[i] = A[i] - C[i] * A[i + 1];
        C[i] = -C[i] * C[i + 1];
    }

    double rr = 1.0 / (1.0 - A[1] * C[0]);
    D[0] = rr * (D[0] - C[0] * D[1]);
    A[0] = rr * A[0];
    C[0] = -rr * C[0] * C[1];

    plan->A_rd[0] = A[0];
    plan->A_rd[1] = A[n_row - 1];
    plan->B_rd[0] = 1.0; plan->B_rd[1] = 1.0;
    plan->C_rd[0] = C[0];
    plan->C_rd[1] = C[n_row - 1];
    plan->D_rd[0] = D[0];
    plan->D_rd[1] = D[n_row - 1];

    MPI_Request request[4];
    MPI_Igather(plan->A_rd.data(), 2, MPI_DOUBLE,
                plan->A_rt.data(), 2, MPI_DOUBLE,
                plan->gather_rank, plan->ptdma_world, &request[0]);
    MPI_Igather(plan->B_rd.data(), 2, MPI_DOUBLE,
                plan->B_rt.data(), 2, MPI_DOUBLE,
                plan->gather_rank, plan->ptdma_world, &request[1]);
    MPI_Igather(plan->C_rd.data(), 2, MPI_DOUBLE,
                plan->C_rt.data(), 2, MPI_DOUBLE,
                plan->gather_rank, plan->ptdma_world, &request[2]);
    MPI_Igather(plan->D_rd.data(), 2, MPI_DOUBLE,
                plan->D_rt.data(), 2, MPI_DOUBLE,
                plan->gather_rank, plan->ptdma_world, &request[3]);
    MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

    if (plan->myrank == plan->gather_rank) {
        tdma_cycl_single(plan->A_rt.data(), plan->B_rt.data(),
                         plan->C_rt.data(), plan->D_rt.data(), plan->n_row_rt);
    }

    MPI_Iscatter(plan->D_rt.data(), 2, MPI_DOUBLE,
                 plan->D_rd.data(), 2, MPI_DOUBLE,
                 plan->gather_rank, plan->ptdma_world, &request[0]);
    MPI_Waitall(1, request, MPI_STATUSES_IGNORE);

    D[0] = plan->D_rd[0];
    D[n_row - 1] = plan->D_rd[1];
    for (int i = 1; i < n_row - 1; ++i) {
        D[i] = D[i] - A[i] * D[0] - C[i] * D[n_row - 1];
    }
}

void PaScaL_TDMA_many_solve_time(ptdma_plan_many* plan, double* __restrict__ A, double* __restrict__ B,
                             double* __restrict__ C, double* __restrict__ D, int n_sys, int n_row) {
    
    constexpr int stamp_tdma = 3;
    if (plan->nprocs == 1) {
        timer::timer_stamp0(stamp_tdma);
        tdma_many(A, B, C, D, n_sys, n_row);
        timer::timer_stamp(10, stamp_tdma);
        return;
    }

    
    timer::timer_stamp0(stamp_tdma);

   double t1 = MPI_Wtime();

{
    for (int i = 0; i < n_sys; ++i) {
        // idx(i,j,ncols) = j * ncols + i;
        std::size_t i0 = idx(i, 0, n_sys);
        std::size_t i1 = idx(i, 1, n_sys);
        
        A[i] /= B[i]; D[i] /= B[i]; C[i] /= B[i];
        A[i1] /= B[i1]; D[i1] /= B[i1]; C[i1] /= B[i1];
    }

    for (int j = 2; j < n_row; ++j) {
        for (int i = 0; i < n_sys; ++i) {
            std::size_t id = idx(i, j, n_sys);
            std::size_t idm1 = idx(i, j - 1, n_sys);
            double r = 1.0 / (B[id] - A[id] * C[idm1]);
            D[id] = r * (D[id] - A[id] * D[idm1]);
            C[id] = r * C[id];
            A[id] = -r * A[id] * A[idm1];
        }
    }

    for (int j = n_row - 3; j >= 1; --j) {
        for (int i = 0; i < n_sys; ++i) {
            std::size_t id = idx(i, j, n_sys);
            std::size_t idp1 = idx(i, j + 1, n_sys);
            D[id] = D[id] - C[id] * D[idp1];
            A[id] = A[id] - C[id] * A[idp1];
            C[id] = -C[id] * C[idp1];
        }
    }

    const int nr_rd = 2;
    for (int i = 0; i < n_sys; ++i) {
        std::size_t i0 = idx(i, 0, n_sys);
        std::size_t i1 = idx(i, 1, n_sys);
        double r = 1.0 / (1.0 - A[i1] * C[i0]);
        D[i0] = r * (D[i0] - C[i0] * D[i1]);
        A[i0] = r * A[i0];
        C[i0] = -r * C[i0] * C[i1];

        // idx = i * n_row + j;
        // Get coefficients from the N-matrix to construct reduced matrix
        // idx(i,j,ncols) = j * ncols + i;
        plan->A_rd[idx(i, 0, n_sys)] = A[i0];
        plan->A_rd[idx(i, 1, n_sys)] = A[idx(i, n_row - 1, n_sys)];
        plan->B_rd[idx(i, 0, n_sys)] = 1.0;
        plan->B_rd[idx(i, 1, n_sys)] = 1.0;
        plan->C_rd[idx(i, 0, n_sys)] = C[i0];
        plan->C_rd[idx(i, 1, n_sys)] = C[idx(i, n_row - 1, n_sys)];
        plan->D_rd[idx(i, 0, n_sys)] = D[i0];
        plan->D_rd[idx(i, 1, n_sys)] = D[idx(i, n_row - 1, n_sys)];
    }
}
   double t2 = MPI_Wtime();
   printf("N자만들기: %8.15f\n", t2 - t1);
   fflush(stdout);

   t1 = MPI_Wtime();
    MPI_Request request[4];
    MPI_Ialltoallw(plan->A_rd.data(), plan->count_send.data(),
                   plan->displ_send.data(), plan->ddtype_FS.data(),
                   plan->A_rt.data(), plan->count_recv.data(),
                   plan->displ_recv.data(), plan->ddtype_BS.data(),
                   plan->ptdma_world, &request[0]);
    MPI_Ialltoallw(plan->B_rd.data(), plan->count_send.data(),
                   plan->displ_send.data(), plan->ddtype_FS.data(),
                   plan->B_rt.data(), plan->count_recv.data(),
                   plan->displ_recv.data(), plan->ddtype_BS.data(),
                   plan->ptdma_world, &request[1]);
    MPI_Ialltoallw(plan->C_rd.data(), plan->count_send.data(),
                   plan->displ_send.data(), plan->ddtype_FS.data(),
                   plan->C_rt.data(), plan->count_recv.data(),
                   plan->displ_recv.data(), plan->ddtype_BS.data(),
                   plan->ptdma_world, &request[2]);
    MPI_Ialltoallw(plan->D_rd.data(), plan->count_send.data(),
                   plan->displ_send.data(), plan->ddtype_FS.data(),
                   plan->D_rt.data(), plan->count_recv.data(),
                   plan->displ_recv.data(), plan->ddtype_BS.data(),
                   plan->ptdma_world, &request[3]);
    MPI_Waitall(4, request, MPI_STATUSES_IGNORE);
   t2 = MPI_Wtime();
    printf("처음 올투올: %8.15f\n", t1 - t2);

   t1 = MPI_Wtime();
    tdma_many(plan->A_rt.data(), plan->B_rt.data(), plan->C_rt.data(),
              plan->D_rt.data(), plan->n_sys_rt, plan->n_row_rt);
   t2 = MPI_Wtime();
    printf("그중 many실제로 푸는 시간: %8.15f\n", t1 - t2);
   t1 = MPI_Wtime();
    MPI_Ialltoallw(plan->D_rt.data(), plan->count_recv.data(),
                   plan->displ_recv.data(), plan->ddtype_BS.data(),
                   plan->D_rd.data(), plan->count_send.data(),
                   plan->displ_send.data(), plan->ddtype_FS.data(),
                   plan->ptdma_world, &request[0]);
    MPI_Waitall(1, request, MPI_STATUSES_IGNORE);
   t2 = MPI_Wtime();
    printf("드번쨰 올투올: %8.15f\n", t1 - t2);
   t1 = MPI_Wtime();
    for (int i = 0; i < n_sys; ++i) {
        D[idx(i, 0, n_sys)] = plan->D_rd[idx(i, 0, n_sys)];
        D[idx(i, n_row - 1, n_sys)] = plan->D_rd[idx(i, 1, n_sys)];
    }
    
    for (int j = 1; j < n_row - 1; ++j) {
        for (int i = 0; i < n_sys; ++i) {
            std::size_t id = idx(i, j, n_sys);
            D[id] = D[id] - A[id] * D[idx(i, 0, n_sys)] -
                     C[id] * D[idx(i, n_row - 1, n_sys)];
        }
    }
   t2 = MPI_Wtime();
   printf("재분배: %8.15f\n", t1 - t2);
}

void PaScaL_TDMA_many_solve(ptdma_plan_many* plan, double* __restrict__ A, double* __restrict__ B,
                             double* __restrict__ C, double* __restrict__ D, int n_sys, int n_row) {
    
    constexpr int stamp_tdma = 3;
    if (plan->nprocs == 1) {
        tdma_many(A, B, C, D, n_sys, n_row);
        return;
    }

// {
//   const int T = omp_get_num_threads();
//   const int t = omp_get_thread_num();
//   const int i_beg = (long long)n_sys * t / T;
//   const int i_end = (long long)n_sys * (t+1) / T;

//   // j=0,1 정규화
//   for (int i = i_beg; i < i_end; ++i) {
//     const size_t i0 = (size_t)i;
//     const size_t i1 = (size_t)n_sys + i;
//     double inv0 = 1.0 / B[i0];
//     A[i0] *= inv0; D[i0] *= inv0; C[i0] *= inv0;

//     double inv1 = 1.0 / B[i1];
//     A[i1] *= inv1; D[i1] *= inv1; C[i1] *= inv1;
//   }

//   // forward sweep: 각 스레드는 자기 i-구간에 대해 j를 혼자 진행 (전역 barrier 없음)
//   for (int j = 2; j < n_row; ++j) {
//     const size_t row   = (size_t)j * n_sys;
//     const size_t rowm1 = (size_t)(j-1) * n_sys;

//     for (int i = i_beg; i < i_end; ++i) {
//       const size_t id   = row   + i;
//       const size_t idm1 = rowm1 + i;
//       const double denom = B[id] - A[id]*C[idm1];
//       const double inv   = 1.0 / denom;
//       D[id] = inv * (D[id] - A[id]*D[idm1]);
//       C[id] = inv *  C[id];
//       A[id] = -inv * A[id] * A[idm1];
//     }
//   }

//   // backward sweep
//   for (int j = n_row - 3; j >= 1; --j) {
//     const size_t row   = (size_t)j   * n_sys;
//     const size_t rowp1 = (size_t)(j+1)* n_sys;

//     for (int i = i_beg; i < i_end; ++i) {
//       const size_t id   = row   + i;
//       const size_t idp1 = rowp1 + i;
//       D[id] -= C[id]*D[idp1];
//       A[id] -= C[id]*A[idp1];
//       C[id]  = -C[id]*C[idp1];
//     }
//   }

//   // 경계 축소 및 rd 작성
//   for (int i = i_beg; i < i_end; ++i) {
//     const size_t i0 = (size_t)i;
//     const size_t i1 = (size_t)(n_row-1)*n_sys + i;
//     const size_t i_mid = (size_t)1*n_sys + i;

//     const double r = 1.0 / (1.0 - A[i_mid]*C[i0]);
//     D[i0] = r * (D[i0] - C[i0]*D[i_mid]);
//     A[i0] = r * A[i0];
//     C[i0] = -r * C[i0] * C[i_mid];

//     // plan->*_rd 채우기
//     plan->A_rd[i0]        = A[i0];
//     plan->A_rd[n_sys+i]   = A[i1];
//     plan->B_rd[i0]        = 1.0;
//     plan->B_rd[n_sys+i]   = 1.0;
//     plan->C_rd[i0]        = C[i0];
//     plan->C_rd[n_sys+i]   = C[i1];
//     plan->D_rd[i0]        = D[i0];
//     plan->D_rd[n_sys+i]   = D[i1];
//   }
// } // parallel 끝
{
    for (int i = 0; i < n_sys; ++i) {
        // idx(i,j,ncols) = j * ncols + i;
        std::size_t i0 = idx(i, 0, n_sys);
        std::size_t i1 = idx(i, 1, n_sys);
        
        A[i0] /= B[i0]; D[i0] /= B[i0]; C[i0] /= B[i0];
        A[i1] /= B[i1]; D[i1] /= B[i1]; C[i1] /= B[i1];
    }

    for (int j = 2; j < n_row; ++j) {
        for (int i = 0; i < n_sys; ++i) {
            std::size_t id = idx(i, j, n_sys);
            std::size_t idm1 = idx(i, j - 1, n_sys);
            double r = 1.0 / (B[id] - A[id] * C[idm1]);
            D[id] = r * (D[id] - A[id] * D[idm1]);
            C[id] = r * C[id];
            A[id] = -r * A[id] * A[idm1];
        }
    }

    for (int j = n_row - 3; j >= 1; --j) {
        for (int i = 0; i < n_sys; ++i) {
            std::size_t id = idx(i, j, n_sys);
            std::size_t idp1 = idx(i, j + 1, n_sys);
            D[id] = D[id] - C[id] * D[idp1];
            A[id] = A[id] - C[id] * A[idp1];
            C[id] = -C[id] * C[idp1];
        }
    }

    const int nr_rd = 2;
    for (int i = 0; i < n_sys; ++i) {
        std::size_t i0 = idx(i, 0, n_sys);
        std::size_t i1 = idx(i, 1, n_sys);
        double r = 1.0 / (1.0 - A[i1] * C[i0]);
        D[i0] = r * (D[i0] - C[i0] * D[i1]);
        A[i0] = r * A[i0];
        C[i0] = -r * C[i0] * C[i1];

        // idx = i * n_row + j;
        // Get coefficients from the N-matrix to construct reduced matrix
        // idx(i,j,ncols) = j * ncols + i;
        plan->A_rd[idx(i, 0, n_sys)] = A[i0];
        plan->A_rd[idx(i, 1, n_sys)] = A[idx(i, n_row - 1, n_sys)];
        plan->B_rd[idx(i, 0, n_sys)] = 1.0;
        plan->B_rd[idx(i, 1, n_sys)] = 1.0;
        plan->C_rd[idx(i, 0, n_sys)] = C[i0];
        plan->C_rd[idx(i, 1, n_sys)] = C[idx(i, n_row - 1, n_sys)];
        plan->D_rd[idx(i, 0, n_sys)] = D[i0];
        plan->D_rd[idx(i, 1, n_sys)] = D[idx(i, n_row - 1, n_sys)];
    }
}

    MPI_Request request[4];
    MPI_Ialltoallw(plan->A_rd.data(), plan->count_send.data(),
                   plan->displ_send.data(), plan->ddtype_FS.data(),
                   plan->A_rt.data(), plan->count_recv.data(),
                   plan->displ_recv.data(), plan->ddtype_BS.data(),
                   plan->ptdma_world, &request[0]);
    MPI_Ialltoallw(plan->B_rd.data(), plan->count_send.data(),
                   plan->displ_send.data(), plan->ddtype_FS.data(),
                   plan->B_rt.data(), plan->count_recv.data(),
                   plan->displ_recv.data(), plan->ddtype_BS.data(),
                   plan->ptdma_world, &request[1]);
    MPI_Ialltoallw(plan->C_rd.data(), plan->count_send.data(),
                   plan->displ_send.data(), plan->ddtype_FS.data(),
                   plan->C_rt.data(), plan->count_recv.data(),
                   plan->displ_recv.data(), plan->ddtype_BS.data(),
                   plan->ptdma_world, &request[2]);
    MPI_Ialltoallw(plan->D_rd.data(), plan->count_send.data(),
                   plan->displ_send.data(), plan->ddtype_FS.data(),
                   plan->D_rt.data(), plan->count_recv.data(),
                   plan->displ_recv.data(), plan->ddtype_BS.data(),
                   plan->ptdma_world, &request[3]);
    MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

   
    tdma_many(plan->A_rt.data(), plan->B_rt.data(), plan->C_rt.data(),
              plan->D_rt.data(), plan->n_sys_rt, plan->n_row_rt);

    MPI_Ialltoallw(plan->D_rt.data(), plan->count_recv.data(),
                   plan->displ_recv.data(), plan->ddtype_BS.data(),
                   plan->D_rd.data(), plan->count_send.data(),
                   plan->displ_send.data(), plan->ddtype_FS.data(),
                   plan->ptdma_world, &request[0]);
    MPI_Waitall(1, request, MPI_STATUSES_IGNORE);

    for (int i = 0; i < n_sys; ++i) {
        D[idx(i, 0, n_sys)] = plan->D_rd[idx(i, 0, n_sys)];
        D[idx(i, n_row - 1, n_sys)] = plan->D_rd[idx(i, 1, n_sys)];
    }
    
    for (int j = 1; j < n_row - 1; ++j) {
        for (int i = 0; i < n_sys; ++i) {
            std::size_t id = idx(i, j, n_sys);
            D[id] = D[id] - A[id] * D[idx(i, 0, n_sys)] -
                     C[id] * D[idx(i, n_row - 1, n_sys)];
        }
    }
}


void PaScaL_TDMA_many_solve_cycle(ptdma_plan_many* plan, double* A, double* B,
                                   double* C, double* D, int n_sys, int n_row) {
    if (plan->nprocs == 1) {
        tdma_cycl_many(A, B, C, D, n_sys, n_row);
        return;
    }

    for (int i = 0; i < n_sys; ++i) {
        std::size_t i0 = idx(i, 0, n_row);
        std::size_t i1 = idx(i, 1, n_row);
        A[i0] /= B[i0]; D[i0] /= B[i0]; C[i0] /= B[i0];
        A[i1] /= B[i1]; D[i1] /= B[i1]; C[i1] /= B[i1];
    }
    for (int j = 2; j < n_row; ++j) {
        for (int i = 0; i < n_sys; ++i) {
            std::size_t id = idx(i, j, n_row);
            std::size_t idm1 = idx(i, j - 1, n_row);
            double r = 1.0 / (B[id] - A[id] * C[idm1]);
            D[id] = r * (D[id] - A[id] * D[idm1]);
            C[id] = r * C[id];
            A[id] = -r * A[id] * A[idm1];
        }
    }
    for (int j = n_row - 3; j >= 1; --j) {
        for (int i = 0; i < n_sys; ++i) {
            std::size_t id = idx(i, j, n_row);
            std::size_t idp1 = idx(i, j + 1, n_row);
            D[id] = D[id] - C[id] * D[idp1];
            A[id] = A[id] - C[id] * A[idp1];
            C[id] = -C[id] * C[idp1];
        }
    }
    const int nr_rd = 2;
    for (int i = 0; i < n_sys; ++i) {
        std::size_t i0 = idx(i, 0, n_row);
        std::size_t i1 = idx(i, 1, n_row);
        double r = 1.0 / (1.0 - A[i1] * C[i0]);
        D[i0] = r * (D[i0] - C[i0] * D[i1]);
        A[i0] = r * A[i0];
        C[i0] = -r * C[i0] * C[i1];

        plan->A_rd[idx(i, 0, nr_rd)] = A[i0];
        plan->A_rd[idx(i, 1, nr_rd)] = A[idx(i, n_row - 1, n_row)];
        plan->B_rd[idx(i, 0, nr_rd)] = 1.0;
        plan->B_rd[idx(i, 1, nr_rd)] = 1.0;
        plan->C_rd[idx(i, 0, nr_rd)] = C[i0];
        plan->C_rd[idx(i, 1, nr_rd)] = C[idx(i, n_row - 1, n_row)];
        plan->D_rd[idx(i, 0, nr_rd)] = D[i0];
        plan->D_rd[idx(i, 1, nr_rd)] = D[idx(i, n_row - 1, n_row)];
    }

    MPI_Request request[4];
    MPI_Ialltoallw(plan->A_rd.data(), plan->count_send.data(),
                   plan->displ_send.data(), plan->ddtype_FS.data(),
                   plan->A_rt.data(), plan->count_recv.data(),
                   plan->displ_recv.data(), plan->ddtype_BS.data(),
                   plan->ptdma_world, &request[0]);
    MPI_Ialltoallw(plan->B_rd.data(), plan->count_send.data(),
                   plan->displ_send.data(), plan->ddtype_FS.data(),
                   plan->B_rt.data(), plan->count_recv.data(),
                   plan->displ_recv.data(), plan->ddtype_BS.data(),
                   plan->ptdma_world, &request[1]);
    MPI_Ialltoallw(plan->C_rd.data(), plan->count_send.data(),
                   plan->displ_send.data(), plan->ddtype_FS.data(),
                   plan->C_rt.data(), plan->count_recv.data(),
                   plan->displ_recv.data(), plan->ddtype_BS.data(),
                   plan->ptdma_world, &request[2]);
    MPI_Ialltoallw(plan->D_rd.data(), plan->count_send.data(),
                   plan->displ_send.data(), plan->ddtype_FS.data(),
                   plan->D_rt.data(), plan->count_recv.data(),
                   plan->displ_recv.data(), plan->ddtype_BS.data(),
                   plan->ptdma_world, &request[3]);
    MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

    tdma_cycl_many(plan->A_rt.data(), plan->B_rt.data(), plan->C_rt.data(),
                   plan->D_rt.data(), plan->n_sys_rt, plan->n_row_rt);

    MPI_Ialltoallw(plan->D_rt.data(), plan->count_recv.data(),
                   plan->displ_recv.data(), plan->ddtype_BS.data(),
                   plan->D_rd.data(), plan->count_send.data(),
                   plan->displ_send.data(), plan->ddtype_FS.data(),
                   plan->ptdma_world, &request[0]);
    MPI_Waitall(1, request, MPI_STATUSES_IGNORE);

    for (int i = 0; i < n_sys; ++i) {
        D[idx(i, 0, n_row)] = plan->D_rd[idx(i, 0, nr_rd)];
        D[idx(i, n_row - 1, n_row)] = plan->D_rd[idx(i, 1, nr_rd)];
    }
    for (int j = 1; j < n_row - 1; ++j) {
        for (int i = 0; i < n_sys; ++i) {
            std::size_t id = idx(i, j, n_row);
            D[id] = D[id] - A[id] * D[idx(i, 0, n_row)] -
                     C[id] * D[idx(i, n_row - 1, n_row)];
        }
    }
}

} // extern "C"