//======================================================================================================================
//> @file        module_mpi_subdomain.f90
//> @brief       This file contains a module of subdomains for PaScaL_TCS.
//> @details     The module includes the informations on the partitioned domains and corresponding derived datatype for
//>              MPI communication.
//> @author      
//>              - Kiha Kim (k-kiha@yonsei.ac.kr), Department of Computational Science & Engineering, Yonsei University
//>              - Ji-Hoon Kang (jhkang@kisti.re.kr), Korea Institute of Science and Technology Information
//>              - Jung-Il Choi (jic@yonsei.ac.kr), Department of Computational Science & Engineering, Yonsei University
//>
//> @date        October 2022
//> @version     1.0
//> @par         Copyright
//>              Copyright (c) 2022 Kiha Kim and Jung-Il choi, Yonsei University and 
//>              Ji-Hoon Kang, Korea Institute of Science and Technology Information, All rights reserved.
//> @par         License     
//>              This project is release under the terms of the MIT License (see LICENSE in )
//======================================================================================================================

//>
//> @brief       Module for building subdomains from the physical domain.
//> @details     This module has simulation parameters for subdomains and communication between the subdomains.
//>

#include "modules.hpp"
#include "pascal_tdma.hpp"
#include <algorithm>
#include <cmath>
#include <fftw3.h>
#include <vector>
#include <complex>

namespace mpi_subdomain {

int n1sub = 0, n2sub = 0, n3sub = 0;
int n1msub = 0, n2msub = 0, n3msub = 0;
int ista = 0, iend = 0, jsta = 0, jend = 0, ksta = 0, kend = 0;

std::vector<double> x1_sub, x2_sub, x3_sub, x1_sub_half, x2_sub_half, x3_sub_half;
std::vector<double> PRHS;
std::vector<double> dx1_sub, dx2_sub, dx3_sub;
std::vector<double> dmx1_sub, dmx2_sub, dmx3_sub;
std::vector<double> fft_amj, fft_apj, fft_acj;

std::vector<int> iC_BC, iS_BC, jC_BC, jS_BC, kC_BC, kS_BC;
int i_indexS = 0, j_indexS = 0, k_indexS = 0;

MPI_Datatype ddtype_sendto_E       = MPI_DATATYPE_NULL;
MPI_Datatype ddtype_recvfrom_W     = MPI_DATATYPE_NULL;
MPI_Datatype ddtype_sendto_W       = MPI_DATATYPE_NULL;
MPI_Datatype ddtype_recvfrom_E     = MPI_DATATYPE_NULL;
MPI_Datatype ddtype_sendto_N       = MPI_DATATYPE_NULL;
MPI_Datatype ddtype_recvfrom_S     = MPI_DATATYPE_NULL;
MPI_Datatype ddtype_sendto_S       = MPI_DATATYPE_NULL;
MPI_Datatype ddtype_recvfrom_N     = MPI_DATATYPE_NULL;
MPI_Datatype ddtype_sendto_F       = MPI_DATATYPE_NULL;
MPI_Datatype ddtype_recvfrom_B     = MPI_DATATYPE_NULL;
MPI_Datatype ddtype_sendto_B       = MPI_DATATYPE_NULL;
MPI_Datatype ddtype_recvfrom_F     = MPI_DATATYPE_NULL;

int h1p = 0, h3p = 0;
int h1pKsub = 0, n2mIsub = 0, n2mKsub = 0, h1pKsub_ista = 0, h1pKsub_iend = 0,
    n2mKsub_jsta = 0, n2mKsub_jend = 0;
int n3msub_Isub = 0, h1psub = 0, h1psub_Ksub = 0, h1psub_Ksub_ista = 0,
    h1psub_Ksub_iend = 0;

std::vector<MPI_Datatype> ddtype_dble_C_in_C2I, ddtype_dble_I_in_C2I;
std::vector<MPI_Datatype> ddtype_cplx_C_in_C2I, ddtype_cplx_I_in_C2I;
std::vector<MPI_Datatype> ddtype_cplx_I_in_I2K, ddtype_cplx_K_in_I2K;
std::vector<MPI_Datatype> ddtype_cplx_C_in_C2K, ddtype_cplx_K_in_C2K;


std::vector<double> Am_r, Ac_r, Ap_r, Be_r;
std::vector<double> Am_c, Ac_c, Ap_c, Be_c;

std::vector<double> Am_r_modified, Ac_r_modified, Ap_r_modified, Be_r_modified;
std::vector<double> Am_c_modified, Ac_c_modified, Ap_c_modified, Be_c_modified;

std::vector<double> buffer_dp1, buffer_dp2;
std::vector<std::complex<double>> buffer_cd1, buffer_cd2;

std::vector<std::complex<double>> RHSIKhat_Kline;
std::vector<std::complex<double>> RHSIhat_Kline2;

fftw_plan plan_fft_x;
fftw_plan plan_fft_z;
fftw_plan plan_ifft_x;
fftw_plan plan_ifft_z;

ptdma_plan_many plan_tdma, plan_tdma_modified;

std::vector<int> countsendI, countdistI, countsendK, countdistK;

void subdomain_para_range(int nsta, int nend, int nprocs, int myrank,
                          int &index_sta, int &index_end) {
    int iwork1 = (nend - nsta + 1) / nprocs;
    int iwork2 = (nend - nsta + 1) % nprocs;
    index_sta = myrank * iwork1 + nsta + std::min(myrank, iwork2);
    index_end = index_sta + iwork1 - 1;
    if (iwork2 > myrank) index_end++;
}

void mpi_subdomain_make() {
    using namespace mpi_topology;
    using namespace global;

    subdomain_para_range(1, n1 - 1, comm_1d_x1.nprocs, comm_1d_x1.myrank, ista, iend);
    n1sub = iend - ista + 2;
    subdomain_para_range(1, n2 - 1, comm_1d_x2.nprocs, comm_1d_x2.myrank, jsta, jend);
    n2sub = jend - jsta + 2;
    subdomain_para_range(1, n3 - 1, comm_1d_x3.nprocs, comm_1d_x3.myrank, ksta, kend);
    n3sub = kend - ksta + 2;

    n1msub = n1sub - 1;
    n2msub = n2sub - 1;
    n3msub = n3sub - 1;


    x1_sub.assign(n1sub + 1, 0.0); dmx1_sub.assign(n1sub + 1, 0.0); dx1_sub.assign(n1sub + 1, 0.0);
    x2_sub.assign(n2sub + 1, 0.0); dmx2_sub.assign(n2sub + 1, 0.0); dx2_sub.assign(n2sub + 1, 0.0);
    x3_sub.assign(n3sub + 1, 0.0); dmx3_sub.assign(n3sub + 1, 0.0); dx3_sub.assign(n3sub + 1, 0.0);
    x1_sub_half.assign(n1sub + 1, 0.0);
    x2_sub_half.assign(n2sub + 1, 0.0);
    x3_sub_half.assign(n3sub + 1, 0.0);

    fft_apj.assign(n2msub, 0.0);
    fft_acj.assign(n2msub, 0.0);
    fft_amj.assign(n2msub, 0.0);

    PRHS.assign(n1msub * n2msub * n3msub, 0.0);

}

void mpi_subdomain_clean() {
    x1_sub.clear(); dmx1_sub.clear(); dx1_sub.clear();
    x2_sub.clear(); dmx2_sub.clear(); dx2_sub.clear();
    x3_sub.clear(); dmx3_sub.clear(); dx3_sub.clear();
    x1_sub_half.clear(); x2_sub_half.clear(); x3_sub_half.clear();

    iC_BC.clear(); iS_BC.clear();
    jC_BC.clear(); jS_BC.clear();
    kC_BC.clear(); kS_BC.clear();

    MPI_Type_free(&ddtype_sendto_E); MPI_Type_free(&ddtype_recvfrom_W);
    MPI_Type_free(&ddtype_sendto_W); MPI_Type_free(&ddtype_recvfrom_E);
    MPI_Type_free(&ddtype_sendto_N); MPI_Type_free(&ddtype_recvfrom_S);
    MPI_Type_free(&ddtype_sendto_S); MPI_Type_free(&ddtype_recvfrom_N);
    MPI_Type_free(&ddtype_sendto_F); MPI_Type_free(&ddtype_recvfrom_B);
    MPI_Type_free(&ddtype_sendto_B); MPI_Type_free(&ddtype_recvfrom_F);

    for (auto &dt : ddtype_dble_C_in_C2I) MPI_Type_free(&dt);
    for (auto &dt : ddtype_dble_I_in_C2I) MPI_Type_free(&dt);
    for (auto &dt : ddtype_cplx_C_in_C2I) MPI_Type_free(&dt);
    for (auto &dt : ddtype_cplx_I_in_C2I) MPI_Type_free(&dt);
    for (auto &dt : ddtype_cplx_I_in_I2K) MPI_Type_free(&dt);
    for (auto &dt : ddtype_cplx_K_in_I2K) MPI_Type_free(&dt);
    for (auto &dt : ddtype_cplx_C_in_C2K) MPI_Type_free(&dt);
    for (auto &dt : ddtype_cplx_K_in_C2K) MPI_Type_free(&dt);

    ddtype_dble_C_in_C2I.clear(); ddtype_dble_I_in_C2I.clear();
    ddtype_cplx_C_in_C2I.clear(); ddtype_cplx_I_in_C2I.clear();
    ddtype_cplx_I_in_I2K.clear();  ddtype_cplx_K_in_I2K.clear();
    ddtype_cplx_C_in_C2K.clear();  ddtype_cplx_K_in_C2K.clear();
    countsendI.clear(); countdistI.clear();
    countsendK.clear(); countdistK.clear();
}

void mpi_subdomain_mesh() {
    using namespace mpi_topology;
    using namespace global;

    std::fill(x1_sub.begin(), x1_sub.end(), 0.0);
    std::fill(dmx1_sub.begin(), dmx1_sub.end(), 0.0);
    std::fill(dx1_sub.begin(), dx1_sub.end(), 0.0);
    std::fill(x2_sub.begin(), x2_sub.end(), 0.0);
    std::fill(dmx2_sub.begin(), dmx2_sub.end(), 0.0);
    std::fill(dx2_sub.begin(), dx2_sub.end(), 0.0);
    std::fill(x3_sub.begin(), x3_sub.end(), 0.0);
    std::fill(dmx3_sub.begin(), dmx3_sub.end(), 0.0);
    std::fill(dx3_sub.begin(), dx3_sub.end(), 0.0);


    std::fill(x1_sub_half.begin(), x1_sub_half.end(), 0.0);
    std::fill(x2_sub_half.begin(), x2_sub_half.end(), 0.0);
    std::fill(x3_sub_half.begin(), x3_sub_half.end(), 0.0);
    std::fill(PRHS.begin(), PRHS.end(), 0.0);
    std::fill(fft_amj.begin(), fft_amj.end(), 0.0);
    std::fill(fft_apj.begin(), fft_apj.end(), 0.0);
    std::fill(fft_acj.begin(), fft_acj.end(), 0.0);


    // X-direction
    for (int i = ista - 1; i <= iend + 1; ++i) {
        if (UNIFORM1 == 1)
            x1_sub[i - ista + 1] = static_cast<double>(i - 1) * L1 / static_cast<double>(n1m) + x1_start;
        else
            x1_sub[i - ista + 1] = L1 * 0.5 * (1.0 + std::tanh(0.5 * GAMMA1 * (2.0 * static_cast<double>(i - 1) / static_cast<double>(n1m) - 1.0)) / std::tanh(GAMMA1 * 0.5)) + x1_start;
    }
    if (!pbc1 && comm_1d_x1.myrank == 0) x1_sub[0] = x1_sub[1];
    for (int i = 1; i <= n1msub; ++i) dx1_sub[i] = x1_sub[i + 1] - x1_sub[i];

    MPI_Request request_S2E, request_S2W; MPI_Status status;
    MPI_Isend(&dx1_sub[n1msub], 1, MPI_DOUBLE, comm_1d_x1.east_rank, 111, comm_1d_x1.mpi_comm, &request_S2E);
    MPI_Irecv(&dx1_sub[0],     1, MPI_DOUBLE, comm_1d_x1.west_rank, 111, comm_1d_x1.mpi_comm, &request_S2E);
    MPI_Wait(&request_S2E, &status);
    MPI_Isend(&dx1_sub[1],     1, MPI_DOUBLE, comm_1d_x1.west_rank, 111, comm_1d_x1.mpi_comm, &request_S2W);
    MPI_Irecv(&dx1_sub[n1sub], 1, MPI_DOUBLE, comm_1d_x1.east_rank, 111, comm_1d_x1.mpi_comm, &request_S2W);
    MPI_Wait(&request_S2W, &status);
    if (!pbc1 && comm_1d_x1.myrank == 0) dx1_sub[0] = 0.0;
    if (!pbc1 && comm_1d_x1.myrank == comm_1d_x1.nprocs - 1) dx1_sub[n1sub] = 0.0;

    for (int i = 1; i <= n1msub; ++i) dmx1_sub[i] = 0.5 * (dx1_sub[i - 1] + dx1_sub[i]);
    MPI_Isend(&dmx1_sub[n1msub], 1, MPI_DOUBLE, comm_1d_x1.east_rank, 111, comm_1d_x1.mpi_comm, &request_S2E);
    MPI_Irecv(&dmx1_sub[0],     1, MPI_DOUBLE, comm_1d_x1.west_rank, 111, comm_1d_x1.mpi_comm, &request_S2E);
    MPI_Wait(&request_S2E, &status);
    MPI_Isend(&dmx1_sub[1],     1, MPI_DOUBLE, comm_1d_x1.west_rank, 111, comm_1d_x1.mpi_comm, &request_S2W);
    MPI_Irecv(&dmx1_sub[n1sub], 1, MPI_DOUBLE, comm_1d_x1.east_rank, 111, comm_1d_x1.mpi_comm, &request_S2W);
    MPI_Wait(&request_S2W, &status);
    if (!pbc1 && comm_1d_x1.myrank == comm_1d_x1.nprocs - 1)
        dmx1_sub[n1sub] = 0.5 * (dx1_sub[n1msub] + dx1_sub[n1sub]);

    // Y-direction
    for (int j = jsta - 1; j <= jend + 1; ++j) {
        if (UNIFORM2 == 1)
            x2_sub[j - jsta + 1] = static_cast<double>(j - 1) * L2 / static_cast<double>(n2m) + x2_start;
        else
            x2_sub[j - jsta + 1] = L2 * 0.5 * (1.0 + std::tanh(0.5 * GAMMA2 * (2.0 * static_cast<double>(j - 1) / static_cast<double>(n2m) - 1.0)) / std::tanh(GAMMA2 * 0.5)) + x2_start;
    }
    if (!pbc2 && comm_1d_x2.myrank == 0) x2_sub[0] = x2_sub[1];
    for (int j = 1; j <= n2msub; ++j) dx2_sub[j] = x2_sub[j + 1] - x2_sub[j];
    MPI_Isend(&dx2_sub[n2msub], 1, MPI_DOUBLE, comm_1d_x2.east_rank, 111, comm_1d_x2.mpi_comm, &request_S2E);
    MPI_Irecv(&dx2_sub[0],     1, MPI_DOUBLE, comm_1d_x2.west_rank, 111, comm_1d_x2.mpi_comm, &request_S2E);
    MPI_Wait(&request_S2E, &status);
    MPI_Isend(&dx2_sub[1],     1, MPI_DOUBLE, comm_1d_x2.west_rank, 111, comm_1d_x2.mpi_comm, &request_S2W);
    MPI_Irecv(&dx2_sub[n2sub], 1, MPI_DOUBLE, comm_1d_x2.east_rank, 111, comm_1d_x2.mpi_comm, &request_S2W);
    MPI_Wait(&request_S2W, &status);
    if (!pbc2 && comm_1d_x2.myrank == 0) dx2_sub[0] = 0.0;
    if (!pbc2 && comm_1d_x2.myrank == comm_1d_x2.nprocs - 1) dx2_sub[n2sub] = 0.0;
    for (int j = 1; j <= n2msub; ++j) dmx2_sub[j] = 0.5 * (dx2_sub[j - 1] + dx2_sub[j]);
    MPI_Isend(&dmx2_sub[n2msub], 1, MPI_DOUBLE, comm_1d_x2.east_rank, 111, comm_1d_x2.mpi_comm, &request_S2E);
    MPI_Irecv(&dmx2_sub[0],     1, MPI_DOUBLE, comm_1d_x2.west_rank, 111, comm_1d_x2.mpi_comm, &request_S2E);
    MPI_Wait(&request_S2E, &status);
    MPI_Isend(&dmx2_sub[1],     1, MPI_DOUBLE, comm_1d_x2.west_rank, 111, comm_1d_x2.mpi_comm, &request_S2W);
    MPI_Irecv(&dmx2_sub[n2sub], 1, MPI_DOUBLE, comm_1d_x2.east_rank, 111, comm_1d_x2.mpi_comm, &request_S2W);
    MPI_Wait(&request_S2W, &status);
    if (!pbc2 && comm_1d_x2.myrank == comm_1d_x2.nprocs - 1)
        dmx2_sub[n2sub] = 0.5 * (dx2_sub[n2msub] + dx2_sub[n2sub]);

    // Z-direction
    for (int k = ksta - 1; k <= kend + 1; ++k) {
        if (UNIFORM3 == 1)
            x3_sub[k - ksta + 1] = static_cast<double>(k - 1) * L3 / static_cast<double>(n3m) + x3_start;
        else
            x3_sub[k - ksta + 1] = L3 * 0.5 * (1.0 + std::tanh(0.5 * GAMMA3 * (2.0 * static_cast<double>(k - 1) / static_cast<double>(n3m) - 1.0)) / std::tanh(GAMMA3 * 0.5)) + x3_start;
    }
    if (!pbc3 && comm_1d_x3.myrank == 0) x3_sub[0] = x3_sub[1];
    for (int k = 1; k <= n3msub; ++k) dx3_sub[k] = x3_sub[k + 1] - x3_sub[k];
    MPI_Isend(&dx3_sub[n3msub], 1, MPI_DOUBLE, comm_1d_x3.east_rank, 111, comm_1d_x3.mpi_comm, &request_S2E);
    MPI_Irecv(&dx3_sub[0],     1, MPI_DOUBLE, comm_1d_x3.west_rank, 111, comm_1d_x3.mpi_comm, &request_S2E);
    MPI_Wait(&request_S2E, &status);
    MPI_Isend(&dx3_sub[1],     1, MPI_DOUBLE, comm_1d_x3.west_rank, 111, comm_1d_x3.mpi_comm, &request_S2W);
    MPI_Irecv(&dx3_sub[n3sub], 1, MPI_DOUBLE, comm_1d_x3.east_rank, 111, comm_1d_x3.mpi_comm, &request_S2W);
    MPI_Wait(&request_S2W, &status);
    if (!pbc3 && comm_1d_x3.myrank == 0) dx3_sub[0] = 0.0;
    if (!pbc3 && comm_1d_x3.myrank == comm_1d_x3.nprocs - 1) dx3_sub[n3sub] = 0.0;
    for (int k = 1; k <= n3msub; ++k) dmx3_sub[k] = 0.5 * (dx3_sub[k - 1] + dx3_sub[k]);
    MPI_Isend(&dmx3_sub[n3msub], 1, MPI_DOUBLE, comm_1d_x3.east_rank, 111, comm_1d_x3.mpi_comm, &request_S2E);
    MPI_Irecv(&dmx3_sub[0],     1, MPI_DOUBLE, comm_1d_x3.west_rank, 111, comm_1d_x3.mpi_comm, &request_S2E);
    MPI_Wait(&request_S2E, &status);
    MPI_Isend(&dmx3_sub[1],     1, MPI_DOUBLE, comm_1d_x3.west_rank, 111, comm_1d_x3.mpi_comm, &request_S2W);
    MPI_Irecv(&dmx3_sub[n3sub], 1, MPI_DOUBLE, comm_1d_x3.east_rank, 111, comm_1d_x3.mpi_comm, &request_S2W);
    MPI_Wait(&request_S2W, &status);
    if (!pbc3 && comm_1d_x3.myrank == comm_1d_x3.nprocs - 1)
        dmx3_sub[n3sub] = 0.5 * (dx3_sub[n3msub] + dx3_sub[n3sub]);


    for (int k = 1; k < n3msub+1;++k) {
        x3_sub_half[k] = 0.5 * (x3_sub[k] + x3_sub[k + 1]);
    }
    for (int j = 1; j < n2msub+1; ++j) {
        x2_sub_half[j] = 0.5 * (x2_sub[j] + x2_sub[j + 1]);
    }
    for (int i = 1; i < n1msub+1;++i) {
        x1_sub_half[i] = 0.5 * (x1_sub[i] + x1_sub[i + 1]);
    }
    for (int j = 0; j < n2msub; ++j) {
        fft_amj[j] = 1.0 / dx2_sub[j + 1] / dx2_sub[j + 1];
        fft_apj[j] = 1.0 / dx2_sub[j + 1] / dx2_sub[j + 1];
        fft_acj[j] = -fft_amj[j] - fft_apj[j];
    }

    // printf("Rank : %8d\n", mpi_topology::myrank);
    //     for (int i = 0; i < x1_sub.size(); i++) {
    //         printf("x1_sub[%8d] : %8.15f\n", i, x1_sub[i]);
    //     }
    //     for (int i = 0; i < x2_sub.size(); i++) {
    //         printf("x2_sub[%8d] : %8.15f\n", i, x2_sub[i]);
    //     }
    //     for (int i = 0; i < x3_sub.size(); i++) {
    //         printf("x3_sub[%8d] : %8.15f\n", i, x3_sub[i]);
    //     }
}

void mpi_subdomain_indices() {
    using namespace mpi_topology;
    using namespace global;

    iC_BC.assign(n1sub + 1, 1); iS_BC.assign(n1sub + 1, 1);
    jC_BC.assign(n2sub + 1, 1); jS_BC.assign(n2sub + 1, 1);
    kC_BC.assign(n3sub + 1, 1); kS_BC.assign(n3sub + 1, 1);

    i_indexS = j_indexS = k_indexS = 1;

    if (!pbc1) {
        if (comm_1d_x1.myrank == 0) {
            i_indexS = 2; iC_BC[0] = 0; iS_BC[0] = 0; iS_BC[1] = 0;
        }
        if (comm_1d_x1.myrank == comm_1d_x1.nprocs - 1) {
            iC_BC[n1sub] = 0; iS_BC[n1sub] = 0;
        }
    }
    if (!pbc2) {
        if (comm_1d_x2.myrank == 0) {
            j_indexS = 2; jC_BC[0] = 0; jS_BC[0] = 0; jS_BC[1] = 0;
        }
        if (comm_1d_x2.myrank == comm_1d_x2.nprocs - 1) {
            jC_BC[n2sub] = 0; jS_BC[n2sub] = 0;
        }
    }
    if (!pbc3) {
        if (comm_1d_x3.myrank == 0) {
            k_indexS = 2; kC_BC[0] = 0; kS_BC[0] = 0; kS_BC[1] = 0;
        }
        if (comm_1d_x3.myrank == comm_1d_x3.nprocs - 1) {
            kC_BC[n3sub] = 0; kS_BC[n3sub] = 0;
        }
    }
}

void mpi_subdomain_indices_clean() {
    iC_BC.clear(); iS_BC.clear();
    jC_BC.clear(); jS_BC.clear();
    kC_BC.clear(); kS_BC.clear();
}

void mpi_subdomain_DDT_ghostcell() {
    int sizes[3], subsizes[3], starts[3];

    sizes[0] = n3sub + 1; sizes[1] = n2sub + 1; sizes[2] = n1sub + 1;

    // East/West faces
    subsizes[0] = n3sub + 1; subsizes[1] = n2sub + 1; subsizes[2] = 1;
    starts[0] = 0; starts[1] = 0; starts[2] = n1sub - 1;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_sendto_E);
    MPI_Type_commit(&ddtype_sendto_E);

    starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_recvfrom_W);
    MPI_Type_commit(&ddtype_recvfrom_W);

    starts[2] = 1;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_sendto_W);
    MPI_Type_commit(&ddtype_sendto_W);

    starts[2] = n1sub;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_recvfrom_E);
    MPI_Type_commit(&ddtype_recvfrom_E);

    // North/South faces
    subsizes[0] = n3sub + 1; subsizes[1] = 1; subsizes[2] = n1sub + 1;
    starts[0] = 0; starts[1] = n2sub - 1; starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_sendto_N);
    MPI_Type_commit(&ddtype_sendto_N);

    starts[1] = 0;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_recvfrom_S);
    MPI_Type_commit(&ddtype_recvfrom_S);

    starts[1] = 1;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_sendto_S);
    MPI_Type_commit(&ddtype_sendto_S);

    starts[1] = n2sub;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_recvfrom_N);
    MPI_Type_commit(&ddtype_recvfrom_N);

    // Front/Back faces
    subsizes[0] = 1; subsizes[1] = n2sub + 1; subsizes[2] = n1sub + 1;
    starts[0] = n3sub - 1; starts[1] = 0; starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_sendto_F);
    MPI_Type_commit(&ddtype_sendto_F);

    starts[0] = 0;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_recvfrom_B);
    MPI_Type_commit(&ddtype_recvfrom_B);

    starts[0] = 1;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_sendto_B);
    MPI_Type_commit(&ddtype_sendto_B);

    starts[0] = n3sub;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_recvfrom_F);
    MPI_Type_commit(&ddtype_recvfrom_F);
}

void mpi_subdomain_DDT_transpose1() {
    using namespace mpi_topology;
    using namespace global;

    ddtype_dble_C_in_C2I.resize(comm_1d_x1.nprocs);
    ddtype_dble_I_in_C2I.resize(comm_1d_x1.nprocs);
    ddtype_cplx_I_in_I2K.resize(comm_1d_x3.nprocs);
    ddtype_cplx_K_in_I2K.resize(comm_1d_x3.nprocs);

    countsendI.assign(comm_1d_x1.nprocs, 1);
    countdistI.assign(comm_1d_x1.nprocs, 0);
    countsendK.assign(comm_1d_x3.nprocs, 1);
    countdistK.assign(comm_1d_x3.nprocs, 0);

    h1p = n1m / 2 + 1;
    h3p = n3m / 2 + 1;

    subdomain_para_range(1, h1p, comm_1d_x3.nprocs, comm_1d_x3.myrank,
                         h1pKsub_ista, h1pKsub_iend);
    h1pKsub = h1pKsub_iend - h1pKsub_ista + 1;

    int indexA, indexB;
    subdomain_para_range(1, n2msub, comm_1d_x1.nprocs, comm_1d_x1.myrank,
                         indexA, indexB);
    n2mIsub = indexB - indexA + 1;
    n2mKsub = n2mIsub;

    std::vector<int> n1msubAll(comm_1d_x1.nprocs);
    std::vector<int> n3msubAll(comm_1d_x3.nprocs);
    std::vector<int> h1pKsubAll(comm_1d_x3.nprocs);
    std::vector<int> n2mIsubAll(comm_1d_x1.nprocs);

    for (int i = 0; i < comm_1d_x1.nprocs; ++i) {
        subdomain_para_range(1, n1m, comm_1d_x1.nprocs, i, indexA, indexB);
        n1msubAll[i] = indexB - indexA + 1;
        subdomain_para_range(1, n2msub, comm_1d_x1.nprocs, i, indexA, indexB);
        n2mIsubAll[i] = indexB - indexA + 1;
    }

    for (int i = 0; i < comm_1d_x3.nprocs; ++i) {
        subdomain_para_range(1, n3m, comm_1d_x3.nprocs, i, indexA, indexB);
        n3msubAll[i] = indexB - indexA + 1;
        subdomain_para_range(1, h1p, comm_1d_x3.nprocs, i, indexA, indexB);
        h1pKsubAll[i] = indexB - indexA + 1;
    }

    subdomain_para_range(1, n2msub, comm_1d_x1.nprocs, comm_1d_x1.myrank,
                         n2mKsub_jsta, n2mKsub_jend);

    int bigsize[3], subsize[3], start[3];

    int offsetJ = 0;
    int offsetI = 0;
    for (int i = 0; i < comm_1d_x1.nprocs; ++i) {
        // C -> I transpose for real data
        bigsize[0] = n3msub;            bigsize[1] = n2msub;          bigsize[2] = n1msub;
        subsize[0] = n3msub;            subsize[1] = n2mIsubAll[i];   subsize[2] = n1msub;
        start[0]  = 0;                  start[1]  = offsetJ;          start[2]  = 0;
        MPI_Type_create_subarray(3, bigsize, subsize, start, MPI_ORDER_C,
                                 MPI_DOUBLE, &ddtype_dble_C_in_C2I[i]);
        MPI_Type_commit(&ddtype_dble_C_in_C2I[i]);

        bigsize[0] = n3msub;            bigsize[1] = n2mIsub;         bigsize[2] = n1m;
        subsize[0] = n3msub;            subsize[1] = n2mIsub;         subsize[2] = n1msubAll[i];
        start[0]  = 0;                  start[1]  = 0;                start[2]  = offsetI;
        MPI_Type_create_subarray(3, bigsize, subsize, start, MPI_ORDER_C,
                                 MPI_DOUBLE, &ddtype_dble_I_in_C2I[i]);
        MPI_Type_commit(&ddtype_dble_I_in_C2I[i]);

        offsetJ += n2mIsubAll[i];
        offsetI += n1msubAll[i];
    }

    int offsetH = 0;
    int offsetK = 0;
    for (int k = 0; k < comm_1d_x3.nprocs; ++k) {
        // I -> K transpose for complex data
        bigsize[0] = n3msub;            bigsize[1] = n2mIsub;         bigsize[2] = h1p;
        subsize[0] = n3msub;            subsize[1] = n2mIsub;         subsize[2] = h1pKsubAll[k];
        start[0]  = 0;                  start[1]  = 0;                start[2]  = offsetH;
        MPI_Type_create_subarray(3, bigsize, subsize, start, MPI_ORDER_C,
                                 MPI_DOUBLE_COMPLEX, &ddtype_cplx_I_in_I2K[k]);
        MPI_Type_commit(&ddtype_cplx_I_in_I2K[k]);

        bigsize[0] = n3m;               bigsize[1] = n2mIsub;         bigsize[2] = h1pKsub;
        subsize[0] = n3msubAll[k];      subsize[1] = n2mIsub;         subsize[2] = h1pKsub;
        start[0]  = offsetK;            start[1]  = 0;                start[2]  = 0;
        MPI_Type_create_subarray(3, bigsize, subsize, start, MPI_ORDER_C,
                                 MPI_DOUBLE_COMPLEX, &ddtype_cplx_K_in_I2K[k]);
        MPI_Type_commit(&ddtype_cplx_K_in_I2K[k]);

        offsetH += h1pKsubAll[k];
        offsetK += n3msubAll[k];
    }
}

void mpi_subdomain_DDT_transpose2() {
    using namespace mpi_topology;
    using namespace global;

    ddtype_dble_C_in_C2I.resize(comm_1d_x1.nprocs);
    ddtype_dble_I_in_C2I.resize(comm_1d_x1.nprocs);
    ddtype_cplx_C_in_C2I.resize(comm_1d_x1.nprocs);
    ddtype_cplx_I_in_C2I.resize(comm_1d_x1.nprocs);
    ddtype_cplx_C_in_C2K.resize(comm_1d_x3.nprocs);
    ddtype_cplx_K_in_C2K.resize(comm_1d_x3.nprocs);

    countsendI.assign(comm_1d_x1.nprocs, 1);
    countdistI.assign(comm_1d_x1.nprocs, 0);
    countsendK.assign(comm_1d_x3.nprocs, 1);
    countdistK.assign(comm_1d_x3.nprocs, 0);

    int indexA, indexB;
    subdomain_para_range(1, n3msub, comm_1d_x1.nprocs, comm_1d_x1.myrank,
                         indexA, indexB);
    n3msub_Isub = indexB - indexA + 1;

    h1p = n1m / 2 + 1;
    h3p = n3m / 2 + 1;

    subdomain_para_range(1, h1p, comm_1d_x1.nprocs, comm_1d_x1.myrank,
                         indexA, indexB);
    h1psub = indexB - indexA + 1;

    subdomain_para_range(indexA, indexB, comm_1d_x3.nprocs, comm_1d_x3.myrank,
                         h1psub_Ksub_ista, h1psub_Ksub_iend);
    h1psub_Ksub = h1psub_Ksub_iend - h1psub_Ksub_ista + 1;

    std::vector<int> n3msub_IsubAll(comm_1d_x1.nprocs);
    std::vector<int> n1msubAll(comm_1d_x1.nprocs);
    std::vector<int> h1psubAll(comm_1d_x1.nprocs);
    std::vector<int> h1psub_KsubAll(comm_1d_x3.nprocs);
    std::vector<int> n3msubAll(comm_1d_x3.nprocs);

    for (int i = 0; i < comm_1d_x1.nprocs; ++i) {
        subdomain_para_range(1, n3msub, comm_1d_x1.nprocs, i, indexA, indexB);
        n3msub_IsubAll[i] = indexB - indexA + 1;
        subdomain_para_range(1, n1m, comm_1d_x1.nprocs, i, indexA, indexB);
        n1msubAll[i] = indexB - indexA + 1;
        subdomain_para_range(1, h1p, comm_1d_x1.nprocs, i, indexA, indexB);
        h1psubAll[i] = indexB - indexA + 1;
    }

    for (int i = 0; i < comm_1d_x3.nprocs; ++i) {
        subdomain_para_range(1, h1psub, comm_1d_x3.nprocs, i, indexA, indexB);
        h1psub_KsubAll[i] = indexB - indexA + 1;
        subdomain_para_range(1, n3 - 1, comm_1d_x3.nprocs, i, indexA, indexB);
        n3msubAll[i] = indexB - indexA + 1;
    }

    int bigsize[3], subsize[3], start[3];

    int offsetK = 0;
    int offsetI = 0;
    for (int i = 0; i < comm_1d_x1.nprocs; ++i) {
        // C -> I transpose for real data
        bigsize[0] = n3msub;                 bigsize[1] = n2msub;          bigsize[2] = n1msub;
        subsize[0] = n3msub_IsubAll[i];      subsize[1] = n2msub;          subsize[2] = n1msub;
        start[0]  = offsetK;                 start[1]  = 0;                start[2]  = 0;
        MPI_Type_create_subarray(3, bigsize, subsize, start, MPI_ORDER_C,
                                 MPI_DOUBLE, &ddtype_dble_C_in_C2I[i]);
        MPI_Type_commit(&ddtype_dble_C_in_C2I[i]);

        bigsize[0] = n3msub_Isub;            bigsize[1] = n2msub;          bigsize[2] = n1m;
        subsize[0] = n3msub_Isub;            subsize[1] = n2msub;          subsize[2] = n1msubAll[i];
        start[0]  = 0;                       start[1]  = 0;                start[2]  = offsetI;
        MPI_Type_create_subarray(3, bigsize, subsize, start, MPI_ORDER_C,
                                 MPI_DOUBLE, &ddtype_dble_I_in_C2I[i]);
        MPI_Type_commit(&ddtype_dble_I_in_C2I[i]);

        offsetK += n3msub_IsubAll[i];
        offsetI += n1msubAll[i];
    }

    offsetK = 0;
    int offsetH = 0;
    for (int i = 0; i < comm_1d_x1.nprocs; ++i) {
        // C -> I transpose for complex data
        bigsize[0] = n3msub;                 bigsize[1] = n2msub;          bigsize[2] = h1psub;
        subsize[0] = n3msub_IsubAll[i];      subsize[1] = n2msub;          subsize[2] = h1psub;
        start[0]  = offsetK;                 start[1]  = 0;                start[2]  = 0;
        MPI_Type_create_subarray(3, bigsize, subsize, start, MPI_ORDER_C,
                                 MPI_DOUBLE_COMPLEX, &ddtype_cplx_C_in_C2I[i]);
        MPI_Type_commit(&ddtype_cplx_C_in_C2I[i]);

        bigsize[0] = n3msub_Isub;            bigsize[1] = n2msub;          bigsize[2] = h1p;
        subsize[0] = n3msub_Isub;            subsize[1] = n2msub;          subsize[2] = h1psubAll[i];
        start[0]  = 0;                       start[1]  = 0;                start[2]  = offsetH;
        MPI_Type_create_subarray(3, bigsize, subsize, start, MPI_ORDER_C,
                                 MPI_DOUBLE_COMPLEX, &ddtype_cplx_I_in_C2I[i]);
        MPI_Type_commit(&ddtype_cplx_I_in_C2I[i]);

        offsetK += n3msub_IsubAll[i];
        offsetH += h1psubAll[i];
    }

    offsetH = 0;
    offsetK = 0;
    for (int i = 0; i < comm_1d_x3.nprocs; ++i) {
        // I -> K transpose for complex data
        bigsize[0] = n3msub;                 bigsize[1] = n2msub;          bigsize[2] = h1psub;
        subsize[0] = n3msub;                 subsize[1] = n2msub;          subsize[2] = h1psub_KsubAll[i];
        start[0]  = 0;                       start[1]  = 0;                start[2]  = offsetH;
        MPI_Type_create_subarray(3, bigsize, subsize, start, MPI_ORDER_C,
                                 MPI_DOUBLE_COMPLEX, &ddtype_cplx_C_in_C2K[i]);
        MPI_Type_commit(&ddtype_cplx_C_in_C2K[i]);

        bigsize[0] = n3 - 1;                 bigsize[1] = n2msub;          bigsize[2] = h1psub_Ksub;
        subsize[0] = n3msubAll[i];           subsize[1] = n2msub;          subsize[2] = h1psub_Ksub;
        start[0]  = offsetK;                 start[1]  = 0;                start[2]  = 0;
        MPI_Type_create_subarray(3, bigsize, subsize, start, MPI_ORDER_C,
                                 MPI_DOUBLE_COMPLEX, &ddtype_cplx_K_in_C2K[i]);
        MPI_Type_commit(&ddtype_cplx_K_in_C2K[i]);

        offsetH += h1psub_KsubAll[i];
        offsetK += n3msubAll[i];
    }

    std::size_t tdma_size = static_cast<std::size_t>(n3m) *
                            h1psub_Ksub * n2msub;
    std::size_t tdma_size_modified = static_cast<std::size_t>(n3m) * n2msub;
    
    Am_r.resize(tdma_size);  Ac_r.resize(tdma_size);
    Ap_r.resize(tdma_size);  Be_r.resize(tdma_size);
    Am_c.resize(tdma_size);  Ac_c.resize(tdma_size);
    Ap_c.resize(tdma_size);  Be_c.resize(tdma_size);

    Am_r_modified.resize(tdma_size_modified);  Ac_r_modified.resize(tdma_size_modified);
    Ap_r_modified.resize(tdma_size_modified);  Be_r_modified.resize(tdma_size_modified);
    Am_c_modified.resize(tdma_size_modified);  Ac_c_modified.resize(tdma_size_modified);
    Ap_c_modified.resize(tdma_size_modified);  Be_c_modified.resize(tdma_size_modified);

    RHSIKhat_Kline.resize(tdma_size);
    RHSIhat_Kline2.resize(tdma_size);

    int buffer_dp_size =
        std::max(n1m * n2msub * n3msub_Isub, n1msub * n2msub * n3msub);
    int buffer_cd_size = std::max({h1p * n2msub * n3msub_Isub,
                                   h1psub * n2msub * n3msub,
                                   h1psub_Ksub * n2msub * n3m});
    buffer_dp1.resize(buffer_dp_size);
    buffer_dp2.resize(buffer_dp_size);
    buffer_cd1.resize(buffer_cd_size);
    buffer_cd2.resize(buffer_cd_size);

    // Create plan
    int n_fft_x[1] = {n1m};
    plan_fft_x = fftw_plan_many_dft_r2c(
        1, n_fft_x, n2msub * n3msub_Isub, buffer_dp1.data(), n_fft_x, 1, n1m,
        reinterpret_cast<fftw_complex*>(buffer_cd1.data()), n_fft_x, 1, h1p,
        FFTW_ESTIMATE);
    
    int n_fft_z[1] = {n3m};
    plan_fft_z = fftw_plan_many_dft(
        1, n_fft_z, h1psub_Ksub * n2msub,
        reinterpret_cast<fftw_complex*>(buffer_cd1.data()), n_fft_z,
        h1psub_Ksub * n2msub, 1,
        reinterpret_cast<fftw_complex*>(RHSIKhat_Kline.data()), n_fft_z, 1, n3m,
        FFTW_FORWARD, FFTW_ESTIMATE);

    PaScaL_TDMA_plan_many_create(&plan_tdma,
                                 n3m * h1psub_Ksub,          // number of systems
                                 comm_1d_x2.myrank, 
                                 comm_1d_x2.nprocs,
                                 comm_1d_x2.mpi_comm);

    PaScaL_TDMA_plan_many_create(&plan_tdma_modified,
                                 n3m,          // number of systems
                                 comm_1d_x2.myrank, 
                                 comm_1d_x2.nprocs,
                                 comm_1d_x2.mpi_comm);

    plan_ifft_z = fftw_plan_many_dft(
        1, n_fft_z, h1psub_Ksub * n2msub,
        reinterpret_cast<fftw_complex*>(RHSIKhat_Kline.data()), n_fft_z, 1, n3m,
        reinterpret_cast<fftw_complex*>(RHSIhat_Kline2.data()), n_fft_z,
        h1psub_Ksub * n2msub, 1, FFTW_BACKWARD, FFTW_ESTIMATE);

    plan_ifft_x = fftw_plan_many_dft_c2r(
        1, n_fft_x, n2msub * n3msub_Isub,
        reinterpret_cast<fftw_complex*>(buffer_cd2.data()), n_fft_x, 1, h1p,
        buffer_dp1.data(), n_fft_x, 1, n1m, FFTW_ESTIMATE);
    
}

void mpi_subdomain_ghostcell_update(double* Value_sub) {
    using namespace mpi_topology;

    MPI_Request request[4];

    MPI_Isend(Value_sub, 1, ddtype_sendto_E,   comm_1d_x1.east_rank, 111, comm_1d_x1.mpi_comm, &request[0]);
    MPI_Irecv(Value_sub, 1, ddtype_recvfrom_W, comm_1d_x1.west_rank, 111, comm_1d_x1.mpi_comm, &request[1]);
    MPI_Isend(Value_sub, 1, ddtype_sendto_W,   comm_1d_x1.west_rank, 222, comm_1d_x1.mpi_comm, &request[2]);
    MPI_Irecv(Value_sub, 1, ddtype_recvfrom_E, comm_1d_x1.east_rank, 222, comm_1d_x1.mpi_comm, &request[3]);
    MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

    MPI_Isend(Value_sub, 1, ddtype_sendto_N,   comm_1d_x2.east_rank, 111, comm_1d_x2.mpi_comm, &request[0]);
    MPI_Irecv(Value_sub, 1, ddtype_recvfrom_S, comm_1d_x2.west_rank, 111, comm_1d_x2.mpi_comm, &request[1]);
    MPI_Isend(Value_sub, 1, ddtype_sendto_S,   comm_1d_x2.west_rank, 222, comm_1d_x2.mpi_comm, &request[2]);
    MPI_Irecv(Value_sub, 1, ddtype_recvfrom_N, comm_1d_x2.east_rank, 222, comm_1d_x2.mpi_comm, &request[3]);
    MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

    MPI_Isend(Value_sub, 1, ddtype_sendto_F,   comm_1d_x3.east_rank, 111, comm_1d_x3.mpi_comm, &request[0]);
    MPI_Irecv(Value_sub, 1, ddtype_recvfrom_B, comm_1d_x3.west_rank, 111, comm_1d_x3.mpi_comm, &request[1]);
    MPI_Isend(Value_sub, 1, ddtype_sendto_B,   comm_1d_x3.west_rank, 222, comm_1d_x3.mpi_comm, &request[2]);
    MPI_Irecv(Value_sub, 1, ddtype_recvfrom_F, comm_1d_x3.east_rank, 222, comm_1d_x3.mpi_comm, &request[3]);
    MPI_Waitall(4, request, MPI_STATUSES_IGNORE);
}

} // namespace mpi_subdomain