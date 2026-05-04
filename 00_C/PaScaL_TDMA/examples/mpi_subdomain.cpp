////======================================================================================================================
//> @file        mpi_subdomain.f90
//> @brief       This file contains a module of subdomains for the example problem of PaScaL_TDMA.
//> @details     The target example problem is the three-dimensional(3D) time-dependent heat conduction problem 
//>              in a unit cube domain applied with the boundary conditions of vertically constant temperature 
//>              and horizontally periodic boundaries.
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
//> @brief       Module for building subdomains from the physical domain.
//> @details     This module has simulation parameters for subdomains and communication between the subdomains.
//>

#include <cmath>
#include <vector>
#include <mpi.h>
#include "modules.hpp"

namespace mpi_subdomain {

int nx_sub, ny_sub, nz_sub;
int ista, iend, jsta, jend, ksta, kend;
std::vector<double> x_sub, y_sub, z_sub;
std::vector<double> dmx_sub, dmy_sub, dmz_sub;
std::vector<double> thetaBC3_sub, thetaBC4_sub;
std::vector<int> jmbc_index, jpbc_index;

MPI_Datatype ddtype_sendto_E, ddtype_recvfrom_W, ddtype_sendto_W, ddtype_recvfrom_E;
MPI_Datatype ddtype_sendto_N, ddtype_recvfrom_S, ddtype_sendto_S, ddtype_recvfrom_N;
MPI_Datatype ddtype_sendto_F, ddtype_recvfrom_B, ddtype_sendto_B, ddtype_recvfrom_F;

void mpi_subdomain_make(int nprocs_in_x, int myrank_in_x,
                        int nprocs_in_y, int myrank_in_y,
                        int nprocs_in_z, int myrank_in_z) {
    using namespace global;
    para_range(1, nx - 1, nprocs_in_x, myrank_in_x, &ista, &iend);
    nx_sub = iend - ista + 2;
    para_range(1, ny - 1, nprocs_in_y, myrank_in_y, &jsta, &jend);
    ny_sub = jend - jsta + 2;
    para_range(1, nz - 1, nprocs_in_z, myrank_in_z, &ksta, &kend);
    nz_sub = kend - ksta + 2;

    x_sub.assign(nx_sub + 1, 0.0);
    y_sub.assign(ny_sub + 1, 0.0);
    z_sub.assign(nz_sub + 1, 0.0);
    dmx_sub.assign(nx_sub + 1, 0.0);
    dmy_sub.assign(ny_sub + 1, 0.0);
    dmz_sub.assign(nz_sub + 1, 0.0);
    thetaBC3_sub.assign((nx_sub + 1) * (nz_sub + 1), 0.0);
    thetaBC4_sub.assign((nx_sub + 1) * (nz_sub + 1), 0.0);
    jmbc_index.assign(ny_sub + 1, 0);
    jpbc_index.assign(ny_sub + 1, 0);
}

void mpi_subdomain_clean() {
    x_sub.clear(); y_sub.clear(); z_sub.clear();
    dmx_sub.clear(); dmy_sub.clear(); dmz_sub.clear();
    thetaBC3_sub.clear(); thetaBC4_sub.clear();
    jmbc_index.clear(); jpbc_index.clear();
    MPI_Type_free(&ddtype_sendto_E); MPI_Type_free(&ddtype_recvfrom_W);
    MPI_Type_free(&ddtype_sendto_W); MPI_Type_free(&ddtype_recvfrom_E);
    MPI_Type_free(&ddtype_sendto_N); MPI_Type_free(&ddtype_recvfrom_S);
    MPI_Type_free(&ddtype_sendto_S); MPI_Type_free(&ddtype_recvfrom_N);
    MPI_Type_free(&ddtype_sendto_F); MPI_Type_free(&ddtype_recvfrom_B);
    MPI_Type_free(&ddtype_sendto_B); MPI_Type_free(&ddtype_recvfrom_F);
}

void mpi_subdomain_make_ghostcell_ddtype() {
    int sizes[3], subsizes[3], starts[3];
    sizes[0] = nz_sub + 1; sizes[1] = ny_sub + 1; sizes[2] = nx_sub + 1;
    subsizes[0] = nz_sub + 1; subsizes[1] = ny_sub + 1; subsizes[2] = 1;
    starts[0] = 0; starts[1] = 0; starts[2] = nx_sub - 1;
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

    starts[2] = nx_sub;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_recvfrom_E);
    MPI_Type_commit(&ddtype_recvfrom_E);

    subsizes[0] = nz_sub + 1; subsizes[1] = 1; subsizes[2] = nx_sub + 1;
    starts[0] = 0; starts[1] = ny_sub - 1; starts[2] = 0;
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

    starts[1] = ny_sub;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_recvfrom_N);
    MPI_Type_commit(&ddtype_recvfrom_N);

    subsizes[0] = 1; subsizes[1] = ny_sub + 1; subsizes[2] = nx_sub + 1;
    starts[0] = nz_sub - 1; starts[1] = 0; starts[2] = 0;
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

    starts[0] = nz_sub;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_recvfrom_F);
    MPI_Type_commit(&ddtype_recvfrom_F);
}

void mpi_subdomain_ghostcell_update(double* theta_sub,
                                    const mpi_topology::cart_comm_1d& comm_1d_x,
                                    const mpi_topology::cart_comm_1d& comm_1d_y,
                                    const mpi_topology::cart_comm_1d& comm_1d_z) {
    MPI_Request request[12];

    MPI_Isend(theta_sub, 1, ddtype_sendto_E,  comm_1d_x.east_rank, 111, comm_1d_x.mpi_comm, &request[0]);
    MPI_Irecv(theta_sub, 1, ddtype_recvfrom_W, comm_1d_x.west_rank, 111, comm_1d_x.mpi_comm, &request[1]);
    MPI_Isend(theta_sub, 1, ddtype_sendto_W,  comm_1d_x.west_rank, 222, comm_1d_x.mpi_comm, &request[2]);
    MPI_Irecv(theta_sub, 1, ddtype_recvfrom_E, comm_1d_x.east_rank, 222, comm_1d_x.mpi_comm, &request[3]);

    MPI_Isend(theta_sub, 1, ddtype_sendto_N,  comm_1d_y.east_rank, 111, comm_1d_y.mpi_comm, &request[4]);
    MPI_Irecv(theta_sub, 1, ddtype_recvfrom_S, comm_1d_y.west_rank, 111, comm_1d_y.mpi_comm, &request[5]);
    MPI_Isend(theta_sub, 1, ddtype_sendto_S,  comm_1d_y.west_rank, 222, comm_1d_y.mpi_comm, &request[6]);
    MPI_Irecv(theta_sub, 1, ddtype_recvfrom_N, comm_1d_y.east_rank, 222, comm_1d_y.mpi_comm, &request[7]);

    MPI_Isend(theta_sub, 1, ddtype_sendto_F,  comm_1d_z.east_rank, 111, comm_1d_z.mpi_comm, &request[8]);
    MPI_Irecv(theta_sub, 1, ddtype_recvfrom_B, comm_1d_z.west_rank, 111, comm_1d_z.mpi_comm, &request[9]);
    MPI_Isend(theta_sub, 1, ddtype_sendto_B,  comm_1d_z.west_rank, 222, comm_1d_z.mpi_comm, &request[10]);
    MPI_Irecv(theta_sub, 1, ddtype_recvfrom_F, comm_1d_z.east_rank, 222, comm_1d_z.mpi_comm, &request[11]);

    MPI_Waitall(12, request, MPI_STATUSES_IGNORE);
}

void mpi_subdomain_indices(int myrank_in_y, int nprocs_in_y) {
    jmbc_index.assign(ny_sub + 1, 1);
    jpbc_index.assign(ny_sub + 1, 1);
    if (myrank_in_y == 0) jmbc_index[1] = 0;
    if (myrank_in_y == nprocs_in_y - 1) jpbc_index[ny_sub - 1] = 0;
}

void mpi_subdomain_mesh(int myrank_in_x, int myrank_in_y, int myrank_in_z,
                        int nprocs_in_x, int nprocs_in_y, int nprocs_in_z) {
    using namespace global;
    dx = lx / static_cast<double>(nx - 1);
    for (int i = 0; i <= nx_sub; ++i) {
        x_sub[i] = static_cast<double>(i - 1 + ista - 1) * dx;
        dmx_sub[i] = dx;
    }
    dy = ly / static_cast<double>(ny);
    for (int j = 0; j <= ny_sub; ++j) {
        y_sub[j] = static_cast<double>(j + jsta - 1) * dy;
        dmy_sub[j] = dy;
    }
    dz = lz / static_cast<double>(nz - 1);
    for (int k = 0; k <= nz_sub; ++k) {
        z_sub[k] = static_cast<double>(k - 1 + ksta - 1) * dz;
        dmz_sub[k] = dz;
    }
    if (myrank_in_x == 0) dmx_sub[0] = dx;
    if (myrank_in_x == nprocs_in_x - 1) dmx_sub[nx_sub] = dx;
    if (myrank_in_y == 0) dmy_sub[0] = dy / 2.0;
    if (myrank_in_y == nprocs_in_y - 1) dmy_sub[ny_sub] = dy / 2.0;
    if (myrank_in_z == 0) dmz_sub[0] = dz;
    if (myrank_in_z == nprocs_in_z - 1) dmz_sub[nz_sub] = dz;
}

void mpi_subdomain_initialization(std::vector<double>& theta_sub,
                                  int myrank_in_y, int nprocs_in_y) {
    auto idx = [=](int i, int j, int k) {
        return (i * (ny_sub + 1) + j) * (nz_sub + 1) + k;
    };
    for (int k = 0; k <= nz_sub; ++k)
        for (int j = 0; j <= ny_sub; ++j)
            for (int i = 0; i <= nx_sub; ++i)
                theta_sub[idx(i, j, k)] = (global::theta_cold - global::theta_hot) / global::ly * y_sub[j] + global::theta_hot
                                        + std::sin(2 * 2.0 * global::PI / global::lx * x_sub[i])
                                        * std::sin(2 * 2.0 * global::PI / global::lz * z_sub[k])
                                        * std::sin(2 * 2.0 * global::PI / global::ly * y_sub[j]);

    for (int k = 0; k <= nz_sub; ++k)
        for (int i = 0; i <= nx_sub; ++i) {
            if (myrank_in_y == 0) theta_sub[idx(i, 0, k)] = global::theta_hot;
            if (myrank_in_y == nprocs_in_y - 1) theta_sub[idx(i, ny_sub, k)] = global::theta_cold;
        }
}

void mpi_subdomain_boundary(const std::vector<double>& theta_sub,
                            int myrank_in_y, int nprocs_in_y) {
    auto idx = [=](int i, int j, int k) {
        return (i * (ny_sub + 1) + j) * (nz_sub + 1) + k;
    };
    auto bidx = [=](int i, int k) {
        return i * (nz_sub + 1) + k;
    };
    for (int k = 0; k <= nz_sub; ++k)
        for (int i = 0; i <= nx_sub; ++i) {
            thetaBC3_sub[bidx(i, k)] = theta_sub[idx(i, 0, k)];
            thetaBC4_sub[bidx(i, k)] = theta_sub[idx(i, ny_sub, k)];
        }
    if (myrank_in_y == 0)
        for (int k = 0; k <= nz_sub; ++k)
            for (int i = 0; i <= nx_sub; ++i)
                thetaBC3_sub[bidx(i, k)] = global::theta_hot;
    if (myrank_in_y == nprocs_in_y - 1)
        for (int k = 0; k <= nz_sub; ++k)
            for (int i = 0; i <= nx_sub; ++i)
                thetaBC4_sub[bidx(i, k)] = global::theta_cold;
}

} // namespace mpi_subdomain
