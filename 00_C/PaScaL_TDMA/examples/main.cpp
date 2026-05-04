//======================================================================================================================
//> @file        main.f90
//> @brief       This file contains the main subroutines for an example problem of PaScaL_TDMA.
//> @details     The target example problem is a three-dimensional(3D) time-dependent heat conduction problem 
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
//> @brief       Main execution program for the example problem of PaScaL_TDMA.
//>

#include "modules.hpp"

#include <mpi.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <array>

using namespace mpi_topology;
using namespace mpi_subdomain;
using namespace global;

void field_file_write(int myrank, int nprocs, const std::vector<double>& theta_sub);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int nprocs = 0, myrank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    period[0] = 1; period[1] = 0; period[2] = 1;

    global_inputpara(np_dim, argc, argv);

    mpi_topology_make();

    mpi_subdomain_make(comm_1d_x.nprocs, comm_1d_x.myrank,
                       comm_1d_y.nprocs, comm_1d_y.myrank,
                       comm_1d_z.nprocs, comm_1d_z.myrank);

    mpi_subdomain_make_ghostcell_ddtype();

    std::vector<double> theta_sub((nx_sub + 1) * (ny_sub + 1) * (nz_sub + 1), 0.0);

    mpi_subdomain_indices(comm_1d_y.myrank, comm_1d_y.nprocs);
    mpi_subdomain_mesh(comm_1d_x.myrank, comm_1d_y.myrank, comm_1d_z.myrank,
                       comm_1d_x.nprocs, comm_1d_y.nprocs, comm_1d_z.nprocs);

    mpi_subdomain_initialization(theta_sub, comm_1d_y.myrank, comm_1d_y.nprocs);
    mpi_subdomain_ghostcell_update(theta_sub.data(), comm_1d_x, comm_1d_y, comm_1d_z);
    mpi_subdomain_boundary(theta_sub, comm_1d_y.myrank, comm_1d_y.nprocs);

    MPI_Barrier(MPI_COMM_WORLD);

    solve_theta::solve_theta_plan_many(theta_sub);

    field_file_write(myrank, nprocs, theta_sub);

    theta_sub.clear();
    mpi_subdomain_clean();
    mpi_topology_clean();

    MPI_Finalize();
    return 0;
}

void field_file_write(int myrank, int nprocs, const std::vector<double>& theta_sub) {
    std::ostringstream filename;
    filename << "mpi_Tfield_sub_af"
             << (myrank / 10000) % 10
             << (myrank / 1000) % 10
             << (myrank / 100) % 10
             << (myrank / 10) % 10
             << myrank % 10 << ".PLT";
    std::ofstream fout(filename.str());
    fout << "VARIABLES=\"X\",\"Y\",\"Z\",\"THETA\"\n";
    fout << "zone t=\"1\"" << "i=" << nx_sub + 1
         << "j=" << ny_sub + 1 << "k=" << nz_sub + 1 << "\n";
    auto idx = [](int i, int j, int k) {
        return (i * (ny_sub + 1) + j) * (nz_sub + 1) + k;
    };
    for (int k = 1; k <= nz_sub - 1; ++k)
        for (int j = 1; j <= ny_sub - 1; ++j)
            for (int i = 1; i <= nx_sub - 1; ++i)
                fout << std::scientific << std::setprecision(6)
                     << x_sub[i] << ' ' << y_sub[j] << ' ' << z_sub[k] << ' '
                     << std::setprecision(14) << theta_sub[idx(i, j, k)] << '\n';
    fout.close();

    std::vector<int> nxm_sub_cnt(comm_1d_x.nprocs), nxm_sub_disp(comm_1d_x.nprocs);
    std::vector<int> nym_sub_cnt(comm_1d_y.nprocs), nym_sub_disp(comm_1d_y.nprocs);
    std::vector<int> nzm_sub_cnt(comm_1d_z.nprocs), nzm_sub_disp(comm_1d_z.nprocs);
    std::vector<MPI_Datatype> ddtype_data_write_recv(nprocs);
    std::vector<MPI_Request> request_recv(nprocs);
    MPI_Datatype ddtype_data_write_send;
    MPI_Request request_send;

    std::vector<double> theta_all;
    if (myrank == 0) theta_all.assign(nxm * nym * nzm, 0.0);

    MPI_Allgather(&nx_sub, 1, MPI_INT, nxm_sub_cnt.data(), 1, MPI_INT, comm_1d_x.mpi_comm);
    MPI_Allgather(&ny_sub, 1, MPI_INT, nym_sub_cnt.data(), 1, MPI_INT, comm_1d_y.mpi_comm);
    MPI_Allgather(&nz_sub, 1, MPI_INT, nzm_sub_cnt.data(), 1, MPI_INT, comm_1d_z.mpi_comm);
    for (int i = 0; i < comm_1d_x.nprocs; ++i) nxm_sub_cnt[i] -= 1;
    for (int i = 0; i < comm_1d_y.nprocs; ++i) nym_sub_cnt[i] -= 1;
    for (int i = 0; i < comm_1d_z.nprocs; ++i) nzm_sub_cnt[i] -= 1;
    nxm_sub_disp[0] = 0;
    for (int i = 1; i < comm_1d_x.nprocs; ++i)
        nxm_sub_disp[i] = nxm_sub_disp[i - 1] + nxm_sub_cnt[i - 1];
    nym_sub_disp[0] = 0;
    for (int i = 1; i < comm_1d_y.nprocs; ++i)
        nym_sub_disp[i] = nym_sub_disp[i - 1] + nym_sub_cnt[i - 1];
    nzm_sub_disp[0] = 0;
    for (int i = 1; i < comm_1d_z.nprocs; ++i)
        nzm_sub_disp[i] = nzm_sub_disp[i - 1] + nzm_sub_cnt[i - 1];

    int sizes[3] = {nx_sub + 1, ny_sub + 1, nz_sub + 1};
    int subsizes[3] = {nx_sub - 1, ny_sub - 1, nz_sub - 1};
    int starts[3] = {1, 1, 1};
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_data_write_send);
    MPI_Type_commit(&ddtype_data_write_send);

    std::vector<std::array<int, 3>> cart_coord(nprocs);
    for (int i = 0; i < nprocs; ++i)
        MPI_Cart_coords(mpi_world_cart, i, 3, cart_coord[i].data());

    for (int i = 0; i < nprocs; ++i) {
        int rsizes[3] = {nxm, nym, nzm};
        int rsubsizes[3] = {nxm_sub_cnt[cart_coord[i][0]],
                            nym_sub_cnt[cart_coord[i][1]],
                            nzm_sub_cnt[cart_coord[i][2]]};
        int rstarts[3] = {nxm_sub_disp[cart_coord[i][0]],
                          nym_sub_disp[cart_coord[i][1]],
                          nzm_sub_disp[cart_coord[i][2]]};
        MPI_Type_create_subarray(3, rsizes, rsubsizes, rstarts, MPI_ORDER_C,
                                 MPI_DOUBLE, &ddtype_data_write_recv[i]);
        MPI_Type_commit(&ddtype_data_write_recv[i]);
    }

    MPI_Isend(theta_sub.data(), 1, ddtype_data_write_send, 0, 101,
              MPI_COMM_WORLD, &request_send);
    if (myrank == 0) {
        for (int i = 0; i < nprocs; ++i)
            MPI_Irecv(theta_all.data(), 1, ddtype_data_write_recv[i], i, 101,
                      MPI_COMM_WORLD, &request_recv[i]);
    }
    MPI_Wait(&request_send, MPI_STATUS_IGNORE);
    if (myrank == 0)
        MPI_Waitall(nprocs, request_recv.data(), MPI_STATUSES_IGNORE);

    if (myrank == 0) {
        std::ofstream all("T_field_all.dat");
        auto gidx = [](int i, int j, int k) {
            return ((i - 1) * nym + (j - 1)) * nzm + (k - 1);
        };
        for (int k = 1; k <= nzm; ++k)
            for (int j = 1; j <= nym; ++j)
                for (int i = 1; i <= nxm; ++i)
                    all << std::scientific << std::setprecision(6)
                        << dx * static_cast<double>(i - 1) << ' '
                        << dy * static_cast<double>(j - 1) << ' '
                        << dz * static_cast<double>(k - 1) << ' '
                        << std::setprecision(14)
                        << theta_all[gidx(i, j, k)] << '\n';
        all.close();
    }

    MPI_Type_free(&ddtype_data_write_send);
    for (int i = 0; i < nprocs; ++i)
        MPI_Type_free(&ddtype_data_write_recv[i]);
}