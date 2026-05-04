//======================================================================================================================
//> @file        mpi_topology.f90
//> @brief       This file contains a module of communication topology for the example problem of PaScaL_TDMA.
//> @details     The target example problem is the three-dimensional time-dependent heat conduction problem 
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
//> @brief       Module for creating the cartesian topology of the MPI processes and subcommunicators.
//> @details     This module has three subcommunicators in each-direction and related subroutines.
//>
#include <mpi.h>
#include "modules.hpp"

namespace mpi_topology {

MPI_Comm mpi_world_cart;
int np_dim[3];
int period[3];

cart_comm_1d comm_1d_x;
cart_comm_1d comm_1d_y;
cart_comm_1d comm_1d_z;

void mpi_topology_clean() {
    MPI_Comm_free(&mpi_world_cart);
}

void mpi_topology_make() {
    MPI_Cart_create(MPI_COMM_WORLD, 3, np_dim, period, 0, &mpi_world_cart);

    int remain[3];

    remain[0] = 1; remain[1] = 0; remain[2] = 0;
    MPI_Cart_sub(mpi_world_cart, remain, &comm_1d_x.mpi_comm);
    MPI_Comm_rank(comm_1d_x.mpi_comm, &comm_1d_x.myrank);
    MPI_Comm_size(comm_1d_x.mpi_comm, &comm_1d_x.nprocs);
    MPI_Cart_shift(comm_1d_x.mpi_comm, 0, 1, &comm_1d_x.west_rank, &comm_1d_x.east_rank);

    remain[0] = 0; remain[1] = 1; remain[2] = 0;
    MPI_Cart_sub(mpi_world_cart, remain, &comm_1d_y.mpi_comm);
    MPI_Comm_rank(comm_1d_y.mpi_comm, &comm_1d_y.myrank);
    MPI_Comm_size(comm_1d_y.mpi_comm, &comm_1d_y.nprocs);
    MPI_Cart_shift(comm_1d_y.mpi_comm, 0, 1, &comm_1d_y.west_rank, &comm_1d_y.east_rank);

    remain[0] = 0; remain[1] = 0; remain[2] = 1;
    MPI_Cart_sub(mpi_world_cart, remain, &comm_1d_z.mpi_comm);
    MPI_Comm_rank(comm_1d_z.mpi_comm, &comm_1d_z.myrank);
    MPI_Comm_size(comm_1d_z.mpi_comm, &comm_1d_z.nprocs);
    MPI_Cart_shift(comm_1d_z.mpi_comm, 0, 1, &comm_1d_z.west_rank, &comm_1d_z.east_rank);
}

} // namespace mpi_topology
