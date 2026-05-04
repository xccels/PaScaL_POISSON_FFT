//======================================================================================================================
//> @file        module_mpi_topology.f90
//> @brief       This file contains a module of communication topology for PaScaL_TCS.
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
//> @brief       Module for creating the cartesian topology of the MPI processes and subcommunicators.
//> @details     This module has three subcommunicators in each-direction and related subroutines.
//>

#include "modules.hpp"

namespace mpi_topology{

    MPI_Comm mpi_world_cart;       //< Communicator for cartesian topology
    int nprocs;                    //< World size
    int myrank;                    //< World rank

    cart_comm_1d comm_1d_x1;
    cart_comm_1d comm_1d_x2;
    cart_comm_1d comm_1d_x3;
    cart_comm_1d comm_1d_x1n2;
    
    //>
    //> @brief       Destroy the communicator for cartesian topology.
    //>
    void mpi_topology_clean() {
        MPI_Comm_free(&mpi_world_cart);
    }

    //>
    //> @brief       Create the cartesian topology for the MPI processes and subcommunicators.
    //>
    void mpi_topology_make() {
        int np_dim[3] = {global::np1, global::np2, global::np3};
        int period[3] = {global::pbc1 ? 1 : 0, global::pbc2 ? 1 : 0, global::pbc3 ? 1 : 0};

        // Create the cartesian topology
        MPI_Cart_create(MPI_COMM_WORLD,    //  input  | integer      | Input communicator (handle).
                        3,                 //  input  | integer      | Number of dimensions of Cartesian grid (integer).
                        np_dim,            //  input  | integer(1:3) | Integer array of size ndims specifying the number of processes in each dimension.
                        period,            //  input  | logical(1:3) | Logical array of size ndims specifying whether the grid is periodic (true=1) or not (false=0) in each dimension.
                        0,                 //  input  | logical      | Ranking may be reordered (true=1) or not (false=0) (logical).
                        &mpi_world_cart    // *output | integer      | Communicator with new Cartesian topology (handle).
                    );

        int remain[3];

        // Create subcommunicators and assign two neighboring processes in the x-direction.
        remain[0] = 1; remain[1] = 0; remain[2] = 0;
        MPI_Cart_sub(mpi_world_cart, remain, &comm_1d_x1.mpi_comm);
        MPI_Comm_rank(comm_1d_x1.mpi_comm, &comm_1d_x1.myrank);
        MPI_Comm_size(comm_1d_x1.mpi_comm, &comm_1d_x1.nprocs);
        MPI_Cart_shift(comm_1d_x1.mpi_comm, 0, 1, &comm_1d_x1.west_rank, &comm_1d_x1.east_rank);
                
        // Create subcommunicators and assign two neighboring processes in the y-direction
        remain[0] = 0; remain[1] = 1; remain[2] = 0;
        MPI_Cart_sub(mpi_world_cart, remain, &comm_1d_x2.mpi_comm);
        MPI_Comm_rank(comm_1d_x2.mpi_comm, &comm_1d_x2.myrank);
        MPI_Comm_size(comm_1d_x2.mpi_comm, &comm_1d_x2.nprocs);
        MPI_Cart_shift(comm_1d_x2.mpi_comm, 0, 1, &comm_1d_x2.west_rank, &comm_1d_x2.east_rank);
        
        // Create subcommunicators and assign two neighboring processes in the z-direction
        remain[0] = 0; remain[1] = 0; remain[2] = 1;
        MPI_Cart_sub(mpi_world_cart, remain, &comm_1d_x3.mpi_comm);
        MPI_Comm_rank(comm_1d_x3.mpi_comm, &comm_1d_x3.myrank);
        MPI_Comm_size(comm_1d_x3.mpi_comm, &comm_1d_x3.nprocs);
        MPI_Cart_shift(comm_1d_x3.mpi_comm, 0, 1, &comm_1d_x3.west_rank, &comm_1d_x3.east_rank);
        
        // For Poisson
        MPI_Comm_split(mpi_world_cart, comm_1d_x3.myrank,
                    comm_1d_x1.myrank + comm_1d_x2.myrank * comm_1d_x1.nprocs,
                    &comm_1d_x1n2.mpi_comm);
        MPI_Comm_rank(comm_1d_x1n2.mpi_comm, &comm_1d_x1n2.myrank);
        MPI_Comm_size(comm_1d_x1n2.mpi_comm, &comm_1d_x1n2.nprocs);
    }

} // namespace mpi_topology