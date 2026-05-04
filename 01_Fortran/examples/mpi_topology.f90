module mpi_topology
    use mpi
    implicit none

    integer :: myrank, nprocs, mpi_world_cart

    type :: cart_comm_1d
        integer :: mpi_comm
        integer :: myrank
        integer :: nprocs
        integer :: west_rank, east_rank
    end type cart_comm_1d

    type(cart_comm_1d) :: comm_1d_x1    !< Subcommunicator in x1-direction
    type(cart_comm_1d) :: comm_1d_x2    !< Subcommunicator in x2-direction
    type(cart_comm_1d) :: comm_1d_x3    !< Subcommunicator in x3-direction
#ifndef USE_GPU
    type(cart_comm_1d) :: comm_1d_x1n2  !< x1-x2 combined communicator for Poisson transpose (CPU only)
#endif

contains

    subroutine mpi_topology_make()
        use global
        implicit none
        integer :: ierr
        integer :: np_dim(1:3)
        logical :: period(1:3)
        logical :: remain(1:3)

        np_dim(1) = np1; np_dim(2) = np2; np_dim(3) = np3
        period(1) = pbc1; period(2) = pbc2; period(3) = pbc3

        call MPI_Cart_create(MPI_COMM_WORLD, 3, np_dim, period, .false., mpi_world_cart, ierr)

        remain = [.true., .false., .false.]
        call create_subcommunicator(remain, comm_1d_x1)

        remain = [.false., .true., .false.]
        call create_subcommunicator(remain, comm_1d_x2)

        remain = [.false., .false., .true.]
        call create_subcommunicator(remain, comm_1d_x3)

#ifndef USE_GPU
        call MPI_Comm_split(mpi_world_cart, comm_1d_x3%myrank,              &
                            comm_1d_x1%myrank + comm_1d_x2%myrank*comm_1d_x1%nprocs, &
                            comm_1d_x1n2%mpi_comm, ierr)
        call MPI_Comm_rank(comm_1d_x1n2%mpi_comm, comm_1d_x1n2%myrank, ierr)
        call MPI_Comm_size(comm_1d_x1n2%mpi_comm, comm_1d_x1n2%nprocs, ierr)
#endif

        call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    end subroutine mpi_topology_make

    subroutine mpi_topology_clean()
        implicit none
        integer :: ierr
        call MPI_Comm_free(comm_1d_x1%mpi_comm, ierr)
        call MPI_Comm_free(comm_1d_x2%mpi_comm, ierr)
        call MPI_Comm_free(comm_1d_x3%mpi_comm, ierr)
#ifndef USE_GPU
        call MPI_Comm_free(comm_1d_x1n2%mpi_comm, ierr)
#endif
    end subroutine mpi_topology_clean

    subroutine create_subcommunicator(remain, comm_1d)
        implicit none
        logical, intent(in)          :: remain(1:3)
        type(cart_comm_1d), intent(out) :: comm_1d
        integer :: ierr

        call MPI_Cart_sub(mpi_world_cart, remain, comm_1d%mpi_comm, ierr)
        call MPI_Comm_rank(comm_1d%mpi_comm, comm_1d%myrank, ierr)
        call MPI_Comm_size(comm_1d%mpi_comm, comm_1d%nprocs, ierr)
        call MPI_Cart_shift(comm_1d%mpi_comm, 0, 1, comm_1d%west_rank, comm_1d%east_rank, ierr)
    end subroutine create_subcommunicator

end module mpi_topology
