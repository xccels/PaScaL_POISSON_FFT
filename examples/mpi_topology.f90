module mpi_topology

    use mpi

    implicit none

    integer :: mpi_world_cart       !< Communicator for cartesian topology
    integer :: nprocs               !< World size
    integer :: myrank               !< World rank

    !> @brief   Type variable for the information of 1D communicator
    type, public :: cart_comm_1d
        integer :: myrank           !< Rank ID in current communicator
        integer :: nprocs                   !< Number of processes in current communicator
        integer :: west_rank                !< Previous rank ID in current communicator
        integer :: east_rank                !< Next rank ID in current communicator
        integer :: mpi_comm                 !< Current communicator
    end type cart_comm_1d

    type(cart_comm_1d)  :: comm_1d_x1       !< Subcommunicator information in x-direction
    type(cart_comm_1d)  :: comm_1d_x2       !< Subcommunicator information in y-direction
    type(cart_comm_1d)  :: comm_1d_x3       !< Subcommunicator information in z-direction

    public  :: mpi_topology_make
    public  :: mpi_topology_clean

    contains

    !>
    !> @brief       Destroy the communicator for cartesian topology.
    !>
    subroutine mpi_topology_clean()

        implicit none
        integer :: ierr

        call MPI_Comm_free(mpi_world_cart, ierr)

    end subroutine mpi_topology_clean

    !>
    !> @brief       Create the cartesian topology for the MPI processes and subcommunicators.
    !>
   
    subroutine mpi_topology_make()
        use global, only : np1, np2, np3, pbc1, pbc2, pbc3
    
        implicit none
    
        integer :: np_dim(1:3)
        logical :: period(1:3)
        logical :: remain(1:3)
        integer :: ierr
    
        np_dim(1) = np1;    np_dim(2) = np2;    np_dim(3) = np3
        period(1) = pbc1;   period(2) = pbc2;   period(3) = pbc3
    
        call MPI_Cart_create( MPI_COMM_WORLD, 3, np_dim, period, .false., mpi_world_cart, ierr)
    
        remain = [.true., .false., .false.]
        call create_subcommunicator(remain, comm_1d_x1)
    
        remain = [.false., .true., .false.]
        call create_subcommunicator(remain, comm_1d_x2)
    
        remain = [.false., .false., .true.]
        call create_subcommunicator(remain, comm_1d_x3)
    end subroutine mpi_topology_make

    subroutine create_subcommunicator(remain, comm_1d)
        implicit none
        logical, intent(in) :: remain(1:3)
        type(cart_comm_1d), intent(out) :: comm_1d
    
        integer :: ierr
    
        call MPI_Cart_sub( mpi_world_cart, remain, comm_1d%mpi_comm, ierr)
        call MPI_Comm_rank(comm_1d%mpi_comm, comm_1d%myrank, ierr)
        call MPI_Comm_size(comm_1d%mpi_comm, comm_1d%nprocs, ierr)
        call MPI_Cart_shift(comm_1d%mpi_comm, 0, 1, comm_1d%west_rank, comm_1d%east_rank, ierr)
    end subroutine create_subcommunicator

end module mpi_topology
