module mpi_subdomain
    use MPI
    use global
    use cudafor
    use mpi_topology, only : cart_comm_1d

    implicit none

    !> @{ Grid numbers in the subdomain
    integer, public ::  n1sub, n2sub, n3sub
    integer, public :: n1msub,n2msub,n3msub
    !> @}
    !> @{ Grid indices of the assigned range
    integer, public :: ista, iend, jsta, jend, ksta, kend
    !> @}

    !> @{ Coordinates of grid points in the subdomain
    real(rp), allocatable, dimension(:) ::   x1  ,   x2  ,   x3
    real(rp), allocatable, dimension(:) ::   x1_g,   x2_g,   x3_g
    !> @}
    !> @{ Grid lengths in the subdomain
    real(rp), allocatable, dimension(:) ::  dx1  ,  dx2  ,  dx3
    real(rp), allocatable, dimension(:) :: dmx1  , dmx2  , dmx3
    real(rp), allocatable, dimension(:) ::  dx1_g,  dx2_g,  dx3_g
    real(rp), allocatable, dimension(:) :: dmx1_g, dmx2_g, dmx3_g

    !> @{ Derived datatype for communication between x-neighbor subdomains
    integer :: ddtype_sendto_E, ddtype_recvfrom_W, ddtype_sendto_W, ddtype_recvfrom_E
    !> @}
    !> @{ Derived datatype for communication between y-neighbor subdomains
    integer :: ddtype_sendto_N, ddtype_recvfrom_S, ddtype_sendto_S, ddtype_recvfrom_N
    !> @}
    !> @{ Derived datatype for communication between z-neighbor subdomains
    integer :: ddtype_sendto_F, ddtype_recvfrom_B, ddtype_sendto_B, ddtype_recvfrom_F
    !> @}

    !> @{ Half grid numbers plus one in the global domain
    integer :: h1p, h2p
    !> @}
    !> @{ Partitioned grid numbers in the subdomain for transpose scheme 1
    integer :: h1pJsub, n2mIsub, n2mJsub, h1pJsub_ista, h1pJsub_iend, n2mJsub_jsta, n2mJsub_jend
    !> @}

    !> @{ Partitioned grid numbers in the subdomain for transpose scheme 2
    integer :: n2msub_Isub, h1psub, h1psub_Jsub, h1psub_Jsub_ista, h1psub_Jsub_iend
    !> @}

    !> @{ Derived datatype for transpose scheme 1 and transpose scheme 2. C: cubic (org.), I: x-aligned, J: y-aligned.
    integer, allocatable, dimension(:) :: ddtype_dble_C_in_C2I, ddtype_dble_I_in_C2I    ! Scheme 1 & 2
    integer, allocatable, dimension(:) :: ddtype_cplx_C_in_C2I, ddtype_cplx_I_in_C2I    ! Scheme 1 & 2
    integer, allocatable, dimension(:) :: ddtype_cplx_I_in_I2J, ddtype_cplx_J_in_I2J    ! Scheme 1
    integer, allocatable, dimension(:) :: ddtype_dble_C_in_C2J, ddtype_dble_J_in_C2J    ! Scheme 2
    integer, allocatable, dimension(:) :: ddtype_cplx_C_in_C2J, ddtype_cplx_J_in_C2J    ! Scheme 2
    !> @}
    !> @{ Array for MPI_Alltoallw
    integer, allocatable, dimension(:) :: countsendI, countdistI
    integer, allocatable, dimension(:) :: countsendJ, countdistJ
    !> @}

    contains

    subroutine mpi_subdomain_environment()
        use mpi_topology, only : nprocs, myrank
        implicit none

        integer :: ierr
        
        call mpi_init(ierr)
        call mpi_comm_size(MPI_COMM_WORLD, nprocs, ierr)
        call mpi_comm_rank(MPI_COMM_WORLD, myrank, ierr)
        call mpi_barrier  (MPI_COMM_WORLD,         ierr)

    end subroutine mpi_subdomain_environment

    subroutine mpi_subdomain_initial()
        use mpi_topology, only : mpi_topology_make
        implicit none

        call mpi_topology_make()
        call mpi_subdomain_make()
        call mpi_subdomain_mesh()
        call mpi_subdomain_DDT_ghostcell()
        call mpi_subdomain_DDT_transpose2()
    end subroutine mpi_subdomain_initial
   
    subroutine mpi_subdomain_make()
        use mpi_topology, only : comm_1d_x1, comm_1d_x2, comm_1d_x3
        use interface, only : allocate_and_init
    
        implicit none
    
        call allocate_and_init(  x1_g, n1); call allocate_and_init(  x2_g, n2) ; call allocate_and_init(  x3_g, n3)
        call allocate_and_init(dmx1_g, n1); call allocate_and_init(dmx2_g, n2) ; call allocate_and_init(dmx3_g, n3)
        call allocate_and_init( dx1_g, n1); call allocate_and_init( dx2_g, n2) ; call allocate_and_init( dx3_g, n3)

        ! Assigning grid numbers and grid indices of my subdomain.
        call subdomain_para_range(1, n1m, comm_1d_x1%nprocs, comm_1d_x1%myrank, ista, iend)
        n1sub = iend - ista + 2
        call subdomain_para_range(1, n2m, comm_1d_x2%nprocs, comm_1d_x2%myrank, jsta, jend)
        n2sub = jend - jsta + 2
        call subdomain_para_range(1, n3m, comm_1d_x3%nprocs, comm_1d_x3%myrank, ksta, kend)
        n3sub = kend - ksta + 2
    
        n1msub=n1sub-1
        n2msub=n2sub-1
        n3msub=n3sub-1
    
        ! Allocate subdomain variables.
        call allocate_and_init(  x1, n1sub); call allocate_and_init(  x2, n2sub) ; call allocate_and_init(  x3, n3sub)
        call allocate_and_init(dmx1, n1sub); call allocate_and_init(dmx2, n2sub) ; call allocate_and_init(dmx3, n3sub)
        call allocate_and_init( dx1, n1sub); call allocate_and_init( dx2, n2sub) ; call allocate_and_init( dx3, n3sub)

    end subroutine mpi_subdomain_make

    subroutine mpi_subdomain_clean()
        implicit none

        deallocate(x1, dmx1, dx1)
        deallocate(x2, dmx2, dx2)
        deallocate(x3, dmx3, dx3)

        deallocate(x1_g, dmx1_g, dx1_g)
        deallocate(x2_g, dmx2_g, dx2_g)
        deallocate(x3_g, dmx3_g, dx3_g)

        deallocate(ddtype_dble_C_in_C2I,ddtype_dble_I_in_C2I)
        deallocate(ddtype_cplx_C_in_C2I,ddtype_cplx_I_in_C2I)
        deallocate(ddtype_dble_C_in_C2J,ddtype_dble_J_in_C2J)
        deallocate(ddtype_cplx_C_in_C2J,ddtype_cplx_J_in_C2J)

        deallocate(countsendI,countdistI)
        deallocate(countsendJ,countdistJ)

    end subroutine mpi_subdomain_clean

    subroutine mpi_subdomain_mesh()
        use mpi_topology,   only : comm_1d_x1, comm_1d_x2, comm_1d_x3
        implicit none

        !X-DIRECTION
        call calculate_x(UNIFORM1, ista, iend, x1_start, x1_end, L1, n1, n1m, GAMMA1, pbc1, comm_1d_x1, x1, n1sub)
        call calculate_dx_dmx(comm_1d_x1, x1, dx1, dmx1, n1sub, n1msub, pbc1)
        call expand_global_grid(comm_1d_x1, n1sub, x1, dx1, dmx1, x1_g, dx1_g, dmx1_g)

        !Y-DIRECTION
        call calculate_x(UNIFORM2, jsta, jend, x2_start, x2_end, L2, n2, n2m, GAMMA2, pbc2, comm_1d_x2, x2, n2sub)
        call calculate_dx_dmx(comm_1d_x2, x2, dx2, dmx2, n2sub, n2msub, pbc2)
        call expand_global_grid(comm_1d_x2, n2sub, x2, dx2, dmx2, x2_g, dx2_g, dmx2_g)

        !Z-DIRECTION
        call calculate_x(UNIFORM3, ksta, kend, x3_start, x3_end, L3, n3, n3m, GAMMA3, pbc3, comm_1d_x3, x3, n3sub)
        call calculate_dx_dmx(comm_1d_x3, x3, dx3, dmx3, n3sub, n3msub, pbc3)
        call expand_global_grid(comm_1d_x3, n3sub, x3, dx3, dmx3, x3_g, dx3_g, dmx3_g)

    end subroutine mpi_subdomain_mesh

    !>
    !> @brief       Build derived datatypes for subdomain communication using ghostcells.
    !>
    subroutine mpi_subdomain_DDT_ghostcell()

        implicit none
        integer :: sizes(0:2), subsizes(0:2), starts(0:2), ierr     ! Local variables for MPI_Type_create_subarray

        ! ddtype sending data to east MPI process (x+ neighbor)
        sizes    = (/n1sub+1,n2sub+1,n3sub+1/)
        subsizes = (/      1,n2sub+1,n3sub+1/)
        starts   = (/n1sub-1,      0,      0/)
        call MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, &
                                    MPI_DOUBLE_PRECISION, ddtype_sendto_E, ierr)
        call MPI_Type_commit(ddtype_sendto_E,ierr)

        ! ddtype receiving data from west MPI process (x- neighbor)
        starts   = (/      0,      0,      0/)
        call MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, &
                                    MPI_DOUBLE_PRECISION, ddtype_recvfrom_W, ierr)
        call MPI_Type_commit(ddtype_recvfrom_W,ierr)

        ! ddtype sending data to west MPI process (x- neighbor)
        starts   = (/      1,      0,      0/)
        call MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, &
                                    MPI_DOUBLE_PRECISION, ddtype_sendto_W, ierr)
        call MPI_Type_commit(ddtype_sendto_W,ierr)

        ! ddtype receiving data from east MPI process (x+ neighbor)
        starts   = (/  n1sub,      0,      0/)
        call MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, &
                                    MPI_DOUBLE_PRECISION, ddtype_recvfrom_E, ierr)
        call MPI_Type_commit(ddtype_recvfrom_E,ierr)

        ! ddtype sending data to north MPI process (y+ neighbor)
        sizes    = (/n1sub+1,n2sub+1,n3sub+1/)
        subsizes = (/n1sub+1,      1,n3sub+1/)
        starts   = (/      0,n2sub-1,      0/)
        call MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, &
                                    MPI_DOUBLE_PRECISION, ddtype_sendto_N, ierr)
        call MPI_Type_commit(ddtype_sendto_N,ierr)

        ! ddtype receiving data from south MPI process (y- neighbor)
        starts   = (/      0,      0,      0/)
        call MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, &
                                    MPI_DOUBLE_PRECISION, ddtype_recvfrom_S, ierr)
        call MPI_Type_commit(ddtype_recvfrom_S,ierr)

        ! ddtype sending data to south MPI process (y- neighbor)
        starts   = (/      0,      1,      0/)
        call MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, &
                                    MPI_DOUBLE_PRECISION, ddtype_sendto_S, ierr)
        call MPI_Type_commit(ddtype_sendto_S,ierr)

        ! ddtype receiving data from north MPI process (y+ neighbor)
        starts   = (/      0,  n2sub,      0/)
        call MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, &
                                    MPI_DOUBLE_PRECISION, ddtype_recvfrom_N, ierr)
        call MPI_Type_commit(ddtype_recvfrom_N,ierr)

        ! ddtype sending data to forth MPI process (z+ neighbor)
        sizes    = (/n1sub+1,n2sub+1,n3sub+1/)
        subsizes = (/n1sub+1,n2sub+1,      1/)
        starts   = (/      0,      0,n3sub-1/)
        call MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, &
                                    MPI_DOUBLE_PRECISION, ddtype_sendto_F, ierr)
        call MPI_Type_commit(ddtype_sendto_F,ierr)

        ! ddtype receiving data from back MPI process (z- neighbor)
        starts   = (/      0,      0,      0/)
        call MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, &
                                    MPI_DOUBLE_PRECISION, ddtype_recvfrom_B, ierr)
        call MPI_Type_commit(ddtype_recvfrom_B,ierr)

        ! ddtype sending data to back MPI process (z- neighbor)
        starts   = (/      0,      0,      1/)
        call MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, &
                                    MPI_DOUBLE_PRECISION, ddtype_sendto_B, ierr)
        call MPI_Type_commit(  ddtype_sendto_B,ierr)

        ! ddtype receiving data from forth MPI process (z+ neighbor)
        starts   = (/      0,      0,  n3sub/)
        call MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, &
                                    MPI_DOUBLE_PRECISION, ddtype_recvfrom_F, ierr)
        call MPI_Type_commit(ddtype_recvfrom_F,ierr)

    end subroutine mpi_subdomain_DDT_ghostcell
!>
    !> @brief       Build derived datatypes for FFT with transpose scheme 1.
    !>
    subroutine mpi_subdomain_DDT_transpose2()
                                    
        use mpi_topology,   only : comm_1d_x1, comm_1d_x2, comm_1d_x3, myrank

        implicit none

        integer :: i
        integer :: bigsize(3), subsize(3), start(3), ierr 
        integer :: indexA, indexB
        integer, allocatable, dimension(:) :: n2msub_IsubAll,n1msubAll,h1psubAll,h1psub_JsubAll,n2msubAll

        ! C means the partitioned domain in cubic shape as in original decomposition.
        ! I means the partitioned domain in x-aligned shaped for FFT in x-direction
        ! J means the partitioned domain in z-aligned shaped for FFT in y-direction
        allocate(ddtype_dble_C_in_C2I(0:comm_1d_x1%nprocs-1),ddtype_dble_I_in_C2I(0:comm_1d_x1%nprocs-1))
        allocate(ddtype_cplx_C_in_C2I(0:comm_1d_x1%nprocs-1),ddtype_cplx_I_in_C2I(0:comm_1d_x1%nprocs-1))
        allocate(ddtype_dble_C_in_C2J(0:comm_1d_x2%nprocs-1),ddtype_dble_J_in_C2J(0:comm_1d_x2%nprocs-1))
        allocate(ddtype_cplx_C_in_C2J(0:comm_1d_x2%nprocs-1),ddtype_cplx_J_in_C2J(0:comm_1d_x2%nprocs-1))

        allocate(countsendI(0:comm_1d_x1%nprocs-1), countdistI(0:comm_1d_x1%nprocs-1))
        allocate(countsendJ(0:comm_1d_x2%nprocs-1), countdistJ(0:comm_1d_x2%nprocs-1))

        countsendI(:)=1
        countdistI(:)=0
        countsendJ(:)=1
        countdistJ(:)=0

        allocate(n2msub_IsubAll(0:comm_1d_x1%nprocs-1),n1msubAll(0:comm_1d_x1%nprocs-1),h1psubAll(0:comm_1d_x1%nprocs-1))
        allocate(h1psub_JsubAll(0:comm_1d_x2%nprocs-1),n2msubAll(0:comm_1d_x2%nprocs-1))
        
        call subdomain_para_range(1, n2msub, comm_1d_x1%nprocs, comm_1d_x1%myrank, indexA, indexB)
        n2msub_Isub= indexB - indexA + 1     
        
        h1p = int(n1m/2)+1
        h2p = int(n2m/2)+1

        call subdomain_para_range(1, h1p, comm_1d_x1%nprocs, comm_1d_x1%myrank, indexA, indexB)
        h1psub= indexB - indexA + 1

        call subdomain_para_range(indexA, indexB, comm_1d_x2%nprocs, comm_1d_x2%myrank, h1psub_Jsub_ista, h1psub_Jsub_iend)
        h1psub_Jsub= h1psub_Jsub_iend - h1psub_Jsub_ista + 1

        do i=0,comm_1d_x1%nprocs-1
            call subdomain_para_range(1, n2msub, comm_1d_x1%nprocs, i, indexA, indexB)
            n2msub_IsubAll(i)= indexB - indexA + 1        
            call subdomain_para_range(1, n1m, comm_1d_x1%nprocs, i, indexA, indexB)
            n1msubAll(i)= indexB - indexA + 1        
            call subdomain_para_range(1, h1p, comm_1d_x1%nprocs, i, indexA, indexB)
            h1psubAll(i)= indexB - indexA + 1
        enddo
        
        do i=0,comm_1d_x2%nprocs-1             
            call subdomain_para_range(1, h1psub, comm_1d_x2%nprocs, i, indexA, indexB)
            h1psub_JsubAll(i) =indexB - indexA + 1 
            call subdomain_para_range(1, n2-1, comm_1d_x2%nprocs, i, indexA, indexB)
            n2msubAll(i) =indexB - indexA + 1 
        enddo

        ! DDT for I-C transpose (real type)
        do i=0,comm_1d_x1%nprocs-1
            bigsize(1) = n1msub
            bigsize(2) = n2msub
            bigsize(3) = n3msub
            subsize(1) = n1msub
            subsize(2) = n2msub_IsubAll(i)
            subsize(3) = n3msub
            start(1) = 0
            start(2) = sum(n2msub_IsubAll(0:i)) - n2msub_IsubAll(i)
            start(3) = 0
                        
            call MPI_TYPE_CREATE_SUBARRAY( 3, bigsize, subsize, start, MPI_ORDER_FORTRAN &
                                         , MPI_DOUBLE_PRECISION, ddtype_dble_C_in_C2I(i), ierr )
            call MPI_TYPE_COMMIT(ddtype_dble_C_in_C2I(i),ierr)
                        
            bigsize(1) = n1m
            bigsize(2) = n2msub_Isub
            bigsize(3) = n3msub
            subsize(1) = n1msubAll(i)
            subsize(2) = n2msub_Isub
            subsize(3) = n3msub
            start(1) = sum(n1msubAll(0:i)) - n1msubAll(i)
            start(2) = 0
            start(3) = 0
                        
            call MPI_TYPE_CREATE_SUBARRAY( 3, bigsize, subsize, start, MPI_ORDER_FORTRAN &
                                         , MPI_DOUBLE_PRECISION, ddtype_dble_I_in_C2I(i), ierr )
            call MPI_TYPE_COMMIT(ddtype_dble_I_in_C2I(i),ierr)
        enddo

        ! DDT for I-C transpose (complex type)
        do i=0,comm_1d_x1%nprocs-1
            bigsize(1) = h1psub
            bigsize(2) = n2msub
            bigsize(3) = n3msub
            subsize(1) = h1psub
            subsize(2) = n2msub_IsubAll(i)
            subsize(3) = n3msub
            start(1) = 0
            start(2) = sum(n2msub_IsubAll(0:i)) - n2msub_IsubAll(i)
            start(3) = 0
            call MPI_TYPE_CREATE_SUBARRAY( 3, bigsize, subsize, start, MPI_ORDER_FORTRAN &
                                         , MPI_DOUBLE_COMPLEX, ddtype_cplx_C_in_C2I(i), ierr )
            call MPI_TYPE_COMMIT(ddtype_cplx_C_in_C2I(i),ierr)
                                                
            bigsize(1) = h1p
            bigsize(2) = n2msub_Isub
            bigsize(3) = n3msub
            subsize(1) = h1psubAll(i)
            subsize(2) = n2msub_Isub
            subsize(3) = n3msub
            start(1) = sum(h1psubAll(0:i)) - h1psubAll(i)
            start(2) = 0
            start(3) = 0
            call MPI_TYPE_CREATE_SUBARRAY( 3, bigsize, subsize, start, MPI_ORDER_FORTRAN &
                                         , MPI_DOUBLE_COMPLEX, ddtype_cplx_I_in_C2I(i), ierr )
            call MPI_TYPE_COMMIT(ddtype_cplx_I_in_C2I(i),ierr)
        enddo

        ! DDT for I-K transpose (complex type)
        do i=0,comm_1d_x2%nprocs-1
            bigsize(1) = h1psub
            bigsize(2) = n2msub
            bigsize(3) = n3msub
            subsize(1) = h1psub_JsubAll(i)
            subsize(2) = n2msub
            subsize(3) = n3msub
            start(1) = sum(h1psub_JsubAll(0:i)) - h1psub_JsubAll(i)
            start(2) = 0
            start(3) = 0
            call MPI_TYPE_CREATE_SUBARRAY( 3, bigsize, subsize, start, MPI_ORDER_FORTRAN &
                                         , MPI_DOUBLE_COMPLEX, ddtype_cplx_C_in_C2J(i), ierr )
            call MPI_TYPE_COMMIT(ddtype_cplx_C_in_C2J(i),ierr)
                                                
            bigsize(1) = h1psub_Jsub
            bigsize(2) = n2m
            bigsize(3) = n3msub
            subsize(1) = h1psub_Jsub
            subsize(2) = n2msubAll(i)
            subsize(3) = n3msub
            start(1) = 0
            start(2) = sum(n2msubAll(0:i)) - n2msubAll(i)
            start(3) = 0
            call MPI_TYPE_CREATE_SUBARRAY( 3, bigsize, subsize, start, MPI_ORDER_FORTRAN &
                                         , MPI_DOUBLE_COMPLEX, ddtype_cplx_J_in_C2J(i), ierr )
            call MPI_TYPE_COMMIT(ddtype_cplx_J_in_C2J(i),ierr)
        enddo
        deallocate(n2msub_IsubAll,n1msubAll,h1psubAll,h1psub_JsubAll,n2msubAll)

    end subroutine mpi_subdomain_DDT_transpose2
    
    subroutine calculate_x(UNIFORM, sta, end, x_start, x_end, L, n, nm, GAMMA, pbc, comm_1d, x, nsub)
        use mpi_topology, only : myrank
        implicit none
        integer, intent(in) :: UNIFORM, sta, end, n, nm, nsub
        real(rp), intent(in) :: x_start, x_end, L, GAMMA
        logical, intent(in) :: pbc
        real(rp), dimension(0:), intent(out) :: x
        type(cart_comm_1d), intent(in) :: comm_1d
    
        selectcase(UNIFORM)
        case(0)
            call non_uniform_channel_grid(sta, end, x, x_start, L, nm, GAMMA)
        case(1)
            call uniform_grid(sta, end, x, x_start, L, nm)
        end select

        if((pbc==.false. ).and. (comm_1d%myrank==0)) x(0)=x(1)
    
    end subroutine calculate_x

    subroutine non_uniform_channel_grid(sta, end, x, x_start, L, nm, GAMMA)
        implicit none
        integer, intent(in) :: sta, end, nm
        real(rp), intent(in)  :: x_start, L, GAMMA
        real(rp), intent(inout) :: x(0:)
    
        integer :: idx
    
        do idx = sta-1, end+1
            x(idx-sta+1) = L*real(0.5,rp)*(real(1.0,rp) + real(tanh(0.5*GAMMA*(2.0*real(idx-1,rp)/real(nm,rp)-1.0))/tanh(GAMMA*0.5),rp)) &
                         + x_start
        end do
    end subroutine non_uniform_channel_grid

    subroutine uniform_grid(sta, end, x, x_start, L, nm)
        implicit none
        integer , intent(in)  :: sta, end, nm
        real(rp), intent(in)  :: x_start, L
        real(rp), intent(inout) :: x(0:)
        integer :: idx
    
        do idx = sta-1, end+1
            x(idx-sta+1) = real(idx-1,rp)* L/real(nm,rp) + x_start
        end do
    end subroutine uniform_grid

    subroutine calculate_dx_dmx(comm_1d, x, dx, dmx, nsub, nmsub, pbc)
        implicit none

        integer, intent(in) :: nsub, nmsub
        logical, intent(in) :: pbc
        real(rp), dimension(0:), intent(inout) :: x, dx, dmx
        type(cart_comm_1d), intent(in) :: comm_1d
        integer :: request_S2E, request_S2W, ierr, STATUS(MPI_STATUS_SIZE)
        integer :: idx

        ! DX
        do idx = 1, nmsub
            dx(idx) = x(idx+1) - x(idx)
        end do

        call mpi_subdomain_ghostcell_1d(dx, comm_1d, nsub, nmsub)

        if((pbc==.false.) .and. (comm_1d%myrank==0))                dx(   0)=real(0.0,rp)
        if((pbc==.false.) .and. (comm_1d%myrank==comm_1d%nprocs-1)) dx(nsub)=real(0.0,rp)
    
        ! DMX
        do idx = 1, nmsub
            dmx(idx) = real(0.5,rp)*(dx(idx-1)+dx(idx))
        end do

        call mpi_subdomain_ghostcell_1d(dmx, comm_1d, nsub, nmsub)

        if((pbc==.false.) .and. (comm_1d%myrank==comm_1d%nprocs-1)) dmx(nsub)=real(0.5,rp)*(dx(nmsub)+dx(nsub))

    end subroutine calculate_dx_dmx

    subroutine expand_global_grid(comm_1d, nsub, x, dx, dmx, x_g, dx_g, dmx_g)
        implicit none

        integer, intent(in) :: nsub
        real(rp), dimension(0:), intent(in ) :: x, dx, dmx
        real(rp), dimension(0:), intent(out) :: x_g, dx_g, dmx_g
        type(cart_comm_1d), intent(in) :: comm_1d
        integer, allocatable, dimension(:) :: sendcounts, displs
        integer :: ierr, idx
    
        allocate(sendcounts(comm_1d%nprocs))
        do idx = 1, comm_1d%nprocs
            sendcounts(idx) = nsub+1
        end do
    
        allocate(displs(comm_1d%nprocs))
        displs(1) = 0
        do idx = 2, comm_1d%nprocs
            displs(idx) = displs(idx-1) + sendcounts(idx-1) - 2
        enddo
    
        call MPI_Allgatherv(  x, (nsub+1), MPI_REAL_TYPE,   x_g, sendcounts, displs, MPI_REAL_TYPE, comm_1d%mpi_comm, ierr)
        call MPI_Allgatherv( dx, (nsub+1), MPI_REAL_TYPE,  dx_g, sendcounts, displs, MPI_REAL_TYPE, comm_1d%mpi_comm, ierr)
        call MPI_Allgatherv(dmx, (nsub+1), MPI_REAL_TYPE, dmx_g, sendcounts, displs, MPI_REAL_TYPE, comm_1d%mpi_comm, ierr)

        deallocate(sendcounts, displs)

    end subroutine expand_global_grid

    
    subroutine subdomain_para_range(nsta, nend, nprocs, myrank, index_sta, index_end)

        implicit none

        integer, intent(in)     :: nsta, nend, nprocs, myrank
        integer, intent(out)    :: index_sta, index_end
        integer :: iwork1, iwork2

        iwork1 = int((nend - nsta + 1) / nprocs)
        iwork2 = mod(nend - nsta + 1, nprocs)
        index_sta = myrank * iwork1 + nsta + min(myrank, iwork2)
        index_end = index_sta + iwork1 - 1
        if (iwork2 > myrank) index_end = index_end + 1

    end subroutine subdomain_para_range

    subroutine mpi_subdomain_ghostcell_update_real(Value_sub)
        use mpi_topology,   only : comm_1d_x1, comm_1d_x2, comm_1d_x3
        use interface, only: allocate_and_init
        implicit none

        real(rp), dimension(0:n1sub, 0:n2sub, 0:n3sub), intent(inout) :: Value_sub
        integer :: i, j, k
        integer :: ierr, request(4), istat
        ! Communication buffer
        real(rp), allocatable, dimension(:,:)   :: sbuf_x0, sbuf_x1, sbuf_y0, sbuf_y1, sbuf_z0, sbuf_z1
        real(rp), allocatable, dimension(:,:)   :: rbuf_x0, rbuf_x1, rbuf_y0, rbuf_y1, rbuf_z0, rbuf_z1

        call allocate_and_init(sbuf_x0, n2sub, n3sub)
        call allocate_and_init(sbuf_x1, n2sub, n3sub)
        call allocate_and_init(sbuf_y0, n1sub, n3sub)
        call allocate_and_init(sbuf_y1, n1sub, n3sub)
        call allocate_and_init(sbuf_z0, n1sub, n2sub)
        call allocate_and_init(sbuf_z1, n1sub, n2sub)
        
        call allocate_and_init(rbuf_x0, n2sub, n3sub)
        call allocate_and_init(rbuf_x1, n2sub, n3sub)
        call allocate_and_init(rbuf_y0, n1sub, n3sub)
        call allocate_and_init(rbuf_y1, n1sub, n3sub)
        call allocate_and_init(rbuf_z0, n1sub, n2sub)
        call allocate_and_init(rbuf_z1, n1sub, n2sub)
        
        ! Update ghostcells
        ! X-Direction
        do k = 0, n3sub
        do j = 0, n2sub
            if(comm_1d_x1%west_rank.ne.MPI_PROC_NULL) then
                sbuf_x0(j,k) = Value_sub(1      ,j,k)
            endif
            if(comm_1d_x1%east_rank.ne.MPI_PROC_NULL) then
                sbuf_x1(j,k) = Value_sub(n1sub-1,j,k)
            endif
        enddo
        enddo

        call MPI_Isend(sbuf_x0, (n2sub+1)*(n3sub+1), MPI_real_type, comm_1d_x1%west_rank, 111, comm_1d_x1%mpi_comm, request(1), ierr)
        call MPI_Irecv(rbuf_x1, (n2sub+1)*(n3sub+1), MPI_real_type, comm_1d_x1%east_rank, 111, comm_1d_x1%mpi_comm, request(2), ierr)
        call MPI_Irecv(rbuf_x0, (n2sub+1)*(n3sub+1), MPI_real_type, comm_1d_x1%west_rank, 222, comm_1d_x1%mpi_comm, request(3), ierr)
        call MPI_Isend(sbuf_x1, (n2sub+1)*(n3sub+1), MPI_real_type, comm_1d_x1%east_rank, 222, comm_1d_x1%mpi_comm, request(4), ierr)
        call MPI_Waitall(4, request, MPI_STATUSES_IGNORE, ierr)

        do k = 0, n3sub
        do j = 0, n2sub
            if(comm_1d_x1%west_rank.ne.MPI_PROC_NULL) then
                Value_sub(0    ,j,k) = rbuf_x0(j,k)
            endif
            if(comm_1d_x1%east_rank.ne.MPI_PROC_NULL) then
                Value_sub(n1sub,j,k) = rbuf_x1(j,k)
            endif
        enddo
        enddo

        ! Y-Direction
        do k = 0, n3sub
        do i = 0, n1sub
            if(comm_1d_x2%west_rank.ne.MPI_PROC_NULL) then
                sbuf_y0(i,k) = Value_sub(i,1      ,k)
            endif
            if(comm_1d_x2%east_rank.ne.MPI_PROC_NULL) then
                sbuf_y1(i,k) = Value_sub(i,n2sub-1,k)
            endif
        enddo
        enddo

        call MPI_Isend(sbuf_y0, (n1sub+1)*(n3sub+1), MPI_real_type, comm_1d_x2%west_rank, 111, comm_1d_x2%mpi_comm, request(1), ierr)
        call MPI_Irecv(rbuf_y1, (n1sub+1)*(n3sub+1), MPI_real_type, comm_1d_x2%east_rank, 111, comm_1d_x2%mpi_comm, request(2), ierr)
        call MPI_Irecv(rbuf_y0, (n1sub+1)*(n3sub+1), MPI_real_type, comm_1d_x2%west_rank, 222, comm_1d_x2%mpi_comm, request(3), ierr)
        call MPI_Isend(sbuf_y1, (n1sub+1)*(n3sub+1), MPI_real_type, comm_1d_x2%east_rank, 222, comm_1d_x2%mpi_comm, request(4), ierr)
        call MPI_Waitall(4, request, MPI_STATUSES_IGNORE, ierr)

        do k = 0, n3sub
        do i = 0, n1sub
            if(comm_1d_x2%west_rank.ne.MPI_PROC_NULL) then
                Value_sub(i,0    ,k) = rbuf_y0(i,k)
            endif
            if(comm_1d_x2%east_rank.ne.MPI_PROC_NULL) then
                Value_sub(i,n2sub,k) = rbuf_y1(i,k)
            endif
        enddo
        enddo

        ! Z-Direction
        do j = 0, n2sub
        do i = 0, n1sub
            if(comm_1d_x3%west_rank.ne.MPI_PROC_NULL) then
                sbuf_z0(i,j) = Value_sub(i,j,1      )
            endif
            if(comm_1d_x3%east_rank.ne.MPI_PROC_NULL) then
                sbuf_z1(i,j) = Value_sub(i,j,n3sub-1)
            endif
        enddo
        enddo

        call MPI_Isend(sbuf_z0, (n1sub+1)*(n2sub+1), MPI_real_type, comm_1d_x3%west_rank, 111, comm_1d_x3%mpi_comm, request(1), ierr)
        call MPI_Irecv(rbuf_z1, (n1sub+1)*(n2sub+1), MPI_real_type, comm_1d_x3%east_rank, 111, comm_1d_x3%mpi_comm, request(2), ierr)
        call MPI_Irecv(rbuf_z0, (n1sub+1)*(n2sub+1), MPI_real_type, comm_1d_x3%west_rank, 222, comm_1d_x3%mpi_comm, request(3), ierr)
        call MPI_Isend(sbuf_z1, (n1sub+1)*(n2sub+1), MPI_real_type, comm_1d_x3%east_rank, 222, comm_1d_x3%mpi_comm, request(4), ierr)
        call MPI_Waitall(4, request, MPI_STATUSES_IGNORE, ierr)

        do j = 0, n2sub
        do i = 0, n1sub
            if(comm_1d_x3%west_rank.ne.MPI_PROC_NULL) then
                Value_sub(i,j,0    ) = rbuf_z0(i,j)
            endif
            if(comm_1d_x3%east_rank.ne.MPI_PROC_NULL) then
                Value_sub(i,j,n3sub) = rbuf_z1(i,j)
            endif
        enddo
        enddo

        deallocate( sbuf_x0, sbuf_x1 )
        deallocate( sbuf_y0, sbuf_y1 )
        deallocate( sbuf_z0, sbuf_z1 )
        deallocate( rbuf_x0, rbuf_x1 )
        deallocate( rbuf_y0, rbuf_y1 )
        deallocate( rbuf_z0, rbuf_z1 )
        
        call MPI_Barrier(MPI_COMM_WORLD, ierr);

    end subroutine mpi_subdomain_ghostcell_update_real

    subroutine mpi_subdomain_ghostcell_update_integer(Value_sub)
        use mpi_topology,   only : comm_1d_x1, comm_1d_x2, comm_1d_x3
        
        implicit none

        integer, dimension(-1:n1sub+1, -1:n2sub+1, -1:n3sub+1), intent(inout) :: Value_sub

        integer :: i, j, k
        integer :: ierr, request(4), istat
        ! Communication buffer
        integer, allocatable, dimension(:,:,:)   :: sbuf_x0, sbuf_x1, sbuf_y0, sbuf_y1, sbuf_z0, sbuf_z1
        integer, allocatable, dimension(:,:,:)   :: rbuf_x0, rbuf_x1, rbuf_y0, rbuf_y1, rbuf_z0, rbuf_z1

        allocate( sbuf_x0(0:n2sub,0:n3sub,0:1), sbuf_x1(0:n2sub,0:n3sub,0:1) )
        allocate( sbuf_y0(0:n1sub,0:n3sub,0:1), sbuf_y1(0:n1sub,0:n3sub,0:1) )
        allocate( sbuf_z0(0:n1sub,0:n2sub,0:1), sbuf_z1(0:n1sub,0:n2sub,0:1) )
        allocate( rbuf_x0(0:n2sub,0:n3sub,0:1), rbuf_x1(0:n2sub,0:n3sub,0:1) )
        allocate( rbuf_y0(0:n1sub,0:n3sub,0:1), rbuf_y1(0:n1sub,0:n3sub,0:1) )
        allocate( rbuf_z0(0:n1sub,0:n2sub,0:1), rbuf_z1(0:n1sub,0:n2sub,0:1) )

        sbuf_x0 = 0; sbuf_x1 = 0
        sbuf_y0 = 0; sbuf_y1 = 0
        sbuf_z0 = 0; sbuf_z1 = 0
        rbuf_x0 = 0; rbuf_x1 = 0
        rbuf_y0 = 0; rbuf_y1 = 0
        rbuf_z0 = 0; rbuf_z1 = 0
        
        ! Update ghostcells
        ! X-Direction
        do k = 0, n3sub
        do j = 0, n2sub
        do i = 0, 1
            if(comm_1d_x1%west_rank.ne.MPI_PROC_NULL) then
                sbuf_x0(j,k,i) = Value_sub(      1+i,j,k)
            endif
            if(comm_1d_x1%east_rank.ne.MPI_PROC_NULL) then
                sbuf_x1(j,k,i) = Value_sub(n1sub-2+i,j,k)
            endif
        enddo
        enddo
        enddo

        call MPI_Isend(sbuf_x0, (n2sub+1)*(n3sub+1)*2, MPI_INTEGER, comm_1d_x1%west_rank, 111, comm_1d_x1%mpi_comm, request(1), ierr)
        call MPI_Irecv(rbuf_x1, (n2sub+1)*(n3sub+1)*2, MPI_INTEGER, comm_1d_x1%east_rank, 111, comm_1d_x1%mpi_comm, request(2), ierr)
        call MPI_Irecv(rbuf_x0, (n2sub+1)*(n3sub+1)*2, MPI_INTEGER, comm_1d_x1%west_rank, 222, comm_1d_x1%mpi_comm, request(3), ierr)
        call MPI_Isend(sbuf_x1, (n2sub+1)*(n3sub+1)*2, MPI_INTEGER, comm_1d_x1%east_rank, 222, comm_1d_x1%mpi_comm, request(4), ierr)
        call MPI_Waitall(4, request, MPI_STATUSES_IGNORE, ierr)

        do k = 0, n3sub
        do j = 0, n2sub
        do i = 0, 1
            if(comm_1d_x1%west_rank.ne.MPI_PROC_NULL) then
                Value_sub(   -1+i,j,k) = rbuf_x0(j,k,i)
            endif
            if(comm_1d_x1%east_rank.ne.MPI_PROC_NULL) then
                Value_sub(n1sub+i,j,k) = rbuf_x1(j,k,i)
            endif
        enddo
        enddo
        enddo
        
        ! Y-Direction
        do k = 0, n3sub
        do i = 0, n1sub
        do j = 0, 1
            if(comm_1d_x2%west_rank.ne.MPI_PROC_NULL) then
                sbuf_y0(i,k,j) = Value_sub(i,      1+j,k)
            endif
            if(comm_1d_x2%east_rank.ne.MPI_PROC_NULL) then
                sbuf_y1(i,k,j) = Value_sub(i,n2sub-2+j,k)
            endif
        enddo
        enddo
        enddo

        call MPI_Isend(sbuf_y0, (n1sub+1)*(n3sub+1)*2, MPI_INTEGER, comm_1d_x2%west_rank, 111, comm_1d_x2%mpi_comm, request(1), ierr)
        call MPI_Irecv(rbuf_y1, (n1sub+1)*(n3sub+1)*2, MPI_INTEGER, comm_1d_x2%east_rank, 111, comm_1d_x2%mpi_comm, request(2), ierr)
        call MPI_Irecv(rbuf_y0, (n1sub+1)*(n3sub+1)*2, MPI_INTEGER, comm_1d_x2%west_rank, 222, comm_1d_x2%mpi_comm, request(3), ierr)
        call MPI_Isend(sbuf_y1, (n1sub+1)*(n3sub+1)*2, MPI_INTEGER, comm_1d_x2%east_rank, 222, comm_1d_x2%mpi_comm, request(4), ierr)
        call MPI_Waitall(4, request, MPI_STATUSES_IGNORE, ierr)

        do k = 0, n3sub
        do i = 0, n1sub
        do j = 0, 1
            if(comm_1d_x2%west_rank.ne.MPI_PROC_NULL) then
                Value_sub(i,   -1+j,k) = rbuf_y0(i,k,j)
            endif
            if(comm_1d_x2%east_rank.ne.MPI_PROC_NULL) then
                Value_sub(i,n2sub+j,k) = rbuf_y1(i,k,j)
            endif
        enddo
        enddo
        enddo

        ! Z-Direction
        do j = 0, n2sub
        do i = 0, n1sub
        do k = 0, 1
            if(comm_1d_x3%west_rank.ne.MPI_PROC_NULL) then
                sbuf_z0(i,j,k) = Value_sub(i,j,      1+k)
            endif
            if(comm_1d_x3%east_rank.ne.MPI_PROC_NULL) then
                sbuf_z1(i,j,k) = Value_sub(i,j,n3sub-2+k)
            endif
        enddo
        enddo
        enddo

        call MPI_Isend(sbuf_z0, (n1sub+1)*(n2sub+1), MPI_real_type, comm_1d_x3%west_rank, 111, comm_1d_x3%mpi_comm, request(1), ierr)
        call MPI_Irecv(rbuf_z1, (n1sub+1)*(n2sub+1), MPI_real_type, comm_1d_x3%east_rank, 111, comm_1d_x3%mpi_comm, request(2), ierr)
        call MPI_Irecv(rbuf_z0, (n1sub+1)*(n2sub+1), MPI_real_type, comm_1d_x3%west_rank, 222, comm_1d_x3%mpi_comm, request(3), ierr)
        call MPI_Isend(sbuf_z1, (n1sub+1)*(n2sub+1), MPI_real_type, comm_1d_x3%east_rank, 222, comm_1d_x3%mpi_comm, request(4), ierr)
        call MPI_Waitall(4, request, MPI_STATUSES_IGNORE, ierr)

        do j = 0, n2sub
        do i = 0, n1sub
        do k = 0, 1
            if(comm_1d_x3%west_rank.ne.MPI_PROC_NULL) then
                Value_sub(i,j,   -1+k) = rbuf_z0(i,j,k)
            endif
            if(comm_1d_x3%east_rank.ne.MPI_PROC_NULL) then
                Value_sub(i,j,n3sub+k) = rbuf_z1(i,j,k)
            endif
        enddo
        enddo
        enddo

        deallocate( sbuf_x0, sbuf_x1 )
        deallocate( sbuf_y0, sbuf_y1 )
        deallocate( sbuf_z0, sbuf_z1 )
        deallocate( rbuf_x0, rbuf_x1 )
        deallocate( rbuf_y0, rbuf_y1 )
        deallocate( rbuf_z0, rbuf_z1 )
        
        call MPI_Barrier(MPI_COMM_WORLD, ierr);

    end subroutine mpi_subdomain_ghostcell_update_integer

    subroutine mpi_subdomain_ghostcell_1d(data, comm_1d, nsub, nmsub)
        implicit none

        real(rp), dimension(0:), intent(inout) :: data
        type(cart_comm_1d), intent(in) :: comm_1d
        integer, intent(in) :: nsub, nmsub
        integer :: ierr, STATUS(MPI_STATUS_SIZE)
        integer :: request_S2E, request_S2W
    
        call MPI_ISEND(data(nmsub),1, MPI_real_type, comm_1d%east_rank, 111, comm_1d%mpi_comm, request_S2E, ierr)
        call MPI_IRECV(data(0)    ,1, MPI_real_type, comm_1d%west_rank, 111, comm_1d%mpi_comm, request_S2E, ierr)
        call MPI_WAIT(request_S2E,STATUS,ierr)
        call MPI_ISEND(data(1)    ,1, MPI_real_type, comm_1d%west_rank, 111, comm_1d%mpi_comm, request_S2W, ierr)
        call MPI_IRECV(data(nsub) ,1, MPI_real_type, comm_1d%east_rank, 111, comm_1d%mpi_comm, request_S2W, ierr)
        call MPI_WAIT(request_S2W,STATUS,ierr)
    end subroutine mpi_subdomain_ghostcell_1d

    subroutine mpi_allreduce_max(Value)
        use mpi_topology,   only : comm_1d_x1, comm_1d_x2, comm_1d_x3, myrank
        implicit none
        real(rp), intent(inout) :: Value
        integer :: ierr
        real(rp) :: ValueI, ValueIJ, ValueIJK
    
        call MPI_ALLREDUCE(Value  , ValueI  , 1, MPI_real_type, MPI_MAX, comm_1d_x1%mpi_comm, ierr)
        call MPI_ALLREDUCE(ValueI , ValueIJ , 1, MPI_real_type, MPI_MAX, comm_1d_x2%mpi_comm, ierr)
        call MPI_ALLREDUCE(ValueIJ, ValueIJK, 1, MPI_real_type, MPI_MAX, comm_1d_x3%mpi_comm, ierr)
        Value=ValueIJK
        if(myrank==0) write(*,*) "Value", Value

    end subroutine mpi_allreduce_max

    ! Calculate the x3-coordinate for a given index, minimum size, n3cut, and stretch ratio
    real(rp) function calc_x3(index, min_size, n3cut, stretch)
        integer, intent(in) :: index
        real(rp), intent(in) :: min_size, stretch
        integer, intent(in) :: n3cut
        real(rp) :: temp_stretch

        if (index <= n3cut + 1) then
            calc_x3 = x3_start + min_size * real(index - 1, rp)
        else
            temp_stretch = (stretch ** real(index - n3cut - 1, rp) - real(1.0,rp) ) / (stretch - real(1.0,rp) )
            calc_x3 = real(n3cut, rp) * min_size + min_size * temp_stretch
        end if

    end function calc_x3

end module mpi_subdomain