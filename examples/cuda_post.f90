module cuda_post
    use mpi_topology
    use global
    use cuda_subdomain
    use mpi_subdomain

    implicit none
    logical :: FirstPrintXY=.true.,FirstPrintXZ=.true.,FirstPrintH=.true.,FirstPrint3=.true.,FirstPrintGrid=.true.,FirstPrint3D=.true.,FirstPrintDiv=.true.

    !> @{ MPI derived datatype for file IO
    integer :: ddtype_inner_domain
    integer :: ddtype_global_domain_pr_IO
    integer :: ddtype_global_domain_MPI_IO
    integer :: ddtype_global_domain_MPI_IO_aggregation
    integer :: ddtype_aggregation_chunk
    !> @}

    !> @{ Temporary variables for building DDT for file IO
    integer, allocatable, DIMENSION(:)  :: cnts_pr, disp_pr, cnts_aggr, disp_aggr
    !> @}

    !> @{ Communicator type for aggregated file IO
    type, private :: comm_IO
        integer :: myrank
        integer :: nprocs
        integer :: mpi_comm
    end type comm_IO
    !> @}

    type(comm_IO)  :: comm_IO_aggregation   !> Communicator for aggregation
    type(comm_IO)  :: comm_IO_master        !> Communicator for master process having aggregated data

    !> @{ Function for file IO
    public :: cuda_Post_initial

    ! public :: cuda_Post_FileOut_InstantField              ! Instanteneous filed print in ascii format

    public :: cuda_Post_FileIn_Continue_Post_Reassembly_IO
    public :: cuda_Post_FileOut_Continue_Post_Reassembly_IO
    !> @}

    real(rp) :: maxdivU

    character(len=17) :: xyzrank

contains

    subroutine cuda_Post_initial()

        implicit none

        integer :: chunk_size=1

        !> @{ Temporary local variables for communicator setup
        integer :: color
        integer :: i, j, k, icur, ierr
        integer :: sizes(3), subsizes(3), starts(3)

        integer :: ddtype_temp
        integer :: r8size
        integer :: mpi_world_group, mpi_master_group
        !> @}

        !> @{ Temporary local variables for communicator setup
        integer, allocatable, dimension(:,:)    :: cart_coord
        integer, allocatable, dimension(:)      :: n1msub_cnt,  n2msub_cnt,  n3msub_cnt
        integer, allocatable, dimension(:)      :: n1msub_disp, n2msub_disp, n3msub_disp
        integer, allocatable, dimension(:)      :: maste_group_rank
        integer(kind=MPI_ADDRESS_KIND)          :: extent, lb
        !> @}

        ! String of rank ID in x, y, and z-direction
        write(xyzrank, '(I5.5,1A1,I5.5,1A1,I5.5)' ) comm_1d_x1%myrank,'_',   &
                                                    comm_1d_x2%myrank,'_',   &
                                                    comm_1d_x3%myrank

        
        !>>>>>>>>>> MPI communicator for data aggregation (comm_IO_aggregation)
        ! comm_IO_aggregation includes MPI processes of chunk_size (mpi_size_aggregation) in z-direction.
        comm_IO_aggregation%nprocs = chunk_size

        if( mod(comm_1d_x3%nprocs, comm_IO_aggregation%nprocs).ne.0 ) then
            if( myrank.eq.0) print *, '[Error] Chunk_size for IO aggregation should be a measure of mpisize in z-direction'
            call MPI_Abort(MPI_COMM_WORLD, 11, ierr)
        endif

        color = myrank / comm_IO_aggregation%nprocs

        call MPI_Comm_split(MPI_COMM_WORLD, color, myrank, comm_IO_aggregation%mpi_comm, ierr )
        call MPI_Comm_rank(comm_IO_aggregation%mpi_comm, comm_IO_aggregation%myrank, ierr)

        !>>>>>>>>>> MPI communicator for aggregation_master (comm_IO_master)
        ! comm_IO_master includes rank0 processes in comm_IO_aggregation
        comm_IO_master%mpi_comm = MPI_COMM_NULL
        comm_IO_master%nprocs = nprocs / comm_IO_aggregation%nprocs
        allocate( maste_group_rank(comm_IO_master%nprocs) )
        do i = 1, comm_IO_master%nprocs
            maste_group_rank(i) = (i-1) * comm_IO_aggregation%nprocs
        enddo
        
        call MPI_Comm_group(MPI_COMM_WORLD, mpi_world_group, ierr)
        call MPI_Group_incl(mpi_world_group, comm_IO_master%nprocs, maste_group_rank, mpi_master_group, ierr)
        call MPI_Comm_create_group(MPI_COMM_WORLD, mpi_master_group, 0, comm_IO_master%mpi_comm, ierr)

        if(comm_IO_master%mpi_comm.ne.MPI_COMM_NULL) then
            call MPI_Comm_rank(comm_IO_master%mpi_comm, comm_IO_master%myrank, ierr)
        endif
        deallocate( maste_group_rank )
                                            
        !>>>>>>>>>> Derived datatype for inner domain without ghost cells
        sizes    = (/ n1msub+2, n2msub+2, n3msub+2 /)
        subsizes = (/ n1msub,   n2msub,   n3msub   /)
        starts   = (/ 1, 1, 1 /)

        call MPI_Type_create_subarray(  3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, &
                                        MPI_DOUBLE_PRECISION, ddtype_inner_domain, ierr)
        call MPI_Type_commit(ddtype_inner_domain, ierr)

        !>>>>>>>>>> Derived datatype for post reassembly IO
        ! Post reassembly can be used when a single node is capable of memory allocation for the global domain.
        ! All data is gathered into rank 0 and scattered from rank 0 using MPI_Scatterv and MPI_Gatherv
        sizes    = (/ n1m,    n2m,    n3m    /)
        subsizes = (/ n1msub, n2msub, n3msub /)
        starts   = (/ 0, 0, 0 /)

        call MPI_Type_create_subarray(  3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, &
                                        MPI_DOUBLE_PRECISION, ddtype_temp, ierr)

        CALL MPI_Type_size(MPI_DOUBLE_PRECISION, r8size, ierr)
        lb = 0
        extent = r8size

        call MPI_Type_create_resized(ddtype_temp, lb, extent, ddtype_global_domain_pr_IO, ierr)
        call MPI_Type_commit(ddtype_global_domain_pr_IO, ierr)

        ! Counts and displacements for MPI_Gatherv and MPI_Scatterv 
        allocate( cart_coord(3,0:nprocs-1) )
        do i = 0, nprocs-1
            call MPI_Cart_coords(mpi_world_cart, i, 3, cart_coord(:,i), ierr )
        enddo

        allocate(n1msub_cnt(0:comm_1d_x1%nprocs-1), n1msub_disp(0:comm_1d_x1%nprocs-1))
        allocate(n2msub_cnt(0:comm_1d_x2%nprocs-1), n2msub_disp(0:comm_1d_x2%nprocs-1))
        allocate(n3msub_cnt(0:comm_1d_x3%nprocs-1), n3msub_disp(0:comm_1d_x3%nprocs-1))

        call MPI_Allgather(n1msub, 1, MPI_INTEGER, n1msub_cnt, 1, MPI_INTEGER, comm_1d_x1%mpi_comm, ierr)
        call MPI_Allgather(n2msub, 1, MPI_INTEGER, n2msub_cnt, 1, MPI_INTEGER, comm_1d_x2%mpi_comm, ierr)
        call MPI_Allgather(n3msub, 1, MPI_INTEGER, n3msub_cnt, 1, MPI_INTEGER, comm_1d_x3%mpi_comm, ierr)

        n1msub_disp(0) = 0
        do i = 1, comm_1d_x1%nprocs-1
            n1msub_disp(i)=sum(n1msub_cnt(0:i-1))
        enddo

        n2msub_disp(0) = 0
        do i = 1, comm_1d_x2%nprocs-1
            n2msub_disp(i)=sum(n2msub_cnt(0:i-1))
        enddo

        n3msub_disp(0) = 0
        do i = 1, comm_1d_x3%nprocs-1
            n3msub_disp(i)=sum(n3msub_cnt(0:i-1))
        enddo

        allocate( cnts_pr(0:nprocs-1) )
        allocate( disp_pr(0:nprocs-1) )
        
        do i = 0, nprocs-1
            cnts_pr(i) = 1
            disp_pr(i) =  n1msub_disp(cart_coord(1,i))  &
                        + n2msub_disp(cart_coord(2,i)) * n1m  &
                        + n3msub_disp(cart_coord(3,i)) * n1m * n2m
        enddo

        deallocate(cart_coord)
        deallocate(n1msub_cnt, n2msub_cnt, n3msub_cnt)
        deallocate(n1msub_disp, n2msub_disp, n3msub_disp)

        !>>>>>>>>>> Derived datatype for MPI IO
        sizes    = (/ n1m,    n2m,    n3m    /)
        subsizes = (/ n1msub, n2msub, n3msub /)
        starts   = (/ ista-1, jsta-1, ksta-1 /)

        call MPI_Type_create_subarray(  3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, &
                                        MPI_DOUBLE_PRECISION, ddtype_global_domain_MPI_IO, ierr)
        call MPI_Type_commit(ddtype_global_domain_MPI_IO, ierr)


        !>>>>>>>>>> Derived datatype for data aggregation.
        sizes    = (/ n1msub, n2msub, n3msub * comm_IO_aggregation%nprocs /)
        subsizes = (/ n1msub, n2msub, n3msub /)
        starts   = (/ 0, 0, n3msub * comm_IO_aggregation%myrank /)
        
        call MPI_Type_create_subarray(  3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, &
                                        MPI_DOUBLE_PRECISION, ddtype_temp, ierr)

        CALL MPI_Type_size(MPI_DOUBLE_PRECISION, r8size, ierr)
        lb = 0
        extent = r8size

        call MPI_Type_create_resized(ddtype_temp, lb, extent, ddtype_aggregation_chunk, ierr)
        call MPI_Type_commit(ddtype_aggregation_chunk, ierr)

        allocate( cnts_aggr(0:comm_IO_aggregation%nprocs-1) )
        allocate( disp_aggr(0:comm_IO_aggregation%nprocs-1) )

        do i = 0, comm_IO_aggregation%nprocs-1
            cnts_aggr(i) = 1
            disp_aggr(i) = n1msub * n2msub * n3msub * i
        enddo

        !>>>>>>>>>> Derived datatype for aggregated MPI IO.
        sizes    = (/ n1m,    n2m,    n3m    /)
        subsizes = (/ n1msub, n2msub, n3msub * comm_IO_aggregation%nprocs /)
        starts   = (/ ista-1, jsta-1, ksta-1 - n3msub * comm_IO_aggregation%myrank /)

        call MPI_Type_create_subarray(  3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, &
                                        MPI_DOUBLE_PRECISION, ddtype_global_domain_MPI_IO_aggregation, ierr)
        call MPI_Type_commit(ddtype_global_domain_MPI_IO_aggregation, ierr)

    end subroutine cuda_Post_initial


    subroutine cuda_Post_FileIn_Continue_Post_Reassembly_IO(myrank,Pin)

        use mpi

        implicit none

        integer :: myrank
        double precision, dimension(0:n1sub, 0:n2sub, 0:n3sub) :: Pin

        double precision, allocatable, dimension(:,:,:) :: var_all
        integer :: ierr

        ! Rank 0 allocates chunk array which have all field array data
        if(myrank.eq.0) allocate( var_all( n1m, n2m, n3m) )

        ! Rank 0 reads P from a single file
        if(myrank.eq.0) then
            open(myrank, FILE=trim(dir_cont_filein)//'cont_P.bin', FORM='UNFORMATTED', &
                STATUS='OLD', ACTION='READ', ACCESS='DIRECT', RECL=sizeof(var_all) )
            read(myrank, REC=1) var_all
            close(myrank)
        endif

        ! Rank 0 scatter P using the defined DDT
        call MPI_Scatterv(var_all, cnts_pr, disp_pr, ddtype_global_domain_pr_IO, Pin, 1, ddtype_inner_domain, 0, MPI_COMM_WORLD, ierr)

        ! Rank 0 deallocates chunk array
        if(myrank.eq.0) deallocate( var_all )

        ! Ghostcell update
        call mpi_subdomain_ghostcell_update_real(Pin)
    
        if(myrank.eq.0) print '(a)', 'Read continue file using post reassembly IO'

    end subroutine cuda_Post_FileIn_Continue_Post_Reassembly_IO

    subroutine cuda_Post_FileOut_Continue_Post_Reassembly_IO(myrank,Pin)
        use mpi

        implicit none

        integer :: myrank
        double precision, dimension(0:n1sub, 0:n2sub, 0:n3sub) :: Pin

        double precision, allocatable, dimension(:,:,:) :: var_all
        integer :: ierr, i, j, k
        real(rp) :: x1g, x2g, x3g


        ! Rank 0 allocates chunk array which have all field array data
        if(myrank.eq.0) allocate( var_all(n1m, n2m, n3m) )

        ! Rank 0 gathers P using the defined DDT
        call MPI_Gatherv(Pin, 1, ddtype_inner_domain, var_all, cnts_pr, disp_pr, ddtype_global_domain_pr_IO, 0, MPI_COMM_WORLD, ierr)

        ! Rank 0 writes P to a single file
        if(myrank.eq.0) then
            open(myrank, FILE=trim(dir_cont_fileout)//'cont_P.bin', FORM='UNFORMATTED', &
                STATUS='REPLACE', ACTION='WRITE', ACCESS='DIRECT', RECL=sizeof(var_all) )
            write(myrank, REC=1) var_all
            close(myrank)
        endif

        ! Rank 0 deallocates chunk array
        if(myrank.eq.0) deallocate( var_all )

        if(myrank.eq.0) print '(a)', 'Write continue file using post reassembly IO'

    end subroutine cuda_Post_FileOut_Continue_Post_Reassembly_IO

    subroutine cuda_Post_Fileout_InstantField_3D(P) ! YMG 210408
        use mpi_subdomain, only : x1, x2, x3
        use mpi_topology , only : myrank
        use mpi

        implicit none

        real(rp),         dimension( 0:n1sub  ,  0:n2sub  ,  0:n3sub         ) :: P

        real(rp) :: pg
        character(len=22) :: filename_instantfield3D, filename, cfile 

        integer :: i,j,k,im,jm,km
    
        filename_instantfield3D='Output_instantfield_3D'

        write(cfile, '(I3.3)') myrank 

        filename = 'P_'//trim(adjustl(cfile))//'.PLT'
    
        open(unit=myrank,file=trim(dir_instantfield)//filename)
            write(myrank,*) 'VARIABLES="X","Y","Z","P"'  !--
            write(myrank,*) 'zone t="',1,'"','i=',(n1sub-1),'j=',(n2sub-1),'k=',(n3sub-1)

            do k=1, n3sub-1
            do j=1, n2sub-1
            do i=1, n1sub-1
                im = i-1 ; jm = j-1; km = k-1;

                pg = calc_C2g(P, i, j, k, im, jm, km)

                write(myrank,'(4(E11.4,1x))' ) x1(i), x2(j), x3(k), pg 
            enddo
            enddo
            enddo
        close(myrank)
    
    end subroutine cuda_Post_Fileout_InstantField_3D

    subroutine cuda_Post_clean()
        implicit none
        deallocate(cnts_pr, disp_pr, cnts_aggr, disp_aggr)

    end subroutine cuda_Post_clean

    real(rp) function calc_C2g(X, i, j, k, im, jm, km) result(Xg)
        use mpi_subdomain,  only : dx1, dx2, dx3, dmx1, dmx2, dmx3
        implicit none
        real(rp), dimension(0:,0:,0:), intent(in) :: X
        integer, intent(in) :: i, j, k, im, jm, km
        real(rp) :: X1, X2, X_y1, X_y2

        X1   = ( X(im,jm,k)*dx3(km)+X(im,jm,km)*dx3(k) )/( dx3(k)+dx3(km) )
        X2   = ( X(i ,jm,k)*dx3(km)+X(i ,jm,km)*dx3(k) )/( dx3(k)+dx3(km) )
        X_y1 = ( X2        *dx1(im)+X1         *dx1(i) )/( dx1(i)+dx1(im) )
        X1   = ( X(im,j ,k)*dx3(km)+X(im,j ,km)*dx3(k) )/( dx3(k)+dx3(km) )
        X2   = ( X(i ,j ,k)*dx3(km)+X(i ,j ,km)*dx3(k) )/( dx3(k)+dx3(km) )
        X_y2 = ( X2        *dx1(im)+X1         *dx1(i) )/( dx1(i)+dx1(im) )
        Xg   = ( X_y2      *dx2(jm)+X_y1       *dx2(j) )/( dx2(j)+dx2(jm) )
    end function calc_C2g

end module cuda_Post