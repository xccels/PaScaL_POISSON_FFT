!>
!> @brief       Module for post-treatment for Poisson equation.
!>
module mpi_Post

    use mpi_topology
    use global
    use mpi_subdomain

    implicit none

    private

    !> @{ Control flag for output print
    logical :: FirstPrint1=.true.,FirstPrint2=.true.,FirstPrint3=.true.
    !> @}

    !> @{ Function for divergence, CFL, and output print
    public :: mpi_Post_error
    !> @}
    

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
    public :: mpi_Post_allocation

    public :: mpi_Post_FileOut_InstanField              ! Instanteneous filed print in ascii format
    
    public :: mpi_Post_FileIn_Continue_Single_Process_IO
    public :: mpi_Post_FileOut_Continue_Single_Process_IO

    public :: mpi_Post_FileIn_Continue_Single_Process_IO_with_Aggregation
    public :: mpi_Post_FileOut_Continue_Single_Process_IO_with_Aggregation

    public :: mpi_Post_FileIn_Continue_MPI_IO
    public :: mpi_Post_FileOut_Continue_MPI_IO

    public :: mpi_Post_FileIn_Continue_MPI_IO_with_Aggregation
    public :: mpi_Post_FileOut_Continue_MPI_IO_with_Aggregation

    public :: mpi_Post_FileIn_Continue_Post_Reassembly_IO
    public :: mpi_Post_FileOut_Continue_Post_Reassembly_IO

    !> @}

    character(len=17) :: xyzrank            !> Rank information in filename

    contains

    !>
    !> @brief       Calculate error
    !> @param       myrank          Rank for print
    !> @param       Pin             Numerical solution for Poisson equation
    !> @param       exact_sol_in    Exact solution for Poisson equation
    !> @param       rms             MSE
    !>
    subroutine mpi_Post_error(myrank, Pin, exact_sol_in, rms)

        use MPI

        use mpi_subdomain,  only : n1msub, n2msub, n3msub
        use mpi_subdomain,  only : n1sub, n2sub, n3sub
        use mpi_subdomain,  only : n1m, n2m, n3m

        implicit none

        integer :: myrank

        double precision, dimension(0:n1sub,0:n2sub,0:n3sub) :: Pin, exact_sol_in
        double precision :: rms_local, rms

        !> @{ Local indexing variable
        integer :: i,j,k
        !> @}


        integer :: ierr
        
        rms_local = 0.0d0
        rms = 0.0d0
        
        ! Find rms in the subdomain 
        do k = 1, n3msub
            do j = 1, n2msub
                do i = 1, n1msub
                    rms_local = rms_local + (Pin(i,j,k) - exact_sol_in(i,j,k))**2
                end do
            end do
        end do

        ! Find maximum divergence in the global domain with three step communication
        call MPI_ALLREDUCE(rms_local, rms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)
        if(myrank.eq.0) print '(a,e20.10)','[Poisson] RMS = ',sqrt(rms/n1m/n2m/n3m)
        
    end subroutine mpi_Post_error



    !>
    !> @brief       Write instantaneous field variable P for post-processing in ascii format to file
    !> @detail      An example subroutine
    !> @param       myrank          Rank for print
    !> @param       Pin             Pressure field for file write
    !>
    subroutine mpi_Post_FileOut_InstanField(myrank,Pin)

        use mpi
        
        implicit none
        
        integer :: myrank
        double precision, dimension(0:n1sub, 0:n2sub, 0:n3sub) :: Pin

        !> @{ File name
        character(len=22) :: filename_instantfieldXY
        character(len=20) :: filename_instantfield
        !> @}

        integer :: i,j,k

        ! Define range
        filename_instantfieldXY='Output_instantfield_XY'
        filename_instantfield  ='Output_instantfield'


        ! Write field variables in X-Y plane
            if(comm_1d_x3%myrank==comm_1d_x3%nprocs-1) then
                open(unit=myrank,file=trim(dir_instantfield)//trim(filename_instantfieldXY)//trim(xyzrank)//'.plt', position='append')
                    if(FirstPrint1) then
                        write(myrank,*) 'VARIABLES="X","Y","Z","P"'  !--
                        write(myrank,*) 'zone t="',1,'"','i=',n1sub+1,'j=',n2sub+1,'k=',1
                        k= (n3sub-1)/2
                        do j=0,n2sub
                        do i=0,n1sub
                            write(myrank,'(3D20.10,1D30.20)') x1_sub(i),x2_sub(j),x3_sub(k),Pin(i,j,k)
                        enddo
                        enddo

                        FirstPrint1=.false.
                    endif
                close(myrank)
            endif


        ! ! Write field variables in 3D domain
        !     open(unit=myrank,file=trim(dir_instantfield)//trim(filename_instantfield)//trim(xyzrank)//'.plt', position='append')
        !         if(FirstPrint2) then
        !             write(myrank,*) 'VARIABLES="X","Y","Z","P"'  !--
        !             write(myrank,*) 'zone t="',1,'"','i=',n1sub+1,'j=',n2sub+1,'k=',n3sub+1
                        
        !             do k=0,n3sub
        !             do j=0,n2sub 
        !             do i=0,n1sub
        !                 write(myrank,'(3D20.10,1D30.20)') x1_sub(i),x2_sub(j),x3_sub(k),Pin(i,j,k)
        !             enddo
        !             enddo
        !             enddo 

        !             FirstPrint2=.false.

        !         endif
        !     close(myrank)

    end subroutine mpi_Post_FileOut_InstanField

    !>
    !> @brief       Initialize the required variables for file IO of field variable
    !> @detail      File IO of field variables is for restart, so it writes whole field variables
    !>              to binary file.
    !> @param       chunk_size  Aggregation size of MPI processes
    !>
    subroutine mpi_Post_allocation(chunk_size)

        implicit none

        integer :: chunk_size

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
        write(xyzrank, '(I2.2,1A1,I2.2,1A1,I2.2)' ) comm_1d_x1%myrank,'_',   &
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

    end subroutine mpi_Post_allocation

    !>
    !> @brief       Read field variables using single process IO
    !> @param       myrank          Rank for print
    !> @param       Pin             Pressure field for file write
    !>
    subroutine mpi_Post_FileIn_Continue_Single_Process_IO(myrank,Pin)

        implicit none

        integer :: myrank
        double precision, dimension(0:n1sub, 0:n2sub, 0:n3sub) :: Pin

        ! Read P
        open(myrank, FILE=trim(dir_cont_filein)//'cont_P_'//xyzrank//'.bin', FORM='UNFORMATTED', &
            STATUS='OLD', ACTION='READ', ACCESS='DIRECT', RECL=sizeof(Pin) )
        read(myrank, REC=1) Pin
        close(myrank)

        if(myrank.eq.0) print '(a)', 'Read continue file using single process IO'

    end subroutine mpi_Post_FileIn_Continue_Single_Process_IO

    !>
    !> @brief       write field variables using single process IO
    !> @param       myrank          Rank for print
    !> @param       Pin             Pressure field for file write
    !>
    subroutine mpi_Post_FileOut_Continue_Single_Process_IO(myrank,Pin)

        implicit none

        integer :: myrank
        double precision, dimension(0:n1sub, 0:n2sub, 0:n3sub) :: Pin

        ! Write P
        open(myrank, FILE=trim(dir_cont_fileout)//'cont_P_'//xyzrank//'.bin', FORM='UNFORMATTED', &
            STATUS='REPLACE', ACTION='WRITE', ACCESS='DIRECT', RECL=sizeof(Pin) )
        write(myrank, REC=1) Pin
        close(myrank)

        if(myrank.eq.0) print '(a)', 'Write continue file using single process IO'

    end subroutine mpi_Post_FileOut_Continue_Single_Process_IO

    !>
    !> @brief       Read field variables using single process IO with explicit aggregation
    !> @param       myrank          Rank for print
    !> @param       Pin             Pressure field for file write
    !>
    subroutine mpi_Post_FileIn_Continue_Single_Process_IO_with_Aggregation(myrank,Pin)
        use mpi
        implicit none
        integer :: myrank
        double precision, dimension(0:n1sub, 0:n2sub, 0:n3sub) :: Pin

        integer :: ierr
        double precision, allocatable, dimension(:,:,:) :: var_chunk
        
        ! The master rank for data aggregation allocates chunk array which have aggregated data
        if(comm_IO_aggregation%myrank.eq.0) then
            allocate( var_chunk(n1msub, n2msub, n3msub * comm_IO_aggregation%nprocs) )
        endif

        ! The master rank for data aggregation reads P
        if(comm_IO_aggregation%myrank.eq.0) then
            open(myrank, FILE=trim(dir_cont_filein)//'cont_P_'//xyzrank//'.bin', FORM='UNFORMATTED', &
                STATUS='OLD', ACTION='READ', ACCESS='DIRECT', RECL=sizeof(var_chunk) )
            read(myrank, REC=1) var_chunk
            close(myrank)
        endif

        ! The master rank for data aggregation scatter the readed P to its local communicator using the defined DDT
        call MPI_Scatterv( var_chunk, cnts_aggr, disp_aggr, ddtype_aggregation_chunk, Pin, 1, ddtype_inner_domain, 0, &
                        comm_IO_aggregation%mpi_comm, ierr)


        ! The master rank for data aggregation deallocates the chunk array for aggregation
        if(comm_IO_aggregation%myrank.eq.0) then
            deallocate( var_chunk )
        endif

        ! Ghostcell update
        call mpi_subdomain_ghostcell_update(Pin)

        if(myrank.eq.0) print '(a)', 'Read continue file using single process IO with aggregation'

    end subroutine mpi_Post_FileIn_Continue_Single_Process_IO_with_Aggregation

    !>
    !> @brief       Write field variables using single process IO with explicit aggregation
    !> @param       myrank          Rank for print
    !> @param       Pin             Pressure field for file write
    !>
    subroutine mpi_Post_FileOut_Continue_Single_Process_IO_with_Aggregation(myrank,Pin)

        use mpi

        implicit none

        integer :: myrank
        double precision, dimension(0:n1sub, 0:n2sub, 0:n3sub) :: Pin

        double precision, allocatable, dimension(:,:,:) :: var_chunk
        integer :: ierr
        
        ! The master rank for data aggregation allocates chunk array which have aggregated data
        if(comm_IO_aggregation%myrank.eq.0) then
            allocate( var_chunk(n1msub, n2msub, n3msub * comm_IO_aggregation%nprocs) )
        endif

        ! The master rank for data aggregation gathers P to its chunk array
        call MPI_Gatherv( Pin, 1, ddtype_inner_domain, var_chunk, cnts_aggr, disp_aggr, ddtype_aggregation_chunk, 0, &
                        comm_IO_aggregation%mpi_comm, ierr)

        ! The master rank for data aggregation writes P
        if(comm_IO_aggregation%myrank.eq.0) then
            open(myrank, FILE=trim(dir_cont_fileout)//'cont_P_'//xyzrank//'.plt', FORM='UNFORMATTED', &
                STATUS='REPLACE', ACTION='WRITE', ACCESS='DIRECT', RECL=sizeof(var_chunk) )
            write(myrank, REC=1) var_chunk
            close(myrank)
        endif

        ! The master rank for data aggregation deallocates the chunk array for aggregation
        if(comm_IO_aggregation%myrank.eq.0) then
            deallocate( var_chunk )
        endif

        if(myrank.eq.0) print '(a)', 'Write continue file using single process IO with aggregation'

    end subroutine mpi_Post_FileOut_Continue_Single_Process_IO_with_Aggregation

    !>
    !> @brief       Read field variables from a single file using MPI IO
    !> @param       myrank          Rank for print
    !> @param       Pin             Pressure field for file write
    !>
    subroutine mpi_Post_FileIn_Continue_MPI_IO(myrank,Pin)

        use mpi

        implicit none

        integer :: myrank
        double precision, dimension(0:n1sub, 0:n2sub, 0:n3sub) :: Pin

        integer :: filep, ierr
        integer(kind=MPI_OFFSET_KIND) :: disp

        disp = 0

        ! Read P from a single file using MPI IO
        call MPI_File_open(MPI_COMM_WORLD, trim(dir_cont_filein)//'cont_P.bin', MPI_MODE_RDONLY, MPI_INFO_NULL, filep, ierr)
        call MPI_File_set_view(filep, disp, MPI_DOUBLE_PRECISION, ddtype_global_domain_MPI_IO, 'native', MPI_INFO_NULL, ierr)
        call MPI_File_read(filep, Pin, 1, ddtype_inner_domain, MPI_STATUS_IGNORE, ierr)
        call MPI_File_close(filep, ierr)

        ! Ghostcell update
        call mpi_subdomain_ghostcell_update(Pin)

        if(myrank.eq.0) print '(a)', 'Read continue file using MPI IO'

    end subroutine mpi_Post_FileIn_Continue_MPI_IO

    !>
    !> @brief       Write field variables to a single file using MPI IO
    !> @param       myrank          Rank for print
    !> @param       Pin             Pressure field for file write
    !>
    subroutine mpi_Post_FileOut_Continue_MPI_IO(myrank,Pin)

        use mpi

        implicit none

        integer :: myrank
        double precision, dimension(0:n1sub, 0:n2sub, 0:n3sub) :: Pin

        integer :: filep, ierr
        integer(kind=MPI_OFFSET_KIND) :: disp

        disp = 0

        ! Write P to a single file using MPI IO
        call MPI_File_open(MPI_COMM_WORLD, trim(dir_cont_fileout)//'cont_P.bin', MPI_MODE_WRONLY+MPI_MODE_CREATE, MPI_INFO_NULL, filep, ierr)
        call MPI_File_set_view(filep, disp, MPI_DOUBLE_PRECISION, ddtype_global_domain_MPI_IO, 'native', MPI_INFO_NULL, ierr)
        call MPI_File_write(filep, Pin, 1, ddtype_inner_domain, MPI_STATUS_IGNORE, ierr)
        call MPI_File_close(filep, ierr)

        if(myrank.eq.0) print '(a)', 'Write continue file using MPI IO'

    end subroutine mpi_Post_FileOut_Continue_MPI_IO

    !>
    !> @brief       Read field variables from a single file using MPI IO with explicit aggregation
    !> @param       myrank          Rank for print
    !> @param       Pin             Pressure field for file write
    !>
    subroutine mpi_Post_FileIn_Continue_MPI_IO_with_Aggregation(myrank,Pin)

        use mpi

        implicit none

        integer :: myrank
        double precision, dimension(0:n1sub, 0:n2sub, 0:n3sub) :: Pin

        integer :: filep, ierr
        integer(kind=MPI_OFFSET_KIND) :: disp
        
        double precision, allocatable, dimension(:,:,:) :: var_chunk

        ! The master rank for data aggregation allocates chunk array which have aggregated data
        if(comm_IO_aggregation%myrank.eq.0) then
            allocate( var_chunk(n1msub, n2msub, n3msub * comm_IO_aggregation%nprocs) )
        endif

        disp = 0

        ! The master rank for data aggregation reads P from a single file using MPI IO
        if(comm_IO_aggregation%myrank.eq.0) then
            call MPI_File_open(comm_IO_master%mpi_comm, trim(dir_cont_filein)//'cont_P.bin', MPI_MODE_RDONLY, MPI_INFO_NULL, filep, ierr)
            call MPI_File_set_view(filep, disp, MPI_DOUBLE_PRECISION, ddtype_global_domain_MPI_IO_aggregation, 'native', MPI_INFO_NULL, ierr)
            call MPI_File_read(filep, var_chunk, size(var_chunk), MPI_DOUBLE_PRECISION, MPI_STATUS_IGNORE, ierr)
            call MPI_File_close(filep, ierr)
        endif

        ! The master rank for data aggregation scatter the readed P to its local communicator using the defined DDT
        call MPI_Scatterv( var_chunk, cnts_aggr, disp_aggr, ddtype_aggregation_chunk, Pin, 1, ddtype_inner_domain, 0, comm_IO_aggregation%mpi_comm, ierr)

        if(comm_IO_aggregation%myrank.eq.0) then
            deallocate( var_chunk )
        endif

        ! Ghostcell update
        call mpi_subdomain_ghostcell_update(Pin)
    
        if(myrank.eq.0) print '(a)', 'Read continue file using MPI IO with aggregation'

    end subroutine mpi_Post_FileIn_Continue_MPI_IO_with_Aggregation

    !>
    !> @brief       Write field variables to a single file using MPI IO with explicit aggregation
    !> @param       myrank          Rank for print
    !> @param       Pin             Pressure field for file write
    !>
    subroutine mpi_Post_FileOut_Continue_MPI_IO_with_Aggregation(myrank,Pin)

        use mpi

        implicit none

        integer :: myrank
        double precision, dimension(0:n1sub, 0:n2sub, 0:n3sub) :: Pin

        integer :: filep, ierr
        integer(kind=MPI_OFFSET_KIND) :: disp
        
        double precision, allocatable, dimension(:,:,:) :: var_chunk

        ! The master rank for data aggregation allocates chunk array which have aggregated data
        if(comm_IO_aggregation%myrank.eq.0) then
            allocate( var_chunk(n1msub, n2msub, n3msub * comm_IO_aggregation%nprocs) )
        endif

        disp = 0

        ! The master rank for data aggregation gathers P to its chunk array
        call MPI_Gatherv( Pin, 1, ddtype_inner_domain, var_chunk, cnts_aggr, disp_aggr, ddtype_aggregation_chunk, 0, &
                        comm_IO_aggregation%mpi_comm, ierr)

        ! The master rank for data aggregation writes P to a single file using MPI IO
        if(comm_IO_aggregation%myrank.eq.0) then
            call MPI_File_open(comm_IO_master%mpi_comm, trim(dir_cont_fileout)//'cont_P.bin', MPI_MODE_WRONLY+MPI_MODE_CREATE, MPI_INFO_NULL, filep, ierr)
            call MPI_File_set_view(filep, disp, MPI_DOUBLE_PRECISION, ddtype_global_domain_MPI_IO_aggregation, 'native', MPI_INFO_NULL, ierr)
            call MPI_File_write(filep, var_chunk, size(var_chunk), MPI_DOUBLE_PRECISION, MPI_STATUS_IGNORE, ierr)
            call MPI_File_close(filep, ierr)
        endif

        ! The master rank for data aggregation deallocates the chunk array for aggregation
        if(comm_IO_aggregation%myrank.eq.0) then
            deallocate( var_chunk )
        endif

        if(myrank.eq.0) print '(a)', 'Write continue file using MPI IO with aggregation'

    end subroutine mpi_Post_FileOut_Continue_MPI_IO_with_Aggregation

    !>
    !> @brief       Read field variables using post-reassembly IO from a single file
    !> @param       myrank          Rank for print
    !> @param       Pin             Pressure field for file write
    !>
    subroutine mpi_Post_FileIn_Continue_Post_Reassembly_IO(myrank,Pin)

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
        call mpi_subdomain_ghostcell_update(Pin)
    
        if(myrank.eq.0) print '(a)', 'Read continue file using post reassembly IO'

    end subroutine mpi_Post_FileIn_Continue_Post_Reassembly_IO

    !>
    !> @brief       Read field variables using post-reassembly IO from a single file
    !> @param       myrank          Rank for print
    !> @param       Pin             Pressure field for file write
    !>
    subroutine mpi_Post_FileOut_Continue_Post_Reassembly_IO(myrank,Pin)

        use mpi

        implicit none

        integer :: myrank
        double precision, dimension(0:n1sub, 0:n2sub, 0:n3sub) :: Pin

        double precision, allocatable, dimension(:,:,:) :: var_all
        integer :: ierr

        ! Rank 0 allocates chunk array which have all field array data
        if(myrank.eq.0) allocate( var_all( n1m, n2m, n3m) )

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

    end subroutine mpi_Post_FileOut_Continue_Post_Reassembly_IO


end module mpi_Post