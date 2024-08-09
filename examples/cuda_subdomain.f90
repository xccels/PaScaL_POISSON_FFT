module cuda_subdomain
    use MPI
    use global
    use cudafor
    use openacc
    use nvtx
    use mpi_subdomain, only : n1sub, n2sub, n3sub, n1msub, n2msub, n3msub

    implicit none
    
    ! Coordinates of grid points in the subdomain in cuda
    real(rp), device, target, allocatable, dimension(:)     ::   x1_d_c,   x2_d_c,   x3_d_c
    real(rp), device, target, allocatable, dimension(:)     ::  dx1_d_c,  dx2_d_c,  dx3_d_c
    real(rp), device, target, allocatable, dimension(:)     :: dmx1_d_c, dmx2_d_c, dmx3_d_c
    integer,  device, target, allocatable, dimension(:)     :: iC_BC_d_c,iS_BC_d_c,jC_BC_d_c,jS_BC_d_c,kC_BC_d_c,kS_BC_d_c
    integer,  device, target, allocatable, dimension(:)     :: iC_index_d_c, iS_index_d_c, jC_index_d_c, jS_index_d_c, kC_index_d_c, kS_index_d_c

    real(rp), device, pointer, dimension(:)     ::   x1_d,   x2_d,   x3_d
    real(rp), device, pointer, dimension(:)     ::  dx1_d,  dx2_d,  dx3_d
    real(rp), device, pointer, dimension(:)     :: dmx1_d, dmx2_d, dmx3_d
    integer,  device, pointer, dimension(:)     :: iC_BC_d,iS_BC_d,jC_BC_d,jS_BC_d,kC_BC_d,kS_BC_d

    ! For SEM
    real(rp), device, pointer, dimension(:)     ::   x1_g_d  ,   x2_g_d  ,   x3_g_d
    real(rp), device, pointer, dimension(:)     ::  dx1_g_d  ,  dx2_g_d  ,  dx3_g_d
    real(rp), device, pointer, dimension(:)     :: dmx1_g_d  , dmx2_g_d  , dmx3_g_d
    real(rp), device, target, allocatable, dimension(:)  ::   x1_g_d_c,   x2_g_d_c,   x3_g_d_c
    real(rp), device, target, allocatable, dimension(:)  ::  dx1_g_d_c,  dx2_g_d_c,  dx3_g_d_c
    real(rp), device, target, allocatable, dimension(:)  :: dmx1_g_d_c, dmx2_g_d_c, dmx3_g_d_c

    ! Block and thread dimension
    integer             :: block_in_x, block_in_y, block_in_z
    integer             :: share_size, share_size_pascal, share_size_fft
    type(dim3)          :: blocks, threads, threads_tdma, threads_fft

    ! Communication buffer
    real(rp), device, target, allocatable, dimension(:)   :: sbuf_0_temp    , sbuf_1_temp    , rbuf_0_temp    , rbuf_1_temp
    integer , device, target, allocatable, dimension(:)   :: sbuf_0_temp_int, sbuf_1_temp_int, rbuf_0_temp_int, rbuf_1_temp_int

    ! For device
    real(rp), device, target, allocatable, dimension(:) :: RHS_buff1, RHS_buff2
    real(rp), device, target, allocatable, dimension(:) :: API_ptr, ACI_ptr, AMI_ptr, APJ_ptr, ACJ_ptr, AMJ_ptr, APK_ptr, ACK_ptr, AMK_ptr

    real(rp), device, target, allocatable, dimension(:) :: temp1, temp2

    interface cuda_subdomain_ghostcell_update
        module procedure cuda_subdomain_ghostcell_update_scalar   ! For 3D Scalar variables
    end interface

    interface ghostcell_update_on_direction
        module procedure ghostcell_update_on_direction_real       ! For Real variables
    end interface

    contains

    subroutine cuda_subdomain_environment()
        use mpi_topology, only : nprocs, myrank
        implicit none
        integer :: istat, nDevices, local_rank, ierr
    
        ! Check the total number of available CUDA devices
        istat = cudaGetDeviceCount(nDevices)
        if (istat /= cudaSuccess .or. nDevices < 1) then
            if (myrank == 0) then
                write(*,*) "Error: GPU devices not found or cudaGetDeviceCount failed."
            endif
            call MPI_Abort(MPI_COMM_WORLD, istat, ierr) ! Use the error code returned by the CUDA function
            return
        endif
    
        ! Select and initialize the GPU based on the MPI rank
        local_rank = mod(myrank, nDevices)
        istat = cudaSetDevice(local_rank)
        if (istat /= cudaSuccess) then
            write(*,*) "Error: Rank", myrank, "failed to set GPU device", local_rank
            call MPI_Abort(MPI_COMM_WORLD, istat, ierr) ! Use the error code returned by the CUDA function
            return
        endif
    
        ! Initialize the GPU with OpenACC
        call acc_set_device_num(local_rank, acc_device_nvidia)
        ! Report the assignment of CPUs to GPUs
        write(*,*) "Rank", myrank, "assigned to GPU", local_rank

        call MPI_Barrier(MPI_COMM_WORLD, ierr);

    end subroutine cuda_subdomain_environment    

    subroutine cuda_subdomain_initial()

        call cuda_subdomain_mesh()
        call cuda_subdomain_blockandthreads()
        
    end subroutine cuda_subdomain_initial

    subroutine cuda_subdomain_mesh
        use mpi_subdomain, only : x1, x2, x3, dx1, dx2, dx3, dmx1, dmx2, dmx3
        use mpi_subdomain, only : x1_g, x2_g, x3_g, dx1_g, dx2_g, dx3_g, dmx1_g, dmx2_g, dmx3_g
        use interface    , only : allocate_and_init, allocate_and_init_device
        implicit none

        ! Allocate subdomain variables.
        allocate( x1_d_c(0:n1sub), dmx1_d_c(0:n1sub), dx1_d_c(0:n1sub));  
        allocate( x2_d_c(0:n2sub), dmx2_d_c(0:n2sub), dx2_d_c(0:n2sub));  
        allocate( x3_d_c(0:n3sub), dmx3_d_c(0:n3sub), dx3_d_c(0:n3sub));  

        ! Allocate global domain variables.
        allocate( x1_g_d_c(0:n1), dmx1_g_d_c(0:n1), dx1_g_d_c(0:n1))
        allocate( x2_g_d_c(0:n2), dmx2_g_d_c(0:n2), dx2_g_d_c(0:n2))
        allocate( x3_g_d_c(0:n3), dmx3_g_d_c(0:n3), dx3_g_d_c(0:n3))
        
          x1_d_c =   x1  ;   x1_d =>   x1_d_c
          x2_d_c =   x2  ;   x2_d =>   x2_d_c
          x3_d_c =   x3  ;   x3_d =>   x3_d_c
         dx1_d_c =  dx1  ;  dx1_d =>  dx1_d_c
         dx2_d_c =  dx2  ;  dx2_d =>  dx2_d_c
         dx3_d_c =  dx3  ;  dx3_d =>  dx3_d_c
        dmx1_d_c = dmx1  ; dmx1_d => dmx1_d_c
        dmx2_d_c = dmx2  ; dmx2_d => dmx2_d_c
        dmx3_d_c = dmx3  ; dmx3_d => dmx3_d_c

        x1_g_d_c =   x1_g  ;   x1_g_d =>   x1_g_d_c
        x2_g_d_c =   x2_g  ;   x2_g_d =>   x2_g_d_c
        x3_g_d_c =   x3_g  ;   x3_g_d =>   x3_g_d_c
       dx1_g_d_c =  dx1_g  ;  dx1_g_d =>  dx1_g_d_c
       dx2_g_d_c =  dx2_g  ;  dx2_g_d =>  dx2_g_d_c
       dx3_g_d_c =  dx3_g  ;  dx3_g_d =>  dx3_g_d_c
      dmx1_g_d_c = dmx1_g  ; dmx1_g_d => dmx1_g_d_c
      dmx2_g_d_c = dmx2_g  ; dmx2_g_d => dmx2_g_d_c
      dmx3_g_d_c = dmx3_g  ; dmx3_g_d => dmx3_g_d_c   

        ! Ghostcell communication temp array
        allocate( sbuf_0_temp((max0(n1sub,n2sub,n3sub)+1)*(max0(n1sub,n2sub,n3sub)+1)), sbuf_1_temp((max0(n1sub,n2sub,n3sub)+1)*(max0(n1sub,n2sub,n3sub)+1)))
        allocate( rbuf_0_temp((max0(n1sub,n2sub,n3sub)+1)*(max0(n1sub,n2sub,n3sub)+1)), rbuf_1_temp((max0(n1sub,n2sub,n3sub)+1)*(max0(n1sub,n2sub,n3sub)+1)))
        sbuf_0_temp = real(0.0,rp); sbuf_1_temp = real(0.0,rp);
        rbuf_0_temp = real(0.0,rp); rbuf_1_temp = real(0.0,rp);

        ! Ghostcell communication temp array
        allocate( sbuf_0_temp_int((max0(n1sub,n2sub,n3sub)+3)*(max0(n1sub,n2sub,n3sub)+3)*2), sbuf_1_temp_int((max0(n1sub,n2sub,n3sub)+3)*(max0(n1sub,n2sub,n3sub)+3)*2))
        allocate( rbuf_0_temp_int((max0(n1sub,n2sub,n3sub)+3)*(max0(n1sub,n2sub,n3sub)+3)*2), rbuf_1_temp_int((max0(n1sub,n2sub,n3sub)+3)*(max0(n1sub,n2sub,n3sub)+3)*2))
        sbuf_0_temp_int = 0; sbuf_1_temp_int = 0;
        rbuf_0_temp_int = 0; rbuf_1_temp_int = 0;

    end subroutine cuda_subdomain_mesh

    subroutine cuda_subdomain_blockandthreads()     
        implicit none
        integer :: ierr

        block_in_x = ceiling(real(n1msub)/real(thread_in_x)) !n1msub/thread_in_x
        if( (block_in_x.eq.0) .or. (mod((n1sub-1), thread_in_x).ne.0) ) then
            print '(a,i5,a,i5,a)', '[Warning] n1sub-1 is not a multiple of thread_in_x. Adjusted block_in_x = ', block_in_x, ', n1sub-1 = ', n1sub-1, '. This may not be fully optimized.'
        endif
        block_in_y = ceiling(real(n2msub)/real(thread_in_y)) !n2msub/thread_in_y
        if( (block_in_y.eq.0) .or. (mod((n2sub-1), thread_in_y).ne.0)) then
            print '(a,i5,a,i5,a)', '[Warning] n2sub-1 is not a multiple of thread_in_y. Adjusted block_in_y = ', block_in_y, ', n2sub-1 = ', n2sub-1, '. This may not be fully optimized.'
        endif
        block_in_z = ceiling(real(n3msub)/real(thread_in_z)) !n3msub/thread_in_z
        if( (block_in_z.eq.0) .or. (mod((n3sub-1), thread_in_z).ne.0)) then
            print '(a,i5,a,i5,a)', '[Warning] n3sub-1 is not a multiple of thread_in_z. Adjusted block_in_z = ', block_in_z, ', n3sub-1 = ', n3sub-1, '. This may not be fully optimized.'
        endif

        ! Set thread and block
        threads = dim3(thread_in_x, thread_in_y, thread_in_z)
        blocks  = dim3( block_in_x,  block_in_y,  block_in_z)

        threads_tdma = dim3(thread_in_x_pascal, thread_in_y_pascal, 1)
        threads_fft  = dim3(thread_in_fft_pascal, 1, 1)

        share_size        = rp*(2+thread_in_x)*(2+thread_in_y)*(2+thread_in_z)
        share_size_pascal = rp*(1+thread_in_x_pascal)*thread_in_y_pascal
        share_size_fft    = rp*(1+thread_in_fft_pascal)

    end subroutine cuda_subdomain_blockandthreads

    subroutine cuda_subdomain_DtoH(Devicedata, Hostdata) 
        real(rp), device, dimension(0:n1sub, 0:n2sub, 0:n3sub), intent(inout)  :: Devicedata
        real(rp),         dimension(0:n1sub, 0:n2sub, 0:n3sub), intent(inout)  :: Hostdata
        Hostdata=Devicedata
    end subroutine cuda_subdomain_DtoH

    subroutine cuda_subdomain_HtoD(Hostdata, Devicedata)
        real(rp), device, dimension(0:n1sub, 0:n2sub, 0:n3sub), intent(inout)  :: Devicedata
        real(rp),         dimension(0:n1sub, 0:n2sub, 0:n3sub), intent(inout)  :: Hostdata
        Devicedata=Hostdata
    end subroutine cuda_subdomain_HtoD


    subroutine cuda_subdomain_ghostcell_update_scalar(Value_sub_d)
        implicit none
        real(rp), device, target, dimension(0:n1sub, 0:n2sub, 0:n3sub), intent(inout)  :: Value_sub_d
        real(rp), pointer, device, dimension(:,:,:) :: Value_sub_ptr

        Value_sub_ptr(0:n1sub,0:n2sub,0:n3sub) => Value_sub_d(0:,0:,0:)
        call cuda_subdomain_ghostcell_update_real(Value_sub_ptr)
    end subroutine cuda_subdomain_ghostcell_update_scalar


    subroutine cuda_subdomain_ghostcell_update_real(Value_sub_d)
        use mpi_topology,   only : comm_1d_x1, comm_1d_x2, comm_1d_x3
        implicit none
        real(rp), device, dimension(0:, 0:, 0:), intent(inout)  :: Value_sub_d
        real(rp), pointer, device, dimension(:,:)   :: sbuf_x0_d, sbuf_x1_d, sbuf_y0_d, sbuf_y1_d, sbuf_z0_d, sbuf_z1_d
        real(rp), pointer, device, dimension(:,:)   :: rbuf_x0_d, rbuf_x1_d, rbuf_y0_d, rbuf_y1_d, rbuf_z0_d, rbuf_z1_d   
        
        sbuf_x0_d(0:n2sub,0:n3sub) => sbuf_0_temp; sbuf_x1_d(0:n2sub,0:n3sub) => sbuf_1_temp
        rbuf_x0_d(0:n2sub,0:n3sub) => rbuf_0_temp; rbuf_x1_d(0:n2sub,0:n3sub) => rbuf_1_temp
        call ghostcell_update_on_direction(comm_1d_x1, 1, pbc1, sbuf_x0_d, sbuf_x1_d, rbuf_x0_d, rbuf_x1_d, n1sub, n2sub, n3sub, Value_sub_d)
        nullify(sbuf_x0_d, sbuf_x1_d, rbuf_x0_d, rbuf_x1_d)

        sbuf_y0_d(0:n1sub,0:n3sub) => sbuf_0_temp; sbuf_y1_d(0:n1sub,0:n3sub) => sbuf_1_temp
        rbuf_y0_d(0:n1sub,0:n3sub) => rbuf_0_temp; rbuf_y1_d(0:n1sub,0:n3sub) => rbuf_1_temp
        call ghostcell_update_on_direction(comm_1d_x2, 2, pbc2, sbuf_y0_d, sbuf_y1_d, rbuf_y0_d, rbuf_y1_d, n2sub, n1sub, n3sub, Value_sub_d)
        nullify(sbuf_y0_d, sbuf_y1_d, rbuf_y0_d, rbuf_y1_d)

        sbuf_z0_d(0:n1sub,0:n2sub) => sbuf_0_temp; sbuf_z1_d(0:n1sub,0:n2sub) => sbuf_1_temp
        rbuf_z0_d(0:n1sub,0:n2sub) => rbuf_0_temp; rbuf_z1_d(0:n1sub,0:n2sub) => rbuf_1_temp
        call ghostcell_update_on_direction(comm_1d_x3, 3, pbc3, sbuf_z0_d, sbuf_z1_d, rbuf_z0_d, rbuf_z1_d, n3sub, n1sub, n2sub, Value_sub_d)
        nullify(sbuf_z0_d, sbuf_z1_d, rbuf_z0_d, rbuf_z1_d)
    end subroutine cuda_subdomain_ghostcell_update_real

    subroutine ghostcell_update_on_direction_real(comm_1d, direction, pbc, sbuf_0_d, sbuf_1_d, rbuf_0_d, rbuf_1_d, nsub_a, nsub_b, nsub_c, Value_sub_d)
        use mpi_topology, only : cart_comm_1d
        implicit none
        type(cart_comm_1d), intent(in) :: comm_1d
        integer, intent(in) :: direction
        logical, intent(in) :: pbc
        real(rp), device, dimension(0:,0:), intent(inout) :: sbuf_0_d, sbuf_1_d, rbuf_0_d, rbuf_1_d
        integer, intent(in) :: nsub_a, nsub_b, nsub_c
        real(rp), device, intent(inout) :: Value_sub_d(0:,0:,0:)
        integer :: idx_b, idx_c, ierr
        integer, dimension(4) :: request
    
        !$cuf kernel do(2) <<< *,* >>>
        do idx_c = 0, nsub_c
        do idx_b = 0, nsub_b
            select case(direction)
            case(1)  ! X direction
                if(comm_1d%west_rank.ne.MPI_PROC_NULL) then
                    sbuf_0_d(idx_b,idx_c) = Value_sub_d(1       ,idx_b,idx_c)
                endif
                if(comm_1d%east_rank.ne.MPI_PROC_NULL) then
                    sbuf_1_d(idx_b,idx_c) = Value_sub_d(nsub_a-1,idx_b,idx_c)
                endif
            case(2)  ! Y direction
                if(comm_1d%west_rank.ne.MPI_PROC_NULL) then
                    sbuf_0_d(idx_b,idx_c) = Value_sub_d(idx_b,1       ,idx_c)
                endif
                if(comm_1d%east_rank.ne.MPI_PROC_NULL) then
                    sbuf_1_d(idx_b,idx_c) = Value_sub_d(idx_b,nsub_a-1,idx_c)
                endif
            case(3)  ! Z direction
                if(comm_1d%west_rank.ne.MPI_PROC_NULL) then
                    sbuf_0_d(idx_b,idx_c) = Value_sub_d(idx_b,idx_c,1       )
                endif
                if(comm_1d%east_rank.ne.MPI_PROC_NULL) then
                    sbuf_1_d(idx_b,idx_c) = Value_sub_d(idx_b,idx_c,nsub_a-1)
                endif
            end select
        enddo
        enddo
    
        ! MPI Area
        if( comm_1d%nprocs.eq.1 .and. pbc.eqv..true. ) then
            !$cuf kernel do(2) <<< *,* >>>
            do idx_c = 0, nsub_c
            do idx_b = 0, nsub_b
                rbuf_1_d(idx_b,idx_c) = sbuf_0_d(idx_b,idx_c)
                rbuf_0_d(idx_b,idx_c) = sbuf_1_d(idx_b,idx_c)
            enddo
            enddo
        else
            
            ierr = cudaStreamSynchronize()
            call MPI_Isend(sbuf_0_d, (nsub_b+1)*(nsub_c+1), MPI_real_type, comm_1d%west_rank, 111, comm_1d%mpi_comm, request(1), ierr)
            call MPI_Irecv(rbuf_1_d, (nsub_b+1)*(nsub_c+1), MPI_real_type, comm_1d%east_rank, 111, comm_1d%mpi_comm, request(2), ierr)
            call MPI_Irecv(rbuf_0_d, (nsub_b+1)*(nsub_c+1), MPI_real_type, comm_1d%west_rank, 222, comm_1d%mpi_comm, request(3), ierr)
            call MPI_Isend(sbuf_1_d, (nsub_b+1)*(nsub_c+1), MPI_real_type, comm_1d%east_rank, 222, comm_1d%mpi_comm, request(4), ierr)
            call MPI_Waitall(4, request, MPI_STATUSES_IGNORE, ierr)    
        endif
    
        !$cuf kernel do(2) <<< *,* >>>
        do idx_c = 0, nsub_c
        do idx_b = 0, nsub_b
            select case(direction)
            case(1)  ! X direction
                if(comm_1d%west_rank.ne.MPI_PROC_NULL) then
                    Value_sub_d(0     ,idx_b,idx_c) = rbuf_0_d(idx_b,idx_c)
                endif
                if(comm_1d%east_rank.ne.MPI_PROC_NULL) then
                    Value_sub_d(nsub_a,idx_b,idx_c) = rbuf_1_d(idx_b,idx_c)
                endif
            case(2)  ! Y direction
                if(comm_1d%west_rank.ne.MPI_PROC_NULL) then
                    Value_sub_d(idx_b,0     ,idx_c) = rbuf_0_d(idx_b,idx_c)
                endif
                if(comm_1d%east_rank.ne.MPI_PROC_NULL) then
                    Value_sub_d(idx_b,nsub_a,idx_c) = rbuf_1_d(idx_b,idx_c)
                endif
            case(3)  ! Z direction
                if(comm_1d%west_rank.ne.MPI_PROC_NULL) then
                    Value_sub_d(idx_b,idx_c,0     ) = rbuf_0_d(idx_b,idx_c)
                endif
                if(comm_1d%east_rank.ne.MPI_PROC_NULL) then
                    Value_sub_d(idx_b,idx_c,nsub_a) = rbuf_1_d(idx_b,idx_c)
                endif
            end select
        enddo
        enddo
    end subroutine ghostcell_update_on_direction_real


    subroutine cuda_neumann_BC(X_d)
        use mpi_topology, only: comm_1d_x1, comm_1d_x2, comm_1d_x3
        implicit none 
        real(rp), device, dimension(0:,0:,0:) :: X_d
    
        call apply_bc_on_direction(X_d, comm_1d_x1, dmx1_d, pbc1, n1msub, n1sub, n2sub, n3sub, 1)
        call apply_bc_on_direction(X_d, comm_1d_x2, dmx2_d, pbc2, n2msub, n2sub, n1sub, n3sub, 2)
        call apply_bc_on_direction(X_d, comm_1d_x3, dmx3_d, pbc3, n3msub, n3sub, n1sub, n2sub, 3)
        
    end subroutine cuda_neumann_BC

    subroutine apply_bc_on_direction(X_d, comm_1d, dmx_d, pbc, nmsub_a, nsub_a, nsub_b, nsub_c, direction)
        use mpi_topology, only : cart_comm_1d
        implicit none
    
        real(rp), device, dimension(0:,0:,0:) :: X_d
        type(cart_comm_1d) :: comm_1d
        real(rp), device, dimension(0:) :: dmx_d
        integer :: nmsub_a, nsub_a, nsub_b, nsub_c, direction
        logical :: pbc
        real(rp) :: BC_a, BC_b
        integer :: idx_b, idx_c
    
        if(pbc==.False.) then
            !$acc parallel loop collapse(2) &
            !$acc& private(BC_a, BC_b)
            do idx_c = 0, nsub_c
            do idx_b = 0, nsub_b
                if(comm_1d%myrank == 0) then
                    BC_a = (dmx_d(1   )+dmx_d(2    ))**real(2.0,rp)/((dmx_d(1   )+dmx_d(2    ))**real(2.0,rp)-dmx_d(1   )**real(2.0,rp))
                    BC_b =  dmx_d(1   )              **real(2.0,rp)/((dmx_d(1   )+dmx_d(2    ))**real(2.0,rp)-dmx_d(1   )**real(2.0,rp))
                    selectcase(direction)
                        case(1)
                            X_d(0,   idx_b,idx_c) = BC_a*X_d(1    ,idx_b,idx_c) - BC_b*X_d(2     ,idx_b,idx_c)
                        case(2)
                            X_d(idx_b,0  , idx_c) = BC_a*X_d(idx_b,1    ,idx_c) - BC_b*X_d(idx_b,     2,idx_c)
                        case(3)
                            X_d(idx_b,idx_c,0   ) = BC_a*X_d(idx_b,idx_c,1    ) - BC_b*X_d(idx_b,idx_c,2     )
                    endselect
                endif
                if(comm_1d%myrank == comm_1d%nprocs-1) then
                    BC_a = (dmx_d(nsub_a)+dmx_d(nmsub_a))**real(2.0,rp)/((dmx_d(nsub_a)+dmx_d(nmsub_a))**real(2.0,rp)-dmx_d(nsub_a)**real(2.0,rp))
                    BC_b =  dmx_d(nsub_a)                **real(2.0,rp)/((dmx_d(nsub_a)+dmx_d(nmsub_a))**real(2.0,rp)-dmx_d(nsub_a)**real(2.0,rp))
                    selectcase(direction)
                        case(1)
                            X_d(nsub_a,idx_b,idx_c) = BC_a*X_d(nmsub_a,idx_b,idx_c) - BC_b*X_d(nsub_a-2,idx_b,idx_c)
                        case(2)
                            X_d(idx_b,nsub_a,idx_c) = BC_a*X_d(idx_b,nmsub_a,idx_c) - BC_b*X_d(idx_b,nsub_a-2,idx_c)
                        case(3)
                            X_d(idx_b,idx_c,nsub_a) = BC_a*X_d(idx_b,idx_c,nmsub_a) - BC_b*X_d(idx_b,idx_c,nsub_a-2)
                    endselect
                endif
            enddo
            enddo
        endif
    end subroutine apply_bc_on_direction

    subroutine cuda_subdomain_clean
        implicit none
        deallocate(x1_d_c, dmx1_d_c, dx1_d_c)
        deallocate(x2_d_c, dmx2_d_c, dx2_d_c)
        deallocate(x3_d_c, dmx3_d_c, dx3_d_c)

        deallocate(x1_g_d_c, dmx1_g_d_c, dx1_g_d_c)
        deallocate(x2_g_d_c, dmx2_g_d_c, dx2_g_d_c)
        deallocate(x3_g_d_c, dmx3_g_d_c, dx3_g_d_c)

        deallocate(sbuf_0_temp, sbuf_1_temp)
        deallocate(rbuf_0_temp, rbuf_1_temp)
        deallocate(sbuf_0_temp_int, sbuf_1_temp_int)
        deallocate(rbuf_0_temp_int, rbuf_1_temp_int)

    end subroutine cuda_subdomain_clean

    subroutine cuda_subdomain_temp_allocation()
        implicit none

        allocate(  API_ptr(n1msub*n2msub*n3msub),  ACI_ptr(n1msub*n2msub*n3msub),  AMI_ptr(n1msub*n2msub*n3msub) )
        allocate(  APJ_ptr(n1msub*n2msub*n3msub),  ACJ_ptr(n1msub*n2msub*n3msub),  AMJ_ptr(n1msub*n2msub*n3msub) )
        allocate(  APK_ptr(n1msub*n2msub*n3msub),  ACK_ptr(n1msub*n2msub*n3msub),  AMK_ptr(n1msub*n2msub*n3msub) )

        allocate( RHS_buff1(n1msub*n2msub*n3msub), RHS_buff2(n1msub*n2msub*n3msub))

        allocate(   temp1((n1sub+1)*(n2sub+1)*(n3sub+1)))
        allocate(   temp2((n1sub+1)*(n2sub+1)*(n3sub+1)))
    end subroutine cuda_subdomain_temp_allocation

    subroutine cuda_destroy_ptr
        implicit none
        deallocate( API_ptr, ACI_ptr, AMI_ptr )
        deallocate( APJ_ptr, ACJ_ptr, AMJ_ptr )
        deallocate( APK_ptr, ACK_ptr, AMK_ptr )

        deallocate(RHS_buff1, RHS_buff2)
        deallocate(temp1, temp2)
    end subroutine cuda_destroy_ptr

    attributes(global) subroutine cuda_transpose_kernel(idata, odata, n1, n2, n3)
        implicit none
        
        real(rp), device, intent(in   ), dimension(:,:,:) :: idata
        real(rp), device, intent(inout), dimension(:,:,:) :: odata
        integer ,  value, intent(in   )                   :: n1, n2, n3
                
        integer :: i,j,k

        i = (blockidx%x-1) * blockdim%x + threadidx%x
        j = (blockidx%y-1) * blockdim%y + threadidx%y
        k = (blockidx%z-1) * blockdim%z + threadidx%z

        if (i <= n1 .and. j <= n2 .and. k <= n3) then
            odata(i,j,k) = idata(k,i,j)
        endif  

    end subroutine cuda_transpose_kernel

end module cuda_subdomain