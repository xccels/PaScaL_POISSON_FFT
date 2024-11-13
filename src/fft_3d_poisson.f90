module fft_poisson
    use openacc
    use nvtx
    use cufft
    use cudafor
    use PaScaL_TDMA_cuda
    use mpi

    implicit none

#ifdef SINGLE_PRECISION
    integer, parameter :: rp = kind(0.0), MPI_real_type = MPI_REAL, MPI_complex_type = MPI_COMPLEX
#else
    integer, parameter :: rp = kind(0.0d0), MPI_real_type = MPI_DOUBLE_PRECISION, MPI_complex_type = MPI_DOUBLE_COMPLEX
#endif

    real(rp), parameter :: PI = real(dacos(-1.0d0),rp)
    type, private :: comm_1d
        integer :: myrank                   !< Rank ID in current communicator
        integer :: nprocs                   !< Number of processes in current communicator
        integer :: west_rank                !< Previous rank ID in current communicator
        integer :: east_rank                !< Next rank ID in current communicator
        integer :: mpi_comm                 !< Current communicator
    end type comm_1d

    type, public :: fft_poisson_plan_cuda
        private

        type(comm_1d)   ::  comm_1d_x1, comm_1d_x2, comm_1d_x3

        integer         ::  n1, n2, n3
        integer         ::  n1sub, n2sub, n3sub
        integer         ::  n1m, n2m, n3m
        integer         ::  n1msub, n2msub, n3msub
        real(rp)        ::  L1, L2, L3

        logical         ::  pbc1, pbc2, pbc3

        type(dim3)      :: threads_tdma, threads_fft

    end type

    integer, dimension(2,2) :: plan_fft
    type(fft_poisson_plan_cuda) ::  p_poi
    type(ptdma_plan_many_cuda)  :: ptdma_plan_cuda_x1, ptdma_plan_cuda_x2, ptdma_plan_cuda_x3, ptdma_plan_cuda_fft

    ! For FFT/DCT TDMA coefficient
    real(rp),    allocatable, device, dimension(:)     :: dxk2, dyk2

    ! Pointer (For DCT)
    complex(rp), device, target, allocatable, dimension(:) :: Buff_c1, Buff_c2
       real(rp), device, target, allocatable, dimension(:) :: Buff_1 , Buff_2

    ! BCtype
    character(len=1) :: BCtype(3)

    interface cuda_Poisson_TDMA_z
        module procedure cuda_Poisson_TDMA_z_real, cuda_Poisson_TDMA_z_complex
    end interface

    interface cuda_Poisson_transpose_f
        module procedure cuda_Poisson_transpose_f_real, cuda_Poisson_transpose_f_complex
    end interface

    interface cuda_Poisson_transpose_b
        module procedure cuda_Poisson_transpose_b_real, cuda_Poisson_transpose_b_complex
    end interface

contains
    subroutine fft_poisson_plan_cuda_create(rank1, rank2, rank3, np1, np2, np3, wrank1, wrank2, wrank3, erank1, erank2, erank3, comm1,  comm2,  comm3,&
                                            n1, n2, n3, n1sub, n2sub, n3sub, L1, L2, L3, pbc1, pbc2, pbc3, threads_tdma, threads_fft)

        implicit none
        
        integer, intent(in)         ::   rank1,  rank2,  rank3
        integer, intent(in)         ::     np1,    np2,    np3
        integer, intent(in)         ::  wrank1, wrank2, wrank3
        integer, intent(in)         ::  erank1, erank2, erank3
        integer, intent(in)         ::   comm1,  comm2,  comm3

        integer, intent(in)         ::      n1,     n2,     n3
        integer, intent(in)         ::   n1sub,  n2sub,  n3sub
        integer                     ::     n1m,    n2m,    n3m
        integer                     ::  n1msub, n2msub, n3msub
        real(rp), intent(in)        ::      L1,     L2,     L3

        logical, intent(in)         ::    pbc1,   pbc2,   pbc3

        type(dim3), intent(in)      :: threads_tdma, threads_fft

        p_poi%comm_1d_x1%myrank     =   rank1; p_poi%comm_1d_x2%myrank     =     rank2; p_poi%comm_1d_x3%myrank     =     rank3;
        p_poi%comm_1d_x1%nprocs     =     np1; p_poi%comm_1d_x2%nprocs     =       np2; p_poi%comm_1d_x3%nprocs     =       np3;
        p_poi%comm_1d_x1%west_rank  =  wrank1; p_poi%comm_1d_x2%west_rank  =    wrank2; p_poi%comm_1d_x3%west_rank  =    wrank3;
        p_poi%comm_1d_x1%east_rank  =  erank1; p_poi%comm_1d_x2%east_rank  =    erank2; p_poi%comm_1d_x3%east_rank  =    erank3;
        p_poi%comm_1d_x1%mpi_comm   =   comm1; p_poi%comm_1d_x2%mpi_comm   =     comm2; p_poi%comm_1d_x3%mpi_comm   =     comm3;

        p_poi%n1    =    n1; p_poi%n2    =    n2; p_poi%n3    =    n3;
        p_poi%n1sub = n1sub; p_poi%n2sub = n2sub; p_poi%n3sub = n3sub;

        p_poi%L1   =   L1; p_poi%L2   =   L2; p_poi%L3   =   L3;
        p_poi%pbc1 = pbc1; p_poi%pbc2 = pbc2; p_poi%pbc3 = pbc3;

        p_poi%n1m    =    n1 - 1; p_poi%n2m    =    n2 - 1; p_poi%n3m    =    n3 - 1;
        p_poi%n1msub = n1sub - 1; p_poi%n2msub = n2sub - 1; p_poi%n3msub = n3sub - 1;

        p_poi%threads_tdma = threads_tdma; p_poi%threads_fft = threads_fft;

    end subroutine fft_poisson_plan_cuda_create

    subroutine cuda_Poisson_FFT_initial()

        call cuda_Poisson_FFT_BCtype()

        call cuda_Poisson_FFT_memory('allocate')
        call cuda_cufft_plan_memory('allocate')

        call cuda_Poisson_FFT_coefficient()

        call cuda_PaScaL_TDMA_plan_many_memory('allocate')

    end subroutine cuda_Poisson_FFT_initial

    subroutine cuda_Poisson_FFT_clean()
        implicit none

        call cuda_Poisson_FFT_memory('clean')
        call cuda_cufft_plan_memory('clean')
        call cuda_PaScaL_TDMA_plan_many_memory('clean')

    end subroutine cuda_Poisson_FFT_clean
    
    subroutine cuda_Poisson_FFT_BCtype()

        if(p_poi%pbc1==.True.) then
            BCtype(1)='P'
        else
            BCtype(1)='N'
        endif
        if(p_poi%pbc2==.True.) then
            BCtype(2)='P'
        else
            BCtype(2)='N'
        endif
        if(p_poi%pbc3==.True.) then
            BCtype(3)='P'
        else
            BCtype(3)='N'
        endif

    end subroutine cuda_Poisson_FFT_BCtype

    subroutine cuda_Poisson_FFT_memory(action)
        implicit none
        character(len=*), intent(in) :: action

        selectcase(action)
        case('allocate')
            if    (BCtype(1)=='N'.and.BCtype(2)=='N') then
                allocate(Buff_c1(p_poi%n1msub*p_poi%n3msub*p_poi%n2msub/2 + max0(p_poi%n1msub,p_poi%n2msub)*p_poi%n3msub)) ! Considering n/2+1
                allocate(Buff_1( p_poi%n1sub*p_poi%n3msub*p_poi%n2sub ), Buff_2( p_poi%n1sub*p_poi%n3msub*p_poi%n2sub ))
            elseif(BCtype(1)=='N'.and.BCtype(2)=='P') then
                allocate(Buff_c1(p_poi%n1msub*p_poi%n3msub*p_poi%n2msub/2 + max0(p_poi%n1msub,p_poi%n2msub)*p_poi%n3msub), Buff_c2(p_poi%n1msub*p_poi%n3msub*p_poi%n2msub/2 + max0(p_poi%n1msub,p_poi%n2msub)*p_poi%n3msub)) ! Considering n/2+1
                allocate(Buff_1( p_poi%n1sub*p_poi%n3msub*p_poi%n2sub ), Buff_2( p_poi%n1sub*p_poi%n3msub*p_poi%n2sub ))
            elseif(BCtype(1)=='P'.and.BCtype(2)=='N') then
                allocate(Buff_c1(p_poi%n2msub*p_poi%n3msub*p_poi%n1msub/2 + max0(p_poi%n1msub,p_poi%n2msub)*p_poi%n3msub), Buff_c2(p_poi%n2msub*p_poi%n3msub*p_poi%n1msub/2 + max0(p_poi%n1msub,p_poi%n2msub)*p_poi%n3msub)) ! Considering n/2+1
                allocate(Buff_1( p_poi%n1sub*p_poi%n3msub*p_poi%n2sub ), Buff_2( p_poi%n1sub*p_poi%n3msub*p_poi%n2sub )) 
            elseif(BCtype(1)=='P'.and.BCtype(2)=='P') then
                allocate(Buff_c1(p_poi%n2msub*p_poi%n3msub*(p_poi%n1msub/2+1)), Buff_c2(p_poi%n2msub*p_poi%n3msub*(p_poi%n1msub/2+1))) ! Considering n/2+1
                allocate(Buff_1(p_poi%n1msub*p_poi%n2msub*p_poi%n3msub ))
            endif

            allocate(dxk2(1:p_poi%n1m), dyk2(1:p_poi%n2m))
        case('clean')

            if    (BCtype(1)=='N'.and.BCtype(2)=='N') then
                deallocate(Buff_c1)
                deallocate(Buff_1,Buff_2)
            elseif(BCtype(1)=='N'.and.BCtype(2)=='P') then
                deallocate(Buff_c1, Buff_c2)
                deallocate(Buff_1,Buff_2)
            elseif(BCtype(1)=='P'.and.BCtype(2)=='N') then
                deallocate(Buff_c1, Buff_c2)
                deallocate(Buff_1,Buff_2)
            elseif(BCtype(1)=='P'.and.BCtype(2)=='P') then
                deallocate(Buff_c1, Buff_c2)
                deallocate(Buff_1)
            endif

            deallocate(dxk2,dyk2)

        endselect

    end subroutine cuda_Poisson_FFT_memory

    subroutine cuda_cufft_plan_memory(action)
        implicit none
        character(len=*), intent(in) :: action
        integer :: nsize, istride, ostride, idist, odist, nbatch, ierr

        selectcase(action)
        case('allocate')
            ! plan_dct_f_x
            nsize = p_poi%n1m
            istride = 1
            ostride = 1
            idist = p_poi%n1m
            odist = int(p_poi%n1m/2)+1
            nbatch = p_poi%n2msub*p_poi%n3msub
            #ifdef SINGLE_PRECISION
                ierr = cufftPlanMany(plan_fft(1,1), 1, nsize, null(), istride, idist, null(), ostride, odist, CUFFT_R2C, nbatch)
            #elif DOUBLE_PRECISION
                ierr = cufftPlanMany(plan_fft(1,1), 1, nsize, null(), istride, idist, null(), ostride, odist, CUFFT_D2Z, nbatch)
            #endif
            ! plan_dct_b_x
            nsize = p_poi%n1m
            istride = 1
            ostride = 1
            idist = int(p_poi%n1m/2)+1
            odist = p_poi%n1m
            nbatch = p_poi%n2msub*p_poi%n3msub
            #ifdef SINGLE_PRECISION
                ierr = cufftPlanMany(plan_fft(2,1), 1, nsize, null(), istride, idist, null(), ostride, odist, CUFFT_C2R, nbatch)	
            #elif DOUBLE_PRECISION
                ierr = cufftPlanMany(plan_fft(2,1), 1, nsize, null(), istride, idist, null(), ostride, odist, CUFFT_Z2D, nbatch)	
            #endif

            if(BCtype(1)=='P'.and.BCtype(2)=='P') then
                ! plan_fft_f_y	
                nsize = p_poi%n2m
                istride = 1
                ostride = 1
                idist = p_poi%n2m
                odist = p_poi%n2m
                nbatch = (int(p_poi%n1m/2)+1)*p_poi%n3msub
                #ifdef SINGLE_PRECISION
                    ierr = cufftPlanMany(plan_fft(1,2), 1, nsize, null(), istride, idist, null(), ostride, odist, CUFFT_C2C, nbatch)	
                #elif DOUBLE_PRECISION
                    ierr = cufftPlanMany(plan_fft(1,2), 1, nsize, null(), istride, idist, null(), ostride, odist, CUFFT_Z2Z, nbatch)	
                #endif
                ! plan_fft_b_y	
                nsize = p_poi%n2m
                istride = 1
                ostride = 1
                idist = p_poi%n2m
                odist = p_poi%n2m
                nbatch = (int(p_poi%n1m/2)+1)*p_poi%n3msub
                #ifdef SINGLE_PRECISION
                    ierr = cufftPlanMany(plan_fft(2,2), 1, nsize, null(), istride, idist, null(), ostride, odist, CUFFT_C2C, nbatch)	
                #elif DOUBLE_PRECISION
                    ierr = cufftPlanMany(plan_fft(2,2), 1, nsize, null(), istride, idist, null(), ostride, odist, CUFFT_Z2Z, nbatch)
                #endif
            else
                ! plan_dct_f_y
                nsize = p_poi%n2m
                istride = 1
                ostride = 1
                idist = p_poi%n2m
                odist = p_poi%n2m/2+1
                nbatch = p_poi%n1m*p_poi%n3msub
                #ifdef SINGLE_PRECISION
                    ierr = cufftPlanMany(plan_fft(1,2), 1, nsize, null(), istride, idist, null(), ostride, odist, CUFFT_R2C, nbatch)	
                #elif DOUBLE_PRECISION
                    ierr = cufftPlanMany(plan_fft(1,2), 1, nsize, null(), istride, idist, null(), ostride, odist, CUFFT_D2Z, nbatch)	
                #endif

                ! plan_dct_b_y
                nsize = p_poi%n2m
                istride = 1
                ostride = 1
                idist = p_poi%n2m/2+1
                odist = p_poi%n2m
                nbatch = p_poi%n1m*p_poi%n3msub
                #ifdef SINGLE_PRECISION
                    ierr = cufftPlanMany(plan_fft(2,2), 1, nsize, null(), istride, idist, null(), ostride, odist, CUFFT_C2R, nbatch)	
                #elif DOUBLE_PRECISION
                    ierr = cufftPlanMany(plan_fft(2,2), 1, nsize, null(), istride, idist, null(), ostride, odist, CUFFT_Z2D, nbatch)	
                #endif
            endif
        case('clean')   
            ierr = cufftDestroy(plan_fft(1,1))
            ierr = cufftDestroy(plan_fft(1,2))
            ierr = cufftDestroy(plan_fft(2,2))
            ierr = cufftDestroy(plan_fft(2,1))
        endselect

    end subroutine cuda_cufft_plan_memory

    subroutine cuda_Poisson_FFT_coefficient()
        implicit none
        real(rp) :: dx1, dx2

        integer :: i, j, k, im, jm, km

        dx1=p_poi%L1/real(p_poi%n1m,rp)
        dx2=p_poi%L2/real(p_poi%n2m,rp)

        if    (BCtype(1)=='N'.and.BCtype(2)=='N') then
            !$acc parallel loop collapse(1) private(im)
            do i = 1, p_poi%n1m
                im = i-1
                dxk2(i) = real(4.0,rp) * ( dsin(real(0.5,rp) * real(im,rp) * PI / real(p_poi%n1m,rp)) )**real(2.0,rp) /(dx1*dx1) 
            enddo
            !$acc parallel loop collapse(1) private(jm)
            do j = 1, p_poi%n2m
                jm = j-1
                dyk2(j) = real(4.0,rp) * ( dsin(real(0.5,rp) * real(jm,rp) * PI / real(p_poi%n2m,rp)) )**real(2.0,rp) /(dx2*dx2) 
            enddo
            !$acc end parallel
        elseif(BCtype(1)=='N'.and.BCtype(2)=='P') then
            !$acc parallel loop collapse(1) private(im)
            do i = 1, p_poi%n1m
                im = i-1
                dxk2(i) = real(4.0,rp) * ( dsin(real(0.5,rp) * real(im,rp) * PI / real(p_poi%n1m,rp)) )**real(2.0,rp) /(dx1*dx1) 
            enddo
            !$acc parallel loop collapse(1) private(jm)
            do j = 1, p_poi%n2m
                jm = j-1
                dyk2(j) = real(2.0,rp) * ( real(1.0,rp) - dcos(real(2.0,rp) * real(jm,rp) * PI / real(p_poi%n2m,rp))) /(dx2*dx2)
            enddo
            !$acc end parallel
        elseif(BCtype(1)=='P'.and.BCtype(2)=='N') then
            !$acc parallel loop collapse(1) private(jm)
            do j = 1, p_poi%n2m
                jm = j-1
                dyk2(j) = real(4.0,rp) * ( dsin(real(0.5,rp) * real(jm,rp) * PI / real(p_poi%n2m,rp)) )**real(2.0,rp) /(dx2*dx2) 
            enddo
            !$acc parallel loop collapse(1) private(im)
            do i = 1, p_poi%n1m
                im = i-1
                dxk2(i) = real(2.0,rp) * ( real(1.0,rp) - dcos(real(2.0,rp) * real(im,rp) * PI / real(p_poi%n1m,rp))) /(dx1*dx1)
            enddo
            !$acc end parallel
        elseif(BCtype(1)=='P'.and.BCtype(2)=='P') then
            !$acc parallel loop collapse(1) private(im)
            do i = 1, p_poi%n1m
                im = i-1
                dxk2(i) = real(2.0,rp) * ( real(1.0,rp) - dcos(real(2.0,rp) * real(im,rp) * PI / real(p_poi%n1m,rp))) /(dx1*dx1)
            enddo
            !$acc parallel loop collapse(1) private(jm)
            do j = 1, p_poi%n2m
                if(j <= p_poi%n2m/2+1) then
                    jm = j - 1
                else
                    jm = p_poi%n2m - j + 1
                endif
                dyk2(j) = real(2.0,rp) * ( real(1.0,rp) - dcos(real(2.0,rp) * real(jm,rp) * PI / real(p_poi%n2m,rp))) /(dx2*dx2) 
            enddo
            !$acc end parallel
        endif

    end subroutine cuda_Poisson_FFT_coefficient

    subroutine cuda_Poisson_FFT_1D(PRHS_d, P_d, dmx1_d, dmx2_d, dmx3_d, dx3_d)
        use cublas
        use MPI
        implicit none

        real(rp),             device, dimension(1:,1:,1:) :: PRHS_d
        real(rp),             device, dimension(0:,0:,0:) :: P_d 
        real(rp),             device, dimension(:)        :: dmx1_d, dmx2_d, dmx3_d
        real(rp),             device, dimension(:)        :: dx3_d

        real(rp),    pointer, device, dimension( :, :, :) :: FFT_x1, FFT_x2, FFT_y1, FFT_y2, FFT_x3, FFT_y3
        complex(rp), pointer, device, dimension( :, :, :) :: FFT_xc, FFT_yc

        integer :: ierr

        integer :: n1msub, n2msub, n3msub
        integer :: n1m, n2m 
        integer :: n1sub, n2sub

        n1msub = p_poi%n1msub; n2msub = p_poi%n2msub; n3msub = p_poi%n3msub;
        n1m = p_poi%n1m; n2m = p_poi%n2m; 
        n1sub = p_poi%n1sub; n2sub = p_poi%n2sub;

        ! Forward dct
        call nvtxStartRange("Poisson")
        call nvtxStartRange("Poisson-F_x")

        if    ((BCtype(1)=='N'.and.BCtype(2)=='N') .or. (BCtype(1)=='N'.and.BCtype(2)=='P')) then ! X-R2R

            FFT_x2(1:n1msub    ,1:n2msub,1:n3msub) => Buff_1
            call cuda_Poisson_DCT_f_pre (PRHS_d, FFT_x2, n1msub, n2msub, n3msub)

            FFT_xc(1:n1msub/2+1,1:n2msub,1:n3msub) => Buff_c1
            ierr = cufftExecD2Z(plan_fft(1,1), FFT_x2, FFT_xc)
            nullify(FFT_x2)

            FFT_x1(1:n1msub    ,1:n2msub,1:n3msub) => Buff_2
            call cuda_Poisson_DCT_f_post(FFT_xc, FFT_x1, n1msub, n2msub, n3msub)
            nullify(FFT_xc)

        elseif(BCtype(1)=='P'.and.BCtype(2)=='N') then ! Y-R2R

            FFT_y1(1:n2msub    ,1:n3msub,1:n1msub) => Buff_2
            call cuda_Poisson_transpose_b_real(PRHS_d, FFT_y1, n2msub, n3msub, n1msub, real(1.0,rp))
            
            FFT_y2(1:n2msub    ,1:n3msub,1:n1msub) => Buff_1
            call cuda_Poisson_DCT_f_pre (FFT_y1, FFT_y2, n2msub, n3msub, n1msub)
            nullify(FFT_y1)

            FFT_yc(1:n2msub/2+1,1:n3msub,1:n1msub) => Buff_c1
            ierr = cufftExecD2Z(plan_fft(1,2), FFT_y2, FFT_yc)
            nullify(FFT_y2)

            FFT_y1(1:n2msub    ,1:n3msub,1:n1msub) => Buff_2
            call cuda_Poisson_DCT_f_post(FFT_yc, FFT_y1, n2msub, n3msub, n1msub)
            nullify(FFT_yc)

        elseif(BCtype(1)=='P'.and.BCtype(2)=='P') then ! X-R2C

            FFT_x1(1:n1msub    ,1:n2msub,1:n3msub) => Buff_1
            call dcopy(n1msub*n2msub*n3msub, PRHS_d, 1, FFT_x1, 1)  

            FFT_xc(1:n1msub/2+1,1:n2msub,1:n3msub) => Buff_c1
            ierr = cufftExecD2Z(plan_fft(1,1), FFT_x1, FFT_xc)
            nullify(FFT_x1)
    
        endif

        call nvtxEndRange
        call nvtxStartRange("Poisson-F_y")

        if    (BCtype(1)=='N'.and.BCtype(2)=='N') then ! Y-R2R

            FFT_y1(1:n2msub    ,1:n3msub,1:n1msub) => Buff_1
            call cuda_Poisson_transpose_b(FFT_x1, FFT_y1, n2msub, n3msub, n1msub, real(1.0,rp))
            nullify(FFT_x1)

            FFT_y2(1:n2msub    ,1:n3msub,1:n1msub) => Buff_2
            call cuda_Poisson_DCT_f_pre (FFT_y1, FFT_y2, n2msub, n3msub, n1msub)
            nullify(FFT_y1)

            FFT_yc(1:n2msub/2+1,1:n3msub,1:n1msub) => Buff_c1
            ierr = cufftExecD2Z(plan_fft(1,2), FFT_y2, FFT_yc)
            nullify(FFT_y2)

            FFT_y1(1:n2msub    ,1:n3msub,1:n1msub) => Buff_1
            call cuda_Poisson_DCT_f_post(FFT_yc, FFT_y1, n2msub, n3msub, n1msub)
            nullify(FFT_yc)

            FFT_x1(1:n1msub    ,1:n2msub,1:n3msub) => Buff_2
            call cuda_Poisson_transpose_f(FFT_y1, FFT_x1, n1msub, n2msub, n3msub, real(1.0,rp))
            nullify(FFT_y1)

        elseif(BCtype(1)=='N'.and.BCtype(2)=='P') then ! Y-R2C

            FFT_y1(1:n2msub    ,1:n3msub,1:n1msub) => Buff_1
            call cuda_Poisson_transpose_b(FFT_x1, FFT_y1, n2msub, n3msub, n1msub, real(1.0,rp))
            nullify(FFT_x1)

            FFT_yc(1:n2msub/2+1,1:n3msub,1:n1msub) => Buff_c1
            ierr = cufftExecD2Z(plan_fft(1,2), FFT_y1, FFT_yc)
            nullify(FFT_y1)

            FFT_xc(1:n1msub,1:n2msub/2+1,1:n3msub) => Buff_c2
            call cuda_Poisson_transpose_f(FFT_yc, FFT_xc, n1msub, n2msub/2+1, n3msub, real(1.0,rp))
            nullify(FFT_yc)

        elseif(BCtype(1)=='P'.and.BCtype(2)=='N') then ! X-R2C

            FFT_x1(1:n1msub    ,1:n2msub,1:n3msub) => Buff_1
            call cuda_Poisson_transpose_f(FFT_y1, FFT_x1, n1msub, n2msub, n3msub, real(1.0,rp))
            nullify(FFT_y1)

            FFT_xc(1:n1msub/2+1,1:n2msub,1:n3msub) => Buff_c1
            ierr = cufftExecD2Z(plan_fft(1,1), FFT_x1, FFT_xc)
            nullify(FFT_x1)
            
        elseif(BCtype(1)=='P'.and.BCtype(2)=='P') then ! Y-C2C

            FFT_yc(1:n2msub,1:n3msub,1:n1msub/2+1) => Buff_c2
            call cuda_Poisson_transpose_b(FFT_xc, FFT_yc, n2msub, n3msub, n1msub/2+1, real(1.0,rp))
            nullify(FFT_xc)

            ierr = cufftExecZ2Z(plan_fft(1,2), FFT_yc, FFT_yc, CUFFT_FORWARD)

            FFT_xc(1:n1msub/2+1,1:n2msub,1:n3msub) => Buff_c1
            call cuda_Poisson_transpose_f(FFT_yc, FFT_xc, n1msub/2+1, n2msub, n3msub, real(1.0,rp))
            nullify(FFT_yc)

        endif

        call nvtxEndRange

        call nvtxStartRange("Poisson-TDMA-z")
        if(BCtype(1)=='N'.and.BCtype(2)=='N') then
            call cuda_Poisson_TDMA_z(FFT_x1, dx3_d, dmx3_d)
        else
            call cuda_Poisson_TDMA_z(FFT_xc, dx3_d, dmx3_d)
        endif
        call nvtxEndRange

        ! Backward DCT
        call nvtxStartRange("Poisson-B_y")

        if(BCtype(1)=='N'.and.BCtype(2)=='N') then

            FFT_y1(1:n2msub,1:n3msub,1:n1msub) => Buff_1
            call cuda_Poisson_transpose_b(FFT_x1, FFT_y1, n2msub, n3msub, n1msub, real(1.0,rp))
            nullify(FFT_x1)

            FFT_yc(1:(n2msub)/2+1,1:n3msub,1:n1msub) => Buff_c1
            FFT_y3(1:n2sub       ,1:n3msub,1:n1msub) => Buff_2
            call cuda_Poisson_DCT_b_pre(FFT_y1, FFT_y3, FFT_yc, n2msub, n3msub, n1msub)
            nullify(FFT_y3, FFT_y1)

            FFT_y1(1:n2msub,1:n3msub,1:n1msub) => Buff_1
            ierr = cufftExecZ2D(plan_fft(2,2), FFT_yc, FFT_y1)
            nullify(FFT_yc)

            FFT_y2(1:n2msub,1:n3msub,1:n1msub) => Buff_2
            call cuda_Poisson_DCT_b_post(FFT_y1, FFT_y2, n2msub, n3msub, n1msub)
            nullify(FFT_y1)

            FFT_x1(1:n1msub,1:n2msub,1:n3msub) => Buff_1
            call cuda_Poisson_transpose_f(FFT_y2, FFT_x1, n1msub, n2msub, n3msub, real(1,rp)/real(2*n2m,rp))
            nullify(FFT_y2)

        elseif(BCtype(1)=='N'.and.BCtype(2)=='P') then ! Y-C2R

            FFT_yc(1:n2msub/2+1,1:n3msub,1:n1msub) => Buff_c1
            call cuda_Poisson_transpose_b(FFT_xc, FFT_yc, n2msub/2+1, n3msub, n1msub, real(1.0,rp))
            nullify(FFT_xc)

            FFT_y2(1:n2msub,1:n3msub,1:n1msub) => Buff_2
            ierr = cufftExecZ2D(plan_fft(2,2), FFT_yc, FFT_y2)
            nullify(FFT_yc)

            FFT_x1(1:n1msub,1:n2msub,1:n3msub) => Buff_1
            call cuda_Poisson_transpose_f(FFT_y2, FFT_x1, n1msub, n2msub, n3msub, real(1,rp)/real(n2m,rp))
            nullify(FFT_y2)

        elseif(BCtype(1)=='P'.and.BCtype(2)=='N') then ! X-C2R

            FFT_x1(1:n1msub,1:n2msub,1:n3msub) => Buff_2
            ierr = cufftExecZ2D(plan_fft(2,1), FFT_xc, FFT_x1)
            nullify(FFT_xc)

            FFT_y1(1:n2msub,1:n3msub,1:n1msub) => Buff_1
            call cuda_Poisson_transpose_b(FFT_x1, FFT_y1, n2msub, n3msub, n1msub, real(1,rp)/real(n1m,rp))
            nullify(FFT_x1)

        elseif(BCtype(1)=='P'.and.BCtype(2)=='P') then ! Y-C2C

            FFT_yc(1:n2msub,1:n3msub,1:n1msub/2+1) => Buff_c2
            call cuda_Poisson_transpose_b(FFT_xc, FFT_yc, n2msub, n3msub, n1msub/2+1, real(1.0,rp))
            nullify(FFT_xc)

            ierr = cufftExecZ2Z(plan_fft(2,2), FFT_yc, FFT_yc, CUFFT_INVERSE)

            FFT_xc(1:n1msub/2+1,1:n2msub,1:n3msub) => Buff_c1
            call cuda_Poisson_transpose_f(FFT_yc, FFT_xc, n1msub/2+1, n2msub, n3msub, real(1,rp)/real(n2m,rp))
            nullify(FFT_yc)

        endif

        call nvtxEndRange

        call nvtxStartRange("Poisson-B_x")

        if((BCtype(1)=='N'.and.BCtype(2)=='N') .or. (BCtype(1)=='N'.and.BCtype(2)=='P')) then ! X-R2R
            FFT_xc(1:n1msub/2+1,1:n2msub,1:n3msub) => Buff_c1
            FFT_x3(1:n1sub     ,1:n2msub,1:n3msub) => Buff_2
            call cuda_Poisson_DCT_b_pre(FFT_x1, FFT_x3, FFT_xc, n1msub, n2msub, n3msub)
            nullify(FFT_x1, FFT_x3)

            FFT_x1(1:n1msub,1:n2msub,1:n3msub) => Buff_1
            ierr = cufftExecZ2D(plan_fft(2,1), FFT_xc, FFT_x1)
            nullify(FFT_xc)

            FFT_x2(1:n1msub,1:n2msub,1:n3msub) => Buff_2
            call cuda_Poisson_DCT_b_post(FFT_x1, FFT_x2, n1msub, n2msub, n3msub)
            nullify(FFT_x1)

            call dcopy(n1msub*n2msub*n3msub, FFT_x2, 1, PRHS_d, 1)  
            nullify(FFT_x2)

            call dscal(n1msub*n2msub*n3msub, real(1,rp)/real(2*n1m,rp), PRHS_d, 1)

        elseif(BCtype(1)=='P'.and.BCtype(2)=='N') then ! Y-R2R
            
            FFT_yc(1:n2msub/2+1,1:n3msub,1:n1msub) => Buff_c1
            FFT_y3(1:n2sub     ,1:n3msub,1:n1msub) => Buff_2
            call cuda_Poisson_DCT_b_pre(FFT_y1, FFT_y3, FFT_yc, n2msub, n3msub, n1msub)
            nullify(FFT_y1, FFT_y3)

            FFT_y1(1:n2msub,1:n3msub,1:n1msub) => Buff_1
            ierr = cufftExecZ2D(plan_fft(2,2), FFT_yc, FFT_y1)
            nullify(FFT_yc)

            FFT_y2(1:n2msub,1:n3msub,1:n1msub) => Buff_2
            call cuda_Poisson_DCT_b_post(FFT_y1, FFT_y2, n2msub, n3msub, n1msub)
            nullify(FFT_y1)

            FFT_x1(1:n1msub,1:n2msub,1:n3msub) => Buff_1
            call cuda_Poisson_transpose_f(FFT_y2, FFT_x1, n1msub, n2msub, n3msub, real(1.0,rp))
            nullify(FFT_y2)

            call dcopy(n1msub*n2msub*n3msub, FFT_x1, 1, PRHS_d, 1)
            nullify(FFT_x1)

            call dscal(n1msub*n2msub*n3msub, real(1,rp)/real(2*n2m,rp), PRHS_d, 1)

        elseif(BCtype(1)=='P'.and.BCtype(2)=='P') then ! X-C2R

            FFT_x1(1:n1msub+1,1:n2msub,1:n3msub) => Buff_1
            ierr = cufftExecZ2D(plan_fft(2,1), FFT_xc, FFT_x1)
            nullify(FFT_xc)

            call dcopy(n1msub*n2msub*n3msub, FFT_x1, 1, PRHS_d, 1)
            nullify(FFT_x1)

            call dscal(n1msub*n2msub*n3msub, real(1,rp)/real(n1m,rp), PRHS_d, 1)

        endif

        call nvtxEndRange

        call cuda_Poisson_average_elimination(P_d, PRHS_d)
        call cuda_neumann_BC(P_d, dmx1_d, dmx2_d, dmx3_d)
        call cuda_ghostcell_update(P_d)

        call nvtxEndRange

    end subroutine cuda_Poisson_FFT_1D
    
    subroutine cuda_Poisson_average_elimination(P_d, PRHS_d)
        use MPI

        implicit none

        real(rp), device, dimension(0:p_poi%n1sub ,0:p_poi%n2sub ,0:p_poi%n3sub )   :: P_d 
        real(rp), device, dimension(1:p_poi%n1msub,1:p_poi%n2msub,1:p_poi%n3msub)   :: PRHS_d
        real(rp) :: AVERsub, AVERmpi_I, AVERmpi_J, AVERmpi_K, AVERmpi
        integer  :: i,j,k, ierr

        AVERsub=real(0.0,rp)
        
        !$cuf kernel do(3) <<<*,*>>>
        do k=1,p_poi%n3msub
        do j=1,p_poi%n2msub
        do i=1,p_poi%n1msub
            AVERsub = AVERsub+PRHS_d(i,j,k)
        enddo
        enddo
        enddo

        AVERmpi_I=real(0.0,rp);AVERmpi_J=real(0.0,rp);AVERmpi_K=real(0.0,rp);AVERmpi=real(0.0,rp);
        call MPI_ALLREDUCE(AVERsub  , AVERmpi_I, 1, MPI_real_type, MPI_SUM, p_poi%comm_1d_x1%mpi_comm, ierr)
        call MPI_ALLREDUCE(AVERmpi_I, AVERmpi_K, 1, MPI_real_type, MPI_SUM, p_poi%comm_1d_x2%mpi_comm, ierr)
        call MPI_ALLREDUCE(AVERmpi_K, AVERmpi  , 1, MPI_real_type, MPI_SUM, p_poi%comm_1d_x3%mpi_comm, ierr)    
        AVERmpi=AVERmpi/real(p_poi%n1m,rp)/real(p_poi%n2m,rp)/real(p_poi%n3m,rp)

        !$cuf kernel do(3) <<<*,*>>>
        do k=1,p_poi%n3msub
        do j=1,p_poi%n2msub
        do i=1,p_poi%n1msub
            P_d(i,j,k)=PRHS_d(i,j,k)-AVERmpi
        enddo
        enddo
        enddo

    end subroutine cuda_Poisson_average_elimination

    subroutine cuda_ghostcell_update(Value_sub_d)
        implicit none
        real(rp), device, target, dimension(0:p_poi%n1sub, 0:p_poi%n2sub, 0:p_poi%n3sub), intent(inout)  :: Value_sub_d
        real(rp), pointer, device, dimension(:,:,:) :: Value_sub_ptr

        Value_sub_ptr(0:p_poi%n1sub,0:p_poi%n2sub,0:p_poi%n3sub) => Value_sub_d(0:,0:,0:)
        call cuda_ghostcell_update_real(Value_sub_ptr)
    end subroutine cuda_ghostcell_update

    subroutine cuda_ghostcell_update_real(Value_sub_d)

        implicit none
        real(rp), device, dimension(0:, 0:, 0:), intent(inout)  :: Value_sub_d
        real(rp), pointer, device, dimension(:,:)   :: sbuf_x0_d, sbuf_x1_d, sbuf_y0_d, sbuf_y1_d, sbuf_z0_d, sbuf_z1_d
        real(rp), pointer, device, dimension(:,:)   :: rbuf_x0_d, rbuf_x1_d, rbuf_y0_d, rbuf_y1_d, rbuf_z0_d, rbuf_z1_d   
        
        real(rp), device, target, allocatable, dimension(:)   :: sbuf_0_temp    , sbuf_1_temp    , rbuf_0_temp    , rbuf_1_temp
        integer , device, target, allocatable, dimension(:)   :: sbuf_0_temp_int, sbuf_1_temp_int, rbuf_0_temp_int, rbuf_1_temp_int

        allocate( sbuf_0_temp((max0(p_poi%n1sub,p_poi%n2sub,p_poi%n3sub)+1)*(max0(p_poi%n1sub,p_poi%n2sub,p_poi%n3sub)+1)), sbuf_1_temp((max0(p_poi%n1sub,p_poi%n2sub,p_poi%n3sub)+1)*(max0(p_poi%n1sub,p_poi%n2sub,p_poi%n3sub)+1)))
        allocate( rbuf_0_temp((max0(p_poi%n1sub,p_poi%n2sub,p_poi%n3sub)+1)*(max0(p_poi%n1sub,p_poi%n2sub,p_poi%n3sub)+1)), rbuf_1_temp((max0(p_poi%n1sub,p_poi%n2sub,p_poi%n3sub)+1)*(max0(p_poi%n1sub,p_poi%n2sub,p_poi%n3sub)+1)))
        sbuf_0_temp = real(0.0,rp); sbuf_1_temp = real(0.0,rp);
        rbuf_0_temp = real(0.0,rp); rbuf_1_temp = real(0.0,rp);

        sbuf_x0_d(0:p_poi%n2sub,0:p_poi%n3sub) => sbuf_0_temp; sbuf_x1_d(0:p_poi%n2sub,0:p_poi%n3sub) => sbuf_1_temp
        rbuf_x0_d(0:p_poi%n2sub,0:p_poi%n3sub) => rbuf_0_temp; rbuf_x1_d(0:p_poi%n2sub,0:p_poi%n3sub) => rbuf_1_temp
        call ghostcell_update_on_direction(p_poi%comm_1d_x1, 1, p_poi%pbc1, sbuf_x0_d, sbuf_x1_d, rbuf_x0_d, rbuf_x1_d, p_poi%n1sub, p_poi%n2sub, p_poi%n3sub, Value_sub_d)
        nullify(sbuf_x0_d, sbuf_x1_d, rbuf_x0_d, rbuf_x1_d)

        sbuf_y0_d(0:p_poi%n1sub,0:p_poi%n3sub) => sbuf_0_temp; sbuf_y1_d(0:p_poi%n1sub,0:p_poi%n3sub) => sbuf_1_temp
        rbuf_y0_d(0:p_poi%n1sub,0:p_poi%n3sub) => rbuf_0_temp; rbuf_y1_d(0:p_poi%n1sub,0:p_poi%n3sub) => rbuf_1_temp
        call ghostcell_update_on_direction(p_poi%comm_1d_x2, 2, p_poi%pbc2, sbuf_y0_d, sbuf_y1_d, rbuf_y0_d, rbuf_y1_d, p_poi%n2sub, p_poi%n1sub, p_poi%n3sub, Value_sub_d)
        nullify(sbuf_y0_d, sbuf_y1_d, rbuf_y0_d, rbuf_y1_d)

        sbuf_z0_d(0:p_poi%n1sub,0:p_poi%n2sub) => sbuf_0_temp; sbuf_z1_d(0:p_poi%n1sub,0:p_poi%n2sub) => sbuf_1_temp
        rbuf_z0_d(0:p_poi%n1sub,0:p_poi%n2sub) => rbuf_0_temp; rbuf_z1_d(0:p_poi%n1sub,0:p_poi%n2sub) => rbuf_1_temp
        call ghostcell_update_on_direction(p_poi%comm_1d_x3, 3, p_poi%pbc3, sbuf_z0_d, sbuf_z1_d, rbuf_z0_d, rbuf_z1_d, p_poi%n3sub, p_poi%n1sub, p_poi%n2sub, Value_sub_d)
        nullify(sbuf_z0_d, sbuf_z1_d, rbuf_z0_d, rbuf_z1_d)

        deallocate( sbuf_0_temp, sbuf_1_temp)
        deallocate( rbuf_0_temp, rbuf_1_temp)
    end subroutine cuda_ghostcell_update_real

    subroutine ghostcell_update_on_direction(comm_1d_x, direction, pbc, sbuf_0_d, sbuf_1_d, rbuf_0_d, rbuf_1_d, nsub_a, nsub_b, nsub_c, Value_sub_d)
        implicit none
        type(comm_1d), intent(in) :: comm_1d_x
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
                if(comm_1d_x%west_rank.ne.MPI_PROC_NULL) then
                    sbuf_0_d(idx_b,idx_c) = Value_sub_d(1       ,idx_b,idx_c)
                endif
                if(comm_1d_x%east_rank.ne.MPI_PROC_NULL) then
                    sbuf_1_d(idx_b,idx_c) = Value_sub_d(nsub_a-1,idx_b,idx_c)
                endif
            case(2)  ! Y direction
                if(comm_1d_x%west_rank.ne.MPI_PROC_NULL) then
                    sbuf_0_d(idx_b,idx_c) = Value_sub_d(idx_b,1       ,idx_c)
                endif
                if(comm_1d_x%east_rank.ne.MPI_PROC_NULL) then
                    sbuf_1_d(idx_b,idx_c) = Value_sub_d(idx_b,nsub_a-1,idx_c)
                endif
            case(3)  ! Z direction
                if(comm_1d_x%west_rank.ne.MPI_PROC_NULL) then
                    sbuf_0_d(idx_b,idx_c) = Value_sub_d(idx_b,idx_c,1       )
                endif
                if(comm_1d_x%east_rank.ne.MPI_PROC_NULL) then
                    sbuf_1_d(idx_b,idx_c) = Value_sub_d(idx_b,idx_c,nsub_a-1)
                endif
            end select
        enddo
        enddo
    
        ! MPI Area
        if( comm_1d_x%nprocs.eq.1 .and. pbc.eqv..true. ) then
            !$cuf kernel do(2) <<< *,* >>>
            do idx_c = 0, nsub_c
            do idx_b = 0, nsub_b
                rbuf_1_d(idx_b,idx_c) = sbuf_0_d(idx_b,idx_c)
                rbuf_0_d(idx_b,idx_c) = sbuf_1_d(idx_b,idx_c)
            enddo
            enddo
        else
            
            ierr = cudaStreamSynchronize()
            call MPI_Isend(sbuf_0_d, (nsub_b+1)*(nsub_c+1), MPI_real_type, comm_1d_x%west_rank, 111, comm_1d_x%mpi_comm, request(1), ierr)
            call MPI_Irecv(rbuf_1_d, (nsub_b+1)*(nsub_c+1), MPI_real_type, comm_1d_x%east_rank, 111, comm_1d_x%mpi_comm, request(2), ierr)
            call MPI_Irecv(rbuf_0_d, (nsub_b+1)*(nsub_c+1), MPI_real_type, comm_1d_x%west_rank, 222, comm_1d_x%mpi_comm, request(3), ierr)
            call MPI_Isend(sbuf_1_d, (nsub_b+1)*(nsub_c+1), MPI_real_type, comm_1d_x%east_rank, 222, comm_1d_x%mpi_comm, request(4), ierr)
            call MPI_Waitall(4, request, MPI_STATUSES_IGNORE, ierr)    
        endif
    
        !$cuf kernel do(2) <<< *,* >>>
        do idx_c = 0, nsub_c
        do idx_b = 0, nsub_b
            select case(direction)
            case(1)  ! X direction
                if(comm_1d_x%west_rank.ne.MPI_PROC_NULL) then
                    Value_sub_d(0     ,idx_b,idx_c) = rbuf_0_d(idx_b,idx_c)
                endif
                if(comm_1d_x%east_rank.ne.MPI_PROC_NULL) then
                    Value_sub_d(nsub_a,idx_b,idx_c) = rbuf_1_d(idx_b,idx_c)
                endif
            case(2)  ! Y direction
                if(comm_1d_x%west_rank.ne.MPI_PROC_NULL) then
                    Value_sub_d(idx_b,0     ,idx_c) = rbuf_0_d(idx_b,idx_c)
                endif
                if(comm_1d_x%east_rank.ne.MPI_PROC_NULL) then
                    Value_sub_d(idx_b,nsub_a,idx_c) = rbuf_1_d(idx_b,idx_c)
                endif
            case(3)  ! Z direction
                if(comm_1d_x%west_rank.ne.MPI_PROC_NULL) then
                    Value_sub_d(idx_b,idx_c,0     ) = rbuf_0_d(idx_b,idx_c)
                endif
                if(comm_1d_x%east_rank.ne.MPI_PROC_NULL) then
                    Value_sub_d(idx_b,idx_c,nsub_a) = rbuf_1_d(idx_b,idx_c)
                endif
            end select
        enddo
        enddo
    end subroutine ghostcell_update_on_direction

    subroutine cuda_neumann_BC(X_d, dmx1_d, dmx2_d, dmx3_d)
        implicit none 
        real(rp),             device, dimension(0:,0:,0:) :: X_d
        real(rp),             device, dimension(0:)       :: dmx1_d, dmx2_d, dmx3_d

        call apply_bc_on_direction(X_d, p_poi%comm_1d_x1, dmx1_d, p_poi%pbc1, p_poi%n1msub, p_poi%n1sub, p_poi%n2sub, p_poi%n3sub, 1)
        call apply_bc_on_direction(X_d, p_poi%comm_1d_x2, dmx2_d, p_poi%pbc2, p_poi%n2msub, p_poi%n2sub, p_poi%n1sub, p_poi%n3sub, 2)
        call apply_bc_on_direction(X_d, p_poi%comm_1d_x3, dmx3_d, p_poi%pbc3, p_poi%n3msub, p_poi%n3sub, p_poi%n1sub, p_poi%n2sub, 3)
        
    end subroutine cuda_neumann_BC

    subroutine apply_bc_on_direction(X_d, comm_1d_x, dmx_d, pbc, nmsub_a, nsub_a, nsub_b, nsub_c, direction)
        implicit none
    
        real(rp), device, dimension(0:,0:,0:) :: X_d
        type(comm_1d) :: comm_1d_x
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
                if(comm_1d_x%myrank == 0) then
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
                if(comm_1d_x%myrank == comm_1d_x%nprocs-1) then
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

    subroutine cuda_PaScaL_TDMA_plan_many_memory(action)
        use PaScaL_TDMA_cuda, only : PaScaL_TDMA_plan_many_create_cuda, PaScaL_TDMA_plan_many_destroy_cuda
        implicit none

        character(len=*), intent(in) :: action
        character(len=1) :: BCtype(3)

        integer ::  n1msub, n2msub, n3msub
        logical :: pbc1, pbc2, pbc3

        n1msub = p_poi%n1msub; n2msub = p_poi%n2msub; n3msub = p_poi%n3msub;
        pbc1   =   p_poi%pbc1; pbc2   =   p_poi%pbc2; pbc3   =   p_poi%pbc3;

        ! For Viscous term
        selectcase(action)
        case('allocate')
            call PaScaL_TDMA_plan_many_create_cuda(ptdma_plan_cuda_x1, n2msub, n3msub, n1msub, p_poi%comm_1d_x1%myrank, p_poi%comm_1d_x1%nprocs, p_poi%comm_1d_x1%mpi_comm, p_poi%threads_tdma)
            call PaScaL_TDMA_plan_many_create_cuda(ptdma_plan_cuda_x2, n3msub, n1msub, n2msub, p_poi%comm_1d_x2%myrank, p_poi%comm_1d_x2%nprocs, p_poi%comm_1d_x2%mpi_comm, p_poi%threads_tdma)
            call PaScaL_TDMA_plan_many_create_cuda(ptdma_plan_cuda_x3, n1msub, n2msub, n3msub, p_poi%comm_1d_x3%myrank, p_poi%comm_1d_x3%nprocs, p_poi%comm_1d_x3%mpi_comm, p_poi%threads_tdma)
        case('clean')
            call PaScaL_TDMA_plan_many_destroy_cuda(ptdma_plan_cuda_x1)
            call PaScaL_TDMA_plan_many_destroy_cuda(ptdma_plan_cuda_x2)
            call PaScaL_TDMA_plan_many_destroy_cuda(ptdma_plan_cuda_x3)
        endselect

        ! For Poisson equation-FFT
        selectcase(action)
        case('allocate')
            if    ((pbc1.eqv..false.).and.(pbc2.eqv..false.)) then !N-N
                call PaScaL_TDMA_plan_many_create_cuda(ptdma_plan_cuda_fft, n1msub, n2msub    , n3msub, p_poi%comm_1d_x3%myrank, p_poi%comm_1d_x3%nprocs, p_poi%comm_1d_x3%mpi_comm, p_poi%threads_tdma)
            elseif((pbc1.eqv..false.).and.(pbc2.eqv..true.) ) then !N-P
                call PaScaL_TDMA_plan_many_create_cuda(ptdma_plan_cuda_fft, n1msub, n2msub/2+1, n3msub, p_poi%comm_1d_x3%myrank, p_poi%comm_1d_x3%nprocs, p_poi%comm_1d_x3%mpi_comm, p_poi%threads_fft )
            elseif((pbc1.eqv..true. ).and.(pbc2.eqv..false.)) then !P-N
                call PaScaL_TDMA_plan_many_create_cuda(ptdma_plan_cuda_fft, n2msub, n1msub/2+1, n3msub, p_poi%comm_1d_x3%myrank, p_poi%comm_1d_x3%nprocs, p_poi%comm_1d_x3%mpi_comm, p_poi%threads_fft )
            elseif((pbc1.eqv..true. ).and.(pbc2.eqv..true. )) then !P-P
                call PaScaL_TDMA_plan_many_create_cuda(ptdma_plan_cuda_fft, n2msub, n1msub/2+1, n3msub, p_poi%comm_1d_x3%myrank, p_poi%comm_1d_x3%nprocs, p_poi%comm_1d_x3%mpi_comm, p_poi%threads_fft )
            endif
        case('clean')
            call PaScaL_TDMA_plan_many_destroy_cuda(ptdma_plan_cuda_fft)
        end select


    end subroutine cuda_PaScaL_TDMA_plan_many_memory

    subroutine cuda_ptdma_core(plan, A_d, B_d, C_d, D_d, pbc)
        use PaScaL_TDMA_Cuda, only : PaScaL_TDMA_many_solve_cycle_cuda, PaScaL_TDMA_many_solve_cuda, ptdma_plan_many_cuda
        implicit none

        type(ptdma_plan_many_cuda)        , intent(inout) :: plan 
        real(rp), device, dimension(:,:,:), intent(inout) :: A_d, B_d, C_d, D_d
        logical                           , intent(in   ) :: pbc

        selectcase(pbc)
        case(.True.)
            call PaScaL_TDMA_many_solve_cycle_cuda(plan, A_d, B_d, C_d, D_d)
        case(.False.)
            call       PaScaL_TDMA_many_solve_cuda(plan, A_d, B_d, C_d, D_d)
        end select

    end subroutine cuda_ptdma_core

    subroutine cuda_Poisson_DCT_f_pre(in,out,n1,n2,n3)
        implicit none

        integer                                              :: n1, n2, n3
        integer                                              :: i, j, k

        real(rp) , device, dimension(1:n1,1:n2,1:n3) :: in
        real(rp) , device, dimension(1:n1,1:n2,1:n3) :: out

        !$acc parallel loop collapse(3)
        do k=1,n3
        do j=1,n2
        do i=1,n1/2
            out(i     ,j,k) = in(2*i-1              ,j,k) !v(n)
            out(i+n1/2,j,k) = in(2*(n1-(i+(n1/2))+1),j,k) !w(n)
        end do
        end do
        end do
        !$acc end parallel

    end subroutine cuda_Poisson_DCT_f_pre

    subroutine cuda_Poisson_DCT_f_post(in,out,n1,n2,n3)
        implicit none

        integer                                              :: n1, n2, n3
        integer                                              :: i, j, k

        complex(rp), device, dimension(1:(n1/2)+1,1:n2,1:n3) :: in      
        real(rp)   , device, dimension(1:n1      ,1:n2,1:n3) :: out
    
        real(rp)                                             :: arg

        !$acc parallel loop collapse(3) private(arg)
        do k=1,n3
        do j=1,n2
        do i=1,n1/2
            arg = -PI*real(i-1,rp)/(real(2.0,rp)*real(n1,rp))
            out(i,j,k) = real(2.0,rp)*(dcos(arg)*real(in(i,j,k),rp) - dsin(arg)*dimag(in(i,j,k)))
        end do
        end do
        end do
        !$acc end parallel
        !$acc parallel loop collapse(3) private(arg)
        do k=1,n3
        do j=1,n2
        do i=n1/2+1, n1
            arg = -PI*real(i-1,rp)/(real(2.0,rp)*real(n1,rp))
            out(i,j,k) = real(2.0,rp)*(dcos(arg)*real(in(n1-i+2,j,k),rp) + dsin(arg)*dimag(in(n1-i+2,j,k)))
        end do
        end do
        end do
        !$acc end parallel

    end subroutine cuda_Poisson_DCT_f_post

    subroutine cuda_Poisson_DCT_b_pre(in1,in2,out,n1,n2,n3)
        implicit none

        integer                                              :: n1, n2, n3
        integer                                              :: i, j, k

        real(rp)   , device, dimension(1:n1      ,1:n2,1:n3) :: in1
        real(rp)   , device, dimension(1:n1+1    ,1:n2,1:n3) :: in2
        complex(rp), device, dimension(1:(n1/2)+1,1:n2,1:n3) :: out
    
        real(rp)                                             :: arg

        !$acc parallel loop collapse(2)
        do k=1,n3
        do j=1,n2
            in2(n1+1,j,k) = real(0.0,rp)
        end do
        end do
        !$acc end parallel
        !$acc parallel loop collapse(3)
        do k=1,n3
        do j=1,n2
        do i=1,n1
            in2(i,j,k) = in1(i,j,k)
        end do
        end do
        end do
        !$acc end parallel
        !$acc parallel loop collapse(3) private(arg)
        do k=1,n3
        do j=1,n2
        do i=1,n1/2+1
            arg        = -PI*real(   i-1,rp)/(real(2.0,rp)*real(n1,rp))
            out(i,j,k) = DCMPLX( ( dcos(arg)*in2(i,j,k)-dsin(arg)*in2(n1+2-i,j,k)), &
                                 (-dsin(arg)*in2(i,j,k)-dcos(arg)*in2(n1+2-i,j,k))  )
        end do
        end do
        end do
        !$acc end parallel

    end subroutine cuda_Poisson_DCT_b_pre

    subroutine cuda_Poisson_DCT_b_post(in,out,n1,n2,n3)
        implicit none

        integer                                              :: n1, n2, n3
        integer                                              :: i, j, k

        real(rp) , device, dimension(1:n1,1:n2,1:n3) :: in
        real(rp) , device, dimension(1:n1,1:n2,1:n3) :: out

        !$acc parallel loop collapse(3)
        do k=1,n3
        do j=1,n2
        do i=1,n1/2
            out(2*i-1              ,j,k) =  in(i     ,j,k)
            out(2*(n1-(i+(n1/2))+1),j,k) =  in(i+n1/2,j,k)
        end do
        end do
        end do
        !$acc end parallel

    end subroutine cuda_Poisson_DCT_b_post

    subroutine cuda_Poisson_transpose_f_real(in,out,n1,n2,n3,coefficient)
        implicit none

        integer                                     :: n1, n2, n3
        integer                                     :: i, j, k

        real(rp), device, dimension(1:n2,1:n3,1:n1) :: in      
        real(rp), device, dimension(1:n1,1:n2,1:n3) :: out
        real(rp)                                    :: coefficient

        !$cuf kernel do(3) <<<*,*>>>
        do k=1,n3
        do j=1,n2
        do i=1,n1
            out(i,j,k)=in(j,k,i)*coefficient
        enddo
        enddo
        enddo

    end subroutine cuda_Poisson_transpose_f_real

    subroutine cuda_Poisson_transpose_b_real(in,out,n1,n2,n3,coefficient)
        implicit none

        integer                                     :: n1, n2, n3
        integer                                     :: i, j, k

        real(rp), device, dimension(1:n3,1:n1,1:n2) :: in      
        real(rp), device, dimension(1:n1,1:n2,1:n3) :: out
        real(rp)                                    :: coefficient


        !$cuf kernel do(3) <<<*,*>>>
        do k=1,n3
        do j=1,n2
        do i=1,n1
            out(i,j,k)=in(k,i,j)*coefficient
        enddo
        enddo
        enddo  

    end subroutine cuda_Poisson_transpose_b_real

    subroutine cuda_Poisson_transpose_f_complex(in,out,n1,n2,n3,coefficient)
        implicit none

        integer                                     :: n1, n2, n3
        integer                                     :: i, j, k

        complex(rp), device, dimension(1:n2,1:n3,1:n1) :: in      
        complex(rp), device, dimension(1:n1,1:n2,1:n3) :: out
        real(rp)                                       :: coefficient

        !$cuf kernel do(3) <<<*,*>>>
        do k=1,n3
        do j=1,n2
        do i=1,n1
            out(i,j,k)=in(j,k,i)*coefficient
        enddo
        enddo
        enddo

    end subroutine cuda_Poisson_transpose_f_complex

    subroutine cuda_Poisson_transpose_b_complex(in,out,n1,n2,n3,coefficient)
        implicit none

        integer                                     :: n1, n2, n3
        integer                                     :: i, j, k

        complex(rp), device, dimension(1:n3,1:n1,1:n2) :: in      
        complex(rp), device, dimension(1:n1,1:n2,1:n3) :: out
        real(rp)                                    :: coefficient


        !$cuf kernel do(3) <<<*,*>>>
        do k=1,n3
        do j=1,n2
        do i=1,n1
            out(i,j,k)=in(k,i,j)*coefficient
        enddo
        enddo
        enddo  

    end subroutine cuda_Poisson_transpose_b_complex 

    subroutine cuda_Poisson_TDMA_z_real(d_d, dx3_d, dmx3_d)
        real(rp),             device, dimension(:,:,:) :: d_d
        real(rp),    pointer, device, dimension(:,:,:) :: a_d, b_d, c_d
        real(rp),             device, dimension(0:)    :: dx3_d, dmx3_d   

        real(rp), device, target, allocatable, dimension(:) :: API_ptr, ACI_ptr, AMI_ptr
        real(rp) :: am, ac, ap
        integer :: i,j,k,kp

        allocate(  API_ptr(p_poi%n1msub*p_poi%n2msub*p_poi%n3msub),  ACI_ptr(p_poi%n1msub*p_poi%n2msub*p_poi%n3msub),  AMI_ptr(p_poi%n1msub*p_poi%n2msub*p_poi%n3msub) )

        ! Note that memory size are different!(Just for efficiency)
        a_d(1:p_poi%n1msub,1:p_poi%n2msub,1:p_poi%n3msub) => AMI_ptr
        b_d(1:p_poi%n1msub,1:p_poi%n2msub,1:p_poi%n3msub) => ACI_ptr
        c_d(1:p_poi%n1msub,1:p_poi%n2msub,1:p_poi%n3msub) => API_ptr

        !$acc parallel loop collapse(3) private(kp, am, ac, ap) copyin(p_poi%comm_1d_x1, p_poi%comm_1d_x2,p_poi%comm_1d_x3)
        do k = 1, p_poi%n3msub
        do j = 1, p_poi%n2m
        do i = 1, p_poi%n1m
            kp = k+1   

            am = real(1.0,rp)/dx3_d(k)/dmx3_d(k ); if(p_poi%comm_1d_x3%myrank==0                  .and.k==1      )  am = real(0.0,rp)
            ap = real(1.0,rp)/dx3_d(k)/dmx3_d(kp); if(p_poi%comm_1d_x3%myrank==p_poi%comm_1d_x3%nprocs-1.and.k==p_poi%n3msub )  ap = real(0.0,rp)

            ac =  - am - ap

            a_d(i,j,k) = am
            b_d(i,j,k) = ac - dxk2(i) - dyk2(j)
            c_d(i,j,k) = ap

            if(p_poi%comm_1d_x1%myrank==0.and.p_poi%comm_1d_x2%myrank==0.and.p_poi%comm_1d_x3%myrank==0.and.i==1.and.j==1.and.k==1) then
                a_d(1,1,1) = real(0.0,rp)
                b_d(1,1,1) = real(1.0,rp)
                c_d(1,1,1) = real(0.0,rp)
                d_d(1,1,1) = real(0.0,rp)
            endif    
        end do
        end do
        end do
        !$acc end parallel

        call cuda_ptdma_core(ptdma_plan_cuda_fft, a_d, b_d, c_d, d_d, p_poi%pbc3)

        deallocate( API_ptr, ACI_ptr, AMI_ptr )

        nullify(a_d, b_d, c_d)     
        !$acc exit data delete(p_poi)

    end subroutine cuda_Poisson_TDMA_z_real
    
    subroutine cuda_Poisson_TDMA_z_complex(d_d, dx3_d, dmx3_d)

        complex(rp),          device, dimension(:,:,:) :: d_d
        real(rp),    pointer, device, dimension(:,:,:) :: a_r_d, b_r_d, c_r_d, d_r_d
        real(rp),    pointer, device, dimension(:,:,:) :: a_c_d, b_c_d, c_c_d, d_c_d
        real(rp),             device, dimension(0:)    :: dmx3_d
        real(rp),             device, dimension(0:)    :: dx3_d

        type(comm_1d)                                  ::  comm_1d_x1, comm_1d_x2, comm_1d_x3

        integer                                        ::  n1msub, n2msub, n3msub
        logical                                        ::  pbc1, pbc2, pbc3

        real(rp), device, target, allocatable, dimension(:) :: RHS_buff1, RHS_buff2
        real(rp), device, target, allocatable, dimension(:) :: API_ptr, ACI_ptr, AMI_ptr, APJ_ptr, ACJ_ptr, AMJ_ptr

        real(rp) :: am, ac, ap, temp
        integer  :: iend, jend, ierr
        integer  :: i,j,k, kp 

        comm_1d_x1 = p_poi%comm_1d_x1; comm_1d_x2 = p_poi%comm_1d_x2; comm_1d_x3 = p_poi%comm_1d_x3;

        n1msub = p_poi%n1msub; n2msub = p_poi%n2msub; n3msub = p_poi%n3msub
        pbc1 = p_poi%pbc1; pbc2 = p_poi%pbc2; pbc3 = p_poi%pbc3

        allocate(  API_ptr(n1msub*n2msub*n3msub),  ACI_ptr(n1msub*n2msub*n3msub),  AMI_ptr(n1msub*n2msub*n3msub) )
        allocate(  APJ_ptr(n1msub*n2msub*n3msub),  ACJ_ptr(n1msub*n2msub*n3msub),  AMJ_ptr(n1msub*n2msub*n3msub) )

        allocate( RHS_buff1(n1msub*n2msub*n3msub), RHS_buff2(n1msub*n2msub*n3msub))

        ! Note that memory size are different!(Just for efficiency) 
        if    (BCtype(1)=='P') then ! P-N, P-P
            a_r_d(1:n2msub,1:n1msub/2+1,1:n3msub) => AMI_ptr   ; a_c_d(1:n2msub,1:n1msub/2+1,1:n3msub) => AMJ_ptr
            b_r_d(1:n2msub,1:n1msub/2+1,1:n3msub) => ACI_ptr   ; b_c_d(1:n2msub,1:n1msub/2+1,1:n3msub) => ACJ_ptr
            c_r_d(1:n2msub,1:n1msub/2+1,1:n3msub) => API_ptr   ; c_c_d(1:n2msub,1:n1msub/2+1,1:n3msub) => APJ_ptr
            d_r_d(1:n2msub,1:n1msub/2+1,1:n3msub) => RHS_buff1 ; d_c_d(1:n2msub,1:n1msub/2+1,1:n3msub) => RHS_buff2
        elseif(BCtype(2)=='P') then ! N-P
            a_r_d(1:n1msub,1:n2msub/2+1,1:n3msub) => AMI_ptr   ; a_c_d(1:n1msub,1:n2msub/2+1,1:n3msub) => AMJ_ptr
            b_r_d(1:n1msub,1:n2msub/2+1,1:n3msub) => ACI_ptr   ; b_c_d(1:n1msub,1:n2msub/2+1,1:n3msub) => ACJ_ptr
            c_r_d(1:n1msub,1:n2msub/2+1,1:n3msub) => API_ptr   ; c_c_d(1:n1msub,1:n2msub/2+1,1:n3msub) => APJ_ptr
            d_r_d(1:n1msub,1:n2msub/2+1,1:n3msub) => RHS_buff1 ; d_c_d(1:n1msub,1:n2msub/2+1,1:n3msub) => RHS_buff2
        endif

        if    (BCtype(1)=='P') then !Transpose for PASCAL_TDMA. First array row should be even number for alltoall communications.
            !$acc parallel loop collapse(3) private(kp, am, ac, ap) copyin(comm_1d_x1, comm_1d_x2,comm_1d_x3)
            do k = 1, n3msub
            do i = 1, n1msub/2+1
            do j = 1, n2msub
                kp = k+1   

                am = real(1.0,rp)/dx3_d(k)/dmx3_d(k ); if(comm_1d_x3%myrank==0                  .and.k==1      )  am = real(0.0,rp)
                ap = real(1.0,rp)/dx3_d(k)/dmx3_d(kp); if(comm_1d_x3%myrank==comm_1d_x3%nprocs-1.and.k==n3msub )  ap = real(0.0,rp)
                ac =  - am - ap

                d_r_d(j,i,k) =  real(d_d(i,j,k),rp)
            #ifdef SINGLE_PRECISION
                d_c_d(j,i,k) =  aimag(d_d(i,j,k))
            #elif  DOUBLE_PRECISION
                d_c_d(j,i,k) =  dimag(d_d(i,j,k))
            #endif

                a_r_d(j,i,k) = am
                b_r_d(j,i,k) = ac - dxk2(i) - dyk2(j)
                c_r_d(j,i,k) = ap
                a_c_d(j,i,k) = am
                b_c_d(j,i,k) = ac - dxk2(i) - dyk2(j)
                c_c_d(j,i,k) = ap

            if(comm_1d_x3%myrank==0.and.i==1.and.j==1.and.k==1) then
                a_r_d(1,1,1) = real(0.0,rp)
                b_r_d(1,1,1) = real(1.0,rp)
                c_r_d(1,1,1) = real(0.0,rp)
                d_r_d(1,1,1) = real(0.0,rp)
                a_c_d(1,1,1) = real(0.0,rp)
                b_c_d(1,1,1) = real(1.0,rp)
                c_c_d(1,1,1) = real(0.0,rp)
                d_c_d(1,1,1) = real(0.0,rp)
            endif

            end do
            end do
            end do
            !$acc end parallel

            call cuda_ptdma_core(ptdma_plan_cuda_fft, a_r_d, b_r_d, c_r_d, d_r_d, pbc3)
            call cuda_ptdma_core(ptdma_plan_cuda_fft, a_c_d, b_c_d, c_c_d, d_c_d, pbc3)

            nullify(a_r_d, b_r_d, c_r_d, a_c_d, b_c_d, c_c_d)     

            !$cuf kernel do(3) <<<*,*>>>
            do k = 1, n3msub
            do j = 1, n2msub
            do i = 1, n1msub/2+1
            #ifdef SINGLE_PRECISION
                d_d(i,j,k) =  CMPLX(d_r_d(j,i,k), d_c_d(j,i,k))
            #elif  DOUBLE_PRECISION
                d_d(i,j,k) = DCMPLX(d_r_d(j,i,k), d_c_d(j,i,k))
            #endif
            enddo
            enddo
            enddo

            !if(myrank.eq.0) call cuda_InstantOutput_mass_3D(d_c_d, myrank)

        elseif(BCtype(2)=='P') then
            !$acc parallel loop collapse(3) private(kp, am, ac, ap) copyin(comm_1d_x1, comm_1d_x2,comm_1d_x3)
            do k = 1, n3msub
            do j = 1, n2msub/2+1
            do i = 1, n1msub
                kp = k+1   

                am = real(1.0,rp)/dx3_d(k)/dmx3_d(k ); if(comm_1d_x3%myrank==0                  .and.k==1      )  am = real(0.0,rp)
                ap = real(1.0,rp)/dx3_d(k)/dmx3_d(kp); if(comm_1d_x3%myrank==comm_1d_x3%nprocs-1.and.k==n3msub )  ap = real(0.0,rp)

                ac =  - am - ap

                d_r_d(i,j,k) =  real(d_d(i,j,k),rp)
            #ifdef SINGLE_PRECISION
                d_c_d(i,j,k) =  aimag(d_d(i,j,k))
            #elif  DOUBLE_PRECISION
                d_c_d(i,j,k) =  dimag(d_d(i,j,k))
            #endif

                a_r_d(i,j,k) = am
                b_r_d(i,j,k) = ac - dxk2(i) - dyk2(j)
                c_r_d(i,j,k) = ap
                a_c_d(i,j,k) = am
                b_c_d(i,j,k) = ac - dxk2(i) - dyk2(j)
                c_c_d(i,j,k) = ap

            if(comm_1d_x3%myrank==0.and.i==1.and.j==1.and.k==1) then
                a_r_d(1,1,1) = real(0.0,rp)
                b_r_d(1,1,1) = real(1.0,rp)
                c_r_d(1,1,1) = real(0.0,rp)
                d_r_d(1,1,1) = real(0.0,rp)
                a_c_d(1,1,1) = real(0.0,rp)
                b_c_d(1,1,1) = real(1.0,rp)
                c_c_d(1,1,1) = real(0.0,rp)
                d_c_d(1,1,1) = real(0.0,rp)
            endif

            end do
            end do
            end do
            !$acc end parallel

            call cuda_ptdma_core(ptdma_plan_cuda_fft, a_r_d, b_r_d, c_r_d, d_r_d, pbc3)
            call cuda_ptdma_core(ptdma_plan_cuda_fft, a_c_d, b_c_d, c_c_d, d_c_d, pbc3)

            nullify(a_r_d, b_r_d, c_r_d, a_c_d, b_c_d, c_c_d)     

            !$cuf kernel do(3) <<<*,*>>>
            do k = 1, n3msub
            do j = 1, n2msub/2+1
            do i = 1, n1msub
            #ifdef SINGLE_PRECISION
                d_d(i,j,k) =  CMPLX(d_r_d(i,j,k), d_c_d(i,j,k))
            #elif  DOUBLE_PRECISION
                d_d(i,j,k) = DCMPLX(d_r_d(i,j,k), d_c_d(i,j,k))
            #endif
            enddo
            enddo
            enddo
        endif

        deallocate( API_ptr, ACI_ptr, AMI_ptr )
        deallocate( APJ_ptr, ACJ_ptr, AMJ_ptr )

        deallocate(RHS_buff1, RHS_buff2)

        nullify(d_r_d, d_c_d)     

    end subroutine cuda_Poisson_TDMA_z_complex

end module