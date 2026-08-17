module fft_poisson
    use openacc
    ! use nvtx
    use cufft
    use cudafor
    use PaScaL_TDMA_cuda
    use mpi
#ifdef POISSON_USE_CUDECOMP
    use fft_cudecomp_bridge, only: fft_cudecomp_initialize, &
        fft_cudecomp_x_to_y, fft_cudecomp_y_to_x, &
        fft_cudecomp_finalize
#endif

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

    type, private :: fft_alltoall_metadata
        integer :: nprocs = -1
        integer :: send_split_extent = -1
        integer :: send_plane_extent = -1
        integer :: recv_split_extent = -1
        integer :: recv_plane_extent = -1
        integer :: send_total = 0
        integer :: recv_total = 0
        integer, allocatable :: sendcounts(:)
        integer, allocatable :: recvcounts(:)
        integer, allocatable :: senddispls(:)
        integer, allocatable :: recvdispls(:)
    end type fft_alltoall_metadata

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

    ! v3 tuning: cache cuFFT plans instead of recreating them inside each Poisson solve.
    ! Cache keys are the 1D transform length, strides, input/output distances,
    ! transform type, and batch count for each plan_fft(i,j) slot.
    integer, dimension(2,2) :: plan_fft_cache_n       = -1
    integer, dimension(2,2) :: plan_fft_cache_istride = -1
    integer, dimension(2,2) :: plan_fft_cache_idist   = -1
    integer, dimension(2,2) :: plan_fft_cache_ostride = -1
    integer, dimension(2,2) :: plan_fft_cache_odist   = -1
    integer, dimension(2,2) :: plan_fft_cache_type    = -1
    integer, dimension(2,2) :: plan_fft_cache_nbatch  = -1
    type(fft_poisson_plan_cuda) ::  p_poi
    type(ptdma_plan_many_cuda)  :: ptdma_plan_cuda_x1, ptdma_plan_cuda_x2, ptdma_plan_cuda_x3, ptdma_plan_cuda_fft

    ! For FFT/DCT TDMA coefficient
    real(rp),    allocatable, device, dimension(:)     :: dxk2, dyk2

    ! Pointer (For DCT)
    complex(rp), device, target, allocatable, dimension(:) :: Buff_c1, Buff_c2
       real(rp), device, target, allocatable, dimension(:) :: Buff_1 , Buff_2

    ! BCtype
    character(len=1) :: BCtype(3)

    logical, parameter :: cufft_plan_cache_enabled = .true.
    logical, save :: tdma_2rhs_env_initialized = .false.
    logical, save :: tdma_2rhs_env_enabled = .true.
    logical, save :: tdma_2rhs_banner_printed = .false.
    logical, save :: tdma_workspace_banner_printed = .false.
    logical, save :: tdma_static_env_initialized = .false.
    logical, save :: tdma_static_env_enabled = .true.
    logical, save :: tdma_static_operator_prepared = .false.
    logical, save :: tdma_static_banner_printed = .false.

    interface cuda_Poisson_TDMA_z
        module procedure cuda_Poisson_TDMA_z_real, cuda_Poisson_TDMA_z_complex
    end interface

    interface cuda_Poisson_transpose_f
        module procedure cuda_Poisson_transpose_f_real, cuda_Poisson_transpose_f_complex
    end interface

    interface cuda_Poisson_transpose_b
        module procedure cuda_Poisson_transpose_b_real, cuda_Poisson_transpose_b_complex
    end interface

    ! -------------------------------------------------------------------------
    ! Contiguous CUDA-aware MPI redistribution workspaces.
    !
    ! The original FFT path passed non-contiguous device subarrays directly to
    ! MPI_Alltoallw through derived datatypes.  On the target MPI/UCX stack that
    ! path spends far more time packing device datatypes than doing the FFT.
    ! These buffers are filled/consumed by GPU kernels and communicated with
    ! ordinary contiguous MPI_Alltoallv calls.
    ! -------------------------------------------------------------------------
    real(rp), allocatable, device, target, save, dimension(:) :: fft_mpi_send_r_d
    real(rp), allocatable, device, target, save, dimension(:) :: fft_mpi_recv_r_d
    complex(rp), allocatable, device, target, save, dimension(:) :: fft_mpi_send_c_d
    complex(rp), allocatable, device, target, save, dimension(:) :: fft_mpi_recv_c_d

    type(fft_alltoall_metadata), save :: fft_meta_c2i_real
    type(fft_alltoall_metadata), save :: fft_meta_i2c_real
    type(fft_alltoall_metadata), save :: fft_meta_c2i_complex
    type(fft_alltoall_metadata), save :: fft_meta_i2c_complex
    type(fft_alltoall_metadata), save :: fft_meta_c2j_complex
    type(fft_alltoall_metadata), save :: fft_meta_j2c_complex

    ! Repeated CFD solves use an invariant local grid shape.  Keep the large
    ! TDMA-Z and ghost-exchange buffers alive until cuda_Poisson_FFT_clean()
    ! instead of paying cudaMalloc/cudaFree inside every Poisson call.
    real(rp), allocatable, device, target, save, dimension(:) :: tdma_api_work_d
    real(rp), allocatable, device, target, save, dimension(:) :: tdma_aci_work_d
    real(rp), allocatable, device, target, save, dimension(:) :: tdma_ami_work_d
    real(rp), allocatable, device, target, save, dimension(:) :: tdma_apj_work_d
    real(rp), allocatable, device, target, save, dimension(:) :: tdma_acj_work_d
    real(rp), allocatable, device, target, save, dimension(:) :: tdma_amj_work_d
    real(rp), allocatable, device, target, save, dimension(:) :: tdma_rhs_real_work_d
    real(rp), allocatable, device, target, save, dimension(:) :: tdma_rhs_imag_work_d

    real(rp), allocatable, device, target, save, dimension(:) :: ghost_send_0_d
    real(rp), allocatable, device, target, save, dimension(:) :: ghost_send_1_d
    real(rp), allocatable, device, target, save, dimension(:) :: ghost_recv_0_d
    real(rp), allocatable, device, target, save, dimension(:) :: ghost_recv_1_d

    logical, parameter :: fft_contiguous_mpi_enabled = .true.

    ! Lightweight timing profiler for the current periodic/periodic FFT path.
    ! The debug executable defines POISSON_DETAILED_PROFILE.  The performance
    ! executable leaves it undefined, so the per-phase synchronization calls
    ! become no-ops while the same numerical path is retained.
#ifdef POISSON_DETAILED_PROFILE
    logical, save :: poisson_profile_enabled = .true.
#else
    logical, parameter :: poisson_profile_enabled = .false.
#endif
    integer, parameter :: poisson_profile_count = 21
    integer, parameter :: tdma_setup_profile_count = 5
    integer, parameter :: fft_route_profile_count = 2
    integer, parameter :: fft_route_phase_count = 3
#ifdef POISSON_DETAILED_PROFILE
    double precision, save :: fft_route_profile_times(fft_route_phase_count, fft_route_profile_count) = 0.0d0
    double precision, save :: fft_route_profile_payload_bytes(fft_route_profile_count) = 0.0d0
    integer, save :: fft_route_profile_calls(fft_route_profile_count) = 0
#endif

    ! Performance-build coarse profiler.  Unlike the 21-boundary debug
    ! profiler, this mode synchronizes only at seven algorithmic boundaries
    ! and accumulates all timed solves before one MPI reduction.  It is an
    ! opt-in diagnostic run; unprofiled runs remain the source of total
    ! seconds/solve.
    integer, parameter :: poisson_coarse_profile_count = 7
#ifdef POISSON_COARSE_PROFILE
    logical, save :: poisson_coarse_profile_initialized = .false.
    logical, save :: poisson_coarse_profile_enabled = .false.
    logical, save :: poisson_coarse_profile_active = .false.
    double precision, save :: poisson_coarse_profile_t0 = 0.0d0
    double precision, save :: poisson_coarse_profile_sum( &
        poisson_coarse_profile_count) = 0.0d0
    integer, save :: poisson_coarse_profile_calls = 0
#endif
contains
    logical function poisson_tdma_2rhs_enabled()
        implicit none

        character(len=32) :: value
        integer :: status, length, ios, numeric_value

        if (.not. tdma_2rhs_env_initialized) then
            tdma_2rhs_env_enabled = .true.
            value = ''
            call get_environment_variable( &
                'POISSON_TDMA_2RHS', value, length, status)
            if (status == 0 .and. length > 0) then
                read(value(1:length), *, iostat=ios) numeric_value
                if (ios == 0) then
                    tdma_2rhs_env_enabled = numeric_value /= 0
                else
                    select case(trim(adjustl(value(1:length))))
                    case('false', 'FALSE', 'off', 'OFF', 'no', 'NO')
                        tdma_2rhs_env_enabled = .false.
                    case default
                        tdma_2rhs_env_enabled = .true.
                    end select
                endif
            endif
            tdma_2rhs_env_initialized = .true.
        endif

        poisson_tdma_2rhs_enabled = tdma_2rhs_env_enabled
    end function poisson_tdma_2rhs_enabled


    logical function poisson_tdma_static_enabled()
        implicit none

        character(len=32) :: value
        integer :: status, length, ios, numeric_value

        if (.not. tdma_static_env_initialized) then
            tdma_static_env_enabled = .true.
            value = ''
            call get_environment_variable( &
                'POISSON_TDMA_STATIC_OPERATOR', value, length, status)
            if (status == 0 .and. length > 0) then
                read(value(1:length), *, iostat=ios) numeric_value
                if (ios == 0) then
                    tdma_static_env_enabled = numeric_value /= 0
                else
                    select case(trim(adjustl(value(1:length))))
                    case('false', 'FALSE', 'off', 'OFF', 'no', 'NO')
                        tdma_static_env_enabled = .false.
                    case default
                        tdma_static_env_enabled = .true.
                    end select
                endif
            endif
            tdma_static_env_initialized = .true.
        endif

        poisson_tdma_static_enabled = tdma_static_env_enabled
    end function poisson_tdma_static_enabled

    subroutine poisson_profile_set_enabled(enabled, reset_counters)
        implicit none

        logical, intent(in) :: enabled
        logical, intent(in), optional :: reset_counters
        logical :: do_reset

        do_reset = .false.
        if (present(reset_counters)) do_reset = reset_counters

#ifdef POISSON_DETAILED_PROFILE
        poisson_profile_enabled = enabled
#endif
        call PaScaL_TDMA_profile_set_enabled(enabled, do_reset)
    end subroutine poisson_profile_set_enabled


    subroutine poisson_coarse_profile_initialize()
        implicit none

#ifdef POISSON_COARSE_PROFILE
        character(len=32) :: value
        integer :: status, length, ios, numeric_value

        if (poisson_coarse_profile_initialized) return

        value = ''
        numeric_value = 0
        call get_environment_variable( &
            'POISSON_COARSE_PROFILE', value, length, status)
        if (status == 0 .and. length > 0) then
            read(value(1:length), *, iostat=ios) numeric_value
            if (ios /= 0) then
                select case(trim(adjustl(value(1:length))))
                case('true', 'TRUE', 'on', 'ON', 'yes', 'YES')
                    numeric_value = 1
                case default
                    numeric_value = 0
                end select
            endif
        endif
        poisson_coarse_profile_enabled = numeric_value /= 0
        poisson_coarse_profile_initialized = .true.
#endif
    end subroutine poisson_coarse_profile_initialize


    subroutine poisson_coarse_profile_reset()
        implicit none

#ifdef POISSON_COARSE_PROFILE
        call poisson_coarse_profile_initialize()
        poisson_coarse_profile_sum = 0.0d0
        poisson_coarse_profile_calls = 0
        poisson_coarse_profile_active = .false.
#endif
    end subroutine poisson_coarse_profile_reset


    subroutine poisson_coarse_profile_begin()
        implicit none

#ifdef POISSON_COARSE_PROFILE
        integer :: ierr_cuda

        call poisson_coarse_profile_initialize()
        if (.not. poisson_coarse_profile_enabled) return

        ierr_cuda = cudaDeviceSynchronize()
        poisson_coarse_profile_t0 = MPI_Wtime()
        poisson_coarse_profile_active = .true.
#endif
    end subroutine poisson_coarse_profile_begin


    subroutine poisson_coarse_profile_mark(phase_id)
        implicit none

        integer, intent(in) :: phase_id

#ifdef POISSON_COARSE_PROFILE
        integer :: ierr_cuda
        double precision :: now

        if (.not. poisson_coarse_profile_active) return
        if (phase_id < 1 .or. &
            phase_id > poisson_coarse_profile_count) return

        ierr_cuda = cudaDeviceSynchronize()
        now = MPI_Wtime()
        poisson_coarse_profile_sum(phase_id) = &
            poisson_coarse_profile_sum(phase_id) + &
            now - poisson_coarse_profile_t0
        poisson_coarse_profile_t0 = now

        if (phase_id == poisson_coarse_profile_count) then
            poisson_coarse_profile_calls = &
                poisson_coarse_profile_calls + 1
            poisson_coarse_profile_active = .false.
        endif
#endif
    end subroutine poisson_coarse_profile_mark


    subroutine poisson_coarse_profile_report()
        implicit none

#ifdef POISSON_COARSE_PROFILE
        double precision :: local_average(poisson_coarse_profile_count)
        double precision :: phase_min(poisson_coarse_profile_count)
        double precision :: phase_sum(poisson_coarse_profile_count)
        double precision :: phase_max(poisson_coarse_profile_count)
        double precision :: phase_avg(poisson_coarse_profile_count)
        character(len=42) :: label
        integer :: calls_min, calls_max
        integer :: ierr_mpi, myrank, nprocs, i

        call poisson_coarse_profile_initialize()
        if (.not. poisson_coarse_profile_enabled) return

        call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr_mpi)
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr_mpi)

        local_average = 0.0d0
        if (poisson_coarse_profile_calls > 0) then
            local_average = poisson_coarse_profile_sum / &
                dble(poisson_coarse_profile_calls)
        endif

        call MPI_Reduce(local_average, phase_min, &
                        poisson_coarse_profile_count, &
                        MPI_DOUBLE_PRECISION, MPI_MIN, 0, &
                        MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(local_average, phase_sum, &
                        poisson_coarse_profile_count, &
                        MPI_DOUBLE_PRECISION, MPI_SUM, 0, &
                        MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(local_average, phase_max, &
                        poisson_coarse_profile_count, &
                        MPI_DOUBLE_PRECISION, MPI_MAX, 0, &
                        MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(poisson_coarse_profile_calls, calls_min, 1, &
                        MPI_INTEGER, MPI_MIN, 0, MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(poisson_coarse_profile_calls, calls_max, 1, &
                        MPI_INTEGER, MPI_MAX, 0, MPI_COMM_WORLD, ierr_mpi)

        if (myrank /= 0) return

        phase_avg = phase_sum / dble(nprocs)

        write(*,'(/,A)') &
            '================ Coarse performance phases ============='
        write(*,'(A)') &
            'Seven major GPU synchronization boundaries are enabled.'
        write(*,'(A)') &
            'Use unprofiled performance runs for total seconds/solve.'
        write(*,'(A,I0)') &
            '[COARSE] Timed solves accumulated: ', calls_min
        if (calls_min /= calls_max) then
            write(*,'(A,2(1X,I0))') &
                '[COARSE-WARN] Rank call-count min/max:', &
                calls_min, calls_max
        endif
        write(*,'(A)') &
            'Columns: rank-min / rank-avg / rank-max [seconds/solve]'

        do i = 1, poisson_coarse_profile_count
            select case(i)
            case(1)
                label = '01 Cube-to-X layout and optional dcopy'
            case(2)
                label = '02 Forward X cuFFT'
            case(3)
                label = '03 X-to-Y transpose and forward Y FFT'
            case(4)
                label = '04 TDMA-Z solve'
            case(5)
                label = '05 Inverse Y FFT and Y-to-X transpose'
            case(6)
                label = '06 Inverse X FFT and X-to-Cube layout'
            case(7)
                label = '07 Pressure post-processing'
            end select

            write(*,'(A,1X,A42,3(1X,ES13.6))') &
                '[COARSE]', label, &
                phase_min(i), phase_avg(i), phase_max(i)
        enddo
        write(*,'(A,/)') &
            '========================================================='
#endif
    end subroutine poisson_coarse_profile_report


    subroutine poisson_profile_start(t0)
        implicit none

        double precision, intent(out) :: t0
        integer :: ierr_cuda

#ifdef POISSON_DETAILED_PROFILE
        if (.not. poisson_profile_enabled) then
            t0 = 0.0d0
            return
        endif
        ierr_cuda = cudaDeviceSynchronize()
        t0 = MPI_Wtime()
#else
        t0 = 0.0d0
#endif
    end subroutine poisson_profile_start


    subroutine poisson_profile_stop(t0, elapsed)
        implicit none

        double precision, intent(in)  :: t0
        double precision, intent(out) :: elapsed
        integer :: ierr_cuda

#ifdef POISSON_DETAILED_PROFILE
        if (.not. poisson_profile_enabled) then
            elapsed = 0.0d0
            return
        endif
        ierr_cuda = cudaDeviceSynchronize()
        elapsed = MPI_Wtime() - t0
#else
        elapsed = 0.0d0
#endif
    end subroutine poisson_profile_stop


    subroutine poisson_profile_report(times)
        implicit none

        double precision, intent(in) :: times(poisson_profile_count)
        double precision :: tmin(poisson_profile_count)
        double precision :: tmax(poisson_profile_count)
        double precision :: tsum(poisson_profile_count)
        double precision :: tavg(poisson_profile_count)
        character(len=44) :: label
        integer :: ierr_mpi, myrank, nprocs, i

#ifndef POISSON_DETAILED_PROFILE
        return
#else
        if (.not. poisson_profile_enabled) return

        call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr_mpi)
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr_mpi)

        call MPI_Reduce(times, tmin, poisson_profile_count, &
                        MPI_DOUBLE_PRECISION, MPI_MIN, 0, &
                        MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(times, tsum, poisson_profile_count, &
                        MPI_DOUBLE_PRECISION, MPI_SUM, 0, &
                        MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(times, tmax, poisson_profile_count, &
                        MPI_DOUBLE_PRECISION, MPI_MAX, 0, &
                        MPI_COMM_WORLD, ierr_mpi)

        if (myrank /= 0) return

        tavg = tsum / dble(nprocs)

        write(*,'(/,A)') &
            '================ Poisson phase profile ================'
        write(*,'(A)') &
            'GPU work is synchronized at every measurement boundary.'
        write(*,'(A)') &
            'Columns: rank-min / rank-avg / rank-max [seconds]'

        do i = 1, poisson_profile_count
            select case(i)
            case(1)
                label = '01 PP FWD-X workspace/cache'
            case(2)
                label = '02 PP FWD-X pack/alltoallv/unpack real'
            case(3)
                label = '03 PP FWD-X device dcopy'
            case(4)
                label = '04 PP FWD-X cuFFT D2Z'
            case(5)
                label = '05 PP FWD-Y workspace/cache'
            case(6)
                label = '06 PP FWD I2J cuDecomp XToY'
            case(7)
                label = '07 PP FWD legacy C2J exchange'
            case(8)
                label = '08 PP FWD legacy J-to-Y transpose'
            case(9)
                label = '09 PP FWD-Y cuFFT Z2Z forward'
            case(10)
                label = '10 PP FWD-Y transpose forward'
            case(11)
                label = '11 PP BWD-Y transpose backward'
            case(12)
                label = '12 PP BWD-Y cuFFT Z2Z inverse'
            case(13)
                label = '13 PP BWD J2I cuDecomp YToX'
            case(14)
                label = '14 PP BWD legacy J2C exchange'
            case(15)
                label = '15 PP BWD legacy C2I exchange'
            case(16)
                label = '16 PP BWD-X cuFFT Z2D'
            case(17)
                label = '17 PP BWD-X I2C contiguous exchange'
            case(18)
                label = '18 PP BWD-X device dscal'
            case(19)
                label = '19 Post average elimination'
            case(20)
                label = '20 Post boundary conditions'
            case(21)
                label = '21 Post ghost-cell update'
            end select

            if (tmax(i) > 0.0d0) then
                write(*,'(A,1X,A44,3(1X,ES13.6))') &
                    '[PROFILE]', label, tmin(i), tavg(i), tmax(i)
            endif
        enddo

        write(*,'(A,/)') &
            '========================================================'
#endif
    end subroutine poisson_profile_report


    subroutine fft_route_profile_reset()
        implicit none

#ifdef POISSON_DETAILED_PROFILE
        if (.not. poisson_profile_enabled) return
        fft_route_profile_times = 0.0d0
        fft_route_profile_payload_bytes = 0.0d0
        fft_route_profile_calls = 0
#endif
    end subroutine fft_route_profile_reset


    subroutine fft_route_profile_record(route_id, phase_times, payload_bytes)
        implicit none

        integer, intent(in) :: route_id
        double precision, intent(in) :: phase_times(fft_route_phase_count)
        double precision, intent(in) :: payload_bytes

#ifdef POISSON_DETAILED_PROFILE
        if (.not. poisson_profile_enabled) return
        if (route_id < 1 .or. route_id > fft_route_profile_count) return

        fft_route_profile_times(:, route_id) = &
            fft_route_profile_times(:, route_id) + phase_times
        fft_route_profile_payload_bytes(route_id) = &
            fft_route_profile_payload_bytes(route_id) + payload_bytes
        fft_route_profile_calls(route_id) = fft_route_profile_calls(route_id) + 1
#endif
    end subroutine fft_route_profile_record


    subroutine fft_route_profile_report()
        implicit none

#ifdef POISSON_DETAILED_PROFILE
        double precision :: tmin(fft_route_phase_count, fft_route_profile_count)
        double precision :: tmax(fft_route_phase_count, fft_route_profile_count)
        double precision :: tsum(fft_route_phase_count, fft_route_profile_count)
        double precision :: tavg(fft_route_phase_count, fft_route_profile_count)
        double precision :: payload_min(fft_route_profile_count)
        double precision :: payload_max(fft_route_profile_count)
        double precision :: payload_sum(fft_route_profile_count)
        double precision :: payload_avg(fft_route_profile_count)
        double precision :: bandwidth(fft_route_profile_count)
        double precision :: bandwidth_min(fft_route_profile_count)
        double precision :: bandwidth_max(fft_route_profile_count)
        double precision :: bandwidth_sum(fft_route_profile_count)
        double precision :: bandwidth_avg(fft_route_profile_count)
        double precision :: route_total(fft_route_profile_count)
        double precision :: route_total_min(fft_route_profile_count)
        double precision :: route_total_max(fft_route_profile_count)
        double precision :: route_total_sum(fft_route_profile_count)
        double precision :: route_total_avg(fft_route_profile_count)
        integer :: calls_min(fft_route_profile_count)
        integer :: calls_max(fft_route_profile_count)
        integer :: ierr_mpi, myrank, nprocs, route_id, phase_id
        character(len=3) :: route_name
        character(len=28) :: phase_name

        if (.not. poisson_profile_enabled) return

        call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr_mpi)
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr_mpi)

        bandwidth = 0.0d0
        do route_id = 1, fft_route_profile_count
            route_total(route_id) = &
                sum(fft_route_profile_times(:, route_id))
            if (route_total(route_id) > 0.0d0) then
                bandwidth(route_id) = &
                    fft_route_profile_payload_bytes(route_id) / &
                    route_total(route_id) / 1.0d9
            endif
        enddo

        call MPI_Reduce(fft_route_profile_times, tmin, &
                        fft_route_phase_count*fft_route_profile_count, &
                        MPI_DOUBLE_PRECISION, MPI_MIN, 0, &
                        MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(fft_route_profile_times, tsum, &
                        fft_route_phase_count*fft_route_profile_count, &
                        MPI_DOUBLE_PRECISION, MPI_SUM, 0, &
                        MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(fft_route_profile_times, tmax, &
                        fft_route_phase_count*fft_route_profile_count, &
                        MPI_DOUBLE_PRECISION, MPI_MAX, 0, &
                        MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(fft_route_profile_payload_bytes, payload_min, &
                        fft_route_profile_count, MPI_DOUBLE_PRECISION, &
                        MPI_MIN, 0, MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(fft_route_profile_payload_bytes, payload_sum, &
                        fft_route_profile_count, MPI_DOUBLE_PRECISION, &
                        MPI_SUM, 0, MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(fft_route_profile_payload_bytes, payload_max, &
                        fft_route_profile_count, MPI_DOUBLE_PRECISION, &
                        MPI_MAX, 0, MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(bandwidth, bandwidth_min, fft_route_profile_count, &
                        MPI_DOUBLE_PRECISION, MPI_MIN, 0, &
                        MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(bandwidth, bandwidth_sum, fft_route_profile_count, &
                        MPI_DOUBLE_PRECISION, MPI_SUM, 0, &
                        MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(bandwidth, bandwidth_max, fft_route_profile_count, &
                        MPI_DOUBLE_PRECISION, MPI_MAX, 0, &
                        MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(route_total, route_total_min, &
                        fft_route_profile_count, MPI_DOUBLE_PRECISION, &
                        MPI_MIN, 0, MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(route_total, route_total_sum, &
                        fft_route_profile_count, MPI_DOUBLE_PRECISION, &
                        MPI_SUM, 0, MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(route_total, route_total_max, &
                        fft_route_profile_count, MPI_DOUBLE_PRECISION, &
                        MPI_MAX, 0, MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(fft_route_profile_calls, calls_min, &
                        fft_route_profile_count, MPI_INTEGER, MPI_MIN, 0, &
                        MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(fft_route_profile_calls, calls_max, &
                        fft_route_profile_count, MPI_INTEGER, MPI_MAX, 0, &
                        MPI_COMM_WORLD, ierr_mpi)

        if (myrank /= 0) return

        tavg = tsum / dble(nprocs)
        payload_avg = payload_sum / dble(nprocs)
        bandwidth_avg = bandwidth_sum / dble(nprocs)
        route_total_avg = route_total_sum / dble(nprocs)

        write(*,'(/,A)') &
            '================ FFT route detail ====================='
        write(*,'(A)') &
            'Debug build only; performance build has no added route sync.'
        write(*,'(A)') &
            'Pack time includes the required CUDA dependency wait before MPI.'
        write(*,'(A)') &
            'Columns: rank-min / rank-avg / rank-max'

        do route_id = 1, fft_route_profile_count
            if (route_id == 1) then
                route_name = 'C2J'
            else
                route_name = 'J2C'
            endif

            if (calls_max(route_id) == 0) cycle

            do phase_id = 1, fft_route_phase_count
                select case(phase_id)
                case(1)
                    phase_name = 'pack + dependency wait [s]'
                case(2)
                    phase_name = 'MPI_Alltoallv [s]'
                case(3)
                    phase_name = 'unpack + completion wait [s]'
                end select
                write(*,'(A,1X,A3,1X,A28,3(1X,ES13.6))') &
                    '[ROUTE]', route_name, phase_name, &
                    tmin(phase_id,route_id), tavg(phase_id,route_id), &
                    tmax(phase_id,route_id)
            enddo

            write(*,'(A,1X,A3,1X,A28,3(1X,ES13.6))') &
                '[ROUTE]', route_name, 'route total [s]', &
                route_total_min(route_id), route_total_avg(route_id), &
                route_total_max(route_id)
            write(*,'(A,1X,A3,1X,A28,3(1X,ES13.6))') &
                '[ROUTE]', route_name, 'payload sent / rank [MiB]', &
                payload_min(route_id)/(1024.0d0*1024.0d0), &
                payload_avg(route_id)/(1024.0d0*1024.0d0), &
                payload_max(route_id)/(1024.0d0*1024.0d0)
            write(*,'(A,1X,A3,1X,A28,3(1X,ES13.6))') &
                '[ROUTE]', route_name, 'effective payload BW [GB/s]', &
                bandwidth_min(route_id), bandwidth_avg(route_id), &
                bandwidth_max(route_id)
            write(*,'(A,1X,A3,1X,A,I0,A,I0)') &
                '[ROUTE]', route_name, 'calls/rank min=', &
                calls_min(route_id), ' max=', calls_max(route_id)
        enddo

        if (maxval(calls_max) == 0) then
            write(*,'(A)') &
                '[ROUTE] No inter-rank C2J/J2C exchange in this solve.'
        endif

        write(*,'(A,/)') &
            '========================================================'
#endif
    end subroutine fft_route_profile_report


    subroutine tdma_setup_profile_report(times)
        implicit none

        double precision, intent(in) :: times(tdma_setup_profile_count)
        double precision :: tmin(tdma_setup_profile_count)
        double precision :: tmax(tdma_setup_profile_count)
        double precision :: tsum(tdma_setup_profile_count)
        double precision :: tavg(tdma_setup_profile_count)
        character(len=44) :: label
        integer :: ierr_mpi, myrank, nprocs, i

#ifndef POISSON_DETAILED_PROFILE
        return
#else
        if (.not. poisson_profile_enabled) return

        call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr_mpi)
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr_mpi)

        call MPI_Reduce(times, tmin, tdma_setup_profile_count, &
                        MPI_DOUBLE_PRECISION, MPI_MIN, 0, &
                        MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(times, tsum, tdma_setup_profile_count, &
                        MPI_DOUBLE_PRECISION, MPI_SUM, 0, &
                        MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(times, tmax, tdma_setup_profile_count, &
                        MPI_DOUBLE_PRECISION, MPI_MAX, 0, &
                        MPI_COMM_WORLD, ierr_mpi)

        if (myrank /= 0) return

        tavg = tsum / dble(nprocs)

        write(*,'(/,A)') &
            '================ TDMA-Z setup profile ================='
        write(*,'(A)') &
            'Columns: rank-min / rank-avg / rank-max [seconds]'

        do i = 1, tdma_setup_profile_count
            select case(i)
            case(1)
                label = '01 ensure cached coefficient/RHS workspaces'
            case(2)
                if (tdma_static_operator_prepared) then
                    label = '02 unpack changing RHS only on GPU'
                else
                    label = '02 rebuild TDMA coefficients/RHS on GPU'
                endif
            case(3)
                if (tdma_static_operator_prepared) then
                    label = '03 reuse static PaScaL_TDMA factors'
                else
                    label = '03 ensure/cache PaScaL_TDMA plan'
                endif
            case(4)
                label = '04 recombine real/imaginary solution'
            case(5)
                label = '05 release local pointer views'
            end select

            if (tmax(i) > 0.0d0) then
                write(*,'(A,1X,A44,3(1X,ES13.6))') &
                    '[TDMA-Z]', label, tmin(i), tavg(i), tmax(i)
            endif
        enddo

        write(*,'(A,/)') &
            '========================================================'
#endif
    end subroutine tdma_setup_profile_report


    subroutine ensure_tdma_z_workspaces(required_elements)
        implicit none

        integer, intent(in) :: required_elements
        logical :: resize_required, need_imag_coefficients

        need_imag_coefficients = .not. poisson_tdma_2rhs_enabled()

        resize_required = .not. allocated(tdma_api_work_d)
        if (.not. resize_required) then
            resize_required = size(tdma_api_work_d) < required_elements
        endif
        if (need_imag_coefficients .and. .not. allocated(tdma_apj_work_d)) then
            resize_required = .true.
        endif
        if (.not. resize_required) return

        if (allocated(tdma_api_work_d)) deallocate(tdma_api_work_d)
        if (allocated(tdma_aci_work_d)) deallocate(tdma_aci_work_d)
        if (allocated(tdma_ami_work_d)) deallocate(tdma_ami_work_d)
        if (allocated(tdma_apj_work_d)) deallocate(tdma_apj_work_d)
        if (allocated(tdma_acj_work_d)) deallocate(tdma_acj_work_d)
        if (allocated(tdma_amj_work_d)) deallocate(tdma_amj_work_d)
        if (allocated(tdma_rhs_real_work_d)) deallocate(tdma_rhs_real_work_d)
        if (allocated(tdma_rhs_imag_work_d)) deallocate(tdma_rhs_imag_work_d)

        allocate(tdma_api_work_d(required_elements))
        allocate(tdma_aci_work_d(required_elements))
        allocate(tdma_ami_work_d(required_elements))
        if (need_imag_coefficients) then
            allocate(tdma_apj_work_d(required_elements))
            allocate(tdma_acj_work_d(required_elements))
            allocate(tdma_amj_work_d(required_elements))
        endif
        allocate(tdma_rhs_real_work_d(required_elements))
        allocate(tdma_rhs_imag_work_d(required_elements))
    end subroutine ensure_tdma_z_workspaces


    subroutine ensure_ghost_workspaces(required_elements)
        implicit none

        integer, intent(in) :: required_elements
        logical :: resize_required

        resize_required = .not. allocated(ghost_send_0_d)
        if (.not. resize_required) then
            resize_required = size(ghost_send_0_d) < required_elements
        endif
        if (.not. resize_required) return

        if (allocated(ghost_send_0_d)) deallocate(ghost_send_0_d)
        if (allocated(ghost_send_1_d)) deallocate(ghost_send_1_d)
        if (allocated(ghost_recv_0_d)) deallocate(ghost_recv_0_d)
        if (allocated(ghost_recv_1_d)) deallocate(ghost_recv_1_d)

        allocate(ghost_send_0_d(required_elements))
        allocate(ghost_send_1_d(required_elements))
        allocate(ghost_recv_0_d(required_elements))
        allocate(ghost_recv_1_d(required_elements))
    end subroutine ensure_ghost_workspaces


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

    subroutine cuda_Poisson_cudecomp_initial( &
        n2msub_Isub, h1psub_Jsub)
        implicit none

        integer, intent(in) :: n2msub_Isub, h1psub_Jsub

#ifdef POISSON_USE_CUDECOMP
        if (BCtype(1) == 'P' .and. BCtype(2) == 'P') then
            call fft_cudecomp_initialize( &
                p_poi%n1m/2+1, p_poi%n2m, p_poi%n3m, &
                p_poi%comm_1d_x1%nprocs, &
                p_poi%comm_1d_x2%nprocs, &
                p_poi%comm_1d_x3%nprocs, &
                n2msub_Isub, p_poi%n3msub, h1psub_Jsub)
        endif
#endif
    end subroutine cuda_Poisson_cudecomp_initial


    !>
    !> @brief Build and factor the invariant P-P-N Poisson TDMA operator.
    !> @details This runs during application initialization, before CFD/Poisson
    !>          iterations.  It is intentionally outside every performance timer.
    !>
    subroutine cuda_Poisson_TDMA_static_initial( &
        n1td, n2td, n3td, dx3_d, dmx3_d, h1psub_Jsub_ista)
        use PaScaL_TDMA_cuda, only : &
            PaScaL_TDMA_plan_many_create_cuda, &
            PaScaL_TDMA_many_prepare_static_cuda
        implicit none

        integer, intent(in) :: n1td, n2td, n3td, h1psub_Jsub_ista
        real(rp), device, dimension(0:), intent(in) :: dx3_d, dmx3_d
        real(rp), pointer, device, contiguous, dimension(:,:,:) :: a_d, b_d, c_d
        type(comm_1d) :: comm_1d_x3
        integer :: i, j, k, kp, i_g, j_g
        integer :: global_rank, ierr
        real(rp) :: am, ac, ap

        call MPI_Comm_rank(MPI_COMM_WORLD, global_rank, ierr)

        if (.not. poisson_tdma_static_enabled()) then
            if (global_rank == 0) then
                write(*,'(A)') &
                    '[TDMA-STATIC] OFF: dynamic coefficient path selected for A/B.'
            endif
            tdma_static_operator_prepared = .false.
            return
        endif

        if (.not. poisson_tdma_2rhs_enabled() .or. p_poi%pbc3 .or. &
            BCtype(1) /= 'P' .or. BCtype(2) /= 'P') then
            if (global_rank == 0) then
                write(*,'(A)') &
                    '[TDMA-STATIC] Unsupported BC/RHS mode; using dynamic TDMA.'
            endif
            tdma_static_operator_prepared = .false.
            return
        endif

        call ensure_tdma_z_workspaces(n1td*n2td*n3td)
        a_d(1:n2td,1:n1td,1:n3td) => tdma_ami_work_d
        b_d(1:n2td,1:n1td,1:n3td) => tdma_aci_work_d
        c_d(1:n2td,1:n1td,1:n3td) => tdma_api_work_d
        comm_1d_x3 = p_poi%comm_1d_x3

        !$acc parallel loop collapse(3) private(kp, am, ac, ap, i_g, j_g) &
        !$acc& copyin(comm_1d_x3)
        do k = 1, n3td
        do i = 1, n1td
        do j = 1, n2td
            i_g = h1psub_Jsub_ista + i - 1
            j_g = j
            kp = k + 1

            am = real(1.0,rp)/dx3_d(k)/dmx3_d(k)
            if (comm_1d_x3%myrank == 0 .and. k == 1) &
                am = real(0.0,rp)
            ap = real(1.0,rp)/dx3_d(k)/dmx3_d(kp)
            if (comm_1d_x3%myrank == comm_1d_x3%nprocs-1 .and. &
                k == n3td) &
                ap = real(0.0,rp)
            ac = -am-ap

            a_d(j,i,k) = am
            b_d(j,i,k) = ac-dxk2(i_g)-dyk2(j_g)
            c_d(j,i,k) = ap

            if (comm_1d_x3%myrank == 0 .and. &
                i_g == 1 .and. j_g == 1 .and. k == 1) then
                a_d(1,1,1) = real(0.0,rp)
                b_d(1,1,1) = real(1.0,rp)
                c_d(1,1,1) = real(0.0,rp)
            endif
        enddo
        enddo
        enddo
        !$acc end parallel

        call PaScaL_TDMA_plan_many_create_cuda( &
            ptdma_plan_cuda_fft, n2td, n1td, n3td, &
            p_poi%comm_1d_x3%myrank, p_poi%comm_1d_x3%nprocs, &
            p_poi%comm_1d_x3%mpi_comm, p_poi%threads_tdma)
        call PaScaL_TDMA_many_prepare_static_cuda( &
            ptdma_plan_cuda_fft, a_d, b_d, c_d)

        tdma_static_operator_prepared = .true.
        if (global_rank == 0) then
            write(*,'(A,3(I0,1X))') &
                '[TDMA-STATIC] Prepared spectral TDMA shape: ', &
                n2td, n1td, n3td
        endif
        nullify(a_d, b_d, c_d)

    end subroutine cuda_Poisson_TDMA_static_initial


    subroutine cuda_Poisson_FFT_clean()
        implicit none

#ifdef POISSON_USE_CUDECOMP
        call fft_cudecomp_finalize()
#endif
        call cuda_Poisson_FFT_memory('clean')
        call cuda_cufft_plan_memory('clean')
        call cuda_PaScaL_TDMA_plan_many_memory('clean')
        tdma_static_operator_prepared = .false.
        tdma_static_banner_printed = .false.

        if (allocated(fft_mpi_send_r_d)) deallocate(fft_mpi_send_r_d)
        if (allocated(fft_mpi_recv_r_d)) deallocate(fft_mpi_recv_r_d)
        if (allocated(fft_mpi_send_c_d)) deallocate(fft_mpi_send_c_d)
        if (allocated(fft_mpi_recv_c_d)) deallocate(fft_mpi_recv_c_d)

        if (allocated(tdma_api_work_d)) deallocate(tdma_api_work_d)
        if (allocated(tdma_aci_work_d)) deallocate(tdma_aci_work_d)
        if (allocated(tdma_ami_work_d)) deallocate(tdma_ami_work_d)
        if (allocated(tdma_apj_work_d)) deallocate(tdma_apj_work_d)
        if (allocated(tdma_acj_work_d)) deallocate(tdma_acj_work_d)
        if (allocated(tdma_amj_work_d)) deallocate(tdma_amj_work_d)
        if (allocated(tdma_rhs_real_work_d)) deallocate(tdma_rhs_real_work_d)
        if (allocated(tdma_rhs_imag_work_d)) deallocate(tdma_rhs_imag_work_d)

        if (allocated(ghost_send_0_d)) deallocate(ghost_send_0_d)
        if (allocated(ghost_send_1_d)) deallocate(ghost_send_1_d)
        if (allocated(ghost_recv_0_d)) deallocate(ghost_recv_0_d)
        if (allocated(ghost_recv_1_d)) deallocate(ghost_recv_1_d)

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
                ! v19 NP fix: N-P uses both x-DCT complex workspace and y-periodic
                ! half-spectrum workspace.  Allocate a safe upper bound based on
                ! global transform lengths to cover all decompositions.
                allocate(Buff_c1(max0((p_poi%n1m/2+1)*p_poi%n2m, p_poi%n1m*(p_poi%n2m/2+1))*p_poi%n3msub), &
                         Buff_c2(max0((p_poi%n1m/2+1)*p_poi%n2m, p_poi%n1m*(p_poi%n2m/2+1))*p_poi%n3msub))
                allocate(Buff_1( p_poi%n1sub*p_poi%n3msub*p_poi%n2sub ), Buff_2( p_poi%n1sub*p_poi%n3msub*p_poi%n2sub ))
            elseif(BCtype(1)=='P'.and.BCtype(2)=='N') then
                allocate(Buff_c1(p_poi%n2msub*p_poi%n3msub*p_poi%n1msub/2 + max0(p_poi%n1msub,p_poi%n2msub)*p_poi%n3msub), Buff_c2(p_poi%n2msub*p_poi%n3msub*p_poi%n1msub/2 + max0(p_poi%n1msub,p_poi%n2msub)*p_poi%n3msub)) ! Considering n/2+1
                allocate(Buff_1( p_poi%n1sub*p_poi%n3msub*p_poi%n2sub ), Buff_2( p_poi%n1sub*p_poi%n3msub*p_poi%n2sub )) 
            elseif(BCtype(1)=='P'.and.BCtype(2)=='P') then
                ! v18 PP fix:
                ! PP uses redistributed complex workspaces such as
                !   FFT_xc(1:h1psub_Jsub,1:n2m,1:n3msub)
                ! after the C2J stage.  For decompositions with np2 > 1
                ! (e.g. np1,np2,np3 = 2,2,1), this can be larger than the
                ! old physical-local estimate
                !   n2msub*n3msub*(n1msub/2+1).
                ! Allocate a conservative full-y half-x spectral workspace
                ! to avoid silent out-of-bounds writes/corruption.
                allocate(Buff_c1((p_poi%n1m/2+1)*p_poi%n2m*p_poi%n3msub), &
                         Buff_c2((p_poi%n1m/2+1)*p_poi%n2m*p_poi%n3msub))
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

    subroutine cuda_cufft_get_cached_plan(ip, jp, nsize, istride, idist, ostride, odist, fft_type, nbatch)
        implicit none

        integer, intent(in) :: ip, jp
        integer, intent(in) :: nsize, istride, idist, ostride, odist, fft_type, nbatch
        integer :: ierr

        if (plan_fft_cache_n(ip,jp)       == nsize    .and. &
            plan_fft_cache_istride(ip,jp) == istride  .and. &
            plan_fft_cache_idist(ip,jp)   == idist    .and. &
            plan_fft_cache_ostride(ip,jp) == ostride  .and. &
            plan_fft_cache_odist(ip,jp)   == odist    .and. &
            plan_fft_cache_type(ip,jp)    == fft_type .and. &
            plan_fft_cache_nbatch(ip,jp)  == nbatch) then
            return
        endif

        if (plan_fft_cache_type(ip,jp) /= -1) then
            ierr = cufftDestroy(plan_fft(ip,jp))
        endif

        ierr = cufftPlanMany(plan_fft(ip,jp), 1, nsize, null(), istride, idist, null(), ostride, odist, fft_type, nbatch)
        if (ierr /= 0) then
            write(*,*) 'ERROR: cufftPlanMany failed in cached plan creation'
            write(*,*) 'slot/ip/jp = ', ip, jp
            write(*,*) 'nsize/istride/idist/ostride/odist/type/nbatch = ', &
                       nsize, istride, idist, ostride, odist, fft_type, nbatch
            write(*,*) 'cufft ierr = ', ierr
            stop 7701
        endif

        plan_fft_cache_n(ip,jp)       = nsize
        plan_fft_cache_istride(ip,jp) = istride
        plan_fft_cache_idist(ip,jp)   = idist
        plan_fft_cache_ostride(ip,jp) = ostride
        plan_fft_cache_odist(ip,jp)   = odist
        plan_fft_cache_type(ip,jp)    = fft_type
        plan_fft_cache_nbatch(ip,jp)  = nbatch
    end subroutine cuda_cufft_get_cached_plan

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
                                call cuda_cufft_get_cached_plan(1, 1, nsize, istride, idist, ostride, odist, CUFFT_R2C, nbatch)
            #elif DOUBLE_PRECISION
                                call cuda_cufft_get_cached_plan(1, 1, nsize, istride, idist, ostride, odist, CUFFT_D2Z, nbatch)
            #endif
            ! plan_dct_b_x
            nsize = p_poi%n1m
            istride = 1
            ostride = 1
            idist = int(p_poi%n1m/2)+1
            odist = p_poi%n1m
            nbatch = p_poi%n2msub*p_poi%n3msub
            #ifdef SINGLE_PRECISION
                                call cuda_cufft_get_cached_plan(2, 1, nsize, istride, idist, ostride, odist, CUFFT_C2R, nbatch)	
            #elif DOUBLE_PRECISION
                                call cuda_cufft_get_cached_plan(2, 1, nsize, istride, idist, ostride, odist, CUFFT_Z2D, nbatch)	
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
                                        call cuda_cufft_get_cached_plan(1, 2, nsize, istride, idist, ostride, odist, CUFFT_C2C, nbatch)	
                #elif DOUBLE_PRECISION
                                        call cuda_cufft_get_cached_plan(1, 2, nsize, istride, idist, ostride, odist, CUFFT_Z2Z, nbatch)	
                #endif
                ! plan_fft_b_y	
                nsize = p_poi%n2m
                istride = 1
                ostride = 1
                idist = p_poi%n2m
                odist = p_poi%n2m
                nbatch = (int(p_poi%n1m/2)+1)*p_poi%n3msub
                #ifdef SINGLE_PRECISION
                                        call cuda_cufft_get_cached_plan(2, 2, nsize, istride, idist, ostride, odist, CUFFT_C2C, nbatch)	
                #elif DOUBLE_PRECISION
                                        call cuda_cufft_get_cached_plan(2, 2, nsize, istride, idist, ostride, odist, CUFFT_Z2Z, nbatch)
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
                                        call cuda_cufft_get_cached_plan(1, 2, nsize, istride, idist, ostride, odist, CUFFT_R2C, nbatch)	
                #elif DOUBLE_PRECISION
                                        call cuda_cufft_get_cached_plan(1, 2, nsize, istride, idist, ostride, odist, CUFFT_D2Z, nbatch)	
                #endif

                ! plan_dct_b_y
                nsize = p_poi%n2m
                istride = 1
                ostride = 1
                idist = p_poi%n2m/2+1
                odist = p_poi%n2m
                nbatch = p_poi%n1m*p_poi%n3msub
                #ifdef SINGLE_PRECISION
                                        call cuda_cufft_get_cached_plan(2, 2, nsize, istride, idist, ostride, odist, CUFFT_C2R, nbatch)	
                #elif DOUBLE_PRECISION
                                        call cuda_cufft_get_cached_plan(2, 2, nsize, istride, idist, ostride, odist, CUFFT_Z2D, nbatch)	
                #endif
            endif
        case('clean')   
            ierr = cufftDestroy(plan_fft(1,1))
            ierr = cufftDestroy(plan_fft(1,2))
            ierr = cufftDestroy(plan_fft(2,2))
            ierr = cufftDestroy(plan_fft(2,1))
            plan_fft_cache_n(:,:)       = -1
            plan_fft_cache_istride(:,:) = -1
            plan_fft_cache_idist(:,:)   = -1
            plan_fft_cache_ostride(:,:) = -1
            plan_fft_cache_odist(:,:)   = -1
            plan_fft_cache_type(:,:)    = -1
            plan_fft_cache_nbatch(:,:)  = -1
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


    subroutine fft_block_range(total_count, nprocs, rank, first_index, last_index)
        implicit none

        integer, intent(in)  :: total_count, nprocs, rank
        integer, intent(out) :: first_index, last_index
        integer :: base_count, remainder

        base_count = total_count / nprocs
        remainder  = mod(total_count, nprocs)

        first_index = rank * base_count + 1 + min(rank, remainder)
        last_index  = first_index + base_count - 1
        if (remainder > rank) last_index = last_index + 1
    end subroutine fft_block_range


    subroutine fft_prepare_alltoall_metadata(metadata, nprocs, &
                                             send_split_extent, &
                                             send_plane_extent, &
                                             recv_split_extent, &
                                             recv_plane_extent)
        implicit none

        type(fft_alltoall_metadata), intent(inout) :: metadata
        integer, intent(in) :: nprocs
        integer, intent(in) :: send_split_extent, send_plane_extent
        integer, intent(in) :: recv_split_extent, recv_plane_extent

        integer :: peer, first_index, last_index, peer_count
        logical :: cache_matches

        cache_matches = allocated(metadata%sendcounts)
        cache_matches = cache_matches .and. metadata%nprocs == nprocs
        cache_matches = cache_matches .and. &
                        metadata%send_split_extent == send_split_extent
        cache_matches = cache_matches .and. &
                        metadata%send_plane_extent == send_plane_extent
        cache_matches = cache_matches .and. &
                        metadata%recv_split_extent == recv_split_extent
        cache_matches = cache_matches .and. &
                        metadata%recv_plane_extent == recv_plane_extent
        if (cache_matches) return

        if (allocated(metadata%sendcounts)) deallocate(metadata%sendcounts)
        if (allocated(metadata%recvcounts)) deallocate(metadata%recvcounts)
        if (allocated(metadata%senddispls)) deallocate(metadata%senddispls)
        if (allocated(metadata%recvdispls)) deallocate(metadata%recvdispls)

        allocate(metadata%sendcounts(0:nprocs-1))
        allocate(metadata%recvcounts(0:nprocs-1))
        allocate(metadata%senddispls(0:nprocs-1))
        allocate(metadata%recvdispls(0:nprocs-1))

        metadata%send_total = 0
        metadata%recv_total = 0
        do peer = 0, nprocs-1
            call fft_block_range(send_split_extent, nprocs, peer, &
                                 first_index, last_index)
            peer_count = max(0, last_index-first_index+1)
            metadata%sendcounts(peer) = peer_count * send_plane_extent
            metadata%senddispls(peer) = metadata%send_total
            metadata%send_total = metadata%send_total + &
                                  metadata%sendcounts(peer)

            call fft_block_range(recv_split_extent, nprocs, peer, &
                                 first_index, last_index)
            peer_count = max(0, last_index-first_index+1)
            metadata%recvcounts(peer) = peer_count * recv_plane_extent
            metadata%recvdispls(peer) = metadata%recv_total
            metadata%recv_total = metadata%recv_total + &
                                  metadata%recvcounts(peer)
        enddo

        metadata%nprocs = nprocs
        metadata%send_split_extent = send_split_extent
        metadata%send_plane_extent = send_plane_extent
        metadata%recv_split_extent = recv_split_extent
        metadata%recv_plane_extent = recv_plane_extent
    end subroutine fft_prepare_alltoall_metadata


    subroutine fft_ensure_real_mpi_workspace(send_size, recv_size)
        implicit none

        integer, intent(in) :: send_size, recv_size
        integer :: send_need, recv_need

        send_need = max(1, send_size)
        recv_need = max(1, recv_size)

        if (.not. allocated(fft_mpi_send_r_d)) then
            allocate(fft_mpi_send_r_d(send_need))
        elseif (size(fft_mpi_send_r_d) < send_need) then
            deallocate(fft_mpi_send_r_d)
            allocate(fft_mpi_send_r_d(send_need))
        endif

        if (.not. allocated(fft_mpi_recv_r_d)) then
            allocate(fft_mpi_recv_r_d(recv_need))
        elseif (size(fft_mpi_recv_r_d) < recv_need) then
            deallocate(fft_mpi_recv_r_d)
            allocate(fft_mpi_recv_r_d(recv_need))
        endif
    end subroutine fft_ensure_real_mpi_workspace


    subroutine fft_ensure_complex_mpi_workspace(send_size, recv_size)
        implicit none

        integer, intent(in) :: send_size, recv_size
        integer :: send_need, recv_need

        send_need = max(1, send_size)
        recv_need = max(1, recv_size)

        if (.not. allocated(fft_mpi_send_c_d)) then
            allocate(fft_mpi_send_c_d(send_need))
        elseif (size(fft_mpi_send_c_d) < send_need) then
            deallocate(fft_mpi_send_c_d)
            allocate(fft_mpi_send_c_d(send_need))
        endif

        if (.not. allocated(fft_mpi_recv_c_d)) then
            allocate(fft_mpi_recv_c_d(recv_need))
        elseif (size(fft_mpi_recv_c_d) < recv_need) then
            deallocate(fft_mpi_recv_c_d)
            allocate(fft_mpi_recv_c_d(recv_need))
        endif
    end subroutine fft_ensure_complex_mpi_workspace


    subroutine fft_pack_real_subarray(src_d, packed_d, i_first, ni, j_first, nj, k_first, nk, offset)
        implicit none

        real(rp), device, dimension(:,:,:), intent(in)  :: src_d
        real(rp), device, dimension(:),     intent(inout) :: packed_d
        integer, intent(in) :: i_first, ni, j_first, nj, k_first, nk, offset
        integer :: i, j, k, packed_index

        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, nk
        do j = 1, nj
        do i = 1, ni
            packed_index = offset + i + ni * ((j - 1) + nj * (k - 1))
            packed_d(packed_index) = src_d(i_first+i-1, j_first+j-1, k_first+k-1)
        enddo
        enddo
        enddo
    end subroutine fft_pack_real_subarray


    subroutine fft_unpack_real_subarray(packed_d, dst_d, i_first, ni, j_first, nj, k_first, nk, offset)
        implicit none

        real(rp), device, dimension(:),     intent(in)  :: packed_d
        real(rp), device, dimension(:,:,:), intent(inout) :: dst_d
        integer, intent(in) :: i_first, ni, j_first, nj, k_first, nk, offset
        integer :: i, j, k, packed_index

        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, nk
        do j = 1, nj
        do i = 1, ni
            packed_index = offset + i + ni * ((j - 1) + nj * (k - 1))
            dst_d(i_first+i-1, j_first+j-1, k_first+k-1) = packed_d(packed_index)
        enddo
        enddo
        enddo
    end subroutine fft_unpack_real_subarray


    subroutine fft_pack_complex_subarray(src_d, packed_d, i_first, ni, j_first, nj, k_first, nk, offset)
        implicit none

        complex(rp), device, dimension(:,:,:), intent(in)  :: src_d
        complex(rp), device, dimension(:),     intent(inout) :: packed_d
        integer, intent(in) :: i_first, ni, j_first, nj, k_first, nk, offset
        integer :: i, j, k, packed_index

        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, nk
        do j = 1, nj
        do i = 1, ni
            packed_index = offset + i + ni * ((j - 1) + nj * (k - 1))
            packed_d(packed_index) = src_d(i_first+i-1, j_first+j-1, k_first+k-1)
        enddo
        enddo
        enddo
    end subroutine fft_pack_complex_subarray


    subroutine fft_unpack_complex_subarray(packed_d, dst_d, i_first, ni, j_first, nj, k_first, nk, offset)
        implicit none

        complex(rp), device, dimension(:),     intent(in)  :: packed_d
        complex(rp), device, dimension(:,:,:), intent(inout) :: dst_d
        integer, intent(in) :: i_first, ni, j_first, nj, k_first, nk, offset
        integer :: i, j, k, packed_index

        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, nk
        do j = 1, nj
        do i = 1, ni
            packed_index = offset + i + ni * ((j - 1) + nj * (k - 1))
            dst_d(i_first+i-1, j_first+j-1, k_first+k-1) = packed_d(packed_index)
        enddo
        enddo
        enddo
    end subroutine fft_unpack_complex_subarray


    subroutine fft_copy_real_3d(src_d, dst_d, n1_copy, n2_copy, n3_copy)
        implicit none

        real(rp), device, dimension(:,:,:), intent(in)  :: src_d
        real(rp), device, dimension(:,:,:), intent(inout) :: dst_d
        integer, intent(in) :: n1_copy, n2_copy, n3_copy
        integer :: i, j, k

        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, n3_copy
        do j = 1, n2_copy
        do i = 1, n1_copy
            dst_d(i,j,k) = src_d(i,j,k)
        enddo
        enddo
        enddo
    end subroutine fft_copy_real_3d


    subroutine fft_copy_complex_3d(src_d, dst_d, n1_copy, n2_copy, n3_copy)
        implicit none

        complex(rp), device, dimension(:,:,:), intent(in)  :: src_d
        complex(rp), device, dimension(:,:,:), intent(inout) :: dst_d
        integer, intent(in) :: n1_copy, n2_copy, n3_copy
        integer :: i, j, k

        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, n3_copy
        do j = 1, n2_copy
        do i = 1, n1_copy
            dst_d(i,j,k) = src_d(i,j,k)
        enddo
        enddo
        enddo
    end subroutine fft_copy_complex_3d


    subroutine fft_c2i_real_contiguous(c_d, i_d, n1c, n2c, n3c, n1i, n2i, comm)
        implicit none

        real(rp), device, dimension(:,:,:), intent(in)  :: c_d
        real(rp), device, dimension(:,:,:), intent(out) :: i_d
        integer, intent(in) :: n1c, n2c, n3c, n1i, n2i, comm

        integer :: ierr, nprocs, myrank, peer
        integer :: first_index, last_index, peer_count
        integer :: send_total, recv_total

        call MPI_Comm_size(comm, nprocs, ierr)
        call MPI_Comm_rank(comm, myrank, ierr)

        if (nprocs == 1) then
            call fft_copy_real_3d(c_d, i_d, n1c, n2c, n3c)
            return
        endif

        call fft_prepare_alltoall_metadata(fft_meta_c2i_real, nprocs, &
                                           n2c, n1c*n3c, &
                                           n1i, n2i*n3c)
        send_total = fft_meta_c2i_real%send_total
        recv_total = fft_meta_c2i_real%recv_total

        call fft_ensure_real_mpi_workspace(send_total, recv_total)

        do peer = 0, nprocs-1
            call fft_block_range(n2c, nprocs, peer, first_index, last_index)
            peer_count = max(0, last_index-first_index+1)
            if (peer_count > 0) then
                call fft_pack_real_subarray(c_d, fft_mpi_send_r_d, &
                    1, n1c, first_index, peer_count, 1, n3c, &
                    fft_meta_c2i_real%senddispls(peer))
            endif
        enddo

        ierr = cudaStreamSynchronize()
        call MPI_Alltoallv(fft_mpi_send_r_d, &
                           fft_meta_c2i_real%sendcounts, &
                           fft_meta_c2i_real%senddispls, MPI_real_type, &
                           fft_mpi_recv_r_d, &
                           fft_meta_c2i_real%recvcounts, &
                           fft_meta_c2i_real%recvdispls, MPI_real_type, &
                           comm, ierr)

        do peer = 0, nprocs-1
            call fft_block_range(n1i, nprocs, peer, first_index, last_index)
            peer_count = max(0, last_index-first_index+1)
            if (peer_count > 0) then
                call fft_unpack_real_subarray(fft_mpi_recv_r_d, i_d, &
                    first_index, peer_count, 1, n2i, 1, n3c, &
                    fft_meta_c2i_real%recvdispls(peer))
            endif
        enddo
    end subroutine fft_c2i_real_contiguous


    subroutine fft_i2c_real_contiguous(i_d, c_d, n1i, n2i, n3c, n1c, n2c, comm)
        implicit none

        real(rp), device, dimension(:,:,:), intent(in)  :: i_d
        real(rp), device, dimension(:,:,:), intent(out) :: c_d
        integer, intent(in) :: n1i, n2i, n3c, n1c, n2c, comm

        integer :: ierr, nprocs, myrank, peer
        integer :: first_index, last_index, peer_count
        integer :: send_total, recv_total

        call MPI_Comm_size(comm, nprocs, ierr)
        call MPI_Comm_rank(comm, myrank, ierr)

        if (nprocs == 1) then
            call fft_copy_real_3d(i_d, c_d, n1c, n2c, n3c)
            return
        endif

        call fft_prepare_alltoall_metadata(fft_meta_i2c_real, nprocs, &
                                           n1i, n2i*n3c, &
                                           n2c, n1c*n3c)
        send_total = fft_meta_i2c_real%send_total
        recv_total = fft_meta_i2c_real%recv_total

        call fft_ensure_real_mpi_workspace(send_total, recv_total)

        do peer = 0, nprocs-1
            call fft_block_range(n1i, nprocs, peer, first_index, last_index)
            peer_count = max(0, last_index-first_index+1)
            if (peer_count > 0) then
                call fft_pack_real_subarray(i_d, fft_mpi_send_r_d, &
                    first_index, peer_count, 1, n2i, 1, n3c, &
                    fft_meta_i2c_real%senddispls(peer))
            endif
        enddo

        ierr = cudaStreamSynchronize()
        call MPI_Alltoallv(fft_mpi_send_r_d, &
                           fft_meta_i2c_real%sendcounts, &
                           fft_meta_i2c_real%senddispls, MPI_real_type, &
                           fft_mpi_recv_r_d, &
                           fft_meta_i2c_real%recvcounts, &
                           fft_meta_i2c_real%recvdispls, MPI_real_type, &
                           comm, ierr)

        do peer = 0, nprocs-1
            call fft_block_range(n2c, nprocs, peer, first_index, last_index)
            peer_count = max(0, last_index-first_index+1)
            if (peer_count > 0) then
                call fft_unpack_real_subarray(fft_mpi_recv_r_d, c_d, &
                    1, n1c, first_index, peer_count, 1, n3c, &
                    fft_meta_i2c_real%recvdispls(peer))
            endif
        enddo
    end subroutine fft_i2c_real_contiguous


    subroutine fft_c2i_complex_contiguous(c_d, i_d, n1c, n2c, n3c, n1i, n2i, comm)
        implicit none

        complex(rp), device, dimension(:,:,:), intent(in)  :: c_d
        complex(rp), device, dimension(:,:,:), intent(out) :: i_d
        integer, intent(in) :: n1c, n2c, n3c, n1i, n2i, comm

        integer :: ierr, nprocs, myrank, peer
        integer :: first_index, last_index, peer_count
        integer :: send_total, recv_total

        call MPI_Comm_size(comm, nprocs, ierr)
        call MPI_Comm_rank(comm, myrank, ierr)

        if (nprocs == 1) then
            call fft_copy_complex_3d(c_d, i_d, n1c, n2c, n3c)
            return
        endif

        call fft_prepare_alltoall_metadata(fft_meta_c2i_complex, nprocs, &
                                           n2c, n1c*n3c, &
                                           n1i, n2i*n3c)
        send_total = fft_meta_c2i_complex%send_total
        recv_total = fft_meta_c2i_complex%recv_total

        call fft_ensure_complex_mpi_workspace(send_total, recv_total)

        do peer = 0, nprocs-1
            call fft_block_range(n2c, nprocs, peer, first_index, last_index)
            peer_count = max(0, last_index-first_index+1)
            if (peer_count > 0) then
                call fft_pack_complex_subarray(c_d, fft_mpi_send_c_d, &
                    1, n1c, first_index, peer_count, 1, n3c, &
                    fft_meta_c2i_complex%senddispls(peer))
            endif
        enddo

        ierr = cudaStreamSynchronize()
        call MPI_Alltoallv(fft_mpi_send_c_d, &
                           fft_meta_c2i_complex%sendcounts, &
                           fft_meta_c2i_complex%senddispls, MPI_complex_type, &
                           fft_mpi_recv_c_d, &
                           fft_meta_c2i_complex%recvcounts, &
                           fft_meta_c2i_complex%recvdispls, MPI_complex_type, &
                           comm, ierr)

        do peer = 0, nprocs-1
            call fft_block_range(n1i, nprocs, peer, first_index, last_index)
            peer_count = max(0, last_index-first_index+1)
            if (peer_count > 0) then
                call fft_unpack_complex_subarray(fft_mpi_recv_c_d, i_d, &
                    first_index, peer_count, 1, n2i, 1, n3c, &
                    fft_meta_c2i_complex%recvdispls(peer))
            endif
        enddo
    end subroutine fft_c2i_complex_contiguous


    subroutine fft_i2c_complex_contiguous(i_d, c_d, n1i, n2i, n3c, n1c, n2c, comm)
        implicit none

        complex(rp), device, dimension(:,:,:), intent(in)  :: i_d
        complex(rp), device, dimension(:,:,:), intent(out) :: c_d
        integer, intent(in) :: n1i, n2i, n3c, n1c, n2c, comm

        integer :: ierr, nprocs, myrank, peer
        integer :: first_index, last_index, peer_count
        integer :: send_total, recv_total

        call MPI_Comm_size(comm, nprocs, ierr)
        call MPI_Comm_rank(comm, myrank, ierr)

        if (nprocs == 1) then
            call fft_copy_complex_3d(i_d, c_d, n1c, n2c, n3c)
            return
        endif

        call fft_prepare_alltoall_metadata(fft_meta_i2c_complex, nprocs, &
                                           n1i, n2i*n3c, &
                                           n2c, n1c*n3c)
        send_total = fft_meta_i2c_complex%send_total
        recv_total = fft_meta_i2c_complex%recv_total

        call fft_ensure_complex_mpi_workspace(send_total, recv_total)

        do peer = 0, nprocs-1
            call fft_block_range(n1i, nprocs, peer, first_index, last_index)
            peer_count = max(0, last_index-first_index+1)
            if (peer_count > 0) then
                call fft_pack_complex_subarray(i_d, fft_mpi_send_c_d, &
                    first_index, peer_count, 1, n2i, 1, n3c, &
                    fft_meta_i2c_complex%senddispls(peer))
            endif
        enddo

        ierr = cudaStreamSynchronize()
        call MPI_Alltoallv(fft_mpi_send_c_d, &
                           fft_meta_i2c_complex%sendcounts, &
                           fft_meta_i2c_complex%senddispls, MPI_complex_type, &
                           fft_mpi_recv_c_d, &
                           fft_meta_i2c_complex%recvcounts, &
                           fft_meta_i2c_complex%recvdispls, MPI_complex_type, &
                           comm, ierr)

        do peer = 0, nprocs-1
            call fft_block_range(n2c, nprocs, peer, first_index, last_index)
            peer_count = max(0, last_index-first_index+1)
            if (peer_count > 0) then
                call fft_unpack_complex_subarray(fft_mpi_recv_c_d, c_d, &
                    1, n1c, first_index, peer_count, 1, n3c, &
                    fft_meta_i2c_complex%recvdispls(peer))
            endif
        enddo
    end subroutine fft_i2c_complex_contiguous


    subroutine fft_c2j_complex_contiguous(c_d, j_d, n1c, n2c, n3c, n1j, n2j, comm)
        implicit none

        complex(rp), device, dimension(:,:,:), intent(in)  :: c_d
        complex(rp), device, dimension(:,:,:), intent(out) :: j_d
        integer, intent(in) :: n1c, n2c, n3c, n1j, n2j, comm

        integer :: ierr, nprocs, myrank, peer
        integer :: first_index, last_index, peer_count
        integer :: send_total, recv_total
#ifdef POISSON_DETAILED_PROFILE
        double precision :: route_t0
        double precision :: route_phase_times(fft_route_phase_count)
        double precision :: route_payload_bytes
#endif

        call MPI_Comm_size(comm, nprocs, ierr)
        call MPI_Comm_rank(comm, myrank, ierr)

        if (nprocs == 1) then
            call fft_copy_complex_3d(c_d, j_d, n1c, n2c, n3c)
            return
        endif

        call fft_prepare_alltoall_metadata(fft_meta_c2j_complex, nprocs, &
                                           n1c, n2c*n3c, &
                                           n2j, n1j*n3c)
        send_total = fft_meta_c2j_complex%send_total
        recv_total = fft_meta_c2j_complex%recv_total

        call fft_ensure_complex_mpi_workspace(send_total, recv_total)

#ifdef POISSON_DETAILED_PROFILE
        route_phase_times = 0.0d0
        route_payload_bytes = 0.0d0
        if (poisson_profile_enabled) route_t0 = MPI_Wtime()
#endif
        do peer = 0, nprocs-1
            call fft_block_range(n1c, nprocs, peer, first_index, last_index)
            peer_count = max(0, last_index-first_index+1)
            if (peer_count > 0) then
                call fft_pack_complex_subarray(c_d, fft_mpi_send_c_d, &
                    first_index, peer_count, 1, n2c, 1, n3c, &
                    fft_meta_c2j_complex%senddispls(peer))
            endif
        enddo

        ierr = cudaStreamSynchronize()
#ifdef POISSON_DETAILED_PROFILE
        if (poisson_profile_enabled) then
            route_phase_times(1) = MPI_Wtime() - route_t0
            route_t0 = MPI_Wtime()
        endif
#endif
        call MPI_Alltoallv(fft_mpi_send_c_d, &
                           fft_meta_c2j_complex%sendcounts, &
                           fft_meta_c2j_complex%senddispls, MPI_complex_type, &
                           fft_mpi_recv_c_d, &
                           fft_meta_c2j_complex%recvcounts, &
                           fft_meta_c2j_complex%recvdispls, MPI_complex_type, &
                           comm, ierr)

#ifdef POISSON_DETAILED_PROFILE
        if (poisson_profile_enabled) then
            route_phase_times(2) = MPI_Wtime() - route_t0
            route_t0 = MPI_Wtime()
        endif
#endif
        do peer = 0, nprocs-1
            call fft_block_range(n2j, nprocs, peer, first_index, last_index)
            peer_count = max(0, last_index-first_index+1)
            if (peer_count > 0) then
                call fft_unpack_complex_subarray(fft_mpi_recv_c_d, j_d, &
                    1, n1j, first_index, peer_count, 1, n3c, &
                    fft_meta_c2j_complex%recvdispls(peer))
            endif
        enddo
#ifdef POISSON_DETAILED_PROFILE
        if (poisson_profile_enabled) then
            ierr = cudaStreamSynchronize()
            route_phase_times(3) = MPI_Wtime() - route_t0
#ifdef SINGLE_PRECISION
            route_payload_bytes = dble(send_total) * 8.0d0
#else
            route_payload_bytes = dble(send_total) * 16.0d0
#endif
            call fft_route_profile_record(1, route_phase_times, &
                                          route_payload_bytes)
        endif
#endif
    end subroutine fft_c2j_complex_contiguous


    subroutine fft_j2c_complex_contiguous(j_d, c_d, n1j, n2j, n3c, n1c, n2c, comm)
        implicit none

        complex(rp), device, dimension(:,:,:), intent(in)  :: j_d
        complex(rp), device, dimension(:,:,:), intent(out) :: c_d
        integer, intent(in) :: n1j, n2j, n3c, n1c, n2c, comm

        integer :: ierr, nprocs, myrank, peer
        integer :: first_index, last_index, peer_count
        integer :: send_total, recv_total
#ifdef POISSON_DETAILED_PROFILE
        double precision :: route_t0
        double precision :: route_phase_times(fft_route_phase_count)
        double precision :: route_payload_bytes
#endif

        call MPI_Comm_size(comm, nprocs, ierr)
        call MPI_Comm_rank(comm, myrank, ierr)

        if (nprocs == 1) then
            call fft_copy_complex_3d(j_d, c_d, n1c, n2c, n3c)
            return
        endif

        call fft_prepare_alltoall_metadata(fft_meta_j2c_complex, nprocs, &
                                           n2j, n1j*n3c, &
                                           n1c, n2c*n3c)
        send_total = fft_meta_j2c_complex%send_total
        recv_total = fft_meta_j2c_complex%recv_total

        call fft_ensure_complex_mpi_workspace(send_total, recv_total)

#ifdef POISSON_DETAILED_PROFILE
        route_phase_times = 0.0d0
        route_payload_bytes = 0.0d0
        if (poisson_profile_enabled) route_t0 = MPI_Wtime()
#endif
        do peer = 0, nprocs-1
            call fft_block_range(n2j, nprocs, peer, first_index, last_index)
            peer_count = max(0, last_index-first_index+1)
            if (peer_count > 0) then
                call fft_pack_complex_subarray(j_d, fft_mpi_send_c_d, &
                    1, n1j, first_index, peer_count, 1, n3c, &
                    fft_meta_j2c_complex%senddispls(peer))
            endif
        enddo

        ierr = cudaStreamSynchronize()
#ifdef POISSON_DETAILED_PROFILE
        if (poisson_profile_enabled) then
            route_phase_times(1) = MPI_Wtime() - route_t0
            route_t0 = MPI_Wtime()
        endif
#endif
        call MPI_Alltoallv(fft_mpi_send_c_d, &
                           fft_meta_j2c_complex%sendcounts, &
                           fft_meta_j2c_complex%senddispls, MPI_complex_type, &
                           fft_mpi_recv_c_d, &
                           fft_meta_j2c_complex%recvcounts, &
                           fft_meta_j2c_complex%recvdispls, MPI_complex_type, &
                           comm, ierr)

#ifdef POISSON_DETAILED_PROFILE
        if (poisson_profile_enabled) then
            route_phase_times(2) = MPI_Wtime() - route_t0
            route_t0 = MPI_Wtime()
        endif
#endif
        do peer = 0, nprocs-1
            call fft_block_range(n1c, nprocs, peer, first_index, last_index)
            peer_count = max(0, last_index-first_index+1)
            if (peer_count > 0) then
                call fft_unpack_complex_subarray(fft_mpi_recv_c_d, c_d, &
                    first_index, peer_count, 1, n2c, 1, n3c, &
                    fft_meta_j2c_complex%recvdispls(peer))
            endif
        enddo
#ifdef POISSON_DETAILED_PROFILE
        if (poisson_profile_enabled) then
            ierr = cudaStreamSynchronize()
            route_phase_times(3) = MPI_Wtime() - route_t0
#ifdef SINGLE_PRECISION
            route_payload_bytes = dble(send_total) * 8.0d0
#else
            route_payload_bytes = dble(send_total) * 16.0d0
#endif
            call fft_route_profile_record(2, route_phase_times, &
                                          route_payload_bytes)
        endif
#endif
    end subroutine fft_j2c_complex_contiguous


    subroutine cuda_Poisson_FFT_1D(PRHS_d, P_d, dmx1_d, dmx2_d, dmx3_d, dx3_d, h1psub, h1psub_Jsub, n2msub_Isub, n1msub_Jsub, &
                                   countsendI, countdistI, countsendJ, countdistJ, ddtype_dble_C_in_C2I,ddtype_dble_I_in_C2I, &
                                   ddtype_dble_J_in_C2J, ddtype_dble_C_in_C2J, ddtype_cplx_I_in_C2I, ddtype_cplx_C_in_C2I,    &
                                   ddtype_cplx_J_in_C2J, ddtype_cplx_C_in_C2J, iend, ista, jend, jsta, h1psub_Jsub_ista, n2msub_Isub_jsta, n1msub_Jsub_ista)
        use cublas
        use MPI
        implicit none

        real(rp),             device, dimension(1:,1:,1:) :: PRHS_d
        real(rp),             device, dimension(0:,0:,0:) :: P_d 
        real(rp),             device, dimension(:)        :: dmx1_d, dmx2_d, dmx3_d
        real(rp),             device, dimension(:)        :: dx3_d

        real(rp),    pointer, device, contiguous, dimension( :, :, :) :: FFT_x1, FFT_x2, FFT_y1, FFT_y2, FFT_x3, FFT_y3
        complex(rp), pointer, device, contiguous, dimension( :, :, :) :: FFT_xc, FFT_yc

        integer :: ierr
        double precision :: prof_t0
        double precision :: prof_times(poisson_profile_count)

        integer :: n1msub, n2msub, n3msub
        integer :: n1m, n2m 
        integer :: n1sub, n2sub

        ! Dongyun
        integer :: iend, ista, jend, jsta, h1psub_Jsub_ista, n2msub_Isub_jsta, n1msub_Jsub_ista
        integer :: buffer_cd_size_d, buffer_dp_size_d, n1, n2
        real(rp), pointer, device, contiguous, dimension(:,:,:) :: PRHS_Iline_d, PRHS_Jline_d, rhsihat_jline_d
        complex(rp), pointer, device, contiguous, dimension(:,:,:) :: prhs_cplx_d
        real(rp), allocatable, device, target, save, dimension(:) :: buffer_dp1_d, buffer_dp2_d
        complex(rp), allocatable, device, target, save, dimension(:) :: buffer_cd1_d, buffer_cd2_d
        integer :: n2msub_Isub, h1psub, h1psub_Jsub
        integer :: n1msub_Jsub
        integer, dimension(:) :: ddtype_dble_C_in_C2I, ddtype_dble_I_in_C2I, ddtype_dble_J_in_C2J, &
                                 ddtype_cplx_I_in_C2I, ddtype_cplx_C_in_C2I, &
                                 ddtype_cplx_J_in_C2J, ddtype_cplx_C_in_C2J, &
                                 ddtype_dble_C_in_C2J, countsendI, countdistI, countsendJ, countdistJ
        
        n2 = p_poi%n2; n1 = p_poi%n1;
        ! Dongyun

        n1msub = p_poi%n1msub; n2msub = p_poi%n2msub; n3msub = p_poi%n3msub;
        n1m = p_poi%n1m; n2m = p_poi%n2m; 
        n1sub = p_poi%n1sub; n2sub = p_poi%n2sub;
        prof_times = 0.0d0
#ifdef POISSON_DETAILED_PROFILE
        call fft_route_profile_reset()
#endif


        ! Forward dct
        ! call nvtxStartRange("Poisson")
        ! call nvtxStartRange("Poisson-F_x")

        call poisson_coarse_profile_begin()

        if    ((BCtype(1)=='N'.and.BCtype(2)=='N') .or. (BCtype(1)=='N'.and.BCtype(2)=='P')) then ! X-R2R
            ! Dongyun
            buffer_dp_size_d = max(n1m * n2msub_Isub * n3msub, n1msub_Jsub * n2m * n3msub) ! 이거 max로 꼭 수정
            if (.not. allocated(buffer_dp1_d)) then
                allocate( buffer_dp1_d( buffer_dp_size_d ) )
            elseif (size(buffer_dp1_d) < buffer_dp_size_d) then
                deallocate( buffer_dp1_d )
                allocate( buffer_dp1_d( buffer_dp_size_d ) )
            endif
            if (.not. allocated(buffer_dp2_d)) then
                allocate( buffer_dp2_d( buffer_dp_size_d ) )
            elseif (size(buffer_dp2_d) < buffer_dp_size_d) then
                deallocate( buffer_dp2_d )
                allocate( buffer_dp2_d( buffer_dp_size_d ) )
            endif
            PRHS_Iline_d(1:n1m,1:n2msub_Isub,1:n3msub) => buffer_dp1_d


            ierr = cudaStreamSynchronize()
            call MPI_alltoallw(PRHS_d,                    &
                               countsendI,                &
                               countdistI,                &
                               ddtype_dble_C_in_C2I,      &
                               PRHS_Iline_d,              &
                               countsendI,                &
                               countdistI,                &
                               ddtype_dble_I_in_C2I,      &
                               p_poi%comm_1d_x1%mpi_comm, &
                               ierr)
            ! x방향 DCT
            ! 나중에 Buff_1 크기 줄여보는것도 고민해보기

            FFT_x2(1:n1m,1:n2msub_Isub,1:n3msub) => Buff_1
            call cuda_Poisson_DCT_f_pre(PRHS_Iline_d, FFT_x2, n1m, n2msub_Isub, n3msub)
            nullify(PRHS_Iline_d)


            FFT_xc(1:n1m/2+1,1:n2msub_Isub,1:n3msub) => Buff_c1

            ! plan 재작성 (batch 개수 수정)
                        call cuda_cufft_get_cached_plan(1, 1, n1m, 1, n1m, 1, n1m/2+1, CUFFT_D2Z, n2msub_Isub*n3msub)
            ierr = cufftExecD2Z(plan_fft(1,1), FFT_x2, FFT_xc)
            nullify(FFT_x2)
            

            FFT_x1(1:n1m,1:n2msub_Isub,1:n3msub) => Buff_2
            call cuda_Poisson_DCT_f_post(FFT_xc, FFT_x1, n1m, n2msub_Isub, n3msub)
            nullify(FFT_xc)
            

            ! 먼저 alltoall transpose하고 나서 cuda transpose
            ! 마지막 저장 장소: FFT_x1(1:n1m,1:n2msub_Isub,1:n3msub) => Buff_2
            
            ! I2C (FFT_x1 -> PRHS_d)
            ierr = cudaStreamSynchronize()
            call MPI_Alltoallw(FFT_x1, &
                               countsendI, &
                               countdistI, &
                               ddtype_dble_I_in_C2I, &
                               PRHS_d, &
                               countsendI, &
                               countdistI, &
                               ddtype_dble_C_in_C2I, &
                               p_poi%comm_1d_x1%mpi_comm, &
                               ierr)
            nullify(FFT_x1)

            ! C2J (PRHS_d -> RHSIhat_Jline_d)
            RHSIhat_Jline_d(1:n1msub_Jsub, 1:n2m,1:n3msub) => buffer_dp2_d
            ierr = cudaStreamSynchronize()
            call MPI_Alltoallw(PRHS_d, &
                               countsendJ, &
                               countdistJ, &
                               ddtype_dble_C_in_C2J, &
                               RHSIhat_Jline_d, &
                               countsendJ, &
                               countdistJ, &
                               ddtype_dble_J_in_C2J, &
                               p_poi%comm_1d_x2%mpi_comm, &
                               ierr)
            ! 이제 RHSIhat_Jline_d <- 여기에 들어가있음
            ! Dongyun
            
            ! x방향 DCT
            ! FFT_x2(1:n1msub    ,1:n2msub,1:n3msub) => Buff_1
            ! call cuda_Poisson_DCT_f_pre (PRHS_d, &! Input array
            !                              FFT_x2, &
            !                              n1msub, &
            !                              n2msub, &
            !                              n3msub)

            ! FFT_xc(1:n1msub/2+1,1:n2msub,1:n3msub) => Buff_c1
            ! ierr = cufftExecD2Z(plan_fft(1,1), FFT_x2, FFT_xc)
            ! nullify(FFT_x2)

            ! FFT_x1(1:n1msub    ,1:n2msub,1:n3msub) => Buff_2
            ! call cuda_Poisson_DCT_f_post(FFT_xc, FFT_x1, n1msub, n2msub, n3msub)
            ! nullify(FFT_xc)

        elseif(BCtype(1)=='P'.and.BCtype(2)=='N') then ! Y-R2R
            ! Dongyun
            buffer_dp_size_d = n1m * n2msub_Isub * n3msub
            if (.not. allocated(buffer_dp1_d)) then
                allocate( buffer_dp1_d( buffer_dp_size_d ) )
            elseif (size(buffer_dp1_d) < buffer_dp_size_d) then
                deallocate( buffer_dp1_d )
                allocate( buffer_dp1_d( buffer_dp_size_d ) )
            endif
            if (.not. allocated(buffer_dp2_d)) then
                allocate( buffer_dp2_d( buffer_dp_size_d ) )
            elseif (size(buffer_dp2_d) < buffer_dp_size_d) then
                deallocate( buffer_dp2_d )
                allocate( buffer_dp2_d( buffer_dp_size_d ) )
            endif
            ! C2J (PRHS_d -> PRHS_Jline_d)
            PRHS_Jline_d(1:n1msub_Jsub, 1:n2m, 1:n3msub) => buffer_dp1_d
            ierr = cudaStreamSynchronize()
            call MPI_Alltoallw(PRHS_d, &
                               countsendJ, &
                               countdistJ, &
                               ddtype_dble_C_in_C2J, &
                               PRHS_Jline_d, &
                               countsendJ, &
                               countdistJ, &
                               ddtype_dble_J_in_C2J, &
                               p_poi%comm_1d_x2%mpi_comm, &
                               ierr)

            ! y 방향 DCT
            FFT_y1(1:n2m, 1:n3msub, 1:n1msub_Jsub) => Buff_2
            call cuda_Poisson_transpose_b_real(PRHS_Jline_d, FFT_y1, n2m, n3msub, n1msub_Jsub, real(1.0,rp))
            FFT_y2(1:n2m, 1:n3msub, 1:n1msub_Jsub) => Buff_1
            call cuda_Poisson_DCT_f_pre (FFT_y1, FFT_y2, n2m, n3msub, n1msub_Jsub)
            nullify(FFT_y1)

            FFT_yc(1:n2m/2+1,1:n3msub,1:n1msub_Jsub) => Buff_c1
                        call cuda_cufft_get_cached_plan(1, 2, p_poi%n2m, 1, p_poi%n2m, 1, p_poi%n2m/2+1, CUFFT_D2Z, n1msub_Jsub*n3msub)
            ierr = cufftExecD2Z(plan_fft(1,2), FFT_y2, FFT_yc)
            if (ierr /= 0) then
                write(*,*) 'ERROR: fwd y cufftExecD2Z failed, ierr=', ierr
                stop 9812
            endif
            nullify(FFT_y2)
            
            FFT_y1(1:n2m    ,1:n3msub,1:n1msub_Jsub) => Buff_2
            call cuda_Poisson_DCT_f_post(FFT_yc, FFT_y1, n2m, n3msub, n1msub_Jsub)
            nullify(FFT_yc)
            nullify(PRHS_Jline_d)
            ! FFT_y1에 있음
            ! Dongyun

            ! y 방향 DCT
            ! FFT_y1(1:n2msub    ,1:n3msub,1:n1msub) => Buff_2
            ! ijk -> kij transpose
            ! call cuda_Poisson_transpose_b_real(PRHS_d, FFT_y1, n2msub, n3msub, n1msub, real(1.0,rp))
            
            ! FFT_y2(1:n2msub    ,1:n3msub,1:n1msub) => Buff_1
            ! call cuda_Poisson_DCT_f_pre (FFT_y1, FFT_y2, n2msub, n3msub, n1msub)
            ! nullify(FFT_y1)

            ! FFT_yc(1:n2msub/2+1,1:n3msub,1:n1msub) => Buff_c1
            ! ierr = cufftExecD2Z(plan_fft(1,2), FFT_y2, FFT_yc)
            ! nullify(FFT_y2)

            ! FFT_y1(1:n2msub    ,1:n3msub,1:n1msub) => Buff_2
            ! call cuda_Poisson_DCT_f_post(FFT_yc, FFT_y1, n2msub, n3msub, n1msub)
            ! nullify(FFT_yc)

        elseif(BCtype(1)=='P'.and.BCtype(2)=='P') then ! X-R2C
            ! Dongyun
            ! C2I
            ! x 방향 FFT
#ifdef POISSON_DIRECT_C2I_FFT
            ! Direct path: the C2I local copy or MPI receive/unpack writes the
            ! exact real buffer consumed by cuFFT.  This removes the complete
            ! PRHS_Iline_d -> FFT_x1 device-to-device pass.
            call poisson_profile_start(prof_t0)
            FFT_x1(1:n1m,1:n2msub_Isub,1:n3msub) => Buff_1
            call poisson_profile_stop(prof_t0, prof_times(1))
            call poisson_profile_start(prof_t0)
            if (fft_contiguous_mpi_enabled) then
                call fft_c2i_real_contiguous(PRHS_d, FFT_x1, &
                    n1msub, n2msub, n3msub, n1m, n2msub_Isub, &
                    p_poi%comm_1d_x1%mpi_comm)
            else
                ierr = cudaStreamSynchronize()
                call MPI_alltoallw(PRHS_d,                    &
                                   countsendI,                &
                                   countdistI,                &
                                   ddtype_dble_C_in_C2I,      &
                                   FFT_x1,                    &
                                   countsendI,                &
                                   countdistI,                &
                                   ddtype_dble_I_in_C2I,      &
                                   p_poi%comm_1d_x1%mpi_comm, &
                                   ierr)
            endif
            call poisson_profile_stop(prof_t0, prof_times(2))
#else
            ! Control path: preserve the original staging buffer and dcopy.
            call poisson_profile_start(prof_t0)
            buffer_dp_size_d = n1m * n2msub_Isub * n3msub
            if (.not. allocated(buffer_dp1_d)) then
                allocate(buffer_dp1_d(buffer_dp_size_d))
            elseif (size(buffer_dp1_d) < buffer_dp_size_d) then
                deallocate(buffer_dp1_d)
                allocate(buffer_dp1_d(buffer_dp_size_d))
            endif
            if (.not. allocated(buffer_dp2_d)) then
                allocate(buffer_dp2_d(buffer_dp_size_d))
            elseif (size(buffer_dp2_d) < buffer_dp_size_d) then
                deallocate(buffer_dp2_d)
                allocate(buffer_dp2_d(buffer_dp_size_d))
            endif
            PRHS_Iline_d(1:n1m,1:n2msub_Isub,1:n3msub) => buffer_dp1_d
            call poisson_profile_stop(prof_t0, prof_times(1))
            call poisson_profile_start(prof_t0)
            if (fft_contiguous_mpi_enabled) then
                call fft_c2i_real_contiguous(PRHS_d, PRHS_Iline_d, &
                    n1msub, n2msub, n3msub, n1m, n2msub_Isub, &
                    p_poi%comm_1d_x1%mpi_comm)
            else
                ierr = cudaStreamSynchronize()
                call MPI_alltoallw(PRHS_d,                    &
                                   countsendI,                &
                                   countdistI,                &
                                   ddtype_dble_C_in_C2I,      &
                                   PRHS_Iline_d,              &
                                   countsendI,                &
                                   countdistI,                &
                                   ddtype_dble_I_in_C2I,      &
                                   p_poi%comm_1d_x1%mpi_comm, &
                                   ierr)
            endif
            call poisson_profile_stop(prof_t0, prof_times(2))
            call poisson_profile_start(prof_t0)
            FFT_x1(1:n1m,1:n2msub_Isub,1:n3msub) => Buff_1
            call dcopy(n1m*n2msub_Isub*n3msub, &
                       PRHS_Iline_d, 1, FFT_x1, 1)
            nullify(PRHS_Iline_d)
            call poisson_profile_stop(prof_t0, prof_times(3))
#endif

            call poisson_coarse_profile_mark(1)

            call poisson_profile_start(prof_t0)
            FFT_xc(1:n1m/2+1,1:n2msub_Isub,1:n3msub) => Buff_c1
            ! plan 재작성 (batch 개수 수정)
                        call cuda_cufft_get_cached_plan(1, 1, n1m, 1, n1m, 1, n1m/2+1, CUFFT_D2Z, n2msub_Isub*n3msub)
            ierr = cufftExecD2Z(plan_fft(1,1), FFT_x1, FFT_xc)
            nullify(FFT_x1)
            call poisson_profile_stop(prof_t0, prof_times(4))
            call poisson_coarse_profile_mark(2)
            ! Dongyun

            ! FFT_x1(1:n1msub    ,1:n2msub,1:n3msub) => Buff_1
            ! call dcopy(n1msub*n2msub*n3msub, PRHS_d, 1, FFT_x1, 1)  

            ! FFT_xc(1:n1msub/2+1,1:n2msub,1:n3msub) => Buff_c1
            ! ierr = cufftExecD2Z(plan_fft(1,1), FFT_x1, FFT_xc)
            ! nullify(FFT_x1)
        endif

        ! call nvtxEndRange
        ! call nvtxStartRange("Poisson-F_y")

        if    (BCtype(1)=='N'.and.BCtype(2)=='N') then ! Y-R2R
            ! Dongyun
            ! 이제 RHSIhat_Jline_d <- 여기에 들어가있음
            ! RHSIhat_Jline_d -> FFT_y1
            FFT_y1(1:n2m,1:n3msub, 1:n1msub_Jsub) => Buff_1
            call cuda_Poisson_transpose_b(RHSIhat_Jline_d, FFT_y1, n2m, n3msub, n1msub_Jsub, real(1.0,rp))


            ! Buff_2도 줄이는거 고민좀
            ! FFT_y1 -> FFT_y2
            FFT_y2(1:n2m,1:n3msub, 1:n1msub_Jsub) => Buff_2
            call cuda_Poisson_DCT_f_pre (FFT_y1, FFT_y2, n2m, n3msub, n1msub_Jsub)
            nullify(FFT_y1)


            FFT_yc(1:n2m/2+1,1:n3msub,1:n1msub_Jsub) => Buff_c1
            ! Plan 재작성
                        call cuda_cufft_get_cached_plan(1, 2, n2m, 1, n2m, 1, n2m/2+1, CUFFT_D2Z, n1msub_Jsub * n3msub)	
            ! yfft (FFT_y2 -> FFT_yc)
            ierr = cufftExecD2Z(plan_fft(1,2), FFT_y2, FFT_yc)
            nullify(FFT_y2)


            FFT_y1(1:n2m,1:n3msub,1:n1msub_Jsub) => Buff_1
            ! FFT_yc -> FFT_y1
            call cuda_Poisson_DCT_f_post(FFT_yc, FFT_y1, n2m, n3msub, n1msub_Jsub)
            nullify(FFT_yc)


            FFT_x1(1:n1msub_Jsub,1:n2m,1:n3msub) => Buff_2
            ! FFT_y1 -> FFT_x1
            call cuda_Poisson_transpose_f(FFT_y1, FFT_x1, n1msub_Jsub, n2m, n3msub, real(1.0,rp))
            nullify(FFT_y1)
            ! Dongyun
            
            ! FFT_y1(1:n2msub    ,1:n3msub,1:n1msub) => Buff_1
            ! call cuda_Poisson_transpose_b(FFT_x1, FFT_y1, n2msub, n3msub, n1msub, real(1.0,rp))
            ! nullify(FFT_x1)

            ! FFT_y2(1:n2msub    ,1:n3msub,1:n1msub) => Buff_2
            ! call cuda_Poisson_DCT_f_pre (FFT_y1, FFT_y2, n2msub, n3msub, n1msub)
            ! nullify(FFT_y1)

            ! FFT_yc(1:n2msub/2+1,1:n3msub,1:n1msub) => Buff_c1
            ! ierr = cufftExecD2Z(plan_fft(1,2), FFT_y2, FFT_yc)
            ! nullify(FFT_y2)

            ! FFT_y1(1:n2msub    ,1:n3msub,1:n1msub) => Buff_1
            ! call cuda_Poisson_DCT_f_post(FFT_yc, FFT_y1, n2msub, n3msub, n1msub)
            ! nullify(FFT_yc)

            ! FFT_x1(1:n1msub    ,1:n2msub,1:n3msub) => Buff_2
            ! call cuda_Poisson_transpose_f(FFT_y1, FFT_x1, n1msub, n2msub, n3msub, real(1.0,rp))
            ! nullify(FFT_y1)
            ! Last : FFT_x1 

        elseif(BCtype(1)=='N'.and.BCtype(2)=='P') then ! Y-R2C
            ! Dongyun
            ! RHSIhat_Jline_d -> FFT_y1
            FFT_y1(1:n2m,1:n3msub, 1:n1msub_Jsub) => Buff_1
            call cuda_Poisson_transpose_b(RHSIhat_Jline_d, FFT_y1, n2m, n3msub, n1msub_Jsub, real(1.0,rp))

            FFT_yc(1:n2m/2+1,1:n3msub,1:n1msub_Jsub) => Buff_c1
                        call cuda_cufft_get_cached_plan(1, 2, n2m, 1, n2m, 1, n2m/2+1, CUFFT_D2Z, n1msub_Jsub * n3msub)	
            ! v19 NP fix: this cufftExecD2Z was accidentally missing.
            ! Without it, FFT_yc is uninitialized/stale before the transpose to FFT_xc.
            ierr = cufftExecD2Z(plan_fft(1,2), FFT_y1, FFT_yc)
            if (ierr /= 0) then
                write(*,*) 'ERROR: fwd y cufftExecD2Z failed, ierr=', ierr
                stop 9821
            endif
            nullify(FFT_y1)

            FFT_xc(1:n1msub_Jsub,1:n2m/2+1,1:n3msub) => Buff_c2
            call cuda_Poisson_transpose_f(FFT_yc, FFT_xc, n1msub_Jsub, n2m/2+1, n3msub, real(1.0,rp))
            nullify(FFT_yc)
            ! Last : FFT_xc
            ! Dongyun

            ! I2C (FFT_1 -> PRHS_d)
            ! FFT_y1(1:n2msub    ,1:n3msub,1:n1msub) => Buff_1
            ! call cuda_Poisson_transpose_b(FFT_x1, FFT_y1, n2msub, n3msub, n1msub, real(1.0,rp))
            ! nullify(FFT_x1)

            ! FFT_yc(1:n2msub/2+1,1:n3msub,1:n1msub) => Buff_c1
            ! ierr = cufftExecD2Z(plan_fft(1,2), FFT_y1, FFT_yc)
            ! nullify(FFT_y1)

            ! FFT_xc(1:n1msub,1:n2msub/2+1,1:n3msub) => Buff_c2
            ! call cuda_Poisson_transpose_f(FFT_yc, FFT_xc, n1msub, n2msub/2+1, n3msub, real(1.0,rp))
            ! nullify(FFT_yc)

        elseif(BCtype(1)=='P'.and.BCtype(2)=='N') then ! X-R2C
            ! Dongyun
            ! FFT_y1에 있음
            ! 먼저 transpose하고 ijk로 정렬한다음에 alltoall을 해야함!!
            FFT_x1(1:n1msub_Jsub    ,1:n2m,1:n3msub) => Buff_1
            call cuda_Poisson_transpose_f(FFT_y1, FFT_x1, n1msub_Jsub, n2m, n3msub, real(1.0,rp))
            nullify(FFT_y1)

            PRHS_Iline_d(1:n1m,1:n2msub_Isub,1:n3msub) => buffer_dp1_d
            ! J2C (FFT_x1 -> PRHS_d)
            ierr = cudaStreamSynchronize()
            call MPI_Alltoallw(FFT_x1, &
                               countsendJ, &
                               countdistJ, &
                               ddtype_dble_J_in_C2J, &
                               PRHS_d, &
                               countsendJ, &
                               countdistJ, &
                               ddtype_dble_C_in_C2J, &
                               p_poi%comm_1d_x2%mpi_comm, &
                               ierr)
            ! C2I (PRHS_d -> PRHS_Iline_d)
            ierr = cudaStreamSynchronize()
            call MPI_Alltoallw(PRHS_d,                    &
                               countsendI,                &
                               countdistI,                &
                               ddtype_dble_C_in_C2I,      &
                               PRHS_Iline_d,              &
                               countsendI,                &
                               countdistI,                &
                               ddtype_dble_I_in_C2I,      &
                               p_poi%comm_1d_x1%mpi_comm, &
                               ierr)
            FFT_xc(1:n1m/2+1,1:n2msub_Isub,1:n3msub) => Buff_c1
            ! plan다시만들어
                        call cuda_cufft_get_cached_plan(1, 1, n1m, 1, n1m, 1, int(n1m/2) + 1, CUFFT_D2Z, n2msub_Isub*n3msub)
            ierr = cufftExecD2Z(plan_fft(1,1), PRHS_Iline_d, FFT_xc)
            if (ierr /= 0) then
                write(*,*) 'ERROR: fwd x cufftExecD2Z failed, ierr=', ierr
                stop 9813
            endif
            nullify(FFT_x1)
            nullify(PRHS_Iline_d)
            ! Dongyun

            ! FFT_x1(1:n1msub    ,1:n2msub,1:n3msub) => Buff_1
            
            ! FFT_xc(1:n1msub/2+1,1:n2msub,1:n3msub) => Buff_c1
            ! ierr = cufftExecD2Z(plan_fft(1,1), FFT_x1, FFT_xc)
            ! nullify(FFT_x1)
            ! Last : FFT_xc
            
        elseif(BCtype(1)=='P'.and.BCtype(2)=='P') then ! Y-C2C
            ! Dongyun
#ifdef POISSON_USE_CUDECOMP
            ! Direct I-pencil -> J/Y-pencil redistribution.  cuDecomp consumes
            ! the X-contiguous FFT_xc layout and produces the Y-contiguous
            ! FFT_yc layout required by the y-direction cuFFT.  This replaces
            ! I2C + C2J and the old local J-to-Y transpose in one operation.
            FFT_yc(1:n2m,1:n3msub,1:h1psub_Jsub) => Buff_c2
            call poisson_profile_start(prof_t0)
            call fft_cudecomp_x_to_y(FFT_xc, FFT_yc)
            nullify(FFT_xc)
            call poisson_profile_stop(prof_t0, prof_times(6))
#else
            ! Legacy complex I2C -> C2J route.
            call poisson_profile_start(prof_t0)
            buffer_cd_size_d = h1psub * n2msub * n3msub
            if (.not. allocated(buffer_cd1_d)) then
                allocate( buffer_cd1_d( buffer_cd_size_d ) )
            elseif (size(buffer_cd1_d) < buffer_cd_size_d) then
                deallocate( buffer_cd1_d )
                allocate( buffer_cd1_d( buffer_cd_size_d ) )
            endif
            PRHS_cplx_d(1:h1psub, 1:n2msub, 1:n3msub) => buffer_cd1_d
            call poisson_profile_stop(prof_t0, prof_times(5))

            call poisson_profile_start(prof_t0)
            if (fft_contiguous_mpi_enabled) then
                call fft_i2c_complex_contiguous(FFT_xc, PRHS_cplx_d, &
                    n1m/2+1, n2msub_Isub, n3msub, &
                    h1psub, n2msub, p_poi%comm_1d_x1%mpi_comm)
            else
                ierr = cudaStreamSynchronize()
                call MPI_Alltoallw(FFT_xc, &
                                   countsendI, &
                                   countdistI, &
                                   ddtype_cplx_I_in_C2I, &
                                   PRHS_cplx_d, &
                                   countsendI, &
                                   countdistI, &
                                   ddtype_cplx_C_in_C2I, &
                                   p_poi%comm_1d_x1%mpi_comm, &
                                   ierr)
            endif
            nullify(FFT_xc)
            call poisson_profile_stop(prof_t0, prof_times(6))
            FFT_xc(1:h1psub_Jsub,1:n2m,1:n3msub) => Buff_c1

            call poisson_profile_start(prof_t0)
            if (fft_contiguous_mpi_enabled) then
                call fft_c2j_complex_contiguous(PRHS_cplx_d, FFT_xc, &
                    h1psub, n2msub, n3msub, &
                    h1psub_Jsub, n2m, p_poi%comm_1d_x2%mpi_comm)
            else
                ierr = cudaStreamSynchronize()
                call MPI_Alltoallw(PRHS_cplx_d, &
                                   countsendJ, &
                                   countdistJ, &
                                   ddtype_cplx_C_in_C2J, &
                                   FFT_xc, &
                                   countsendJ, &
                                   countdistJ, &
                                   ddtype_cplx_J_in_C2J, &
                                   p_poi%comm_1d_x2%mpi_comm, &
                                   ierr)
            endif
            nullify(PRHS_cplx_d)
            call poisson_profile_stop(prof_t0, prof_times(7))
            call poisson_profile_start(prof_t0)
            FFT_yc(1:n2m,1:n3msub,1:h1psub_Jsub) => Buff_c2
            call cuda_Poisson_transpose_b(FFT_xc, FFT_yc, n2m, n3msub, h1psub_Jsub, real(1.0,rp))
            nullify(FFT_xc)
            call poisson_profile_stop(prof_t0, prof_times(8))
#endif
            ! plan 다시 만들기
            call poisson_profile_start(prof_t0)
                        call cuda_cufft_get_cached_plan(1, 2, n2m, 1, n2m, 1, n2m, CUFFT_Z2Z, h1psub_Jsub*n3msub)	
            ierr = cufftExecZ2Z(plan_fft(1,2), FFT_yc, FFT_yc, CUFFT_FORWARD)
            call poisson_profile_stop(prof_t0, prof_times(9))

            call poisson_profile_start(prof_t0)
            FFT_xc(1:h1psub_Jsub,1:n2m,1:n3msub) => Buff_c1
            call cuda_Poisson_transpose_f(FFT_yc, FFT_xc, h1psub_Jsub, n2m, n3msub, real(1.0,rp))
            nullify(FFT_yc)
            call poisson_profile_stop(prof_t0, prof_times(10))
            ! Dongyun

            ! FFT_yc(1:n2msub,1:n3msub,1:n1msub/2+1) => Buff_c2

            ! call cuda_Poisson_transpose_b(FFT_xc, FFT_yc, n2msub, n3msub, n1msub/2+1, real(1.0,rp))
            ! nullify(FFT_xc)
            ! ierr = cufftExecZ2Z(plan_fft(1,2), FFT_yc, FFT_yc, CUFFT_FORWARD)

            ! FFT_xc(1:n1msub/2+1,1:n2msub,1:n3msub) => Buff_c1
            ! call cuda_Poisson_transpose_f(FFT_yc, FFT_xc, n1msub/2+1, n2msub, n3msub, real(1.0,rp))
            ! nullify(FFT_yc)
            ! Last : FFT_xc

        endif

        call poisson_coarse_profile_mark(3)

        ! call nvtxEndRange

        ! call nvtxStartRange("Poisson-TDMA-z")
        if(BCtype(1)=='N'.and.BCtype(2)=='N') then
            ! call cuda_Poisson_TDMA_z(FFT_x1, dx3_d, dmx3_d)
            call cuda_Poisson_TDMA_z_real(FFT_x1, &
                                          n1msub_Jsub, n2m, n3msub,&
                                          dx3_d, dmx3_d,&
                                          iend, ista, jend, jsta)
        else
            call cuda_Poisson_TDMA_z_complex(FFT_xc, &
                                            size(FFT_xc,1), size(FFT_xc,2), size(FFT_xc,3),&
                                            dx3_d, dmx3_d,&
                                            iend, ista, jend, jsta, h1psub_Jsub_ista, n2msub_Isub_jsta, n1msub_Jsub_ista)
        endif
        call poisson_coarse_profile_mark(4)
        ! call nvtxEndRange

        ! Backward DCT
        ! call nvtxStartRange("Poisson-B_y")

        if(BCtype(1)=='N'.and.BCtype(2)=='N') then
            ! Dongyun
            FFT_y1(1:n2m,1:n3msub,1:n1msub_Jsub) => Buff_1
            call cuda_Poisson_transpose_b(FFT_x1, FFT_y1, n2m, n3msub, n1msub_Jsub, real(1.0,rp))
            nullify(FFT_x1)

            FFT_yc(1:(n2m)/2+1,1:n3msub,1:n1msub_Jsub) => Buff_c1
            FFT_y3(1:n2       ,1:n3msub,1:n1msub_Jsub) => Buff_2
            call cuda_Poisson_DCT_b_pre(FFT_y1, FFT_y3, FFT_yc, n2m, n3msub, n1msub_Jsub)
            nullify(FFT_y3, FFT_y1)

            FFT_y1(1:n2m,1:n3msub,1:n1msub_Jsub) => Buff_1
            ! Plan 다시만들기
                        call cuda_cufft_get_cached_plan(2, 2, n2m, 1, n2m/2+1, 1, n2m, CUFFT_Z2D, n3msub*n1msub_Jsub)	
            ierr = cufftExecZ2D(plan_fft(2,2), FFT_yc, FFT_y1)
            nullify(FFT_yc)

            FFT_y2(1:n2m,1:n3msub,1:n1msub_Jsub) => Buff_2
            call cuda_Poisson_DCT_b_post(FFT_y1, FFT_y2, n2m, n3msub, n1msub_Jsub)
            nullify(FFT_y1)


            FFT_x1(1:n1msub_Jsub,1:n2m,1:n3msub) => Buff_1
            call cuda_Poisson_transpose_f(FFT_y2, FFT_x1, n1msub_Jsub, n2m, n3msub, real(1,rp)/real(2*n2m,rp))
            nullify(FFT_y2)
            ! Dongyun

            ! FFT_y1(1:n2msub,1:n3msub,1:n1msub) => Buff_1
            ! call cuda_Poisson_transpose_b(FFT_x1, FFT_y1, n2msub, n3msub, n1msub, real(1.0,rp))
            ! nullify(FFT_x1)

            ! FFT_yc(1:(n2msub)/2+1,1:n3msub,1:n1msub) => Buff_c1
            ! FFT_y3(1:n2sub       ,1:n3msub,1:n1msub) => Buff_2
            ! call cuda_Poisson_DCT_b_pre(FFT_y1, FFT_y3, FFT_yc, n2msub, n3msub, n1msub)
            ! nullify(FFT_y3, FFT_y1)

            ! FFT_y1(1:n2msub,1:n3msub,1:n1msub) => Buff_1
            ! ierr = cufftExecZ2D(plan_fft(2,2), FFT_yc, FFT_y1)
            ! nullify(FFT_yc)

            ! FFT_y2(1:n2msub,1:n3msub,1:n1msub) => Buff_2
            ! call cuda_Poisson_DCT_b_post(FFT_y1, FFT_y2, n2msub, n3msub, n1msub)
            ! nullify(FFT_y1)

            ! FFT_x1(1:n1msub,1:n2msub,1:n3msub) => Buff_1
            ! call cuda_Poisson_transpose_f(FFT_y2, FFT_x1, n1msub, n2msub, n3msub, real(1,rp)/real(2*n2m,rp))
            ! nullify(FFT_y2)

        elseif(BCtype(1)=='N'.and.BCtype(2)=='P') then ! Y-C2R
            ! Dongyun
            FFT_yc(1:n2m/2+1,1:n3msub,1:n1msub_Jsub) => Buff_c1
            call cuda_Poisson_transpose_b(FFT_xc, FFT_yc, n2m/2+1, n3msub, n1msub_Jsub, real(1.0,rp))
            nullify(FFT_xc)

            FFT_y2(1:n2m,1:n3msub,1:n1msub_Jsub) => Buff_2
                        call cuda_cufft_get_cached_plan(2, 2, n2m, 1, n2m/2+1, 1, n2m, CUFFT_Z2D, n3msub*n1msub_Jsub)	
            ierr = cufftExecZ2D(plan_fft(2,2), FFT_yc, FFT_y2)
            nullify(FFT_yc)

            FFT_x1(1:n1msub_Jsub,1:n2m,1:n3msub) => Buff_1
            call cuda_Poisson_transpose_f(FFT_y2, FFT_x1, n1msub_Jsub, n2m, n3msub, real(1,rp)/real(n2m,rp))
            nullify(FFT_y2)
            ! Dongyun
            
            ! FFT_yc(1:n2msub/2+1,1:n3msub,1:n1msub) => Buff_c1
            ! call cuda_Poisson_transpose_b(FFT_xc, FFT_yc, n2msub/2+1, n3msub, n1msub, real(1.0,rp))
            ! nullify(FFT_xc)

            ! FFT_y2(1:n2msub,1:n3msub,1:n1msub) => Buff_2
            ! ierr = cufftExecZ2D(plan_fft(2,2), FFT_yc, FFT_y2)
            ! nullify(FFT_yc)

            ! FFT_x1(1:n1msub,1:n2msub,1:n3msub) => Buff_1
            ! call cuda_Poisson_transpose_f(FFT_y2, FFT_x1, n1msub, n2msub, n3msub, real(1,rp)/real(n2m,rp))
            ! nullify(FFT_y2)

        elseif(BCtype(1)=='P'.and.BCtype(2)=='N') then ! X-C2R
            ! Dongyun
            FFT_x1(1:n1m,1:n2msub_Isub,1:n3msub) => Buff_1
                        call cuda_cufft_get_cached_plan(2, 1, n1m, 1, int(n1m/2)+1, 1, n1m, CUFFT_Z2D, n2msub_Isub*n3msub)	
            ierr = cufftExecZ2D(plan_fft(2,1), FFT_xc, FFT_x1)
            if (ierr /= 0) then
                write(*,*) 'ERROR: back x cufftExecZ2D failed, ierr=', ierr
                stop 9814
            endif
            nullify(FFT_xc)

            FFT_x2(1:n1msub_Jsub, 1:n2m, 1:n3msub) => Buff_2
            ! 여기서 다시 I2C
            ierr = cudaStreamSynchronize()
            call MPI_Alltoallw(FFT_x1,                    &
                               countsendI,                &
                               countdistI,                &
                               ddtype_dble_I_in_C2I,      &
                               PRHS_d,              &
                               countsendI,                &
                               countdistI,                &
                               ddtype_dble_C_in_C2I,      &
                               p_poi%comm_1d_x1%mpi_comm, &
                               ierr)
            ! C2J
            ierr = cudaStreamSynchronize()
            call MPI_Alltoallw(PRHS_d, &
                               countsendJ, &
                               countdistJ, &
                               ddtype_dble_C_in_C2J, &
                               FFT_x2, &
                               countsendJ, &
                               countdistJ, &
                               ddtype_dble_J_in_C2J, &
                               p_poi%comm_1d_x2%mpi_comm, &
                               ierr)
            nullify(FFT_x1)
            FFT_y1(1:n2m,1:n3msub,1:n1msub_Jsub) => Buff_1
            call cuda_Poisson_transpose_b(FFT_x2, FFT_y1, n2m, n3msub, n1msub_Jsub, real(1,rp)/real(n1m,rp))
            ! Dongyun

            ! FFT_x1(1:n1msub,1:n2msub,1:n3msub) => Buff_2
            ! ierr = cufftExecZ2D(plan_fft(2,1), FFT_xc, FFT_x1)
            ! nullify(FFT_xc)

            ! FFT_y1(1:n2msub,1:n3msub,1:n1msub) => Buff_1
            ! call cuda_Poisson_transpose_b(FFT_x1, FFT_y1, n2msub, n3msub, n1msub, real(1,rp)/real(n1m,rp))
            ! nullify(FFT_x1)

        elseif(BCtype(1)=='P'.and.BCtype(2)=='P') then ! Y-C2C
            ! Dongyun
            call poisson_profile_start(prof_t0)
            FFT_yc(1:n2m,1:n3msub,1:h1psub_Jsub) => Buff_c2
            call cuda_Poisson_transpose_b(FFT_xc, FFT_yc, n2m, n3msub, h1psub_Jsub, real(1.0,rp))
            nullify(FFT_xc)
            call poisson_profile_stop(prof_t0, prof_times(11))

            call poisson_profile_start(prof_t0)
                        call cuda_cufft_get_cached_plan(2, 2, n2m, 1, n2m, 1, n2m, CUFFT_Z2Z, n3msub*h1psub_Jsub)
            ierr = cufftExecZ2Z(plan_fft(2,2), FFT_yc, FFT_yc, CUFFT_INVERSE)
            call poisson_profile_stop(prof_t0, prof_times(12))

            call poisson_profile_start(prof_t0)
#ifdef POISSON_USE_CUDECOMP
            ! Direct Y/J-pencil -> X/I-pencil redistribution.  The 1/n2m
            ! inverse-y normalization is fused into the final real dscal
            ! after the inverse-x transform.
            FFT_xc(1:n1m/2+1,1:n2msub_Isub,1:n3msub) => Buff_c1
            call fft_cudecomp_y_to_x(FFT_yc, FFT_xc)
            nullify(FFT_yc)
#else
            FFT_xc(1:h1psub_Jsub,1:n2m,1:n3msub) => Buff_c1
            call cuda_Poisson_transpose_f(FFT_yc, FFT_xc, h1psub_Jsub, n2m, n3msub, real(1,rp)/real(n2m,rp))
            nullify(FFT_yc)
#endif
            call poisson_profile_stop(prof_t0, prof_times(13))
            ! Dongyun

            ! FFT_yc(1:n2msub,1:n3msub,1:n1msub/2+1) => Buff_c2
            ! call cuda_Poisson_transpose_b(FFT_xc, FFT_yc, n2msub, n3msub, n1msub/2+1, real(1.0,rp))
            ! nullify(FFT_xc)
            ! ierr = cufftExecZ2Z(plan_fft(2,2), FFT_yc, FFT_yc, CUFFT_INVERSE)

            ! FFT_xc(1:n1msub/2+1,1:n2msub,1:n3msub) => Buff_c1
            ! call cuda_Poisson_transpose_f(FFT_yc, FFT_xc, n1msub/2+1, n2msub, n3msub, real(1,rp)/real(n2m,rp))
            ! nullify(FFT_yc)
        endif

        call poisson_coarse_profile_mark(5)

        ! call nvtxEndRange
        ! call nvtxStartRange("Poisson-B_x")

        if((BCtype(1)=='N'.and.BCtype(2)=='N') .or. (BCtype(1)=='N'.and.BCtype(2)=='P')) then ! X-R2R
            ! Dongyun
            ! NP : FFT_x1(1:n1msub_Jsub,1:n2m,1:n3msub)
            ! NN : FFT_x1(1:n1msub_Jsub,1:n2m,1:n3msub)
            ! J2C (FFT_x1 -> PRHS_d)
            ierr = cudaStreamSynchronize()
            call MPI_Alltoallw(FFT_x1, &
                               countsendJ, &
                               countdistJ, &
                               ddtype_dble_J_in_C2J, &
                               PRHS_d, &
                               countsendJ, &
                               countdistJ, &
                               ddtype_dble_C_in_C2J, &
                               p_poi%comm_1d_x2%mpi_comm, &
                               ierr)
            PRHS_Iline_d(1:n1m,1:n2msub_Isub,1:n3msub) => buffer_dp1_d
            ! C2I (PRHS_d -> PRHS_Iline_d)
            ierr = cudaStreamSynchronize()
            call MPI_Alltoallw(PRHS_d,                    &
                               countsendI,                &
                               countdistI,                &
                               ddtype_dble_C_in_C2I,      &
                               PRHS_Iline_d,              &
                               countsendI,                &
                               countdistI,                &
                               ddtype_dble_I_in_C2I,      &
                               p_poi%comm_1d_x1%mpi_comm, &
                               ierr)
            FFT_xc(1:n1m/2+1,1:n2msub_Isub,1:n3msub) => Buff_c1
            FFT_x3(1:n1     ,1:n2msub_Isub,1:n3msub) => Buff_2
            

            call cuda_Poisson_DCT_b_pre(PRHS_Iline_d, FFT_x3, FFT_xc, n1m, n2msub_Isub, n3msub)
            nullify(FFT_x1, FFT_x3)

            FFT_x1(1:n1m,1:n2msub_Isub,1:n3msub) => Buff_1
                        call cuda_cufft_get_cached_plan(2, 1, n1m, 1, int(n1m/2)+1, 1, n1m, CUFFT_Z2D, n2msub_Isub*n3msub)	
            ierr = cufftExecZ2D(plan_fft(2,1), FFT_xc, FFT_x1)
            nullify(FFT_xc)
            
            FFT_x2(1:n1m,1:n2msub_Isub,1:n3msub) => Buff_2
            call cuda_Poisson_DCT_b_post(FFT_x1, FFT_x2, n1m, n2msub_Isub, n3msub)
            nullify(FFT_x1)
            ! 여기서 다시 I2C
            ierr = cudaStreamSynchronize()
            call MPI_Alltoallw(FFT_x2,                    &
                               countsendI,                &
                               countdistI,                &
                               ddtype_dble_I_in_C2I,      &
                               PRHS_d,              &
                               countsendI,                &
                               countdistI,                &
                               ddtype_dble_C_in_C2I,      &
                               p_poi%comm_1d_x1%mpi_comm, &
                               ierr)
            nullify(FFT_x2)
            call dscal(n1msub*n2msub*n3msub, real(1,rp)/real(2*n1m,rp), PRHS_d, 1)
            ! Dongyun
            
            ! FFT_xc(1:n1msub/2+1,1:n2msub,1:n3msub) => Buff_c1
            ! FFT_x3(1:n1sub     ,1:n2msub,1:n3msub) => Buff_2
            ! call cuda_Poisson_DCT_b_pre(FFT_x1, FFT_x3, FFT_xc, n1msub, n2msub, n3msub)
            ! nullify(FFT_x1, FFT_x3)

            ! FFT_x1(1:n1msub,1:n2msub,1:n3msub) => Buff_1
            ! ierr = cufftExecZ2D(plan_fft(2,1), FFT_xc, FFT_x1)
            ! nullify(FFT_xc)

            ! FFT_x2(1:n1msub,1:n2msub,1:n3msub) => Buff_2
            ! call cuda_Poisson_DCT_b_post(FFT_x1, FFT_x2, n1msub, n2msub, n3msub)
            ! nullify(FFT_x1)

            ! call dcopy(n1msub*n2msub*n3msub, FFT_x2, 1, PRHS_d, 1)  
            ! nullify(FFT_x2)

            ! call dscal(n1msub*n2msub*n3msub, real(1,rp)/real(2*n1m,rp), PRHS_d, 1)

        elseif(BCtype(1)=='P'.and.BCtype(2)=='N') then ! Y-R2R
            ! Dongyun
            ! FFT_y1(1:n2m,1:n3msub,1:n1msub_Jsub)
            FFT_yc(1:n2m/2+1,1:n3msub,1:n1msub_Jsub) => Buff_c1
            FFT_y3(1:n2     ,1:n3msub,1:n1msub_Jsub) => Buff_2
            call cuda_Poisson_DCT_b_pre(FFT_y1, FFT_y3, FFT_yc, n2m, n3msub, n1msub_Jsub)
            nullify(FFT_y1, FFT_y3)
            
            FFT_y1(1:n2m,1:n3msub,1:n1msub_Jsub) => Buff_1
            call cuda_cufft_get_cached_plan(2, 2, n2m, 1, n2m/2+1, 1, n2m, CUFFT_Z2D, n3msub*n1msub_Jsub)
            ierr = cufftExecZ2D(plan_fft(2,2), FFT_yc, FFT_y1)
            if (ierr /= 0) then
                write(*,*) 'ERROR: back y cufftExecZ2D failed, ierr=', ierr
                stop 9815
            endif
            nullify(FFT_yc)

            FFT_y2(1:n2m,1:n3msub,1:n1msub_Jsub) => Buff_2
            call cuda_Poisson_DCT_b_post(FFT_y1, FFT_y2, n2m, n3msub, n1msub_Jsub)
            nullify(FFT_y1)

            FFT_x1(1:n1msub_Jsub,1:n2m,1:n3msub) => Buff_1
            call cuda_Poisson_transpose_f(FFT_y2, FFT_x1, n1msub_Jsub, n2m, n3msub, real(1.0,rp))
            nullify(FFT_y2)

            ierr = cudaStreamSynchronize()
            call MPI_Alltoallw(FFT_x1, &
                               countsendJ, &
                               countdistJ, &
                               ddtype_dble_J_in_C2J, &
                               PRHS_d, &
                               countsendJ, &
                               countdistJ, &
                               ddtype_dble_C_in_C2J, &
                               p_poi%comm_1d_x2%mpi_comm, &
                               ierr)
            nullify(FFT_x1)
            call dscal(n1msub*n2msub*n3msub, real(1,rp)/real(2*n2m,rp), PRHS_d, 1)
            ! Dongyun

            ! FFT_yc(1:n2msub/2+1,1:n3msub,1:n1msub) => Buff_c1
            ! FFT_y3(1:n2sub     ,1:n3msub,1:n1msub) => Buff_2
            ! call cuda_Poisson_DCT_b_pre(FFT_y1, FFT_y3, FFT_yc, n2msub, n3msub, n1msub)
            ! nullify(FFT_y1, FFT_y3)

            ! FFT_y1(1:n2msub,1:n3msub,1:n1msub) => Buff_1
            ! ierr = cufftExecZ2D(plan_fft(2,2), FFT_yc, FFT_y1)
            ! nullify(FFT_yc)

            ! FFT_y2(1:n2msub,1:n3msub,1:n1msub) => Buff_2
            ! call cuda_Poisson_DCT_b_post(FFT_y1, FFT_y2, n2msub, n3msub, n1msub)
            ! nullify(FFT_y1)

            ! FFT_x1(1:n1msub,1:n2msub,1:n3msub) => Buff_1
            ! call cuda_Poisson_transpose_f(FFT_y2, FFT_x1, n1msub, n2msub, n3msub, real(1.0,rp))
            ! nullify(FFT_y2)

            ! call dcopy(n1msub*n2msub*n3msub, FFT_x1, 1, PRHS_d, 1)
            ! nullify(FFT_x1)

            ! call dscal(n1msub*n2msub*n3msub, real(1,rp)/real(2*n2m,rp), PRHS_d, 1)

        elseif(BCtype(1)=='P'.and.BCtype(2)=='P') then ! X-C2R
            ! Dongyun
#ifndef POISSON_USE_CUDECOMP
            ! J2C, C2I
            ! FFT_xc(1:h1psub_Jsub,1:n2m,1:n3msub) 
            PRHS_cplx_d(1:h1psub, 1:n2msub, 1:n3msub) => buffer_cd1_d
            call poisson_profile_start(prof_t0)
            if (fft_contiguous_mpi_enabled) then
                call fft_j2c_complex_contiguous(FFT_xc, PRHS_cplx_d, &
                    h1psub_Jsub, n2m, n3msub, &
                    h1psub, n2msub, p_poi%comm_1d_x2%mpi_comm)
            else
                ierr = cudaStreamSynchronize()
                call MPI_Alltoallw(FFT_xc, &
                                   countsendJ, &
                                   countdistJ, &
                                   ddtype_cplx_J_in_C2J, &
                                   PRHS_cplx_d, &
                                   countsendJ, &
                                   countdistJ, &
                                   ddtype_cplx_C_in_C2J, &
                                   p_poi%comm_1d_x2%mpi_comm, &
                                   ierr)
            endif
            nullify(FFT_xc)
            call poisson_profile_stop(prof_t0, prof_times(14))
            FFT_xc(1:n1m/2+1,1:n2msub_Isub,1:n3msub) => Buff_c1
            call poisson_profile_start(prof_t0)
            if (fft_contiguous_mpi_enabled) then
                call fft_c2i_complex_contiguous(PRHS_cplx_d, FFT_xc, &
                    h1psub, n2msub, n3msub, &
                    n1m/2+1, n2msub_Isub, p_poi%comm_1d_x1%mpi_comm)
            else
                ierr = cudaStreamSynchronize()
                call MPI_Alltoallw(PRHS_cplx_d, &
                                   countsendI, &
                                   countdistI, &
                                   ddtype_cplx_C_in_C2I, &
                                   FFT_xc, &
                                   countsendI, &
                                   countdistI, &
                                   ddtype_cplx_I_in_C2I, &
                                   p_poi%comm_1d_x1%mpi_comm, &
                                   ierr)
            endif
            nullify(PRHS_cplx_d)
            call poisson_profile_stop(prof_t0, prof_times(15))
#endif
            call poisson_profile_start(prof_t0)
            FFT_x1(1:n1m,1:n2msub_Isub,1:n3msub) => Buff_1
                        call cuda_cufft_get_cached_plan(2, 1, n1m, 1, int(n1m/2)+1, 1, n1m, CUFFT_Z2D, n2msub_Isub*n3msub)	
            ierr = cufftExecZ2D(plan_fft(2,1), FFT_xc, FFT_x1)
            nullify(FFT_xc)
            call poisson_profile_stop(prof_t0, prof_times(16))

            ! 다시 I2C (double)
            call poisson_profile_start(prof_t0)
            if (fft_contiguous_mpi_enabled) then
                call fft_i2c_real_contiguous(FFT_x1, PRHS_d, &
                    n1m, n2msub_Isub, n3msub, &
                    n1msub, n2msub, p_poi%comm_1d_x1%mpi_comm)
            else
                ierr = cudaStreamSynchronize()
                call MPI_alltoallw(FFT_x1,                    &
                                   countsendI,                &
                                   countdistI,                &
                                   ddtype_dble_I_in_C2I,      &
                                   PRHS_d,                    &
                                   countsendI,                &
                                   countdistI,                &
                                   ddtype_dble_C_in_C2I,      &
                                   p_poi%comm_1d_x1%mpi_comm, &
                                   ierr)
            endif
            nullify(FFT_x1)
            call poisson_profile_stop(prof_t0, prof_times(17))
            call poisson_profile_start(prof_t0)
#ifdef POISSON_USE_CUDECOMP
            call dscal(n1msub*n2msub*n3msub, &
                real(1,rp)/real(n1m*n2m,rp), PRHS_d, 1)
#else
            call dscal(n1msub*n2msub*n3msub, real(1,rp)/real(n1m,rp), PRHS_d, 1)
#endif
            call poisson_profile_stop(prof_t0, prof_times(18))
            ! Dongyun

            ! FFT_x1(1:n1msub+1,1:n2msub,1:n3msub) => Buff_1
            ! ierr = cufftExecZ2D(plan_fft(2,1), FFT_xc, FFT_x1)
            ! nullify(FFT_xc)

            ! call dcopy(n1msub*n2msub*n3msub, FFT_x1, 1, PRHS_d, 1)
            ! nullify(FFT_x1)

            ! call dscal(n1msub*n2msub*n3msub, real(1,rp)/real(n1m,rp), PRHS_d, 1)

        endif

        call poisson_coarse_profile_mark(6)

        ! call nvtxEndRange

        call poisson_profile_start(prof_t0)
        call cuda_Poisson_average_elimination(P_d, PRHS_d)
        call poisson_profile_stop(prof_t0, prof_times(19))

        call poisson_profile_start(prof_t0)
        call cuda_neumann_BC(P_d, dmx1_d, dmx2_d, dmx3_d)
        call poisson_profile_stop(prof_t0, prof_times(20))

        call poisson_profile_start(prof_t0)
        call cuda_ghostcell_update(P_d)
        call poisson_profile_stop(prof_t0, prof_times(21))
        call poisson_coarse_profile_mark(7)
#ifdef POISSON_DETAILED_PROFILE
        call fft_route_profile_report()
#endif
        call poisson_profile_report(prof_times)

        ! Dongyun
        ! v5 workspace cache: keep buffer_dp1_d/buffer_dp2_d allocated across calls.
        ! They are SAVE local device allocatables and will be released at program termination.
        ! Dongyun
        
        ! call nvtxEndRange

    end subroutine cuda_Poisson_FFT_1D
    
    subroutine cuda_Poisson_average_elimination(P_d, PRHS_d)
        use MPI

        implicit none

        real(rp), device, dimension(0:p_poi%n1sub ,0:p_poi%n2sub ,0:p_poi%n3sub )   :: P_d 
        real(rp), device, dimension(1:p_poi%n1msub,1:p_poi%n2msub,1:p_poi%n3msub)   :: PRHS_d
        real(rp) :: AVERsub, AVERmpi
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

        ! A reduction over x1, then x2, then x3 is mathematically identical
        ! to one reduction over the Cartesian product communicator.  Use the
        ! world communicator directly to remove two collective-latency steps
        ! from every Poisson solve without changing the decomposition.
        AVERmpi = real(0.0,rp)
        call MPI_ALLREDUCE(AVERsub, AVERmpi, 1, MPI_real_type, &
                           MPI_SUM, MPI_COMM_WORLD, ierr)
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
        integer :: required_elements

        required_elements = &
            (max0(p_poi%n1sub,p_poi%n2sub,p_poi%n3sub)+1) * &
            (max0(p_poi%n1sub,p_poi%n2sub,p_poi%n3sub)+1)
        call ensure_ghost_workspaces(required_elements)
        ! v4 tuning: do not zero-fill the ghost buffers here.
        ! The pack kernels overwrite send buffers for active neighbors,
        ! and receive buffers are read only when the matching neighbor exists.
        ! Avoiding these full-buffer device assignments removes four extra
        ! device memset/fill operations per ghost-cell update.

        sbuf_x0_d(0:p_poi%n2sub,0:p_poi%n3sub) => ghost_send_0_d; sbuf_x1_d(0:p_poi%n2sub,0:p_poi%n3sub) => ghost_send_1_d
        rbuf_x0_d(0:p_poi%n2sub,0:p_poi%n3sub) => ghost_recv_0_d; rbuf_x1_d(0:p_poi%n2sub,0:p_poi%n3sub) => ghost_recv_1_d
        call ghostcell_update_on_direction(p_poi%comm_1d_x1, 1, p_poi%pbc1, sbuf_x0_d, sbuf_x1_d, rbuf_x0_d, rbuf_x1_d, p_poi%n1sub, p_poi%n2sub, p_poi%n3sub, Value_sub_d)
        nullify(sbuf_x0_d, sbuf_x1_d, rbuf_x0_d, rbuf_x1_d)

        sbuf_y0_d(0:p_poi%n1sub,0:p_poi%n3sub) => ghost_send_0_d; sbuf_y1_d(0:p_poi%n1sub,0:p_poi%n3sub) => ghost_send_1_d
        rbuf_y0_d(0:p_poi%n1sub,0:p_poi%n3sub) => ghost_recv_0_d; rbuf_y1_d(0:p_poi%n1sub,0:p_poi%n3sub) => ghost_recv_1_d
        call ghostcell_update_on_direction(p_poi%comm_1d_x2, 2, p_poi%pbc2, sbuf_y0_d, sbuf_y1_d, rbuf_y0_d, rbuf_y1_d, p_poi%n2sub, p_poi%n1sub, p_poi%n3sub, Value_sub_d)
        nullify(sbuf_y0_d, sbuf_y1_d, rbuf_y0_d, rbuf_y1_d)

        sbuf_z0_d(0:p_poi%n1sub,0:p_poi%n2sub) => ghost_send_0_d; sbuf_z1_d(0:p_poi%n1sub,0:p_poi%n2sub) => ghost_send_1_d
        rbuf_z0_d(0:p_poi%n1sub,0:p_poi%n2sub) => ghost_recv_0_d; rbuf_z1_d(0:p_poi%n1sub,0:p_poi%n2sub) => ghost_recv_1_d
        call ghostcell_update_on_direction(p_poi%comm_1d_x3, 3, p_poi%pbc3, sbuf_z0_d, sbuf_z1_d, rbuf_z0_d, rbuf_z1_d, p_poi%n3sub, p_poi%n1sub, p_poi%n2sub, Value_sub_d)
        nullify(sbuf_z0_d, sbuf_z1_d, rbuf_z0_d, rbuf_z1_d)
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

        ! v4 tuning: if this direction has only one MPI rank and is non-periodic,
        ! there is no neighbor exchange to perform. Boundary ghost values have
        ! already been set by cuda_neumann_BC before this routine is called.
        if (comm_1d_x%nprocs == 1 .and. .not. pbc) return
    
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

    subroutine cuda_ptdma_core_2rhs(plan, A_d, B_d, C_d, D1_d, D2_d)
        use PaScaL_TDMA_Cuda, only : PaScaL_TDMA_many_solve_2rhs_cuda, &
                                    ptdma_plan_many_cuda
        implicit none

        type(ptdma_plan_many_cuda), intent(inout) :: plan
        real(rp), device, dimension(:,:,:), intent(inout) :: A_d, B_d, C_d
        real(rp), device, dimension(:,:,:), intent(inout) :: D1_d, D2_d

        call PaScaL_TDMA_many_solve_2rhs_cuda( &
            plan, A_d, B_d, C_d, D1_d, D2_d)

    end subroutine cuda_ptdma_core_2rhs

    subroutine cuda_ptdma_core_static_2rhs(plan, A_d, B_d, C_d, D1_d, D2_d)
        use PaScaL_TDMA_Cuda, only : &
            PaScaL_TDMA_many_solve_static_2rhs_cuda, &
            ptdma_plan_many_cuda
        implicit none

        type(ptdma_plan_many_cuda), intent(inout) :: plan
        real(rp), device, dimension(:,:,:), intent(inout) :: A_d, B_d, C_d
        real(rp), device, dimension(:,:,:), intent(inout) :: D1_d, D2_d

        call PaScaL_TDMA_many_solve_static_2rhs_cuda( &
            plan, A_d, B_d, C_d, D1_d, D2_d)

    end subroutine cuda_ptdma_core_static_2rhs

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

        ! v9 tuning: remove the explicit zero-fill of in2(n1+1,:,:).
        ! Nsight Compute reported __pgi_dev_cumemset_16n as a hot kernel;
        ! this zero plane is one source.  Keep the original contiguous copy
        ! in1 -> in2, but avoid reading in2(n1+1,:,:) by handling i=1
        ! directly in the output kernel.
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
            arg = -PI*real(i-1,rp)/(real(2.0,rp)*real(n1,rp))
            if (i == 1) then
                out(i,j,k) = DCMPLX( dcos(arg)*in2(i,j,k), &
                                    -dsin(arg)*in2(i,j,k)  )
            else
                out(i,j,k) = DCMPLX( ( dcos(arg)*in2(i,j,k)-dsin(arg)*in2(n1+2-i,j,k)), &
                                     (-dsin(arg)*in2(i,j,k)-dcos(arg)*in2(n1+2-i,j,k))  )
            endif
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

    subroutine cuda_Poisson_TDMA_z_real(d_d,  &
                                        n1td, n2td, n3td,  &
                                        dx3_d, dmx3_d, &
                                        iend, ista, jend, jsta)
        real(rp),             device, dimension(:,:,:) :: d_d
        real(rp),    pointer, device, dimension(:,:,:) :: a_d, b_d, c_d
        real(rp),             device, dimension(0:)    :: dx3_d, dmx3_d   

        real(rp) :: am, ac, ap
        integer :: i,j,k,kp
        
        ! Dongyun
        integer :: n1td, n2td, n3td, iend, ista, jend, jsta
        integer :: ig, jg
        integer :: rank_x1, rank_x2, rank_x3, nprocs_x3, n3last
        call ensure_tdma_z_workspaces(n1td*n2td*n3td)
        ! Dongyun

        ! allocate(  API_ptr(p_poi%n1msub*p_poi%n2msub*p_poi%n3msub),  ACI_ptr(p_poi%n1msub*p_poi%n2msub*p_poi%n3msub),  AMI_ptr(p_poi%n1msub*p_poi%n2msub*p_poi%n3msub) )

        ! Note that memory size are different!(Just for efficiency)
        ! a_d(1:p_poi%n1msub,1:p_poi%n2msub,1:p_poi%n3msub) => AMI_ptr
        ! b_d(1:p_poi%n1msub,1:p_poi%n2msub,1:p_poi%n3msub) => ACI_ptr
        ! c_d(1:p_poi%n1msub,1:p_poi%n2msub,1:p_poi%n3msub) => API_ptr

        ! Dongyun
        a_d(1:n1td,1:n2td,1:n3td) => tdma_ami_work_d
        b_d(1:n1td,1:n2td,1:n3td) => tdma_aci_work_d
        c_d(1:n1td,1:n2td,1:n3td) => tdma_api_work_d
        
        rank_x1   = p_poi%comm_1d_x1%myrank
        rank_x2   = p_poi%comm_1d_x2%myrank
        rank_x3   = p_poi%comm_1d_x3%myrank
        nprocs_x3 = p_poi%comm_1d_x3%nprocs
        n3last = p_poi%n3msub
        
        !$acc parallel loop collapse(3) private(kp,am,ac,ap,ig,jg)
        do k = 1, n3td
        do j = 1, n2td
        do i = 1, n1td

            ! Global spectral indices in J-line layout
            ig = ista + rank_x2*n1td + i - 1
            jg = j

            kp = k + 1

            am = real(1.0,rp)/dx3_d(k)/dmx3_d(k)
            if (rank_x3 == 0 .and. k == 1) then
                am = real(0.0,rp)
            endif

            ap = real(1.0,rp)/dx3_d(k)/dmx3_d(kp)
            if (rank_x3 == nprocs_x3-1 .and. k == n3last) then
                ap = real(0.0,rp)
            endif

            ac = -am - ap

            a_d(i,j,k) = am
            b_d(i,j,k) = ac - dxk2(ig) - dyk2(jg)
            c_d(i,j,k) = ap

            if (ig == 1 .and. jg == 1 .and. &
                rank_x3 == 0 .and. k == 1) then

                a_d(i,j,k) = real(0.0,rp)
                b_d(i,j,k) = real(1.0,rp)
                c_d(i,j,k) = real(0.0,rp)
                d_d(i,j,k) = real(0.0,rp)
            endif

        enddo
        enddo
        enddo
        !$acc end parallel
        ! Dongyun

        ! !$acc parallel loop collapse(3) private(kp, am, ac, ap) copyin(p_poi%comm_1d_x1, p_poi%comm_1d_x2,p_poi%comm_1d_x3)
        ! do k = 1, p_poi%n3msub
        ! do j = 1, p_poi%n2m
        ! do i = 1, p_poi%n1m
        !     kp = k+1   

        !     am = real(1.0,rp)/dx3_d(k)/dmx3_d(k ); if(p_poi%comm_1d_x3%myrank==0                  .and.k==1      )  am = real(0.0,rp)
        !     ap = real(1.0,rp)/dx3_d(k)/dmx3_d(kp); if(p_poi%comm_1d_x3%myrank==p_poi%comm_1d_x3%nprocs-1.and.k==p_poi%n3msub )  ap = real(0.0,rp)

        !     ac =  - am - ap

        !     a_d(i,j,k) = am
        !     b_d(i,j,k) = ac - dxk2(i) - dyk2(j)
        !     c_d(i,j,k) = ap

        !     if(p_poi%comm_1d_x1%myrank==0.and.p_poi%comm_1d_x2%myrank==0.and.p_poi%comm_1d_x3%myrank==0.and.i==1.and.j==1.and.k==1) then
        !         a_d(1,1,1) = real(0.0,rp)
        !         b_d(1,1,1) = real(1.0,rp)
        !         c_d(1,1,1) = real(0.0,rp)
        !         d_d(1,1,1) = real(0.0,rp)
        !     endif    
        ! end do
        ! end do
        ! end do
        ! !$acc end parallel
        call PaScaL_TDMA_plan_many_create_cuda(ptdma_plan_cuda_fft, &
                                               n1td, &
                                               n2td    , &
                                               n3td, &
                                               p_poi%comm_1d_x3%myrank, &
                                               p_poi%comm_1d_x3%nprocs, &
                                               p_poi%comm_1d_x3%mpi_comm, &
                                               p_poi%threads_tdma)
        call cuda_ptdma_core(ptdma_plan_cuda_fft, a_d, b_d, c_d, d_d, p_poi%pbc3)

        nullify(a_d, b_d, c_d)     
        !$acc exit data delete(p_poi)

    end subroutine cuda_Poisson_TDMA_z_real
    
    ! subroutine cuda_Poisson_TDMA_z_real(d_d, dx3_d, dmx3_d)
    !     real(rp),             device, dimension(:,:,:) :: d_d
    !     real(rp),    pointer, device, dimension(:,:,:) :: a_d, b_d, c_d
    !     real(rp),             device, dimension(0:)    :: dx3_d, dmx3_d   

    !     real(rp), device, target, allocatable, dimension(:) :: API_ptr, ACI_ptr, AMI_ptr
    !     real(rp) :: am, ac, ap
    !     integer :: i,j,k,kp

    !     allocate(  API_ptr(p_poi%n1msub*p_poi%n2msub*p_poi%n3msub),  ACI_ptr(p_poi%n1msub*p_poi%n2msub*p_poi%n3msub),  AMI_ptr(p_poi%n1msub*p_poi%n2msub*p_poi%n3msub) )

    !     ! Note that memory size are different!(Just for efficiency)
    !     a_d(1:p_poi%n1msub,1:p_poi%n2msub,1:p_poi%n3msub) => AMI_ptr
    !     b_d(1:p_poi%n1msub,1:p_poi%n2msub,1:p_poi%n3msub) => ACI_ptr
    !     c_d(1:p_poi%n1msub,1:p_poi%n2msub,1:p_poi%n3msub) => API_ptr

    !     !$acc parallel loop collapse(3) private(kp, am, ac, ap) copyin(p_poi%comm_1d_x1, p_poi%comm_1d_x2,p_poi%comm_1d_x3)
    !     do k = 1, p_poi%n3msub
    !     do j = 1, p_poi%n2m
    !     do i = 1, p_poi%n1m
    !         kp = k+1   

    !         am = real(1.0,rp)/dx3_d(k)/dmx3_d(k ); if(p_poi%comm_1d_x3%myrank==0                  .and.k==1      )  am = real(0.0,rp)
    !         ap = real(1.0,rp)/dx3_d(k)/dmx3_d(kp); if(p_poi%comm_1d_x3%myrank==p_poi%comm_1d_x3%nprocs-1.and.k==p_poi%n3msub )  ap = real(0.0,rp)

    !         ac =  - am - ap

    !         a_d(i,j,k) = am
    !         b_d(i,j,k) = ac - dxk2(i) - dyk2(j)
    !         c_d(i,j,k) = ap

    !         if(p_poi%comm_1d_x1%myrank==0.and.p_poi%comm_1d_x2%myrank==0.and.p_poi%comm_1d_x3%myrank==0.and.i==1.and.j==1.and.k==1) then
    !             a_d(1,1,1) = real(0.0,rp)
    !             b_d(1,1,1) = real(1.0,rp)
    !             c_d(1,1,1) = real(0.0,rp)
    !             d_d(1,1,1) = real(0.0,rp)
    !         endif    
    !     end do
    !     end do
    !     end do
    !     !$acc end parallel

    !     call cuda_ptdma_core(ptdma_plan_cuda_fft, a_d, b_d, c_d, d_d, p_poi%pbc3)

    !     deallocate( API_ptr, ACI_ptr, AMI_ptr )

    !     nullify(a_d, b_d, c_d)     
    !     !$acc exit data delete(p_poi)

    ! end subroutine cuda_Poisson_TDMA_z_real
    
    subroutine cuda_Poisson_TDMA_z_complex(d_d, n1td, n2td, n3td, dx3_d, dmx3_d, iend, ista, jend, jsta, h1psub_Jsub_ista, n2msub_Isub_jsta, n1msub_Jsub_ista)

        complex(rp),          device, dimension(:,:,:) :: d_d
        real(rp),    pointer, device, dimension(:,:,:) :: a_r_d, b_r_d, c_r_d, d_r_d
        real(rp),    pointer, device, dimension(:,:,:) :: a_c_d, b_c_d, c_c_d, d_c_d
        real(rp),             device, dimension(0:)    :: dmx3_d
        real(rp),             device, dimension(0:)    :: dx3_d

        type(comm_1d)                                  ::  comm_1d_x1, comm_1d_x2, comm_1d_x3

        integer                                        ::  n1msub, n2msub, n3msub
        logical                                        ::  pbc1, pbc2, pbc3

        real(rp) :: am, ac, ap, temp
        double precision :: duplicate_coeff_mib
        integer  :: iend, jend, ierr, global_rank
        integer  :: i,j,k, kp, required_elements
        logical  :: use_tdma_2rhs, use_static_tdma
        
        ! Dongyun
        integer :: n1td, n2td, n3td, ista, jsta, h1psub_Jsub_ista, n2msub_Isub_jsta, n1msub_Jsub_ista, i_g, j_g
        double precision :: tdma_prof_t0
        double precision :: tdma_prof_times(tdma_setup_profile_count)

        tdma_prof_times = 0.0d0
        use_tdma_2rhs = poisson_tdma_2rhs_enabled() .and. (.not. p_poi%pbc3)
        use_static_tdma = use_tdma_2rhs .and. &
                          poisson_tdma_static_enabled() .and. &
                          tdma_static_operator_prepared
        call MPI_Comm_rank(MPI_COMM_WORLD, global_rank, ierr)
        if (.not. tdma_2rhs_banner_printed) then
            if (global_rank == 0) then
                if (use_tdma_2rhs) then
                    write(*,'(A)') &
                        '[TDMA-Z] Stage-4 complex solve: shared operator + 2 RHS'
                elseif (poisson_tdma_2rhs_enabled() .and. p_poi%pbc3) then
                    write(*,'(A)') &
                        '[TDMA-Z] Stage-4 2-RHS unavailable for cyclic Z; using legacy path'
                else
                    write(*,'(A)') &
                        '[TDMA-Z] Legacy complex solve: separate real/imag RHS'
                endif
            endif
            tdma_2rhs_banner_printed = .true.
        endif
        if (.not. tdma_static_banner_printed) then
            if (global_rank == 0) then
                if (use_static_tdma) then
                    write(*,'(A)') &
                        '[TDMA-Z] Static operator apply: RHS-only kernels + 2 collectives'
                else
                    write(*,'(A)') &
                        '[TDMA-Z] Dynamic operator apply: rebuild/factor + 5 collectives'
                endif
            endif
            tdma_static_banner_printed = .true.
        endif
        call poisson_profile_start(tdma_prof_t0)
        required_elements = n1td*n2td*n3td
        call ensure_tdma_z_workspaces(required_elements)
        if (use_tdma_2rhs .and. .not. tdma_workspace_banner_printed) then
            duplicate_coeff_mib = dble(3 * required_elements) &
                                  * dble(storage_size(am) / 8) &
                                  / (1024.0d0 * 1024.0d0)
            if (global_rank == 0) then
                write(*,'(A,F10.3,A)') &
                    '[TDMA-Z] Duplicate complex coefficient storage removed/rank: ', &
                    duplicate_coeff_mib, ' MiB'
            endif
            tdma_workspace_banner_printed = .true.
        endif
        ! Dongyun

        comm_1d_x1 = p_poi%comm_1d_x1; comm_1d_x2 = p_poi%comm_1d_x2; comm_1d_x3 = p_poi%comm_1d_x3;

        n1msub = p_poi%n1msub; n2msub = p_poi%n2msub; n3msub = p_poi%n3msub
        pbc1 = p_poi%pbc1; pbc2 = p_poi%pbc2; pbc3 = p_poi%pbc3

        ! allocate(  API_ptr(n1msub*n2msub*n3msub),  ACI_ptr(n1msub*n2msub*n3msub),  AMI_ptr(n1msub*n2msub*n3msub) )
        ! allocate(  APJ_ptr(n1msub*n2msub*n3msub),  ACJ_ptr(n1msub*n2msub*n3msub),  AMJ_ptr(n1msub*n2msub*n3msub) )

        ! allocate( RHS_buff1(n1msub*n2msub*n3msub), RHS_buff2(n1msub*n2msub*n3msub))

        ! Dongyun
        ! a_r_d(1:n1td,1:n2td,1:n3td) => AMI_ptr   ; a_c_d(1:n1td,1:n2td,1:n3td) => AMJ_ptr
        ! b_r_d(1:n1td,1:n2td,1:n3td) => ACI_ptr   ; b_c_d(1:n1td,1:n2td,1:n3td) => ACJ_ptr
        ! c_r_d(1:n1td,1:n2td,1:n3td) => API_ptr   ; c_c_d(1:n1td,1:n2td,1:n3td) => APJ_ptr
        ! d_r_d(1:n1td,1:n2td,1:n3td) => RHS_buff1 ; d_c_d(1:n1td,1:n2td,1:n3td) => RHS_buff2

        if (BCtype(1)=='P') then ! BCType(1)이 P면 (j,i,k)로 가야함
            a_r_d(1:n2td,1:n1td,1:n3td) => tdma_ami_work_d
            b_r_d(1:n2td,1:n1td,1:n3td) => tdma_aci_work_d
            c_r_d(1:n2td,1:n1td,1:n3td) => tdma_api_work_d
            if (use_tdma_2rhs) then
                a_c_d(1:n2td,1:n1td,1:n3td) => tdma_ami_work_d
                b_c_d(1:n2td,1:n1td,1:n3td) => tdma_aci_work_d
                c_c_d(1:n2td,1:n1td,1:n3td) => tdma_api_work_d
            else
                a_c_d(1:n2td,1:n1td,1:n3td) => tdma_amj_work_d
                b_c_d(1:n2td,1:n1td,1:n3td) => tdma_acj_work_d
                c_c_d(1:n2td,1:n1td,1:n3td) => tdma_apj_work_d
            endif
            d_r_d(1:n2td,1:n1td,1:n3td) => tdma_rhs_real_work_d
            d_c_d(1:n2td,1:n1td,1:n3td) => tdma_rhs_imag_work_d
        elseif(BCtype(2)=='P') then ! BCType(2)이 P면 (i,j,k)로 가야함
            a_r_d(1:n1td,1:n2td,1:n3td) => tdma_ami_work_d
            b_r_d(1:n1td,1:n2td,1:n3td) => tdma_aci_work_d
            c_r_d(1:n1td,1:n2td,1:n3td) => tdma_api_work_d
            if (use_tdma_2rhs) then
                a_c_d(1:n1td,1:n2td,1:n3td) => tdma_ami_work_d
                b_c_d(1:n1td,1:n2td,1:n3td) => tdma_aci_work_d
                c_c_d(1:n1td,1:n2td,1:n3td) => tdma_api_work_d
            else
                a_c_d(1:n1td,1:n2td,1:n3td) => tdma_amj_work_d
                b_c_d(1:n1td,1:n2td,1:n3td) => tdma_acj_work_d
                c_c_d(1:n1td,1:n2td,1:n3td) => tdma_apj_work_d
            endif
            d_r_d(1:n1td,1:n2td,1:n3td) => tdma_rhs_real_work_d
            d_c_d(1:n1td,1:n2td,1:n3td) => tdma_rhs_imag_work_d
        endif
        call poisson_profile_stop( &
            tdma_prof_t0, tdma_prof_times(1))
        ! Dongyun

        ! Note that memory size are different!(Just for efficiency) 
        ! if    (BCtype(1)=='P') then ! P-N, P-P
        !     a_r_d(1:n2msub,1:n1msub/2+1,1:n3msub) => AMI_ptr   ; a_c_d(1:n2msub,1:n1msub/2+1,1:n3msub) => AMJ_ptr
        !     b_r_d(1:n2msub,1:n1msub/2+1,1:n3msub) => ACI_ptr   ; b_c_d(1:n2msub,1:n1msub/2+1,1:n3msub) => ACJ_ptr
        !     c_r_d(1:n2msub,1:n1msub/2+1,1:n3msub) => API_ptr   ; c_c_d(1:n2msub,1:n1msub/2+1,1:n3msub) => APJ_ptr
        !     d_r_d(1:n2msub,1:n1msub/2+1,1:n3msub) => RHS_buff1 ; d_c_d(1:n2msub,1:n1msub/2+1,1:n3msub) => RHS_buff2
        ! elseif(BCtype(2)=='P') then ! N-P
        !     a_r_d(1:n1msub,1:n2msub/2+1,1:n3msub) => AMI_ptr   ; a_c_d(1:n1msub,1:n2msub/2+1,1:n3msub) => AMJ_ptr
        !     b_r_d(1:n1msub,1:n2msub/2+1,1:n3msub) => ACI_ptr   ; b_c_d(1:n1msub,1:n2msub/2+1,1:n3msub) => ACJ_ptr
        !     c_r_d(1:n1msub,1:n2msub/2+1,1:n3msub) => API_ptr   ; c_c_d(1:n1msub,1:n2msub/2+1,1:n3msub) => APJ_ptr
        !     d_r_d(1:n1msub,1:n2msub/2+1,1:n3msub) => RHS_buff1 ; d_c_d(1:n1msub,1:n2msub/2+1,1:n3msub) => RHS_buff2
        ! endif

        if    (BCtype(1)=='P') then !Transpose for PASCAL_TDMA. First array row should be even number for alltoall communications.
            call poisson_profile_start(tdma_prof_t0)
            ! !$acc parallel loop collapse(3) private(kp, am, ac, ap) copyin(comm_1d_x1, comm_1d_x2,comm_1d_x3)
            ! do k = 1, n3msub
            ! do i = 1, n1msub/2+1
            ! do j = 1, n2msub
            !     kp = k+1   

            !     am = real(1.0,rp)/dx3_d(k)/dmx3_d(k ); if(comm_1d_x3%myrank==0                  .and.k==1      )  am = real(0.0,rp)
            !     ap = real(1.0,rp)/dx3_d(k)/dmx3_d(kp); if(comm_1d_x3%myrank==comm_1d_x3%nprocs-1.and.k==n3msub )  ap = real(0.0,rp)
            !     ac =  - am - ap

            !     d_r_d(j,i,k) =  real(d_d(i,j,k),rp)
            ! #ifdef SINGLE_PRECISION
            !     d_c_d(j,i,k) =  aimag(d_d(i,j,k))
            ! #elif  DOUBLE_PRECISION
            !     d_c_d(j,i,k) =  dimag(d_d(i,j,k))
            ! #endif

            !     a_r_d(j,i,k) = am
            !     b_r_d(j,i,k) = ac - dxk2(i) - dyk2(j)
            !     c_r_d(j,i,k) = ap
            !     a_c_d(j,i,k) = am
            !     b_c_d(j,i,k) = ac - dxk2(i) - dyk2(j)
            !     c_c_d(j,i,k) = ap

            ! if(comm_1d_x3%myrank==0.and.i==1.and.j==1.and.k==1) then
            !     a_r_d(1,1,1) = real(0.0,rp)
            !     b_r_d(1,1,1) = real(1.0,rp)
            !     c_r_d(1,1,1) = real(0.0,rp)
            !     d_r_d(1,1,1) = real(0.0,rp)
            !     a_c_d(1,1,1) = real(0.0,rp)
            !     b_c_d(1,1,1) = real(1.0,rp)
            !     c_c_d(1,1,1) = real(0.0,rp)
            !     d_c_d(1,1,1) = real(0.0,rp)
            ! endif

            ! end do
            ! end do
            ! end do
            ! !$acc end parallel

            ! Dongyun
            ! v13 PN fix:
            ! Do not put character tests such as BCtype(2)=='N' inside an
            ! OpenACC device kernel.  NVFORTRAN 23.7 can crash in fort2 on
            ! that pattern.  Split P-N and P-P on the host, then launch a
            ! branch-free GPU loop.
            if (BCtype(2) == 'N') then
                ! P-N: x is periodic half-spectrum.  i=1 is kx=0 and must
                ! map to dxk2(1), not dxk2(0).  y is Neumann-distributed,
                ! so use the distributed 1-based y start from n2msub_Isub_jsta.
                !$acc parallel loop collapse(3) private(kp, am, ac, ap, i_g, j_g) &
                !$acc& copyin(comm_1d_x1, comm_1d_x2, comm_1d_x3, use_tdma_2rhs)
                do k = 1, n3td
                do i = 1, n1td
                do j = 1, n2td
                    i_g = i
                    j_g = n2msub_Isub_jsta + j - 1
                    kp = k+1

                    am = real(1.0,rp)/dx3_d(k)/dmx3_d(k ); if(comm_1d_x3%myrank==0                  .and.k==1      )  am = real(0.0,rp)
                    ap = real(1.0,rp)/dx3_d(k)/dmx3_d(kp); if(comm_1d_x3%myrank==comm_1d_x3%nprocs-1.and.k==n3td )  ap = real(0.0,rp)
                    ac = -am - ap

                    d_r_d(j,i,k) = real(d_d(i,j,k),rp)
                #ifdef SINGLE_PRECISION
                    d_c_d(j,i,k) = aimag(d_d(i,j,k))
                #elif  DOUBLE_PRECISION
                    d_c_d(j,i,k) = dimag(d_d(i,j,k))
                #endif

                    a_r_d(j,i,k) = am
                    b_r_d(j,i,k) = ac - dxk2(i_g) - dyk2(j_g)
                    c_r_d(j,i,k) = ap
                    if (.not. use_tdma_2rhs) then
                        a_c_d(j,i,k) = am
                        b_c_d(j,i,k) = ac - dxk2(i_g) - dyk2(j_g)
                        c_c_d(j,i,k) = ap
                    endif

                    if(comm_1d_x3%myrank==0.and.i_g==1.and.j_g==1.and.k==1) then
                        a_r_d(1,1,1) = real(0.0,rp)
                        b_r_d(1,1,1) = real(1.0,rp)
                        c_r_d(1,1,1) = real(0.0,rp)
                        d_r_d(1,1,1) = real(0.0,rp)
                        if (.not. use_tdma_2rhs) then
                            a_c_d(1,1,1) = real(0.0,rp)
                            b_c_d(1,1,1) = real(1.0,rp)
                            c_c_d(1,1,1) = real(0.0,rp)
                        endif
                        d_c_d(1,1,1) = real(0.0,rp)
                    endif

                enddo
                enddo
                enddo
                !$acc end parallel
            else
                ! P-P:
                ! Use the actual 1-based J-layout half-spectrum start passed
                ! from the caller.  In np2 > 1 decompositions, comm_x1 rank
                ! alone is not enough to determine the local x-spectral block.
                if (use_static_tdma) then
                    ! Static operator: only unpack the changing complex RHS.
                    !$acc parallel loop collapse(3) private(i_g, j_g) &
                    !$acc& copyin(comm_1d_x3)
                    do k = 1, n3td
                    do i = 1, n1td
                    do j = 1, n2td
                        i_g = h1psub_Jsub_ista + i - 1
                        j_g = j
                        d_r_d(j,i,k) = real(d_d(i,j,k),rp)
                    #ifdef SINGLE_PRECISION
                        d_c_d(j,i,k) = aimag(d_d(i,j,k))
                    #elif  DOUBLE_PRECISION
                        d_c_d(j,i,k) = dimag(d_d(i,j,k))
                    #endif
                        if (comm_1d_x3%myrank == 0 .and. &
                            i_g == 1 .and. j_g == 1 .and. k == 1) then
                            d_r_d(1,1,1) = real(0.0,rp)
                            d_c_d(1,1,1) = real(0.0,rp)
                        endif
                    enddo
                    enddo
                    enddo
                    !$acc end parallel
                else
                    !$acc parallel loop collapse(3) private(kp, am, ac, ap, i_g, j_g) &
                    !$acc& copyin(comm_1d_x1, comm_1d_x2, comm_1d_x3, use_tdma_2rhs)
                    do k = 1, n3td
                    do i = 1, n1td
                    do j = 1, n2td
                        i_g = h1psub_Jsub_ista + i - 1
                        j_g = j
                        kp = k+1

                        am = real(1.0,rp)/dx3_d(k)/dmx3_d(k ); if(comm_1d_x3%myrank==0                  .and.k==1      )  am = real(0.0,rp)
                        ap = real(1.0,rp)/dx3_d(k)/dmx3_d(kp); if(comm_1d_x3%myrank==comm_1d_x3%nprocs-1.and.k==n3td )  ap = real(0.0,rp)
                        ac = -am - ap

                        d_r_d(j,i,k) = real(d_d(i,j,k),rp)
                    #ifdef SINGLE_PRECISION
                        d_c_d(j,i,k) = aimag(d_d(i,j,k))
                    #elif  DOUBLE_PRECISION
                        d_c_d(j,i,k) = dimag(d_d(i,j,k))
                    #endif

                        a_r_d(j,i,k) = am
                        b_r_d(j,i,k) = ac - dxk2(i_g) - dyk2(j_g)
                        c_r_d(j,i,k) = ap
                        if (.not. use_tdma_2rhs) then
                            a_c_d(j,i,k) = am
                            b_c_d(j,i,k) = ac - dxk2(i_g) - dyk2(j_g)
                            c_c_d(j,i,k) = ap
                        endif

                        if(comm_1d_x3%myrank==0.and.i_g==1.and.j_g==1.and.k==1) then
                            a_r_d(1,1,1) = real(0.0,rp)
                            b_r_d(1,1,1) = real(1.0,rp)
                            c_r_d(1,1,1) = real(0.0,rp)
                            d_r_d(1,1,1) = real(0.0,rp)
                            if (.not. use_tdma_2rhs) then
                                a_c_d(1,1,1) = real(0.0,rp)
                                b_c_d(1,1,1) = real(1.0,rp)
                                c_c_d(1,1,1) = real(0.0,rp)
                            endif
                            d_c_d(1,1,1) = real(0.0,rp)
                        endif
                    enddo
                    enddo
                    enddo
                    !$acc end parallel
                endif
            endif
            call poisson_profile_stop( &
                tdma_prof_t0, tdma_prof_times(2))
            ! Dongyun

            ! v14 PN/PP decomposition fix:
            ! For BCtype(1)=='P', the complex TDMA arrays above are laid out as
            ! a_*(1:n2td,1:n1td,1:n3td) via a_*(j,i,k).  The initialization-time
            ! ptdma_plan_cuda_fft can have stale physical-subdomain dimensions
            ! such as (n2msub,n1msub/2+1,n3msub), which is wrong for decompositions
            ! like np1,np2,np3 = 2,2,1.  Recreate the plan with the actual local
            ! spectral dimensions before solving.
            call poisson_profile_start(tdma_prof_t0)
            call PaScaL_TDMA_plan_many_create_cuda(ptdma_plan_cuda_fft, &
                                                   n2td, &
                                                   n1td, &
                                                   n3td, &
                                                   p_poi%comm_1d_x3%myrank, &
                                                   p_poi%comm_1d_x3%nprocs, &
                                                   p_poi%comm_1d_x3%mpi_comm, &
                                                   p_poi%threads_tdma)
            call poisson_profile_stop( &
                tdma_prof_t0, tdma_prof_times(3))

            if (use_static_tdma) then
                call cuda_ptdma_core_static_2rhs( &
                    ptdma_plan_cuda_fft, a_r_d, b_r_d, c_r_d, d_r_d, d_c_d)
            elseif (use_tdma_2rhs) then
                call cuda_ptdma_core_2rhs( &
                    ptdma_plan_cuda_fft, a_r_d, b_r_d, c_r_d, d_r_d, d_c_d)
            else
                call cuda_ptdma_core(ptdma_plan_cuda_fft, a_r_d, b_r_d, c_r_d, d_r_d, pbc3)
                call cuda_ptdma_core(ptdma_plan_cuda_fft, a_c_d, b_c_d, c_c_d, d_c_d, pbc3)
            endif

            nullify(a_r_d, b_r_d, c_r_d, a_c_d, b_c_d, c_c_d)     

            ! !$cuf kernel do(3) <<<*,*>>>
            ! do k = 1, n3msub
            ! do j = 1, n2msub
            ! do i = 1, n1msub/2+1
            ! #ifdef SINGLE_PRECISION
            !     d_d(i,j,k) =  CMPLX(d_r_d(j,i,k), d_c_d(j,i,k))
            ! #elif  DOUBLE_PRECISION
            !     d_d(i,j,k) = DCMPLX(d_r_d(j,i,k), d_c_d(j,i,k))
            ! #endif
            ! enddo
            ! enddo
            ! enddo

            ! Dongyun
            call poisson_profile_start(tdma_prof_t0)
            !$cuf kernel do(3) <<<*,*>>>
            do k = 1, n3td
            do j = 1, n2td
            do i = 1, n1td
            #ifdef SINGLE_PRECISION
                d_d(i,j,k) =  CMPLX(d_r_d(j,i,k), d_c_d(j,i,k))
            #elif  DOUBLE_PRECISION
                d_d(i,j,k) = DCMPLX(d_r_d(j,i,k), d_c_d(j,i,k))
            #endif
            enddo
            enddo
            enddo
            call poisson_profile_stop( &
                tdma_prof_t0, tdma_prof_times(4))
            ! Dongyun

            !if(myrank.eq.0) call cuda_InstantOutput_mass_3D(d_c_d, myrank)

        elseif(BCtype(2)=='P') then !NP
            call poisson_profile_start(tdma_prof_t0)
            !$acc parallel loop collapse(3) private(kp, am, ac, ap, i_g, j_g) &
            !$acc& copyin(comm_1d_x1, comm_1d_x2, comm_1d_x3, use_tdma_2rhs)
            do k = 1, n3td
            do j = 1, n2td
            do i = 1, n1td
                j_g = j
                i_g = n1msub_Jsub_ista + i - 1
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
                b_r_d(i,j,k) = ac - dxk2(i_g) - dyk2(j_g)
                c_r_d(i,j,k) = ap
                if (.not. use_tdma_2rhs) then
                    a_c_d(i,j,k) = am
                    b_c_d(i,j,k) = ac - dxk2(i_g) - dyk2(j_g)
                    c_c_d(i,j,k) = ap
                endif

            if(comm_1d_x3%myrank==0.and.i_g==1.and.j_g==1.and.k==1) then
                a_r_d(1,1,1) = real(0.0,rp)
                b_r_d(1,1,1) = real(1.0,rp)
                c_r_d(1,1,1) = real(0.0,rp)
                d_r_d(1,1,1) = real(0.0,rp)
                if (.not. use_tdma_2rhs) then
                    a_c_d(1,1,1) = real(0.0,rp)
                    b_c_d(1,1,1) = real(1.0,rp)
                    c_c_d(1,1,1) = real(0.0,rp)
                endif
                d_c_d(1,1,1) = real(0.0,rp)
            endif

            end do
            end do
            end do
            !$acc end parallel
            call poisson_profile_stop( &
                tdma_prof_t0, tdma_prof_times(2))

            call poisson_profile_start(tdma_prof_t0)
            call PaScaL_TDMA_plan_many_create_cuda(ptdma_plan_cuda_fft, &
                                                n1td, &
                                                n2td    , &
                                                n3td, &
                                                p_poi%comm_1d_x3%myrank, &
                                                p_poi%comm_1d_x3%nprocs, &
                                                p_poi%comm_1d_x3%mpi_comm, &
                                                p_poi%threads_tdma)
            call poisson_profile_stop( &
                tdma_prof_t0, tdma_prof_times(3))
            if (use_tdma_2rhs) then
                call cuda_ptdma_core_2rhs( &
                    ptdma_plan_cuda_fft, a_r_d, b_r_d, c_r_d, d_r_d, d_c_d)
            else
                call cuda_ptdma_core(ptdma_plan_cuda_fft, a_r_d, b_r_d, c_r_d, d_r_d, pbc3)
                call cuda_ptdma_core(ptdma_plan_cuda_fft, a_c_d, b_c_d, c_c_d, d_c_d, pbc3)
            endif

            nullify(a_r_d, b_r_d, c_r_d, a_c_d, b_c_d, c_c_d)     

            ! v19 NP fix: copy back the actual local TDMA shape.
            ! n1td/n2td/n3td may differ from physical n1msub/n2msub in decomposed layouts.
            call poisson_profile_start(tdma_prof_t0)
            !$cuf kernel do(3) <<<*,*>>>
            do k = 1, n3td
            do j = 1, n2td
            do i = 1, n1td
            #ifdef SINGLE_PRECISION
                d_d(i,j,k) =  CMPLX(d_r_d(i,j,k), d_c_d(i,j,k))
            #elif  DOUBLE_PRECISION
                d_d(i,j,k) = DCMPLX(d_r_d(i,j,k), d_c_d(i,j,k))
            #endif
            enddo
            enddo
            enddo
            call poisson_profile_stop( &
                tdma_prof_t0, tdma_prof_times(4))
        endif

        call poisson_profile_start(tdma_prof_t0)
        nullify(d_r_d, d_c_d)     
        call poisson_profile_stop( &
            tdma_prof_t0, tdma_prof_times(5))
        call tdma_setup_profile_report(tdma_prof_times)

    end subroutine cuda_Poisson_TDMA_z_complex

end module
