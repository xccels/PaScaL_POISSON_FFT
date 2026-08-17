module wrapper_module
        use global
        ! #ifdef use_nvtx
        ! use nvtx ! for profiling
        ! #endif
        ! mpi
        use mpi
        use mpi_subdomain
        ! cuda
        use cudafor
        use openacc
        use cuda_subdomain
        ! poisson solver
        use fft_poisson
        use poisson_timer, only : poisson_timer_reset, poisson_timer_report
        ! physics
        use cuda_pressure
        ! for postprocessing
        use cuda_post

        double precision :: timer(4)
end module

program mpm_std
        use wrapper_module
        implicit none
        integer :: ierr, istat
        integer :: ierr_cuda
        integer :: iter, warmup_solves, timed_solves
        integer :: debug_warmup_solves, debug_cold_profile
        integer :: write_solution
        integer :: mpi_size
        double precision :: batch_start, batch_elapsed_local
        double precision :: batch_elapsed_max
        double precision :: perf_total_min, perf_total_avg, perf_total_max
        double precision, allocatable :: perf_times(:)
        double precision, allocatable :: perf_min(:)
        double precision, allocatable :: perf_sum(:)
        double precision, allocatable :: perf_max(:)

        call initial

        !! allocatable memory
        call cuda_subdomain_temp_allocation() ! temp1: dv_d,dp_d temp2: dw_d
        
        if(myrank==0) write(*,*) "[[[============== all setup finished ==============]]]"
        if(myrank==0) write(*,*) "[[[=========== main simulation starts! ============]]]"

        prhs_d(1:n1msub,1:n2msub,1:n3msub) => temp1

#ifdef POISSON_DIRECT_C2I_FFT
        if (myrank == 0) then
            write(*,'(A)') &
                '[C2I-BUILD] direct: Cube-to-X writes FFT_x1; dcopy removed.'
        endif
#else
        if (myrank == 0) then
            write(*,'(A)') &
                '[C2I-BUILD] legacy: Cube-to-X stages through PRHS_Iline_d.'
        endif
#endif

        ! Batch/scaling runs need only the RMS value for correctness.  Large
        ! per-rank Tecplot solution dumps are therefore opt-in.
        call read_integer_environment( &
            'POISSON_WRITE_SOLUTION', 0, 0, write_solution)

#ifdef PERF_BUILD
        call read_integer_environment( &
            'POISSON_WARMUP', 5, 0, warmup_solves)
        call read_integer_environment( &
            'POISSON_ITERS', 20, 1, timed_solves)

        allocate(perf_times(timed_solves))
        allocate(perf_min(timed_solves))
        allocate(perf_sum(timed_solves))
        allocate(perf_max(timed_solves))
        perf_times = 0.0d0
        perf_min = 0.0d0
        perf_sum = 0.0d0
        perf_max = 0.0d0

        call MPI_Comm_size(MPI_COMM_WORLD, mpi_size, ierr)

        if (myrank == 0) then
            write(*,'(/,A)') &
                '================ PERFORMANCE BUILD =================='
            write(*,'(A,I0)') &
                '[PERF] Warm-up Poisson solves : ', warmup_solves
            write(*,'(A,I0)') &
                '[PERF] Timed Poisson solves   : ', timed_solves
            write(*,'(A)') &
                '[PERF] Detailed phase synchronization/profiling: OFF'
            write(*,'(A)') &
                '[PERF] Optional coarse phases: POISSON_COARSE_PROFILE=1'
            write(*,'(A)') &
                '[PERF] RHS regeneration is excluded from solve time.'
            write(*,'(A)') &
                '[PERF] Final RMS check runs once after all timers stop.'
            write(*,'(A,/)') &
                '====================================================='
        endif

        ! Warm up CUDA-aware MPI, CUDA IPC/GPUDirect registration, cuFFT plans,
        ! cached workspaces, and the GPU kernels before strong-scaling timing.
        do iter = 1, warmup_solves
            call cuda_pressure_RHS(prhs_d)
            call solve_poisson_once()
        enddo

        ! Discard any optional coarse-profile samples collected during warm-up.
        call poisson_coarse_profile_reset()
        ! Same for the computation/communication split: warm-up carries cuFFT plan
        ! creation and cuDecomp autotuning, which are not part of a steady solve.
        call poisson_timer_reset()

        ierr_cuda = cudaDeviceSynchronize()
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        batch_start = MPI_Wtime()

        do iter = 1, timed_solves
            ! cuda_Poisson_FFT_1D uses PRHS_d as working storage. Rebuild the
            ! deterministic RHS before every solve, as a CFD time step would.
            ! Synchronizing here keeps RHS generation outside the solve timer.
            call cuda_pressure_RHS(prhs_d)
            ierr_cuda = cudaDeviceSynchronize()

            timer(1) = MPI_Wtime()
            call solve_poisson_once()
            ierr_cuda = cudaDeviceSynchronize()
            perf_times(iter) = MPI_Wtime() - timer(1)
        enddo

        batch_elapsed_local = MPI_Wtime() - batch_start

        ! Reduce only after all timed solves. This avoids inserting an extra
        ! MPI collective between consecutive CFD-style Poisson solves.
        call MPI_Reduce(perf_times, perf_min, timed_solves, &
                        MPI_DOUBLE_PRECISION, MPI_MIN, 0, &
                        MPI_COMM_WORLD, ierr)
        call MPI_Reduce(perf_times, perf_sum, timed_solves, &
                        MPI_DOUBLE_PRECISION, MPI_SUM, 0, &
                        MPI_COMM_WORLD, ierr)
        call MPI_Reduce(perf_times, perf_max, timed_solves, &
                        MPI_DOUBLE_PRECISION, MPI_MAX, 0, &
                        MPI_COMM_WORLD, ierr)
        call MPI_Reduce(batch_elapsed_local, batch_elapsed_max, 1, &
                        MPI_DOUBLE_PRECISION, MPI_MAX, 0, &
                        MPI_COMM_WORLD, ierr)

        ! One reduction after all solve timers.  This prints nothing when the
        ! optional coarse profiler is disabled.
        call poisson_coarse_profile_report()
        call poisson_timer_report()

        if (myrank == 0) then
            perf_total_min = sum(perf_min) / dble(timed_solves)
            perf_total_avg = sum(perf_sum) / &
                             dble(timed_solves * mpi_size)
            perf_total_max = sum(perf_max) / dble(timed_solves)

            write(*,'(/,A)') &
                '================ Poisson performance ================='
            write(*,'(A)') &
                'Columns: rank-min / rank-avg / rank-max [seconds/solve]'
            write(*,'(A,3(1X,ES13.6))') &
                '[PERF] Steady-state average', &
                perf_total_min, perf_total_avg, perf_total_max
            write(*,'(A,1X,ES13.6)') &
                '[PERF] Best critical-path solve', minval(perf_max)
            write(*,'(A,1X,ES13.6)') &
                '[PERF] Worst critical-path solve', maxval(perf_max)
            write(*,'(A,1X,ES13.6)') &
                '[PERF] Critical-path solves/second', &
                1.0d0 / perf_total_max
            write(*,'(A,1X,ES13.6)') &
                '[PERF] Batch wall/solve including RHS reset', &
                batch_elapsed_max / dble(timed_solves)
            write(*,'(A,/)') &
                '========================================================'

            ! Backward-compatible labels for existing scaling parsers.
            write(*,*) 'Total calculation time: ', sum(perf_max)
            write(*,*) 'Average Poisson solve time: ', perf_total_max
        endif

        deallocate(perf_times, perf_min, perf_sum, perf_max)
#else
        call read_integer_environment( &
            'POISSON_DEBUG_WARMUP', 5, 0, debug_warmup_solves)
        call read_integer_environment( &
            'POISSON_DEBUG_COLD_PROFILE', 1, 0, debug_cold_profile)

        if (myrank == 0) then
            write(*,'(/,A)') &
                '================ DEBUG PROFILE BUILD ================='
            write(*,'(A)') &
                '[DEBUG] Detailed phase synchronization/profiling: ON'
            write(*,'(A,I0)') &
                '[DEBUG] Unprofiled warm-up solves: ', debug_warmup_solves
            write(*,'(A,L1)') &
                '[DEBUG] Include cold-start profile: ', &
                debug_cold_profile /= 0
            write(*,'(A)') &
                '[DEBUG] Final RMS check runs once after all timers stop.'
            write(*,'(A,/)') &
                '========================================================'
        endif

        ! Keep a cold-start profile for initialization diagnostics, then turn
        ! profiling off while CUDA-aware MPI, cuFFT, kernels, and workspaces
        ! warm up.  The final detailed profile is the steady-state path that
        ! should be compared with the performance executable.
        if (debug_cold_profile /= 0) then
            if (myrank == 0) then
                write(*,'(/,A)') &
                    '----- COLD-START DETAILED PROFILE -----'
            endif
            call poisson_profile_set_enabled(.true., .true.)
            call cuda_pressure_RHS(prhs_d)
            ierr_cuda = cudaDeviceSynchronize()
            call MPI_Barrier(MPI_COMM_WORLD, ierr)

            timer(1) = MPI_Wtime()
            call solve_poisson_once()
            ierr_cuda = cudaDeviceSynchronize()
            timer(2) = MPI_Wtime()

            if (myrank == 0) then
                write(*,'(A,1X,ES13.6,A)') &
                    '[DEBUG] Cold-start profiled solve:', &
                    timer(2)-timer(1), ' s'
            endif
        endif

        call poisson_profile_set_enabled(.false., .true.)
        do iter = 1, debug_warmup_solves
            call cuda_pressure_RHS(prhs_d)
            call solve_poisson_once()
        enddo

        ierr_cuda = cudaDeviceSynchronize()
        call MPI_Barrier(MPI_COMM_WORLD, ierr)

        call poisson_timer_reset()
        call poisson_profile_set_enabled(.true., .true.)
        if (myrank == 0) then
            write(*,'(/,A)') &
                '----- WARMED STEADY-STATE DETAILED PROFILE -----'
        endif
        call cuda_pressure_RHS(prhs_d)
        ierr_cuda = cudaDeviceSynchronize()
        call MPI_Barrier(MPI_COMM_WORLD, ierr)

        timer(1) = MPI_Wtime()
        call solve_poisson_once()
        ierr_cuda = cudaDeviceSynchronize()
        timer(4) = MPI_Wtime()
        call poisson_profile_set_enabled(.false.)
        call poisson_timer_report()

        if (myrank == 0) then
            write(*,'(A,1X,ES13.6,A)') &
                '[DEBUG] Warmed profiled solve:', &
                timer(4)-timer(1), ' s'
            write(*,*) 'Total calculation time: ', timer(4)-timer(1)
        endif
#endif

        nullify(prhs_d)

        ! Correctness is checked only for the final solution.  Both the D2H
        ! copy and RMS reductions are intentionally outside every solve and
        ! batch timing interval.
        if (myrank == 0) then
            write(*,'(/,A)') &
                '[CORRECTNESS] Final-solution RMS check (outside timers).'
        endif
        call cuda_subdomain_DtoH(P_d, P)

        call post_rms_error(P, myrank)
        if (write_solution /= 0) then
            call cuda_Post_fileout_instantfield_3d(P)
        elseif (myrank == 0) then
            write(*,'(A)') &
                '[OUTPUT] Solution dump disabled; RMS correctness only.'
        endif

        call clean

end program


subroutine solve_poisson_once()
        use wrapper_module
        implicit none

        call cuda_Poisson_FFT_1D( &
            PRHS_d, P_d, dmx1_d, dmx2_d, dmx3_d, dx3_d, &
            h1psub, h1psub_Jsub, n2msub_Isub, n1msub_Jsub, &
            countsendI, countdistI, countsendJ, countdistJ, &
            ddtype_dble_C_in_C2I, ddtype_dble_I_in_C2I, &
            ddtype_dble_J_in_C2J, ddtype_dble_C_in_C2J, &
            ddtype_cplx_I_in_C2I, ddtype_cplx_C_in_C2I, &
            ddtype_cplx_J_in_C2J, ddtype_cplx_C_in_C2J, &
            iend, ista, jend, jsta, &
            h1psub_Jsub_ista, n2msub_Isub_jsta, &
            n1msub_Jsub_ista)

end subroutine solve_poisson_once


subroutine read_integer_environment(name, default_value, minimum_value, value)
        implicit none

        character(len=*), intent(in) :: name
        integer, intent(in) :: default_value, minimum_value
        integer, intent(out) :: value

        character(len=64) :: text
        integer :: env_status, text_length, read_status

        value = default_value
        text = ''

        call get_environment_variable( &
            name, text, length=text_length, status=env_status)

        if (env_status == 0 .and. text_length > 0) then
            read(text(1:text_length), *, iostat=read_status) value
            if (read_status /= 0 .or. value < minimum_value) then
                value = default_value
            endif
        endif

end subroutine read_integer_environment

subroutine initial
        use wrapper_module
        implicit none
        integer :: ierr
        
        ! #ifdef use_nvtx
        ! call nvtxstartrange("initial")
        ! #endif
        call  mpi_subdomain_environment()                                       ; if(myrank==0) write(*,*) "00: mpi initializing" 
        call cuda_subdomain_environment()                                       ; if(myrank==0) write(*,*) "01: cuda initializing"
        call global_inputpara(myrank)                                           ; if(myrank==0) write(*,*) "02: module_global setting"

        call  mpi_subdomain_initial()                                           ; if(myrank==0) write(*,*) "04: mpi subdomain initializing"
        call cuda_subdomain_initial()                                           ; if(myrank==0) write(*,*) "05: cuda subdomain initializing"

        call fft_poisson_plan_cuda_create(comm_1d_x1%myrank   , comm_1d_x2%myrank   , comm_1d_x3%myrank   ,&
                                          comm_1d_x1%nprocs   , comm_1d_x2%nprocs   , comm_1d_x3%nprocs   ,&
                                          comm_1d_x1%west_rank, comm_1d_x2%west_rank, comm_1d_x3%west_rank,&
                                          comm_1d_x1%east_rank, comm_1d_x2%east_rank, comm_1d_x3%east_rank,&
                                          comm_1d_x1%mpi_comm , comm_1d_x2%mpi_comm , comm_1d_x3%mpi_comm ,&
                                          n1, n2, n3, n1sub, n2sub, n3sub, L1, L2, L3, pbc1, pbc2, pbc3, threads_tdma, threads_fft)
        call cuda_Poisson_FFT_initial()
        call cuda_Poisson_cudecomp_initial( &
            n2msub_Isub, h1psub_Jsub)
        call cuda_Poisson_TDMA_static_initial( &
            h1psub_Jsub, n2m, n3msub, dx3_d, dmx3_d, &
            h1psub_Jsub_ista)
        call cuda_pressure_memory('allocate')                                   ; if(myrank==0) write(*,*) "12: pressure initializing"         

        call cuda_post_initial()                                                ; if(myrank==0) write(*,*) "15: post process allocation"

        call cuda_subdomain_ghostcell_update(P_d)
        call cuda_subdomain_DtoH(P_d, P)                                        ; if(myrank==0) write(*,*) "15: ghostcell update and host-device synchronizing"

        call mpi_barrier(mpi_comm_world,ierr)
        ierr = cudadevicesynchronize()

        if (ContinueFilein==.true.) then                                             ; if(myrank==0) write(*,*) "16: continue calculation setting"      
                call cuda_Post_FileIn_Continue_Post_Reassembly_IO(myrank,P)
                call cuda_subdomain_HtoD(P, P_d)
        end if

        ! #ifdef use_nvtx
        ! call nvtxendrange
        ! #endif

end subroutine initial


subroutine clean
        use wrapper_module
        implicit none
        integer :: ierr
        call cuda_destroy_ptr()
        call cuda_post_clean()
        call cuda_pressure_memory('clean')
        call cuda_Poisson_FFT_clean()

        call cuda_subdomain_clean()
        call mpi_topology_clean()
        call mpi_subdomain_clean()
        
        call mpi_finalize(ierr)

        if(myrank==0) write(*,*) '[main] The main simulation complete.'
        
end subroutine clean

! Dongyun
subroutine post_rms_error(P, myrank)
    use global, only : rp, PI, MPI_real_type, n1m, n2m, n3m
    use mpi
    use mpi_subdomain, only : n1sub, n2sub, n3sub,       &
                             n1msub, n2msub, n3msub,    &
                             x1, x2, x3
    implicit none

    real(rp), intent(in) :: P(0:n1sub,0:n2sub,0:n3sub)
    integer, intent(in)  :: myrank

    real(rp) :: err_local, err_global
    real(rp) :: p_exact, rms
    real(rp) :: xc, yc, zc
    integer :: i, j, k, ierr

    err_local = real(0.0,rp)

    do k = 1, n3msub
        zc = real(0.5,rp) * (x3(k) + x3(k+1))

        do j = 1, n2msub
            yc = real(0.5,rp) * (x2(j) + x2(j+1))

            do i = 1, n1msub
                xc = real(0.5,rp) * (x1(i) + x1(i+1))

                p_exact = cos(PI*xc) &
                         *cos(PI*yc) &
                         *cos(PI*zc)

                err_local = err_local + &
                    (P(i,j,k)-p_exact)**2
            enddo
        enddo
    enddo

    ! This correctness-only collective executes once, after all solve timers.
    call MPI_Allreduce(err_local, err_global, 1,          &
                       MPI_real_type, MPI_SUM,            &
                       MPI_COMM_WORLD, ierr)

    rms = sqrt(err_global / real(n1m*n2m*n3m,rp))

    if (myrank == 0) then
        write(*,*) "RMS error against analytic solution:", rms
    endif

end subroutine post_rms_error
! Dongyun
