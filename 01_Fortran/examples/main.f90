module wrapper_module
    use mpi
    use global
#ifdef USE_GPU
    use cudafor
    use openacc
    use mpi_subdomain
    use cuda_subdomain
    use fft_poisson
    use cuda_pressure
    use cuda_post
#ifdef use_nvtx
    use nvtx
#endif
    double precision :: timer(4)
#else
    use mpi_topology
    use mpi_subdomain
    use mpi_poisson
    use mpi_Post
    use timer
#endif
end module wrapper_module


program main
    use wrapper_module
    implicit none
    integer :: ierr
#ifndef USE_GPU
    integer :: timestep, Tstepmax = 10
    character(len=64) :: timer_str(64)
    integer, parameter :: stamp_main = 1, stamp_total = 4
#endif

    call initial

#ifdef USE_GPU
    call cuda_subdomain_temp_allocation()

    if(myrank==0) write(*,*) "[[[============== all setup finished ==============]]]"
    if(myrank==0) write(*,*) "[[[=========== main simulation starts! ============]]]"

    timer(1) = mpi_wtime()

#ifdef use_nvtx
    call nvtxstartrange("Poisson")
#endif

    prhs_d(1:n1msub,1:n2msub,1:n3msub) => temp1
    call cuda_pressure_RHS(prhs_d)
    call cuda_Poisson_FFT_1D(PRHS_d, P_d, dmx1_d, dmx2_d, dmx3_d, dx3_d)
    nullify(prhs_d)
    call cuda_subdomain_DtoH(P_d, P)

#ifdef use_nvtx
    call nvtxendrange
#endif

    timer(4) = mpi_wtime()
    if(myrank==0) write(*,*) "Total calculation time: ", timer(4)-timer(1)

    call cuda_Post_fileout_instantfield_3d(P)
#else
    timer_str(1)  = '[Main] poisson RHS             '
    timer_str(2)  = '[Main] poisson FFT             '
    timer_str(3)  = '[Main] ghostcell update        '
    timer_str(4)  = '[fft] TDMA                     '
    timer_str(5)  = '[fft] FFT                      '
    timer_str(6)  = '[fft] ALLTOALL                 '
    timer_str(7)  = '[fft] UPDATE SOLUTION          '
    timer_str(8)  = '[fft] others                   '
    timer_str(9)  = '[fft] build TDMA               '
    timer_str(10) = '[tdma] calc TDMA               '
    timer_str(11) = '[tdma] comm TDMA               '
    timer_str(12) = '[RHS] allocate                 '
    timer_str(13) = '[RHS] calc                     '
    timer_str(14) = '[Main] total                   '

    if(myrank==0) write(*,*) '[Main] Iteration starts!'
    do timestep = 1, Tstepmax
        if(myrank==0) write(*,*) '[Main] tstep=', timestep

        call timer_init(14,timer_str)
        call MPI_Barrier(MPI_COMM_WORLD,ierr)

        call timer_stamp0(stamp_main)
        call timer_stamp0(stamp_total)
        call mpi_poisson_RHS
        call timer_stamp(1,stamp_main)
        call mpi_Poisson_FFT1(dx2_sub,dmx2_sub,P)
        call timer_stamp(2,stamp_main)
        call mpi_subdomain_ghostcell_update(P)
        call timer_stamp(3,stamp_main)
        call timer_stamp(14,stamp_total)

        call mpi_poisson_exact_sol()
        call mpi_Post_error(myrank, P, exact_sol, rms)

        call timer_reduction()
        call timer_output(myrank, nprocs)
    enddo
#endif

    call clean
end program main


subroutine initial
    use wrapper_module
    implicit none
    integer :: ierr

#ifdef USE_GPU
#ifdef use_nvtx
    call nvtxstartrange("initial")
#endif
    call mpi_subdomain_environment()                                        ; if(myrank==0) write(*,*) "00: mpi initializing"
    call cuda_subdomain_environment()                                       ; if(myrank==0) write(*,*) "01: cuda initializing"
    call global_inputpara(myrank)                                           ; if(myrank==0) write(*,*) "02: global parameters loaded"
    call mpi_subdomain_initial()                                            ; if(myrank==0) write(*,*) "04: mpi subdomain initializing"
    call cuda_subdomain_initial()                                           ; if(myrank==0) write(*,*) "05: cuda subdomain initializing"
    call fft_poisson_plan_cuda_create(comm_1d_x1%myrank   , comm_1d_x2%myrank   , comm_1d_x3%myrank   ,&
                                      comm_1d_x1%nprocs   , comm_1d_x2%nprocs   , comm_1d_x3%nprocs   ,&
                                      comm_1d_x1%west_rank, comm_1d_x2%west_rank, comm_1d_x3%west_rank,&
                                      comm_1d_x1%east_rank, comm_1d_x2%east_rank, comm_1d_x3%east_rank,&
                                      comm_1d_x1%mpi_comm , comm_1d_x2%mpi_comm , comm_1d_x3%mpi_comm ,&
                                      n1, n2, n3, n1sub, n2sub, n3sub, L1, L2, L3, pbc1, pbc2, pbc3, threads_tdma, threads_fft)
    call cuda_Poisson_FFT_initial()
    call cuda_pressure_memory('allocate')                                   ; if(myrank==0) write(*,*) "12: pressure memory allocated"
    call cuda_post_initial()                                                ; if(myrank==0) write(*,*) "15: post-processing initialized"
    call cuda_subdomain_ghostcell_update(P_d)
    call cuda_subdomain_DtoH(P_d, P)                                        ; if(myrank==0) write(*,*) "16: ghostcell update and H-D sync done"
    call mpi_barrier(mpi_comm_world,ierr)
    ierr = cudadevicesynchronize()
    if (ContinueFilein==.true.) then                                        ; if(myrank==0) write(*,*) "17: loading continue file"
        call cuda_Post_FileIn_Continue_Post_Reassembly_IO(myrank,P)
        call cuda_subdomain_HtoD(P, P_d)
    end if
#ifdef use_nvtx
    call nvtxendrange
#endif
#else
    call MPI_Init(ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)

    if(myrank==0) write(*,*) '[Main] The simulation starts!'

    if(myrank==0) call system('mkdir -p ./data/1_continue')
    if(myrank==0) call system('mkdir -p ./data/2_instanfield')
    call MPI_Barrier(MPI_COMM_WORLD,ierr)

    call global_inputpara()
    if(myrank==0) write(*,*) '[Main] Global parameters loaded.'

    call mpi_topology_make()
    call mpi_subdomain_make()
    call mpi_subdomain_mesh()
    call mpi_subdomain_indices()
    call mpi_subdomain_DDT_ghostcell()
    call mpi_poisson_allocation()
    call mpi_subdomain_DDT_transpose1()
    call mpi_poisson_wave_number()
    call mpi_subdomain_ghostcell_update(P)
    call MPI_Barrier(MPI_COMM_WORLD,ierr)

    if(myrank==0) write(*,*) '[Main] Setup complete.'
#endif
end subroutine initial


subroutine clean
    use wrapper_module
    implicit none
    integer :: ierr

#ifdef USE_GPU
    call cuda_destroy_ptr()
    call cuda_post_clean()
    call cuda_pressure_memory('clean')
    call cuda_Poisson_FFT_clean()
    call cuda_subdomain_clean()
    call mpi_topology_clean()
    call mpi_subdomain_clean()
    call mpi_finalize(ierr)
    if(myrank==0) write(*,*) '[Main] Simulation complete. (GPU)'
#else
    call mpi_poisson_clean()
    call mpi_subdomain_indices_clean()
    call mpi_subdomain_clean()
    call mpi_topology_clean()
    call MPI_FINALIZE(ierr)
    if(myrank==0) write(*,*) '[Main] Simulation complete. (CPU)'
#endif
end subroutine clean
