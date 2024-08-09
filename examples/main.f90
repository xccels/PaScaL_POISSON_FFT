module wrapper_module
        use global
        #ifdef use_nvtx
        use nvtx ! for profiling
        #endif
        ! mpi
        use mpi
        use mpi_subdomain
        ! cuda
        use cudafor
        use openacc
        use cuda_subdomain
        ! poisson solver
        use fft_poisson
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

        call initial

        !! allocatable memory
        call cuda_subdomain_temp_allocation() ! temp1: dv_d,dp_d temp2: dw_d
        
        if(myrank==0) write(*,*) "[[[============== all setup finished ==============]]]"
        if(myrank==0) write(*,*) "[[[=========== main simulation starts! ============]]]"

        timer(1)=mpi_wtime()


        #ifdef use_nvtx
        call nvtxstartrange("Poisson")
        #endif
                prhs_d(1:n1msub,1:n2msub,1:n3msub) => temp1 ! warning: grid size are different
                call cuda_pressure_RHS(prhs_d) 

                call cuda_Poisson_FFT_1D(PRHS_d, P_d, dmx1_d, dmx2_d, dmx3_d, dx3_d)
                nullify(prhs_d) ! nullify temp1  
                
                call cuda_subdomain_DtoH(P_d, P)   

        #ifdef use_nvtx
        call nvtxendrange
        #endif
       

        timer(4)=mpi_wtime()
        if(myrank==0) write(*,*) "Total calculation time: ", timer(4)-timer(1)

        call cuda_Post_fileout_instantfield_3d(P)

        call clean

end program

subroutine initial
        use wrapper_module
        implicit none
        integer :: ierr
        
        #ifdef use_nvtx
        call nvtxstartrange("initial")
        #endif
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

        #ifdef use_nvtx
        call nvtxendrange
        #endif

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