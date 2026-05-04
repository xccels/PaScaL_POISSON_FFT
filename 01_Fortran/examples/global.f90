module global
    use mpi
    implicit none

#ifdef USE_GPU
#ifdef SINGLE_PRECISION
    integer, parameter :: rp = kind(0.0),   MPI_real_type = MPI_REAL,             MPI_complex_type = MPI_COMPLEX
#else
    integer, parameter :: rp = kind(0.0d0), MPI_real_type = MPI_DOUBLE_PRECISION, MPI_complex_type = MPI_DOUBLE_COMPLEX
#endif
    real(rp), parameter :: PI = real(dacos(-1.0d0), rp)
#else
    double precision, parameter :: PI = acos(-1.d0)
#endif

    ! Common: mesh size
    integer :: n1, n2, n3
    integer :: n1m, n2m, n3m
    integer :: n1p, n2p, n3p

    ! Common: MPI decomposition
    integer :: np1, np2, np3

    ! Common: boundary conditions and mesh style
    logical :: pbc1, pbc2, pbc3
    integer :: uniform1, uniform2, uniform3

    ! Common: domain geometry
#ifdef USE_GPU
    real(rp) :: gamma1, gamma2, gamma3
    real(rp) :: L1, L2, L3
    real(rp) :: x1_start, x2_start, x3_start
    real(rp) :: x1_end,   x2_end,   x3_end
    real(rp) :: Volume
    real(rp) :: resolution_x, resolution_y, resolution_z

    ! GPU: CUDA thread configuration
    integer  :: thread_in_x, thread_in_y, thread_in_z
    integer  :: thread_in_x_pascal, thread_in_y_pascal, thread_in_fft_pascal
    integer  :: Poisson_model

    ! GPU: simulation control
    logical  :: ContinueFilein, ContinueFileout
#else
    double precision :: gamma1, gamma2, gamma3
    double precision :: L1, L2, L3
    double precision :: H, Aspect1, Aspect2, Aspect3
    double precision :: x1_start, x2_start, x3_start
    double precision :: x1_end,   x2_end,   x3_end
    double precision :: rms, rms_local

    ! CPU: simulation control
    integer :: ContinueFilein, ContinueFileout
    integer :: print_start_step, print_interval_step
    integer :: print_j_index_wall, print_j_index_bulk, print_k_index
#endif

    ! Common: I/O directories
    character(len=128) :: dir_cont_filein, dir_cont_fileout, dir_instantfield

contains

#ifdef USE_GPU
    subroutine global_inputpara(myrank)
        implicit none
        integer, intent(in) :: myrank
        integer :: ierr

        namelist /domain_size/             L1, L2, L3
        namelist /mesh/                    n1m, n2m, n3m
        namelist /grid_style/              pbc1, pbc2, pbc3, uniform1, uniform2, uniform3, gamma1, gamma2, gamma3
        namelist /MPI_procs/               np1, np2, np3
        namelist /cuda_thread/             thread_in_x, thread_in_y, thread_in_z
        namelist /cuda_pascal_tdma_thread/ thread_in_x_pascal, thread_in_y_pascal, thread_in_fft_pascal
        namelist /Poisson/                 Poisson_model
        namelist /post_option/             ContinueFilein, ContinueFileout
        namelist /directories/             dir_cont_filein, dir_cont_fileout, dir_instantfield

        open(unit=1, file="../run/PARA_INPUT.dat")
            read(1, domain_size)
            read(1, mesh)
            read(1, grid_style)
            read(1, MPI_procs)
            read(1, cuda_thread)
            read(1, cuda_pascal_tdma_thread)
            read(1, Poisson)
            read(1, post_option)
            read(1, directories)
        close(1)

        x1_start = real(0.0, rp); x2_start = real(0.0, rp); x3_start = real(0.0, rp)
        x1_end = x1_start+L1;     x2_end = x2_start+L2;     x3_end = x3_start+L3
        Volume = L1*L2*L3

        n1=n1m+1; n1p=n1+1
        n2=n2m+1; n2p=n2+1
        n3=n3m+1; n3p=n3+1

        resolution_x = real(n1m,rp)/L1
        resolution_y = real(n2m,rp)/L2
        resolution_z = real(n3m,rp)/L3

        if (myrank == 0) then
            write(*,'(A)') '========== Unified Poisson FFT Solver (GPU) =========='
            write(*,'(A,3F10.2)') '  Domain  (L1,L2,L3)        :', L1, L2, L3
            write(*,'(A,3I10)')   '  Mesh    (n1m,n2m,n3m)     :', n1m, n2m, n3m
            write(*,'(A,3L10)')   '  PBC     (x1, x2, x3)      :', pbc1, pbc2, pbc3
            write(*,'(A,3I10)')   '  MPI     (np1,np2,np3)      :', np1, np2, np3
            write(*,'(A,3I10)')   '  Threads (x,y,z)            :', thread_in_x, thread_in_y, thread_in_z
            write(*,'(A,3I10)')   '  TDMA    (x,y,fft)          :', thread_in_x_pascal, thread_in_y_pascal, thread_in_fft_pascal
            write(*,'(A)') '======================================================'
        endif

        if (n1m < (2*np1*np3 - 2)) then
            write(*,*) "n1m must be >= 2*np1*np3-2. Aborting."
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        elseif (mod(n1m,2)/=0 .or. mod(n2m,2)/=0 .or. mod(n3m,2)/=0) then
            write(*,*) "n1m, n2m, n3m must all be even. Aborting."
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif

        call system('mkdir -p ../run/data/1_continue')
        call system('mkdir -p ../run/data/2_instantfield')
    end subroutine global_inputpara

#else

    subroutine global_inputpara()
        implicit none
        double precision :: tttmp(1:3)

        namelist /sim_continue/      ContinueFilein, ContinueFileout, dir_cont_filein, dir_cont_fileout, dir_instantfield
        namelist /meshes/            n1m, n2m, n3m
        namelist /MPI_procs/         np1, np2, np3
        namelist /periodic_boundary/ pbc1, pbc2, pbc3
        namelist /uniform_mesh/      uniform1, uniform2, uniform3
        namelist /mesh_stretch/      gamma1, gamma2, gamma3
        namelist /aspect_ratio/      H, Aspect1, Aspect2, Aspect3

        open(unit=1, file="PARA_INPUT.dat")
        read(1, sim_continue)
        read(1, meshes)
        read(1, MPI_procs)
        read(1, periodic_boundary)
        read(1, uniform_mesh)
        read(1, mesh_stretch)
        read(1, aspect_ratio)
        close(1)

        n1=n1m+1; n1p=n1+1
        n2=n2m+1; n2p=n2+1
        n3=n3m+1; n3p=n3+1

        L1=H*Aspect1; L2=H*Aspect2; L3=H*Aspect3
        x1_start=0.0d0; x2_start=0.0d0; x3_start=0.0d0
        x1_end=x1_start+L1; x2_end=x2_start+L2; x3_end=x3_start+L3

        tttmp(1)=L1/dble(n1-1); tttmp(2)=L2/dble(n2-1); tttmp(3)=L3/dble(n3-1)
    end subroutine global_inputpara

#endif

end module global
