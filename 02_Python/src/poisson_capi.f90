!=======================================================================================================================
!> @file        poisson_capi.f90
!> @brief       C-callable wrapper module around the CUDA FFT Poisson solver of PaScaL_POISSON_FFT.
!> @details     The wrapper exposes the fft_poisson module through ISO_C_BINDING so that the
!>              device-resident GPU solver can be driven from other languages, e.g. Python (ctypes)
!>              with CuPy device arrays and CUDA-aware mpi4py.
!>
!>              Division of labour between this file and the Python side:
!>              - Fortran (here): everything that touches the fft_poisson module state, i.e. plan
!>                creation, cuDecomp/static-TDMA initialisation, the solve itself, and teardown.
!>              - Python: the MPI cartesian topology, the block decomposition, the grid metrics,
!>                and the MPI derived datatypes for the C<->I and C<->J transposes.  The solver
!>                receives all of those as ordinary arguments, so it does not care who built them.
!>
!>              Two things cross the language boundary as opaque integers:
!>              - Device arrays are passed as raw addresses typed integer(c_intptr_t).  A Fortran
!>                device pointer is rebuilt from the address with c_devptr and c_f_pointer, so the
!>                data never leaves the GPU.
!>              - MPI communicators and derived datatypes are passed as Fortran integer handles,
!>                i.e. what mpi4py returns from Comm.py2f() / Datatype.py2f().
!>
!>              Array conventions expected from the caller (all Fortran/column-major order,
!>              double precision).  n1sub/n2sub/n3sub and n1msub = n1sub-1 etc. follow the
!>              subdomain definition of the reference example:
!>                PRHS   (n1msub,   n2msub,   n3msub  )   right-hand side, also used as scratch
!>                P      (n1sub+1,  n2sub+1,  n3sub+1 )   solution including one ghost layer
!>                dmx1   (n1sub+1), dmx2 (n2sub+1), dmx3 (n3sub+1), dx3 (n3sub+1)
!>              Only the size and the start address of these arrays matter: every dummy argument
!>              downstream is assumed-shape and re-establishes its own index origin.
!>
!> @author
!>              - Jungwoo Kim (yasandy@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
!>
!> @date        August 2026
!> @version     1.0
!> @par         License
!>              This project is released under the terms of the MIT License (see LICENSE file).
!=======================================================================================================================

!>
!> @brief       Module wrapping fft_poisson behind a C-compatible interface.
!> @details     fft_poisson keeps its plan in a single module-level variable (p_poi), so there is
!>              exactly one solver instance per process and no handle table is needed.  The sizes
!>              given at plan creation are cached here and reused to rebuild device pointers and to
!>              size the derived-datatype arrays in poisson_solve.
!>
module poisson_capi

    use iso_c_binding
    use cudafor
    use poisson_timer, only : poisson_timer_reset, poisson_timer_values
    use fft_poisson, only : rp,                             &
                            fft_poisson_plan_cuda_create,   &
                            cuda_Poisson_FFT_initial,       &
                            cuda_Poisson_cudecomp_initial,  &
                            cuda_Poisson_TDMA_static_initial,   &
                            cuda_Poisson_FFT_1D,            &
                            cuda_Poisson_FFT_clean
    implicit none

    !> Block decomposition used by the library.  External subroutine in src/para_range.f90;
    !> declared here so poisson_para_range calls the very same arithmetic the library uses.
    interface
        subroutine para_range(n1, n2, nprocs, myrank, ista, iend)
            integer, intent(in)  :: n1, n2, nprocs, myrank
            integer, intent(out) :: ista, iend
        end subroutine para_range
    end interface

    !> Sizes cached at plan creation.  p_poi is private inside fft_poisson, so the wrapper cannot
    !> read them back and keeps its own copy.
    type :: poisson_state
        logical :: created = .false.
        integer :: np1 = 0, np2 = 0, np3 = 0            ! process counts per direction
        integer :: n1sub = 0, n2sub = 0, n3sub = 0      ! subdomain sizes (ghost layer included)
        integer :: n1msub = 0, n2msub = 0, n3msub = 0   ! n?sub - 1
    end type poisson_state

    type(poisson_state), save :: st

contains

    !>
    !> @brief       Block-distribute the range [nsta, nend] over nprocs processes.
    !> @details     Thin wrapper around the library's para_range so that the Python side derives its
    !>              subdomain indices from the same arithmetic instead of reimplementing it.  A
    !>              one-off mismatch here would not raise an error, it would silently change the
    !>              wavenumbers the solver works on.
    !>
    subroutine poisson_para_range(nsta, nend, nprocs, myrank, index_sta, index_end) &
                                  bind(C, name="poisson_para_range")
        integer(c_int), value :: nsta, nend, nprocs, myrank
        integer(c_int), intent(out) :: index_sta, index_end
        integer :: isbeg, isend

        call para_range(nsta, nend, nprocs, myrank, isbeg, isend)
        index_sta = isbeg
        index_end = isend
    end subroutine poisson_para_range

    !>
    !> @brief       Create the Poisson plan and allocate the solver workspaces.
    !> @details     Combines fft_poisson_plan_cuda_create with cuda_Poisson_FFT_initial, which the
    !>              reference example always calls as a pair: the latter derives the boundary-condition
    !>              type and every workspace size from the plan just stored.
    !> @param       rank1/2/3       Rank ID in the 1D communicator of each direction
    !> @param       np1/2/3         Process count in each direction
    !> @param       wrank1/2/3      West (previous) neighbour rank in each direction
    !> @param       erank1/2/3      East (next) neighbour rank in each direction
    !> @param       comm1/2/3       Fortran handles of the 1D communicators (mpi4py Comm.py2f())
    !> @param       n1/2/3          Global grid point counts
    !> @param       n1sub/2sub/3sub Subdomain grid point counts
    !> @param       L1/2/3          Domain lengths
    !> @param       ipbc1/2/3       Periodic flag per direction, 0 = Neumann, non-zero = periodic
    !>                              (an int, not a Fortran logical, whose C representation is compiler-defined)
    !> @param       tdma_tx/ty/tz   CUDA thread block for the TDMA kernels
    !> @param       fft_tx/ty/tz    CUDA thread block for the FFT kernels
    !>
    subroutine poisson_plan_create(rank1, rank2, rank3, np1, np2, np3,          &
                                   wrank1, wrank2, wrank3,                      &
                                   erank1, erank2, erank3,                      &
                                   comm1, comm2, comm3,                         &
                                   n1, n2, n3, n1sub, n2sub, n3sub,             &
                                   L1, L2, L3, ipbc1, ipbc2, ipbc3,             &
                                   tdma_tx, tdma_ty, tdma_tz,                   &
                                   fft_tx, fft_ty, fft_tz)                      &
                                   bind(C, name="poisson_plan_create")
        integer(c_int), value :: rank1, rank2, rank3, np1, np2, np3
        integer(c_int), value :: wrank1, wrank2, wrank3
        integer(c_int), value :: erank1, erank2, erank3
        integer(c_int), value :: comm1, comm2, comm3
        integer(c_int), value :: n1, n2, n3, n1sub, n2sub, n3sub
        real(c_double), value :: L1, L2, L3
        integer(c_int), value :: ipbc1, ipbc2, ipbc3
        integer(c_int), value :: tdma_tx, tdma_ty, tdma_tz
        integer(c_int), value :: fft_tx, fft_ty, fft_tz

        type(dim3) :: threads_tdma, threads_fft

        threads_tdma = dim3(tdma_tx, tdma_ty, tdma_tz)
        threads_fft  = dim3(fft_tx,  fft_ty,  fft_tz)

        call fft_poisson_plan_cuda_create(rank1, rank2, rank3,      &
                                          np1, np2, np3,            &
                                          wrank1, wrank2, wrank3,   &
                                          erank1, erank2, erank3,   &
                                          comm1, comm2, comm3,      &
                                          n1, n2, n3,               &
                                          n1sub, n2sub, n3sub,      &
                                          real(L1, rp), real(L2, rp), real(L3, rp), &
                                          ipbc1 /= 0, ipbc2 /= 0, ipbc3 /= 0,       &
                                          threads_tdma, threads_fft)

        call cuda_Poisson_FFT_initial()

        st%np1 = np1;       st%np2 = np2;       st%np3 = np3
        st%n1sub = n1sub;   st%n2sub = n2sub;   st%n3sub = n3sub
        st%n1msub = n1sub - 1
        st%n2msub = n2sub - 1
        st%n3msub = n3sub - 1
        st%created = .true.
    end subroutine poisson_plan_create

    !>
    !> @brief       Initialise the cuDecomp transpose backend.
    !> @details     No-op unless the library was built with -DPOISSON_USE_CUDECOMP, and inside the
    !>              library it only takes effect for the periodic/periodic horizontal case.
    !> @param       n2msub_Isub     y-extent of the x-aligned slab
    !> @param       h1psub_Jsub     x-extent (in complex wavenumbers) of the y-aligned slab
    !>
    subroutine poisson_cudecomp_init(n2msub_Isub, h1psub_Jsub) &
                                     bind(C, name="poisson_cudecomp_init")
        integer(c_int), value :: n2msub_Isub, h1psub_Jsub

        if (.not. st%created) return
        call cuda_Poisson_cudecomp_initial(n2msub_Isub, h1psub_Jsub)
    end subroutine poisson_cudecomp_init

    !>
    !> @brief       Build and factor the invariant spectral TDMA operator once, outside the solve loop.
    !> @details     Optional: the library falls back to rebuilding the coefficients inside every solve
    !>              when this is skipped or when the boundary conditions are unsupported.
    !> @param       n1td, n2td, n3td    Spectral TDMA system shape; the example passes
    !>                                  (h1psub_Jsub, n2m, n3msub)
    !> @param       dx3_addr            Device address of dx3,  n3sub+1 elements
    !> @param       dmx3_addr           Device address of dmx3, n3sub+1 elements
    !> @param       h1psub_Jsub_ista    Global start index of this rank's wavenumber slab
    !>
    subroutine poisson_tdma_static_init(n1td, n2td, n3td, dx3_addr, dmx3_addr, h1psub_Jsub_ista) &
                                        bind(C, name="poisson_tdma_static_init")
        integer(c_int), value :: n1td, n2td, n3td, h1psub_Jsub_ista
        integer(c_intptr_t), value :: dx3_addr, dmx3_addr

        type(c_devptr) :: cptr
        real(rp), device, pointer, dimension(:) :: dx3_p, dmx3_p

        if (.not. st%created) return

        cptr = transfer(dx3_addr,  cptr); call c_f_pointer(cptr, dx3_p,  [st%n3sub + 1])
        cptr = transfer(dmx3_addr, cptr); call c_f_pointer(cptr, dmx3_p, [st%n3sub + 1])

        call cuda_Poisson_TDMA_static_initial(n1td, n2td, n3td, dx3_p, dmx3_p, h1psub_Jsub_ista)

        nullify(dx3_p, dmx3_p)
    end subroutine poisson_tdma_static_init

    !>
    !> @brief       Solve the 3D Poisson equation once.
    !> @details     On return P holds the solution; PRHS is destroyed because the solver reuses it as
    !>              working storage.  Regenerate the right-hand side before every call.
    !>
    !>              The twelve integer arrays are host arrays of MPI handles and counts, indexed by
    !>              rank within the corresponding 1D communicator.  Sizes are np1 for the *I* family
    !>              (C<->I transpose, x direction) and np2 for the *J* family (C<->J transpose,
    !>              y direction); they are taken from the cached plan, so the caller passes bare
    !>              pointers and does not repeat the lengths.
    !>
    !> @param       prhs_addr   Device address of PRHS, (n1msub, n2msub, n3msub)
    !> @param       p_addr      Device address of P,    (n1sub+1, n2sub+1, n3sub+1)
    !> @param       dmx1_addr   Device address of dmx1, n1sub+1 elements
    !> @param       dmx2_addr   Device address of dmx2, n2sub+1 elements
    !> @param       dmx3_addr   Device address of dmx3, n3sub+1 elements
    !> @param       dx3_addr    Device address of dx3,  n3sub+1 elements
    !>
    subroutine poisson_solve(prhs_addr, p_addr,                                     &
                             dmx1_addr, dmx2_addr, dmx3_addr, dx3_addr,             &
                             h1psub, h1psub_Jsub, n2msub_Isub, n1msub_Jsub,         &
                             countsendI, countdistI, countsendJ, countdistJ,        &
                             ddt_dble_C_in_C2I, ddt_dble_I_in_C2I,                  &
                             ddt_dble_J_in_C2J, ddt_dble_C_in_C2J,                  &
                             ddt_cplx_I_in_C2I, ddt_cplx_C_in_C2I,                  &
                             ddt_cplx_J_in_C2J, ddt_cplx_C_in_C2J,                  &
                             iend, ista, jend, jsta,                                &
                             h1psub_Jsub_ista, n2msub_Isub_jsta, n1msub_Jsub_ista)  &
                             bind(C, name="poisson_solve")
        integer(c_intptr_t), value :: prhs_addr, p_addr
        integer(c_intptr_t), value :: dmx1_addr, dmx2_addr, dmx3_addr, dx3_addr
        integer(c_int), value :: h1psub, h1psub_Jsub, n2msub_Isub, n1msub_Jsub
        integer(c_int), value :: iend, ista, jend, jsta
        integer(c_int), value :: h1psub_Jsub_ista, n2msub_Isub_jsta, n1msub_Jsub_ista

        integer(c_int) :: countsendI(*), countdistI(*), countsendJ(*), countdistJ(*)
        integer(c_int) :: ddt_dble_C_in_C2I(*), ddt_dble_I_in_C2I(*)
        integer(c_int) :: ddt_dble_J_in_C2J(*), ddt_dble_C_in_C2J(*)
        integer(c_int) :: ddt_cplx_I_in_C2I(*), ddt_cplx_C_in_C2I(*)
        integer(c_int) :: ddt_cplx_J_in_C2J(*), ddt_cplx_C_in_C2J(*)

        type(c_devptr) :: cptr
        real(rp), device, pointer, dimension(:,:,:) :: prhs_p, p_p
        real(rp), device, pointer, dimension(:)     :: dmx1_p, dmx2_p, dmx3_p, dx3_p
        integer :: ni, nj

        if (.not. st%created) return
        ni = st%np1
        nj = st%np2

        cptr = transfer(prhs_addr, cptr)
        call c_f_pointer(cptr, prhs_p, [st%n1msub, st%n2msub, st%n3msub])
        cptr = transfer(p_addr, cptr)
        call c_f_pointer(cptr, p_p, [st%n1sub + 1, st%n2sub + 1, st%n3sub + 1])

        cptr = transfer(dmx1_addr, cptr); call c_f_pointer(cptr, dmx1_p, [st%n1sub + 1])
        cptr = transfer(dmx2_addr, cptr); call c_f_pointer(cptr, dmx2_p, [st%n2sub + 1])
        cptr = transfer(dmx3_addr, cptr); call c_f_pointer(cptr, dmx3_p, [st%n3sub + 1])
        cptr = transfer(dx3_addr,  cptr); call c_f_pointer(cptr, dx3_p,  [st%n3sub + 1])

        call cuda_Poisson_FFT_1D(prhs_p, p_p,                                   &
                                 dmx1_p, dmx2_p, dmx3_p, dx3_p,                 &
                                 h1psub, h1psub_Jsub, n2msub_Isub, n1msub_Jsub, &
                                 countsendI(1:ni), countdistI(1:ni),            &
                                 countsendJ(1:nj), countdistJ(1:nj),            &
                                 ddt_dble_C_in_C2I(1:ni), ddt_dble_I_in_C2I(1:ni),  &
                                 ddt_dble_J_in_C2J(1:nj), ddt_dble_C_in_C2J(1:nj),  &
                                 ddt_cplx_I_in_C2I(1:ni), ddt_cplx_C_in_C2I(1:ni),  &
                                 ddt_cplx_J_in_C2J(1:nj), ddt_cplx_C_in_C2J(1:nj),  &
                                 iend, ista, jend, jsta,                            &
                                 h1psub_Jsub_ista, n2msub_Isub_jsta, n1msub_Jsub_ista)

        nullify(prhs_p, p_p, dmx1_p, dmx2_p, dmx3_p, dx3_p)
    end subroutine poisson_solve

    !>
    !> @brief       Drop the computation/communication accumulators, e.g. after warm-up solves.
    !> @details     No-op unless POISSON_TIMER=1.  Warm-up carries cuFFT plan creation and cuDecomp
    !>              autotuning, which do not belong in a steady-state measurement.
    !>
    subroutine poisson_timer_clear() bind(C, name="poisson_timer_clear")

        call poisson_timer_reset()
    end subroutine poisson_timer_clear

    !>
    !> @brief       This rank's timing accumulators, summed over the solves since the last clear.
    !> @details     Raw per-rank totals; the caller reduces them.  All zero when POISSON_TIMER is
    !>              unset.  comp + comm equals total by construction, so a caller can use their
    !>              ratio as a check that no part of the solve escaped instrumentation.
    !> @param       comp    Seconds of local computation
    !> @param       comm    Seconds inside MPI or NCCL calls
    !> @param       total   Seconds for the whole solve
    !> @param       nsolve  Number of solves accumulated
    !>
    subroutine poisson_timer_read(comp, comm, total, nsolve) bind(C, name="poisson_timer_read")
        real(c_double), intent(out) :: comp, comm, total
        integer(c_int), intent(out) :: nsolve

        double precision :: a, b, c
        integer :: n

        call poisson_timer_values(a, b, c, n)
        comp = a; comm = b; total = c; nsolve = n
    end subroutine poisson_timer_read

    !>
    !> @brief       Block until every queued GPU operation has completed.
    !> @details     The solve is asynchronous with respect to the host.  Python needs this before
    !>              reading results back or stopping a timer.
    !> @param       istat   CUDA status code (0 = cudaSuccess)
    !>
    subroutine poisson_sync(istat) bind(C, name="poisson_sync")
        integer(c_int), intent(out) :: istat

        istat = cudaDeviceSynchronize()
    end subroutine poisson_sync

    !>
    !> @brief       Release every solver workspace and invalidate the plan.
    !>
    subroutine poisson_destroy() bind(C, name="poisson_destroy")

        if (.not. st%created) return
        call cuda_Poisson_FFT_clean()
        st%created = .false.
    end subroutine poisson_destroy

end module poisson_capi
