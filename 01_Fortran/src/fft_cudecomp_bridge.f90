module fft_cudecomp_bridge
    use cudafor
    use mpi
    use cudecomp
    use poisson_timer, only : poisson_timer_comm_enter, poisson_timer_comm_exit
    use, intrinsic :: iso_fortran_env, only: int64

    implicit none

#ifdef SINGLE_PRECISION
    integer, parameter :: rp_bridge = kind(0.0)
    integer, parameter :: cudecomp_complex_type = CUDECOMP_FLOAT_COMPLEX
#else
    integer, parameter :: rp_bridge = kind(0.0d0)
    integer, parameter :: cudecomp_complex_type = CUDECOMP_DOUBLE_COMPLEX
#endif

    type(cudecompHandle), save :: fft_cudecomp_handle
    type(cudecompGridDesc), save :: fft_cudecomp_grid
    type(cudecompPencilInfo), save :: fft_cudecomp_x_info
    type(cudecompPencilInfo), save :: fft_cudecomp_y_info
    complex(rp_bridge), pointer, device, contiguous, save :: &
        fft_cudecomp_work_d(:) => null()

    logical, save :: fft_cudecomp_ready = .false.
    integer, save :: fft_cudecomp_backend = CUDECOMP_TRANSPOSE_COMM_NCCL
    character(len=16), save :: fft_cudecomp_backend_name = 'nccl'

contains

    subroutine fft_cudecomp_abort(istat, where)
        implicit none

        integer, intent(in) :: istat
        character(len=*), intent(in) :: where
        integer :: ierr, myrank

        if (istat == CUDECOMP_RESULT_SUCCESS) return

        call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
        write(*,'(A,I0,2A,I0)') &
            '[CUDECOMP-ERROR] rank=', myrank, ' call=', trim(where), istat
        call MPI_Abort(MPI_COMM_WORLD, 9200 + istat, ierr)
    end subroutine fft_cudecomp_abort


    subroutine fft_cudecomp_select_backend()
        implicit none

        character(len=64) :: raw
        integer :: env_status, ierr, myrank

        raw = 'nccl'
        call get_environment_variable( &
            'POISSON_CUDECOMP_BACKEND', raw, status=env_status)
        if (env_status /= 0 .or. len_trim(raw) == 0) raw = 'nccl'

        select case(trim(adjustl(raw)))
        case('nccl')
            fft_cudecomp_backend = CUDECOMP_TRANSPOSE_COMM_NCCL
            fft_cudecomp_backend_name = 'nccl'
        case('nccl_pl')
            fft_cudecomp_backend = CUDECOMP_TRANSPOSE_COMM_NCCL_PL
            fft_cudecomp_backend_name = 'nccl_pl'
        case('mpi_a2a')
            fft_cudecomp_backend = CUDECOMP_TRANSPOSE_COMM_MPI_A2A
            fft_cudecomp_backend_name = 'mpi_a2a'
        case('mpi_p2p')
            fft_cudecomp_backend = CUDECOMP_TRANSPOSE_COMM_MPI_P2P
            fft_cudecomp_backend_name = 'mpi_p2p'
        case default
            call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
            if (myrank == 0) then
                write(*,'(3A)') &
                    '[CUDECOMP-ERROR] Unsupported backend: ', &
                    trim(adjustl(raw)), &
                    ' (use nccl, nccl_pl, mpi_a2a, or mpi_p2p)'
            endif
            call MPI_Abort(MPI_COMM_WORLD, 9209, ierr)
        end select
    end subroutine fft_cudecomp_select_backend


    subroutine fft_cudecomp_initialize( &
        h1p, n2m, n3m, np1, np2, np3, &
        n2msub_iline, n3msub, h1psub_jline)

        implicit none

        integer, intent(in) :: h1p, n2m, n3m
        integer, intent(in) :: np1, np2, np3
        integer, intent(in) :: n2msub_iline, n3msub, h1psub_jline

        type(cudecompGridDescConfig) :: config
        integer(int64) :: workspace_elements
        integer :: expected_x(3), expected_y(3)
        integer :: expected_x_order(3), expected_y_order(3)
        integer :: istat, ierr, myrank, nranks
        integer :: local_bad, global_bad

        if (fft_cudecomp_ready) return

        call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
        call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)

        if (np1*np2*np3 /= nranks) then
            if (myrank == 0) then
                write(*,'(A,4(1X,I0))') &
                    '[CUDECOMP-ERROR] decomposition/world mismatch:', &
                    np1, np2, np3, nranks
            endif
            call MPI_Abort(MPI_COMM_WORLD, 9210, ierr)
        endif

        call fft_cudecomp_select_backend()

        istat = cudecompInit(fft_cudecomp_handle, MPI_COMM_WORLD)
        call fft_cudecomp_abort(istat, 'cudecompInit')

        istat = cudecompGridDescConfigSetDefaults(config)
        call fft_cudecomp_abort(istat, 'cudecompGridDescConfigSetDefaults')

        ! PaScaL keeps its np1 x np2 x np3 cube ownership.  After the
        ! x-direction FFT, the I pencil is distributed over np1*np2 in y and
        ! np3 in z.  The J pencil uses the matching cuDecomp Y-pencil layout.
        config%gdims = [h1p, n2m, n3m]
        config%pdims = [np1*np2, np3]
        config%transpose_comm_backend = fft_cudecomp_backend
        config%halo_comm_backend = CUDECOMP_HALO_COMM_MPI
        config%transpose_axis_contiguous = [.true., .true., .true.]

        istat = cudecompGridDescCreate( &
            fft_cudecomp_handle, fft_cudecomp_grid, config)
        call fft_cudecomp_abort(istat, 'cudecompGridDescCreate')

        istat = cudecompGetPencilInfo( &
            fft_cudecomp_handle, fft_cudecomp_grid, &
            fft_cudecomp_x_info, 1)
        call fft_cudecomp_abort(istat, 'cudecompGetPencilInfo(X)')
        istat = cudecompGetPencilInfo( &
            fft_cudecomp_handle, fft_cudecomp_grid, &
            fft_cudecomp_y_info, 2)
        call fft_cudecomp_abort(istat, 'cudecompGetPencilInfo(Y)')

        expected_x = [h1p, n2msub_iline, n3msub]
        expected_y = [n2m, n3msub, h1psub_jline]
        expected_x_order = [1, 2, 3]
        expected_y_order = [2, 3, 1]
        local_bad = 0
        if (any(fft_cudecomp_x_info%shape /= expected_x)) local_bad = 1
        if (any(fft_cudecomp_y_info%shape /= expected_y)) local_bad = 1
        if (any(fft_cudecomp_x_info%order /= expected_x_order)) local_bad = 1
        if (any(fft_cudecomp_y_info%order /= expected_y_order)) local_bad = 1
        call MPI_Allreduce( &
            local_bad, global_bad, 1, MPI_INTEGER, MPI_MAX, &
            MPI_COMM_WORLD, ierr)

        if (global_bad /= 0) then
            write(*,'(A,I0,A,3(I0,1X),A,3(I0,1X))') &
                '[CUDECOMP-ERROR] rank=', myrank, &
                ' X actual/expected=', fft_cudecomp_x_info%shape, &
                '/', expected_x
            write(*,'(A,I0,A,3(I0,1X),A,3(I0,1X))') &
                '[CUDECOMP-ERROR] rank=', myrank, &
                ' Y actual/expected=', fft_cudecomp_y_info%shape, &
                '/', expected_y
            write(*,'(A,I0,A,3(I0,1X),A,3(I0,1X))') &
                '[CUDECOMP-ERROR] rank=', myrank, &
                ' X order actual/expected=', fft_cudecomp_x_info%order, &
                '/', expected_x_order
            write(*,'(A,I0,A,3(I0,1X),A,3(I0,1X))') &
                '[CUDECOMP-ERROR] rank=', myrank, &
                ' Y order actual/expected=', fft_cudecomp_y_info%order, &
                '/', expected_y_order
            call MPI_Abort(MPI_COMM_WORLD, 9211, ierr)
        endif

        istat = cudecompGetTransposeWorkspaceSize( &
            fft_cudecomp_handle, fft_cudecomp_grid, workspace_elements)
        call fft_cudecomp_abort( &
            istat, 'cudecompGetTransposeWorkspaceSize')
        istat = cudecompMalloc( &
            fft_cudecomp_handle, fft_cudecomp_grid, &
            fft_cudecomp_work_d, workspace_elements)
        call fft_cudecomp_abort(istat, 'cudecompMalloc(transpose)')

        fft_cudecomp_ready = .true.

        if (myrank == 0) then
            write(*,'(/,A)') &
                '================ cuDecomp I<->J route ================'
            write(*,'(A,A)') &
                '[CUDECOMP] Backend        : ', &
                trim(fft_cudecomp_backend_name)
            write(*,'(A,3(I0,1X))') &
                '[CUDECOMP] Spectral gdims : ', config%gdims
            write(*,'(A,2(I0,1X))') &
                '[CUDECOMP] Process pdims  : ', config%pdims
            write(*,'(A,3(I0,1X))') &
                '[CUDECOMP] Rank-0 X shape : ', &
                fft_cudecomp_x_info%shape
            write(*,'(A,3(I0,1X))') &
                '[CUDECOMP] Rank-0 Y shape : ', &
                fft_cudecomp_y_info%shape
            write(*,'(A,3(I0,1X))') &
                '[CUDECOMP] X memory order  : ', &
                fft_cudecomp_x_info%order
            write(*,'(A,3(I0,1X))') &
                '[CUDECOMP] Y memory order  : ', &
                fft_cudecomp_y_info%order
            write(*,'(A,I0)') &
                '[CUDECOMP] Workspace elems: ', workspace_elements
            write(*,'(A,/)') &
                '========================================================'
        endif
    end subroutine fft_cudecomp_initialize


    subroutine fft_cudecomp_x_to_y(input_d, output_d)
        implicit none

        complex(rp_bridge), device :: input_d(*), output_d(*)
        integer :: istat, ierr

        if (.not. fft_cudecomp_ready) then
            call MPI_Abort(MPI_COMM_WORLD, 9220, ierr)
        endif

        call poisson_timer_comm_enter()
        istat = cudecompTransposeXToY( &
            fft_cudecomp_handle, fft_cudecomp_grid, &
            input_d, output_d, fft_cudecomp_work_d, &
            cudecomp_complex_type)
        call fft_cudecomp_abort(istat, 'cudecompTransposeXToY')
        call poisson_timer_comm_exit()
    end subroutine fft_cudecomp_x_to_y


    subroutine fft_cudecomp_y_to_x(input_d, output_d)
        implicit none

        complex(rp_bridge), device :: input_d(*), output_d(*)
        integer :: istat, ierr

        if (.not. fft_cudecomp_ready) then
            call MPI_Abort(MPI_COMM_WORLD, 9221, ierr)
        endif

        call poisson_timer_comm_enter()
        istat = cudecompTransposeYToX( &
            fft_cudecomp_handle, fft_cudecomp_grid, &
            input_d, output_d, fft_cudecomp_work_d, &
            cudecomp_complex_type)
        call fft_cudecomp_abort(istat, 'cudecompTransposeYToX')
        call poisson_timer_comm_exit()
    end subroutine fft_cudecomp_y_to_x


    subroutine fft_cudecomp_finalize()
        implicit none

        integer :: istat

        if (.not. fft_cudecomp_ready) return

        if (associated(fft_cudecomp_work_d)) then
            istat = cudecompFree( &
                fft_cudecomp_handle, fft_cudecomp_grid, &
                fft_cudecomp_work_d)
            call fft_cudecomp_abort(istat, 'cudecompFree(transpose)')
            nullify(fft_cudecomp_work_d)
        endif

        istat = cudecompGridDescDestroy( &
            fft_cudecomp_handle, fft_cudecomp_grid)
        call fft_cudecomp_abort(istat, 'cudecompGridDescDestroy')

        istat = cudecompFinalize(fft_cudecomp_handle)
        call fft_cudecomp_abort(istat, 'cudecompFinalize')

        fft_cudecomp_ready = .false.
    end subroutine fft_cudecomp_finalize

end module fft_cudecomp_bridge
